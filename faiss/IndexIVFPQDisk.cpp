#include <faiss/IndexIVFPQDisk.h>

#include <omp.h>
#include <cstdint>
#include <memory>
#include <mutex>

#include <algorithm>
#include <map>
#include <cinttypes>
#include <cstdio>
#include <limits>

#include <fstream>
#include <unistd.h>   // 包含 read, lseek 函数
#include <fcntl.h>    // 包含 open 函数和相关的文件控制常量（如 O_RDONLY 等）
#include <sys/types.h> // 包含 ssize_t 类型
#include <sys/stat.h>  // 包含 open 相关的模式常量
#include <cerrno>      // 包含 errno 及其处理
#include <stdexcept>   // 包含 std::runtime_error
#include <sys/mman.h>

#include <faiss/utils/Heap.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>
#include <faiss/index_io.h>
//#include <faiss/tsl/robin_set.h>
//#include <unordered_set>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/DiskIOProcessor.h>

#include <faiss/impl/code_distance/code_distance.h>

#include <iostream>

#define USING_ASYNC
//#define USING SYNC

namespace faiss{

IndexIVFPQDisk::IndexIVFPQDisk(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        size_t top,
        float estimate_factor,
        float prune_factor,
        const std::string& diskPath,
        const std::string& valueType,
        MetricType metric): 
    IndexIVFPQ(quantizer, d, nlist, M, nbits_per_idx, metric), 
                top(top), 
                estimate_factor(estimate_factor), 
                prune_factor(prune_factor), 
                disk_path(diskPath),
                disk_vector_offset(d * sizeof(float)),
                valueType(valueType) {
    estimate_factor_partial = estimate_factor;
    //this->code_size = d * sizeof(float);  // code_size refer to vectors stored in disk.
    //clusters = new size_t[nlist];
    //len = new size_t[nlist];
    std::cout << valueType << " " << this->valueType << std::endl;
    if(valueType!="float" && valueType!= "uint8" && valueType!= "int16"){
       
        FAISS_THROW_FMT("Unsupported type %s", valueType.c_str());
    }

    //FAISS_ASSERT_MSG(valueType!="float" && valueType!= "uint8" && valueType!= "int16", "Unsupported value type");
    
    this->aligned_cluster_info = nullptr;
    clusters = nullptr;
    len = nullptr;
}

IndexIVFPQDisk::IndexIVFPQDisk() {}

IndexIVFPQDisk::~IndexIVFPQDisk() {
    if(clusters != nullptr)
        delete[] clusters;
    if(len != nullptr)
        delete[] len;
    if(aligned_cluster_info!=nullptr)
        delete[] aligned_cluster_info;
}

void IndexIVFPQDisk::train_graph(){
    std::cout << "The index is of type Disk.\n";
    IndexFlat* flat_quantizer = dynamic_cast<IndexFlat*>(quantizer);
    if (flat_quantizer != nullptr) {
        faiss::IndexHNSWFlat index(d, 16);
        index.add(quantizer->ntotal, flat_quantizer->get_xb()); 
        faiss::write_index(&index, this->centroid_index_path.c_str());
        std::cout << "Output centroid index.\n";
        load_hnsw_centroid_index();
    } else {
        // Handle the case where quantizer is not of type IndexFlat
        std::cerr << "Quantizer is not an IndexFlat." << std::endl;
    }
}

void IndexIVFPQDisk::load_hnsw_centroid_index(){
    if (centroid_index_path.empty()) {
        throw std::runtime_error("Centroid index path is not set.");
    }
    // Load the HNSW index from the specified path
    faiss::Index* loaded_index = faiss::read_index(centroid_index_path.c_str());

    // Attempt to cast the loaded index to faiss::IndexHNSW
    centroid_index = dynamic_cast<faiss::IndexHNSWFlat*>(loaded_index);

    if (centroid_index == nullptr) {
        throw std::runtime_error("Failed to cast the loaded index to faiss::IndexHNSW.");
    }

    std::cout << "HNSW centroid index loaded successfully from " << centroid_index_path << std::endl;
    
}

namespace{

    // Function to calculate the dot product of two vectors
double dotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2, size_t d) {
    double dot = 0.0;
    for (size_t i = 0; i < d; i++) {
        dot += vec1[i] * vec2[i];
    }
    return dot;
}

// Function to calculate the norm (magnitude) of a vector
double vectorNorm(const std::vector<double>& vec, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; i++) {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

// Function to calculate the residual vector (difference between two vectors)
std::vector<double> calculateResidual(const std::vector<double>& vec, const std::vector<double>& centroid, size_t d) {
    std::vector<double> residual(d);
    for (size_t i = 0; i < d; i++) {
        residual[i] = vec[i] - centroid[i];
    }
    return residual;
}

// Function to retrieve a centroid vector from centroidData given an index
std::vector<double> getCentroidVector(const float* centroidData, idx_t centroidIdx, size_t d) {
    std::vector<double> centroid(d);
    for (size_t i = 0; i < d; i++) {
        centroid[i] = static_cast<double>(centroidData[centroidIdx * d + i]);
    }
    return centroid;
}

// Function to retrieve a vector from xb given an index
std::vector<double> getVector(const float* xb, size_t vecIdx, size_t d) {
    std::vector<double> vec(d);
    for (size_t i = 0; i < d; i++) {
        vec[i] = static_cast<double>(xb[vecIdx * d + i]);
    }
    return vec;
}

// Function to find the primary and secondary centroid assignments for each vector
void SOAR(const float* xb, const float* centroidData, 
                     idx_t* coarse_idx, idx_t* assign,
                     size_t nb, size_t d, size_t k) {
    // Loop through each vector in xb
    #pragma omp parallel for
    for (size_t vecIdx = 0; vecIdx < nb; vecIdx++) {
        // Retrieve the current vector from xb
        std::vector<double> vector = getVector(xb, vecIdx, d);

        // Step 1: Use the first centroid as the primary assignment
        size_t primaryCentroidIdx = 0;
        assign[2 * vecIdx] = coarse_idx[vecIdx * k + primaryCentroidIdx];  // Primary assignment for the current vector

        // Step 2: Retrieve the primary centroid vector
        std::vector<double> primaryCentroid = getCentroidVector(centroidData, assign[2 * vecIdx], d);
        
        // Calculate the residual vector with the primary centroid
        std::vector<double> primaryResidual = calculateResidual(vector, primaryCentroid, d);
        double primaryResidualNorm = vectorNorm(primaryResidual, d);

        // Step 3: Find the secondary centroid with the smallest cosine similarity to the primary residual
        double minCosineSimilarity = std::numeric_limits<double>::max();
        size_t secondaryCentroidIdx = primaryCentroidIdx;

        for (size_t i = 1; i < k; i++) {  // Start from 1 since 0 is already the primary centroid
            idx_t centroidId = coarse_idx[vecIdx * k + i];
            std::vector<double> currentCentroid = getCentroidVector(centroidData, centroidId, d);
            std::vector<double> currentResidual = calculateResidual(vector, currentCentroid, d);
            double currentResidualNorm = vectorNorm(currentResidual, d);

            // Calculate the cosine similarity
            double cosineSimilarity = dotProduct(primaryResidual, currentResidual, d) / 
                                      (primaryResidualNorm * currentResidualNorm);

            if (cosineSimilarity < minCosineSimilarity) {
                minCosineSimilarity = cosineSimilarity;
                secondaryCentroidIdx = i;
            }
        }
        if(secondaryCentroidIdx ==  primaryCentroidIdx){
            secondaryCentroidIdx=primaryCentroidIdx + 1;
        }
        // Store the secondary assignment for the current vector
        assign[2 * vecIdx + 1] = coarse_idx[vecIdx * k + secondaryCentroidIdx];
    }
}
}


// void IndexIVFPQDisk::add_with_ids(idx_t n, const float* x, const idx_t* xids){
//     idx_t k = this->assign_replicas;
//     std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n * k]);
   
//     if(centroid_index != nullptr && n*k > 10000){
//         std::cout << "Using hnsw centroid assign\n";
//         centroid_index->hnsw.efSearch = k * 4/3;
//         centroid_index->assign(n, x, coarse_idx.get(), k);
//     }else{
//          std::cout << "Using quantizer assign\n";
//         quantizer->assign(n, x, coarse_idx.get(), k);
//     }

//     add_core(n, x, xids, coarse_idx.get());
// }

void IndexIVFPQDisk::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    // idx_t k = this->assign_replicas;
    idx_t k = 5;
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n * k]);
    printf("IndexIVFPQDisk::add_with_ids::k=%ld\n",k);
    float* D = new float[k * n];
    if (n * k < 100000) {
        quantizer->assign(n, x, coarse_idx.get(), k);
    } else {
        printf("IndexIVFPQDisk::add_with_ids::hnsw way, n=%ld, k=%ld \n", n, k);
       
        centroid_index->hnsw.efSearch = 400;
        centroid_index->search(n, x, k, D, coarse_idx.get());
        // quantizer->assign(n, x, coarse_idx.get(), k);
    }
    if (k != 1) {
        printf("Using SOAR\n");
        std::unique_ptr<idx_t[]> assign(new idx_t[n * 2]);
        IndexFlat* flat_quantizer = dynamic_cast<IndexFlat*>(quantizer);
        float* centroidData = flat_quantizer->get_xb();
        SOAR(x, centroidData, coarse_idx.get(), assign.get(), n, d, k);
        add_core(n, x, xids, assign.get());
        return;
    }
    //     if (k != 1) {
    //     adjustReplica(n, k, coarse_idx.get(), D);
    // }
    add_core(n, x, xids, coarse_idx.get());
}

void IndexIVFPQDisk::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    add_core_o(n, x, xids, nullptr, coarse_idx, inverted_list_context);
    initial_location(n, x);
}

void IndexIVFPQDisk::initial_location(idx_t n, const float* data) {
    if (!invlists) {
        throw std::runtime_error("invlists is not initialized.");
    }

    // Cast invlists to ArrayInvertedLists to access the underlying data
    ArrayInvertedLists* array_invlists = dynamic_cast<ArrayInvertedLists*>(invlists);
    if (!array_invlists) {
        throw std::runtime_error("invlists is not of type ArrayInvertedLists.");
    }

    size_t* tmp_clusters = nullptr;
    size_t* tmp_len = nullptr;
    // the first time to add
    if(clusters == nullptr && len == nullptr){
        this->aligned_cluster_info = new Aligned_Cluster_Info[nlist];
        clusters = new size_t[nlist];
        len = new size_t[nlist];
    }else{
        tmp_clusters = new size_t[nlist];
        tmp_len = new size_t[nlist];
        for(size_t i = 0; i < nlist; ++i){
            tmp_clusters[i] = clusters[i];
            tmp_len[i] = len[i];
        }
    }
    
    size_t current_offset = 0;
    for (size_t i = 0; i < nlist; ++i) {
        clusters[i] = current_offset;
        len[i] = array_invlists->ids[i].size();
        current_offset += len[i];
    }
    if(verbose){
        printf("Cluster info initialized!");
    }

    std::unique_ptr<DiskIOProcessor> io_processor(get_DiskIOBuildProcessor());
    //get_DiskIOBuildProcessor();
    // reorg it at last, and save it to file.
    
    //reorganize_vectors(n, data, tmp_clusters ,tmp_len);
    if(io_processor->reorganize_vectors(n, 
                                    data, 
                                    tmp_clusters, 
                                    tmp_len, 
                                    clusters, 
                                    len, 
                                    aligned_cluster_info,
                                    nlist, 
                                    array_invlists->ids)){
        this->disk_path = disk_path + ".clustered";
    }
}

// Method to reorganize vectors based on clustering
void IndexIVFPQDisk::reorganize_vectors(idx_t n, const float* data, size_t* old_clusters, size_t* old_len) {
    
    ArrayInvertedLists* array_invlists = dynamic_cast<ArrayInvertedLists*>(invlists);
    if (!array_invlists) {
        throw std::runtime_error("invlists is not of type ArrayInvertedLists.");
    }

    idx_t old_total = this->ntotal - n;

    if (old_clusters == nullptr && old_len == nullptr) {
        // Reorganize vectors and write to the new file
        //disk_path = disk_path + ".clustered";

        //std::cout << "disk_path_clustered: " << disk_path_clustered << std::endl;
        //std::cout << "disk_path          : " << disk_path << std::endl;
        set_disk_write(disk_path);
        for (size_t i = 0; i < nlist; ++i) {
            size_t count = len[i];
            for (size_t j = 0; j < count; ++j) {
                idx_t id = array_invlists->ids[i][j];
                const float* vector = &data[id * d];
                disk_data_write.write(reinterpret_cast<const char*>(vector), d * sizeof(float));
            }
        }
        disk_data_write.close();
    } else {
        std::string tmp_disk = disk_path + ".tmp";
        // 1. Rename disk_path_clustered to tmp_disk
        int file_result = std::rename(disk_path.c_str(), tmp_disk.c_str());
        if(file_result == 0)
            std::cout << "Success rename: " << tmp_disk << std::endl;
        else
            std::cout << "Fail: "<< tmp_disk << std::endl;
        // 2. Open temp_disk for reading
        std::ifstream temp_disk_read(tmp_disk, std::ios::binary);
        if (!temp_disk_read.is_open()) {
            throw std::runtime_error("Failed to open temporary disk file for reading.");
        }

        // 3. Set up for writing to the new clustered disk path
        set_disk_write(disk_path);
        
        // 4. Write old data and new data to the file
        for (size_t i = 0; i < nlist; ++i) {
            size_t old_offset = old_clusters[i];
            size_t old_count = old_len[i];

            // Read old cluster data
            std::vector<float> old_cluster(old_count * d);
            temp_disk_read.seekg(old_offset * d * sizeof(float), std::ios::beg);
            temp_disk_read.read(reinterpret_cast<char*>(old_cluster.data()), old_count * d * sizeof(float));
            disk_data_write.write(reinterpret_cast<const char*>(old_cluster.data()), old_count * d * sizeof(float));

            // Write new data
            size_t count = len[i];
            for (size_t j = old_count; j < count; ++j) {
                idx_t id = array_invlists->ids[i][j];
                const float* vector = &data[(id - old_total) * d];
                disk_data_write.write(reinterpret_cast<const char*>(vector), d * sizeof(float));
            }
        }
        disk_data_write.close();
        temp_disk_read.close();

        // 5. Delete the temporary file
        std::remove(tmp_disk.c_str());
    }

    if (verbose) {
        printf("Vectors reorganized and written to %s\n", disk_path.c_str());
    }
}

// Method to set the disk path and open read stream
void IndexIVFPQDisk::set_disk_read(const std::string& diskPath) {
    disk_path = diskPath;
    if (disk_data_read.is_open()) {
        disk_data_read.close();
    }
    disk_data_read.open(disk_path, std::ios::binary);
    if (!disk_data_read.is_open()) {
        throw std::runtime_error("IndexIVFPQDisk: Failed to open disk file for reading");
    }
}

// Method to set the disk path and open write stream
void IndexIVFPQDisk::set_disk_write(const std::string& diskPath) {
    disk_path = diskPath;
    if (disk_data_write.is_open()) {
        disk_data_write.close();
    }
    disk_data_write.open(disk_path, std::ios::binary);
    if (!disk_data_write.is_open()) {
        throw std::runtime_error("IndexIVFPQDisk: Failed to open disk file for writing");
    }
}

void IndexIVFPQDisk::load_from_offset(size_t list_no, size_t offset, float* original_vector) {
    if (!disk_data_read.is_open()) {
        throw std::runtime_error("Disk read stream is not open.");
    }

    // 检查偏移量是否有效
    assert(offset < len[list_no]);

    // 计算全局偏移量
    size_t global_offset = (clusters[list_no] + offset) * d * sizeof(float);
    disk_data_read.seekg(global_offset, std::ios::beg);

    // 读取向量数据
    disk_data_read.read(reinterpret_cast<char*>(original_vector), d * sizeof(float));

    if (disk_data_read.fail()) {
        throw std::runtime_error("Failed to read vector from disk.");
    }
}

void IndexIVFPQDisk::load_clusters(size_t list_no, float* original_vectors) {
    if (!disk_data_read.is_open()) {
        throw std::runtime_error("Disk read stream is not open.");
    }

    // 计算全局偏移量
    size_t global_offset = clusters[list_no] * d * sizeof(float);
    disk_data_read.seekg(global_offset, std::ios::beg);

    // 读取所有向量数据
    disk_data_read.read(reinterpret_cast<char*>(original_vectors), d * sizeof(float) * len[list_no]);

    if (disk_data_read.fail()) {
        throw std::runtime_error("Failed to read vectors from disk.");
    }
}

namespace{
     int sort_coarse(
        std::vector<idx_t>& listno,            // input: cluster ids
        std::vector<size_t>& sorted_listno,    // output: sorted cluster ids based on frequency
        size_t* lens,                          // input: array of cluster sizes (number of vectors in each cluster)
        size_t nlist,                          // input: maximum number of clusters to cache
        size_t nvec)                           // input: maximum number of vectors to cache
    {
        std::map<idx_t, size_t> freq_map;
        for (const auto& id : listno) {
            freq_map[id]++;
        }

        std::vector<std::pair<idx_t, size_t>> freq_pairs(freq_map.begin(), freq_map.end());

        std::sort(freq_pairs.begin(), freq_pairs.end(), 
            [](const std::pair<idx_t, size_t>& a, const std::pair<idx_t, size_t>& b) {
                return a.second > b.second;
            });

        size_t total_vecs = 0;

        for (const auto& pair : freq_pairs) {
            idx_t cluster_id = pair.first;
            //std::cout <<"cluster_id"<<cluster_id <<std::endl;
            if (nlist != static_cast<size_t>(-1) && sorted_listno.size() >= nlist) {
                break;
            }

            if (nvec != static_cast<size_t>(-1)) {
                total_vecs += lens[cluster_id];
                if (total_vecs >= nvec) {
                    // Stop if the total vectors exceed nvec
                    sorted_listno.push_back(cluster_id);
                    break;
                }
            }

            sorted_listno.push_back(cluster_id);
        }

        return sorted_listno.size();
    }

    int warm_up(DiskInvertedListHolder& holder, std::vector<size_t>& indice){
        holder.warm_up(indice);
        // TODO return What???
        return 0;
    }
}

int IndexIVFPQDisk::warm_up_nlist(size_t n, float* x, size_t w_nprobe, size_t warm_list){
    std::vector<idx_t> idx(n * w_nprobe);
    std::vector<float> coarse_dis(n * w_nprobe);
    std::vector<size_t> sorted_idx;
    // Search to get the nearest cluster IDs
    quantizer->search(n, x, w_nprobe, coarse_dis.data(), idx.data(), nullptr);

    sort_coarse(idx, sorted_idx, this->len, warm_list, static_cast<size_t>(-1));
    
    size_t code_size = get_code_size();

    diskInvertedListHolder.set_holder(disk_path, nlist, code_size, aligned_cluster_info);
    // for(int i = 0; i< warm_list;i++){
    //     std::cout << "idx:"<<sorted_idx.data()[i] << "  length:" << this->len[sorted_idx.data()[i]] << std::endl;
    // }

    warm_up(diskInvertedListHolder, sorted_idx);
    
    return warm_list;
}

int IndexIVFPQDisk::warm_up_nvec(size_t n, float* x, size_t w_nprobe, size_t nvec){
    if(nvec==0){
        return 0;
    }
    std::vector<idx_t> idx(n * w_nprobe);
    std::vector<float> coarse_dis(n * w_nprobe);
    std::vector<size_t> sorted_idx;
    quantizer->search(n, x, w_nprobe, coarse_dis.data(), idx.data(), nullptr);
    
    int warm_list = sort_coarse(idx, sorted_idx, this->len, static_cast<size_t>(-1), nvec);
    //std::cout << "warm_list: "<< warm_list << std::endl;
    size_t code_size = get_code_size();

    diskInvertedListHolder.set_holder(disk_path, nlist, code_size, aligned_cluster_info);
    warm_up(diskInvertedListHolder, sorted_idx);

    return warm_list;
}


/** It is a sad fact of software that a conceptually simple function like this
 * becomes very complex when you factor in several ways of parallelizing +
 * interrupt/error handling + collecting stats + min/max collection. The
 * codepath that is used 95% of time is the one for parallel_mode = 0 */
void IndexIVFPQDisk::search(
        idx_t n,
        const float* x,
        idx_t k_r,
        float* distances_result,
        idx_t* labels_result,
        const SearchParameters* params_in ) const {
    FAISS_THROW_IF_NOT(k_r > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    // : make a new distances and labels to contain replica*k results
    //       new k_replica = k * this->reolica
    idx_t k = k_r * this->assign_replicas;
    std::unique_ptr<idx_t[]> del1(new idx_t[n * k]);
    std::unique_ptr<float[]> del2(new float[n * k]);
    idx_t* labels = del1.get();
    float* distances = del2.get();

    // search function for a subset of queries
    auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        auto time_start = std::chrono::high_resolution_clock::now();      // time begin

        double t0 = getmillisecs();

        if(centroid_index != nullptr && nprobe > 60){
            std::cout << "Searching in HNSW\n";
            centroid_index->hnsw.efSearch = 2 * nprobe;
            centroid_index->search(n,x,nprobe,coarse_dis.get(),idx.get());
        }
        else{
            std::cout << "Searching by quantizer\n";
            quantizer->search(
                    n,
                    x,
                    nprobe,
                    coarse_dis.get(),
                    idx.get(),
                    params ? params->quantizer_params : nullptr);
        }
        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

        auto time_end = std::chrono::high_resolution_clock::now(); 
        indexIVFPQDisk_stats.coarse_elapsed += time_end - time_start;

        search_preassigned(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                params,
                ivf_stats);
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice]);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle parallelization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
    // 

    // for(idx_t ii = 0; ii < n;ii++){
    //     idx_t begin = ii*k;
    //     for(idx_t jj = 0; jj < k; jj++){
    //         std::cout<<jj<<":\t label:" <<  labels[begin+jj] << "  distance:"<<distances[begin+jj] << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    
    auto time_start = std::chrono::high_resolution_clock::now();      // time begin

    for(idx_t ii = 0; ii < n;ii++){
        idx_t begin_r = ii*k_r;
        idx_t begin = ii*k;
        idx_t limit = 0;

        for(idx_t jj = 0; jj < k; jj++){
            //if(ii==3202)
                //std::cout << "ii: "<<ii <<"  jj:"<< jj << " :"<<labels[begin+jj]<< std::endl;
            if(jj == 0){
                distances_result[begin_r] = distances[begin];
                labels_result[begin_r] = labels[begin];
                limit++;
            }
            else{
                if(labels[begin+jj] != labels[begin+jj-1]){
                    distances_result[begin_r+limit] = distances[begin+jj];
                    labels_result[begin_r+limit] = labels[begin+jj];
                    limit++;
                }
                if(limit>=k_r)
                    break;

            }
        }
    }
    auto time_end = std::chrono::high_resolution_clock::now();       // time end
    indexIVFPQDisk_stats.rank_elapsed += time_end - time_start;
}

void IndexIVFPQDisk::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) const {
    
    auto time_start = std::chrono::high_resolution_clock::now();      // time begin
    
    FAISS_THROW_IF_NOT(k > 0);               //1. 参数检查

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);           //2. nprobe参数初始化 以及检查

    const size_t top_cluster = this->top;     // 设置使用不同load_strategy的边界

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;
    const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr; // use special IDSelectorRange processing
        } else {
            selr = nullptr; // use generic processing
        }
    }

    FAISS_THROW_IF_NOT_MSG(
            !(sel && store_pairs),
            "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
            !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");              //3. 参数检查

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;                       //4. 统计变量初始化

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
            max_codes == 0 || pmode == 0 || pmode == 3,
            "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? nprobe > 1 
                                  : nprobe * n > 1);            // 5.并行模式检查

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;
    
    const float p_factor = this->prune_factor;

    auto time_end = std::chrono::high_resolution_clock::now();       // time end
    indexIVFPQDisk_stats.others_elapsed += time_end - time_start;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)    //6. 并行查找
    { 
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel));
        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };            // 1. 初始化结果

        auto add_local_results = [&](const float* local_dis,
                                     const idx_t* local_idx,
                                     float* simi,
                                     idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }   
        };      // 2. 增加计算好的结果到堆

        auto reorder_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };           // 3. 重排序函数

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 idx_t list_size_max) {
            //auto time_start = std::chrono::high_resolution_clock::now();      // time begin
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            // don't waste time on empty lists
            if (invlists->is_empty(key, inverted_list_context)) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            //auto time_end = std::chrono::high_resolution_clock::now();       // time end
            //indexIVFPQDisk_stats.others_elapsed += time_end - time_start;

            try {
                if (invlists->use_iterator) {
                    size_t list_size = 0;

                    std::unique_ptr<InvertedListsIterator> it(
                            invlists->get_iterator(key, inverted_list_context));

                    nheap += scanner->iterate_codes(
                            it.get(), simi, idxi, k, list_size);

                    return list_size;
                } else {
                    size_t list_size = invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t* codes = scodes.get();

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                                invlists, key);
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                                list_size, ids, &jmin, &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }
                    // if(ids == nullptr){
                    //     std::cout << "NULLPTR!!" << std::endl;
                    // }else{
                    //     std::cout << "NONE NULL" << std::endl;
                    // }
                    
                    nheap += scanner->scan_codes(
                            list_size, codes, ids, simi, idxi, k);

                    return list_size;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         * pmode == 0或pmode == 3：对每个查询向量进行并行处理，每个查询向量都扫描所有的倒排列表。
           pmode == 1：每个查询向量依次进行处理，但对倒排列表的扫描是并行的。
           pmode == 2：每个倒排列表的扫描是并行的，查询向量的结果在最后进行合并。
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }
                auto time_start = std::chrono::high_resolution_clock::now();      // time begin
                // loop over queries
                scanner->set_query(x + i * d);  // todo: reset hash table in scanner
                
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;
                auto time_end = std::chrono::high_resolution_clock::now();       // time end
                indexIVFPQDisk_stats.others_elapsed += time_end - time_start;
                // TODO scan async??? here
                size_t valid_probe = 0;
                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {

                     /*
                    prune redundant clusters
                    */
                    if(coarse_dis[i * nprobe + ik] > p_factor * coarse_dis[i * nprobe])
                    {
                        //std::cout << "coarse_dis[i * nprobe + ik]: " << coarse_dis[i * nprobe + ik] ;
                        //std::cout << " <>" << p_factor << "*"<< coarse_dis[i * nprobe];
                        //std::cout << " = " << p_factor * coarse_dis[i * nprobe + ik] << std::endl;
                        break; 
                    }

                    // if(ik>=21 && ik <= 30){
                    //     continue;
                    // }

                    valid_probe++;
                    indexIVFPQDisk_stats.pruned++;
                    if(ik >= top_cluster){
                        scanner->set_strategy(PARTIALLY);
#ifdef USING_ASYNC  
                        scanner.get()->async_submit(valid_probe);
                        if(ik!=0 && ik%20 == 0){
                            //std::cout << "ik:" <<ik <<"   Async_submit!!!!" << std::endl;
                            scanner.get()->async_submit();
                        }
#endif
                    }
                    else{
                        scanner->set_strategy(FULLY);
                    }

                    nscan += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            simi,
                            idxi,
                            max_codes - nscan);
                    if (nscan >= max_codes) {
                        break;
                    }
                }
#ifdef USING_ASYNC
                scanner.get()->async_submit();
                //std::cout << "Valid_probe " << valid_probe << std::endl;
#endif

                ndis += nscan;
                reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        } else if (pmode == 1) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {

                    if(ik > top_cluster)
                        scanner->set_strategy(PARTIALLY);

                    ndis += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            local_dis.data(),
                            local_idx.data(),
                            unlimited_list_size);

                    // can't do the test on max_codes
                }
                // merge thread-local results

                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
#pragma omp single
                init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(), local_idx.data(), simi, idxi);
                }
#pragma omp barrier
#pragma omp single
                reorder_result(simi, idxi);
            }
        } else if (pmode == 2) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                size_t i = ij / nprobe;

                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                ndis += scan_one_list(
                        keys[ij],
                        coarse_dis[ij],
                        local_dis.data(),
                        local_idx.data(),
                        unlimited_list_size);
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(),
                            local_idx.data(),
                            distances + i * k,
                            labels + i * k);
                }
            }
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats == nullptr) {
        ivf_stats = &indexIVF_stats;
    }
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}


// ----------------------modification from the original IVFPQ----------------------------------------------

namespace {

#define TIC t0 = get_cycles()
#define TOC get_cycles() - t0

using HeapForIP = CMin<float, idx_t>;
using HeapForL2 = CMax<float, idx_t>;

/** QueryTables manages the various ways of searching an
 * IndexIVFPQ. The code contains a lot of branches, depending on:
 * - metric_type: are we computing L2 or Inner product similarity?
 * - by_residual: do we encode raw vectors or residuals?
 * - use_precomputed_table: are x_R|x_C tables precomputed?
 * - polysemous_ht: are we filtering with polysemous codes?
 */
struct QueryTables {
    /*****************************************************
     * General data from the IVFPQ
     *****************************************************/

    const IndexIVFPQDisk& ivfpq_disk;
    const IVFSearchParameters* params;

    // copied from IndexIVFPQDisk for easier access
    int d;
    const ProductQuantizer& pq;
    MetricType metric_type;
    bool by_residual;
    int use_precomputed_table;
    int polysemous_ht;

    // pre-allocated data buffers
    float *sim_table, *sim_table_2;
    float *residual_vec, *decoded_vec;

    // single data buffer
    std::vector<float> mem;

    // for table pointers
    std::vector<const float*> sim_table_ptrs;

    explicit QueryTables(
            const IndexIVFPQDisk& ivfpq_disk,
            const IVFSearchParameters* params)
            : ivfpq_disk(ivfpq_disk),
              d(ivfpq_disk.d),
              pq(ivfpq_disk.pq),
              metric_type(ivfpq_disk.metric_type),
              by_residual(ivfpq_disk.by_residual),
              use_precomputed_table(ivfpq_disk.use_precomputed_table) {
        mem.resize(pq.ksub * pq.M * 2 + d * 2);
        sim_table = mem.data();
        sim_table_2 = sim_table + pq.ksub * pq.M;
        residual_vec = sim_table_2 + pq.ksub * pq.M;
        decoded_vec = residual_vec + d;

        // for polysemous
        polysemous_ht = ivfpq_disk.polysemous_ht;
        if (auto ivfpq_disk_params =
                    dynamic_cast<const IVFPQSearchParameters*>(params)) {
            polysemous_ht = ivfpq_disk_params->polysemous_ht;
        }
        if (polysemous_ht != 0) {
            q_code.resize(pq.code_size);
        }
        init_list_cycles = 0;
        sim_table_ptrs.resize(pq.M);
    }

    /*****************************************************
     * What we do when query is known
     *****************************************************/

    // field specific to query
    const float* qi;

    // query-specific initialization
    void init_query(const float* qi) {
        this->qi = qi;
        if (metric_type == METRIC_INNER_PRODUCT)
            init_query_IP();
        else
            init_query_L2();
        if (!by_residual && polysemous_ht != 0)
            pq.compute_code(qi, q_code.data());
    }

    void init_query_IP() {
        // precompute some tables specific to the query qi
        pq.compute_inner_prod_table(qi, sim_table);
    }

    void init_query_L2() {
        if (!by_residual) {
            pq.compute_distance_table(qi, sim_table);
        } else if (use_precomputed_table) {
            pq.compute_inner_prod_table(qi, sim_table_2);
        }
    }

    /*****************************************************
     * When inverted list is known: prepare computations
     *****************************************************/

    // fields specific to list
    idx_t key;
    float coarse_dis;
    std::vector<uint8_t> q_code;

    uint64_t init_list_cycles;

    /// once we know the query and the centroid, we can prepare the
    /// sim_table that will be used for accumulation
    /// and dis0, the initial value
    float precompute_list_tables() {
        float dis0 = 0;
        uint64_t t0;
        TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
                dis0 = precompute_list_tables_IP();
            else
                dis0 = precompute_list_tables_L2();
        }
        init_list_cycles += TOC;
        return dis0;
    }

    float precompute_list_table_pointers() {
        float dis0 = 0;
        uint64_t t0;
        TIC;
        if (by_residual) {
            if (metric_type == METRIC_INNER_PRODUCT)
                FAISS_THROW_MSG("not implemented");
            else
                dis0 = precompute_list_table_pointers_L2();
        }
        init_list_cycles += TOC;
        return dis0;
    }

    /*****************************************************
     * compute tables for inner prod
     *****************************************************/

    float precompute_list_tables_IP() {
        // prepare the sim_table that will be used for accumulation
        // and dis0, the initial value
        ivfpq_disk.quantizer->reconstruct(key, decoded_vec);
        // decoded_vec = centroid
        float dis0 = fvec_inner_product(qi, decoded_vec, d);

        if (polysemous_ht) {
            for (int i = 0; i < d; i++) {
                residual_vec[i] = qi[i] - decoded_vec[i];
            }
            pq.compute_code(residual_vec, q_code.data());
        }
        return dis0;
    }

    /*****************************************************
     * compute tables for L2 distance
     *****************************************************/

    float precompute_list_tables_L2() {
        float dis0 = 0;

        if (use_precomputed_table == 0 || use_precomputed_table == -1) {
            ivfpq_disk.quantizer->compute_residual(qi, residual_vec, key);
            pq.compute_distance_table(residual_vec, sim_table);

            if (polysemous_ht != 0) {
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            fvec_madd(
                    pq.M * pq.ksub,
                    ivfpq_disk.precomputed_table.data() + key * pq.ksub * pq.M,
                    -2.0,
                    sim_table_2,
                    sim_table);

            if (polysemous_ht != 0) {
                ivfpq_disk.quantizer->compute_residual(qi, residual_vec, key);
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq_disk.quantizer);
            FAISS_THROW_IF_NOT(miq);
            const ProductQuantizer& cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            const float* qtab = sim_table_2; // query-specific table
            float* ltab = sim_table;         // (output) list-specific table

            long k = key;
            for (int cm = 0; cm < cpq.M; cm++) {
                // compute PQ index
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                // get corresponding table
                const float* pc = ivfpq_disk.precomputed_table.data() +
                        (ki * pq.M + cm * Mf) * pq.ksub;

                if (polysemous_ht == 0) {
                    // sum up with query-specific table
                    fvec_madd(Mf * pq.ksub, pc, -2.0, qtab, ltab);
                    ltab += Mf * pq.ksub;
                    qtab += Mf * pq.ksub;
                } else {
                    for (int m = cm * Mf; m < (cm + 1) * Mf; m++) {
                        q_code[m] = fvec_madd_and_argmin(
                                pq.ksub, pc, -2, qtab, ltab);
                        pc += pq.ksub;
                        ltab += pq.ksub;
                        qtab += pq.ksub;
                    }
                }
            }
        }

        return dis0;
    }

    float precompute_list_table_pointers_L2() {
        float dis0 = 0;

        if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            const float* s =
                    ivfpq_disk.precomputed_table.data() + key * pq.ksub * pq.M;
            for (int m = 0; m < pq.M; m++) {
                sim_table_ptrs[m] = s;
                s += pq.ksub;
            }
        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq_disk.quantizer);
            FAISS_THROW_IF_NOT(miq);
            const ProductQuantizer& cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            long k = key;
            int m0 = 0;
            for (int cm = 0; cm < cpq.M; cm++) {
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                const float* pc = ivfpq_disk.precomputed_table.data() +
                        (ki * pq.M + cm * Mf) * pq.ksub;

                for (int m = m0; m < m0 + Mf; m++) {
                    sim_table_ptrs[m] = pc;
                    pc += pq.ksub;
                }
                m0 += Mf;
            }
        } else {
            FAISS_THROW_MSG("need precomputed tables");
        }

        if (polysemous_ht) {
            FAISS_THROW_MSG("not implemented");
            // Not clear that it makes sense to implemente this,
            // because it costs M * ksub, which is what we wanted to
            // avoid with the tables pointers.
        }

        return dis0;
    }
};

// This way of handling the selector is not optimal since all distances
// are computed even if the id would filter it out.
template <class C, bool use_sel>
struct KnnSearchResults {
    idx_t key;
    const idx_t* ids;
    const IDSelector* sel;

    // heap params
    size_t k;
    float* heap_sim;
    idx_t* heap_ids;

    size_t nup;

    inline bool check_repeat(idx_t j){
        // idx_t id = ids ? ids[j] : lo_build(key, j);
        // for (size_t ii = 0; ii < k; ii++) {
        //     if (id == heap_ids[ii])
        //         return true;   
        // }
        return false;
    }

    inline bool skip_entry(idx_t j) {
        return use_sel && !sel->is_member(ids[j]);
    }

    inline void add(idx_t j, float dis) {
        if (C::cmp(heap_sim[0], dis)) {
            idx_t id = ids ? ids[j] : lo_build(key, j);
            heap_replace_top<C>(k, heap_sim, heap_ids, dis, id);
            nup++;
        }
    }
};

template <class C, bool use_sel>
struct RangeSearchResults {
    idx_t key;
    const idx_t* ids;
    const IDSelector* sel;

    // wrapped result structure
    float radius;
    RangeQueryResult& rres;

    inline bool skip_entry(idx_t j) {
        return use_sel && !sel->is_member(ids[j]);
    }

    inline void add(idx_t j, float dis) {
        if (C::cmp(radius, dis)) {
            idx_t id = ids ? ids[j] : lo_build(key, j);
            rres.add(dis, id);
        }
    }
};

/*****************************************************
 * Scaning the codes.
 * The scanning functions call their favorite precompute_*
 * function to precompute the tables they need.
 *****************************************************/

// global
std::unordered_set<idx_t> visited;

template <typename IDType, MetricType METRIC_TYPE, class PQDecoder>
struct IVFPQDiskScannerT : QueryTables {
    const uint8_t* list_codes;
    const IDType* list_ids;
    size_t list_size;

    IVFPQDiskScannerT(const IndexIVFPQDisk& ivfpq_disk, const IVFSearchParameters* params)
            : QueryTables(ivfpq_disk, params) {
        assert(METRIC_TYPE == metric_type);
    }

    float dis0;

    void init_list(idx_t list_no, float coarse_dis, int mode) {
        this->key = list_no;
        this->coarse_dis = coarse_dis;

        if (mode == 2) {
            dis0 = precompute_list_tables();
            //std::cout << "dis0:" <<dis0 << std::endl;
        } else if (mode == 1) {
            dis0 = precompute_list_table_pointers();
        }
    }

    
    /// store all result
    template <class SearchResultType>
    int scan_list_with_table(
            size_t ncode,
            const uint8_t* codes,
            float* local_dis,
            idx_t* local_ids,
            idx_t* knn_ids,              // skip some existed entries directly
            const idx_t* list_ids,
            size_t k,
            SearchResultType& res) const {
        int counter = 0;
        int operation = 0;

        size_t saved_j[4] = {0, 0, 0, 0};
        for (size_t j = 0; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }
            /* skip scanned entry, simple implementation*/
            // bool scanned = false;
            // for (size_t ii = 0; ii < k; ii++) {
            //     if (list_ids[j] == knn_ids[ii])
            //         scanned = true;   
            // }
            // if (scanned) {
            //     continue;
            // }
            // if (visited.find(list_ids[j]) != visited.end()) {
            //     continue; 
            // }
            // visited.insert(list_ids[j]);
            //std::cout << "list_ids[j]: " << list_ids[j] << std::endl;

            // res.add() discard repetitive result

            ///> if we try to remove repetitive vector here, it would be extraordinarily slow

            

            saved_j[0] = (counter == 0) ? j : saved_j[0];
            saved_j[1] = (counter == 1) ? j : saved_j[1];
            saved_j[2] = (counter == 2) ? j : saved_j[2];
            saved_j[3] = (counter == 3) ? j : saved_j[3];

            counter += 1;
            if (counter == 4) {
                float distance_0 = 0;
                float distance_1 = 0;
                float distance_2 = 0;
                float distance_3 = 0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);
                *(local_dis++) = dis0 + distance_0;
                *(local_ids++) = saved_j[0];
                *(local_dis++) = dis0 + distance_1;
                *(local_ids++) = saved_j[1];
                *(local_dis++) = dis0 + distance_2;
                *(local_ids++) = saved_j[2];
                *(local_dis++) = dis0 + distance_3;
                *(local_ids++) = saved_j[3];

                operation +=4;
                //res.add(saved_j[0], dis0 + distance_0);
                //res.add(saved_j[1], dis0 + distance_1);
                //res.add(saved_j[2], dis0 + distance_2);
                //res.add(saved_j[3], dis0 + distance_3);
                counter = 0;
            }
        }

        if (counter >= 1) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[0] * pq.code_size);
            *(local_dis++) = dis;
            *(local_ids++) = saved_j[0];
            operation ++;
            //res.add(saved_j[0], dis);
        }
        if (counter >= 2) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[1] * pq.code_size);
            *(local_dis++) = dis;
            *(local_ids++) = saved_j[1];
            operation ++;
            //res.add(saved_j[1], dis);
        }
        if (counter >= 3) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[2] * pq.code_size);
            *(local_dis++) = dis;
            *(local_ids++) = saved_j[2];
            operation ++;
            //res.add(saved_j[2], dis);
        }
        return operation;
    }

    /// version of the scan where we use precomputed tables.
    template <class SearchResultType>
    void scan_list_with_table(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        int counter = 0;

        size_t saved_j[4] = {0, 0, 0, 0};
        for (size_t j = 0; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }

            saved_j[0] = (counter == 0) ? j : saved_j[0];
            saved_j[1] = (counter == 1) ? j : saved_j[1];
            saved_j[2] = (counter == 2) ? j : saved_j[2];
            saved_j[3] = (counter == 3) ? j : saved_j[3];

            counter += 1;
            if (counter == 4) {
                float distance_0 = 0;
                float distance_1 = 0;
                float distance_2 = 0;
                float distance_3 = 0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0);
                res.add(saved_j[1], dis0 + distance_1);
                res.add(saved_j[2], dis0 + distance_2);
                res.add(saved_j[3], dis0 + distance_3);
                counter = 0;
            }
        }

        if (counter >= 1) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[0] * pq.code_size);
            res.add(saved_j[0], dis);
        }
        if (counter >= 2) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[1] * pq.code_size);
            res.add(saved_j[1], dis);
        }
        if (counter >= 3) {
            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[2] * pq.code_size);
            res.add(saved_j[2], dis);
        }
    }

    /// tables are not precomputed, but pointers are provided to the
    /// relevant X_c|x_r tables
    template <class SearchResultType>
    void scan_list_with_pointer(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
            if (res.skip_entry(j)) {
                continue;
            }
            PQDecoder decoder(codes, pq.nbits);
            float dis = dis0;
            const float* tab = sim_table_2;

            for (size_t m = 0; m < pq.M; m++) {
                int ci = decoder.decode();
                dis += sim_table_ptrs[m][ci] - 2 * tab[ci];
                tab += pq.ksub;
            }
            res.add(j, dis);
        }
    }

    /// nothing is precomputed: access residuals on-the-fly
    template <class SearchResultType>
    void scan_on_the_fly_dist(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        const float* dvec;
        float dis0 = 0;
        if (by_residual) {
            if (METRIC_TYPE == METRIC_INNER_PRODUCT) {
                ivfpq_disk.quantizer->reconstruct(key, residual_vec);
                dis0 = fvec_inner_product(residual_vec, qi, d);
            } else {
                ivfpq_disk.quantizer->compute_residual(qi, residual_vec, key);
            }
            dvec = residual_vec;
        } else {
            dvec = qi;
            dis0 = 0;
        }

        for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
            if (res.skip_entry(j)) {
                continue;
            }
            pq.decode(codes, decoded_vec);

            float dis;
            if (METRIC_TYPE == METRIC_INNER_PRODUCT) {
                dis = dis0 + fvec_inner_product(decoded_vec, qi, d);
            } else {
                dis = fvec_L2sqr_simd(decoded_vec, dvec, d);
            }
            res.add(j, dis);
        }
    }

    template <class HammingComputer, class SearchResultType>
    void scan_list_polysemous_hc(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        int ht = ivfpq_disk.polysemous_ht;
        size_t n_hamming_pass = 0;

        int code_size = pq.code_size;

        size_t saved_j[8];
        int counter = 0;

        HammingComputer hc(q_code.data(), code_size);

        for (size_t j = 0; j < (ncode / 4) * 4; j += 4) {
            const uint8_t* b_code = codes + j * code_size;

            // Unrolling is a key. Basically, doing multiple popcount
            // operations one after another speeds things up.

            // 9999999 is just an arbitrary large number
            int hd0 = (res.skip_entry(j + 0))
                    ? 99999999
                    : hc.hamming(b_code + 0 * code_size);
            int hd1 = (res.skip_entry(j + 1))
                    ? 99999999
                    : hc.hamming(b_code + 1 * code_size);
            int hd2 = (res.skip_entry(j + 2))
                    ? 99999999
                    : hc.hamming(b_code + 2 * code_size);
            int hd3 = (res.skip_entry(j + 3))
                    ? 99999999
                    : hc.hamming(b_code + 3 * code_size);

            saved_j[counter] = j + 0;
            counter = (hd0 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 1;
            counter = (hd1 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 2;
            counter = (hd2 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 3;
            counter = (hd3 < ht) ? (counter + 1) : counter;

            if (counter >= 4) {
                // process four codes at the same time
                n_hamming_pass += 4;

                float distance_0 = dis0;
                float distance_1 = dis0;
                float distance_2 = dis0;
                float distance_3 = dis0;
                distance_four_codes<PQDecoder>(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0);
                res.add(saved_j[1], dis0 + distance_1);
                res.add(saved_j[2], dis0 + distance_2);
                res.add(saved_j[3], dis0 + distance_3);

                //
                counter -= 4;
                saved_j[0] = saved_j[4];
                saved_j[1] = saved_j[5];
                saved_j[2] = saved_j[6];
                saved_j[3] = saved_j[7];
            }
        }

        for (size_t kk = 0; kk < counter; kk++) {
            n_hamming_pass++;

            float dis = dis0 +
                    distance_single_code<PQDecoder>(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[kk] * pq.code_size);

            res.add(saved_j[kk], dis);
        }

        // process leftovers
        for (size_t j = (ncode / 4) * 4; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }
            const uint8_t* b_code = codes + j * code_size;
            int hd = hc.hamming(b_code);
            if (hd < ht) {
                n_hamming_pass++;

                float dis = dis0 +
                        distance_single_code<PQDecoder>(
                                    pq.M,
                                    pq.nbits,
                                    sim_table,
                                    codes + j * code_size);

                res.add(j, dis);
            }
        }

#pragma omp critical
        { indexIVFPQ_stats.n_hamming_pass += n_hamming_pass; }
    }

    template <class SearchResultType>
    struct Run_scan_list_polysemous_hc {
        using T = void;
        template <class HammingComputer, class... Types>
        void f(const IVFPQDiskScannerT* scanner, Types... args) {
            scanner->scan_list_polysemous_hc<HammingComputer, SearchResultType>(
                    args...);
        }
    };

    template <class SearchResultType>
    void scan_list_polysemous(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        Run_scan_list_polysemous_hc<SearchResultType> r;
        dispatch_HammingComputer(pq.code_size, r, this, ncode, codes, res);
    }
};




namespace{

//compute distance
void compute_precise_dis_simd(const float* query, const float* data, size_t* pos, int D, float* distances, size_t vec_num){
    for(size_t i = 0; i < vec_num; i++){
        distances[i] = fvec_L2sqr_simd(query, data + pos[i] * D, D);
    }
}

void compute_precise_dis_batch(const float* query, const float* data, size_t* pos, int D, float* distances, size_t vec_num){ 
    size_t i = 0;
    for (; i + 4 <= vec_num; i += 4) {
        fvec_L2sqr_batch_4(query, 
                           data + pos[i] * D, 
                           data + pos[i + 1] * D, 
                           data + pos[i + 2] * D, 
                           data + pos[i + 3] * D, 
                           D,
                           distances[i],
                           distances[i + 1],
                           distances[i + 2],
                           distances[i + 3]); 
    }
    
    for (; i < vec_num; i++) {
        distances[i] = fvec_L2sqr_simd(query, data + pos[i] * D, D);
    }
}



//fread
template<class C, bool use_sel>
void disk_full_fread(
    KnnSearchResults<C, use_sel>* res,
    int D,
    size_t len,
    size_t cluster_begin,
    size_t single_offset,
    int real_heap,

    float factor,
    float factor_partial,

    float* heap_sim,
    float* list_sim,
    idx_t* list_ids,

    float* query,
    // file descriptor?
    FILE* disk_data
){
    int stats_compare = 0;
    int stats_rerank = 0;
    auto time_start = std::chrono::high_resolution_clock::now(); //time begin
    std::vector<float> vec(D * len);
    size_t offset = cluster_begin * single_offset;

    if (fseek(disk_data, offset, SEEK_SET) != 0) {
        throw std::runtime_error("Failed to seek in disk file");
    }

    // read data by fread
    size_t read_count = fread(vec.data(), sizeof(float), D * len, disk_data);
    float* cluster_data = vec.data();

    auto time_end = std::chrono::high_resolution_clock::now();   //time end
    indexIVFPQDisk_stats.disk_full_elapsed += time_end - time_start;
    
    time_start = std::chrono::high_resolution_clock::now(); //time begin

    for(size_t i = 0; i < real_heap; i++ ){
        if (list_sim[i] < heap_sim[0] * factor) {   // rerank
            // skip repetitive entries


            float distance = fvec_L2sqr_simd(query, cluster_data + list_ids[i] * D, D);
            res->add(list_ids[i], distance);
            stats_rerank++;
        }
        stats_compare++;
    }

    time_end = std::chrono::high_resolution_clock::now(); //time end
    indexIVFPQDisk_stats.memory_2_elapsed += time_end - time_start;

    indexIVFPQDisk_stats.full_cluster_rerank += stats_rerank;
    indexIVFPQDisk_stats.full_cluster_compare += stats_compare;


}

template<class C, bool use_sel>
void disk_partial_fread(
    KnnSearchResults<C, use_sel>* res,
    int D,
    size_t len,
    size_t cluster_begin,
    size_t single_offset,
    int real_heap,

    float factor,
    float factor_partial,

    float* heap_sim,
    float* list_sim,
    idx_t* list_ids,

    float* query,
    // file descriptor?
    FILE* disk_data)
{
    int stats_compare = 0;
    int stats_rerank = 0;
    std::vector<float> vec(D);
    size_t offset = cluster_begin * single_offset;
    for (size_t i = 0; i < real_heap; i++) {
        if (list_sim[i] < heap_sim[0] * factor_partial) {

            auto time_start = std::chrono::high_resolution_clock::now();  // time begin
            // skip repetitive entries
            //if(res->check_repeat(list_ids[i]))
                //continue;
            
            fseek(disk_data, offset, SEEK_SET);
            fseek(disk_data, list_ids[i]* single_offset, SEEK_CUR);
 
            size_t read_count = fread(vec.data(), sizeof(float), D, disk_data);

            auto time_end = std::chrono::high_resolution_clock::now();
            indexIVFPQDisk_stats.disk_partial_elapsed += time_end - time_start;  // time end

            time_start = std::chrono::high_resolution_clock::now();  // time begin

            float distance = fvec_L2sqr_simd(query, vec.data(), D);
            res->add(list_ids[i], distance);
            //std::cout << "list_ids[i]:" << list_ids[i] << " distance:"<<distance << std::endl;
            stats_rerank++;

            time_end = std::chrono::high_resolution_clock::now();  // time end
            indexIVFPQDisk_stats.memory_2_elapsed += time_end - time_start;
        }

        stats_compare++;
    }

    indexIVFPQDisk_stats.partial_cluster_rerank += stats_rerank;
    indexIVFPQDisk_stats.partial_cluster_compare += stats_compare;


}

template<class C, bool use_sel>
void disk_fully_process(KnnSearchResults<C, use_sel>* res,
                        int D,
                        size_t len,
                        size_t listno,
                        Aligned_Cluster_Info* acInfo,


                        float factor,
                        float factor_partial,

                        float* heap_sim,
                        float* list_sim,
                        idx_t* list_ids,

                        float* query,
                        DiskIOProcessor* diskIOprocessor){
    int stats_compare = 0;
    int stats_rerank = 0;
    auto time_start = std::chrono::high_resolution_clock::now(); //time begin
    std::vector<float> vec(D * len);
    // ********Disk Process Begin**********
    diskIOprocessor->disk_io_all(D, len, listno, vec.data(), acInfo);
    // ********Disk Process End ***********
    float* cluster_data = vec.data();
    for(size_t i = 0; i < len; i++ ){
        if (list_sim[i] < heap_sim[0] * factor) {   // rerank
            float distance = fvec_L2sqr_simd(query, cluster_data + list_ids[i] * D, D);
            res->add(list_ids[i], distance);
            stats_rerank++;
        }
        stats_compare++;
    }

}

template<class C, bool use_sel>
void disk_partially_process(KnnSearchResults<C, use_sel>* res,
                            int D,
                            size_t len,
                            size_t listno,
                            Aligned_Cluster_Info* acInfo,

                            float factor,
                            float factor_partial,

                            float* heap_sim,
                            float* list_sim,
                            idx_t* list_ids,

                            float* query,
                            DiskIOProcessor* diskIOprocessor){
    int stats_compare = 0;
    int stats_rerank = 0;
    std::vector<float> vec(D);
    for (size_t i = 0; i < len; i++) {
        if (list_sim[i] < heap_sim[0] * factor_partial) {

            auto time_start = std::chrono::high_resolution_clock::now();  // time begin
            diskIOprocessor->disk_io_single(D, len, listno, i, vec.data(), acInfo);

            auto time_end = std::chrono::high_resolution_clock::now();
            indexIVFPQDisk_stats.disk_partial_elapsed += time_end - time_start;  // time end

            time_start = std::chrono::high_resolution_clock::now();  // time begin

            float distance = fvec_L2sqr_simd(query, vec.data(), D);
            res->add(list_ids[i], distance);
            //std::cout << "list_ids[i]:" << list_ids[i] << " distance:"<<distance << std::endl;
            stats_rerank++;

            time_end = std::chrono::high_resolution_clock::now();  // time end
            indexIVFPQDisk_stats.memory_2_elapsed += time_end - time_start;
        }

        stats_compare++;
    }

    indexIVFPQDisk_stats.partial_cluster_rerank += stats_rerank;
    indexIVFPQDisk_stats.partial_cluster_compare += stats_compare;


}


template<class C, bool use_sel>
void disk_fully_async(KnnSearchResults<C, use_sel>* res,
                        int D,
                        size_t len,
                        size_t listno,
                        Aligned_Cluster_Info* acInfo,


                        float factor,
                        float factor_partial,

                        float* heap_sim,
                        idx_t* heap_ids,

                        std::shared_ptr<std::vector<float>>& list_sim,
                        std::shared_ptr<std::vector<idx_t>>& list_ids,

                        float* query,
                        DiskIOProcessor* diskIOprocessor){
    int stats_compare = 0;
    int stats_rerank = 0;
    auto time_start = std::chrono::high_resolution_clock::now(); //time begin
    
    Aligned_Cluster_Info* cluster_info = &acInfo[listno];

    //auto buffer = std::make_shared<std::vector<char>>(cluster_info->page_count * diskIOprocessor->page_size);
    std::shared_ptr<AsyncReadRequest> request = std::make_shared<AsyncReadRequest>();
    // TODO 补全request的条件
    // page number
    //request->m_offset = cluster_info->page_start * diskIOprocessor->page_size;
    //request->m_readSize = cluster_info->page_count * diskIOprocessor->page_size;

    request->m_offset = cluster_info->page_start * PAGE_SIZE;
    request->m_readSize = cluster_info->page_count * PAGE_SIZE;
    //request->m_buffer = buffer->data();
    request->m_status = 63;
    request->m_payload = (void*)cluster_info; 
    request->m_success = false;
    request->len = len;
    request->D = D;

    request->rerank_info.factor_fully = factor;
    request->rerank_info.factor_partially = factor_partial;
    request->rerank_info.heap_ids=heap_ids;
    request->rerank_info.heap_sim=heap_sim;
    request->rerank_info.list_ids = list_ids;
    request->rerank_info.list_sim = list_sim;
    request->rerank_info.k = res->k;
    request->rerank_info.key = res->key;
    request->rerank_info.query = query;
    request->rerank_info.ids = res->ids;

    faiss::indexIVFPQDisk_stats.searched_vector_full+=request->len;
    
    //std::cout << " page_count: "<< cluster_info->page_count << " page_start: " << cluster_info->page_start << std::endl;

    request->m_callback = [&](AsyncReadRequest* requested){

        //diskIOprocessor->test();
        //std::cout << "len:" << len << " D:" << D <<" Request:" << requested->m_status << std::endl; 
        //std::cout << "list_sim[0]:" << list_sim[0] << " factor:" << factor<<" Request:" << requested->m_status << std::endl; 
        // for(int i = 0; i < 100; i++){
        //     std::cout << reinterpret_cast<float*>(requested->m_buffer)[i] << std::endl;
        // }
        //
        //std::vector<float> vec(D*len);
        //void* buffer_disk = requested->m_buffer;
        //std::cout << " begin call back " << std::endl;
        KnnSearchResults<C, false> res_callback = {
                /* key */ requested->rerank_info.key,
                /* ids */ requested->rerank_info.ids,
                /* sel */ nullptr,
                /* k */ requested->rerank_info.k,
                /* heap_sim */ requested->rerank_info.heap_sim,
                /* heap_ids */ requested->rerank_info.heap_ids,
                /* nup */ 0};
        
        // for(int i = 0; i < requested->rerank_info.k; i++){
        //     std::cout << "i:" << i << "  heap_sim:" << requested->rerank_info.heap_sim[i] << std::endl;
        // }
        // std::cout << "\n\n\n\n\n";

        idx_t* c_ids = requested->rerank_info.list_ids.get()->data();
        float* c_sim = requested->rerank_info.list_sim.get()->data();

        // std::cout << "Q:";
        // for(int i = 0; i < 128; i++){
        //     std::cout  <<requested->rerank_info.query[i] << "  ";
        // }
        // std::cout << std::endl;
        // std::cout << "D:";
        // for(int i = 0; i < 128; i++){
        //     std::cout  <<(requested->converted_buffer + c_ids[0])[i] << "  ";
        // }
        // std::cout << std::endl << "dis=" << fvec_L2sqr_simd(requested->rerank_info.query, requested->converted_buffer + c_ids[0] * requested->D, requested->D);
        // std::cout << std::endl;
        // std::cout << std::endl;

        //diskIOprocessor->convert_to_float(len, vec.data(), buffer_disk);     
        for(size_t i = 0; i < requested->len; i++ ){
            if (c_sim[i] < requested->rerank_info.heap_sim[0] * requested->rerank_info.factor_fully) {   // rerank
                float distance = fvec_L2sqr_simd(requested->rerank_info.query, requested->converted_buffer + c_ids[i] * requested->D, requested->D);
                res_callback.add(c_ids[i], distance);
                // std::cout << "c_ids:"<<c_ids[i] << "   dis:"<< distance << std::endl;
            }
        }
        //std::cout << "Call Back!!!!" << std::endl;
    };

    // ******** Disk Process Begin **********
    diskIOprocessor->disk_io_all_async(request);
    // ******** Disk Process End   **********
    //diskIOprocessor->submit(1);
}

namespace{

struct PageSegment {
    int start_page;
    int page_count;
    std::vector<int> in_buffer_offsets;
    std::vector<size_t> in_buffer_ids;
    PageSegment() = default;

    // 自定义构造函数，接受三个参数
    PageSegment(int start, int count, const std::vector<int>& offsets, const std::vector<size_t>& ids)
        : start_page(start), page_count(count), 
            in_buffer_offsets(offsets), in_buffer_ids(ids) {}
};

}// anonymous namespace

template<class C, bool use_sel>
void disk_partially_async(KnnSearchResults<C, use_sel>* res,
                        int D,
                        size_t len,
                        size_t listno,
                        Aligned_Cluster_Info* acInfo,


                        float factor,
                        float factor_partial,

                        float* heap_sim,
                        idx_t* heap_ids,

                        std::shared_ptr<std::vector<float>>& list_sim,
                        std::shared_ptr<std::vector<idx_t>>& list_ids,

                        float* query,
                        DiskIOProcessor* diskIOprocessor){
    int stats_compare = 0;
    int stats_rerank = 0;
    auto time_start = std::chrono::high_resolution_clock::now(); //time begin
    std::shared_ptr<std::vector<int>> vector_to_search = std::make_shared<std::vector<int>>();
    
    //std::cout << "heap_sim[0] = " << heap_sim[0] << std::endl;
    for(size_t i = 0; i < len; i++){
        if (list_sim->data()[i] < heap_sim[0] * factor_partial){
            vector_to_search->push_back(i);
        }
    }

    // for(size_t i = 0; i < len; i++){
    //     std::cout << list_sim->data()[i] << " ";
    // }    

    size_t vectors_num = vector_to_search->size();

    //std::cout << "List_no:" << listno << "  pushed:" << vectors_num << " total:"<<len << "\n\n\n"<< std::endl;

    // for(size_t i = 0; i < pages_num; i++){
    //     //TODO Heap_sim is non_initialed
    //     //std::cout << "VECTOR: " << vector_to_search->data()[i] << "  Heap_sim:" << heap_sim[0] << std::endl;
    // }
    //std::cout << "vectors_num:" << vectors_num << std::endl;

    
    std::shared_ptr<std::vector<int>> page_to_search = std::make_shared<std::vector<int>>(vectors_num);
    std::unique_ptr<size_t> vec_page_proj(new size_t[vectors_num]);

    int num_page_to_search = diskIOprocessor->process_page(vector_to_search->data(), page_to_search->data(), 
                                                           vec_page_proj.get(), vectors_num);

    // for(int i = 0; i < vectors_num; i++){
    //     std::cout << "vector->" << i << "   id_rank:"<<vector_to_search->data()[i] <<
    //               "  page: "<< vec_page_proj.get()[i] << std::endl; 
    // }
    
    Aligned_Cluster_Info* cluster_info = &acInfo[listno];

    // for(size_t i = 0; i < num_page_to_search; i++){

    //     std::cout << "Total:" << cluster_info->page_count <<"    num_page_to_search:" << page_to_search->data()[i]<<std::endl;
    // }



    //auto buffer = std::make_shared<std::vector<char>>(cluster_info->page_count * diskIOprocessor->page_size);
    //std::shared_ptr<std::vector<AsyncReadRequest_Partial>> request_p = std::make_shared<std::vector<AsyncReadRequest_Partial>>();
    auto request_p = std::make_shared<std::vector<AsyncReadRequest_Partial>>();

    const int per_page_element = diskIOprocessor->get_per_page_element();
    //std::cout << "per_page_element:" << per_page_element << std::endl;
    const int per_page_vector = per_page_element/D;
    
    // TODO 如果一个页面不能有整数个向量 要跨边界..
    bool not_aligned = false;
    if(per_page_element%D != 0)
        not_aligned = true;

    int begin_page = 0;
    int record_begin_page = 0;

    faiss::indexIVFPQDisk_stats.searched_vector_partial+=vectors_num;

    if (num_page_to_search > 0) {
        // 用于保存合并后的页信息
        std::vector<PageSegment> merged_segments;
        // 合并连续页
        int start_page = page_to_search->at(0);
        int page_count = 1;
        std::vector<int> in_buffer_offsets;  //?
        std::vector<size_t> in_buffer_ids;

        int vec_rank = 0;
        int page_rank = 0;
        int* ptr_vector_to_search = vector_to_search->data();
        int* ptr_page_to_search = page_to_search->data();
        
        // 第一个请求的开始读取的页面为
        begin_page = ptr_page_to_search[0];
        record_begin_page = ptr_page_to_search[0];
        // for(int i = 0; i < num_page_to_search ;i++){
        //     std::cout << "   num_page_to_search:"  << ptr_page_to_search[i] << std::endl;   
        // }

        //std::cout << "begin_page: "<<begin_page << "  record_begin_page:" <<record_begin_page << std::endl;
        // 逐页添加
        while (vec_rank<vectors_num && *(vec_page_proj.get() + vec_rank) == begin_page) {
            int inbuffer = ptr_vector_to_search[vec_rank] * D - per_page_element * record_begin_page;
            //std::cout << "record_begin_page:" << record_begin_page << " begin_page:" << begin_page<< std::endl; 
            // std::cout << "Offset: " << inbuffer << " = " << ptr_vector_to_search[vec_rank] << "*"<< D 
            //          << "-" << per_page_element << "*" << record_begin_page 
            //          << " ptr_vector_to_search->" <<vec_rank << " "<<ptr_vector_to_search[vec_rank]  << std::endl; 
            //std::cout << "Offset: " << inbuffer << std::endl;
            in_buffer_offsets.emplace_back(inbuffer);
            //TODO
            in_buffer_ids.emplace_back(ptr_vector_to_search[vec_rank]);
            vec_rank++;
        }
        page_rank++;

        for (int i = 1; i < num_page_to_search; ++i) {
            int current_page = page_to_search->at(i);
            int previous_page = page_to_search->at(i - 1);

            if (current_page == previous_page + 1) {
                // 如果是连续的页，增加页数
                page_count++;
                begin_page = ptr_page_to_search[page_rank];
                //?????
                while ( vec_rank<vectors_num && *(vec_page_proj.get() + vec_rank) == begin_page) {
                    int inbuffer = ptr_vector_to_search[vec_rank] * D - per_page_element * record_begin_page;
                    
                    //std::cout << "record_begin_page:" << record_begin_page << " begin_page:" << begin_page << std::endl; 
                    // std::cout << "Offset: " << inbuffer << " = " << ptr_vector_to_search[vec_rank] << "*"<< D 
                    //  << "-" << per_page_element << "*" << record_begin_page  
                    //  << " ptr_vector_to_search->" <<vec_rank << " "<<ptr_vector_to_search[vec_rank]  << std::endl; 
                    //std::cout << "Offset: " << inbuffer << std::endl;
                    in_buffer_offsets.emplace_back(inbuffer);
                    in_buffer_ids.emplace_back(ptr_vector_to_search[vec_rank]);
                    vec_rank++;
                }
                page_rank++;
            } else {
                // 保存当前合并段
                merged_segments.emplace_back(start_page, page_count, in_buffer_offsets, in_buffer_ids);

                // 重新初始化新的合并段
                start_page = current_page;
                page_count = 1;
                in_buffer_offsets.clear();
                in_buffer_ids.clear();
                record_begin_page = ptr_page_to_search[page_rank];
                begin_page = ptr_page_to_search[page_rank];
                
                while (vec_rank<vectors_num && *(vec_page_proj.get() + vec_rank) == begin_page) {
                    
                    int inbuffer = ptr_vector_to_search[vec_rank] * D - per_page_element * record_begin_page;
                    //std::cout << "record_begin_page:" << record_begin_page << " begin_page:" << begin_page << std::endl; 
                    // std::cout << "Offset: " << inbuffer << " = " << ptr_vector_to_search[vec_rank] << "*"<< D 
                    //  << "-" << per_page_element << "*" << record_begin_page 
                    //  << " ptr_vector_to_search->" <<vec_rank << " "<<ptr_vector_to_search[vec_rank]  << std::endl; 
                    //std::cout << "Offset: " << inbuffer << std::endl;
                    in_buffer_offsets.emplace_back(inbuffer);
                    in_buffer_ids.emplace_back(ptr_vector_to_search[vec_rank]);
                    vec_rank++;
                }
                page_rank++;
            }
        }
        // 添加最后一个合并段
        merged_segments.emplace_back(start_page, page_count, in_buffer_offsets, in_buffer_ids);

        // for(int i = 0; i < merged_segments.size(); i++){
        //     std::cout << " START_PAGE: " << merged_segments[i].start_page << std::endl;
        //     std::cout << " PAGE_COUNT: " << merged_segments[i].page_count << std::endl;
        //     for(int j = 0; j < merged_segments[i].in_buffer_offsets.size(); j++){
        //         std::cout <<"   offsets: " << merged_segments[i].in_buffer_offsets[j] << std::endl;
        //         std::cout <<"   ids    : " << merged_segments[i].in_buffer_ids[j] << std::endl;
        //     }
        // }
        // std::cout << std::endl;

        size_t global_start = cluster_info->page_start;

        // 第二步：调整 request_p 的大小，并初始化每个请求
        request_p->resize(merged_segments.size());
        for (size_t i = 0; i < merged_segments.size(); ++i) {
            const auto& segment = merged_segments[i];
            AsyncReadRequest_Partial& request = (*request_p)[i];
            //request.m_offset = (global_start + segment.start_page) * diskIOprocessor->page_size;
            //request.m_readSize = segment.page_count * diskIOprocessor->page_size;

            request.m_offset = (global_start + segment.start_page) * PAGE_SIZE;
            request.m_readSize = segment.page_count * PAGE_SIZE;
            request.D = D;
            request.in_buffer_offsets = segment.in_buffer_offsets;
            request.in_buffer_ids = segment.in_buffer_ids;
            
            //**************  request information begin ******************
            // 
            request.rerank_info.factor_fully = factor;
            request.rerank_info.factor_partially = factor_partial;
            request.rerank_info.heap_ids=heap_ids;
            request.rerank_info.heap_sim=heap_sim;
            request.rerank_info.list_ids = list_ids;
            request.rerank_info.list_sim = list_sim;
            request.rerank_info.k = res->k;
            request.rerank_info.key = res->key;
            request.rerank_info.query = query;
            request.rerank_info.ids = res->ids;
            //**************  request information end ******************

            request.m_callback_calculation = [&](AsyncReadRequest_Partial* requested, 
                                                std::vector<float>& global_dis, std::vector<size_t>& global_ids){
                
                int* element_offsets = requested->in_buffer_offsets.data();
                size_t* element_ids = requested->in_buffer_ids.data();
                float distance;

                // std::cout << "Q:";
                // for(int i = 0; i < 128; i++){
                //     std::cout  <<requested->rerank_info.query[i] << "  ";
                // }
                // std::cout << std::endl;
                // std::cout << "D:";
                // for(int i = 0; i < 128; i++){
                //     std::cout  <<(requested->converted_buffer + element_offsets[0])[i] << "  ";
                // }
                // std::cout << std::endl << "dis=" << fvec_L2sqr_simd(requested->rerank_info.query, requested->converted_buffer + element_offsets[0] * requested->D, requested->D);
                // std::cout << std::endl;
                // std::cout << std::endl;

                for(int i = 0; i < requested->in_buffer_offsets.size(); i++){
                    distance = fvec_L2sqr_simd(requested->rerank_info.query, requested->converted_buffer + element_offsets[i], requested->D);
                    //distance = fvec_L2sqr_simd(requested->converted_buffer + element_offsets[0], requested->converted_buffer + element_offsets[i], requested->D);
                    //distance = fvec_L2sqr_simd(requested->rerank_info.query, requested->rerank_info.query, requested->D);
                    
                    //std::cout << "Key:"  << requested->rerank_info.key<< "  element_offsets:" <<  element_offsets[i] <<"  ID:" << element_ids[i] <<"   DIS:" << distance << std::endl;
                    //std::cout << element_offsets[i] << " ";
                    // 需要一个地方有存储的ID？ 
                    global_ids.emplace_back(element_ids[i]);
                    global_dis.emplace_back(distance);
                }

                //std::cout << "\n";
            };

            request.m_callback = [&](AsyncReadRequest_Partial* requested,
                                      std::vector<float>& global_dis, std::vector<size_t>& global_ids){
                KnnSearchResults<C, false> res_callback = {
                        /* key */ requested->rerank_info.key,
                        /* ids */ requested->rerank_info.ids,
                        /* sel */ nullptr,
                        /* k */ requested->rerank_info.k,
                        /* heap_sim */ requested->rerank_info.heap_sim,
                        /* heap_ids */ requested->rerank_info.heap_ids,
                        /* nup */ 0};
                // for(size_t i = 0; i < global_dis.size(); i++ ){
                //     std::cout << " global_ids->i: " << global_ids[i]<< "  global_dis[i]:" <<  global_dis[i] << std::endl;
                // }
                // for(size_t i = 0; i <requested->rerank_info.k; i++ ){
                //     std::cout << " heap_sim->i: " << requested->rerank_info.heap_sim[i]<< 
                //     "  heap_ids[i]:" <<  requested->rerank_info.heap_ids[i] << std::endl;
                // }

                for(size_t i = 0; i < global_dis.size(); i++ ){
                    res_callback.add(global_ids[i], global_dis[i]);
                }
                
            };
        }

        // 提交请求
        diskIOprocessor->disk_io_partial_async(request_p);
    }
}


// if warmed up
template<class C, bool use_sel>
void memory_all_scan(KnnSearchResults<C, use_sel>* res, 
                        int D, 
                        const float* vec_data, 
                        float* query, 
                        const idx_t* list_ids, 
                        size_t len){
    std::vector<size_t> temp_pos(len);
    for(int i = 0; i < len; i++){
        temp_pos[i] = i;
    }
    std::vector<float> distances(len);
    float* p_dis = distances.data();

    compute_precise_dis_simd(query, vec_data, temp_pos.data(), D, distances.data(),len);
    for(size_t i = 0; i < len; i++){
        res->add(i, p_dis[i]);
        //std::cout <<"i:" <<i<<"list_ids[i]:" << list_ids[i] << " distance:"<<distances.data()[i] << std::endl;
        // if(distances.data()[i] > 1000000)
        // {
        //     for(int j = 0; j < D; j++){
        //         std::cout << *(vec_data + i*D +j) << " ";
        //     }
        //     std::cout<< std::endl;
        // }
    }
}

template<class C, bool use_sel>
void memory_selective_scan(
    KnnSearchResults<C, use_sel>* res, 
        int D, 
        const float* vec_data, 
        float* query, 
        float* list_sim,
        const idx_t* list_ids,
        float* heap_sim,
        size_t len, 
        float factor){
    // std::vector<size_t> positions(len);
    // size_t* p_pos = positions.data();
    // size_t* p_begin = p_pos;
    // // 2. find the smaller one of threshold
    // float threshold = C::cmp(list_sim[0],heap_sim_0)?heap_sim_0:list_sim[0];
    // // 3. record vectors number 
    // size_t vec_num = 0;

    // for(size_t i = 0; i < len; i++ ){
    //     if (list_sim[i] < threshold * factor) {
    //         *p_pos = i;
    //         p_pos++;
    //         vec_num++;
    //     }
    // }
    
    // std::vector<float> distances(vec_num);
    // float* p_dis = distances.data();
    // std::vector<size_t> temp_pos(vec_num);
    // for(int i = 0; i < vec_num; i++){
    //     temp_pos[i] = i;
    // }

    // compute_precise_dis_batch(query, vec_data, temp_pos.data(), D, distances.data(),vec_num);
    
    // for(size_t i = 0; i < vec_num; i++){
    //     res->add(list_ids[p_begin[i]], p_dis[i]);
    // }
    for (size_t i = 0; i < len; i++) {
        if (list_sim[i] < heap_sim[0] * factor) {
            float distance = fvec_L2sqr_simd(query, vec_data + i*D, D);
            res->add(i, distance);
            //std::cout << "list_ids[i]:" << list_ids[i] << " distance:"<<distance << std::endl;
        }
     }
}

}

/* We put as many parameters as possible in template. Hopefully the
 * gain in runtime is worth the code bloat.
 *
 * C is the comparator < or >, it is directly related to METRIC_TYPE.
 *
 * precompute_mode is how much we precompute (2 = precompute distance tables,
 * 1 = precompute pointers to distances, 0 = compute distances one by one).
 * Currently only 2 is supported
 *
 * use_sel: store or ignore the IDSelector
 */
//#define DISK_READ
//#define DISK_FREAD
//#define DISK_MMAP
//#define DISK_IFSTREAM
//#define DISK_MMAP_ALL

template <MetricType METRIC_TYPE, class C, class PQDecoder, bool use_sel>
struct IVFPQDiskScanner : IVFPQDiskScannerT<idx_t, METRIC_TYPE, PQDecoder>,
                      InvertedListScanner {
    int precompute_mode;
    const IDSelector* sel;
    float* raw_query  = nullptr;

#ifdef DISK_FREAD
    mutable FILE* disk_data = nullptr;    // fread
#endif
    
    //std::unique_ptr<DiskIOProcessor> diskIOprocessor;
    //std::shared_ptr<DiskIOProcessor> diskIOprocessor;
    DiskIOProcessor* diskIOprocessor;

    IVFPQDiskScanner(
            const IndexIVFPQDisk& ivfpq_disk,
            bool store_pairs,
            int precompute_mode,
            const IDSelector* sel)
            : IVFPQDiskScannerT<idx_t, METRIC_TYPE, PQDecoder>(ivfpq_disk, nullptr),
              precompute_mode(precompute_mode),sel(sel) {
#ifdef DISK_FREAD
        disk_data = fopen(ivfpq_disk.get_disk_path().c_str(), "rb");
        if (!disk_data) {
            throw std::runtime_error("IVFPQDiskScanner: Failed to open disk file for reading");
        }
#endif
        // diskIOprocessor.reset(ivfpq_disk.get_DiskIOSearchProcessor());
        // diskIOprocessor.get()->initial();
        diskIOprocessor = ivfpq_disk.get_DiskIOSearchProcessor();
        diskIOprocessor->initial();  // 使用箭头操作符访问成员函数

        this->store_pairs = store_pairs;
        this->keep_max = is_similarity_metric(METRIC_TYPE);
    }
    
    ~IVFPQDiskScanner() {
#ifdef DISK_FREAD
    if (disk_data)        
            fclose(disk_data);    // fread
#endif
    }

    void async_submit(int num) override{
        //std::cout << "Ready to submit: " << num << std::endl;
        diskIOprocessor->submit(num);
    }

    void set_query(const float* query) override {
        this->raw_query = const_cast<float*>(query);   // ? const seems to be not very suitable here
        this->init_query(query);
        // reset hashtable
        //this->visited.clear();
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        this->init_list(list_no, coarse_dis, precompute_mode);
    }

    float distance_to_code(const uint8_t* code) const override {
        assert(precompute_mode == 2);
        float dis = this->dis0 +
                distance_single_code<PQDecoder>(
                            this->pq.M, this->pq.nbits, this->sim_table, code);
        return dis;
    }

    size_t scan_codes(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float* heap_sim,
            idx_t* heap_ids,
            size_t k) const override {

        KnnSearchResults<C, use_sel> res = {
                /* key */ this->key,
                /* ids */ this->store_pairs ? nullptr : ids,
                /* sel */ this->sel,
                /* k */ k,
                /* heap_sim */ heap_sim,
                /* heap_ids */ heap_ids,
                /* nup */ 0};

        auto time_start = std::chrono::high_resolution_clock::now();      // time begin
        size_t list_code_num = this->ivfpq_disk.invlists->list_size(this->list_no);
        const idx_t* list_entry = this->ivfpq_disk.invlists->get_ids(this->list_no);   // check whether an entry has been scanned
        
        // store local result
        //std::vector<float> list_sim(list_code_num);
        //std::vector<idx_t> list_ids(list_code_num);
        std::shared_ptr<std::vector<float>> list_sim = std::make_shared<std::vector<float>>(list_code_num);
        std::shared_ptr<std::vector<idx_t>> list_ids = std::make_shared<std::vector<idx_t>>(list_code_num);

        if (this->ivfpq_disk.metric_type == METRIC_INNER_PRODUCT) {
            heap_heapify<HeapForIP>(list_code_num, list_sim->data(), list_ids->data());
        } else {
            heap_heapify<HeapForL2>(list_code_num, list_sim->data(), list_ids->data());
        }

        int real_heap = 0;

        float factor = this->ivfpq_disk.get_estimate_factor();
        float factor_partial = this->ivfpq_disk.get_estimate_factor_partial();

        auto time_end = std::chrono::high_resolution_clock::now();       // time end
        indexIVFPQDisk_stats.others_elapsed += time_end - time_start;
        
        int is_cached = this->ivfpq_disk.diskInvertedListHolder.is_cached(this->list_no);
        if(is_cached >= 0){
            indexIVFPQDisk_stats.cached_list_access++;
            if(this->load_strategy == FULLY){
                const float* vecs = reinterpret_cast<const float*>(this->ivfpq_disk.diskInvertedListHolder.get_cache_data(is_cached));
                //std::cout << "In cache:"<<this->ivfpq_disk.diskInvertedListHolder.clusters_length[is_cached];
                //std::cout << " Acctually:" << list_code_num << std::endl;
                //std::cout << "cached:" << is_cached << " block" << std::endl;
                memory_all_scan(&res, this->ivfpq_disk.get_dim(), vecs, raw_query,list_entry,list_code_num);
                //std::cout << "scan finished"<< std::endl;
                return res.nup;
            }
            
        }

        time_start = std::chrono::high_resolution_clock::now();      // time begin
            
        if (this->polysemous_ht > 0) {
            assert(precompute_mode == 2);
            this->scan_list_polysemous(ncode, codes, res);
        } else if (precompute_mode == 2) {
            real_heap = this->scan_list_with_table(ncode, codes, list_sim->data(), list_ids->data(),
                                                    heap_ids, list_entry, k, res);
        } else if (precompute_mode == 1) {
            this->scan_list_with_pointer(ncode, codes, res);
        } else if (precompute_mode == 0) {
            this->scan_on_the_fly_dist(ncode, codes, res);
        } else {
            FAISS_THROW_MSG("bad precomp mode");
        }

        time_end = std::chrono::high_resolution_clock::now();       // time end
        indexIVFPQDisk_stats.memory_1_elapsed += time_end - time_start;

        if(is_cached >= 0 && load_strategy == PARTIALLY){
            std::cout << "Cache Scan!" << std::endl;
            const float* vecs = reinterpret_cast<const float*>(this->ivfpq_disk.diskInvertedListHolder.get_cache_data(is_cached));
            memory_selective_scan(&res, 
                                    this->ivfpq_disk.get_dim(),
                                    vecs, 
                                    raw_query,
                                    list_sim->data(),
                                    list_entry,
                                    heap_sim,
                                    list_code_num,
                                    factor_partial);
            return res.nup;
        }

        if(real_heap != 0){
            size_t cluster_begin = this->ivfpq_disk.get_cluster_location(this->key);
            size_t len = this->ivfpq_disk.get_cluster_len(this->key);
            size_t single_offset = this->ivfpq_disk.get_vector_offset();
            int D = this->ivfpq_disk.get_dim();
            float* query = raw_query;

            if (this->load_strategy == FULLY) {
#ifdef USING_SYNC
                disk_fully_process(&res, D, len,
                                    list_no,
                                    this->ivfpq_disk.aligned_cluster_info,
                                    factor,
                                    factor_partial,
                                    heap_sim,
                                    list_sim.data(),
                                    list_ids.data(),
                                    query,
                                    this->diskIOprocessor.get());
#else
                disk_fully_async(&res, D, len,
                                list_no,
                                this->ivfpq_disk.aligned_cluster_info,
                                factor,
                                factor_partial,
                                heap_sim,
                                heap_ids,
                                list_sim,
                                list_ids,
                                query,
                                this->diskIOprocessor);           
#endif
        
#ifdef USING_SYNC
        disk_full_fread(&res,
                        D,
                        len,
                        cluster_begin,
                        single_offset,
                        real_heap,
                        factor,
                        factor_partial,
                        heap_sim,
                        list_sim.data(),
                        list_ids.data(),
                        query,
                        this->disk_data);

    
#endif
    } else {
        // disk_partially_process(&res, D, len,list_no,
        //                         this->ivfpq_disk.aligned_cluster_info,
        //                         factor,
        //                         factor_partial,
        //                         heap_sim,
        //                         list_sim.data(),
        //                         list_ids.data(),
        //                         query,
        //                         this->diskIOprocessor.get());
#ifdef USING_SYNC
        disk_partial_fread(&res,
                           D,
                           len,
                           cluster_begin,
                           single_offset,
                           real_heap,
                           factor,
                           factor_partial,
                           heap_sim,
                           list_sim.data(),
                           list_ids.data(),
                           query,
                           this->disk_data);
#else
        disk_partially_async(&res, D, len,
                                    list_no,
                                    this->ivfpq_disk.aligned_cluster_info,
                                    factor,
                                    factor_partial,
                                    heap_sim,
                                    heap_ids,
                                    list_sim,
                                    list_ids,
                                    query,
                                    this->diskIOprocessor);  
#endif
    }
        }
        return res.nup;
    }

    void scan_codes_range(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& rres) const override {
        RangeSearchResults<C, use_sel> res = {
                /* key */ this->key,
                /* ids */ this->store_pairs ? nullptr : ids,
                /* sel */ this->sel,
                /* radius */ radius,
                /* rres */ rres};

        if (this->polysemous_ht > 0) {
            assert(precompute_mode == 2);
            this->scan_list_polysemous(ncode, codes, res);
        } else if (precompute_mode == 2) {
            this->scan_list_with_table(ncode, codes, res);
        } else if (precompute_mode == 1) {
            this->scan_list_with_pointer(ncode, codes, res);
        } else if (precompute_mode == 0) {
            this->scan_on_the_fly_dist(ncode, codes, res);
        } else {
            FAISS_THROW_MSG("bad precomp mode");
        }
    }
};

template <class PQDecoder, bool use_sel>
InvertedListScanner* get_InvertedListScanner1(
        const IndexIVFPQDisk& index,
        bool store_pairs,
        const IDSelector* sel) {
    if (index.metric_type == METRIC_INNER_PRODUCT) {
        return new IVFPQDiskScanner<
                METRIC_INNER_PRODUCT,
                CMin<float, idx_t>,
                PQDecoder,
                use_sel>(index, store_pairs, 2, sel);
    } else if (index.metric_type == METRIC_L2) {
        return new IVFPQDiskScanner<
                METRIC_L2,
                CMax<float, idx_t>,
                PQDecoder,
                use_sel>(index, store_pairs, 2, sel);
    }
    return nullptr;
}

template <bool use_sel>
InvertedListScanner* get_InvertedListScanner2(
        const IndexIVFPQDisk& index,
        bool store_pairs,
        const IDSelector* sel) {
    if (index.pq.nbits == 8) {
        return get_InvertedListScanner1<PQDecoder8, use_sel>(
                index, store_pairs, sel);
    } else if (index.pq.nbits == 16) {
        return get_InvertedListScanner1<PQDecoder16, use_sel>(
                index, store_pairs, sel);
    } else {
        return get_InvertedListScanner1<PQDecoderGeneric, use_sel>(
                index, store_pairs, sel);
    }
}

} // anonymous namespace

InvertedListScanner* IndexIVFPQDisk::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel) const {
    if (sel) {
        return get_InvertedListScanner2<true>(*this, store_pairs, sel);
    } else {
        return get_InvertedListScanner2<false>(*this, store_pairs, sel);
    }
    return nullptr;
}
namespace{

    template <typename ValueType>
    DiskIOProcessor* get_DiskIOBuildProcessor_2(std::string& disk_path, size_t d, size_t ntotal){
        return new IVF_DiskIOBuildProcessor<ValueType>(disk_path, d, ntotal);

    }


    template <typename ValueType>
    DiskIOProcessor* get_DiskIOSearchProcessor_2(const std::string& disk_path, const size_t d) {
    #ifndef USING_ASYNC
        return new IVF_DiskIOSearchProcessor<ValueType>(disk_path, d);
    #else 
        return new IVF_DiskIOSearchProcessor_Async<ValueType>(disk_path, d);
    #endif
    }
}



DiskIOProcessor* IndexIVFPQDisk::get_DiskIOBuildProcessor() {

    if(this->valueType == "float"){
         return get_DiskIOBuildProcessor_2<float>(this->disk_path, d, ntotal);
    }else if(this->valueType == "uint8"){
        return get_DiskIOBuildProcessor_2<uint8_t>(this->disk_path, d, ntotal);
    }else if(this->valueType == "int16"){
         return get_DiskIOBuildProcessor_2<int16_t>(this->disk_path, d, ntotal);
    }else{
        FAISS_THROW_MSG("Unsupported type");
    }
}

DiskIOProcessor* IndexIVFPQDisk::get_DiskIOSearchProcessor() const{
    if(this->valueType == "float"){
        return get_DiskIOSearchProcessor_2<float>(this->disk_path, d);
    }else if(this->valueType == "uint8"){
        return get_DiskIOSearchProcessor_2<uint8_t>(this->disk_path, d);
    }else if(this->valueType == "int16"){
        return get_DiskIOSearchProcessor_2<int16_t>(this->disk_path, d);
    }else{
        FAISS_THROW_FMT("Unsupported type %s", this->valueType.c_str());

    }

}


IndexIVFPQDiskStats indexIVFPQDisk_stats;

void IndexIVFPQDiskStats::reset() {
    memset(this, 0, sizeof(*this));
}

} // namespace faiss