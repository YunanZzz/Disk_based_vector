#include <faiss/IndexIVFPQDisk2.h>

#include <omp.h>
#include <cstdint>
#include <memory>
#include <mutex>

#include <algorithm>
#include <map>
#include <queue>
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
#include <iomanip>
#include <string>

#define USING_ASYNC
//#define USING SYNC



namespace faiss{

IndexIVFPQDisk2::IndexIVFPQDisk2(
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

    // We want a new type of InvertedList here.
    if (own_invlists && invlists) {
        delete invlists;
    }
    // Assign a new ClusteredArrayInvertedLists instance
    invlists = new ClusteredArrayInvertedLists(nlist, code_size);
    own_invlists = true;  // Ensure the IndexIVF class takes ownership


    //FAISS_ASSERT_MSG(valueType!="float" && valueType!= "uint8" && valueType!= "int16", "Unsupported value type");

    this->aligned_cluster_info = nullptr;
    clusters = nullptr;
    len = nullptr;
}

IndexIVFPQDisk2::IndexIVFPQDisk2() {}

IndexIVFPQDisk2::~IndexIVFPQDisk2() {
    if(clusters != nullptr)
        delete[] clusters;
    if(len != nullptr)
        delete[] len;
    if(aligned_cluster_info!=nullptr)
        delete[] aligned_cluster_info;
}

void IndexIVFPQDisk2::train_graph(){
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

void IndexIVFPQDisk2::load_hnsw_centroid_index(){
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

void SOAR2(const float* xb, 
          const float* centroidData,
          const idx_t* coarse_idx,  // [nb * k], the candidate centroid IDs
          idx_t* assign,           // [nb * num_centroid] output (chosen IDs)
          size_t nb,               // number of vectors
          size_t d,                // dimension
          size_t k,                // how many centroid candidates per vector
          size_t num_centroid)     // how many we actually pick
{
    if (num_centroid > k) {
        fprintf(stderr, 
                "ERROR: num_centroid (%zu) cannot exceed k (%zu)\n", 
                num_centroid, k);
        return;
    }

#pragma omp parallel for
    for (size_t vecIdx = 0; vecIdx < nb; vecIdx++) 
    {
        // 1) Load the current vector
        std::vector<double> vec = getVector(xb, vecIdx, d);

        // 2) Find the primary centroid (closest in L2 distance among the k)
        double minDist = std::numeric_limits<double>::max();
        idx_t bestCentroidID = 0;

        for (size_t i = 0; i < k; i++) {
            idx_t cID = coarse_idx[vecIdx * k + i];
            std::vector<double> cvec = getCentroidVector(centroidData, cID, d);

            // L2 distance squared = dot(residual, residual)
            std::vector<double> tmpResidual = calculateResidual(vec, cvec, d);
            double dist = dotProduct(tmpResidual, tmpResidual, d);  
            if (dist < minDist) {
                minDist = dist;
                bestCentroidID = cID;
            }
        }

        // Store the primary centroid
        assign[vecIdx * num_centroid + 0] = bestCentroidID;

        // 3) Compute the "primary residual"
        std::vector<double> primaryCentroid = getCentroidVector(centroidData, bestCentroidID, d);
        std::vector<double> primaryResidual = calculateResidual(vec, primaryCentroid, d);
        double primaryResidualNorm = vectorNorm(primaryResidual, d);

        // 4) Pick the remaining (num_centroid - 1) by smallest cosine similarity
        //    to that same primaryResidual
        std::unordered_set<idx_t> used;
        used.insert(bestCentroidID);

        for (size_t pick = 1; pick < num_centroid; pick++) {
            double minCosSim = std::numeric_limits<double>::max();
            idx_t bestSecID  = bestCentroidID;

            for (size_t i = 0; i < k; i++) {
                idx_t cID = coarse_idx[vecIdx * k + i];
                if (used.count(cID) > 0) {
                    // already picked this centroid
                    continue;
                }

                // compute residual wrt. cID
                std::vector<double> cvec = getCentroidVector(centroidData, cID, d);
                std::vector<double> currentResidual = calculateResidual(vec, cvec, d);
                double currentResidualNorm = vectorNorm(currentResidual, d);

                // Cosine similarity = dot(primaryResidual, currentResidual)
                //                    / (|primaryResidual| * |currentResidual|)
                // + small epsilon to avoid division by zero
                double cosSim = dotProduct(primaryResidual, currentResidual, d) / 
                                (primaryResidualNorm * currentResidualNorm + 1e-12);

                if (cosSim < minCosSim) {
                    minCosSim = cosSim;
                    bestSecID = cID;
                }
            }

            // Fallback if for some reason nothing changed
            if (bestSecID == bestCentroidID && used.size() < k) {
                // pick anything not used yet
                for (size_t i = 0; i < k; i++) {
                    idx_t cID = coarse_idx[vecIdx * k + i];
                    if (!used.count(cID)) {
                        bestSecID = cID;
                        break;
                    }
                }
            }

            // Store it
            assign[vecIdx * num_centroid + pick] = bestSecID;
            used.insert(bestSecID);
        }
    } // end parallel for
}



void IndexIVFPQDisk2::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    idx_t k = this->assign_replicas;
    //idx_t k = 5;
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n * k * 2]);
    printf("IndexIVFPQDisk::add_with_ids::k=%ld\n",k);
    float* D = new float[k * n * 2];

    if (n * k * 2 < 5000) 
    //if(1)
    {
        //quantizer->assign(n, x, coarse_idx.get(), k);
        quantizer->search(n, x, k * 2, D, coarse_idx.get());
    } else {
        printf("IndexIVFPQDisk2::add_with_ids::hnsw way, n=%ld, k=%ld \n", n, k * 2);

        centroid_index->hnsw.efSearch = 400;
        centroid_index->search(n, x, k * 2, D, coarse_idx.get());
        // quantizer->assign(n, x, coarse_idx.get(), k);
    }
    if (k != 1) {
        printf("Using SOAR\n");
        std::unique_ptr<idx_t[]> assign(new idx_t[n * k]);
        IndexFlat* flat_quantizer = dynamic_cast<IndexFlat*>(quantizer);
        float* centroidData = flat_quantizer->get_xb();
        // SOAR(x, centroidData, coarse_idx.get(), assign.get(), n, d, k * 2);
        SOAR2(x, centroidData, coarse_idx.get(), assign.get(), n, d, k * 2, k);
        add_core(n, x, xids, assign.get());
        return;
    }
    //     if (k != 1) {
    //     adjustReplica(n, k, coarse_idx.get(), D);
    // }
    add_core(n, x, xids, coarse_idx.get());
}

void IndexIVFPQDisk2::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    add_core_o(n, x, xids, nullptr, coarse_idx, inverted_list_context);
    initial_location(n, x);
}

//#define BUILD_IN_MEMORY

#ifdef BUILD_IN_MEMORY
namespace{
    void add_original_data(const float* data, ArrayInvertedLists* build_invlists, ClusteredArrayInvertedLists* c_array_invlists, size_t old_total, size_t d) {

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < c_array_invlists->nlist; ++i) {
            size_t new_count = c_array_invlists->ids[i].size();
            size_t old_count = build_invlists->ids[i].size();

            for (size_t j = old_count; j < new_count; ++j) {
                idx_t id = c_array_invlists->ids[i][j];
                const float* vector_to_add = &data[(id-old_total) * d];
                build_invlists->add_entry(i, id, (uint8_t*)vector_to_add);  // 假设 ArrayInvertedLists 提供 add_vector 方法来添加向量
            }
        }
    }

}
#endif

void IndexIVFPQDisk2::initial_location(idx_t n, const float* data) {
    if (!invlists) {
        throw std::runtime_error("invlists is not initialized.");
    }

    // Cast invlists to ArrayInvertedLists to access the underlying data
    ClusteredArrayInvertedLists* array_invlists = dynamic_cast<ClusteredArrayInvertedLists*>(invlists);
    if (!array_invlists) {
        throw std::runtime_error("invlists is not of type ClusteredArrayInvertedLists.");
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
        // update maps
        array_invlists->add_maps(i, len[i]);
    }
    if(verbose){
        printf("Cluster info initialized!");
    }

    std::unique_ptr<DiskIOProcessor> io_processor(get_DiskIOBuildProcessor());
    //get_DiskIOBuildProcessor();
    // reorg it at last, and save it to file.

    //reorganize_vectors(n, data, tmp_clusters ,tmp_len);
    ClusteredArrayInvertedLists* c_array_invlists = dynamic_cast<ClusteredArrayInvertedLists*>(invlists);
    if (!c_array_invlists) {
        throw std::runtime_error("invlists is not of type ArrayInvertedLists.");
    }
    bool do_in_list_cluster = false;
    bool keep_in_disk = false;
    this->actual_batch_num++;
    if(this->actual_batch_num == this->add_batch_num){
        do_in_list_cluster = true;
        keep_in_disk = true;
    }
        
    std::cout << "batch:" << this->actual_batch_num << std::endl;
#ifndef BUILD_IN_MEMORY 
    if(io_processor->reorganize_vectors_2(n,
                                    data,
                                    tmp_clusters,
                                    tmp_len,
                                    clusters,
                                    len,
                                    aligned_cluster_info,
                                    nlist,
                                    c_array_invlists,
                                    do_in_list_cluster)){
        this->disk_path = disk_path + ".clustered";
    }
#else
    if(this->build_invlists == nullptr){
        this->build_invlists = new ArrayInvertedLists(nlist, d*sizeof(float));
    }
    add_original_data(data, build_invlists, c_array_invlists, this->ntotal-n, d);

    if(io_processor->reorganize_vectors_in_memory(n,
                                    data,
                                    tmp_clusters,
                                    tmp_len,
                                    clusters,
                                    len,
                                    aligned_cluster_info,
                                    nlist,
                                    c_array_invlists,
                                    build_invlists,
                                    do_in_list_cluster,
                                    keep_in_disk)){
        this->disk_path = disk_path + ".clustered";
    }
#endif


    // if(reorganize_lists && do_in_list_cluster){
    //     io_processor->reorganize_list(*quantizer, c_array_invlists, aligned_cluster_info, clusters, len, nlist);
    // }

    if(select_lists && do_in_list_cluster){
        // assume it's ClusteredArrayInvertedLists

        // TODO make sure pq.code_size is right
        std::cout << "pq_size = " << this->pq.code_size << "  idx_t = " << sizeof(faiss::idx_t)<< "  size_t = " << sizeof(size_t) << "\n";

        //size_t entry_size = this->pq.code_size + sizeof(faiss::idx_t) + sizeof(size_t);
        // No need store map
        size_t entry_size = this->pq.code_size + sizeof(faiss::idx_t) + 0;
        this->aligned_inv_info = new Aligned_Invlist_Info[nlist];

        //io_processor->organize_select_list(this->pq.code_size, entry_size, c_array_invlists, aligned_inv_info, nlist, this->select_lists_path);
        io_processor->organize_select_list(this->pq.code_size, entry_size, c_array_invlists, aligned_inv_info, nlist, this->disk_path);

    }

    // if(do_in_list_cluster){
    //     train_graph();
    // }

    if(tmp_clusters != nullptr){
        delete[] tmp_clusters;
        delete[] tmp_len;
    }

}


/*  需要一个新的reorganize_vectors函数
    如果是这是第一批写入文件的向量：
        每个list进行一下聚类（把相近的向量放在一起，同时要重新组织invlist->ids和invlist->codes)
    如果在之前已经存在向量了：
        读入之前的list向量，和新的向量进行聚类操作（重新组织invlist->ids和invlist->codes）

    analysis:
        1. 一个list大概300向量，所以对单个list进行聚类应该时间不会很多
        2. 每次处理一个聚类，所以内存开销为当前处理的向量+之前的向量

    optimization:
        1. 提前得知需要处理的批次，只在最后一批时，对每个list进行聚类

    utilization: 聚类操作API
    // Clustering clus(d, nlist, cp);
    // IndexFlatL2 assigner(d);
    // clus.train(n, x, assigner);
    // 如果要保存聚类中心 eg:
    // quantizer->add(nlist, clus.centroids.data());
*/

// Method to reorganize vectors based on clustering
void IndexIVFPQDisk2::reorganize_vectors(idx_t n, const float* data, size_t* old_clusters, size_t* old_len) {

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

namespace{
    // void in_list_map_reassign(size_t n, float* nx, Index& index, ClusteredArrayInvertedLists* invlists, size_t list_no){
    //     std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n * 1]);
    //     index.assign(n, nx, coarse_idx.get(), 1);

    //     std::vector<float> reordered_nx(n * index.d); // Temporary buffer for reordered nx

    //     // Track position within each cluster
    //     std::vector<size_t> cluster_counts(index.ntotal, 0);
    //     for (size_t i = 0; i < n; ++i) {
    //         cluster_counts[coarse_idx[i]]++;
    //     }

    //     std::vector<size_t> cluster_offsets(index.ntotal, 0);
    //     size_t cumulative_offset = 0;
    //     for (size_t cluster_id = 0; cluster_id < index.ntotal; ++cluster_id) {
    //         cluster_offsets[cluster_id] = cumulative_offset;
    //         cumulative_offset += cluster_counts[cluster_id];
    //     }

    //     for (size_t i = 0; i < n; ++i) {
    //         size_t cluster_id = coarse_idx[i];
    //         size_t new_pos = cluster_offsets[cluster_id]++; // Cumulative position within entire list
    //         invlists->updata_inlist_map(list_no, i, new_pos);
    //         std::memcpy(&reordered_nx[new_pos * index.d], &nx[i * index.d], index.d * sizeof(float));
    //     }
    //     // copy back
    //     std::memcpy(nx, reordered_nx.data(), n * index.d * sizeof(float));

    // }

}


// Method to set the disk path and open read stream
void IndexIVFPQDisk2::set_disk_read(const std::string& diskPath) {
    disk_path = diskPath;
    if (disk_data_read.is_open()) {
        disk_data_read.close();
    }
    disk_data_read.open(disk_path, std::ios::binary);
    if (!disk_data_read.is_open()) {
        throw std::runtime_error("IndexIVFPQDisk2: Failed to open disk file for reading");
    }
}

// Method to set the disk path and open write stream
void IndexIVFPQDisk2::set_disk_write(const std::string& diskPath) {
    disk_path = diskPath;
    if (disk_data_write.is_open()) {
        disk_data_write.close();
    }
    disk_data_write.open(disk_path, std::ios::binary);
    if (!disk_data_write.is_open()) {
        throw std::runtime_error("IndexIVFPQDisk2: Failed to open disk file for writing");
    }
}

void IndexIVFPQDisk2::load_from_offset(size_t list_no, size_t offset, float* original_vector) {
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

void IndexIVFPQDisk2::load_clusters(size_t list_no, float* original_vectors) {
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


        // int valid_size = 0;
        // int cul_vec = 0;
        // for (size_t i = 0; i < freq_pairs.size(); ++i) {
        //     cul_vec += freq_pairs[i].second;
        //     if( freq_pairs[i].second > 4){

        //         std::cout <<"total_size:"<< freq_pairs.size() << " cul_vec:" << cul_vec << " valid:" << valid_size++ <<"   idx: " << freq_pairs[i].first << ", times: " << freq_pairs[i].second << '\n';
        //     }
        // }

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

int IndexIVFPQDisk2::warm_up_nlist(size_t n, float* x, size_t w_nprobe, size_t warm_list){
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

int IndexIVFPQDisk2::warm_up_nvec(size_t n, float* x, size_t w_nprobe, size_t nvec){
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

int IndexIVFPQDisk2::warm_up_page(size_t n, float* x, size_t w_nprobe, size_t npage){
    return 0;
}

namespace{
    void warm_invlist(
        std::string select_lists_path,
        std::vector<size_t>& sorted_idx,
        faiss::Aligned_Invlist_Info *aligned_inv_info,
        faiss::ClusteredArrayInvertedLists* c_invlists)
    {
        std::ifstream in_file(select_lists_path, std::ios::binary);
        if (!in_file.is_open()) {
            throw std::runtime_error("Failed to open file: " + select_lists_path);
        }

        for (size_t idx : sorted_idx) {
            // Get cluster info
            const faiss::Aligned_Invlist_Info& inv_info = aligned_inv_info[idx];

            size_t ids_size = inv_info.list_size * sizeof(faiss::idx_t);
            size_t codes_size = inv_info.list_size * c_invlists->code_size;
            //size_t map_size = inv_info.list_size * sizeof(size_t);

            c_invlists->ids[idx].resize(inv_info.list_size);
            c_invlists->codes[idx].resize(codes_size);
            //c_invlists->inlist_maps[idx].resize(inv_info.list_size);
            // Calculate the start offset
            size_t data_offset = inv_info.page_start * PAGE_SIZE;
            in_file.seekg(data_offset, std::ios::beg);

            in_file.read(reinterpret_cast<char*>(c_invlists->ids[idx].data()), ids_size);
            in_file.read(reinterpret_cast<char*>(c_invlists->codes[idx].data()), codes_size);
            //in_file.read(reinterpret_cast<char*>(c_invlists->inlist_maps[idx].data()), map_size);
        }

    }
}


int IndexIVFPQDisk2::warm_up_index_info(size_t n, float* x, size_t w_nprobe, size_t warm_list){
    this->cached_list_info = new bool[nlist];

    std::vector<idx_t> idx(n * w_nprobe);
    std::vector<float> coarse_dis(n * w_nprobe);
    std::vector<size_t> sorted_idx;
    // Search to get the nearest cluster IDs
    quantizer->search(n, x, w_nprobe, coarse_dis.data(), idx.data(), nullptr);

    sort_coarse(idx, sorted_idx, this->len, warm_list, static_cast<size_t>(-1));

    for(int i = 0; i < nlist; i++){
        this->cached_list_info[i] = false;
    }

    for(int i = 0; i < sorted_idx.size(); i++){
        this->cached_list_info[sorted_idx[i]] = true;
    }

    warm_invlist(disk_path, sorted_idx, this->aligned_inv_info, dynamic_cast<ClusteredArrayInvertedLists*>(this->invlists));

    return 0;
}



void IndexIVFPQDisk2::set_disk(int n_threads){
    diskIOprocessors.resize(n_threads);
    for (int i = 0; i < n_threads; i++) {
        diskIOprocessors[i].reset(get_DiskIOSearchProcessor());
        diskIOprocessors[i]->initial();
    }
}

void IndexIVFPQDisk2::end_disk(int n_threads){
    diskIOprocessors.clear();
}

/** It is a sad fact of software that a conceptually simple function like this
 * becomes very complex when you factor in several ways of parallelizing +
 * interrupt/error handling + collecting stats + min/max collection. The
 * codepath that is used 95% of time is the one for parallel_mode = 0 */
void IndexIVFPQDisk2::search(
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
            centroid_index->hnsw.efSearch = std::min(nprobe,(size_t)200);
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

        // for(int i = 0; i < n; i++){
        //     idx_t* start = idx.get() + i * nprobe;
        //     idx_t* end = start + nprobe;
        //     // 对该组进行排序
        //     std::sort(start, end);

        //     for(int j = 0; j < nprobe; j++){
        //         std::cout << idx[i*nprobe + j] << " ";
        //     }
        //     std::cout <<"\n\n\n";
        // }

        double t1 = getmillisecs();
        invlists->prefetch_lists(idx.get(), n * nprobe);

        auto time_end = std::chrono::high_resolution_clock::now();
        indexIVFPQDisk2_stats.coarse_elapsed += time_end - time_start;

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

    int nslice = omp_get_max_threads();
    //sub_search_func(n, x, distances, labels, &indexIVF_stats);
#pragma omp parallel for
    for (int slice = 0; slice < nslice; slice++) {
        idx_t i0 = n * slice / nslice;
        idx_t i1 = n * (slice + 1) / nslice;
        idx_t n_slice = i1 - i0;
        const float* x_i = x + i0 * d; // 假设每个向量维度为 d
        float* dis_i = distances + i0 * k;
        idx_t* lab_i = labels + i0 * k;

        // 创建线程私有的统计对象
        IndexIVFStats local_stats;

        // 调用子搜索函数，处理数据的子集
        sub_search_func(n_slice, x_i, dis_i, lab_i, &local_stats);

        // 将本线程的统计信息累加到全局统计信息中
        // indexIVF_stats.ndis += local_stats.ndis;
        // indexIVF_stats.nlist_visited += local_stats.nlist_visited;
    }

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
                if(limit>=k_r){
                    //std::cout << "jj:" << jj << "  ratio:" <<jj*1.0/k*1.0 << "\n";
                    break;
                }

            }
        }
    }
    // for(int i = 0; i < 20; i++){
    //     for(int j = 0; j < k; j++){
    //         std::cout << "k: " << k << "  dis:" << distances[i*k + j] << "   id: "<<labels[i*k + j] << std::endl;
    //     }
    // }

    auto time_end = std::chrono::high_resolution_clock::now();       // time end
    indexIVFPQDisk2_stats.rerank_elapsed += time_end - time_start;
}


namespace{

struct DiskResultHandler{
    virtual void add(size_t q, float d0, idx_t id) = 0;
    //virtual void add(size_t q, float d0, idx_t id) = 0;
    virtual void end() = 0;
};

template<class C>
struct DiskHeapHandler : DiskResultHandler {
    float* dis;
    idx_t* ids;

    size_t k; // number of results to keep
    size_t nq;
    size_t nup;

    DiskHeapHandler(size_t nq, size_t k, float* dis, idx_t* ids)
            : nq(nq),
              dis(dis),
              ids(ids),
              k(k),
              nup(0) {
        heap_heapify<C>(k * nq, dis, ids);
    }

    void add(size_t q, float d0, idx_t id) override {
        float* current_dis = dis + q*k;
        idx_t* current_ids = ids + q*k;
        if (C::cmp(current_dis[0], d0)) {
            heap_replace_top<C>(k, current_dis, current_ids, d0, id);
            nup++;
        }

    }

    void end() override{
        for(int i = 0; i < nq; i++){
            heap_reorder<C>(k, dis + i*k, ids + i*k);
        }
    }

};


DiskResultHandler* get_result_handler(idx_t n, idx_t k, float* distances, idx_t* labels, MetricType metricType = METRIC_L2){
    if(metricType == METRIC_L2){
        return new DiskHeapHandler<CMax<float, idx_t>>(n, k, distances, labels);
    }else if(metricType == METRIC_INNER_PRODUCT){
        return new DiskHeapHandler<CMin<float, idx_t>>(n, k, distances, labels);
    }else{
        FAISS_THROW_MSG("Not support now!");
    }
}



struct UncachedList{
    size_t q;
    //std::vector<faiss::idx_t> list_no;
    std::vector<faiss::idx_t> list_pos;    // store the offset, keys and coarse_dis
};

}


void IndexIVFPQDisk2::search_fully_qps(
    idx_t n,
    const float* x,
    idx_t k,
    idx_t nprobe,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    DiskIOProcessor* diskIOprocessor,
    DiskResultHandler* heap_handler) const {

    if (n == 0) {
        return;
    }

    //DiskHeapHandler<C> heap_handler(n, k, distances, labels);

    struct QC {
        int qno;     // query number
        int list_no; // list to visit
        int rank;    // rank in coarse quantizer results
    };

    idx_t probe_fully_scan = std::min((idx_t)top, nprobe);

    // DiskIOProcessor* diskIOProcessor = get_DiskIOSearchProcessor();
    // diskIOProcessor->initial();

    std::vector<QC> qcs;

    // Step 1: Prepare the QC structure

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < probe_fully_scan; j++) {
            if (keys[i*nprobe + j] >= 0) {    // ensure it is a valid list
                //TODO check distance to ignore some probes
                qcs.push_back(QC{i, int(keys[i*nprobe + j]), int(j)});
            }

        }
    }

    std::sort(qcs.begin(), qcs.end(), [](const QC& a, const QC& b) {
        return a.list_no < b.list_no;
    });

    size_t ndis = 0;
    size_t i0 = 0;

    while (i0 < qcs.size()) {
        int list_no = qcs[i0].list_no;
        size_t i1 = i0 + 1;

        while (i1 < qcs.size() && qcs[i1].list_no == list_no) {
            i1++;
        }

        size_t list_size = len[list_no];
        if (list_size == 0) {
            i0 = i1;
            continue;
        }
        InvertedLists::ScopedIds ids(invlists, list_no);
        std::vector<float> codes(list_size*d);

        auto time_start = std::chrono::high_resolution_clock::now();
        diskIOprocessor->disk_io_all(d, list_size, list_no, codes.data(), aligned_cluster_info);
        auto time_end = std::chrono::high_resolution_clock::now();
        indexIVFPQDisk2_stats.disk_full_elapsed+=time_end - time_start;



        for (size_t i = i0; i < i1; i++) {

            const QC& qc = qcs[i];
            int q_index = qc.qno;
            //std::cout << "i0: " << i0 << " q_index:" << q_index<< std::endl;
            for (size_t j = 0; j < list_size; j++) {
                float distance = fvec_L2sqr_simd(x + q_index * d, codes.data() + j*d, d);
                heap_handler->add(q_index, distance, ids[j]);
            }
        }

        ndis += (i1 - i0) * list_size;
        i0 = i1;
    }
    //heap_handler->end();

}


// decode PQ begin
namespace{

#define TIC t0 = get_cycles()
#define TOC get_cycles() - t0
struct QueryTables {
    /*****************************************************
     * General data from the IVFPQ
     *****************************************************/

    const IndexIVFPQ& ivfpq;
    const IVFSearchParameters* params;

    // copied from IndexIVFPQ for easier access
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
            const IndexIVFPQ& ivfpq,
            const IVFSearchParameters* params)
            : ivfpq(ivfpq),
              d(ivfpq.d),
              pq(ivfpq.pq),
              metric_type(ivfpq.metric_type),
              by_residual(ivfpq.by_residual),
              use_precomputed_table(ivfpq.use_precomputed_table) {
        mem.resize(pq.ksub * pq.M * 2 + d * 2);
        sim_table = mem.data();
        sim_table_2 = sim_table + pq.ksub * pq.M;
        residual_vec = sim_table_2 + pq.ksub * pq.M;
        decoded_vec = residual_vec + d;

        // for polysemous
        polysemous_ht = ivfpq.polysemous_ht;
        if (auto ivfpq_params =
                    dynamic_cast<const IVFPQSearchParameters*>(params)) {
            polysemous_ht = ivfpq_params->polysemous_ht;
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
        ivfpq.quantizer->reconstruct(key, decoded_vec);
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
            ivfpq.quantizer->compute_residual(qi, residual_vec, key);
            pq.compute_distance_table(residual_vec, sim_table);

            if (polysemous_ht != 0) {
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 1) {
            dis0 = coarse_dis;

            fvec_madd(
                    pq.M * pq.ksub,
                    ivfpq.precomputed_table.data() + key * pq.ksub * pq.M,
                    -2.0,
                    sim_table_2,
                    sim_table);

            if (polysemous_ht != 0) {
                ivfpq.quantizer->compute_residual(qi, residual_vec, key);
                pq.compute_code(residual_vec, q_code.data());
            }

        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
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
                const float* pc = ivfpq.precomputed_table.data() +
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
                    ivfpq.precomputed_table.data() + key * pq.ksub * pq.M;
            for (int m = 0; m < pq.M; m++) {
                sim_table_ptrs[m] = s;
                s += pq.ksub;
            }
        } else if (use_precomputed_table == 2) {
            dis0 = coarse_dis;

            const MultiIndexQuantizer* miq =
                    dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
            FAISS_THROW_IF_NOT(miq);
            const ProductQuantizer& cpq = miq->pq;
            int Mf = pq.M / cpq.M;

            long k = key;
            int m0 = 0;
            for (int cm = 0; cm < cpq.M; cm++) {
                int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                k >>= cpq.nbits;

                const float* pc = ivfpq.precomputed_table.data() +
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


template <typename IDType, class PQDecoder>
struct IVFPQScannerT : QueryTables {
    const uint8_t* list_codes;
    const IDType* list_ids;
    size_t list_size;

    IVFPQScannerT(const IndexIVFPQ& ivfpq, const IVFSearchParameters* params)
            : QueryTables(ivfpq, params) {
    }

    float dis0;

    void init_list(idx_t list_no, float coarse_dis, int mode) {
        this->key = list_no;
        this->coarse_dis = coarse_dis;

        if (mode == 2) {
            dis0 = precompute_list_tables();
            //std::cout << "dis0:" << dis0 << std::endl;
        } else if (mode == 1) {
            dis0 = precompute_list_table_pointers();
        }
    }

    size_t scan_list_with_table(
            size_t ncode,
            const uint8_t* codes,
            float* local_dis,
            idx_t* local_ids) const {
        int counter = 0;
        size_t operations = 0;

        //std::cout << " pq.M:" << pq.M << "  pq.nbits:" << pq.nbits << std::endl;
        //assert(codes != nullptr);
        //assert(local_dis != nullptr);
        //assert(local_ids != nullptr);
        size_t saved_j[4] = {0, 0, 0, 0};
        for (size_t j = 0; j < ncode; j++) {
            //std::cout << "ncode:"<< ncode <<"  j:" << j << std::endl;
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

                counter = 0;
                operations+=4;
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
            operations++;
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
            operations++;
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
            operations++;
        }

        return operations;
    }

};

struct BaseDecoder{

    idx_t list_no;
    bool store_pairs;
    virtual void set_query(const float* query) = 0;
    virtual void set_list(idx_t list_no, float coarse_dis) = 0;
    virtual float distance_to_code(const uint8_t* code) const = 0;
    virtual size_t scan_codes(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            float* decoded_dis,
            idx_t* decoded_ids) const = 0;

};

template <class C, class PQDecoder>
struct IVFPQDecoder : IVFPQScannerT<idx_t, PQDecoder>, BaseDecoder {

    int precompute_mode;
    const IDSelector* sel;

    IVFPQDecoder(
            const IndexIVFPQ& ivfpq,
            bool store_pairs,
            int precompute_mode,
            const IDSelector* sel)
            : IVFPQScannerT<idx_t, PQDecoder>(ivfpq, nullptr),
              precompute_mode(precompute_mode),
              sel(sel) {
        this->store_pairs = store_pairs;
    }

    void set_query(const float* query) override {
        this->init_query(query);
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
            float* decoded_dis,
            idx_t* decoded_ids) const override{

        //std::cout << " scanning codes" <<std::endl;
        return this->scan_list_with_table(ncode, codes, decoded_dis, decoded_ids);
    }

    // void scan_codes_range(
    //         size_t ncode,
    //         const uint8_t* codes,
    //         const idx_t* ids,
    //         float radius,
    //         RangeQueryResult& rres) const override {

    // }
};

template<class C, class PQDecoder>
BaseDecoder* get_pqdecoder1(const IndexIVFPQ& index){
    return new IVFPQDecoder<C, PQDecoder>(index, false, 2, nullptr);
}

template<class C>
BaseDecoder* get_pqdecoder2(const IndexIVFPQ& index){
    if (index.pq.nbits == 8) {
        return get_pqdecoder1<C, PQDecoder8>(
                index);
    } else if (index.pq.nbits == 16) {
        return get_pqdecoder1<C, PQDecoder16>(
                index);
    } else {
        return get_pqdecoder1<C, PQDecoderGeneric>(
                index);
    }
}

BaseDecoder* get_pqdecoder(const IndexIVFPQ& index, MetricType metricType){
    if(metricType == METRIC_L2){
        return get_pqdecoder2<CMax<float, idx_t>>(index);
    }else if(metricType == METRIC_INNER_PRODUCT){
        return get_pqdecoder2<CMin<float, idx_t>>(index);
    }else{
        FAISS_THROW_MSG("Do not support !!");
    }

}

// // read fully
struct ListSegment{
    // 需要合并读取的信息，以及map，和list id, list dis之类的信息
};

// read partially
struct PageSegment {
    idx_t list_no;
    int start_page;
    int page_count;
    int* in_buffer_offsets;    // 使用指针替代 vector
    size_t* in_buffer_ids;     // 使用指针替代 vector
    size_t length;             // 新增的长度参数

    PageSegment()
        : in_buffer_offsets(nullptr), in_buffer_ids(nullptr), length(0) {}

    PageSegment(int start, int count, idx_t list_no, const int* offsets, const size_t* ids, size_t len)
        : start_page(start), page_count(count), list_no(list_no), length(len) {

        // 分配内存并进行深拷贝
        in_buffer_offsets = (int*)malloc(length * sizeof(int));
        in_buffer_ids = (size_t*)malloc(length * sizeof(size_t));

        if (in_buffer_offsets && in_buffer_ids) {
            std::memcpy(in_buffer_offsets, offsets, length * sizeof(int));
            std::memcpy(in_buffer_ids, ids, length * sizeof(size_t));
        } else {
            // 处理内存分配失败的情况
            free(in_buffer_offsets);
            free(in_buffer_ids);
            in_buffer_offsets = nullptr;
            in_buffer_ids = nullptr;
            length = 0;
        }
    }

    // 析构函数，释放动态分配的内存
    ~PageSegment() {
        if(in_buffer_offsets != nullptr){
            free(in_buffer_offsets);
            free(in_buffer_ids);
        }
    }

    PageSegment(const PageSegment& other)
        : start_page(other.start_page), page_count(other.page_count), list_no(other.list_no), length(other.length) {

        in_buffer_offsets = (int*)malloc(length * sizeof(int));
        in_buffer_ids = (size_t*)malloc(length * sizeof(size_t));

        if (in_buffer_offsets && in_buffer_ids) {
            std::memcpy(in_buffer_offsets, other.in_buffer_offsets, length * sizeof(int));
            std::memcpy(in_buffer_ids, other.in_buffer_ids, length * sizeof(size_t));
        } else {
            free(in_buffer_offsets);
            free(in_buffer_ids);
            in_buffer_offsets = nullptr;
            in_buffer_ids = nullptr;
            length = 0;
        }
    }
};

} // anonymous namespace
// decode PQ end

// sort func begin
namespace{
    template <typename T1, typename T2>
    std::vector<size_t> sort_two_array(const T1* first_begin, const T2* second_begin, T1* result_first, T2* result_second, size_t num) {
        std::vector<size_t> indices(num);
        for (size_t i = 0; i < num; i++) {
            indices[i] = i;
        }

        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return first_begin[a] < first_begin[b];
        });

        std::vector<T1> sorted_first(num);
        std::vector<T2> sorted_second(num);

        for (size_t i = 0; i < num; i++) {
            sorted_first[i] = first_begin[indices[i]];
            sorted_second[i] = second_begin[indices[i]];
        }

        std::copy(sorted_first.begin(), sorted_first.end(), result_first);
        std::copy(sorted_second.begin(), sorted_second.end(), result_second);
        return indices;
    }

    void decode_pq_lists(BaseDecoder* pqdecoder,
                     const float* query,
                     const idx_t* list_ids,
                     const float* list_dis,
                     size_t probe_batch,                 // 一批次解码的数量
                     InvertedLists* invlists,
                     std::vector<std::vector<float>>& pq_distances,
                     std::vector<std::vector<idx_t>>& pq_ids,
                     const bool* cached_list_info = nullptr)
    {
        // 设置查询向量，有时候不需要设置
        if(query)
            pqdecoder->set_query(query);

        for (size_t i = 0; i < probe_batch; i++) {
            idx_t list_no = list_ids[i];
            float list_distance = list_dis[i];

            size_t list_size = invlists->list_size(list_no);

            //pq_distances[i].clear();
            //pq_ids[i].clear();

            pq_distances[i].resize(list_size);
            pq_ids[i].resize(list_size);

#ifdef CACHE_MODE
            if(cached_list_info != nullptr){
                if(!cached_list_info[list_no])
                    continue;                  // leave it empty so that we can know when to get it from disk
            }
#endif
            // 设置当前列表和解码
            pqdecoder->set_list(list_no, list_distance);    // list_no用于寻找聚类中心向量位置
            pqdecoder->scan_codes(list_size,
                                invlists->get_codes(list_no),
                                invlists->get_ids(list_no),
                                pq_distances[i].data(),
                                pq_ids[i].data());

        }
    }

    void decode_pq_lists(BaseDecoder* pqdecoder,
                     const float* query,
                     const idx_t* list_ids,
                     const float* list_dis,
                     size_t pqed_list,
                     size_t probe_batch,                 // 一批次解码的数量
                     InvertedLists* invlists,
                     std::vector<std::vector<float>>& pq_distances,
                     std::vector<std::vector<idx_t>>& pq_ids,
                     const bool* cached_list_info = nullptr)
    {
        // 设置查询向量，有时候不需要设置
        if(query)
            pqdecoder->set_query(query);

        for (size_t i = 0; i < probe_batch; i++) {
            idx_t list_no = list_ids[pqed_list + i];
            float list_distance = list_dis[pqed_list + i];

            size_t list_size = invlists->list_size(list_no);

            //pq_distances[i].clear();
            //pq_ids[i].clear();

            pq_distances[pqed_list + i].resize(list_size);
            pq_ids[pqed_list + i].resize(list_size);

#ifdef CACHE_MODE
            if(cached_list_info != nullptr){
                if(!cached_list_info[list_no])
                    continue;                  // leave it empty so that we can know when to get it from disk
            }
#endif
            // 设置当前列表和解码
            pqdecoder->set_list(list_no, list_distance);    // list_no用于寻找聚类中心向量位置
            pqdecoder->scan_codes(list_size,
                                invlists->get_codes(list_no),
                                invlists->get_ids(list_no),
                                pq_distances[pqed_list + i].data(),
                                pq_ids[pqed_list + i].data());

        }
    }

    struct DiskInvlist{

        size_t list_size;
        size_t code_size;
        size_t list_no;   // Optional, may not be useful

        std::vector<faiss::idx_t> disk_ids;
        std::vector<uint8_t> disk_codes;
        std::vector<size_t> disk_map;

        DiskInvlist() : list_size(0), code_size(0), list_no(0) {}

        // 构造函数
        DiskInvlist(void* disk_buffer, size_t list_size, size_t code_size, size_t list_no = 0)
            : list_size(list_size), code_size(code_size), list_no(list_no) {
            set(disk_buffer, list_size, code_size, list_no);
        }

        // set 函数
        void set(void* disk_buffer, size_t list_size, size_t code_size, size_t list_no = 0) {
            this->list_size = list_size;
            this->code_size = code_size;
            this->list_no = list_no;

            size_t ids_size = list_size * sizeof(faiss::idx_t);
            size_t codes_size = list_size * code_size;
            size_t map_size = list_size * sizeof(size_t);

            char* buffer = static_cast<char*>(disk_buffer);
            disk_ids.assign(reinterpret_cast<faiss::idx_t*>(buffer),
                            reinterpret_cast<faiss::idx_t*>(buffer + ids_size));
            buffer += ids_size;

            disk_codes.assign(reinterpret_cast<uint8_t*>(buffer),
                            reinterpret_cast<uint8_t*>(buffer + codes_size));
            buffer += codes_size;

            disk_map.assign(reinterpret_cast<size_t*>(buffer),
                            reinterpret_cast<size_t*>(buffer + map_size));
        }

        size_t get_size(){
            return disk_ids.size();
        }

        faiss::idx_t* get_ids() {
            return disk_ids.data();
        }

        uint8_t* get_codes() {
            return disk_codes.data();
        }

        size_t* get_map() {
            return disk_map.data();
        }

    };




    void decode_pq_list(
        BaseDecoder* pqdecoder,
        const float* query,
        idx_t list_no,
        float list_distance,
        size_t list_size,
        DiskInvlist* disk_invlist,
        std::vector<float>& pq_distance,
        std::vector<idx_t>& pq_ids){

        if(query)
            pqdecoder->set_query(query);   // when decode several discrete lists, we can set query out of this function

        pq_distance.resize(list_size);
        pq_ids.resize(list_size);


        pqdecoder->set_list(list_no, list_distance);    // list_no用于寻找聚类中心向量位置
        pqdecoder->scan_codes(list_size,
                            disk_invlist->get_codes(),
                            disk_invlist->get_ids(),
                            pq_distance.data(),
                            pq_ids.data());


    }

}
// sort func end

void IndexIVFPQDisk2::search_fully(
    idx_t n,
    const float* x,
    idx_t k,
    idx_t nprobe,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    DiskIOProcessor* diskIOprocessor,
    DiskResultHandler* heap_handler
#ifdef CACHE_MODE
    ,UncachedLists& uncached_lists
#endif
    ) const{
        // 1. 这里search_fully的情况可以用pq
        // 2. 记得聚类要重排序！
        if (n == 0) {
            return;
        }
        // 避免top>nprobe的情况
        const size_t probes = std::min((size_t)nprobe, top);

        const bool* cached_list_info = this->cached_list_info;

        std::vector<idx_t> working_lists_ids(probes);
        std::vector<float> working_lists_dis(probes);
        idx_t* p_working_lists_ids = working_lists_ids.data();
        float* p_working_lists_dis = working_lists_dis.data();

        std::vector<std::vector<float>> pq_distances;
        std::vector<std::vector<idx_t>> pq_ids;
        pq_distances.resize(probes);
        pq_ids.resize(probes);

        BaseDecoder* pqdecoder = get_pqdecoder(*this,this->metric_type);

/*

*************************
不能访问没有cache的list, 通过cached_list_info判断。
*************************

*/
        //std::cout << "FULL 1\n";
        for(int cur_q = 0; cur_q < n; cur_q++){
            const float* current_query = x + cur_q*d;
            float* heap_sim = distances + cur_q*k;
            idx_t* heap_ids = labels + cur_q*k;


            //std::cout << "Query:  " << cur_q << "\n";


            int probes_begin = cur_q*nprobe;
            std::vector<size_t> indices = sort_two_array<idx_t, float>(keys + probes_begin,
                                             coarse_dis + probes_begin,
                                             p_working_lists_ids,
                                             p_working_lists_dis,
                                             probes);
            size_t* pq_map = indices.data();
                //sort list_no and list_dis

#ifdef CACHE_MODE
            std::vector<idx_t> record_uncached;
            record_uncached.reserve(nprobe);
#endif

            // 用于控制pq的变量
            // bool pq_cross_query = false;
            // ******pre PQ******
            //std::cout << "PQ  begin" << "\n";
            if(cur_q == 0){
                auto time_start = std::chrono::high_resolution_clock::now();
                //由于程序一开始没有解码，所以需要预先解码一些内容
                decode_pq_lists(pqdecoder, x+cur_q*d, keys + cur_q*nprobe, coarse_dis+cur_q*nprobe,
                               probes, invlists, pq_distances, pq_ids, cached_list_info);
                auto time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.memory_1_elapsed += time_end - time_start;
            }
            //std::cout << "PQ  end" << "\n";
            // ******pre PQ******

            // std::cout << "list_no:" << keys[cur_q*nprobe] << " list_size:" << invlists->list_size(keys[cur_q*nprobe]) << " ";
            // for(int ii = 0; ii < pq_distances[0].size(); ii++){
            //     std::cout << pq_distances[0][ii] << " ";
            // }
            // std::cout << "\n";

            AsyncReadRequests_Full_PQDecode requests_full;
            requests_full.list_requests.reserve(probes);

            Aligned_Cluster_Info* acinfo;
            //std::cout << "request_num" << request_num << std::endl;
            int pushed_lists = 0;
            int pre_pushed_lists = 0;
            int pushed_requests = 0;
            //std::cout << "Full 2 " << "\n";
            while(pushed_lists < probes){

                size_t list_no = p_working_lists_ids[pushed_lists];

                //std::cout << "list_no:" << list_no << "\n";
#ifdef CACHE_MODE
                if(!cached_list_info[list_no]){
                    //std::cout << "pushed_lists continue:" << pushed_lists << std::endl;
                    //assert(list_no == keys[cur_q*nprobe + pq_map[pushed_lists]]);
                    record_uncached.push_back(pq_map[pushed_lists++]);
                    continue;
                }
#endif

                size_t list_size = invlists->list_size(list_no);
                const size_t* map = invlists->get_inlist_map(list_no);  // convert result
                const idx_t* ids = invlists->get_ids(list_no);
                acinfo = &aligned_cluster_info[list_no];

                /*
                TODO 如果某个list已经cache了，那么就直接计算，不用加进去
                */
                auto cce_time_start = std::chrono::high_resolution_clock::now();
                int cluster_pos = diskInvertedListHolder.is_cached(list_no);
                if(cluster_pos!= -1){
                    //std::cout << "using cached data, listno=" << list_no << "  cluster_pos="  << cluster_pos << std::endl;
                    float* pq_dis_c = pq_distances[pq_map[pushed_lists]].data();
                    const void* buffer = diskInvertedListHolder.get_cache_data(cluster_pos);
                    float distance;
                    for(int m = 0; m < list_size; m++){
                        if(pq_dis_c[m] < heap_sim[0]*1.10){
                            if(this->valueType == "uint8")
                                distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)buffer) +d*map[m], d);
                            else
                                distance = fvec_L2sqr_simd(current_query, (float*)buffer +d*map[m], d);
                            heap_handler->add(cur_q, distance, ids[m]);
                        }
                    }
                    indexIVFPQDisk2_stats.cached_list_access += 1;
                    indexIVFPQDisk2_stats.cached_vector_access += list_size;
                    pushed_lists++;

                    //std::cout << "cache list end" << std::endl;
                    auto cce_time_end = std::chrono::high_resolution_clock::now();
                    indexIVFPQDisk2_stats.cached_calculate_elapsed += cce_time_end - cce_time_start;
                    continue;
                }

                if(pushed_lists!=0 && pre_pushed_lists!=0 && (p_working_lists_ids[pushed_lists] == p_working_lists_ids[pre_pushed_lists] + 1)){
                    // 合并request，增长单次IO的长度
                    requests_full.list_requests[pushed_requests-1].push_request(
                                                        acinfo->page_count,
                                                        list_size,
                                                        0,
                                                        list_size,
                                                        0,
                                                        map,
                                                        ids,
                                                        pq_distances[pq_map[pushed_lists]].data());
                    //std::cout << "list_size:"<< list_size << "  pq_size:" << pq_distances[pq_map[pushed_lists]].size() << std::endl;
                    assert(list_size == pq_distances[pq_map[pushed_lists]].size());
                    pushed_lists++;
                    pre_pushed_lists++;
                    //std::cout << "pushed_requests:"<< pushed_requests << std::endl;
                    //std::cout << "acinfo->page_count:" << acinfo->page_count<<" list_size:" << list_size << "\n";
                }else{
                    // 新增一个request
                    requests_full.list_requests.emplace_back(acinfo->page_start * PAGE_SIZE,
                                                        acinfo->page_count,
                                                        list_size,
                                                        0,
                                                        list_size,
                                                        0,
                                                        map,
                                                        ids,
                                                        pq_distances[pq_map[pushed_lists]].data());
                    //std::cout << "list_size:"<< list_size << "  pq_size:" << pq_distances[pq_map[pushed_lists]].size() << std::endl;
                    assert(list_size == pq_distances[pq_map[pushed_lists]].size());
                    pushed_lists++;
                    pushed_requests++;
                    pre_pushed_lists++;
                    //std::cout << "pushed_lists:" << pushed_lists << "  pushed_requests:"<< pushed_requests << std::endl;
                    //std::cout << "acinfo->page_count:" << acinfo->page_count<<" list_size:" << list_size << "\n";
                }
                //indexIVFPQDisk2_stats.searched_vector_full+=list_size;
                indexIVFPQDisk2_stats.searched_page_full+=acinfo->page_count;
                faiss::indexIVFPQDisk2_stats.searched_lists++;
            }
            //std::cout << "\n\n";
            indexIVFPQDisk2_stats.requests_full+=pushed_requests;
            //std::cout << "pushed_requests:" << pushed_requests << std::endl;
            /*
            Callback function
            */
            requests_full.cal_callback = [&](AsyncRequest_Full* requested, void* buffer){
                auto time_start = std::chrono::high_resolution_clock::now();

                float distance;
                const size_t* map = requested->map;
                const idx_t * list_ids = requested->ids;
                const float* pq_dis_c = requested->pq_dis.data();
                //size_t vector_num = requested->vectors_num;
                //idx_t tmp_list_no = requested->list_no;
                //const idx_t* list_ids = invlists->get_ids(tmp_list_no);

                std::vector<float> float_vector(d);
                //std::cout << "vector number = " << requested->vectors_num << "\n";
                //std::cout << "heap_sim:" << heap_sim[0] << "\n";
                for(int m = 0; m < requested->vectors_num; m++){
                    //distance = fvec_L2sqr_simd(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(), buffer, d*map[m]), d);
                    //std::cout << "pq:" << pq_dis_c[m] << "  heap_sim:" << heap_sim[0] << "\n";
                    if(pq_dis_c[m] < heap_sim[0]*1.1)
                    {
                        if(this->valueType == "uint8")
                            distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)buffer) +d*map[m], d);
                        else
                            distance = fvec_L2sqr_simd(current_query, (float*)buffer +d*map[m], d);
                        //distance = fvec_L2sqr_simd(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(), buffer, d*map[m]), d);
                        //distance = fvec_L2sqr(current_query, (float*)buffer +d*map[m], d);
                        heap_handler->add(cur_q, distance, list_ids[m]);

                        // for(int ii = 0; ii < d; ii++){
                        //     std::cout << ((float*)buffer +d*map[m])[d] << " ";
                        // }
                        //std::cout << std::endl;
                        // std::cout << "pq:" << pq_dis_c[m] << "  heap_sim:" << heap_sim[0] << " ";
                        // std::cout << "cur_q:" << cur_q << " distance:" << distance << " id:" << list_ids[m] << std::endl;
                        indexIVFPQDisk2_stats.searched_vector_full++;
                    }
                    //std::cout << "cur_q:" << cur_q << " distance:" << distance << " id:" << list_ids[m] << std::endl;
                }
                auto time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.rank_elapsed+=time_end - time_start;


            };
            requests_full.pq_callback = [&](){

                auto time_start = std::chrono::high_resolution_clock::now();
                //std::cout << "pq:" << cur_q+1 << "\n" ;
                if(cur_q < n-1){
                    int next_q = cur_q + 1;
                    decode_pq_lists(pqdecoder, x + next_q*d, keys + next_q*nprobe, coarse_dis + next_q*nprobe,
                                    probes, invlists, pq_distances, pq_ids, cached_list_info);
                }
                auto time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.pq_elapsed+=time_end - time_start;

            };

            //std::cout << "page_buffer prepare" << std::endl;
            diskIOprocessor->disk_io_full_async_pq(requests_full);
            //std::cout << "page_buffer ok" << std::endl;


            auto time_start = std::chrono::high_resolution_clock::now();
            diskIOprocessor->submit(pushed_requests);
            auto time_end = std::chrono::high_resolution_clock::now();

#ifdef CACHE_MODE
                if(!record_uncached.empty()){
                    UncachedList new_list;
                    new_list.q = cur_q;  // 假设有一个 `query_id`
                    new_list.list_pos = std::move(record_uncached);  // 移动数据以避免拷贝
                    uncached_lists.push_back(std::move(new_list));  // 移动到列表中
                }
#endif

            indexIVFPQDisk2_stats.disk_full_elapsed+=time_end - time_start;
        }


}



// merge pages
namespace{

// 中间有一些页也进行合并
void merge_pages_2(
    std::vector<PageSegment>& merged_segments,
    int* ptr_page_to_search,
    int* ptr_vector_to_submit,
    int* ptr_vector_to_search,
    size_t* vec_page_proj,

    size_t vectors_num,
    int num_page_to_search,
    const int per_page_element,
    const int per_page_vector,
    const size_t d,
    const idx_t list_no,
    const size_t max_pages
    ){
        int max_vectors_per_request = num_page_to_search*per_page_vector;

        int begin_page = 0;
        int record_begin_page = 0;
        int start_page = ptr_page_to_search[0];
        int page_count = 1;
        //std::vector<int> in_buffer_offsets;  //?
        //std::vector<size_t> in_buffer_ids;
        int in_buffer_offsets[max_vectors_per_request];
        size_t in_buffer_ids[max_vectors_per_request];
        size_t offset_count = 0;

        int vec_rank = 0;
        int page_rank = 0;
        //int* ptr_vector_to_search = vector_to_search.data();
        //int* ptr_page_to_search = page_to_search.data();

        begin_page = ptr_page_to_search[0];
        record_begin_page = ptr_page_to_search[0];

        while (vec_rank<vectors_num && vec_page_proj[vec_rank] == begin_page) {
            int inbuffer = ptr_vector_to_submit[vec_rank] * d - per_page_element * record_begin_page;

            //std::cout << "record_begin_page:" << record_begin_page << " begin_page:" << begin_page << std::endl;

            in_buffer_offsets[offset_count] = inbuffer;
            //TODO
            in_buffer_ids[offset_count] = ptr_vector_to_search[vec_rank];
            offset_count++;
            vec_rank++;
        }
        page_rank++;

        for (int i = 1; i < num_page_to_search; ++i) {
            int current_page = ptr_page_to_search[i];
            int previous_page = ptr_page_to_search[i-1];

            int page_gap = current_page - previous_page;

            if (page_gap < max_pages) {
                // 如果是连续的页，增加页数
                page_count += page_gap;
                begin_page = ptr_page_to_search[page_rank];
                while (vec_rank < vectors_num && vec_page_proj[vec_rank] == begin_page) {
                    int inbuffer = ptr_vector_to_submit[vec_rank] * d - per_page_element * record_begin_page;

                    //std::cout << "record_begin_page:" << record_begin_page << " begin_page:" << begin_page << std::endl;

                    in_buffer_offsets[offset_count] = inbuffer;
                    in_buffer_ids[offset_count] = ptr_vector_to_search[vec_rank];
                    offset_count++;
                    vec_rank++;
                }
                page_rank++;
            } else {
                merged_segments.push_back(PageSegment(start_page, page_count, list_no,
                                             in_buffer_offsets, in_buffer_ids, offset_count));
                start_page = current_page;
                page_count = 1;

                offset_count = 0;

                record_begin_page = ptr_page_to_search[page_rank];
                begin_page = ptr_page_to_search[page_rank];

                while (vec_rank<vectors_num && vec_page_proj[vec_rank] == begin_page) {

                    int inbuffer = ptr_vector_to_submit[vec_rank] * d - per_page_element * record_begin_page;

                    //std::cout << "record_begin_page:" << record_begin_page << " begin_page:" << begin_page << std::endl;

                    in_buffer_offsets[offset_count] = inbuffer;
                    in_buffer_ids[offset_count] = ptr_vector_to_search[vec_rank];
                    offset_count++;
                    vec_rank++;
                }
                page_rank++;
            }
        }
        // if(page_count > 40){
        //     return;
        // }
        merged_segments.push_back(PageSegment(start_page, page_count, list_no,
                                    in_buffer_offsets, in_buffer_ids, offset_count));
        //std::cout << "merged_segments:" << merged_segments.size() << std::endl;
}

void merge_pages_transpage(
    std::vector<PageSegment>& merged_segments,
    Page_to_Search* ptr_page_to_search,
    int* ptr_vector_to_submit,
    int* ptr_vector_to_search,
    size_t* vec_page_proj,

    size_t vectors_num,
    int num_page_to_search,
    const int per_page_element,
    const int per_page_vector,
    const size_t d,
    const idx_t list_no,
    const size_t max_pages
    )
{
    //per_page_vector 向下取整
    int max_vectors_per_request = num_page_to_search * (per_page_vector+1);  

    int current_page = ptr_page_to_search[0].first;  // 初始化为第一组的起始页
    int record_begin_page = current_page;
    int page_count = 1;

    int in_buffer_offsets[max_vectors_per_request];
    size_t in_buffer_ids[max_vectors_per_request];
    size_t offset_count = 0;

    int vec_rank = 0;
    int page_rank = 0;

    // 处理第一个页面的向量
    while (vec_rank < vectors_num && ptr_page_to_search[vec_rank].first == current_page) {
        // 距离页面开始相差的元素。（相差的维度，在process时自动换为float或uint8）
        int inbuffer = ptr_vector_to_submit[vec_rank] * d - per_page_element * record_begin_page;

        in_buffer_offsets[offset_count] = inbuffer;
        in_buffer_ids[offset_count] = ptr_vector_to_search[vec_rank];
        // 如果向量跨页(不要if也可以)
        //if(ptr_page_to_search[vec_rank].last != current_page){
            page_count += ptr_page_to_search[vec_rank].last - ptr_page_to_search[vec_rank].first;
        //}
        offset_count++;
        vec_rank++;
    }
    page_rank++;

    for (int i = 1; i < num_page_to_search; ++i) {
        int previous_last_page = ptr_page_to_search[i - 1].last;
        int current_first_page = ptr_page_to_search[i].first;

        int page_gap = current_first_page - previous_last_page;

        if (page_gap < max_pages) {
            // 页之间间隔小于 max_pages，可以合并
            // page_count
            page_count += (ptr_page_to_search[i].last - ptr_page_to_search[i-1].last);
            current_page = ptr_page_to_search[page_rank].first;

            // 开头是一样的，但是跨页的部分怎么办？ 需要比较开头？，如果头尾都在，那还可以加入
            // TODO beam search还需要一些参数，也许可以加在后面？
            while (vec_rank < vectors_num && ptr_page_to_search[vec_rank].first == current_page) {
                int inbuffer = ptr_vector_to_submit[vec_rank] * d - per_page_element * record_begin_page;

                in_buffer_offsets[offset_count] = inbuffer;
                in_buffer_ids[offset_count] = ptr_vector_to_search[vec_rank];
                // 如果跨页，那么现在肯定是循环最后一次执行
                page_count += ptr_page_to_search[vec_rank].last - ptr_page_to_search[vec_rank].first;

                offset_count++;
                vec_rank++;
            }
            page_rank++;
        } else {
            // 页之间间隔大于 max_pages，创建新的 PageSegment
            merged_segments.push_back(PageSegment(record_begin_page, page_count, list_no,
                                                  in_buffer_offsets, in_buffer_ids, offset_count));

            // 更新新的页段信息
            record_begin_page = ptr_page_to_search[page_rank].first;
            current_page = ptr_page_to_search[page_rank].first;
            page_count = ptr_page_to_search[page_rank].last - ptr_page_to_search[page_rank].first + 1;

            offset_count = 0;

            while (vec_rank < vectors_num && ptr_page_to_search[vec_rank].first == current_page) {
                int inbuffer = ptr_vector_to_submit[vec_rank] * d - per_page_element * record_begin_page;

                in_buffer_offsets[offset_count] = inbuffer;
                in_buffer_ids[offset_count] = ptr_vector_to_search[vec_rank];

                page_count += ptr_page_to_search[vec_rank].last - ptr_page_to_search[vec_rank].first;
                
                offset_count++;
                vec_rank++;
            }
            page_rank++;
        }
    }

    // 添加最后一个页段
    merged_segments.push_back(PageSegment(record_begin_page, page_count, list_no,
                                          in_buffer_offsets, in_buffer_ids, offset_count)); 
}




}



void IndexIVFPQDisk2::search_partially(
    idx_t n,
    const float* x,
    idx_t k,
    idx_t nprobe,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    DiskIOProcessor* diskIOprocessor,
    DiskResultHandler* heap_handler
#ifdef CACHE_MODE
    ,UncachedLists& uncached_lists
#endif
    ) const {

    if (n == 0) {
        return;
    }

    // assume submit 10 probes once a time
    int submit_per_round = 20;
    std::vector<std::vector<float>> pq_distances;
    std::vector<std::vector<idx_t>> pq_ids;     // the ids here are from 0 to listno-1;

    pq_distances.resize(submit_per_round);
    pq_ids.resize(submit_per_round);


    BaseDecoder* pqdecoder = get_pqdecoder(*this,this->metric_type);

    // for future work
    // 对每一个query而言，记录已经解码、即将解码、已经完成的list的数据
    int working_list = top;
    int pq_done = 0;       // 从partial开始算起
    bool pq_switch = false;
    int pq_todo = 0;

    bool pq_cross_query = false;  // 在程序末尾 计算下一个query的pq
    bool pq_cross_extra_decode = true;  // 有些回调函数由于查询原因不会执行

    for(int cur_q = 0; cur_q < n; cur_q++){
        //std::cout << "cur_q:" << cur_q << "\n\n\n\n\n" << std::endl;
        const float* current_query = x + cur_q*d;

        float* heap_sim = distances + cur_q*k;
        idx_t* heap_ids = labels + cur_q*k;
        //pqdecoder->set_query(x + cur_q*d);

#ifdef CACHE_MODE
            std::vector<idx_t> record_uncached;
            record_uncached.reserve(nprobe);
#endif

        pq_cross_query = false;
        // 避免超出范围
        int actual_submit = std::min(submit_per_round, (int)nprobe - working_list);
        //std::cout << "query phrase." << std::endl;
        // 先计算第一批IO提交的pq id以及dis  TIP: 只在程序开始时进行计算
        //std::cout << "\n";
        if(pq_cross_extra_decode){

            working_list = top;
            actual_submit = std::min(submit_per_round, (int)nprobe - working_list);
            //std::cout << "pq_cross_extra_decoding. working_list:" << working_list << " actual_submit:" <<actual_submit <<"\n" ;
            pq_done = 0;
            pq_switch = false;
            pq_todo = 0;
            //std::cout << " 补充pq解码"<< std::endl;

            decode_pq_lists(pqdecoder, x+cur_q*d, keys + cur_q*nprobe + pq_done + top, coarse_dis + cur_q*nprobe + pq_done + top,
                             actual_submit, invlists, pq_distances, pq_ids, this->cached_list_info);

            // pq_distances.resize(actual_submit);
            // pq_ids.resize(actual_submit);
            // pqdecoder->set_query(x + cur_q*d);

            // for(int i = 0; i < actual_submit; i++){

            //     idx_t list_no =        keys[cur_q*nprobe + i + pq_done + top];
            //     float list_dis = coarse_dis[cur_q*nprobe + i + pq_done + top];
            //     size_t list_size = invlists->list_size(list_no);
            //     pq_distances[i].resize(list_size);
            //     pq_ids[i].resize(list_size);

            //     //std::cout << "pq key:" << cur_q*nprobe + i + pq_done + top << "  pq list_no:" << list_no << std::endl;

            //     auto time_start = std::chrono::high_resolution_clock::now();
            //     //std::cout << "list_no:" << list_no << " list_dis:" << list_dis <<std::endl;
            //     pqdecoder->set_list(list_no, list_dis);
            //     //std::cout << "pqdecoder set list :" << list_no  << std::endl;
            //     pqdecoder->scan_codes(list_size,
            //                         invlists->get_codes(list_no),
            //                         invlists->get_ids(list_no),
            //                         pq_distances[i].data(),
            //                         pq_ids[i].data());

            //     auto time_end = std::chrono::high_resolution_clock::now();
            //     indexIVFPQDisk2_stats.memory_1_elapsed+=time_end - time_start;
            // }
            indexIVFPQDisk2_stats.pq_list_partial += actual_submit;
            pq_done += actual_submit;      // pq_done 一定会小于nprobe
        }
        pq_cross_extra_decode = false;
        //std::cout << "ASYNC phrase_1." << std::endl;
        // 开始进行异步IO

        while(working_list < nprobe){
            // 一次提交submit_per_round 或更少
            actual_submit = std::min((idx_t)submit_per_round, nprobe - (idx_t)working_list);
            pq_todo = std::min(submit_per_round, (int)nprobe - (working_list + actual_submit));
            if(pq_todo > 0){
                pq_switch = true;
                pq_todo = std::min(submit_per_round, pq_todo);
            }
            if(pq_todo <= 0 && cur_q != n - 1){
                pq_cross_extra_decode = true;
                pq_cross_query = true;
            }
            //std::cout << "\n\n\n\n";
            //std::cout << "Actual_submit begin" << std::endl;

            // std::cout << "actual_submit:" << actual_submit
            //           << " pq_distances.size():" << pq_distances.size()
            //           << " nprobe:" << nprobe << " working_list:"<< working_list
            //           << std::endl;
            //assert(pq_distances.size() == actual_submit);

            AsyncReadRequests_Partial_PQDecode request_p;

            for(int i = 0; i < actual_submit; i++){

                idx_t list_no = keys[cur_q*nprobe + i + working_list];

#ifdef CACHE_MODE
                if(!cached_list_info[list_no]){
                    //std::cout << "pushed_lists continue:" << pushed_lists << std::endl;
                    //assert(list_no == keys[cur_q*nprobe + pq_map[pushed_lists]]);
                    record_uncached.push_back(i + working_list);
                    continue;
                }
#endif

                float cur_dis = coarse_dis[cur_q*nprobe + i + working_list];
                float base_dis = coarse_dis[cur_q*nprobe];
                if(cur_dis > base_dis * prune_factor){
                    break;
                }

                std::vector<int> vector_to_search;
                std::vector<int> vector_to_submit;
                size_t list_size = invlists->list_size(list_no);

                // 准备将有希望称为结果的candidates进行提交
                const size_t* map = invlists->get_inlist_map(list_no);

                /*
                TODO L1.如果map为空，那么就在这里读取pq码和list
                     pq_distance 肯定也是为空的，empty
                */

                auto time_start = std::chrono::high_resolution_clock::now();
                int reserve_size = 0;
                float* dis_line = pq_distances[i].data();
                if(pq_distances[i].size() != list_size){
                    FAISS_THROW_MSG("pq_size and list_size are not equal");
                }

                float dynamic_estimate_factor_partial = 1.03;
                if(i + working_list > 200){
                    dynamic_estimate_factor_partial = estimate_factor_partial;
                }else{
                    dynamic_estimate_factor_partial = estimate_factor_partial;
                }
                //std::cout << "heap_sim:" << heap_sim[0] << "\n";
                for(size_t j = 0; j < list_size; j++){
                    if (dis_line[j] < heap_sim[0] * dynamic_estimate_factor_partial){
                        reserve_size++;
                    }
                }

                if(reserve_size == 0){
                    continue;
                }
                faiss::indexIVFPQDisk2_stats.searched_lists++;

                //std::cout << "reserve_size:" << reserve_size << std::endl;
                vector_to_search.reserve(reserve_size + 10);
                vector_to_submit.reserve(reserve_size + 10);
                //std::cout << "reserve_end"<< std::endl;
                int reserve_next = 0;

                bool fully_read = false;
                if(working_list < top)
                    fully_read = true;

                for(size_t j = 0; j < list_size; j++){
                    if (dis_line[j] < heap_sim[0] * dynamic_estimate_factor_partial){
                        //if(working_list < top)
                        if(fully_read)
                        {
                            vector_to_search.push_back(j);
                            vector_to_submit.push_back(j);
                        }else{
                            vector_to_search.push_back(j);
                            vector_to_submit.push_back(map[j]);
                        }
                        reserve_next++;
                    }
                }
                //std::cout << "reserve_next:" << reserve_next << std::endl;
                size_t vectors_num = vector_to_search.size();

                auto cce_time_start = std::chrono::high_resolution_clock::now();
                int cluster_pos = diskInvertedListHolder.is_cached(list_no);
                if(cluster_pos!= -1){
                    // 使用cache数据
                    //std::cout << "cache!\n" ;
                    int* vec_pos = vector_to_search.data();
                    const void* buffer = diskInvertedListHolder.get_cache_data(cluster_pos);
                    const idx_t* ids = invlists->get_ids(list_no);
                    float distance;
                    for(int m = 0; m < vectors_num; m++){
                        if(this->valueType == "uint8")
                            distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)buffer) +d*map[vec_pos[m]], d);
                        else
                            distance = fvec_L2sqr_simd(current_query, (float*)buffer +d*map[m], d);
                        heap_handler->add(cur_q, distance, ids[vec_pos[m]]);
                        //TODO 增加一个统计项目
                    }
                    indexIVFPQDisk2_stats.cached_vector_access += vectors_num;
                    auto cce_time_end = std::chrono::high_resolution_clock::now();
                    indexIVFPQDisk2_stats.cached_calculate_elapsed += cce_time_end - cce_time_start;
                    continue;
                }
                else{
                }

                if(vectors_num <= 1){
                    continue;
                }
                if(!fully_read)
                {
                    std::vector<size_t> indices(vectors_num);
                    for(int ii = 0; ii < vectors_num; ii++){
                        indices.data()[ii] = ii;
                    }

                    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                        return vector_to_submit[a] < vector_to_submit[b];
                    });
                    // 按 indices 顺序重新排列 vector_to_search 和 vector_to_submit
                    std::vector<int> sorted_vector_to_search(vectors_num);
                    std::vector<int> sorted_vector_to_submit(vectors_num);

                    for (size_t i = 0; i < vectors_num; ++i) {
                        sorted_vector_to_search[i] = vector_to_search[indices[i]];
                        sorted_vector_to_submit[i] = vector_to_submit[indices[i]];
                    }
                    vector_to_search = std::move(sorted_vector_to_search);
                    vector_to_submit = std::move(sorted_vector_to_submit);
                }

                auto time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.memory_2_elapsed+=time_end - time_start;
                time_start = std::chrono::high_resolution_clock::now();

                std::vector<int> page_to_search(vectors_num);
                std::vector<size_t> vec_page_proj(vectors_num);

                int num_page_to_search = diskIOprocessor->process_page(vector_to_submit.data(), page_to_search.data(),
                                                           vec_page_proj.data(), vectors_num);


                Aligned_Cluster_Info* cluster_info = &aligned_cluster_info[list_no];

                const int per_page_element = diskIOprocessor->get_per_page_element();
                const int per_page_vector = per_page_element/d;

                bool not_aligned = false;
                if(per_page_element%d != 0)
                    not_aligned = true;

                std::vector<PageSegment> merged_segments;
                int max_continous_pages = 6;
                merged_segments.reserve(num_page_to_search);

                if (num_page_to_search > 0) {
                    // TODO 这里时间消耗比较多！！合并一些连续的页
                    merge_pages_2(merged_segments, page_to_search.data(), vector_to_submit.data(), vector_to_search.data(), vec_page_proj.data(),
                                vectors_num, num_page_to_search, per_page_element, per_page_vector, d, list_no,max_continous_pages);
                }else{
                    continue;
                }

                if(working_list < top){
                    faiss::indexIVFPQDisk2_stats.searched_vector_full += vectors_num;
                    faiss::indexIVFPQDisk2_stats.searched_page_full += num_page_to_search;
                    faiss::indexIVFPQDisk2_stats.requests_full += merged_segments.size();
                }
                else{
                    faiss::indexIVFPQDisk2_stats.searched_vector_partial += vectors_num;

                    for(int seg = 0; seg<merged_segments.size(); seg++){
                        faiss::indexIVFPQDisk2_stats.searched_page_partial += merged_segments[seg].page_count;
                    }
                    faiss::indexIVFPQDisk2_stats.requests_partial += merged_segments.size();
                }


                /* DISKIO里测试一下实际IO的速度？*/


                /*Async IO info*/
                size_t global_start = cluster_info->page_start;
                size_t prepare_size = request_p.list_requests.size();
                request_p.list_requests.reserve(prepare_size + merged_segments.size());

                //std::cout << "request prepare:" << merged_segments.size() << std::endl;

                for (size_t j = 0; j < merged_segments.size(); ++j) {
                    const auto& segment = merged_segments[j];

                    size_t page_num = segment.page_count;
                    size_t offset = (global_start + segment.start_page) * PAGE_SIZE;

                    size_t total_vector_num = page_num * per_page_vector;   // beam search
                    size_t begin_idx = segment.start_page * per_page_vector;

                    /*
                    TODO L2: 如果map和ids为null的，此时前面已经读取好了，需要保存一下，也许unique？
                    然后在这里使用
                    */

                    const size_t* map = invlists->get_inlist_map(segment.list_no);  // convert result
                    const idx_t* ids = invlists->get_ids(segment.list_no);

                    int* in_buffer_begin = segment.in_buffer_offsets;
                    int* in_buffer_end = segment.in_buffer_offsets + segment.length;

                    size_t* in_ids_start = segment.in_buffer_ids;
                    size_t* in_ids_end = segment.in_buffer_ids + segment.length;

                    request_p.list_requests.emplace_back(page_num, total_vector_num, begin_idx, offset, 0, map, ids,
                    in_buffer_begin, in_buffer_end, in_ids_start, in_ids_end);

                }

                time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.memory_3_elapsed+=time_end - time_start;
                //std::cout << "pq end\n";
            }

            request_p.cal_callback = [&](AsyncRequest_Partial* requested, void* buffer){
                auto time_start = std::chrono::high_resolution_clock::now();
                int* element_offsets = requested->in_buffer_offsets.data();
                size_t* element_ids = requested->in_buffer_ids.data();

                float distance;
                const size_t* map = requested->map;
                const idx_t * list_ids = requested->ids;

                std::vector<float> float_vector(d);

                for(int m = 0; m < requested->in_buffer_offsets.size(); m++){
                    distance = fvec_L2sqr_simd(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, element_offsets[m]), d);
                    heap_handler->add(cur_q, distance, list_ids[element_ids[m]]);
                    //indexIVFPQDisk2_stats.searched_vector_partial++;
                }
                auto time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.rank_elapsed+=time_end - time_start;
            };

            request_p.pq_callback = [&](){
                auto time_start = std::chrono::high_resolution_clock::now();
                if(pq_switch){
                    //std::cout << " Decoding PQ while IO."  << " pqdone:" << pq_todo<< std::endl;
                    //pq_distances.resize(pq_todo);
                    //pq_ids.resize(pq_todo);

                    decode_pq_lists(pqdecoder, nullptr, keys+cur_q*nprobe + pq_done + top, coarse_dis+cur_q*nprobe + pq_done + top,
                                    pq_todo, invlists, pq_distances, pq_ids, this->cached_list_info);

                    // for(int m = 0; m < pq_todo; m++){
                    //     //std::cout << k_s + lis<< " \n" ;
                    //     idx_t list_no_cb = keys[cur_q*nprobe + m + pq_done + top];

                    //     //std::cout << "pq_cb key:" << cur_q*nprobe + m + pq_done << "  pq_cb list_no:" << list_no_cb << std::endl;
                    //     float list_dis_cb = coarse_dis[cur_q*nprobe + m + pq_done + top];
                    //     size_t list_size_cb = invlists->list_size(list_no_cb);

                    //     /*
                    //     TODO L3:pq下一批次的内容，如果此时得到的结果为空，那么就只初始化pq_distances和pq_ids的大小，不需要存储内容
                    //     */
                    //     //std::cout << "list_no:" << list_no_cb << " list_dis:" << list_dis_cb <<std::endl;
                    //     pq_distances[m].resize(list_size_cb);
                    //     pq_ids[m].resize(list_size_cb);
                    //     pqdecoder->set_list(list_no_cb, list_dis_cb);
                    //     pqdecoder->scan_codes(list_size_cb,
                    //                         invlists->get_codes(list_no_cb),
                    //                         invlists->get_ids(list_no_cb),
                    //                         pq_distances[m].data(),
                    //                         pq_ids[m].data());
                    // }
                    indexIVFPQDisk2_stats.pq_list_partial+=pq_todo;
                    pq_done += pq_todo;
                }

                pq_todo = 0;
                pq_switch = false;

                if(pq_cross_query){
                    //std::cout << "query:" <<cur_q<<"  pq decode in advance!!!" << std::endl;
                    int next_q = cur_q+1;
                    working_list = top;
                    pq_done = 0;

                    actual_submit = std::min(submit_per_round, (int)nprobe - working_list);
                    //pq_distances.resize(actual_submit);
                    //pq_ids.resize(actual_submit);

                    decode_pq_lists(pqdecoder, x + next_q*d, keys + next_q*nprobe + pq_done + top, coarse_dis + next_q*nprobe + pq_done + top,
                                    actual_submit, invlists, pq_distances, pq_ids, this->cached_list_info);

                    /*
                    TODO L3:pq下一批次的内容，如果此时得到的结果为空，那么就只初始化pq_distances和pq_ids的大小，不需要存储内容
                    */
                    // pqdecoder->set_query(x + next_q*d);
                    // //std::cout << "working_list: " << working_list << "  pq_done: " << pq_done << std::endl;
                    // for(int i = 0; i < actual_submit; i++){
                    //     idx_t list_no_cq = keys[next_q*nprobe + i + pq_done + top];
                    //     float list_dis_cq = coarse_dis[next_q*nprobe + i + pq_done + top];
                    //     size_t list_size_cq = invlists->list_size(list_no_cq);
                    //     pq_distances[i].resize(list_size_cq);
                    //     pq_ids[i].resize(list_size_cq);
                    //     pqdecoder->set_list(list_no_cq, list_dis_cq);
                    //     pqdecoder->scan_codes(list_size_cq,
                    //                         invlists->get_codes(list_no_cq),
                    //                         invlists->get_ids(list_no_cq),
                    //                         pq_distances[i].data(),
                    //                         pq_ids[i].data());

                    // }
                    // indexIVFPQDisk2_stats.pq_list_partial += actual_submit;
                    pq_cross_extra_decode = false;
                    pq_done += actual_submit;      // pq_done 一定会小于nprobe

                }

                auto time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.pq_elapsed+=time_end - time_start;

            };

            //std::cout << "pq begin\n";
            diskIOprocessor->disk_io_partial_async_pq(request_p);

            //pq_distances.clear();
            //pq_ids.clear();
            auto time_start = std::chrono::high_resolution_clock::now();
            diskIOprocessor->submit();  // 执行IO和回调函数
            auto time_end = std::chrono::high_resolution_clock::now();
            indexIVFPQDisk2_stats.disk_partial_elapsed+=time_end - time_start;
            if(pq_distances.empty()){
                //std::cout << "NO DATA!!!" << std::endl;
                if(cur_q != n-1)
                    pq_cross_extra_decode = true;
                break;
            }

            if(pq_cross_query){
                //std::cout << " Execute cross\n";
                pq_cross_query = false;
                break;
            }

            working_list+=actual_submit;
        }
#ifdef CACHE_MODE
        if(!record_uncached.empty()){
            UncachedList new_list;
            new_list.q = cur_q;  // 假设有一个 `query_id`
            new_list.list_pos = std::move(record_uncached);  // 移动数据以避免拷贝
            uncached_lists.push_back(std::move(new_list));  // 移动到列表中
        }
#endif

    }
}

//  1. Beam search
#define BLOCK_BEAM_SEARCH




//  2. PQ 提前
#define FULL_DECODE_VOLUME 6       // ---- 120
#define PARTIAL_ONE_DECODE_VOLUME 5  // ---- 100
#define PARTIAL_TWO_DECODE_VOLUME 5  // ---- 100


//#define FULL_RATIO_STATS


// 统计函数
namespace{
void visualizeRange(int min, int max, int total) {
    if (total <= 0 || min < 0 || max < 0 || min > max || max > total) {
        std::cerr << "Invalid input values." << std::endl;
        return;
    }

    // 设置比例尺和绘图符号
    const int scale = 50; // 可视化总长度
    const char filledChar = '='; // 填充区间的符号
    const char emptyChar = '.'; // 非填充区间的符号

    // 计算比例
    double minRatio = static_cast<double>(min) / total;
    double maxRatio = static_cast<double>(max) / total;

    // 计算填充的起始和结束位置
    int minPos = static_cast<int>(minRatio * scale);
    int maxPos = static_cast<int>(maxRatio * scale);

    // 构造可视化字符串
    std::string bar(scale, emptyChar);
    for (int i = minPos; i < maxPos; ++i) {
        bar[i] = filledChar;
    }

    // 输出可视化结果
    std::cout << "0 ";
    std::cout << bar;
    std::cout << " " << total << std::endl;

    // 输出百分比信息
    double rangePercent = (max - min) * 100.0 / total;
    std::cout << "Range: [" << min << ", " << max << "] ";
    std::cout << "Percentage: " << std::fixed << std::setprecision(2) << rangePercent << "%" << std::endl;
}
}



// 3.1 直接砍掉一些full（精度会下降）

#define FETCH_FULL 1.0

// 3.2 根据PQ来减少一些full

#define REDUCED_FULL_LIST
#define LOSSLESS_FULL_REDUCTION



#ifdef REDUCED_FULL_LIST
namespace{
    struct DiskReadResult {
        size_t begin_idx;  // 起始向量位置
        size_t end_idx;    // 结束向量位置（不包含）
        size_t start_page; // 起始磁盘页
        size_t end_page;   // 结束磁盘页
        size_t iobuffer_offset;
    };

    DiskReadResult optimizeDiskRead(size_t min, size_t max, size_t list_size, size_t per_page_element_num, size_t d) {
        
        size_t vector_size_bytes = d * PAGE_SIZE / per_page_element_num;
        size_t start_page = (min * vector_size_bytes) / PAGE_SIZE;
        size_t end_page = (max * vector_size_bytes) / PAGE_SIZE;

        size_t start_offset = start_page * PAGE_SIZE;

        size_t begin_idx;
        size_t iobuffer_offset = 0;
        if (start_offset % vector_size_bytes == 0) {
            begin_idx = start_offset / vector_size_bytes;
        } else {
            // 跳过页中不完整部分的数据
            iobuffer_offset = vector_size_bytes - (start_offset % vector_size_bytes);
            begin_idx = (start_offset + iobuffer_offset) / vector_size_bytes;
        }

        // 确定结束位置的索引（防止越界）
        size_t end_offset = std::min((end_page + 1) * PAGE_SIZE, list_size * vector_size_bytes);
        size_t end_idx = end_offset / vector_size_bytes;

        return {begin_idx, end_idx, start_page, end_page, iobuffer_offset};
    }

    struct MinMaxStats {
        size_t min = std::numeric_limits<size_t>::max();
        size_t min_2 = std::numeric_limits<size_t>::max();
        size_t max = 0;
        size_t max_2 = 0;
        bool skip = true;

        inline void not_skip(){
            skip = false;
        }

        inline bool skip_2(){
            return min_2 > max_2;
        }

        void update(size_t value) {
            // 更新最大值和次大值
            if (value > max) {
                max_2 = max;  // 当前最大值变为次大值
                max = value;
            } else if (value > max_2 && value < max) {
                max_2 = value;
            }

            // 更新最小值和次小值
            if (value < min) {
                min_2 = min;  // 当前最小值变为次小值
                min = value;
            } else if (value < min_2 && value > min) {
                min_2 = value;
            }
        }

        size_t get_min2(){
            if(min_2 > 500000)
                return 0;
            else{
                if(max_2 < min_2)
                    return max_2; 
                else
                    return min_2;
            }

                
        }

        size_t get_max2(){
            if(min_2 > 500000)
                return 0;
            if(max_2 < min_2)
                return min_2; 
            else
                return max_2;
        }
    };


    void calculateMinMax(const float* pq_dis_c, size_t vectors_num, float estimator, size_t k, std::priority_queue<float>& queue,
                     MinMaxStats& stats) {
        
        for (size_t m = 0; m < vectors_num; m++) {
            float value = pq_dis_c[m];
            if (queue.size() < k) {
                queue.push(value);
            } 
            // 如果堆已满，且当前值比堆顶小，替换堆顶
            else if (value < queue.top()) {
                queue.pop();
                queue.push(value);
            }
        }


        for (size_t m = 0; m < vectors_num; m++) {
            if (pq_dis_c[m] < queue.top() * estimator) {
                stats.update(m);
                stats.not_skip();
            }
        }
    }

}
#endif

void IndexIVFPQDisk2::search_o(
    idx_t n,
    const float* x,
    idx_t k,
    idx_t nprobe,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    DiskIOProcessor* diskIOprocessor,
    DiskResultHandler* heap_handler
#ifdef CACHE_MODE
    ,UncachedLists& uncached_lists
#endif
    ) const{
        if (n == 0) {
            return;
        }
        // used by top and partial
        std::vector<std::vector<float>> pq_distances;
        std::vector<std::vector<idx_t>> pq_ids;
        pq_distances.resize(nprobe);
        pq_ids.resize(nprobe);

        // global variable
        size_t current_pqed_list = 0;
        size_t pq_stage = 0;   // 0 for full
                               // 1 for partial stage 1
                               // 2 for partial stage 2
                               // 3 for others
        // part top:
        const size_t top_probes = std::min((size_t)nprobe, top);
        
        const bool* cached_list_info = this->cached_list_info;
        std::vector<idx_t> working_lists_ids(top_probes);
        std::vector<float> working_lists_dis(top_probes);
        //idx_t* p_working_lists_ids = working_lists_ids.data();
        //float* p_working_lists_dis = working_lists_dis.data();

        // part_partial:
        // TODO: modify some parameters
        // 对每一个query而言，记录已经解码、即将解码、已经完成的list的数据
        int working_list = top;
        int pq_done = 0;       // 从partial开始算起
        bool pq_switch = false;
        int pq_todo = 0;
        bool pq_cross_query = false;  // 在程序末尾 计算下一个query的pq
        bool pq_cross_extra_decode = true;  // 有些回调函数由于查询原因不会执行

        size_t submit_per_round = 10;

        // used by top and partial
        BaseDecoder* pqdecoder = get_pqdecoder(*this,this->metric_type);

        for(int cur_q = 0; cur_q < n; cur_q++){
            const float* current_query = x + cur_q*d;
            float* heap_sim = distances + cur_q*k;
            idx_t* heap_ids = labels + cur_q*k;

            current_pqed_list = top_probes;
            pq_stage = 0;

            auto time_start = std::chrono::high_resolution_clock::now();

            size_t pre_decode = std::min(top, (size_t)nprobe);

            //1. 对每一个query，先直接解码full所需要的内容
            decode_pq_lists(pqdecoder, x+cur_q*d, keys + cur_q*nprobe, coarse_dis+cur_q*nprobe,
                            0, pre_decode, invlists, pq_distances, pq_ids, cached_list_info);

            //std::cout << "pre_pq " << cur_q << " \n";
            auto time_end = std::chrono::high_resolution_clock::now();
            indexIVFPQDisk2_stats.memory_1_elapsed += time_end - time_start;

            //------------------- full stage begin ----------------- //
            AsyncReadRequests_Full_PQDecode requests_full;
            requests_full.list_requests.reserve(top_probes);

            Aligned_Cluster_Info* acinfo;
            //std::cout << "request_num" << request_num << std::endl;
            int pushed_lists = 0;
            int pre_pushed_lists = 0;
            int pushed_requests = 0;
#ifdef CACHE_MODE
            std::vector<idx_t> record_uncached;
            record_uncached.reserve(nprobe);
#endif
#ifdef FULL_RATIO_STATS
            size_t list_count = 0;
#endif  

#ifdef REDUCED_FULL_LIST
        std::priority_queue<float> pq_queue;
#endif

            //std::cout << "Full 2 " << "\n";
            while(pushed_lists < top_probes){
                size_t list_no = keys[cur_q*nprobe + pushed_lists];
                //std::cout << "list_no:" << list_no << "\n";
#ifdef CACHE_MODE
                if(!cached_list_info[list_no]){
                    //std::cout << "pushed_lists continue:" << pushed_lists << std::endl;
                    //assert(list_no == keys[cur_q*nprobe + pq_map[pushed_lists]]);
                    record_uncached.push_back(pushed_lists++);
                    continue;
                }
#endif
                size_t list_size = invlists->list_size(list_no);
                //const size_t* map = invlists->get_inlist_map(list_no);  // convert result
                const idx_t* ids = invlists->get_ids(list_no);
                acinfo = &aligned_cluster_info[list_no];
                /*
                TODO 如果某个list已经cache了，那么就直接计算，不用加进去
                */
#ifdef CACHE_MODE
                auto cce_time_start = std::chrono::high_resolution_clock::now();
                int cluster_pos = diskInvertedListHolder.is_cached(list_no);
                if(cluster_pos!= -1){
                    //std::cout << "using cached data, listno=" << list_no << "  cluster_pos="  << cluster_pos << std::endl;
                    float* pq_dis_c = pq_distances[pushed_lists].data();
                    const void* buffer = diskInvertedListHolder.get_cache_data(cluster_pos);
                    float distance;
                    for(int m = 0; m < list_size; m++){
                        if(pq_dis_c[m] < heap_sim[0]*1.10){
                            if(this->valueType == "uint8")
                                distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)buffer) +d*m, d);
                            else
                                distance = fvec_L2sqr_simd(current_query, (float*)buffer +d*m, d);
                            heap_handler->add(cur_q, distance, ids[m]);
                        }
                    }
                    indexIVFPQDisk2_stats.cached_list_access += 1;
                    indexIVFPQDisk2_stats.cached_vector_access += list_size;
                    pushed_lists++;

                    //std::cout << "cache list end" << std::endl;
                    auto cce_time_end = std::chrono::high_resolution_clock::now();
                    indexIVFPQDisk2_stats.cached_calculate_elapsed += cce_time_end - cce_time_start;
                    continue;
                }
#endif
                //std::cout << "FETCH_FULL:" << (size_t)(acinfo->page_count * FETCH_FULL) << "\n";
#ifndef REDUCED_FULL_LIST
                size_t reduced_begin_page = acinfo->page_start * PAGE_SIZE;
                size_t reduced_fetch_pages = (size_t)(acinfo->page_count * FETCH_FULL);
                if(reduced_fetch_pages == 0) reduced_fetch_pages++;
                size_t reduced_fetch_size = (size_t)(list_size * FETCH_FULL);
                size_t reduced_begin_idx = 0;
                size_t reduced_iobuffer_offset = 0;
#else
                
                MinMaxStats full_prune;
                calculateMinMax(pq_distances[pushed_lists].data(), list_size, estimate_factor_high_dim, k,  pq_queue, full_prune);
                //visualizeRange(full_prune.min, full_prune.max, list_size);
                
#ifdef LOSSLESS_FULL_REDUCTION
                //std::cout << "min:" <<full_prune.min << " max:" << full_prune.max  << "\n";
                if(full_prune.skip){
                    pushed_lists++;
                    continue;
                }
                DiskReadResult redret = optimizeDiskRead(full_prune.min, full_prune.max, list_size, diskIOprocessor->get_per_page_element(), d);
#else   
                //std::cout << "min_2:" <<full_prune.min_2 << " max_2:" << full_prune.max_2  << "\n";
                if(full_prune.skip_2()){
                    pushed_lists++;
                    continue;
                }
                DiskReadResult redret = optimizeDiskRead(full_prune.get_min2(), full_prune.get_max2(), list_size, diskIOprocessor->get_per_page_element(), d);
#endif
                size_t reduced_begin_page = (acinfo->page_start + redret.start_page)  * PAGE_SIZE;
                size_t reduced_fetch_pages = redret.end_page - redret.start_page + 1;
                size_t reduced_begin_idx = redret.begin_idx;
                size_t reduced_fetch_size = redret.end_idx;
                size_t reduced_iobuffer_offset = redret.iobuffer_offset;
                // std::cout << "reduced_fetch_pages: " << reduced_fetch_pages << "\n";
                // std::cout << "begin_pages: " << redret.start_page << "\n";
                // std::cout << "end_pages: " << redret.end_page << "\n";

                // std::cout << "reduced_fetch_pages: " << reduced_fetch_pages << "\n";
                // std::cout << "reduced_begin_idx: " << reduced_begin_idx << "\n";
                // std::cout << "reduced_end_idx: " << reduced_fetch_size << "\n\n";

#endif
                // 新增一个request
                if(reduced_fetch_pages == 0){
                    continue;
                }
                requests_full.list_requests.emplace_back(reduced_begin_page,
                                                    reduced_fetch_pages,
                                                    reduced_fetch_size,
                                                    reduced_begin_idx,
                                                    list_size,
                                                    reduced_iobuffer_offset,
                                                    nullptr,
                                                    ids,
                                                    pq_distances[pushed_lists].data());
                //std::cout << "list_size:"<< list_size << "  pq_size:" << pq_distances[pq_map[pushed_lists]].size() << std::endl;
                assert(list_size == pq_distances[pushed_lists].size());
                pushed_lists++;
                pushed_requests++;
                pre_pushed_lists++;
                //std::cout << "pushed_lists:" << pushed_lists << "  pushed_requests:"<< pushed_requests << std::endl;
                //std::cout << "acinfo->page_count:" << acinfo->page_count<<" list_size:" << list_size << "\n";
                
                indexIVFPQDisk2_stats.searched_vector_full+=reduced_fetch_size;
                indexIVFPQDisk2_stats.searched_page_full+=acinfo->page_count;
                faiss::indexIVFPQDisk2_stats.searched_lists++;
            }
            //std::cout << "\n\n";
            indexIVFPQDisk2_stats.requests_full+=pushed_requests;
            //std::cout << "pushed_requests:" << pushed_requests << std::endl;
            /*
            Callback function
            */
            requests_full.cal_callback = [&](AsyncRequest_Full* requested, void* buffer){
                auto time_start = std::chrono::high_resolution_clock::now();

                float distance;
                //const size_t* map = requested->map;
                const idx_t* list_ids = requested->ids;
                const float* pq_dis_c = requested->pq_dis.data();
                //size_t vector_num = requested->vectors_num;
                //idx_t tmp_list_no = requested->list_no;
                //const idx_t* list_ids = invlists->get_ids(tmp_list_no);

                //std::cout << "vector number = " << requested->vectors_num << "\n";
                //std::cout << "heap_sim:" << heap_sim[0] << "\n";
// full 统计
#ifdef FULL_RATIO_STATS
                size_t total_f = requested->vectors_num;
                size_t min = total_f, max = 0;
                size_t min_2 = total_f, max_2 = 0;
                size_t calcu = 0;
#endif  
// full 统计
                /*
                    begin_idx 32的倍数  vectors_num 取min(list_size, the last one in the block)
                */
                int buffer_m = 0;
                uint8_t* right_buffer = ((uint8_t*)buffer) + requested->iobuffer_offset;
                for(int m = requested->begin_idx; m < requested->vectors_num; m++, buffer_m++){
                    //distance = fvec_L2sqr_simd(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(), buffer, d*map[m]), d);
                    //std::cout << "pq:" << pq_dis_c[m] << "  heap_sim:" << heap_sim[0] << "\n";
                    if(pq_dis_c[m] < heap_sim[0]*this->estimate_factor)
                    {
                        if(this->valueType == "uint8")
                            distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)right_buffer) +d*buffer_m, d);
                        else
                            distance = fvec_L2sqr_simd(current_query, (float*)right_buffer +d*buffer_m, d);
                        
                        //distance = fvec_L2sqr_simd(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(), buffer, d*map[m]), d);
                        //distance = fvec_L2sqr(current_query, (float*)buffer +d*map[m], d);
                        heap_handler->add(cur_q, distance, list_ids[m]);

                        // for(int ii = 0; ii < d; ii++){
                        //     std::cout << ((float*)buffer +d*map[m])[d] << " ";
                        // }
                        //std::cout << std::endl;
                        // std::cout << "pq:" << pq_dis_c[m] << "  heap_sim:" << heap_sim[0] << " ";
                        //std::cout << "cur_q:" << cur_q << " distance:" << distance << " id:" << list_ids[m] << std::endl;
                        indexIVFPQDisk2_stats.searched_vector_full++;
// full 统计
#ifdef FULL_RATIO_STATS
                        if (max < m) {
                            max_2 = max;
                            max = m;
                        } else if (max_2 < m) {
                            max_2 = m;
                        }

                        if (min > m) {
                            min_2 = min;
                            min = m;
                        } else if (min_2 > m) {
                            min_2 = m;
                        }
                        calcu ++;
#endif  
// full 统计
                    }
                    //std::cout << "cur_q:" << cur_q << " distance:" << distance << " id:" << list_ids[m] << std::endl;
                }
                auto time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.rank_elapsed+=time_end - time_start;

               // full 统计
#ifdef FULL_RATIO_STATS
                if((list_count++)%10 == 0 ){
                    std::cout << "\n\nNew query\n";
                }
                std::cout << "\n\n First:\n ";
                visualizeRange(min,max,total_f);
                std::cout << " Second:\n";
                std::cout << " min_2 = " << min_2 << "  max_2 =" << max_2 << "\n";
                visualizeRange(min_2,max_2,total_f);
                std::cout << " calcu time:" << calcu << "\n";
#endif  
// full 统计 
            };
            requests_full.pq_callback = [&](){

                auto time_start = std::chrono::high_resolution_clock::now();
                //std::cout << "pq:" << cur_q+1 << "\n" ;
                if(top_probes < nprobe){
                    // eg. top = 20, 此时解码 20*6个list， 如果不足，则解码剩余的
                    
                    size_t decode_batch = std::min(top*FULL_DECODE_VOLUME, nprobe - top);
                    if(nprobe - top > submit_per_round){
                        decode_batch = submit_per_round;
                    }
                    decode_pq_lists(pqdecoder, nullptr, keys + cur_q*nprobe, coarse_dis + cur_q*nprobe,
                                    current_pqed_list, decode_batch, invlists, pq_distances, pq_ids, cached_list_info);
                    current_pqed_list += decode_batch;
                    pq_stage = 1;
                }
                auto time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.pq_elapsed+=time_end - time_start;
            };

            //std::cout << "page_buffer prepare" << std::endl;
            diskIOprocessor->disk_io_full_async_pq(requests_full);
            //std::cout << "page_buffer ok" << std::endl;


            time_start = std::chrono::high_resolution_clock::now();
            diskIOprocessor->submit(pushed_requests);
            time_end = std::chrono::high_resolution_clock::now();
            indexIVFPQDisk2_stats.disk_full_elapsed+=time_end - time_start;

            //std::cout << "FULL PQ finished." << std::endl;
            //------------------- full stage end   ----------------- //
            /*
            *
            *
            */
            // ------------------ partial stage begin  -----------------//

            working_list = top;
            size_t actual_submit = 0;
            while(working_list < nprobe){
                // 一次提交submit_per_round 或更少

                //std::cout << "working_list:" << working_list << std::endl;

                actual_submit = std::min((idx_t)submit_per_round, nprobe - (idx_t)working_list);

                if(pq_stage == 1){
                    pq_todo = std::min(submit_per_round * PARTIAL_ONE_DECODE_VOLUME, nprobe - current_pqed_list);
                }else if(pq_stage == 2){
                    pq_todo = std::min(submit_per_round * PARTIAL_TWO_DECODE_VOLUME, nprobe - current_pqed_list);
                }else if(pq_stage >= 3){
                    pq_todo = std::min(submit_per_round, nprobe - current_pqed_list);
                }
                //current_pqed_list += pq_todo;
                pq_stage += 1;
                
                AsyncReadRequests_Partial_PQDecode request_p;
                //std::cout << "stage 1." << std::endl;
                for(int i = 0; i < actual_submit; i++){
                    idx_t list_no = keys[cur_q*nprobe + i + working_list];

                    //std::cout << "list_no:" << list_no << std::endl;
#ifdef CACHE_MODE
                    if(!cached_list_info[list_no]){
                        //std::cout << "pushed_lists continue:" << pushed_lists << std::endl;
                        //assert(list_no == keys[cur_q*nprobe + pq_map[pushed_lists]]);
                        record_uncached.push_back(i + working_list);
                        continue;
                    }
#endif
                    //std::cout << "stage 1.1" << std::endl;
                    float cur_dis = coarse_dis[cur_q*nprobe + i + working_list];
                    float base_dis = coarse_dis[cur_q*nprobe];
                    if(cur_dis > base_dis * prune_factor){
                        break;
                    }

                    std::vector<int> vector_to_search;
                    std::vector<int> vector_to_submit;
                    size_t list_size = invlists->list_size(list_no);
             
                    auto time_start = std::chrono::high_resolution_clock::now();
                    int reserve_size = 0;
                    float* dis_line = pq_distances[working_list + i].data();
                    if(pq_distances[working_list + i].size() != list_size){
                        //FAISS_THROW_MSG("pq_size and list_size are not equal");
                        std::cout << "working list + i:" << working_list + i << "  list_size:" << list_size << std::endl;
                    }

                    //std::cout << "stage 2." << std::endl;

                    float dynamic_estimate_factor_partial = 1.03;
                    if(i + working_list > 200){
                        dynamic_estimate_factor_partial = estimate_factor_partial;
                    }else{
                        dynamic_estimate_factor_partial = estimate_factor_partial;
                    }
                    //std::cout << "heap_sim:" << heap_sim[0] << "\n";
                    for(size_t j = 0; j < list_size; j++){
                        if (dis_line[j] < heap_sim[0] * dynamic_estimate_factor_partial){
                            reserve_size++;
                        }
                    }

                    if(reserve_size == 0){
                        continue;
                    }
                    faiss::indexIVFPQDisk2_stats.searched_lists++;

                    //std::cout << "reserve_size:" << reserve_size << std::endl;
                    vector_to_search.reserve(reserve_size + 10);
                    vector_to_submit.reserve(reserve_size + 10);
                    //std::cout << "reserve_end"<< std::endl;
                    int reserve_next = 0;

                    for(size_t j = 0; j < list_size; j++){
                        if (dis_line[j] < heap_sim[0] * dynamic_estimate_factor_partial){
                            //if(working_list < top)
                            vector_to_search.push_back(j);
                            vector_to_submit.push_back(j);
                            reserve_next++;
                        }
                    }
                    //std::cout << "reserve_next:" << reserve_next << std::endl;
                    size_t vectors_num = vector_to_search.size();

#ifdef CACHE_MODE
                    auto cce_time_start = std::chrono::high_resolution_clock::now();
                    int cluster_pos = diskInvertedListHolder.is_cached(list_no);
                    //std::cout << "cluster_pos:" << cluster_pos << std::endl;
                    if(cluster_pos!= -1){
                        // 使用cache数据
                        std::cout << "cache!\n" ;
                        int* vec_pos = vector_to_search.data();
                        const void* buffer = diskInvertedListHolder.get_cache_data(cluster_pos);
                        const idx_t* ids = invlists->get_ids(list_no);
                        float distance;
                        for(int m = 0; m < vectors_num; m++){
                            if(this->valueType == "uint8")
                                distance = vec_L2sqr_h<uint8_t>(current_query, ((uint8_t*)buffer) +d*vec_pos[m], d);
                            else
                                distance = fvec_L2sqr_simd(current_query, (float*)buffer +d*m, d);
                            heap_handler->add(cur_q, distance, ids[vec_pos[m]]);
                            //TODO 增加一个统计项目
                        }
                        indexIVFPQDisk2_stats.cached_vector_access += vectors_num;
                        auto cce_time_end = std::chrono::high_resolution_clock::now();
                        indexIVFPQDisk2_stats.cached_calculate_elapsed += cce_time_end - cce_time_start;
                        continue;
                    }

#endif
                    if(vectors_num <= 1){
                        continue;
                    }

                    auto time_end = std::chrono::high_resolution_clock::now();
                    indexIVFPQDisk2_stats.memory_2_elapsed+=time_end - time_start;
                    time_start = std::chrono::high_resolution_clock::now();
                    std::vector<size_t> vec_page_proj(vectors_num);
                    
                    
                    // std::vector<int> page_to_search(vectors_num);
                    // int num_page_to_search = diskIOprocessor->process_page(vector_to_submit.data(), page_to_search.data(),
                    //                                         vec_page_proj.data(), vectors_num);
                    std::vector<Page_to_Search> page_to_search(vectors_num);
                    int num_page_to_search = diskIOprocessor->process_page_transpage(vector_to_submit.data(), page_to_search.data(),
                                                            vec_page_proj.data(), vectors_num);

                    Aligned_Cluster_Info* cluster_info = &aligned_cluster_info[list_no];

                    const int per_page_element = diskIOprocessor->get_per_page_element();
                    const int per_page_vector = per_page_element/d;

                    bool not_aligned = false;
                    if(per_page_element%d != 0)
                        not_aligned = true;

                    std::vector<PageSegment> merged_segments;
                    int max_continous_pages = 10;
                    merged_segments.reserve(num_page_to_search);

                    if (num_page_to_search > 0) {
                        // TODO 这里时间消耗比较多！！合并一些连续的页
                        // merge_pages_2(merged_segments, page_to_search.data(), vector_to_submit.data(), vector_to_search.data(), vec_page_proj.data(),
                        //             vectors_num, num_page_to_search, per_page_element, per_page_vector, d, list_no,max_continous_pages);
                        merge_pages_transpage(merged_segments, page_to_search.data(), vector_to_submit.data(), vector_to_search.data(), vec_page_proj.data(),
                                    vectors_num, num_page_to_search, per_page_element, per_page_vector, d, list_no,max_continous_pages);
                    }else{
                        continue;
                    }

                    if(working_list < top){
                        faiss::indexIVFPQDisk2_stats.searched_vector_full += vectors_num;
                        faiss::indexIVFPQDisk2_stats.searched_page_full += num_page_to_search;
                        faiss::indexIVFPQDisk2_stats.requests_full += merged_segments.size();
                    }
                    else{
                        faiss::indexIVFPQDisk2_stats.searched_vector_partial += vectors_num;

                        for(int seg = 0; seg<merged_segments.size(); seg++){
                            faiss::indexIVFPQDisk2_stats.searched_page_partial += merged_segments[seg].page_count;
                        }
                        faiss::indexIVFPQDisk2_stats.requests_partial += merged_segments.size();
                    }

                    /*Async IO info*/
                    size_t global_start = cluster_info->page_start;
                    size_t prepare_size = request_p.list_requests.size();
                    request_p.list_requests.reserve(prepare_size + merged_segments.size());

                    //std::cout << "request prepare:" << merged_segments.size() << std::endl;

                    for (size_t j = 0; j < merged_segments.size(); ++j) {
                        const auto& segment = merged_segments[j];

                        size_t page_num = segment.page_count;
                        size_t start_page = segment.start_page;
                        size_t offset = (global_start + segment.start_page) * PAGE_SIZE;

                        //std::cout <<"page_num: " << segment.page_count << "\n";

                        // block search需要，从第几个元素开始算
                        size_t iobuffer_offset = (d - (start_page*per_page_element) %d ) % d; //? sift ==0

                        size_t total_vector_num = (page_num * per_page_element - iobuffer_offset)/d;

                        //size_t total_vector_num = page_num * per_page_vector;   // block search
                        //size_t begin_idx = segment.start_page * per_page_vector;

                        size_t begin_idx = (start_page*per_page_element + d - 1)/d;
                        if(begin_idx + total_vector_num > list_size){          // 防止超出list_size
                            total_vector_num = list_size - begin_idx;
                        }

                        /*
                        TODO L2: 如果map和ids为null的，此时前面已经读取好了，需要保存一下，也许unique？
                        然后在这里使用
                        */

                        //const size_t* map = invlists->get_inlist_map(segment.list_no);  // convert result
                        const size_t* map = nullptr;
                        const idx_t* ids = invlists->get_ids(segment.list_no);

                        int* in_buffer_begin = segment.in_buffer_offsets;
                        int* in_buffer_end = segment.in_buffer_offsets + segment.length;

                        size_t* in_ids_start = segment.in_buffer_ids;
                        size_t* in_ids_end = segment.in_buffer_ids + segment.length;

                        request_p.list_requests.emplace_back(page_num, total_vector_num, begin_idx, offset, iobuffer_offset, map, ids,
                        in_buffer_begin, in_buffer_end, in_ids_start, in_ids_end);

                    }

                    time_end = std::chrono::high_resolution_clock::now();
                    indexIVFPQDisk2_stats.memory_3_elapsed+=time_end - time_start;
                    //std::cout << "pq end\n";
                }

                request_p.cal_callback = [&](AsyncRequest_Partial* requested, void* buffer){
                    auto time_start = std::chrono::high_resolution_clock::now();
                    int* element_offsets = requested->in_buffer_offsets.data();
                    size_t* element_ids = requested->in_buffer_ids.data();

                    float distance;
                    const idx_t * list_ids = requested->ids;
                    std::vector<float> float_vector(d);
#ifndef BLOCK_BEAM_SEARCH
                    for(int m = 0; m < requested->in_buffer_offsets.size(); m++){
                        distance = fvec_L2sqr_simd(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, element_offsets[m]), d);
                        heap_handler->add(cur_q, distance, list_ids[element_ids[m]]);
                        //indexIVFPQDisk2_stats.searched_vector_partial++;
                    }
#else
                    for(int m = 0; m < requested->vectors_num; m++){
                        distance = fvec_L2sqr_simd(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, m*d + requested->iobuffer_offset), d);
                        heap_handler->add(cur_q, distance, list_ids[requested->begin_idx + m]);
                        //indexIVFPQDisk2_stats.searched_vector_partial++;
                    }
#endif
                    auto time_end = std::chrono::high_resolution_clock::now();
                    indexIVFPQDisk2_stats.rank_elapsed+=time_end - time_start;
                };

                request_p.pq_callback = [&](){
                    auto time_start = std::chrono::high_resolution_clock::now();
                    if(pq_todo > 0){
                        decode_pq_lists(pqdecoder, nullptr, keys+cur_q*nprobe, coarse_dis+cur_q*nprobe,
                                        current_pqed_list, pq_todo, invlists, pq_distances, pq_ids, this->cached_list_info);
                        current_pqed_list += pq_todo;
                        indexIVFPQDisk2_stats.pq_list_partial+=pq_todo;
                        pq_done += pq_todo;
                    }
                    auto time_end = std::chrono::high_resolution_clock::now();
                    indexIVFPQDisk2_stats.pq_elapsed+=time_end - time_start;
                };

                //std::cout << "begin disk io" << std::endl;
                diskIOprocessor->disk_io_partial_async_pq(request_p);

                //std::cout << "disk io well" << std::endl;

                auto time_start = std::chrono::high_resolution_clock::now();
                diskIOprocessor->submit();  // 执行IO和回调函数
                auto time_end = std::chrono::high_resolution_clock::now();
                indexIVFPQDisk2_stats.disk_partial_elapsed+=time_end - time_start;

                working_list+=actual_submit;
                //std::cout << "submit well" << std::endl;
            }

            // ------------------ partial stage end  -----------------//

#ifdef CACHE_MODE
            if(!record_uncached.empty()){
                UncachedList new_list;
                new_list.q = cur_q;  // 假设有一个 `query_id`
                new_list.list_pos = std::move(record_uncached);  // 移动数据以避免拷贝
                uncached_lists.push_back(std::move(new_list));  // 移动到列表中
            }
#endif
        }
    }




namespace{
    inline bool skip_list(float base_dis, float dis, float prune_factor){
        if(dis > prune_factor * base_dis)
            return true;
        return false;
    }

}



void IndexIVFPQDisk2::search_uncached(
    idx_t n,
    const float* x,
    idx_t k,
    idx_t nprobe,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    DiskIOProcessor* diskIOprocessor,
    DiskResultHandler* heap_handler,
    UncachedLists& uncached_lists
    )const {

    if (n == 0) {
        return;
    }

    size_t uncached_query_num = uncached_lists.size();

    if(uncached_query_num == 0){
        return;
    }

    std::vector<DiskInvlist> disk_invlists;   // calculate pq first and then store the info of next query

    /* try to find the max number*/
    int max_lists = 0;
    for(int i = 0; i < uncached_query_num; i++){
        if(uncached_lists[i].list_pos.size() > max_lists )
            max_lists = uncached_lists[i].list_pos.size();
    }

    disk_invlists.resize(max_lists);
    /* try to find the max number*/

    std::vector<std::vector<float>> pq_distances;
    std::vector<std::vector<idx_t>> pq_ids;     // the ids here are from 0 to listno-1;

    pq_distances.resize(max_lists);
    pq_ids.resize(max_lists);

    BaseDecoder* pqdecoder = get_pqdecoder(*this,this->metric_type);

    size_t pq_code_size = this->pq.code_size;
    //std::cout << "pq_code_size:" << pq_code_size << std::endl;

    for(int i = 0; i < uncached_query_num; i++){

        // List num per query
        size_t q = uncached_lists[i].q;

        const float* current_query = x + q*d;

        size_t lists_per_query = uncached_lists[i].list_pos.size();
        idx_t* list_pos = uncached_lists[i].list_pos.data();

        //std::cout << "lists_per_query: " << lists_per_query << "\n";

        float* heap_sim = distances + q*k;
        idx_t* heap_ids = labels + q*k;



        /* 获取disk_invlists信息*/

        auto disk_time_start = std::chrono::high_resolution_clock::now();
        AsyncRequests_IndexInfo requests_info;
        requests_info.info_requests.reserve(lists_per_query);
        int prune_threshold = lists_per_query;
        for(int j = 0; j < lists_per_query; j++){
            size_t list_no = keys[q*nprobe + list_pos[j]];
            /* skip some lists*/
            if(skip_list(coarse_dis[q*nprobe], coarse_dis[q*nprobe + list_pos[j]], prune_factor)){
                prune_threshold = j;
                break;
            }
            Aligned_Invlist_Info* aii = aligned_inv_info + list_no;
            size_t page_num = aii->page_count;
            size_t list_size = aii->list_size;
            std::uint64_t m_readSize = page_num*PAGE_SIZE;
            std::uint64_t m_offset = aii->page_start*PAGE_SIZE;

            requests_info.info_requests.emplace_back(page_num, m_offset, m_readSize, nullptr, list_no, list_size);
        }
        //std::cout << "info_requests: " << requests_info.info_requests.size() << "\n";
        diskIOprocessor->disk_io_info_async(requests_info);
        diskIOprocessor->submit(-2);
        //std::cout << "query " << q << " Info Get!" << std::endl;

        for(int j = 0; j < prune_threshold; j++){
            AsyncRequest_IndexInfo& request = requests_info.info_requests[j];
            disk_invlists[j].set(request.m_buffer, request.list_size, pq_code_size);

            //assert(keys[q*nprobe + list_pos[j]] == requests_info.info_requests[j].list_no);

            // for(int kk =0 ; kk < request.list_size; kk++){
            //     std::cout << "ids " << kk << " :" << disk_invlists[j].get_ids()[kk] << std::endl;
            // }
            // std::cout << "\n\n\n";
        }
        auto disk_time_end = std::chrono::high_resolution_clock::now();
        indexIVFPQDisk2_stats.disk_uncache_info_elapsed+=disk_time_end - disk_time_start;

        auto pq_time_start = std::chrono::high_resolution_clock::now();
        if(true)
        {
            pqdecoder->set_query(x + q*d);
            for(int j = 0; j < prune_threshold; j++){

                //std::cout << "pq decoding: " << j << "\n";

                idx_t list_no = keys[q*nprobe + list_pos[j]];

                size_t list_size = this->aligned_inv_info[list_no].list_size;

                decode_pq_list(pqdecoder, nullptr, list_no, coarse_dis[q*nprobe + list_pos[j]],
                                list_size, &disk_invlists[j], pq_distances[j], pq_ids[j]);
            }
        }

        auto pq_time_end = std::chrono::high_resolution_clock::now();
        indexIVFPQDisk2_stats.pq_uncache_elapsed+=pq_time_end - pq_time_start;



        AsyncReadRequests_Partial_PQDecode request_p;

        //std::cout << "making requests: " << "\n";
        for(int j = 0; j < lists_per_query; j++){
            idx_t list_no = keys[q*nprobe + list_pos[j]];
            size_t list_size = this->aligned_inv_info[list_no].list_size;

            /* skip some lists*/
            if(skip_list(coarse_dis[q*nprobe], coarse_dis[q*nprobe + list_pos[j]], prune_factor)){
                break;
            }

            std::vector<int> vector_to_search;
            std::vector<int> vector_to_submit;
            //const size_t* map = disk_invlists[j].get_map();       // 1. 准备将有希望称为结果的candidates进行提交

            auto time_start = std::chrono::high_resolution_clock::now();

            int reserve_size = 0;                                 // 2. 获取pq数据并检查
            float* dis_line = pq_distances[j].data();


            //assert(pq_distances[j].size() == list_size);
            //assert(disk_invlists[j].get_size() == list_size);

            if(pq_distances[j].size() != list_size){
                FAISS_THROW_MSG("UncachedList: pq_size and list_size are not equal");
            }

            for(size_t kk = 0; kk < list_size; kk++){               // 3. 预先计算需要读取的向量个数
                if (dis_line[kk] < heap_sim[0] * estimate_factor_partial){
                        reserve_size++;
                }
            }
            //std::cout << "list_size: " << list_size <<"  reserve_size:" << reserve_size << "\n";
            if(reserve_size == 0){
                continue;
            }
            faiss::indexIVFPQDisk2_stats.searched_lists++;

            vector_to_search.reserve(reserve_size + 10);         // 4. 记录需要读入的向量在磁盘上的存储位置
            vector_to_submit.reserve(reserve_size + 10);

            for(size_t kk = 0; kk < list_size; kk++){
                if (dis_line[kk] < heap_sim[0] * estimate_factor_partial){
                    vector_to_search.push_back(kk);
                    vector_to_submit.push_back(kk);
                }
            }
            size_t vectors_num = vector_to_search.size();
            if(vectors_num <= 1){
                continue;
            }

            auto time_end = std::chrono::high_resolution_clock::now();
            indexIVFPQDisk2_stats.memory_uncache_elapsed+=time_end - time_start;
            time_start = std::chrono::high_resolution_clock::now();

            std::vector<int> page_to_search(vectors_num);                      // 6. 计算出需要读取的页范围
            std::vector<size_t> vec_page_proj(vectors_num);

            int num_page_to_search = diskIOprocessor->process_page(vector_to_submit.data(), page_to_search.data(),
                                                        vec_page_proj.data(), vectors_num);

            Aligned_Cluster_Info* cluster_info = &aligned_cluster_info[list_no];   // 7. 合并相邻的页
            const int per_page_element = diskIOprocessor->get_per_page_element();
            const int per_page_vector = per_page_element/d;

            std::vector<PageSegment> merged_segments;
            int max_continous_pages = 2;
            merged_segments.reserve(num_page_to_search);

            if (num_page_to_search > 0) {
                merge_pages_2(merged_segments, page_to_search.data(), vector_to_submit.data(), vector_to_search.data(), vec_page_proj.data(),
                            vectors_num, num_page_to_search, per_page_element, per_page_vector, d, list_no, max_continous_pages);
            }else{
                continue;
            }
            {                                                                        // 8. 收集信息
                faiss::indexIVFPQDisk2_stats.searched_vector_partial += vectors_num;
                for(int seg = 0; seg<merged_segments.size(); seg++){
                    faiss::indexIVFPQDisk2_stats.searched_page_partial += merged_segments[seg].page_count;
                }
                faiss::indexIVFPQDisk2_stats.requests_partial += merged_segments.size();
            }

            size_t global_start = cluster_info->page_start;                         // 9. 准备request
            size_t prepare_size = request_p.list_requests.size();
            request_p.list_requests.reserve(prepare_size + merged_segments.size());

            for (size_t kk = 0; kk < merged_segments.size(); ++kk) {
                const auto& segment = merged_segments[kk];

                size_t page_num = segment.page_count;
                size_t offset = (global_start + segment.start_page) * PAGE_SIZE;

                size_t total_vector_num = page_num * per_page_vector;   // beam search
                size_t begin_idx = segment.start_page * per_page_vector;
                if(begin_idx + total_vector_num > list_size){          // 防止超出list_size
                    total_vector_num = list_size - begin_idx;
                }

                // TODO 一个list可能有多个request， 所以我的一个disk_invlist需要与多个list匹配，how？
                //const size_t* map = disk_invlists[j].get_map();
                const size_t* map = nullptr;
                const idx_t* ids = disk_invlists[j].get_ids();

                int* in_buffer_begin = segment.in_buffer_offsets;
                int* in_buffer_end = segment.in_buffer_offsets + segment.length;

                size_t* in_ids_start = segment.in_buffer_ids;
                size_t* in_ids_end = segment.in_buffer_ids + segment.length;

                request_p.list_requests.emplace_back(page_num, total_vector_num, begin_idx, offset, 0,map, ids,
                        in_buffer_begin, in_buffer_end, in_ids_start, in_ids_end);
            }

            time_end = std::chrono::high_resolution_clock::now();
            indexIVFPQDisk2_stats.memory_uncache_elapsed+=time_end - time_start;

        }
        request_p.cal_callback = [&](AsyncRequest_Partial* requested, void* buffer){
            auto time_start = std::chrono::high_resolution_clock::now();
            int* element_offsets = requested->in_buffer_offsets.data();
            size_t* element_ids = requested->in_buffer_ids.data();

            float distance;
            const idx_t * list_ids = requested->ids;

            std::vector<float> float_vector(d);

#ifndef BLOCK_BEAM_SEARCH
            for(int m = 0; m < requested->in_buffer_offsets.size(); m++){
                distance = fvec_L2sqr_simd(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, element_offsets[m]), d);
                heap_handler->add(q, distance, list_ids[element_ids[m]]);
                //indexIVFPQDisk2_stats.searched_vector_partial++;
            }
#else
            for(int m = 0; m < requested->vectors_num; m++){
                distance = fvec_L2sqr_simd(current_query, diskIOprocessor->convert_to_float_single(float_vector.data(),buffer, m*d), d);
                heap_handler->add(q, distance, list_ids[requested->begin_idx + m]);
                //indexIVFPQDisk2_stats.searched_vector_partial++;
            }
#endif
            auto time_end = std::chrono::high_resolution_clock::now();
            indexIVFPQDisk2_stats.rank_uncache_elapsed+=time_end - time_start;
        };

        request_p.pq_callback = [&](){
            // do nothing
            // 现在先把disk 和 pqdecode都放出来，后面继承再弄
        };

        diskIOprocessor->disk_io_partial_async_pq(request_p);
        auto time_start = std::chrono::high_resolution_clock::now();
        diskIOprocessor->submit();  // 执行IO和回调函数
        auto time_end = std::chrono::high_resolution_clock::now();
        indexIVFPQDisk2_stats.disk_uncache_calc_elapsed+=time_end - time_start;

        requests_info.page_buffers.clear();
    }
}



void IndexIVFPQDisk2::search_preassigned(
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
    //indexIVFPQDisk2_stats.others_elapsed += time_end - time_start;

    DiskResultHandler* rh = get_result_handler(n, k, distances, labels);

    int thread_id = omp_get_thread_num();
    DiskIOProcessor* local_processor = diskIOprocessors[thread_id].get();

#ifdef CACHE_MODE
    UncachedLists ul;
    std::cout << "Searching in search_o. Cache mode" << std::endl;
    search_o(n, x, k, nprobe, keys, coarse_dis, distances, labels, local_processor, rh, ul);
    search_uncached(n, x, k, nprobe, keys, coarse_dis, distances, labels, local_processor, rh, ul);
#else
    std::cout << "Searching in search_o." << std::endl;
    search_o(n, x, k, nprobe, keys, coarse_dis, distances, labels, local_processor, rh);
#endif

    auto d_time_start = std::chrono::high_resolution_clock::now();
    rh->end();
    //delete diskIOprocessor;
    auto d_time_end = std::chrono::high_resolution_clock::now();
    indexIVFPQDisk2_stats.delete_elapsed += d_time_end - d_time_start;

    

}

namespace{
    template <typename ValueType>
    DiskIOProcessor* get_DiskIOBuildProcessor_2(std::string& disk_path, size_t d, size_t ntotal){
        return new IVF_DiskIOBuildProcessor<ValueType>(disk_path, d, ntotal);

    }


    template <typename ValueType>
    DiskIOProcessor* get_DiskIOSearchProcessor_2(const std::string& disk_path, const size_t d) {
        //std::cout << "Get Search Processor!" << std::endl;
    #ifndef USING_ASYNC
        return new IVF_DiskIOSearchProcessor<ValueType>(disk_path, d);
    #else
        return new IVF_DiskIOSearchProcessor_Async_PQ<ValueType>(disk_path, d);
    #endif
    }
}


DiskIOProcessor* IndexIVFPQDisk2::get_DiskIOBuildProcessor() {

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

DiskIOProcessor* IndexIVFPQDisk2::get_DiskIOSearchProcessor() const{
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



IndexIVFPQDisk2Stats indexIVFPQDisk2_stats;

void IndexIVFPQDisk2Stats::reset() {
    memset(this, 0, sizeof(*this));
}


}