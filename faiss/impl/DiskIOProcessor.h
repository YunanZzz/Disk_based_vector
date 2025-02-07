#ifndef FAISS_DISKIOPROCESSOR_H
#define FAISS_DISKIOPROCESSOR_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <functional>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>     // for open, O_DIRECT
#include <immintrin.h>
#include <cstddef>

#include <faiss/impl/FaissAssert.h>
#include <faiss/MetricType.h>
#include <faiss/invlists/InvertedLists.h>
#include <cmath>    // For std::ceil
#include <faiss/impl/AsyncIOExtension.h>
#include <faiss/impl/DiskIOStructure.h>

// #include <chrono>
// using namespace std::chrono;



//#include<>

namespace faiss{

struct Rerank_Info{
    std::shared_ptr<std::vector<float>> list_sim;
    std::shared_ptr<std::vector<idx_t>> list_ids;
    float* heap_sim;
    idx_t* heap_ids;

    size_t k;
    idx_t key;

    float factor_fully;
    float factor_partially;

    float* query;

    const idx_t* ids;

};

struct AsyncReadRequest
{
    std::uint64_t m_offset;
    std::uint64_t m_readSize;
    char* m_buffer;
    float* converted_buffer;
    std::function<void(AsyncReadRequest*)> m_callback;
    int m_status;    // 与多文件控制有关，分为前16位与后16位
    size_t len;      // help create a buffer to store float data
    
    // use it when search partially
    // size_t* in_page_offset;
    size_t D;
    Rerank_Info rerank_info;
    // Carry items like counter for callback to process.
    void* m_payload;
    bool m_success;
    // Carry exension metadata needed by some DiskIO implementations
    void* m_extension;

    AsyncReadRequest() : m_offset(0), m_readSize(0), m_buffer(nullptr), m_status(0), m_payload(nullptr), m_success(false), m_extension(nullptr) {}
};


struct AsyncReadRequest_Partial
{
    std::uint64_t m_offset;
    std::uint64_t m_readSize;

    // offset from at the begining of the PAGE. Counted by element, not vectors. 
    // Every offset represent a vector.
    std::vector<int> in_buffer_offsets;
    std::vector<size_t> in_buffer_ids;    


    char* m_buffer;
    float* converted_buffer;
    std::function<void(AsyncReadRequest_Partial*, std::vector<float>&, std::vector<size_t>&)> m_callback;
    std::function<void(AsyncReadRequest_Partial*, std::vector<float>&, std::vector<size_t>&)> m_callback_calculation;
    size_t len;      // help create a buffer to store float data
    //std::vector<size_t> in_page_offset;  // TODO store local varibles

    size_t D;
    Rerank_Info rerank_info;
    // Carry items like counter for callback to process.
    void* m_payload;
    bool m_success;
    // Carry exension metadata needed by some DiskIO implementations
    void* m_extension;

    AsyncReadRequest_Partial() : m_offset(0), m_readSize(0), m_buffer(nullptr), m_payload(nullptr), m_success(false), m_extension(nullptr) {}
};


struct AsyncReadRequest_Partial_PQDecode{
    std::uint64_t m_offset;
    std::uint64_t m_readSize;

    // offset from at the begining of the PAGE. Counted by element, not vectors. 
    // Every offset represent a vector.
    std::vector<int> in_buffer_offsets;
    std::vector<size_t> in_buffer_ids; 

    // must be aligned
    char* m_buffer;

    float* converted_buffer;


    size_t len;      // help create a buffer to store float data
    size_t D;
    idx_t list_no;


    // necessary?
    Rerank_Info rerank_info;

    // Carry items like counter for callback to process.
    void* m_payload;
    bool m_success;
    // Carry exension metadata needed by some DiskIO implementations
    void* m_extension;

    std::function<void(AsyncReadRequest_Partial_PQDecode* requested)> callback_calculation;
    std::function<void()> callback_pqdecode;
};



// make sure each cluster aligning with page
// struct Aligned_Cluster_Info{
//     size_t page_start;    // 1. begining site of the cluster
//     size_t padding_offset;   // 2. padding offset of the cluster
//     size_t page_count;    // 3. page usage of the cluster
// };

// 挪到AsyncIOExtention里
// template<typename T>
// class PageBuffer
// {
// public:
//     PageBuffer()
//         : m_pageBufferSize(0)
//     {
//     }

struct DiskIOProcessor{
    std::string disk_path;
    size_t d;
    
    size_t total_page;

    // only use it in async mode
    //std::function<void(bool)> callback_partial;
    //std::function<void(bool)> callback_full;

    bool verbose = true;

    DiskIOProcessor(std::string disk_path, size_t d, size_t total_page = 0): disk_path(disk_path), d(d), total_page(total_page){
    }

    virtual ~DiskIOProcessor(){
        
    }

    virtual bool align_cluster_page(size_t* clusters,
                                size_t* len,
                                Aligned_Cluster_Info* acInfo,
                                size_t nlist){
                                    FAISS_THROW_MSG("Func reorganize_vectors: Processor Base Class does not support write operation");
                                    return true;
                                };

    virtual bool reorganize_vectors(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            std::vector<std::vector<faiss::idx_t>> ids){
                            FAISS_THROW_MSG("Func reorganize_vectors: Processor Base Class does not support write operation");
                            return true;

                            };
    virtual bool reorganize_vectors_2(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            ClusteredArrayInvertedLists* c_array_invlists,
                            bool do_in_list_cluster){
                                return true;
                            }

    // memory 版本
    virtual bool reorganize_vectors_in_memory(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            ClusteredArrayInvertedLists* c_array_invlists,
                            ArrayInvertedLists* build_invlists,
                            bool do_in_list_cluster,
                            bool keep_disk = false){
                                return true;
                            }

    // 1. invlist按照已有的函数来
    // 2. 改invlist里的map
    // 3. 改acInfo的信息 
    virtual size_t reorganize_list(
        Index& quantizer, 
        ClusteredArrayInvertedLists* c_array_invlists, 
        Aligned_Cluster_Info* acInfo,
        size_t* clusters,
        size_t* len,
        size_t nlist
    ){
        FAISS_THROW_MSG("Func reorganize_vectors: Processor Base Class does not support write operation");
        return 0;         
    }

    virtual bool organize_select_list(
        size_t pq_size,
        size_t entry_size,
        ClusteredArrayInvertedLists* c_array_invlists, 
        Aligned_Invlist_Info* invInfo, 
        size_t nlist,
        std::string select_lists_path
    ){
        FAISS_THROW_MSG("Func organize_select_list: Processor Base Class does not support write operation");
        return false;  
    }
    
    virtual void disk_io_all(int D,
                            size_t len,
                            size_t listno, 
                            float* vectors,
                            Aligned_Cluster_Info* acInfo){
        FAISS_THROW_MSG("Do not call virtual function disk_io_all!");
    }

    virtual void disk_io_single(int D,
                                size_t len,
                                size_t listno,
                                size_t nth,
                                float* vector,
                                Aligned_Cluster_Info* acInfo){
        
    }

    virtual void disk_io_all_async(std::shared_ptr<AsyncReadRequest>& asyncReadRequest){
        FAISS_THROW_MSG("Base IO processor should not be used : disk_io_all_async");
    }

    virtual void submit(int num = -1){
        FAISS_THROW_MSG("Base IO processor should not be used : submit");
    }

    virtual void initial(std::uint64_t maxIOSize = (1 << 20),
                        std::uint32_t maxReadRetries = 2,
                        std::uint32_t maxWriteRetries = 2,
                        std::uint16_t threadPoolSize = 4){
        FAISS_THROW_MSG("Do not call virtual function initial!");
    }

    virtual int process_page(int* vector_to_submit, int* page_to_search, size_t* vec_page_proj, size_t len_p){
            FAISS_THROW_MSG("Func process_page: Processor Base Class does not support this operation");
                            
            return -1;
    }

    virtual int process_page_transpage(int* vector_to_submit, Page_to_Search* page_to_search, size_t* vec_page_proj, size_t len_p){
            FAISS_THROW_MSG("Func process_page_transpage: Processor Base Class does not support this operation");
                            
            return -1;
    }

    virtual void disk_io_partial_async(std::shared_ptr<std::vector<AsyncReadRequest_Partial>>& asyncReadRequests_p){
        
    }

    virtual void disk_io_partial_async_pq(AsyncReadRequests_Partial_PQDecode& asyncReadRequests_p){
        
    }

    virtual void disk_io_full_async_pq(AsyncReadRequests_Full_PQDecode& asyncReadRequests_f){
        
    }

    virtual void disk_io_info_async(AsyncRequests_IndexInfo& asyncReadRequests_i){

    }
    
    virtual void convert_to_float(size_t n, float* vectors, void* disk_data){

    }

    virtual float* convert_to_float_single(float* vectors, void* disk_data, int begin){
        FAISS_THROW_MSG("Func convert_to_float_single: Processor Base Class does not support this operation");
        return nullptr;
    }

    virtual void test(){
        std::cout << "DiskIOBase:" << std::endl;
    }

    virtual int get_per_page_element(){
        return 0;
    }

    virtual int shut_down(){
        FAISS_THROW_MSG("Do not call virtual function shutdown!");
        return 0;
    }
    
};


namespace{

    void in_list_pq_ids_reassign(ClusteredArrayInvertedLists* c_array_invlists, size_t list_no){
        const size_t* map = c_array_invlists->get_inlist_map(list_no);
        if (!map) return;

        // 获取当前 list 的 codes 和 ids
        std::vector<uint8_t>& codes = c_array_invlists->codes[list_no];
        std::vector<idx_t>& ids = c_array_invlists->ids[list_no];
        size_t list_size = ids.size();
        size_t code_size = c_array_invlists->code_size;

        if (list_size == 0) return; // 如果 list 是空的，直接返回

        std::vector<uint8_t> new_codes(list_size * code_size);
        std::vector<idx_t> new_ids(list_size);

        for (size_t i = 0; i < list_size; ++i) {
            size_t new_pos = map[i];
            if (new_pos >= list_size) {
                throw std::runtime_error("Invalid map index");
            }
            // PQ CODE
            std::copy(
                codes.begin() + i * code_size,
                codes.begin() + (i + 1) * code_size,
                new_codes.begin() + new_pos * code_size
            );
            // ID
            new_ids[new_pos] = ids[i];
        }

        // 更新原始数据
        codes = std::move(new_codes);
        ids = std::move(new_ids);
    }

    void in_list_map_reassign(
        size_t n, 
        float* nx, 
        Index& index, 
        ClusteredArrayInvertedLists* invlists, 
        size_t list_no){
        
        assert(n!=0);
        assert(index.ntotal != 0);
        std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n * 1]);
        //std::cout << "Assigning...\n"; 
        index.assign(n, nx, coarse_idx.get(), 1);
        //std::cout << "Assigned! \n";

        std::vector<float> reordered_nx(n * index.d); // Temporary buffer for reordered nx
        //std::cout << "reordered! \n";
        // Track position within each cluster
        std::vector<size_t> cluster_counts(index.ntotal, 0);

        //std::cout << "n:" << n << "  index.ntotal:" << index.ntotal << std::endl;
        for (size_t i = 0; i < n; ++i) {
            //std::cout << "coarse_idx: " << coarse_idx[i] << std::endl;
            cluster_counts[coarse_idx[i]]++;
        }
        //std::cout << "cluster_counts! \n";
        std::vector<size_t> cluster_offsets(index.ntotal, 0);
        size_t cumulative_offset = 0;
        for (size_t cluster_id = 0; cluster_id < index.ntotal; ++cluster_id) {
            cluster_offsets[cluster_id] = cumulative_offset;
            cumulative_offset += cluster_counts[cluster_id];
        }

        //std::cout << "Clusters assigned! \n";

        for (size_t i = 0; i < n; ++i) {
            //std::cout << "Deal " << i << " Vector\n";
            size_t cluster_id = coarse_idx[i];
            size_t new_pos = cluster_offsets[cluster_id]++; // Cumulative position within entire list
            invlists->updata_inlist_map(list_no, i, new_pos);
            std::memcpy(&reordered_nx[new_pos * index.d], &nx[i * index.d], index.d * sizeof(float));
        }
        //std::cout << "\n";
        // copy back
        std::memcpy(nx, reordered_nx.data(), n * index.d * sizeof(float));

    }

    std::vector<idx_t> sort_vectors_by_proximity(size_t n, idx_t* labels) {
        std::vector<bool> visited(n, false);      // 记录是否访问过
        std::vector<idx_t> new_order;            // 存储最终排序结果
        new_order.reserve(n);

        // 从第一个向量开始
        size_t current = 0;
        visited[current] = true;
        new_order.push_back(current);

        for (size_t step = 1; step < n; ++step) { 
            bool found_next = false;

            for (size_t j = 0; j < n; ++j) {
                size_t neighbor = labels[current * n + j];
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    new_order.push_back(neighbor);
                    current = neighbor; 
                    found_next = true;
                    break;
                }
            }

            if (!found_next) {
                for (size_t j = 0; j < n; ++j) {
                    if (!visited[j]) {
                        visited[j] = true;
                        new_order.push_back(j);
                        current = j; 
                        break;
                    }
                }
            }
        }

        return new_order;
    }

    std::vector<idx_t> in_list_centroid_reassign(Index& index){
        size_t total = index.ntotal;
        size_t d = index.d;
        std::vector<float> dis(total*total);
        std::vector<idx_t> ids(total*total);

        std::vector<float> base_vector(d*total);
        index.reconstruct_n(0, total, base_vector.data());
        index.search(total, base_vector.data(), total, dis.data(), ids.data());
        auto new_order = sort_vectors_by_proximity(total, ids.data());


        // std::vector<float> base_vector(d);
        // index.reconstruct(0, base_vector.data());

        // index.search(1,base_vector.data(),total,dis.data(),ids.data());

        std::vector<float> reordered_centroids(total * d);
        for (size_t i = 0; i < total; ++i) {
            size_t original_index = new_order[i];
            index.reconstruct(original_index, &reordered_centroids[i * d]);
        }
        index.reset();
        index.add(total, reordered_centroids.data());
        return new_order;
    }

    std::vector<faiss::idx_t> in_list_centroid_reassign2(faiss::IndexFlatL2& assigner, size_t num_clusters) {
        size_t total = assigner.ntotal;
        size_t d = assigner.d;

        // Step 1: Retrieve all centroids
        std::vector<float> centroids(num_clusters * d);
        for (size_t i = 0; i < num_clusters; ++i) {
            assigner.reconstruct(i, &centroids[i * d]);
        }

        // Step 2: Assign each vector to its nearest centroid
        std::vector<faiss::idx_t> cluster_assignments(total);
        std::vector<float> distances(total);
        assigner.search(total, centroids.data(), 1, distances.data(), cluster_assignments.data());

        // Step 3: Sort vectors within each cluster by distance to centroid
        std::vector<faiss::idx_t> new_order(total);
        std::iota(new_order.begin(), new_order.end(), 0);

        std::sort(new_order.begin(), new_order.end(), [&](faiss::idx_t a, faiss::idx_t b) {
            return cluster_assignments[a] < cluster_assignments[b];
        });

        // Step 4: Rebuild the index in the new order
        std::vector<float> reordered_vectors(total * d);
        for (size_t i = 0; i < total; ++i) {
            assigner.reconstruct(new_order[i], &reordered_vectors[i * d]);
        }

        assigner.reset();
        assigner.add(total, reordered_vectors.data());

        return new_order;
    }

    std::vector<size_t> generate_inverted_map(const std::vector<idx_t>& map, size_t nlist) {
        std::vector<size_t> inverted_map(nlist);
        for (size_t i = 0; i < nlist; ++i) {
            inverted_map[map[i]] = i;
        }
        return inverted_map;
    }
}

// The members of this class is redundent
// So, there is no need to write it in disk
// Just make it when needed( read_index and make one)
template<typename ValueType>
struct IVF_DiskIOBuildProcessor : DiskIOProcessor{

    size_t ntotal;

    //size_t add_batch_num = 10;    // add a big file in batchs
    //size_t actual_batch_num = 0;  // temperory varible..... delete it later

    IVF_DiskIOBuildProcessor(std::string disk_path, 
                             size_t d,
                             size_t ntotal): DiskIOProcessor(disk_path, d), ntotal(ntotal){    
    }

    std::vector<ValueType> convert_to_ValueType(const std::vector<float>& data) {
        std::cout << "Convert to ValueType." << std::endl;
        std::vector<ValueType> converted(data.size());
        std::transform(data.begin(), data.end(), converted.begin(), [](float value) {
            return static_cast<ValueType>(value);
        });
        std::cout << "Convert to ValueType." << std::endl;
        return converted;
    }

    void convert_to_float(size_t n, float* vectors, void* disk_data) override {
        //std::cout << " Convert:" << std::endl;
        ValueType* original_vectors = reinterpret_cast<ValueType*>(disk_data);
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < d; j++)
            {
                vectors[i*d+j] = static_cast<float>(original_vectors[i*d+j]);
            }
        }
    }
    /* TODO                     
            // 写入 posting list 的详细信息：
            // 1. Page number（起始页号）
            // 2. Page offset（页面中的偏移量）
            // 3. Number of vectors（该 posting list 中的向量数量）
            // 4. Page count（该 posting list 所占用的页数）
    */

    bool align_cluster_page(size_t* clusters,
                        size_t* len,
                        Aligned_Cluster_Info* acInfo,
                        size_t nlist) override {
        
        size_t cumulative_pages = 0;   // 累计页数，用于记录每个聚类的起始页号

        for (size_t i = 0; i < nlist; ++i) {
            size_t total_bytes = len[i] * d * sizeof(ValueType);
            size_t page_count = (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE;

            size_t aligned_bytes = page_count * PAGE_SIZE;

            size_t padding_offset = aligned_bytes - total_bytes;

            acInfo[i].page_start = cumulative_pages;
            acInfo[i].page_count = page_count;
            acInfo[i].padding_offset = padding_offset;

            cumulative_pages += page_count;
        }

        this->total_page = cumulative_pages;

        std::cout << "Page alignment completed!" << std::endl;
        return true;
    }

    bool reorganize_vectors(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            std::vector<std::vector<faiss::idx_t>> ids) override {
        idx_t old_total = this->ntotal - n;
        std::cout << "old_total = "  << old_total << "  ntotal = " << this->ntotal << " n = " << n << std::endl;

        if (old_clusters == nullptr && old_len == nullptr) {
            align_cluster_page(clusters, len, acInfo, nlist);
            std::string clustered_disk_path = disk_path + ".clustered";

            std::ofstream disk_data_write(clustered_disk_path, std::ios::binary);
            if (!disk_data_write.is_open()) {
                throw std::runtime_error("Failed to open clustered disk file for writing.");
            }
            for (size_t i = 0; i < nlist; ++i) {
                size_t count = len[i];
                for (size_t j = 0; j < count; ++j) {
                    idx_t id = ids[i][j];
                    const float* vector = &data[id * d];
                    std::vector<ValueType> converted_vector(d);
                    for (size_t k = 0; k < d; ++k) {
                        converted_vector[k] = static_cast<ValueType>(vector[k]); 
                    }
                    disk_data_write.write(reinterpret_cast<const char*>(converted_vector.data()), d * sizeof(ValueType));
                }
                size_t padding = acInfo[i].padding_offset / sizeof(ValueType);
                if (padding > 0) {
                    std::vector<ValueType> padding_data(padding, 0);
                    disk_data_write.write(reinterpret_cast<const char*>(padding_data.data()), padding * sizeof(ValueType));
                }
            }

            disk_data_write.close();
            std::cout << "Reorganize vectors: Initial write completed with page alignment." << std::endl;
            return true;
        } else {
            std::vector<Aligned_Cluster_Info> old_acInfo(nlist);
            for (size_t i = 0; i < nlist; ++i) {
                old_acInfo[i] = acInfo[i];
            }
            align_cluster_page(clusters, len, acInfo, nlist);

            std::string tmp_disk = disk_path + ".tmp";
            int file_result = std::rename(disk_path.c_str(), tmp_disk.c_str());
            if(file_result == 0)
                std::cout << "Successfully renamed to: " << tmp_disk << std::endl;
            else {
                std::cerr << "Failed to rename to: " << tmp_disk << std::endl;
                throw std::runtime_error("Failed to rename disk file.");
            }

            std::ifstream temp_disk_read(tmp_disk, std::ios::binary);
            if (!temp_disk_read.is_open()) {
                throw std::runtime_error("Failed to open temporary disk file for reading.");
            }

            std::ofstream disk_data_write(disk_path, std::ios::binary);
            if (!disk_data_write.is_open()) {
                throw std::runtime_error("Failed to open clustered disk file for writing.");
            }

            for (size_t i = 0; i < nlist; ++i) {
                size_t old_count = old_len[i];
                size_t new_count = len[i];
                size_t total_count = len[i];

                std::cout << " old_len = " << old_len[i] << "  len = " << len[i] << std::endl;

                size_t total_bytes = total_count * d * sizeof(ValueType);
                std::vector<ValueType> combined_data(total_count * d, 0);
                if (old_count > 0) {
                    size_t old_bytes = old_count * d * sizeof(ValueType);
                    size_t old_offset_bytes = old_acInfo[i].page_start * PAGE_SIZE;
                    temp_disk_read.seekg(old_offset_bytes, std::ios::beg);
                    temp_disk_read.read(reinterpret_cast<char*>(combined_data.data()), old_bytes);
                    if (temp_disk_read.gcount() != static_cast<std::streamsize>(old_bytes)) {
                        throw std::runtime_error("Failed to read complete old cluster data.");
                    }
                }

                for (size_t j = old_count; j < new_count ; ++j) {
                    idx_t id = ids[i][j];
                    const float* vector = &data[(id-old_total) * d];  // new 
                    for (size_t k = 0; k < d; ++k) {
                        combined_data[j * d + k] = static_cast<ValueType>(vector[k]);
                    }
                }

                disk_data_write.write(reinterpret_cast<const char*>(combined_data.data()), total_count * d * sizeof(ValueType));

                if(verbose)
                {   
                    size_t written_pages = (total_count * d * sizeof(ValueType) + PAGE_SIZE - 1) / PAGE_SIZE;
                    size_t expected_pages = acInfo[i].page_count;
                }

                size_t new_padding = acInfo[i].padding_offset;
                if (new_padding > 0) {
                    std::vector<ValueType> padding_data(new_padding / sizeof(ValueType), 0);
                    disk_data_write.write(reinterpret_cast<const char*>(padding_data.data()), new_padding);
                }
            }
            disk_data_write.close();
            temp_disk_read.close();

            std::remove(tmp_disk.c_str());

            std::cout << "Reorganize vectors: Merged old and new data with page alignment." << std::endl;
            return false;
        }
    }

    bool reorganize_vectors_2(idx_t n, 
                          const float* data, 
                          size_t* old_clusters, 
                          size_t* old_len,
                          size_t* clusters,
                          size_t* len,
                          Aligned_Cluster_Info* acInfo,
                          size_t nlist,
                          ClusteredArrayInvertedLists* c_array_invlists,
                          bool do_in_list_cluster) override 
{
    idx_t old_total = this->ntotal - n;
    //case 1
    if (old_clusters == nullptr && old_len == nullptr) {

        align_cluster_page(clusters, len, acInfo, nlist);
        std::string clustered_disk_path = disk_path + ".clustered";

        std::ofstream disk_data_write(clustered_disk_path, std::ios::binary);
        if (!disk_data_write.is_open()) {
            throw std::runtime_error("Failed to open clustered disk file for writing.");
        }

        // We will first build per-list data in parallel (vectors, possibly re-cluster),
        // then do a single pass of writes in correct order.

        // Step A: Prepare buffers in parallel
        std::vector< std::vector<ValueType> > all_list_buffers(nlist);  
        // We also need to store how many ValueType elements each list has, for writing.
        std::vector<size_t> list_buffer_sizes(nlist, 0);

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nlist; ++i) {
            size_t count = len[i];
            // Build a float buffer for clustering
            std::vector<float> vectors(count * d);
            for (size_t j = 0; j < count; ++j) {
                idx_t id = c_array_invlists->ids[i][j];
                std::copy(&data[id * d],
                          &data[(id + 1) * d],
                          vectors.data() + j * d);
            }
            //in_list_clustering
            if (do_in_list_cluster) {
                size_t in_list_cluster_num = count / 32;
                if (in_list_cluster_num > 1) {
                    Clustering clus(d, in_list_cluster_num);
                    IndexFlatL2 assigner(d);
                    clus.train(count, vectors.data(), assigner);
                    in_list_centroid_reassign2(assigner, in_list_cluster_num);
                    in_list_map_reassign(count, vectors.data(), assigner, c_array_invlists, i);
                    in_list_pq_ids_reassign(c_array_invlists, i);
                } else {
                    // If too few vectors, just keep them in the same order.
                    for (size_t m = 0; m < count; m++) {
                        c_array_invlists->updata_inlist_map(i, m, m);
                    }
                }
            }

            // Convert floats -> ValueType
            auto converted_vectors = convert_to_ValueType(vectors);

            // Prepare final buffer for this list:
            //   (vectors + any padding)
            size_t padding_vals = acInfo[i].padding_offset / sizeof(ValueType);
            size_t total_vals   = converted_vectors.size() + padding_vals;

            all_list_buffers[i].resize(total_vals, 0);
            // copy the real vector data
            std::memcpy(all_list_buffers[i].data(),
                        converted_vectors.data(),
                        converted_vectors.size() * sizeof(ValueType));
            list_buffer_sizes[i] = total_vals; 
        } // end parallel for

        // Step B: Single-threaded writing in correct order
        for (size_t i = 0; i < nlist; ++i) {
            // Write out the entire buffer for list i
            if (!all_list_buffers[i].empty()) {
                disk_data_write.write(
                    reinterpret_cast<const char*>(all_list_buffers[i].data()),
                    list_buffer_sizes[i] * sizeof(ValueType));
            }
        }

        disk_data_write.close();
        std::cout << "Reorganize vectors: Initial write completed with page alignment." << std::endl;
        return true;
    }

    // Case 2: Merge old and new data
    else {
        std::vector<Aligned_Cluster_Info> old_acInfo(nlist);
        for (size_t i = 0; i < nlist; ++i) {
            old_acInfo[i] = acInfo[i];
        }
        align_cluster_page(clusters, len, acInfo, nlist);

        std::string tmp_disk = disk_path + ".tmp";
        int file_result = std::rename(disk_path.c_str(), tmp_disk.c_str());
        if (file_result == 0) {
            std::cout << "Successfully renamed to: " << tmp_disk << std::endl;
        } else {
            std::cerr << "Failed to rename to: " << tmp_disk << std::endl;
            throw std::runtime_error("Failed to rename disk file.");
        }

        std::ifstream temp_disk_read(tmp_disk, std::ios::binary);
        if (!temp_disk_read.is_open()) {
            throw std::runtime_error("Failed to open temporary disk file for reading.");
        }
        std::ofstream disk_data_write(disk_path, std::ios::binary);
        if (!disk_data_write.is_open()) {
            throw std::runtime_error("Failed to open clustered disk file for writing.");
        }

        std::vector< std::vector<ValueType> > all_list_buffers(nlist);
        std::vector<size_t> list_buffer_sizes(nlist, 0);

        std::vector<size_t> old_offsets(nlist);
        for (size_t i = 0; i < nlist; ++i) {
            old_offsets[i] = old_acInfo[i].page_start * PAGE_SIZE;
        }

#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nlist; ++i) {
            size_t old_offset = old_offsets[i];
            size_t old_count  = old_len[i];
            size_t new_count  = len[i] - old_count;
            size_t count      = len[i];

            std::vector<ValueType> disk_vectors(old_count * d, 0);

            {
                // Each thread opens its own ifstream to avoid concurrency issues
                // with a shared stream.  This is safer for random-access reads.
                std::ifstream local_in(tmp_disk, std::ios::binary);
                if (!local_in.is_open()) {
                    throw std::runtime_error("Thread failed to open tmp file for reading.");
                }
                local_in.seekg(old_offset, std::ios::beg);
                local_in.read(reinterpret_cast<char*>(disk_vectors.data()),
                              old_count * d * sizeof(ValueType));
                local_in.close();
            }

            // Convert the old vectors to float for potential re-clustering
            std::vector<float> all_vectors(count * d, 0.0f);
            convert_to_float(old_count, all_vectors.data(), disk_vectors.data());

            // Append new vectors (from RAM)
            for (size_t j = old_count; j < count; ++j) {
                idx_t id = c_array_invlists->ids[i][j];
                // shift by old_total in data array
                std::copy(&data[(id - old_total) * d],
                          &data[(id - old_total + 1) * d],
                          all_vectors.data() + j * d);
            }

            // Optional in-list clustering
            if (do_in_list_cluster) {
                size_t in_list_cluster_num = count / 32;
                if (in_list_cluster_num > 1) {
                    Clustering clus(d, in_list_cluster_num);
                    IndexFlatL2 assigner(d);
                    clus.train(count, all_vectors.data(), assigner);
                    in_list_centroid_reassign2(assigner, in_list_cluster_num);
                    in_list_map_reassign(count, all_vectors.data(), assigner, c_array_invlists, i);
                    in_list_pq_ids_reassign(c_array_invlists, i);
                } else {
                    for (size_t m = 0; m < count; m++) {
                        c_array_invlists->updata_inlist_map(i, m, m);
                    }
                }
            }

            // Convert floats -> ValueType
            auto converted_vectors = convert_to_ValueType(all_vectors);

            // Build the final buffer for list i (data + padding)
            size_t new_padding = acInfo[i].padding_offset;
            size_t padding_vals = new_padding / sizeof(ValueType);

            size_t total_vals = converted_vectors.size() + padding_vals;
            all_list_buffers[i].resize(total_vals, 0);
            std::memcpy(all_list_buffers[i].data(),
                        converted_vectors.data(),
                        converted_vectors.size() * sizeof(ValueType));

            list_buffer_sizes[i] = total_vals;
        } // end parallel for

        // Single-threaded final writes
        for (size_t i = 0; i < nlist; ++i) {
            if (!all_list_buffers[i].empty()) {
                disk_data_write.write(
                    reinterpret_cast<const char*>(all_list_buffers[i].data()),
                    list_buffer_sizes[i] * sizeof(ValueType));
            }
        }

        disk_data_write.close();
        temp_disk_read.close();
        std::remove(tmp_disk.c_str());

        std::cout << "Reorganize vectors: Merged old and new data with page alignment." << std::endl;
        return false;
    }
}

    
    bool reorganize_vectors_in_memory(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            ClusteredArrayInvertedLists* c_array_invlists,
                            ArrayInvertedLists* build_invlists,
                            bool do_in_list_cluster,
                            bool keep_disk = false)override
    {
        idx_t old_total = this->ntotal - n;
        //case 1
        //if (old_clusters == nullptr && old_len == nullptr) 
        if(1)
        {    
            if(keep_disk){
                align_cluster_page(clusters, len, acInfo, nlist);
                std::string clustered_disk_path = disk_path + ".clustered";

                std::ofstream disk_data_write(clustered_disk_path, std::ios::binary);
                if (!disk_data_write.is_open()) {
                    throw std::runtime_error("Failed to open clustered disk file for writing.");
                }

                // We will first build per-list data in parallel (vectors, possibly re-cluster),
                // then do a single pass of writes in correct order.

                // Step A: Prepare buffers in parallel
                std::vector< std::vector<ValueType> > all_list_buffers(nlist);  
                // We also need to store how many ValueType elements each list has, for writing.
                std::vector<size_t> list_buffer_sizes(nlist, 0);

#pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < nlist; ++i) {
                    std::cout << "clustering " << i << "\n";
                    size_t count = len[i];
                    // Build a float buffer for clustering
                    std::vector<float> vectors(count * d);

                    const float* original_data = (const float*)build_invlists->get_codes(i);

                    std::copy(original_data, original_data + count*d, vectors.data());

                    // for (size_t j = 0; j < count; ++j) {
                    //     idx_t id = c_array_invlists->ids[i][j];
                    //     std::copy(&data[id * d],
                    //             &data[(id + 1) * d],
                    //             vectors.data() + j * d);
                    // }
                    //in_list_clustering
                    if (do_in_list_cluster) {
                        size_t in_list_cluster_num = count / 32;
                        if (in_list_cluster_num > 1) {
                            Clustering clus(d, in_list_cluster_num);
                            IndexFlatL2 assigner(d);
                            clus.train(count, vectors.data(), assigner);
                            in_list_centroid_reassign2(assigner, in_list_cluster_num);
                            in_list_map_reassign(count, vectors.data(), assigner, c_array_invlists, i);
                            in_list_pq_ids_reassign(c_array_invlists, i);
                        } else {
                            // If too few vectors, just keep them in the same order.
                            for (size_t m = 0; m < count; m++) {
                                c_array_invlists->updata_inlist_map(i, m, m);
                            }
                        }
                    }

                    // Convert floats -> ValueType
                    auto converted_vectors = convert_to_ValueType(vectors);
                    std::cout << "Converted to original format...\n";

                    // Prepare final buffer for this list:
                    //   (vectors + any padding)
                    size_t padding_vals = acInfo[i].padding_offset / sizeof(ValueType);
                    size_t total_vals   = converted_vectors.size() + padding_vals;

                    all_list_buffers[i].resize(total_vals, 0);
                    // copy the real vector data
                    std::memcpy(all_list_buffers[i].data(),
                                converted_vectors.data(),
                                converted_vectors.size() * sizeof(ValueType));
                    list_buffer_sizes[i] = total_vals; 
                } // end parallel for

                // Step B: Single-threaded writing in correct order
                for (size_t i = 0; i < nlist; ++i) {
                    // Write out the entire buffer for list i
                    if (!all_list_buffers[i].empty()) {
                        disk_data_write.write(
                            reinterpret_cast<const char*>(all_list_buffers[i].data()),
                            list_buffer_sizes[i] * sizeof(ValueType));
                    }
                }

                disk_data_write.close();
                std::cout << "Stored in disk!" << std::endl;
                return true;
            }
            std::cout << "Skip mid clusters" << std::endl;
            return false;
        }
        else{
    //         std::vector<Aligned_Cluster_Info> old_acInfo(nlist);
    //         for (size_t i = 0; i < nlist; ++i) {
    //             old_acInfo[i] = acInfo[i];
    //         }
    //         align_cluster_page(clusters, len, acInfo, nlist);

    //         std::string tmp_disk = disk_path + ".tmp";
    //         int file_result = std::rename(disk_path.c_str(), tmp_disk.c_str());
    //         if (file_result == 0) {
    //             std::cout << "Successfully renamed to: " << tmp_disk << std::endl;
    //         } else {
    //             std::cerr << "Failed to rename to: " << tmp_disk << std::endl;
    //             throw std::runtime_error("Failed to rename disk file.");
    //         }

    //         std::ifstream temp_disk_read(tmp_disk, std::ios::binary);
    //         if (!temp_disk_read.is_open()) {
    //             throw std::runtime_error("Failed to open temporary disk file for reading.");
    //         }
    //         std::ofstream disk_data_write(disk_path, std::ios::binary);
    //         if (!disk_data_write.is_open()) {
    //             throw std::runtime_error("Failed to open clustered disk file for writing.");
    //         }

    //         std::vector< std::vector<ValueType> > all_list_buffers(nlist);
    //         std::vector<size_t> list_buffer_sizes(nlist, 0);

    //         std::vector<size_t> old_offsets(nlist);
    //         for (size_t i = 0; i < nlist; ++i) {
    //             old_offsets[i] = old_acInfo[i].page_start * PAGE_SIZE;
    //         }

    // #pragma omp parallel for schedule(dynamic)
    //         for (size_t i = 0; i < nlist; ++i) {
    //             size_t old_offset = old_offsets[i];
    //             size_t old_count  = old_len[i];
    //             size_t new_count  = len[i] - old_count;
    //             size_t count      = len[i];

    //             std::vector<ValueType> disk_vectors(old_count * d, 0);

    //             {
    //                 // Each thread opens its own ifstream to avoid concurrency issues
    //                 // with a shared stream.  This is safer for random-access reads.
    //                 std::ifstream local_in(tmp_disk, std::ios::binary);
    //                 if (!local_in.is_open()) {
    //                     throw std::runtime_error("Thread failed to open tmp file for reading.");
    //                 }
    //                 local_in.seekg(old_offset, std::ios::beg);
    //                 local_in.read(reinterpret_cast<char*>(disk_vectors.data()),
    //                             old_count * d * sizeof(ValueType));
    //                 local_in.close();
    //             }

    //             // Convert the old vectors to float for potential re-clustering
    //             std::vector<float> all_vectors(count * d, 0.0f);
    //             convert_to_float(old_count, all_vectors.data(), disk_vectors.data());

    //             // Append new vectors (from RAM)
    //             for (size_t j = old_count; j < count; ++j) {
    //                 idx_t id = c_array_invlists->ids[i][j];
    //                 // shift by old_total in data array
    //                 std::copy(&data[(id - old_total) * d],
    //                         &data[(id - old_total + 1) * d],
    //                         all_vectors.data() + j * d);
    //             }

    //             // Optional in-list clustering
    //             if (do_in_list_cluster) {
    //                 size_t in_list_cluster_num = count / 32;
    //                 if (in_list_cluster_num > 1) {
    //                     Clustering clus(d, in_list_cluster_num);
    //                     IndexFlatL2 assigner(d);
    //                     clus.train(count, all_vectors.data(), assigner);
    //                     in_list_centroid_reassign2(assigner, in_list_cluster_num);
    //                     in_list_map_reassign(count, all_vectors.data(), assigner, c_array_invlists, i);
    //                     in_list_pq_ids_reassign(c_array_invlists, i);
    //                 } else {
    //                     for (size_t m = 0; m < count; m++) {
    //                         c_array_invlists->updata_inlist_map(i, m, m);
    //                     }
    //                 }
    //             }

    //             // Convert floats -> ValueType
    //             auto converted_vectors = convert_to_ValueType(all_vectors);

    //             // Build the final buffer for list i (data + padding)
    //             size_t new_padding = acInfo[i].padding_offset;
    //             size_t padding_vals = new_padding / sizeof(ValueType);

    //             size_t total_vals = converted_vectors.size() + padding_vals;
    //             all_list_buffers[i].resize(total_vals, 0);
    //             std::memcpy(all_list_buffers[i].data(),
    //                         converted_vectors.data(),
    //                         converted_vectors.size() * sizeof(ValueType));

    //             list_buffer_sizes[i] = total_vals;
    //         } // end parallel for

    //         // Single-threaded final writes
    //         for (size_t i = 0; i < nlist; ++i) {
    //             if (!all_list_buffers[i].empty()) {
    //                 disk_data_write.write(
    //                     reinterpret_cast<const char*>(all_list_buffers[i].data()),
    //                     list_buffer_sizes[i] * sizeof(ValueType));
    //             }
    //         }

    //         disk_data_write.close();
    //         temp_disk_read.close();
    //         std::remove(tmp_disk.c_str());

    //         std::cout << "Reorganize vectors: Merged old and new data with page alignment." << std::endl;
            return false;


        }
        
    }


    // if it is true, then add .cluster to the disk_path in the index.
    bool okd_reorganize_vectors(idx_t n, 
                            const float* data, 
                            size_t* old_clusters, 
                            size_t* old_len,
                            size_t* clusters,
                            size_t* len,
                            Aligned_Cluster_Info* acInfo,
                            size_t nlist,
                            std::vector<std::vector<faiss::idx_t>> ids) {
        idx_t old_total = this->ntotal - n;

        if (old_clusters == nullptr && old_len == nullptr) {
            // Reorganize vectors and write to the new file
            // disk_path should be changed in outer
            disk_path = disk_path + ".clustered";

            //std::cout << "disk_path_clustered: " << disk_path_clustered << std::endl;
            //std::cout << "disk_path          : " << disk_path << std::endl;
            //set_disk_write(disk_path);

            std::ofstream disk_data_write(disk_path, std::ios::binary);
            if (!disk_data_write.is_open()) {
                throw std::runtime_error("Failed to open temporary disk file for reading.");
            }
            for (size_t i = 0; i < nlist; ++i) {
                size_t count = len[i];

                for (size_t j = 0; j < count; ++j) {
                    idx_t id = ids[i][j];
                    const float* vector = &data[id * d];
                    std::vector<ValueType> converted_vector(d);
                    for (size_t k = 0; k < d; ++k) {
                        // TODO How to convert???
                        converted_vector[k] = static_cast<ValueType>(vector[k]); 
                    }
                    disk_data_write.write(reinterpret_cast<const char*>(converted_vector.data()), d * sizeof(ValueType));
                    //disk_data_write.write(reinterpret_cast<const char*>(vector), d * sizeof(float));
                }
            }
            disk_data_write.close();
            
            return true;
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
            // set_disk_write(disk_path);
            std::ofstream disk_data_write(disk_path, std::ios::binary);
            if (!disk_data_write.is_open()) {
                throw std::runtime_error("Failed to open temporary disk file for reading.");
            }
            // 4. Write old data and new data to the file
            for (size_t i = 0; i < nlist; ++i) {
                size_t old_offset = old_clusters[i];
                size_t old_count = old_len[i];

                // Read old cluster data
                std::vector<ValueType> old_cluster(old_count * d);
                temp_disk_read.seekg(old_offset * d * sizeof(ValueType), std::ios::beg);
                temp_disk_read.read(reinterpret_cast<char*>(old_cluster.data()), old_count * d * sizeof(ValueType));
                disk_data_write.write(reinterpret_cast<const char*>(old_cluster.data()), old_count * d * sizeof(ValueType));

                // Write new data
                size_t count = len[i];
                for (size_t j = old_count; j < count; ++j) {
                    idx_t id = ids[i][j];
                    const float* vector = &data[(id - old_total) * d];
                    std::vector<ValueType> converted_vector(d);
                    for (size_t k = 0; k < d; ++k) {
                        // TODO How to convert???
                        converted_vector[k] = static_cast<ValueType>(vector[k]); 
                    }
                    disk_data_write.write(reinterpret_cast<const char*>(converted_vector.data()), d * sizeof(ValueType));
                }
            }
            disk_data_write.close();
            temp_disk_read.close();

            // 5. Delete the temporary file
            std::remove(tmp_disk.c_str());
            return false;
        }
    }
    
    void reorganize_list_disk(
        Aligned_Cluster_Info* acInfo,
        std::vector<idx_t>& map,
        size_t nlist){
        
        std::cout << "reorganize_list_disk." << std::endl;
        // 原始聚类文件
        std::string temp_file = disk_path + ".temp";
        if (std::rename(disk_path.c_str(), temp_file.c_str()) != 0) {
            throw std::runtime_error("Failed to rename the source file to a temporary file.");
        }
        std::ifstream source(temp_file, std::ios::binary);
        if (!source.is_open()) {
            throw std::runtime_error("Failed to open the temporary file for reading.");
        }
        //更新后的聚类文件
        std::ofstream destination(disk_path, std::ios::binary);
        if (!destination.is_open()) {
            throw std::runtime_error("Failed to open the new file for writing.");
        }

        //std::vector<size_t> inverted_map = generate_inverted_map(map, nlist);
        
        for (size_t i = 0; i < nlist; i++) {
            size_t o = map[i];
            FAISS_THROW_IF_NOT(o < nlist);
            size_t offset = acInfo[o].page_start * PAGE_SIZE;
            size_t data_size = acInfo[o].page_count * PAGE_SIZE;
            
            source.seekg(offset, std::ios::beg);
            std::vector<char> buffer(data_size);
            source.read(buffer.data(), data_size);
            if (source.gcount() != data_size) {
                throw std::runtime_error("Failed to read data for cluster from the temporary file.");
            }

            destination.write(buffer.data(), data_size);
            if (!destination) {
                throw std::runtime_error("Failed to write data to the new file.");
            }
        }

        source.close();
        destination.close();

        if (std::remove(temp_file.c_str()) != 0) {
            throw std::runtime_error("Failed to delete the temporary file.");
        }
    }

    size_t reorganize_list(
        Index& quantizer, 
        ClusteredArrayInvertedLists* c_array_invlists, 
        Aligned_Cluster_Info* acInfo,
        size_t* clusters,
        size_t* len,
        size_t nlist) override{
        std::cout << "permuting invlists" << std::endl;

        std::vector<idx_t> list_reorg_map = in_list_centroid_reassign2(reinterpret_cast<IndexFlatL2&>(quantizer), nlist);
        // 更改磁盘的存储顺序
        reorganize_list_disk(acInfo, list_reorg_map, nlist);

        c_array_invlists->permute_invlists(list_reorg_map.data());

        std::cout << "updating info" << std::endl;
        size_t current_offset = 0;
        for (size_t i = 0; i < nlist; ++i) {
            clusters[i] = current_offset;
            len[i] = c_array_invlists->ids[i].size();
            current_offset += len[i];
        }
        align_cluster_page(clusters, len, acInfo, nlist);

        return true;         
    }


    bool align_list_page(size_t entry_size,   // clustered: sizeof(idx) + sizeof(pq_size)
                        size_t* len,
                        Aligned_Invlist_Info* invInfo,
                        size_t nlist) {
        
        //TODO 累计页数，用于记录每个聚类的起始页号。
        //TODO 有两种情况，第一种是新开一个文件，cumulative = 0 （文件打开模式改成trunc）
        //TODO 第二种情况是记录在向量文件后面，cumulative = total_page
        //size_t cumulative_pages = 0;   
        size_t cumulative_pages = this->total_page;   

        for (size_t i = 0; i < nlist; ++i) {
            size_t total_bytes = len[i] * entry_size;
            size_t page_count = (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE;

            size_t aligned_bytes = page_count * PAGE_SIZE;

            size_t padding_offset = aligned_bytes - total_bytes;

            invInfo[i].page_start = cumulative_pages;
            invInfo[i].page_count = page_count;
            invInfo[i].padding_offset = padding_offset;
            invInfo[i].list_size = len[i];

            cumulative_pages += page_count;
        }

        std::cout << "Inverted list pages alignment completed!" << std::endl;
        return true;
    }


    bool organize_select_list(
        size_t pq_size,
        size_t entry_size,
        ClusteredArrayInvertedLists* c_array_invlists, 
        Aligned_Invlist_Info* invInfo, 
        size_t nlist,
        std::string select_lists_path)override
    {
        std::vector<size_t> list_size(nlist);

        for(int i = 0; i < nlist; i++){
            list_size[i] = c_array_invlists->list_size(i);
            invInfo[i].list_size = list_size[i];
        }

        align_list_page(entry_size, list_size.data(), invInfo, nlist);

        std::ofstream out_file(select_lists_path, std::ios::binary | std::ios::app);  // 如果写在一个文件里面
        //std::ofstream out_file(select_lists_path, std::ios::binary | std::ios::trunc); // 如果分开写


        if (!out_file.is_open()) {
            std::cerr << "Failed to open file: " << select_lists_path << std::endl;
            return false;
        }

        std::cout << "pq_size:" << pq_size << "  idx_t:" << sizeof(idx_t) << "  size_t:" << sizeof(size_t) <<"\n";

        for (size_t i = 0; i < nlist; ++i) {
            size_t ids_size = list_size[i] * sizeof(idx_t);
            size_t codes_size = list_size[i] * pq_size;
            //size_t map_size = list_size[i] * sizeof(size_t);

            const idx_t* ids = c_array_invlists->get_ids(i);
            const uint8_t* codes = c_array_invlists->get_codes(i);
            //const size_t* inlist_map = c_array_invlists->get_inlist_map(i);

            out_file.write(reinterpret_cast<const char*>(ids), ids_size);
            out_file.write(reinterpret_cast<const char*>(codes), codes_size);
            //out_file.write(reinterpret_cast<const char*>(inlist_map), map_size);

            // 填充对齐的padding
            size_t padding_bytes = invInfo[i].padding_offset;
            if (padding_bytes > 0) {
                std::vector<char> padding(padding_bytes, 0);
                out_file.write(padding.data(), padding_bytes);
            }
        }

        out_file.close();
        std::cout << "Data written successfully to " << select_lists_path << std::endl;
        return true;
    }
};

// sync form
template<typename ValueType>
struct IVF_DiskIOSearchProcessor : DiskIOProcessor{

    FILE* file_ptr;

    IVF_DiskIOSearchProcessor(std::string disk_path, 
                             size_t d): DiskIOProcessor(disk_path, d), file_ptr(nullptr){
        file_ptr = fopen(disk_path.c_str(), "rb");
        if (!file_ptr) {
            throw std::runtime_error("Failed to open disk file for reading.");
        }
    }

    ~IVF_DiskIOSearchProcessor() {
        if (file_ptr) {
            fclose(file_ptr);
            std::cout << "File closed successfully." << std::endl;
        }
    }
    
    void disk_io_all(int D,
                    size_t len,
                    size_t listno,
                    float* vectors,
                    Aligned_Cluster_Info* acInfo) override {
        
        if (!file_ptr) {
            std::cerr << "File is not open for reading!" << std::endl;
            return;
        }
        // assert(listno<??);
        // 获取聚类的信息
        size_t page_start = acInfo[listno].page_start;
        size_t padding_offset = acInfo[listno].padding_offset;
        size_t page_count = acInfo[listno].page_count;

        // 计算磁盘文件中该list的起始位置（字节偏移量）
        size_t offset = page_start * PAGE_SIZE;

        // 计算需要读取的总字节数，去除填充字节数
        size_t total_bytes = page_count * PAGE_SIZE - padding_offset;

        if (fseek(file_ptr, offset, SEEK_SET) != 0) {
            std::cerr << "Failed to seek to the required position in the file." << std::endl;
            return;
        }

        // 使用临时缓冲区存储读取的数据
        std::vector<ValueType> buffer(len * D);

        // 使用 fread 读取数据
        size_t read_count = fread(buffer.data(), sizeof(ValueType), total_bytes / sizeof(ValueType), file_ptr);
        if (read_count != total_bytes / sizeof(ValueType)) {
            std::cerr << "Failed to read the expected number of bytes from disk!" << std::endl;
            return;
        }

        // 将读取的数据转换为 float 类型并存储到 vectors 中
        for (size_t i = 0; i < len; ++i) {
            for (size_t j = 0; j < D; ++j) {
                vectors[i * D + j] = static_cast<float>(buffer[i * D + j]);
            }
        }


    }

    void disk_io_single(int D,
                        size_t len,
                        size_t listno,
                        size_t nth,
                        float* vector,
                        Aligned_Cluster_Info* acInfo) override {
        if (!file_ptr) {
            std::cerr << "File is not open for reading!" << std::endl;
            return;
        }
        // 获取聚类的起始页位置和偏移量
        size_t page_start = acInfo[listno].page_start;
        size_t padding_offset = acInfo[listno].padding_offset;

        // 计算目标向量在磁盘中的偏移量：聚类起始位置 + nth个向量的位置
        size_t offset = page_start * PAGE_SIZE + nth * D * sizeof(ValueType);

        // 定位到目标向量的位置
        if (fseek(file_ptr, offset, SEEK_SET) != 0) {
            std::cerr << "Failed to seek to the required position in the file." << std::endl;
            return;
        }

        // 读取单个向量的数据（d维度）
        std::vector<ValueType> buffer(D);
        size_t read_count = fread(buffer.data(), sizeof(ValueType), D, file_ptr);
        if (read_count != D) {
            std::cerr << "Failed to read the expected number of bytes for the vector!" << std::endl;
            return;
        }

        // 将读取的数据转换为 float 类型，并存储到 vector 中
        for (size_t i = 0; i < D; ++i) {
            vector[i] = static_cast<float>(buffer[i]);
        }

        // std::cout << "Read single vector successfully from cluster " << listno 
        //           << " at position " << nth << "." << std::endl;                 
    }

};

// Async form
template<typename ValueType>
struct IVF_DiskIOSearchProcessor_Async : DiskIOProcessor{

    size_t list_to_search;   // 这个的作用？？？
    size_t factor_full;
    size_t factor_partial;
    std::vector<std::shared_ptr<AsyncReadRequest>> full_diskRequests;
    std::vector<std::shared_ptr<std::vector<AsyncReadRequest_Partial>>> partial_diskRequests;  // different
    std::vector<PageBuffer<uint8_t>> page_buffers;

    //std::vector<std::vector<float>> converted_data;  // store converted result, we need nlist 


    // 文件描述符
    int fd;

    // AIO context
    aio_context_t aio_ctx;


    IVF_DiskIOSearchProcessor_Async(std::string disk_path, 
                             size_t d): DiskIOProcessor(disk_path, d), aio_ctx(0){
        // TODO 暂时full_diskRequests 是全部的，后面再partial
        // 打开文件时需要加上o_direct
        //int maxIOSize = 128; // 设定同时处理的最大异步 I/O 事件数
        //if (io_setup(max_events, &aio_ctx) < 0) {}
       
    }

    ~IVF_DiskIOSearchProcessor_Async() {
        if (syscall(__NR_io_destroy, aio_ctx) < 0) {
            std::cerr << "Failed to destroy AIO context." << std::endl;
        }

        if (fd >= 0) {
            close(fd);
            std::cout << "AIO context destroyed and file closed successfully." << std::endl;
        }
    }

    void initial(std::uint64_t maxIOSize = (1 << 20),
                 std::uint32_t maxReadRetries = 2,
                 std::uint32_t maxWriteRetries = 2,
                 std::uint16_t threadPoolSize = 4) override {
        maxIOSize = 1<<10;
        // O_DIRECT
        fd = open(disk_path.c_str(), O_RDONLY | O_DIRECT);   // Must align
        if (fd < 0) {
            perror("Failed to open file with O_DIRECT");
            throw std::runtime_error("Failed to open file");
        }
        
        auto ret = syscall(__NR_io_setup, (int)maxIOSize, &aio_ctx);
        if(ret < 0){
            perror("io_setup failed");
            throw std::runtime_error("Failed to initialize AIO context");
        }
        std::cout << "AIO context initialized and file opened with O_DIRECT: " << disk_path << std::endl;
    }


    void convert_to_float(size_t n, float* vectors, void* disk_data) override {
        //std::cout << " Convert:" << std::endl;
        ValueType* original_vectors = reinterpret_cast<ValueType*>(disk_data);
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < d; j++)
            {
                //std::cout << "i" << i << "  j " << j << std::endl;
                //std::cout <<  static_cast<float>(original_vectors[i*d+j]) << std::endl;
                vectors[i*d+j] = static_cast<float>(original_vectors[i*d+j]);
            }
        }
    }

    void disk_io_all_async(std::shared_ptr<AsyncReadRequest>& asyncReadRequest) override {

        // 一个query提交一次，然后要processor要进行reset
        //第一步：设置好第i个nlist的各种信息
        //第二步：full_diskRequests.push();
        // TODO ERROR!!!!!!!!!!!!!!!!!   invalid asycReadRequest
        
        PageBuffer<uint8_t> page_buffer;
        page_buffer.ReservePageBuffer(asyncReadRequest->m_readSize); // 设置 buffer 大小
        page_buffers.emplace_back(std::move(page_buffer)); // 直接构造并移动到 vector 中
        full_diskRequests.push_back(asyncReadRequest);
        // std::vector<float> tmp_converted_data;
        // tmp_converted_data.resize(d * asyncReadRequest->len);
        // converted_data.push_back(tmp_converted_data);
        

        //std::cout <<"SIZE: " << full_diskRequests.size() << std::endl;
        //converted_data.resize(d * asyncReadRequest->len);
    }

    void disk_io_partial_async(std::shared_ptr<std::vector<AsyncReadRequest_Partial>>& asyncReadRequests_p ){
        AsyncReadRequest_Partial* ptr_asyncReadRequests = asyncReadRequests_p->data();
        int r_size = asyncReadRequests_p->size();

        //std::cout << "r_size:" << r_size << std::endl;
        for(int i = 0; i < r_size; i++){
            PageBuffer<uint8_t> page_buffer;
            page_buffer.ReservePageBuffer(ptr_asyncReadRequests[i].m_readSize); // 设置 buffer 大小
            page_buffers.emplace_back(std::move(page_buffer)); // 直接构造并移动到 vector 中
        }
        //std::cout << "Partial push back." << std::endl;
        // TODO 这里没有push
        partial_diskRequests.push_back(asyncReadRequests_p);


    };

    struct timespec AIOTimeout {0, 30000};
    void submit_fully(int num){
        // TODO async but only 1 file

        // int handler_size = handler.size();
        const int handler_size = 1;
        const int handler_num = handler_size - 1;
        //AsyncReadRequest* readRequests = full_diskRequests.data().get();
        num = full_diskRequests.size();
        std::vector<struct iocb> myiocbs(num);
        std::vector<std::vector<struct iocb*>> iocbs(handler_size);   
        std::vector<int> submitted(handler_size, 0);
        std::vector<int> done(handler_size, 0);

        //std::vector<float>test(10*128, 9);
        
        int totalToSubmit = 0;
        //std::cout <<num <<  " num\n";
        memset(myiocbs.data(), 0, num * sizeof(struct iocb));
        for (int i = 0; i < num; i++) {
            AsyncReadRequest* readRequest = full_diskRequests[i].get();
            //std::cout << "read_size:" << readRequest->m_readSize << "  read_offset:" << readRequest->m_offset << std::endl;
            //std::cout <<i <<  " Processing necessary info stage 1\n";
            //channel = readRequest->m_status & 0xffff;
            //int fileid = (readRequest->m_status >> 16);
            struct iocb* myiocb = &(myiocbs[totalToSubmit++]);
            myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
            
            myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
            //myiocb->aio_fildes = ((AsyncFileIO*)(handlers[fileid].get()))->GetFileHandler();
            myiocb->aio_fildes = this->fd;
            // myiocb->aio_buf = (std::uint64_t)(readRequest->m_buffer);
            //std::cout <<i <<  " Processing necessary info stage 2\n";
            myiocb->aio_nbytes = readRequest->m_readSize;
            //std::cout <<i <<  " Processing necessary info stage 3\n";
            myiocb->aio_buf = (std::uint64_t)(page_buffers[i].GetBuffer());
            readRequest->m_buffer = (char*)page_buffers[i].GetBuffer();
            
            // myiocb->aio_buf = (std::uint64_t)(test.data());
            // myiocb->aio_nbytes = 5120;
            
            
            myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);
            //myiocb->aio_offset = static_cast<std::int64_t>(4096);

            //iocbs[fileid].emplace_back(myiocb);
            iocbs[handler_num].emplace_back(myiocb);
        }
        //std::cout << "event creating\n";
        std::vector<struct io_event> events(totalToSubmit);
        int totalDone = 0, totalSubmitted = 0, totalQueued = 0;

        //std::cout<<"At begining totalDone: " << totalDone << " totalToSubmit" << totalToSubmit << std::endl;
        // while(totalDone<totalToSubmit)
        int kk = 0;
        while (totalDone < totalToSubmit) 
        {
            
            //std::cout << kk++ << std::endl;
            if (totalSubmitted < totalToSubmit) {
                //for (int i = 0; i < handlers.size(); i++) {
                for(int i = 0; i < handler_size; i++){
                    if (submitted[i] < iocbs[i].size()) {
                        //AsyncFileIO* handler = (AsyncFileIO*)(handlers[i].get());
                        //int s = syscall(__NR_io_submit, handler->GetIOCP(channel), iocbs[i].size() - submitted[i], iocbs[i].data() + submitted[i]);
                        int s = syscall(__NR_io_submit, aio_ctx, iocbs[i].size() - submitted[i], iocbs[i].data() + submitted[i]);
                        // handler->GetIOCP  : aio_context_t
                        //std::cout <<  i << " System call " << std::endl;
                        //std::cout << "s:" << s << "  totalToSubmit:" << totalToSubmit << std::endl;

                        if (s > 0) {
                            submitted[i] += s;
                            totalSubmitted += s;
                        }
                        else {
                            //SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "fid:%d channel %d, to submit:%d, submitted:%s\n", i, channel, iocbs[i].size() - submitted[i], strerror(-s));
                            FAISS_THROW_FMT("fid:%d channel %d, to submit:%ld, submitted:%s\n", i, 404, iocbs[i].size() - submitted[i], strerror(-s));
                        }
                    }
                }
            }

            for (int i = totalQueued; i < totalDone; i++) {
                AsyncReadRequest* req = reinterpret_cast<AsyncReadRequest*>((events[i].data));
                //std::cout << "Rq: ";
                //if(0)
                if (nullptr != req)
                {
                    //std::cout << "True" << std::endl;
                    std::vector<float> tmp(req->len * d);
                    convert_to_float(req->len, tmp.data(), req->m_buffer);
                    req->converted_buffer = tmp.data();
                    req->m_callback(req);
                }
                //std::cout << "False" << std::endl;
            }
            totalQueued = totalDone;

            //for (int i = 0; i < handlers.size(); i++) {
            for (int i = 0; i < handler_size; i++){
                if (done[i] < submitted[i]) {
                    int wait = submitted[i] - done[i];
                    //AsyncFileIO* handler = (AsyncFileIO*)(handlers[i].get());
                    //auto d = syscall(__NR_io_getevents, handler->GetIOCP(channel), wait, wait, events.data() + totalDone, &AIOTimeout);

                    //std::cout << "done:" << done[i] << "  submitted:" << submitted[i] << std::endl;
                    auto _d = syscall(__NR_io_getevents, aio_ctx, wait, wait, events.data() + totalDone, &AIOTimeout);
                    //std::cerr << "aio_ctx:"<< aio_ctx <<" wait:" << wait <<" I/O num: " << events[i].res << " Error:" << strerror(-events[i].res)<< " _d="<<_d <<" i:" << i <<std::endl;
                    //std::cerr << "aio_ctx:"<< aio_ctx <<" wait:" << wait << std::endl;
                    // for(size_t x = done[i]; x < done[i] + _d; x++){
                    //     std::cout << x<<" Read:"<<events[x].res << std::endl;
                    //     if( _d > 0){
                    //         for(int j = 0; j < 128; j++){
                    //             std::cout << ((float*)(page_buffers[x].GetBuffer()))[j] << "  ";
                    //         }
                    //         std::cout << "\n";
                    //     }
                    // }

                    //std::cerr<< "totalDone:" << totalDone << std::endl;
                    done[i] += _d;
                    totalDone += _d;
                    //std::cout <<" Finished: " << _d << std::endl;

                    

                    // if( _d > 0){
                    //     for(int j = 0; j < 10; j++){
                    //         if(test.data()[j] == 9)
                    //             std::cout<<".";
                    //         else
                    //             std::cout << "vector: "<<test.data()[j] << std::endl;
                    //     }
                    // }
                }
            }
            
            // std::cout << "done: "<< done[0] << " totalDone: " << totalDone << std::endl;
        }

        for (int i = totalQueued; i < totalDone; i++) {
            AsyncReadRequest* req = reinterpret_cast<AsyncReadRequest*>((events[i].data));
            if (nullptr != req)
            {
                std::vector<float> tmp(req->len * d);
                convert_to_float(req->len, tmp.data(), req->m_buffer);
                req->converted_buffer = tmp.data();
                req->m_callback(req);
            }
        }

    }
    

    void submit_partially(){
        const int handler_size = 1;
        const int handler_num = handler_size - 1;
        
        // 一共有n个聚类
        int num_cluster = partial_diskRequests.size();

        int iocb_num = 0;
        for(int i = 0; i < num_cluster; i++){
            //auto requests_l = partial_diskRequests[i]->data();
            iocb_num += partial_diskRequests[i]->size();
        }

        std::vector<struct iocb> myiocbs(iocb_num);
        std::vector<std::vector<struct iocb*>> iocbs(handler_size);   
        std::vector<int> submitted(handler_size, 0);
        std::vector<int> done(handler_size, 0);

        //std::vector<float>test(10*128, 9);
        
        int totalToSubmit = 0;
        int totalVector = 0;
        memset(myiocbs.data(), 0, iocb_num * sizeof(struct iocb));

        idx_t* keys = new idx_t[num_cluster];

        for (int i = 0; i < num_cluster; i++) {
            auto readRequests = partial_diskRequests[i]->data();
            
            //std::cout << "num_cluster:" << num_cluster<< " i:" << i << "   requests:" << partial_diskRequests[i]->size() ;
            //std::cout << "Page_buffer: " << page_buffers.size()<< std::endl;
            keys[i] = readRequests[0].rerank_info.key;
            for(int j = 0; j < partial_diskRequests[i]->size(); j++){
                AsyncReadRequest_Partial* readRequest = readRequests + j;
                totalVector += readRequest->in_buffer_offsets.size();
                struct iocb* myiocb = &(myiocbs[totalToSubmit]);
                myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
                myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
                myiocb->aio_fildes = this->fd;
                myiocb->aio_nbytes = readRequest->m_readSize;
                //  一个list有很多buffer，所以不能用i或者j来。
                myiocb->aio_buf = (std::uint64_t)(page_buffers[totalToSubmit].GetBuffer());
                readRequest->m_buffer = (char*)page_buffers[totalToSubmit].GetBuffer();
                myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);
                iocbs[handler_num].emplace_back(myiocb);
                totalToSubmit++;
            }
            // AsyncReadRequest* readRequest = full_diskRequests[i].get();
            // //std::cout << "read_size:" << readRequest->m_readSize << "  read_offset:" << readRequest->m_offset << std::endl;
            // //std::cout <<i <<  " Processing necessary info stage 1\n";
            // //channel = readRequest->m_status & 0xffff;
            // //int fileid = (readRequest->m_status >> 16);
            // struct iocb* myiocb = &(myiocbs[totalToSubmit++]);
            // myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
            
            // myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
            // //myiocb->aio_fildes = ((AsyncFileIO*)(handlers[fileid].get()))->GetFileHandler();
            // myiocb->aio_fildes = this->fd;
            // // myiocb->aio_buf = (std::uint64_t)(readRequest->m_buffer);
            // //std::cout <<i <<  " Processing necessary info stage 2\n";
            // myiocb->aio_nbytes = readRequest->m_readSize;
            // //std::cout <<i <<  " Processing necessary info stage 3\n";
            // myiocb->aio_buf = (std::uint64_t)(page_buffers[i].GetBuffer());
            // readRequest->m_buffer = (char*)page_buffers[i].GetBuffer();
            
            // // myiocb->aio_buf = (std::uint64_t)(test.data());
            // // myiocb->aio_nbytes = 5120;
            
            
            // myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);
            // //myiocb->aio_offset = static_cast<std::int64_t>(4096);

            // //iocbs[fileid].emplace_back(myiocb);
            // iocbs[handler_num].emplace_back(myiocb);
        }
        //std::cout << "event creating\n";
        std::vector<struct io_event> events(totalToSubmit);
        int totalDone = 0, totalSubmitted = 0, totalQueued = 0;

        std::map<idx_t, std::vector<float>> global_dis;
        std::map<idx_t, std::vector<size_t>> global_ids;

        for(int i = 0;i < num_cluster;i++){
            global_dis[keys[i]] = std::vector<float>();   // 初始化为空的vector
            global_ids[keys[i]] = std::vector<size_t>(); 
        }
        //global_dis.resize(num_cluster);
        //global_ids.resize(num_cluster);
        //std::cout <<"num_cluster:" <<num_cluster << std::endl;
        //std::cout << "totalVector:" << totalVector << std::endl ;

        //std::cout<<"At begining totalDone: " << totalDone << " totalToSubmit" << totalToSubmit << std::endl;
        int kk = 0;
        while (totalDone < totalToSubmit) 
        {
            
            //std::cout << kk++ << std::endl;
            if (totalSubmitted < totalToSubmit) {
                //for (int i = 0; i < handlers.size(); i++) {
                for(int i = 0; i < handler_size; i++){
                    if (submitted[i] < iocbs[i].size()) {
                        //AsyncFileIO* handler = (AsyncFileIO*)(handlers[i].get());
                        //int s = syscall(__NR_io_submit, handler->GetIOCP(channel), iocbs[i].size() - submitted[i], iocbs[i].data() + submitted[i]);
                        int s = syscall(__NR_io_submit, aio_ctx, iocbs[i].size() - submitted[i], iocbs[i].data() + submitted[i]);
                        // handler->GetIOCP  : aio_context_t
                        //std::cout <<  i << " System call " << std::endl;
                        //std::cout << "s:" << s << "  totalToSubmit:" << totalToSubmit << std::endl;

                        if (s > 0) {
                            submitted[i] += s;
                            totalSubmitted += s;
                        }
                        else {
                            //SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "fid:%d channel %d, to submit:%d, submitted:%s\n", i, channel, iocbs[i].size() - submitted[i], strerror(-s));
                            FAISS_THROW_FMT("fid:%d channel %d, to submit:%ld, submitted:%s\n", i, 404, iocbs[i].size() - submitted[i], strerror(-s));
                        }
                    }
                }
            }

            for (int i = totalQueued; i < totalDone; i++) {
                AsyncReadRequest_Partial* req = reinterpret_cast<AsyncReadRequest_Partial*>((events[i].data));
                //std::cout << "Rq: ";
                //if(0)
                if (nullptr != req)
                {
                    //std::cout << "True" << std::endl;
                    // TODO convert only vectors to be calculated
                    // std::vector<float> tmp(req->len * d);
                    // convert_to_float(req->len, tmp.data(), req->m_buffer);
                    // req->converted_buffer = tmp.data();
                    req->converted_buffer = reinterpret_cast<float*>(req->m_buffer);
                    req->m_callback_calculation(req, global_dis[req->rerank_info.key], global_ids[req->rerank_info.key]);
                }
                //std::cout << "False" << std::endl;
            }
            totalQueued = totalDone;
            //std::cout << "totalQueued: " << totalQueued << "  totalDone:" <<  totalDone<< std::endl;

            //for (int i = 0; i < handlers.size(); i++) {
            for (int i = 0; i < handler_size; i++){
                if (done[i] < submitted[i]) {
                    int wait = submitted[i] - done[i];
                    //AsyncFileIO* handler = (AsyncFileIO*)(handlers[i].get());
                    //auto d = syscall(__NR_io_getevents, handler->GetIOCP(channel), wait, wait, events.data() + totalDone, &AIOTimeout);

                    //std::cout << "done:" << done[i] << "  submitted:" << submitted[i] << std::endl;
                    auto _d = syscall(__NR_io_getevents, aio_ctx, wait, wait, events.data() + totalDone, &AIOTimeout);
                    
                    //for(size_t x = done[i]; x < done[i] + _d; x++){
                        //std::cerr << "aio_ctx:"<< aio_ctx <<" wait:" << wait <<" I/O num: " << events[x].res << " Error:" << strerror(-events[i].res)<< " _d="<<_d <<" i:" << i <<std::endl;
                        //std::cerr << "aio_ctx:"<< aio_ctx <<" wait:" << wait << std::endl;
                        //std::cout << x<<" Read:"<<events[x].res << std::endl;
                        // if( _d > 0){
                        //     for(int j = 0; j < 128; j++){
                        //         std::cout << ((float*)(page_buffers[x].GetBuffer()))[j] << "  ";
                        //     }
                        //     std::cout << "\n";
                        // }
                    //}

                    //std::cerr<< "totalDone:" << totalDone << std::endl;
                    done[i] += _d;
                    totalDone += _d;
                    //std::cout <<" Finished: " << _d << std::endl;

                    

                    // if( _d > 0){
                    //     for(int j = 0; j < 10; j++){
                    //         if(test.data()[j] == 9)
                    //             std::cout<<".";
                    //         else
                    //             std::cout << "vector: "<<test.data()[j] << std::endl;
                    //     }
                    // }
                }
            }
            
            //std::cout << "done: "<< done[0] << " totalDone: " << totalDone << std::endl;
        }

        for (int i = totalQueued; i < totalDone; i++) {
            AsyncReadRequest_Partial* req = reinterpret_cast<AsyncReadRequest_Partial*>((events[i].data));
            //std::cout << "Rq: ";
            //if(0)
            if (nullptr != req)
            {
                //std::cout << "True" << std::endl;
                // TODO convert only vectors to be calculated
                // std::vector<float> tmp(req->len * d);
                // convert_to_float(req->len, tmp.data(), req->m_buffer);
                // req->converted_buffer = tmp.data();
                req->converted_buffer = reinterpret_cast<float*>(req->m_buffer);
                req->m_callback_calculation(req, global_dis[req->rerank_info.key], global_ids[req->rerank_info.key]);
            }
            //std::cout << "False" << std::endl;
        }

         for (int i = 0; i < totalDone; i++) {
            AsyncReadRequest_Partial* req = reinterpret_cast<AsyncReadRequest_Partial*>((events[i].data));
            if (nullptr != req)
            {
                // for(int i = 0; i < global_dis[req->rerank_info.key].size(); i++){
                //      std::cout << "key:" << req->rerank_info.key << "  dis"<<global_dis[req->rerank_info.key][i] 
                //      << "  ids:" << global_ids[req->rerank_info.key][i] << std::endl; 
                // }
               // 重复加了很多？
                if(!global_dis[req->rerank_info.key].empty()){
                    req->m_callback(req, global_dis[req->rerank_info.key], global_ids[req->rerank_info.key]);
                    global_dis[req->rerank_info.key].clear();
                    global_ids[req->rerank_info.key].clear();
                }
                
            }
            //std::cout << "False" << std::endl;
        }

        // for(int i = 0; i < global_dis.size(); i++){
        //     std::cout << "ids: "<< global_ids[i]   << "    dis: " << global_dis[i] << std::endl;
        // }
    }

    void clear(){
        // After finishing a query, delete existed requests and buffers.
        full_diskRequests.clear();
        partial_diskRequests.clear();
        page_buffers.clear();
        //converted_data.clear();

    }

    void submit(int num = -1) override {
        
        if(num > 0){
            //std::cout << " Async submitting" << std::endl;
            if(!full_diskRequests.empty()){
                //submit
                //std::cout << "Submit_full: "<< num << std::endl;
                submit_fully(num);
                //clear();
                full_diskRequests.clear();
                page_buffers.clear();
                //std::cout << " Async submit successfully" << std::endl;
            }
            
        }else{
            if(!partial_diskRequests.empty()){
                //submit
                //std::cout << "Submit_partial: " << partial_diskRequests.size() << "=size" << std::endl;
                submit_partially();
                //clear();
                partial_diskRequests.clear();
                page_buffers.clear();
            }else{
                //std::cout << "Partial NULL!!" << std::endl;
            }
            
        }

        
        

    }

    // TODO we assume that now that page size divide by vector size. 
    int process_page(int* vector_to_submit, int* page_to_search, size_t* vec_page_proj, size_t len_p) override {
        
        int vector_size = sizeof(ValueType) * d;
        int vec_per_page = PAGE_SIZE/vector_size;  // TODO maybe not divisible

        int page_num = -1;
        int total_page = 0;
        for(int i = 0; i < len_p; i++){
            int tmp_num = vector_to_submit[i]/vec_per_page;
            vec_page_proj[i] = tmp_num;   // calculate the brgin scope of vectors in each page
            if(tmp_num != page_num){
                page_to_search[total_page] = tmp_num;
                page_num = tmp_num;
                total_page++;
            }
        }
        return total_page;
    }

    // ?????? 
    int in_request_rank(size_t in_cluster_id, size_t start_page, size_t read_size){
        return 0;
    }

    void test() override {
        std::cout << "DiskIOAsync:" << std::endl;
    }

    int get_per_page_element() override {

        return PAGE_SIZE/sizeof(ValueType) ;
    }
    

};


template<typename ValueType>
struct IVF_DiskIOSearchProcessor_Async_PQ : DiskIOProcessor{
    size_t factor_partial;
    
    // partial没有集成到一起
    //std::vector<PageBuffer<uint8_t>> page_buffers;
    //std::vector<std::vector<AsyncReadRequest_Partial_PQDecode>> partial_diskRequests; 
    
    AsyncReadRequests_Partial_PQDecode*  partial_diskRequests;

    // full的page_buffers和具体的request在full_diskRequests里
    AsyncReadRequests_Full_PQDecode* full_diskRequests;

    AsyncRequests_IndexInfo* info_diskRequests;

    int fd;
    aio_context_t aio_ctx;    // AIO context

    IVF_DiskIOSearchProcessor_Async_PQ(std::string disk_path, 
                               size_t d): DiskIOProcessor(disk_path, d), aio_ctx(0){
    }

    void initial(std::uint64_t maxIOSize = (1 << 20),
                 std::uint32_t maxReadRetries = 2,
                 std::uint32_t maxWriteRetries = 2,
                 std::uint16_t threadPoolSize = 4) override {
        maxIOSize = 1<<10;
        // O_DIRECT
        fd = open(disk_path.c_str(), O_RDONLY | O_DIRECT);   // Must align
        if (fd < 0) {
            perror("Failed to open file with O_DIRECT");
            throw std::runtime_error("Failed to open file");
        }
        
        auto ret = syscall(__NR_io_setup, (int)maxIOSize, &aio_ctx);
        if(ret < 0){
            perror("io_setup failed");
            throw std::runtime_error("Failed to initialize AIO context");
        }
        std::cout << "AIO context initialized and file opened with O_DIRECT: " << disk_path << std::endl;
    }

    ~IVF_DiskIOSearchProcessor_Async_PQ() {
        if (syscall(__NR_io_destroy, aio_ctx) < 0) {
            std::cerr << "Failed to destroy AIO context." << std::endl;
        }

        if (fd >= 0) {
            close(fd);
            std::cout << "AIO context destroyed and file closed successfully." << std::endl;
        }
    }


    void convert_to_float(size_t n, float* vectors, void* disk_data) override {
        //std::cout << " Convert:" << std::endl;
        // ValueType* original_vectors = reinterpret_cast<ValueType*>(disk_data);
        // for(size_t i = 0; i < n; i++){
        //     for(size_t j = 0; j < d; j++)
        //     {
        //         vectors[i*d+j] = static_cast<float>(original_vectors[i*d+j]);
        //     }
        // }
        uint8_t* data = static_cast<uint8_t*>(disk_data);

        size_t i = 0;
        const size_t simd_step = 32; // 每次可以处理 32 个 uint8_t

        for (size_t i = 0; i < n; i += 32) {
            // 加载 32 个 uint8 数据
            __m256i raw_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data[i]));

            // 提取前 16 字节（低 128 位）和后 16 字节（高 128 位）
            __m128i data_lo = _mm256_castsi256_si128(raw_data); // 前 16 字节
            __m128i data_hi = _mm256_extracti128_si256(raw_data, 1); // 后 16 字节

            // 将每部分扩展为 uint16
            __m256i uint16_lo = _mm256_cvtepu8_epi16(data_lo);
            __m256i uint16_hi = _mm256_cvtepu8_epi16(data_hi);

            // 转换为浮点数
            __m256 float_lo = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(uint16_lo, _mm256_setzero_si256()));
            __m256 float_hi = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(uint16_lo, _mm256_setzero_si256()));
            __m256 float_lo2 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(uint16_hi, _mm256_setzero_si256()));
            __m256 float_hi2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(uint16_hi, _mm256_setzero_si256()));

            // 按顺序存储
            _mm256_storeu_ps(&vectors[i + 0], float_lo);
            _mm256_storeu_ps(&vectors[i + 8], float_hi);
            _mm256_storeu_ps(&vectors[i + 16], float_lo2);
            _mm256_storeu_ps(&vectors[i + 24], float_hi2);
        }

        // 处理剩余的元素
        for (; i < n; ++i) {
            vectors[i] = static_cast<float>(data[i]);
        }
        
        // for(int ii = 0;ii < n/d; ii++){
        //     for(int jj = 0; jj < d; jj++){
        //        std::cout << vectors[ii*d + jj] << " " ;
        //     }
        //     std::cout << "\n\n";
        //     for(int jj = 0; jj < d; jj++){
        //        std::cout << (int)data[ii*d + jj] << " " ;
        //     }
        //     std::cout << "\n\n\n\n";
        // }

    }

    float* convert_to_float_single(float* vector, void* disk_data, int begin){
        
        ValueType* original_vector = reinterpret_cast<ValueType*>(disk_data) + begin;
        for(int i = 0; i < d; i++){
            //vector[i] = static_cast<float>(original_vector[i]);
            vector[i] = (float)(original_vector[i]);
        }
        return vector;
    }


    struct timespec AIOTimeout {0, 30000};
    void submit_fully(int num){
        int iocb_num = num;
        std::vector<struct iocb> myiocbs(iocb_num);
        std::vector<struct iocb*> iocbs;
        int submitted = 0;
        int done = 0;
        //std::cout << "submitting!!!" << std::endl;
        
        // reserve
        iocbs.reserve(iocb_num);
        int totalToSubmit = 0;
        int totalVector = 0;
        memset(myiocbs.data(), 0, iocb_num * sizeof(struct iocb));

        for(int i = 0; i < iocb_num; i++){
            AsyncRequest_Full_Batch* readRequest = full_diskRequests->list_requests.data() + i;
            // totalVector += readRequest->in_buffer_offsets.size();
            struct iocb* myiocb = &(myiocbs[totalToSubmit]);
            myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
            myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
            myiocb->aio_fildes = this->fd;
            myiocb->aio_nbytes = readRequest->m_readSize;

            myiocb->aio_buf = (std::uint64_t)(full_diskRequests->page_buffers[totalToSubmit].GetBuffer());
            // readRequest->m_buffer = (char*)page_buffers[totalToSubmit].GetBuffer();  这个在fill_buffer里已经确认了
            myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);

            iocbs.emplace_back(myiocb);

            totalToSubmit++;
        }

        std::vector<struct io_event> events(totalToSubmit);
        int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
        bool pq_decode = true;
        int kk = 0;

        if(totalToSubmit == 0){
            if(pq_decode){
                full_diskRequests->pq_callback();
                pq_decode = false;
            }
            return;
        }

        //std::cout << "pq to submit:" << totalToSubmit << std::endl;

        while (totalDone < totalToSubmit) 
        {
            if (totalSubmitted < totalToSubmit) {
                
                if (submitted < iocbs.size()) {
                    int s = syscall(__NR_io_submit, aio_ctx, iocbs.size() - submitted, iocbs.data() + submitted);
                    if (s > 0) {
                        submitted += s;
                        totalSubmitted += s;
                    }
                    else {
                        FAISS_THROW_FMT("To submit:%ld, submitted:%s\n", iocbs.size() - submitted, strerror(-s));
                    }
                }
                
            }
            if(pq_decode){
                full_diskRequests->pq_callback();
                pq_decode = false;
            }

            for (int i = totalQueued; i < totalDone; i++) {
                AsyncRequest_Full_Batch* req = reinterpret_cast<AsyncRequest_Full_Batch*>((events[i].data));
                if (nullptr != req)
                {   
                    AsyncRequest_Full* single_list = req->request_full.data();
                    for(int j = 0; j < req->list_num; j++)
                    {
                        // std::vector<float> tmp( (single_list + j)->vectors_num * d);
                        // convert_to_float( (single_list + j)->vectors_num * d, tmp.data(), (single_list + j)->m_buffer);
                        // full_diskRequests->cal_callback(single_list + j, tmp.data());

                        full_diskRequests->cal_callback(single_list + j, (single_list + j)->m_buffer);
                        // for(int ii = 0; ii < (single_list + j)->vectors_num * d; ii++){
                        //     std::cout << (tmp.data())[ii] << ". ";
                        //     if(ii%d == 0){
                        //         std::cout <<"\n\n";
                        //     }
                        // }
                    }   
                }
            }
            totalQueued = totalDone;
            if (done < submitted) {
                int wait = submitted - done;
                auto _d = syscall(__NR_io_getevents, aio_ctx, wait, wait, events.data() + totalDone, &AIOTimeout);
                //std::cout << "_d:" << _d << std::endl; 
                done += _d;
                totalDone += _d;
            }
        }

        
        //std::cout << totalQueued << " ";
        for (int i = totalQueued; i < totalDone; i++) {

            AsyncRequest_Full_Batch* req = reinterpret_cast<AsyncRequest_Full_Batch*>((events[i].data));
            if (nullptr != req)
            {  
                AsyncRequest_Full* single_list = req->request_full.data();
                for(int j = 0; j < req->list_num; j++)
                {   
                    // std::vector<float> tmp((single_list + j)->vectors_num * d);
                    // convert_to_float((single_list + j)->vectors_num * d, tmp.data(), (single_list + j)->m_buffer);
                    // full_diskRequests->cal_callback(single_list + j, tmp.data());

                    full_diskRequests->cal_callback(single_list + j, (single_list + j)->m_buffer);
                }   
            }

        }

    }


    void submit_partially(){
        int iocb_num = partial_diskRequests->list_requests.size();
        std::vector<struct iocb> myiocbs(iocb_num);
        std::vector<struct iocb*> iocbs;
        int submitted = 0;
        int done = 0;
        //std::cout << "iocb_num:" << iocb_num << "\n";

        iocbs.reserve(iocb_num);
        int totalToSubmit = 0;
        int totalVector = 0;
        memset(myiocbs.data(), 0, iocb_num * sizeof(struct iocb));

        for(int i = 0; i < iocb_num; i++){
            AsyncRequest_Partial* readRequest = partial_diskRequests->list_requests.data() + i;
            // totalVector += readRequest->in_buffer_offsets.size();
            struct iocb* myiocb = &(myiocbs[totalToSubmit]);
            myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
            myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
            myiocb->aio_fildes = this->fd;
            myiocb->aio_nbytes = readRequest->m_readSize;

            myiocb->aio_buf = (std::uint64_t)(partial_diskRequests->page_buffers[totalToSubmit].GetBuffer());
            // readRequest->m_buffer = (char*)page_buffers[totalToSubmit].GetBuffer();  这个在fill_buffer里已经确认了
            myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);

            //std::cout << "myiocb->aio_nbytes: " << myiocb->aio_nbytes << "  myiocb->aio_offset:" << myiocb->aio_offset << std::endl;

            iocbs.emplace_back(myiocb);
            totalToSubmit++;
        }

        std::vector<struct io_event> events(totalToSubmit);
        int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
        bool pq_decode = true;
        int kk = 0;

        if(totalToSubmit == 0){
            if(pq_decode){
                partial_diskRequests->pq_callback();
                pq_decode = false;
            }
            return;
        }

        while (totalDone < totalToSubmit) 
        {
            if (totalSubmitted < totalToSubmit) {
                
                if (submitted < iocbs.size()) {
                    int s = syscall(__NR_io_submit, aio_ctx, iocbs.size() - submitted, iocbs.data() + submitted);
                    if (s > 0) {
                        submitted += s;
                        totalSubmitted += s;
                    }
                    else {
                        std::cout << "submit fails:" << s << "  submitted:" << submitted << "  total:" << totalToSubmit << std::endl;
                        FAISS_THROW_FMT("To submit:%ld, submitted:%s\n", iocbs.size() - submitted, strerror(-s));
                    }
                }
            }
            
            if(pq_decode){
                partial_diskRequests->pq_callback();
                pq_decode = false;
            }

            for (int i = totalQueued; i < totalDone; i++) {
                AsyncRequest_Partial* req = reinterpret_cast<AsyncRequest_Partial*>((events[i].data));
                if (nullptr != req)
                {   
                    partial_diskRequests->cal_callback(req, req->m_buffer);
                }
            }
            totalQueued = totalDone;
            if (done < submitted) {
                int wait = submitted - done;
                auto _d = syscall(__NR_io_getevents, aio_ctx, wait, wait, events.data() + totalDone, &AIOTimeout);
                if(_d < 0){
                    std::cout << "_d: " << _d << "\n";
                    FAISS_THROW_FMT("To get:%d, error:%s\n", wait, strerror(-_d));
                    exit(1);
                }
                done += _d;
                totalDone += _d;
            }
        }
        for (int i = totalQueued; i < totalDone; i++) {
            AsyncRequest_Partial* req = reinterpret_cast<AsyncRequest_Partial*>((events[i].data));
            if (nullptr != req)
            {   
                partial_diskRequests->cal_callback(req, req->m_buffer);
            }
        }

    }

    void submit_info(){
        int iocb_num = info_diskRequests->info_requests.size();
        std::vector<struct iocb> myiocbs(iocb_num);
        std::vector<struct iocb*> iocbs;
        int submitted = 0;
        int done = 0;
        //std::cout << "iocb_num:" << iocb_num << "\n";

        iocbs.reserve(iocb_num);
        int totalToSubmit = 0;
        int totalVector = 0;
        memset(myiocbs.data(), 0, iocb_num * sizeof(struct iocb));

        for(int i = 0; i < iocb_num; i++){
            AsyncRequest_IndexInfo* readRequest = info_diskRequests->info_requests.data() + i;
            // totalVector += readRequest->in_buffer_offsets.size();
            struct iocb* myiocb = &(myiocbs[totalToSubmit]);
            myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
            myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
            myiocb->aio_fildes = this->fd;
            myiocb->aio_nbytes = readRequest->m_readSize;

            myiocb->aio_buf = (std::uint64_t)(info_diskRequests->page_buffers[totalToSubmit].GetBuffer());
            // readRequest->m_buffer = (char*)page_buffers[totalToSubmit].GetBuffer();  这个在fill_buffer里已经确认了
            myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);
            iocbs.emplace_back(myiocb);
            totalToSubmit++;
        }

        std::vector<struct io_event> events(totalToSubmit);
        int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
        bool pq_decode = true;
        while (totalDone < totalToSubmit) 
        {
            if (totalSubmitted < totalToSubmit) {
                
                if (submitted < iocbs.size()) {
                    int s = syscall(__NR_io_submit, aio_ctx, iocbs.size() - submitted, iocbs.data() + submitted);
                    if (s > 0) {
                        submitted += s;
                        totalSubmitted += s;
                    }
                    else {
                        FAISS_THROW_FMT("To submit:%ld, submitted:%s\n", iocbs.size() - submitted, strerror(-s));
                    }
                }
            }

            //totalQueued = totalDone;
            if (done < submitted) {
                int wait = submitted - done;
                auto _d = syscall(__NR_io_getevents, aio_ctx, wait, wait, events.data() + totalDone, &AIOTimeout);
                done += _d;
                totalDone += _d;
                //std::cout << "totalDone:" << totalDone << "  totalToSubmit:" << totalToSubmit << "\n";
            }
        }


    }


    void submit(int num = -1) override{
        
        if(num == -1){
            submit_partially();
            partial_diskRequests->page_buffers.clear();

        }else if(num >= 0){
            submit_fully(num);
            // full_diskRequests里应该只是局部变量，把page_buffers清空就好了
            full_diskRequests->page_buffers.clear();
        }else if(num == -2){
            submit_info();
            // 可以用回调函数来帮助放置
        }
        else{
            return;
        }
    }

    int process_page(int* vector_to_submit, int* page_to_search, size_t* vec_page_proj, size_t len_p) override {

        int vector_size = sizeof(ValueType) * d;
        int vec_per_page = PAGE_SIZE/vector_size;  // TODO maybe not divisible

        int page_num = -1;
        int total_page = 0;
        for(int i = 0; i < len_p; i++){
            int tmp_num = vector_to_submit[i]/vec_per_page;
            vec_page_proj[i] = tmp_num;   // calculate the brgin scope of vectors in each page
            if(tmp_num != page_num){
                page_to_search[total_page] = tmp_num;
                page_num = tmp_num;
                total_page++;
            }
        }
        return total_page;
    }

    int process_page_transpage(int* vector_to_submit, Page_to_Search* page_to_search, size_t* vec_page_proj, size_t len_p)override{
        int vector_size = sizeof(ValueType) * d;

        // 每页的字节大小
        //constexpr int PAGE_SIZE = 4096;

        // 页中能存储的完整向量数量（可能不是整数倍）
        int vec_per_page = PAGE_SIZE / vector_size;

        // 页编号变量
        int current_first_page = -1;
        int current_last_page = -1;
        int total_page_count = 0;

        for (size_t i = 0; i < len_p; i++) {
            // 当前向量的字节偏移量
            size_t vector_offset = vector_to_submit[i] * vector_size;

            // 计算向量的起始页号和结束页号
            int start_page = vector_offset / PAGE_SIZE;
            int end_page = (vector_offset + vector_size - 1) / PAGE_SIZE; // 考虑跨页部分

            // 更新 vec_page_proj 映射信息
            vec_page_proj[i] = start_page;

            // 如果是新的一组页
            if (start_page != current_first_page || end_page != current_last_page) {
                // 如果当前页范围有效，记录到 page_to_search
                if (current_first_page != -1) {
                    page_to_search[total_page_count].first = current_first_page;
                    page_to_search[total_page_count].last = current_last_page;

                    total_page_count++;
                }

                // 更新当前页范围
                current_first_page = start_page;
                current_last_page = end_page;
            }
        }

        // 最后一组页范围需要处理
        if (current_first_page != -1) {
            page_to_search[total_page_count].first = current_first_page;
            page_to_search[total_page_count].last = current_last_page;
            total_page_count++;
        }

        return total_page_count;
    }

    int get_per_page_element() override {

        return PAGE_SIZE/sizeof(ValueType) ;
    }

    void disk_io_partial_async_pq(AsyncReadRequests_Partial_PQDecode& asyncReadRequests_p) override{
        asyncReadRequests_p.fill_buffer();
        this->partial_diskRequests = &asyncReadRequests_p;
    }

    void disk_io_full_async_pq(AsyncReadRequests_Full_PQDecode& asyncReadRequests_f)override{
        int r_size = asyncReadRequests_f.list_requests.size();
        
        if(r_size!=0){
            asyncReadRequests_f.page_buffers.reserve(r_size);

            for(int i = 0; i < r_size; i++){
                PageBuffer<uint8_t> page_buffer;
                page_buffer.ReservePageBuffer(asyncReadRequests_f.list_requests[i].m_readSize); // 设置 buffer 大小
                asyncReadRequests_f.page_buffers.emplace_back(std::move(page_buffer));
            }
            asyncReadRequests_f.fill_buffer(); // TODO 检查是否分配成功
        }
        this->full_diskRequests = &asyncReadRequests_f;
    }

    void disk_io_info_async(AsyncRequests_IndexInfo& asyncReadRequests_i)override{
        int r_size = asyncReadRequests_i.info_requests.size();
        if(r_size!= 0){
            asyncReadRequests_i.fill_buffer();
        }
        this->info_diskRequests = &asyncReadRequests_i;
    }

    void disk_io_all(int D,
                    size_t len,
                    size_t listno, 
                    float* vectors,
                    Aligned_Cluster_Info* acInfo){
         if (fd < 0) {
            std::cerr << "File descriptor is invalid!" << std::endl;
            return;
        }

        // 获取聚类的信息
        size_t page_start = acInfo[listno].page_start;
        size_t padding_offset = acInfo[listno].padding_offset;
        size_t page_count = acInfo[listno].page_count;

        // std::cout << "page_start:" << page_start << 
        //             "  padding_offset:" << padding_offset <<
        //             "  page_count:" << page_count << std::endl;

        // 计算磁盘文件中该list的起始位置（字节偏移量）
        size_t offset = page_start * PAGE_SIZE;

        // 计算需要读取的总字节数，减去填充字节数
        size_t total_bytes = page_count * PAGE_SIZE;

        // 分配对齐的缓冲区（O_DIRECT需要内存对齐）
        PageBuffer<uint8_t> page_buffer;
        page_buffer.ReservePageBuffer(total_bytes); // 设置 buffer 大小

        // 准备异步I/O请求
        struct iocb cb;
        struct iocb* cbs[1];
        memset(&cb, 0, sizeof(cb));

        cb.aio_fildes = fd;
        cb.aio_lio_opcode = IOCB_CMD_PREAD;
        cb.aio_buf = reinterpret_cast<std::uint64_t>(page_buffer.GetBuffer());
        cb.aio_offset = static_cast<std::int64_t>(offset);
        cb.aio_nbytes = total_bytes;
        cb.aio_data = reinterpret_cast<std::uint64_t>(&cb); // 用于在回调中识别

        cbs[0] = &cb;

        // 提交异步I/O请求
        int ret = syscall(__NR_io_submit, aio_ctx, 1, cbs);
        //std::cout << "ret: " << ret << std::endl;

        // 等待I/O完成
        struct io_event events[1];
        memset(events, 0, sizeof(events));
        int num_events = 0;
        while(num_events < 1){
            auto _d = syscall(__NR_io_getevents, aio_ctx, 1, 1, events, &AIOTimeout);
            num_events += _d;
        }
        
        
        if (reinterpret_cast<uintptr_t>(page_buffer.GetBuffer()) % 512 != 0) {
            std::cerr << "Buffer alignment error for O_DIRECT." << std::endl;
        }
        //std::cout << "num_events:" << num_events << std::endl;

        //std::cout << "total_bytes:" << total_bytes <<std::endl;
        //std::cout << "events.res :" << events[0].res << std::endl;
        // 检查I/O是否成功
        if (events[0].res != static_cast<size_t>(total_bytes)) {
            std::cerr << "Asynchronous read failed or incomplete!" << std::endl;

        }
        // 将读取的数据转换为 float 类型并存储到 vectors 中
        ValueType* data = reinterpret_cast<ValueType*>(page_buffer.GetBuffer());
        for (size_t i = 0; i < len; ++i) {
            for (size_t j = 0; j < D; ++j) {
                vectors[i * D + j] = static_cast<float>(data[i * D + j]);
            }
        }
    }
        

};

} // namespace faiss

#endif