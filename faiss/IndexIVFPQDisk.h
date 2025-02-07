#ifndef FAISS_INDEX_IVFPQDisk_H
#define FAISS_INDEX_IVFPQDisk_H

#include <vector>

#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/impl/DiskInvertedListHolder.h>
#include <faiss/impl/DiskIOProcessor.h>

#include <string>
#include <fstream>   

#include <chrono>
using namespace std::chrono;


namespace faiss {

class IndexIVFPQDisk : public IndexIVFPQ {
public:
    
    IndexIVFPQDisk(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits_per_idx,
            size_t top,
            float estimate_factor,
            float prune_factor,
            const std::string& diskPath,
            const std::string& valueType = "float",
            MetricType metric = METRIC_L2);
    
    IndexIVFPQDisk();

    ~IndexIVFPQDisk();


    void set_disk_read(const std::string& diskPath); // Method to set the disk path and open read stream

    void set_disk_write(const std::string& diskPath);  // Method to set the disk path and open write stream

    void initial_location(idx_t n, const float* data); // Method to initialize the location arrays.  Maybe
                             // can put it in some other functions. But I'd
                             // prefer to explicitly calling it now. Must call it before search.
    void reorganize_vectors(idx_t n, const float* data, size_t* old_clusters,size_t* old_len);

    void load_from_offset(
            size_t list_no,
            size_t offset,
            float* original_vector);

    void load_clusters(size_t list_no, float* original_vectors);

    size_t get_cluster_location(size_t key) const {
        /*
        assert()?
        */
        return clusters[key];
    }
    size_t get_cluster_len(size_t key) const {
        /*
        assert()?
        */
        return len[key];
    }
    size_t get_vector_offset() const {
        return this->disk_vector_offset;
    }

    const std::string& get_disk_path() const {
        return this->disk_path;
    }
    
    int get_dim() const {
        return this->d;
    }

    size_t get_top() {
        return this->top;
    }

    void set_top(size_t top) {
        this->top = top;
    }

    float get_estimate_factor() const {
        return this->estimate_factor;
    }

    void set_estimate_factor(float factor) {
        this->estimate_factor = factor;
    }

    
    float get_estimate_factor_partial() const {
        return this->estimate_factor_partial;
    }

     // manually set it, or it is same with estimate_factor
    void set_estimate_factor_partial(float factor) {
        this->estimate_factor_partial = factor;
    }


    float get_prune_factor() const {
        return this->prune_factor;
    }

    void set_prune_factor(float prune_factor) {
        this->prune_factor = prune_factor;
    }

    void set_centroid_index_path(const std::string& centroid_path) {
        centroid_index_path = centroid_path;
    }

    template<typename ValueType>
    void set_value_type(){

    }

    void train_graph() override;

    void load_hnsw_centroid_index();

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params_in = nullptr ) const override;

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* keys,
            const float* coarse_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params,
            IndexIVFStats* ivf_stats) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel) const override;

    DiskIOProcessor* get_DiskIOBuildProcessor();

    DiskIOProcessor* get_DiskIOSearchProcessor() const;

    // warm up according to vectors
    int warm_up_nvec(size_t n, float* x, size_t w_nprobe, size_t nvec);

    // warm up according to nlist
    int warm_up_nlist(size_t n, float* x, size_t w_nprobe, size_t nlist);

    size_t get_code_size(){
        if(valueType == "float"){
            return d*sizeof(float);
        }else if(valueType == "uint8_t"){
            return d*sizeof(uint8_t);
        }else{
            FAISS_THROW_MSG("get_code_size() only support float & uint8_t");
            return 0;
        }
    }


//private:
// search parameters
    // 1. array to help locate where is the vector in disk. (File is reorganized
    // by clusters)
    size_t* clusters;     // ith cluster begin at clusters[i]
    size_t* len;          // with length of len[i]
    size_t top;
    float estimate_factor;
    float estimate_factor_partial;   // usually same with estimate_factor
    float prune_factor;              // prune some lists
    size_t disk_vector_offset;

    DiskInvertedListHolder diskInvertedListHolder;

    Aligned_Cluster_Info* aligned_cluster_info;
    
// build parameters
    // 2. disk operations
    size_t add_batch_num = 1;    // add a big file in batchs
    size_t actual_batch_num = 0;  // temperory varible..... delete it later

    std::string disk_path;
    //std::string disk_path_clustered;
    std::ifstream disk_data_read;
    std::ofstream disk_data_write;

    // 3. extra graph index
    std::string centroid_index_path;
    faiss::IndexHNSWFlat* centroid_index = nullptr;

    // 4. value type, must set when constructing
    std::string valueType;

};

struct IndexIVFPQDiskStats {

    size_t full_cluster_compare;
    ///< compare times in FULL load strategy
    size_t partial_cluster_compare;
    ///< compare times in PARTIAL load strategy

    size_t full_cluster_rerank;
    ///< rerank times in FULL load strategy
    size_t partial_cluster_rerank;
    ///< rerank times in PARTIAL load strategy

    size_t cached_list_access;

    size_t searched_vector_full;
    size_t searched_vector_partial;

    size_t searched_page_full;
    size_t searched_page_partial;

    std::chrono::duration<double, std::micro> memory_1_elapsed;
    std::chrono::duration<double, std::micro> memory_2_elapsed;
    std::chrono::duration<double, std::micro> disk_full_elapsed;
    std::chrono::duration<double, std::micro> disk_partial_elapsed;
    std::chrono::duration<double, std::micro> others_elapsed;
    std::chrono::duration<double, std::micro> coarse_elapsed;
    std::chrono::duration<double, std::micro> rank_elapsed;
    size_t pruned;

    IndexIVFPQDiskStats() {
        reset();
    }
    void reset();
};

// global var that collects them all
FAISS_API extern IndexIVFPQDiskStats indexIVFPQDisk_stats;

}

#endif