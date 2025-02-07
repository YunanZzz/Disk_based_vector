#ifndef FAISS_INDEX_IVFPQDisk2_H
#define FAISS_INDEX_IVFPQDisk2_H

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

//#define CACHE_MODE

namespace faiss {

namespace{
    struct DiskResultHandler;

    struct UncachedList;
    typedef std::vector<UncachedList> UncachedLists;

}


class IndexIVFPQDisk2 : public IndexIVFPQ {
public:
    
    IndexIVFPQDisk2(
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
    
    IndexIVFPQDisk2();

    ~IndexIVFPQDisk2();


    void set_disk_read(const std::string& diskPath); // Method to set the disk path and open read stream

    void set_disk_write(const std::string& diskPath);  // Method to set the disk path and open write stream

    void initial_location(idx_t n, const float* data); // Method to initialize the location arrays.  Maybe
                             // can put it in some other functions. But I'd
                             // prefer to explicitly calling it now. Must call it before search.
    void reorganize_vectors(idx_t n, const float* data, size_t* old_clusters,size_t* old_len);
    void reorganize_vectors_2(idx_t n, const float* data, size_t* old_clusters,size_t* old_len);

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

    void set_estimate_factor_high_dim(float factor) {
        this->estimate_factor_high_dim = factor;
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

    void in_list_clustered_id(
        idx_t n, 
        const float* x, 
        const faiss::idx_t* coarse_idsx,
        idx_t begin_id, 
        idx_t* clustered_xids);

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

// Batch, not well developed so far
    void search_fully_qps(
            idx_t n,
            const float* x,
            idx_t k,
            idx_t nprobe,
            const idx_t* keys,
            const float* coarse_dis,
            float* distances,
            idx_t* labels,
            DiskIOProcessor* diskIOprocessor,
            DiskResultHandler* heap_handler) const;

    void search_fully(
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
            ) const;


    void search_partially(
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
            ) const;

    void search_o(
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
        ) const;
    
    void search_uncached(
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
            ) const;

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

    DiskIOProcessor* get_DiskIOBuildProcessor();

    DiskIOProcessor* get_DiskIOSearchProcessor() const;

    void set_disk(int n_threads);
    void end_disk(int n_threads);
    // warm up according to vectors
    int warm_up_nvec(size_t n, float* x, size_t w_nprobe, size_t nvec);

    // warm up according to nlist
    int warm_up_nlist(size_t n, float* x, size_t w_nprobe, size_t nlist);

    int warm_up_page(size_t n, float* x, size_t w_nprobe, size_t npage);

    int warm_up_index_info(size_t n, float* x, size_t w_nprobe, size_t warm_list);

    size_t get_code_size(){
        if(valueType == "float"){
            return d*sizeof(float);
        }else if(valueType == "uint8"){
            return d*sizeof(uint8_t);
        }else{
            FAISS_THROW_MSG("get_code_size() only support float & uint8_t");
            return 0;
        }
    }


    int set_select_lists_mode(std::string select_lists_path = ""){
        // 存放在一个文件里面
        select_lists = true;
        // this->select_lists_path = select_lists_path;
        this->select_lists_path = disk_path;

        cached_list_info = new bool[nlist];

        return 0;
    }

    

// Search parameters
    // 1. array to help locate where is the vector in disk. (File is reorganized
    // by clusters)
    size_t* clusters;     // ith cluster begin at clusters[i]
    size_t* len;          // with length of len[i]
    size_t top;
    float estimate_factor;
    float estimate_factor_partial;   // usually same with estimate_factor
    float estimate_factor_high_dim = 1;   // use in very high dim vector
    float prune_factor;              // prune some lists
    size_t disk_vector_offset;

    mutable std::vector<std::unique_ptr<DiskIOProcessor>> diskIOprocessors;

    DiskInvertedListHolder diskInvertedListHolder;

    Aligned_Cluster_Info* aligned_cluster_info;
    

// Build parameters
    // 1. build all in memory, only support build by float!
    ArrayInvertedLists* build_invlists;

    // 2. disk operations
    size_t add_batch_num = 2;    // add a big file in batchs
    size_t actual_batch_num = 0;  // temperory varible..... delete it later
    bool reorganize_lists = true;

    //instead reading invlist from IndexRead, reading from compress_list 
    bool select_lists = false;
    std::string select_lists_path = "";
    Aligned_Invlist_Info* aligned_inv_info;

    std::string disk_path;
    //std::string disk_path_clustered;
    std::ifstream disk_data_read;
    std::ofstream disk_data_write;

    // 3. extra graph index
    std::string centroid_index_path;
    faiss::IndexHNSWFlat* centroid_index = nullptr;

    // 4. value type, must set when constructing
    std::string valueType;

// Cache parameter
    // cached list info(pq + index + map)
    bool* cached_list_info;

};

struct IndexIVFPQDisk2Stats {

    size_t full_cluster_compare;
    ///< compare times in FULL load strategy
    size_t partial_cluster_compare;
    ///< compare times in PARTIAL load strategy

    size_t full_cluster_rerank;
    ///< rerank times in FULL load strategy
    size_t partial_cluster_rerank;
    ///< rerank times in PARTIAL load strategy

    size_t cached_list_access;
    size_t cached_vector_access;

    size_t pq_list_full;
    size_t pq_list_partial;

    size_t searched_vector_full;
    size_t searched_vector_partial;

    size_t searched_page_full;
    size_t searched_page_partial;

    size_t requests_full;
    size_t requests_partial;

    std::chrono::duration<double, std::micro> memory_1_elapsed;
    std::chrono::duration<double, std::micro> memory_2_elapsed;
    std::chrono::duration<double, std::micro> memory_3_elapsed;
    std::chrono::duration<double, std::micro> disk_full_elapsed;
    std::chrono::duration<double, std::micro> disk_partial_elapsed;
    std::chrono::duration<double, std::micro> others_elapsed;
    std::chrono::duration<double, std::micro> coarse_elapsed;
    std::chrono::duration<double, std::micro> rank_elapsed;  
    std::chrono::duration<double, std::micro> rerank_elapsed;
    std::chrono::duration<double, std::micro> pq_elapsed;
    std::chrono::duration<double, std::micro> cached_calculate_elapsed;
    std::chrono::duration<double, std::micro> delete_elapsed;

    std::chrono::duration<double, std::micro> memory_uncache_elapsed;
    std::chrono::duration<double, std::micro> rank_uncache_elapsed;  
    std::chrono::duration<double, std::micro> disk_uncache_calc_elapsed;
    std::chrono::duration<double, std::micro> disk_uncache_info_elapsed;
    std::chrono::duration<double, std::micro> pq_uncache_elapsed;


    

    

    size_t searched_lists;
    size_t pruned;

    IndexIVFPQDisk2Stats() {
        reset();
    }
    void reset();
};

// global var that collects them all
FAISS_API extern IndexIVFPQDisk2Stats indexIVFPQDisk2_stats;

}

#endif