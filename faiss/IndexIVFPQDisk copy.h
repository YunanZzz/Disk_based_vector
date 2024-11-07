#ifndef FAISS_INDEX_IVFPQDisk_H
#define FAISS_INDEX_IVFPQDisk_H

#include <vector>

#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/AlignedTable.h>

#include <string>
#include <fstream>   


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
            MetricType metric = METRIC_L2);
    
    IndexIVFPQDisk();

    ~IndexIVFPQDisk();


    void set_disk_read(const std::string& diskPath); // Method to set the disk path and open read stream

    void set_disk_write(const std::string& diskPath);  // Method to set the disk path and open write stream

    void initial_location(float* data); // Method to initialize the location arrays.  Maybe
                             // can put it in some other functions. But I'd
                             // prefer to explicitly calling it now. Must call it before search.
    void reorganize_vectors(float* data);

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


//private:
    // 1. array to help locate where is the vector in disk. (File is reorganized
    // by clusters)
    size_t* clusters;     // ith cluster begin at clusters[i]
    size_t* len;          // with length of len[i]
    size_t top;
    float estimate_factor;
    float estimate_factor_partial;   // usually same with estimate_factor
    float prune_factor;              // prune some lists
    size_t disk_vector_offset;
    

    // 2. disk operations
    std::string disk_path;
    std::ifstream disk_data_read;
    std::ofstream disk_data_write;
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
    

    double full_cluster_time;  // Time spent in I/O for FULL load strategy (seconds)
    double partial_cluster_time; // Time spent in I/O for Partial load strategy (seconds)
    double PQScan_time;
    double nprobeSearch_time;
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