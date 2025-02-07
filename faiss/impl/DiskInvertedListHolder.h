#ifndef FAISS_INVLIST_HOLDER_H
#define FAISS_INVLIST_HOLDER_H

#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/DiskIOStructure.h>


namespace faiss {

struct DiskHolder{
    std::string disk_path;
    size_t code_size;           // code_size is calculated by B.  eg 128D float--> 128*4B,  128D uint8_t--> 128B  
    size_t nlist;
    size_t cached_vector;
    Aligned_Cluster_Info* aligned_cluster_info;

    DiskHolder();

    DiskHolder(std::string& path,
                size_t nlist,
                size_t code_size,    // vector length sizeof(uint8_t) or sizeof(float)
                Aligned_Cluster_Info* aligned_cluster_info);
    
    virtual void set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info);

    virtual void warm_up(const std::vector<uint64_t>& cluster_indices); 

    virtual const unsigned char* get_cache_data(uint64_t cluster_idx) const;

    virtual int is_cached(uint64_t listno, uint64_t pageno) const;

    //used in warm_up() so that we can override with listno << 32 | pageno
    inline uint64_t lp_build(uint64_t list_id, uint64_t offset) {
        return list_id << 32 | offset;
    }

    inline uint64_t lp_listno(uint64_t lo) {
        return lo >> 32;
    }

    inline uint64_t lp_pageno(uint64_t lo) {
        return lo & 0xffffffff;
    }
};



struct DiskInvertedListHolder : DiskHolder {
    std::vector<std::vector<unsigned char>> holder; // Vector to hold clusters data in memory
    //std::unordered_map<uint64_t, int> cache_lists;
    std::vector<int> cached_lists;
    
    DiskInvertedListHolder();

    DiskInvertedListHolder(std::string& path,
                           size_t nlist,
                           size_t code_size,    // vector length sizeof(uint8_t) or sizeof(float)
                           Aligned_Cluster_Info* aligned_cluster_info);
    
    ~DiskInvertedListHolder();

    void set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info) override;

    // Warm up the specified clusters by reading them into memory
    void warm_up(const std::vector<uint64_t>& cluster_indices) override; 

    // Access cluster data from memory
    const unsigned char* get_cache_data(uint64_t cluster_idx) const override;

    int is_cached(uint64_t listno, uint64_t page_no = 0) const override{
        return cached_lists[listno];
        // if (cached_lists[listno] != -1) {
        //     return it->second;  // Return the position in holder
        // } else {
        //     return -1;  // Return -1 if not cached
        // }
    }
};



struct DiskPageHolder : DiskHolder{
    std::vector<std::vector<unsigned char>> holder; // Vector to hold clusters data in memory
    //std::unordered_map<uint64_t, int> cache_lists;
    std::vector<std::vector<int>> cached_page;

    DiskPageHolder();

    DiskPageHolder(std::string& path,
                    size_t nlist,
                    size_t code_size,    // vector length sizeof(uint8_t) or sizeof(float)
                    Aligned_Cluster_Info* aligned_cluster_info);

    void set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info) override;

    // Warm up the specified clusters by reading them into memory
    void warm_up(const std::vector<uint64_t>& page_indices) override; 

    // Access cluster data from memory
    const unsigned char* get_cache_data(uint64_t page_idx) const override;

    int is_cached(uint64_t listno, uint64_t page_no) const override{
        return cached_page[listno][page_no];
    }
};


} // namespace faiss

#endif
