#include <iostream>
#include <fstream>
#include <stdexcept>
#include <faiss/impl/DiskInvertedListHolder.h>

namespace faiss{

    DiskHolder::DiskHolder(){
        this->aligned_cluster_info = nullptr;
        this->code_size = 0;
        this->disk_path = "";
        this->nlist = 0;
    }

    DiskHolder::DiskHolder(std::string& path,
                            size_t nlist,
                            size_t code_size,    // vector length sizeof(uint8_t) or sizeof(float)
                            Aligned_Cluster_Info* aligned_cluster_info)
    :disk_path(path), nlist(nlist), aligned_cluster_info(aligned_cluster_info),cached_vector(0){

    }

    void DiskHolder::set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info){
    
        FAISS_THROW_MSG("DiskHolder::set_holder() is base function.");
    }

    void DiskHolder::warm_up(const std::vector<uint64_t>& cluster_indices){
        FAISS_THROW_MSG("DiskHolder::warm_up() is base function.");
    }

    const unsigned char* DiskHolder::get_cache_data(uint64_t cluster_idx) const{
        FAISS_THROW_MSG("DiskHolder::get_cache_data() is base function.");
    }

    int DiskHolder::is_cached(uint64_t listno, uint64_t pageno = 0) const{
        FAISS_THROW_MSG("DiskHolder::is_cached() is base function.");
    }



    DiskInvertedListHolder::DiskInvertedListHolder(){};

    DiskInvertedListHolder::DiskInvertedListHolder(
                           std::string& path,
                           size_t nlist,
                           size_t code_size,
                           Aligned_Cluster_Info* aligned_cluster_info)
        : DiskHolder(path, nlist, code_size, aligned_cluster_info){
            cached_lists.resize(nlist, -1);
        }
    
    DiskInvertedListHolder::~DiskInvertedListHolder(){

    }

    void DiskInvertedListHolder::set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info){

        disk_path = path;                 
        this->nlist = nlist;
        this->code_size = code_size;
        this->aligned_cluster_info = aligned_cluster_info;    
        cached_lists.resize(nlist, -1);   
    }

    void DiskInvertedListHolder::warm_up(const std::vector<uint64_t>& cluster_indices) {
        std::ifstream file(disk_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + disk_path);
        }

        for (uint64_t list_no : cluster_indices) {
            if (list_no >= nlist) {
                throw std::out_of_range("Cluster index out of range");
            }

            Aligned_Cluster_Info* acinfo = aligned_cluster_info + list_no;

            size_t cluster_size = acinfo->page_count*PAGE_SIZE;
            size_t cluster_offset = acinfo->padding_offset;
            size_t cluster_begin = acinfo->page_start*PAGE_SIZE;

            this->cached_vector += (cluster_size - cluster_offset)/code_size;

            file.seekg(cluster_begin, std::ios::beg);

            std::vector<unsigned char> buffer(cluster_size);
            file.read(reinterpret_cast<char*>(buffer.data()), cluster_size);

            if (!file) {
                throw std::runtime_error("Error reading cluster from file");
            }

            holder.push_back(std::move(buffer));
            cached_lists[list_no] = holder.size() - 1;
        }

        file.close();
    }

    const unsigned char* DiskInvertedListHolder::get_cache_data(uint64_t cluster_idx) const {
        if (cluster_idx >= holder.size()) {
            throw std::out_of_range("Cluster index out of range");
        }
        return holder[cluster_idx].data();
    }



    DiskPageHolder::DiskPageHolder(){}

    DiskPageHolder::DiskPageHolder(
        std::string& path,
        size_t nlist,
        size_t code_size,    // vector length sizeof(uint8_t) or sizeof(float)
        Aligned_Cluster_Info* aligned_cluster_info)
        : DiskHolder(path, nlist, code_size, aligned_cluster_info){

    }

    void DiskPageHolder::set_holder(
        std::string& path, 
        size_t nlist,
        size_t code_size,
        Aligned_Cluster_Info* aligned_cluster_info) {
            
            disk_path = path;                 
            this->nlist = nlist;
            this->code_size = code_size;
            this->aligned_cluster_info = aligned_cluster_info; 
            cached_page.resize(nlist);
            // for(int i = 0; i < nlist; i++){

            // }
    }

    // Warm up the specified clusters by reading them into memory
    void DiskPageHolder::warm_up(const std::vector<uint64_t>& page_indices) {
        std::ifstream file(disk_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + disk_path);
        }

        // 将所有list的容量进行调整
        for(int i = 0; i < nlist; i++){
            Aligned_Cluster_Info* acinfo = aligned_cluster_info + i;
            cached_page[i].resize(acinfo->page_count, -1);
        }

        for (uint64_t lp_no : page_indices) {

            uint64_t list_no = this->lp_listno(lp_no);
            uint64_t page_no = this->lp_pageno(lp_no);

            if (list_no >= nlist) {
                throw std::out_of_range("DiskPageHolder::warm_up(): Cluster index out of range");
            }

            Aligned_Cluster_Info* acinfo = aligned_cluster_info + list_no;

            size_t cluster_size = acinfo->page_count*PAGE_SIZE;
            size_t cluster_offset = acinfo->padding_offset;
            size_t cluster_begin = acinfo->page_start*PAGE_SIZE;

            if(page_no > cluster_size){
                throw std::out_of_range("DiskPageHolder::warm_up(): Page index out of range");
            }

            size_t page_size = PAGE_SIZE;
            size_t page_begin = cluster_begin + page_no*PAGE_SIZE;

            this->cached_vector += page_size/code_size;

            file.seekg(page_begin, std::ios::beg);

            std::vector<unsigned char> buffer(page_size);
            file.read(reinterpret_cast<char*>(buffer.data()), page_size);

            if (!file) {
                throw std::runtime_error("Error reading cluster from file");
            }

            holder.push_back(std::move(buffer));
            cached_page[list_no][page_no] = holder.size() - 1;
        }

        file.close();
    } 

    // Access cluster data from memory
    const unsigned char* DiskPageHolder::get_cache_data(uint64_t page_idx) const {
        return this->holder[page_idx].data();
    }


}