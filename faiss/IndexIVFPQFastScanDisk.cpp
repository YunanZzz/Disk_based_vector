#include <faiss/IndexIVFPQFastScanDisk.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <set>

#include <omp.h>

#include <memory>

#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/code_distance/code_distance.h>
#include <faiss/utils/distances.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/LookupTableScaler.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/quantize_lut.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/Heap.h>
#include <iostream>


namespace faiss {

using namespace simd_result_handlers;

IndexIVFPQFastScanDisk::IndexIVFPQFastScanDisk(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        size_t top,
        float estimate_factor,
        float prune_factor,
        const std::string& diskPath,
        MetricType metric)
        : IndexIVFPQFastScan(quantizer, d, nlist, M, nbits_per_idx, metric),
          top(top),
          estimate_factor(estimate_factor),
          prune_factor(prune_factor),
          disk_path(diskPath),
          disk_vector_offset(d * sizeof(float)) {
    estimate_factor_partial = estimate_factor;
    clusters = new size_t[nlist];
    len = new size_t[nlist];
}

IndexIVFPQFastScanDisk::IndexIVFPQFastScanDisk(){}

IndexIVFPQFastScanDisk::~IndexIVFPQFastScanDisk() {
    delete[] clusters;
    delete[] len;
}

void IndexIVFPQFastScanDisk::initial_location(float* data) {
    if (!invlists) {
        throw std::runtime_error("invlists is not initialized.");
    }

    // Cast invlists to BlockInvertedLists to access the underlying data
    BlockInvertedLists* block_invlists =
            dynamic_cast<BlockInvertedLists*>(invlists);
    if (!block_invlists) {
        throw std::runtime_error("invlists is not of type BlockInvertedLists.");
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < nlist; ++i) {
        clusters[i] = current_offset;
        len[i] = block_invlists->ids[i].size();
        current_offset += len[i];
    }
    if (verbose) {
        printf("Cluster info initialized!");
    }
    // reorg it at last, and save it to file.
    reorganize_vectors(data);
}

// Method to reorganize vectors based on clustering
void IndexIVFPQFastScanDisk::reorganize_vectors(float* data) {
    // if (!disk_data_read.is_open()) {
    //     throw std::runtime_error("Disk read stream is not open.");
    // }
    //

    //// Read all vectors into memory
    // size_t total_size = ntotal * d * sizeof(float);
    // std::unique_ptr<float[]> data(new float[ntotal * d]);
    // disk_data_read.read(reinterpret_cast<char*>(data.get()), total_size);
    // disk_data_read.close();

    // Create a new disk path with ".clustered" suffix
    std::string new_disk_path = disk_path + ".clustered";
    disk_path = new_disk_path;
    set_disk_write(new_disk_path);

    // Cast invlists to ArrayInvertedLists to access the underlying data
    BlockInvertedLists* block_invlists =
            dynamic_cast<BlockInvertedLists*>(invlists);
    if (!block_invlists) {
        throw std::runtime_error("invlists is not of type BlockInvertedLists.");
    }

    // Reorganize vectors and write to the new file
    for (size_t i = 0; i < nlist; ++i) {
        size_t offset = clusters[i];
        size_t count = len[i];
        for (size_t j = 0; j < count; ++j) {
            idx_t id = block_invlists->ids[i][j];
            const float* vector = &data[id * d];
            disk_data_write.write(
                    reinterpret_cast<const char*>(vector), d * sizeof(float));
        }
    }

    disk_data_write.close();

    if (verbose) {
        printf("Vectors reorganized and written to %s\n",
               new_disk_path.c_str());
    }
}

// Method to set the disk path and open read stream
void IndexIVFPQFastScanDisk::set_disk_read(const std::string& diskPath) {
    disk_path = diskPath;
    if (disk_data_read.is_open()) {
        disk_data_read.close();
    }
    disk_data_read.open(disk_path, std::ios::binary);
    if (!disk_data_read.is_open()) {
        throw std::runtime_error(
                "IndexIVFPQDisk: Failed to open disk file for reading");
    }
}

// Method to set the disk path and open write stream
void IndexIVFPQFastScanDisk::set_disk_write(const std::string& diskPath) {
    disk_path = diskPath;
    if (disk_data_write.is_open()) {
        disk_data_write.close();
    }
    disk_data_write.open(disk_path, std::ios::binary);
    if (!disk_data_write.is_open()) {
        throw std::runtime_error(
                "IndexIVFPQDisk: Failed to open disk file for writing");
    }
}

void IndexIVFPQFastScanDisk::load_from_offset(
        size_t list_no,
        size_t offset,
        float* original_vector) {
    if (!disk_data_read.is_open()) {
        throw std::runtime_error("Disk read stream is not open.");
    }

    // 检查偏移量是否有效
    assert(offset < len[list_no]);

    // 计算全局偏移量
    size_t global_offset = (clusters[list_no] + offset) * d * sizeof(float);
    disk_data_read.seekg(global_offset, std::ios::beg);

    // 读取向量数据
    disk_data_read.read(
            reinterpret_cast<char*>(original_vector), d * sizeof(float));

    if (disk_data_read.fail()) {
        throw std::runtime_error("Failed to read vector from disk.");
    }
}

void IndexIVFPQFastScanDisk::load_clusters(
        size_t list_no,
        float* original_vectors) {
    if (!disk_data_read.is_open()) {
        throw std::runtime_error("Disk read stream is not open.");
    }

    // 计算全局偏移量
    size_t global_offset = clusters[list_no] * d * sizeof(float);
    disk_data_read.seekg(global_offset, std::ios::beg);

    // 读取所有向量数据
    disk_data_read.read(
            reinterpret_cast<char*>(original_vectors),
            d * sizeof(float) * len[list_no]);

    if (disk_data_read.fail()) {
        throw std::runtime_error("Failed to read vectors from disk.");
    }
}

/*-----------------------modified from IVFPQFastScan---------------------------*/


namespace {

template <class C>
ResultHandlerCompare<C,false>* make_knn_handler_fixC(
        int impl,
        idx_t n,
        idx_t k,
        float* distances,
        idx_t* labels) {
    //using HeapHC = HeapHandler<C, with_id_map>;
    //using ReservoirHC = ReservoirHandler<C, with_id_map>;
   // using SingleResultHC = SingleResultHandler<C, with_id_map>;
    using SingleQueryResultHC = SingleQueryResultHandler<C, false>;
    return new SingleQueryResultHC(0, k, distances, labels);
}

SIMDResultHandlerToFloat* make_knn_handler(
        bool is_max,
        int impl,
        idx_t n,
        idx_t k,
        float* distances,
        idx_t* labels) {
    if (is_max) {
        return make_knn_handler_fixC<CMax<uint16_t, int64_t>>(
                impl, n, k, distances, labels);
    } else {
        return make_knn_handler_fixC<CMin<uint16_t, int64_t>>(
                impl, n, k, distances, labels);
    }
}

template <class C>
struct KnnSearchResults {
    size_t k;

    float* heap_sim;
    idx_t* heap_ids;
    size_t nup;

    KnnSearchResults(size_t k, float* heap_sim, idx_t* heap_ids)
            : k(k), heap_sim(heap_sim), heap_ids(heap_ids) {
        //heap_heapify<C>(k, heap_sim, heap_ids);
        nup = 0;
    }

    inline void add(idx_t id, float dis) {
        if (C::cmp(heap_sim[0], dis)) {
            heap_replace_top<C>(k, heap_sim, heap_ids, dis, id);
            nup++;
        }
    }
};

using CoarseQuantized = IndexIVFFastScan::CoarseQuantized;

struct CoarseQuantizedWithBuffer : CoarseQuantized {
    explicit CoarseQuantizedWithBuffer(const CoarseQuantized& cq)
            : CoarseQuantized(cq) {}

    bool done() const {
        return ids != nullptr;
    }

    std::vector<idx_t> ids_buffer;
    std::vector<float> dis_buffer;

    void quantize(const Index* quantizer, idx_t n, const float* x) {
        dis_buffer.resize(nprobe * n);
        ids_buffer.resize(nprobe * n);
        quantizer->search(n, x, nprobe, dis_buffer.data(), ids_buffer.data());
        dis = dis_buffer.data();
        ids = ids_buffer.data();
    }
};

struct CoarseQuantizedSlice : CoarseQuantizedWithBuffer {
    size_t i0, i1;
    CoarseQuantizedSlice(const CoarseQuantized& cq, size_t i0, size_t i1)
            : CoarseQuantizedWithBuffer(cq), i0(i0), i1(i1) {
        if (done()) {
            dis += nprobe * i0;
            ids += nprobe * i0;
        }
    }

    void quantize_slice(const Index* quantizer, const float* x) {
        quantize(quantizer, i1 - i0, x + quantizer->d * i0);
    }
};

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

int compute_search_nslice(
        const IndexIVFFastScan* index,
        size_t n,
        size_t nprobe) {
    int nslice;
    if (n <= omp_get_max_threads()) {
        nslice = n;
    } else if (index->lookup_table_is_3d()) {
        // make sure we don't make too big LUT tables
        size_t lut_size_per_query = index->M * index->ksub * nprobe *
                (sizeof(float) + sizeof(uint8_t));

        size_t max_lut_size = precomputed_table_max_bytes;
        // how many queries we can handle within mem budget
        size_t nq_ok = std::max(max_lut_size / lut_size_per_query, size_t(1));
        nslice = roundup(std::max(size_t(n / nq_ok), size_t(1)), omp_get_max_threads());
    } else {
        // LUTs unlikely to be a limiting factor
        nslice = omp_get_max_threads();
    }
    return nslice;
}

} // namespace

void IndexIVFPQFastScanDisk::search_dispatch_implem(
        idx_t n,
        const float* x,
        idx_t k_r,
        float* distances_result,
        idx_t* labels_result,
        const CoarseQuantized& cq_in,
        const NormTableScaler* scaler,
        const IVFSearchParameters* params) const {
    const idx_t nprobe = params ? params->nprobe : this->nprobe;
    const IDSelector* sel = (params) ? params->sel : nullptr;
    const SearchParameters* quantizer_params =
            params ? params->quantizer_params : nullptr;
    
    bool is_max = !is_similarity_metric(metric_type);
    using RH = SIMDResultHandlerToFloat;

    if (n == 0) {
        return;
    }

    idx_t k = k_r * this->assign_replicas;
    // k=800;
    printf("ivfpqfs::k=%d\n",k);
    std::unique_ptr<idx_t[]> del1(new idx_t[n * k]);
    std::unique_ptr<float[]> del2(new float[n * k]);
    idx_t* labels = del1.get();
    float* distances = del2.get();

    // actual implementation used
    int impl = implem;

     if (impl == 0) {
        if (bbs == 32) {
            impl = 12;
        } else {
            impl = 10;
        }
        if (k > 20) { // use reservoir rather than heap
            impl++;
        }
    }

    bool multiple_threads =
            n > 1 && impl >= 10 && impl <= 13 && omp_get_max_threads() > 1;

    CoarseQuantizedWithBuffer cq(cq_in);
    size_t ndis = 0, nlist_visited = 0;
    cq.nprobe = nprobe;
    if(!multiple_threads){
        cq.quantize(quantizer, n, x);
        int max_buffer_size = 0;    
        std::unique_ptr<RH> handler(make_knn_handler(is_max, impl, 1, max_buffer_size, nullptr, nullptr));
        search_implem_10(
                n,
                k,
                x,
                *handler.get(),
                cq,
                &ndis,
                &nlist_visited,
                distances,
                labels,
                scaler);
    }else{
        int nslice = compute_search_nslice(this, n, cq.nprobe);

#pragma omp parallel for reduction(+ : ndis, nlist_visited)
            for (int slice = 0; slice < nslice; slice++) {
                idx_t i0 = n * slice / nslice;
                idx_t i1 = n * (slice + 1) / nslice;
                float* dis_i = distances + i0 * k;
                idx_t* lab_i = labels + i0 * k;
                CoarseQuantizedSlice cq_i(cq, i0, i1);
                if (!cq_i.done()) {
                    cq_i.quantize_slice(quantizer, x);
                }
                int max_buffer_size = 0; 
                std::unique_ptr<RH> handler(make_knn_handler(is_max, impl, 1, max_buffer_size, nullptr, nullptr));
                // clang-format off
                search_implem_10(
                        i1 - i0, k, x + i0 * d, *handler.get(),
                        cq_i, &ndis, &nlist_visited, dis_i, lab_i, scaler);
                
                // clang-format on
            }

    }
    indexIVFPQFastScanDisk_stats.nq += n;
    indexIVFPQFastScanDisk_stats.ndis += ndis;
    indexIVFPQFastScanDisk_stats.nlist += nlist_visited;

    auto time_start = std::chrono::high_resolution_clock::now();      // time begin

    for(idx_t ii = 0; ii < n;ii++){
        idx_t begin_r = ii*k_r;
        idx_t begin = ii*k;
        idx_t limit = 0;
        for(idx_t jj = 0; jj < k; jj++){

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
    indexIVFPQFastScanDisk_stats.rank_elapsed += time_end - time_start;
}

namespace {

bool skip_replica(int k, idx_t id, idx_t* ids){
    for(int i = 0; i<k; i++){
        if(id == ids[i])
            return true;
    }
    return false;
}

// template <class C>
// void disk_rerank(
//         std::ifstream& disk_data,
//         int k,
//         size_t cluster_begin,
//         size_t len,
//         size_t single_offset,
//         int D,
//         float factor,
//         float factor_partial,
//         const float* query,
//         float* list_sim,
//         idx_t* list_ids,
//         float* heap_sim,
//         idx_t* heap_ids,
//         const idx_t* id_map,
//         Load_Strategy load_strategy) {
//     KnnSearchResults<C> res(k, heap_sim, heap_ids);  // ?
//     auto time_start = std::chrono::high_resolution_clock::now();      // time begin
//     if (load_strategy == FULLY) {
//         std::vector<float> vec(D * len);
//         size_t offset = cluster_begin * single_offset;
//         disk_data.seekg(offset, std::ios::beg);
//         disk_data.read(reinterpret_cast<char*>(vec.data()), D * len * sizeof(float));
//         float* cluster_data = vec.data();

//         auto time_end = std::chrono::high_resolution_clock::now();       // time end
//         indexIVFPQFastScanDisk_stats.disk_full_elapsed += time_end - time_start;
//         time_start = std::chrono::high_resolution_clock::now();      // time begin

//         for (size_t i = 0; i < len; i++) {
 
//             if (list_sim[i] < heap_sim[0] * factor) {
//                 float distance =
//                         fvec_L2sqr(query, cluster_data + list_ids[i] * D, D);
//                 res.add(id_map[list_ids[i]], distance);
//                 indexIVFPQFastScanDisk_stats.full_cluster_rerank++;
//             }
//             indexIVFPQFastScanDisk_stats.full_cluster_compare++;
//         }

//         time_end = std::chrono::high_resolution_clock::now();       // time end
//         indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;

//     } else {
//         std::vector<float> vec(D);
//         size_t offset = cluster_begin * single_offset;
//         auto time_end = std::chrono::high_resolution_clock::now();       // time end
//         for (size_t i = 0; i < len; i++) {            
//             if (list_sim[i] < heap_sim[0] * factor_partial) {

//                 time_start = std::chrono::high_resolution_clock::now();      // time begin

//                 disk_data.seekg(offset, std::ios::beg);
//                 disk_data.seekg(list_ids[i] * single_offset, std::ios::cur);
//                 disk_data.read(reinterpret_cast<char*>(vec.data()), D * sizeof(float));

//                 time_end = std::chrono::high_resolution_clock::now(); // time end
//                 indexIVFPQFastScanDisk_stats.disk_partial_elapsed += time_end - time_start;
//                 time_start = std::chrono::high_resolution_clock::now();      // time begin

//                 float distance = fvec_L2sqr(query, vec.data(), D);
//                 res.add(id_map[list_ids[i]], distance);
//                 indexIVFPQFastScanDisk_stats.partial_cluster_rerank++;

//                 time_end = std::chrono::high_resolution_clock::now(); // time end
//                 indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;
//             }

//             indexIVFPQFastScanDisk_stats.partial_cluster_compare++;
//         }
//     }
// }

template <class C>
void disk_rerank(
        FILE* disk_data, // 修改为 FILE*
        int k,
        size_t cluster_begin,
        size_t len,
        size_t single_offset,
        int D,
        float factor,
        float factor_partial,
        const float* query,
        float* list_sim,
        idx_t* list_ids,
        float* heap_sim,
        idx_t* heap_ids,
        const idx_t* id_map,
        Load_Strategy load_strategy) {
    KnnSearchResults<C> res(k, heap_sim, heap_ids);
    auto time_start = std::chrono::high_resolution_clock::now(); // time begin
    if (load_strategy == FULLY) {
        std::vector<float> vec(D * len);
        size_t offset = cluster_begin * single_offset;
        if (fseek(disk_data, offset, SEEK_SET) != 0) {
                    throw std::runtime_error("Failed to seek in disk file");
                }
        fread(vec.data(), sizeof(float), D * len, disk_data); 
        float* cluster_data = vec.data();
        auto time_end = std::chrono::high_resolution_clock::now(); // time end
        indexIVFPQFastScanDisk_stats.disk_full_elapsed += time_end - time_start;

        time_start = std::chrono::high_resolution_clock::now(); // time begin
        for (size_t i = 0; i < len; i++) {
            if (list_sim[i] < heap_sim[0] * factor) {
                float distance = fvec_L2sqr(query, cluster_data + list_ids[i] * D, D);
                res.add(id_map[list_ids[i]], distance);
                indexIVFPQFastScanDisk_stats.full_cluster_rerank++;
            }
            indexIVFPQFastScanDisk_stats.full_cluster_compare++;
        }
        time_end = std::chrono::high_resolution_clock::now(); // time end
        indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;
        
    } else {
        std::vector<float> vec(D);
        size_t offset = cluster_begin * single_offset;
        auto time_end = std::chrono::high_resolution_clock::now(); // time end
        indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;

        for (size_t i = 0; i < len; i++) {            
            if (list_sim[i] < heap_sim[0] * factor_partial) {
                time_start = std::chrono::high_resolution_clock::now(); // time begin
                fseek(disk_data, offset, SEEK_SET); 
                fseek(disk_data, list_ids[i] * single_offset, SEEK_CUR); 
                fread(vec.data(), sizeof(float), D, disk_data); 
                time_end = std::chrono::high_resolution_clock::now(); // time end
                indexIVFPQFastScanDisk_stats.disk_partial_elapsed += time_end - time_start;
                
                time_start = std::chrono::high_resolution_clock::now(); // time begin
                float distance = fvec_L2sqr(query, vec.data(), D);
                res.add(id_map[list_ids[i]], distance);
                indexIVFPQFastScanDisk_stats.partial_cluster_rerank++;
                time_end = std::chrono::high_resolution_clock::now(); // time end
                indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;
            }

            indexIVFPQFastScanDisk_stats.partial_cluster_compare++;
        }
        
    }
}

} // namespace


//#define USE_IFSTREAM
#define USE_FREAD

void IndexIVFPQFastScanDisk::search_implem_10(
        idx_t n,
        idx_t k,
        const float* x,
        SIMDResultHandlerToFloat& handler,
        const CoarseQuantized& cq,
        size_t* ndis_out,
        size_t* nlist_out,
        float* distances,
        idx_t* ids,
        const NormTableScaler* scaler) const {
    
    auto time_start = std::chrono::high_resolution_clock::now();      // time begin

    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::unique_ptr<float[]> normalizers(new float[2 * n]);
    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    compute_LUT_uint8(n, x, cq, dis_tables, biases, normalizers.get());

    bool single_LUT = !lookup_table_is_3d();
    size_t ndis = 0;
    int qmap1[1];

    handler.q_map = qmap1;
    // handler_l.begin(skip & 16 ? nullptr : normalizers.get());
    float* normalizers_begin = normalizers.get();
    size_t nprobe = cq.nprobe;

    size_t single_offset = this->get_vector_offset();
    int D = this->get_dim();
    const float refine_factor = this->estimate_factor;
    const float factor_partial = this->estimate_factor_partial;
 
    /* open file here */
#ifdef USE_IFSTREAM
    std::ifstream disk_data;
    disk_data.open(this->disk_path, std::ios::binary);
#endif
#ifdef USE_FREAD
    FILE* disk_data = fopen(this->disk_path.c_str(), "rb"); 
#endif

    float* current_dis;
    idx_t* current_ids;

    auto time_end = std::chrono::high_resolution_clock::now();       // time end
    indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;
   
    for (idx_t i = 0; i < n; i++) {

        time_start = std::chrono::high_resolution_clock::now();      // time begin

        const uint8_t* LUT = nullptr;
        qmap1[0] = i;
        // initialize result set
        current_dis = distances + i * k;
        current_ids = ids + i * k;
        if (this->metric_type==METRIC_L2)
            heap_heapify<HeapForL2>(k, current_dis, current_ids);
        else
            heap_heapify<HeapForIP>(k, current_dis, current_ids);

        // acquire current query
        const float* current_query = x + D * i;

        if (single_LUT) {
            LUT = dis_tables.get() + i * dim12;
        }

        time_end = std::chrono::high_resolution_clock::now();       // time end
        indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;

        for (idx_t j = 0; j < nprobe; j++) {

            time_start = std::chrono::high_resolution_clock::now();      // time begin

            handler.begin(skip & 16 ? nullptr : normalizers_begin + 2 * i);

            size_t ij = i * nprobe + j;
            if (!single_LUT) {
                LUT = dis_tables.get() + ij * dim12;
            }
            if (biases.get()) {
                handler.dbias = biases.get() + ij;
            }

            idx_t list_no = cq.ids[ij];
            if (list_no < 0) {
                continue;
            }
            size_t ls = invlists->list_size(list_no);
            if (ls == 0) {
                continue;
            }

            std::vector<idx_t> local_ids(ls);
            std::vector<float> local_dis(ls);


             // TODO  set total
            handler.set_total(ls, local_ids.data(), local_dis.data());

            InvertedLists::ScopedCodes codes(invlists, list_no);
            InvertedLists::ScopedIds ids(invlists, list_no);

            handler.ntotal = ls;
            handler.id_map = ids.get();

            time_end = std::chrono::high_resolution_clock::now();       // time end
            indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;
            time_start = std::chrono::high_resolution_clock::now();      // time begin
            pq4_accumulate_loop(
                    1,
                    roundup(ls, bbs),
                    bbs,
                    M2,
                    codes.get(),
                    LUT,
                    handler,
                    scaler);
            
            time_end = std::chrono::high_resolution_clock::now();       // time end
            indexIVFPQFastScanDisk_stats.memory_pq_elapsed += time_end - time_start;
            time_start = std::chrono::high_resolution_clock::now();      // time begin
            handler.end();
            ndis++;
            Load_Strategy load_strategy;
            if (j < this->top)
                load_strategy = FULLY;
            else
                load_strategy = PARTIALLY;
            time_end = std::chrono::high_resolution_clock::now();       // time end
            indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;

            /*---------------------DISK Operations----------------------------*/


            size_t cluster_begin = this->get_cluster_location(list_no);
            size_t len = this->get_cluster_len(list_no);
            if (this->metric_type == METRIC_L2)
                disk_rerank<HeapForL2>(
                    disk_data,
                    k,
                    cluster_begin,
                    len,
                    single_offset,
                    D,
                    refine_factor, 
                    factor_partial,
                    current_query,
                    local_dis.data(),
                    local_ids.data(),
                    current_dis,
                    current_ids,
                        ids.get(),
                    load_strategy);
            else
                disk_rerank<HeapForIP>(
                    disk_data,
                    k,
                    cluster_begin,
                    len,
                    single_offset,
                    D,
                    refine_factor,
                    factor_partial,
                    current_query,
                    local_dis.data(),
                    local_ids.data(),
                    current_dis,
                    current_ids,
                        ids.get(),
                    load_strategy);
            /*-------------------------DISK Operations---------------------------------------*/
        }
        time_start = std::chrono::high_resolution_clock::now();      // time begin
        if (this->metric_type==METRIC_L2)
            heap_reorder<HeapForL2>(k, current_dis, current_ids);
        else
            heap_reorder<HeapForIP>(k, current_dis, current_ids);
        time_end = std::chrono::high_resolution_clock::now();       // time end
        indexIVFPQFastScanDisk_stats.memory_2_elapsed += time_end - time_start;

    }
    *ndis_out = ndis;
    *nlist_out = nlist;
}

/* statistics */

IndexIVFPQFastScanDiskStats indexIVFPQFastScanDisk_stats;

void IndexIVFPQFastScanDiskStats::reset() {
    memset(this, 0, sizeof(*this));
}


} // namespace faiss