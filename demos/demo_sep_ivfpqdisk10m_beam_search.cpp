
#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQDisk2.h>

#include <faiss/index_io.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <omp.h>

using idx_t = faiss::idx_t;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float* load_and_convert_to_float(const char* fname, size_t* d_out, size_t* n_out, size_t batch_size = 1000000) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fname);
        perror("");
        abort();
    }

    // 读取向量维度
    int d;
    fread(&d, sizeof(int), 1, f);
    assert((d > 0 && d < 1000000) || !"Unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    // 检查文件大小是否符合预期
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;

    // 每个向量包含：1个int (维度指示符) + d个uint8数据
    size_t per_vector_size = sizeof(int) + d * sizeof(uint8_t);
    assert(sz % per_vector_size == 0 || !"Weird file size");

    size_t n = sz / per_vector_size; // 向量数量
    *d_out = d;
    *n_out = n;

    // 分配最终存储的float数组
    float* result = new float[n * d];

    size_t vectors_left = n;
    size_t offset = 0; // 用于在 result 中定位当前写入位置

    while (vectors_left > 0) {
        size_t current_batch_size = std::min(batch_size, vectors_left);

        // 临时缓冲区：每个向量前有1个int头，后接d个uint8数据
        std::vector<uint8_t> temp(per_vector_size * current_batch_size);
        size_t nr = fread(temp.data(), sizeof(uint8_t), temp.size(), f);
        std::cout << "vector size:"<<  current_batch_size <<" nr:" << nr << "  temp size:" << temp.size() << std::endl;
        assert(nr == temp.size() || !"Could not read batch");

        // 转换并存入 result
        for (size_t i = 0; i < current_batch_size; i++) {
            // 跳过维度指示符 (1个int)
            const uint8_t* data_ptr = temp.data() + i * per_vector_size + sizeof(int);

            // 将 uint8 转换为 float
            for (size_t j = 0; j < d; j++) {
                result[offset + i * d + j] = static_cast<float>(data_ptr[j]);
            }
        }

        offset += current_batch_size * d; // 更新写入偏移
        vectors_left -= current_batch_size;
    }

    fclose(f);
    return result;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

template<typename T>
void disk_build(const char* base_filepath, int nb, int d, int ratio, 
                const std::string& index_store_path,    
                const std::string& disk_store_path,
                const std::string& centroid_index_path) {
    double t0 = elapsed();
    size_t dd; // dimension
    size_t nt; // number of vectors

    float* xb = load_and_convert_to_float(base_filepath, &dd, &nt);

    std::vector<float> trainvecs(nb / ratio * d);
    srand(static_cast<int>(time(0)));
    for (int i = 0; i < nb / ratio; i++) {
        int rng = (rand() % (nb + 1));
        for (int j = 0; j < d; j++) {
            trainvecs[d * i + j] = xb[rng * d + j];
        }
    }

    int nlist = 40000;
    int m = 16;
    int nbits = 8;
    int top_clusters = 5;
    float estimate_factor = 1.2;
    float prune_factor = 1.9;
    omp_set_num_threads(12);
    int M = 32;
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFPQDisk2 index(&quantizer, d, nlist, m, nbits, top_clusters, 
                                estimate_factor, prune_factor, index_store_path, "uint8");
    index.set_assign_replicas(2);
    index.set_centroid_index_path(centroid_index_path);
    index.set_select_lists_mode();
    index.verbose = false;

    printf("[%.3f s] IndexIVFPQ_disk start to train\n", elapsed() - t0);
    index.train(nb / ratio, trainvecs.data());
    printf("[%.3f s] IndexIVFPQ_disk train finished\n", elapsed() - t0);

    printf("[%.3f s] IndexIVFPQ_disk start to add\n", elapsed() - t0);
    //index.add(nb, xb);
    int nadd = 10;
    index.add_batch_num = nadd;
    for(int i = 0; i < nadd; i++){
        index.add(nb/nadd, xb + (nb/nadd * i) * d);
    }
    printf("[%.3f s] IndexIVFPQ_disk add finished\n", elapsed() - t0);

    printf("[%.3f s] IndexIVFPQ_disk start to reorg\n", elapsed() - t0);
    //index.initial_location(xb);
    printf("[%.3f s] IndexIVFPQ_disk reorg finished\n", elapsed() - t0);

    // Write index to disk
    faiss::write_index(&index, disk_store_path.c_str());
    printf("[%.3f s] IndexIVFPQ_disk written to disk\n", elapsed() - t0);

    delete[] xb;
}

template<typename T>
void search(const char* query_filepath, const char* ground_truth_filepath, 
            int nq, int d, int k, 
            const std::string& disk_store_path,
            const std::string& centroid_index_path) {
    double t0 = elapsed();
    size_t dd2; // dimension
    size_t nt2; // number of queries

    float* xq = load_and_convert_to_float(query_filepath, &dd2, &nt2);

    size_t nq2;
    size_t kk;
    int* gt_int = ivecs_read(ground_truth_filepath, &kk, &nq2);
    std::cout << "ground truth:" << kk << std::endl;

    float estimate_factor = 1.1;
    float estimate_factor_partial =1.15;
    float prune_factor =5.0;
    std::cout << "Reading Index: "<<std::endl;
    
    faiss::IndexIVFPQDisk2* index = dynamic_cast<faiss::IndexIVFPQDisk2*>(faiss::read_index(disk_store_path.c_str()));
    std::cout << "Disk index:" << index->disk_path << std::endl;
    

    index->set_centroid_index_path(centroid_index_path);
    index->load_hnsw_centroid_index();

    index->set_top(10);
    index->set_assign_replicas(2);
    index->set_estimate_factor(estimate_factor);
    index->set_estimate_factor_partial(estimate_factor_partial);
    index->set_prune_factor(prune_factor);
    

    std::cout << "Reading Finished " <<std::endl;

    std::vector<double> search_times;
    std::vector<double> recalls;


    idx_t* I = new idx_t[k * nq];
    float* D = new float[k * nq];
    faiss::idx_t* gt = new faiss::idx_t[kk * nq];
    int jj=0;
    for (int i = 0; i < nq; i++) {
        for(int j = 0; j < k; j++){
            gt[jj] =  static_cast<faiss::idx_t>(gt_int[i*kk+j]); //long int / int
            jj+=1;
        }
        
    }
    delete[] gt_int;

    //int arr[] = {20, 40, 60};
    //int arr[] = {20,40,60,80,100};
    //int arr[] = {400, 500, 600};
    int arr[] = {20,40,60,80,100,120,140,160,200,240,300,400};
    //int arr[] = {10,30,50,70,90,110,130,150,170,230,290,390};
    //int arr[] = {10, 40, 60, 80, 100, 120, 140};
    
    std::cout << "Warming up disk index: " <<std::endl;
    
    index->warm_up_nlist(1000, xq, 200, 0);

    index->warm_up_index_info(1000, xq, 200,40000);
    //index->warm_up_nvec(100, xq, 20, 300);

    
    std::cout << "Warming up finished cached: " << index->diskInvertedListHolder.cached_vector << " vectors"<<std::endl;
    //std::unordered_map<size_t, int> temp = index->diskInvertedListHolder.cache_lists;
    //return;
    int n_threads = 12;

    omp_set_num_threads(n_threads);
    //std::cout << "get_num_threads:" << omp_get_num_threads() << "\n";
    index->set_disk(n_threads);
    
    
    for (int i : arr) {
        printf("-----------now, the nprobe is=%d---------\n", i);

        index->nprobe = i;
        printf("[%.3f s] IndexIVFPQ_disk start to search\n", elapsed() - t0);
        double t1 = elapsed();
        index->search(nq, xq, k, D, I);
        double t2 = elapsed();
        double search_time = t2 - t1;
        printf("[%.3f s] IndexIVFPQ_disk search finished\n", elapsed() - t0);

        size_t fcc = faiss::indexIVFPQDisk2_stats.full_cluster_compare;
        size_t fcr = faiss::indexIVFPQDisk2_stats.full_cluster_rerank;
        size_t pcc = faiss::indexIVFPQDisk2_stats.partial_cluster_compare;
        size_t pcr = faiss::indexIVFPQDisk2_stats.partial_cluster_rerank;
        size_t sl  = faiss::indexIVFPQDisk2_stats.searched_lists;

        double me1 = faiss::indexIVFPQDisk2_stats.memory_1_elapsed.count() / 1000; 
        double me2 = faiss::indexIVFPQDisk2_stats.memory_2_elapsed.count() / 1000; 
        double me3 = faiss::indexIVFPQDisk2_stats.memory_3_elapsed.count() / 1000; 
        double dfe = faiss::indexIVFPQDisk2_stats.disk_full_elapsed.count() / 1000; 
        double dpe = faiss::indexIVFPQDisk2_stats.disk_partial_elapsed.count() / 1000; 
        double oe = faiss::indexIVFPQDisk2_stats.others_elapsed.count() / 1000; 
        double ce = faiss::indexIVFPQDisk2_stats.coarse_elapsed.count() / 1000; 
        double re = faiss::indexIVFPQDisk2_stats.rank_elapsed.count() / 1000; 
        double rre = faiss::indexIVFPQDisk2_stats.rerank_elapsed.count() / 1000; 
        double pe = faiss::indexIVFPQDisk2_stats.pq_elapsed.count() / 1000; 
        double cce = faiss::indexIVFPQDisk2_stats.cached_calculate_elapsed.count()/1000;
        double de = faiss::indexIVFPQDisk2_stats.delete_elapsed.count()/1000;

        double mue = faiss::indexIVFPQDisk2_stats.memory_uncache_elapsed.count() / 1000;
        double rue = faiss::indexIVFPQDisk2_stats.rank_uncache_elapsed.count() / 1000;
        double duie = faiss::indexIVFPQDisk2_stats.disk_uncache_info_elapsed.count() / 1000;
        double duce = faiss::indexIVFPQDisk2_stats.disk_uncache_calc_elapsed.count() / 1000;
        double pue = faiss::indexIVFPQDisk2_stats.pq_uncache_elapsed.count() / 1000;

        size_t svf = faiss::indexIVFPQDisk2_stats.searched_vector_full;
        size_t svp = faiss::indexIVFPQDisk2_stats.searched_vector_partial;
        size_t spf = faiss::indexIVFPQDisk2_stats.searched_page_full;
        size_t spp = faiss::indexIVFPQDisk2_stats.searched_page_partial;
        size_t rf = faiss::indexIVFPQDisk2_stats.requests_full;
        size_t rp = faiss::indexIVFPQDisk2_stats.requests_partial;
        size_t lf = faiss::indexIVFPQDisk2_stats.pq_list_full;
        size_t lp = faiss::indexIVFPQDisk2_stats.pq_list_partial;

        size_t cla = faiss::indexIVFPQDisk2_stats.cached_list_access;
        size_t cva = faiss::indexIVFPQDisk2_stats.cached_vector_access;

        std::cout << "full_cluster_compare      :" << fcc/nq << std::endl;
        std::cout << "full_cluster_rerank       :" << fcr/nq << std::endl;
        std::cout << "partial_cluster_compare   :" << pcc/nq << std::endl;
        std::cout << "partial_cluster_rerank    :" << pcr/nq << std::endl;

        std::cout << "AVG rerank ratio(full)    :" << static_cast<double>(fcr) / fcc << std::endl;
        std::cout << "AVG rerank ratio(partial) :" << static_cast<double>(pcr) / pcc << std::endl;

        std::cout << "Scanned lists total       :" << sl << std::endl;
        std::cout << "Scanned lists per query   :" << static_cast<double>(sl) / nq << std::endl;
        std::cout << "Scanned lists account for :" << static_cast<double>(sl) / (nq * i) << std::endl;
        std::cout << "TIME evaluate:\n";
        std::cout << "memory_1_elapsed        :" << me1/nq << std::endl;
        std::cout << "memory_2_elapsed        :" << me2/nq << std::endl;
        std::cout << "memory_3_elapsed        :" << me3/nq << std::endl;
        std::cout << "disk_full_elapsed       :" <<dfe/nq << std::endl;
        std::cout << "disk_partial_elapsed    :" << dpe/nq << std::endl;
        std::cout << "others_elapsed    :" << oe/nq << std::endl;
        std::cout << "coarse_elapsed    :" << ce/nq << std::endl;
        std::cout << "rank_elapsed    :" << re/nq << std::endl;
        std::cout << "rerank_elapsed    :" << rre/nq << std::endl;
        std::cout << "pq_elapsed    :" << pe/nq << std::endl;
        std::cout << "cache_elapsed    :" << cce/nq << std::endl;
        std::cout << "delete_elapsed    :" << de/nq << std::endl;
        std::cout << "\n\n\n";
        std::cout << "memory_uncache_elapsed   :" << mue/nq << std::endl;
        std::cout << "rank_uncache_elapsed     :" << rue/nq << std::endl;
        std::cout << "disk_uncache_info_elapsed     :" << duie/nq << std::endl;
        std::cout << "disk_uncache_calc_elapsed     :" << duce/nq << std::endl;
        std::cout << "pq_uncache_elapsed       :" << pue/nq << std::endl;



        std::cout << "searched_vector_full   : " << svf/nq <<std::endl;
        std::cout << "searched_vector_partial: " << svp/nq <<std::endl;
        std::cout << "partial/full           : " << svp/(svf+1e6) << std::endl;

        std::cout << "pq_list_full           : " << lf/nq <<std::endl;
        std::cout << "pq_list_partial       : " << lp/nq <<std::endl;

        std::cout << "searched_page_full   : " << spf/nq <<std::endl;
        std::cout << "searched_page_partial: " << spp/nq <<std::endl;

        std::cout << "requests_full   : " << rf/nq <<std::endl;
        std::cout << "requests_partial: " << rp/nq <<std::endl;

        std::cout << "cached lists    : " << cla/nq <<std::endl;
        std::cout << "cached vectors  : " << cva/nq <<std::endl;

        std::cout << "\n\n\n\n";

        faiss::indexIVFPQDisk2_stats.reset();

        // int repeated = 0;
        // for (int i = 0; i < nq; i++)
        // {
        //     std::unordered_set<size_t> seen_ids;

        //     for (int j = 0; j < k; j++)
        //     {
        //         size_t temp_id = I[i * k + j];  // 获取第 i 个查询的第 j 个结果的 ID
                
        //         if (seen_ids.find(temp_id) != seen_ids.end())
        //         {
        //             repeated++;
        //             std::cout << "nq:" << i << "  nr:" << j  << "  id:" << temp_id<< std::endl; 
        //         }
        //         else
        //         {
        //             seen_ids.insert(temp_id);
        //         }
        //     }
        // }
        // std::cout << "repeated = " << repeated << std::endl;

        // evaluate result
        int n2_100 = 0;
        int temp = 0;
        for (int i = 0; i < nq; i++) {
            std::map<float, int> umap;
            for (int j = 0; j < k; j++) {
                umap.insert({gt[i * k + j], 0});
            }
            for (int l = 0; l < k; l++) {
                if (umap.find(I[i * k + l]) != umap.end()) {
                    n2_100++;
                }
            }
            //std::cout << n2_100 - temp << " ";
            temp = n2_100;
            umap.clear();
        }
        printf("Intersection R@100 = %.4f\n", n2_100 / float(nq * k));
        double recall = n2_100 / float(nq * k);
        search_times.push_back(search_time / nq * 1000);
        recalls.push_back(recall);
    }
    index->end_disk(n_threads);
    delete[] I;
    delete[] D;
    delete[] xq;
    delete[] gt;

    // Print the search times and recalls
    std::cout << "Search times: [";
    for (size_t i = 0; i < search_times.size(); i++) {
        std::cout << search_times[i];
        if (i < search_times.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "QPS         : [";
    for (size_t i = 0; i < search_times.size(); i++) {
        std::cout << (1000.0 / search_times[i]);
        if (i < search_times.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "Recalls     : [";
    for (size_t i = 0; i < recalls.size(); i++) {
        std::cout << recalls[i];
        if (i < recalls.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "Time Gap    : [";
    for(int i = 1; i < search_times.size(); i++){
        std::cout << search_times[i] - search_times[i-1];
        if (i < search_times.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    delete index;
}

// 1. 异步IO 3.2s，如果加入pq与rerank变为3.6s

int main(int argc, char *argv[]) {
    int type = atoi(argv[1]); // Set to 0 for build, 1 for search

    int d = 128;      // dimension
    int nb = 1000000*10; // database size
    int nq = 1000;   // number of queries
    int ratio = 50;   // ratio for training
    int k = 100;      // number of nearest neighbors to search

    const char* base_filepath = "/mnt/d/VectorDB/sift/sift10m_uint8/bigann_10m_base.bvecs";
    const char* query_filepath = "/mnt/d/VectorDB/sift/sift10m_uint8/bigann_query.bvecs";
    const char* ground_truth_filepath = "/mnt/d/VectorDB/sift/sift10m_uint8/idx_10M.ivecs";
    // data
    
    // std::string index_store_path =    "/home/granthe/ivfpq_test_data/sift10M/sift10M_list50k_greedybuild";
    // std::string disk_store_path =     "/home/granthe/ivfpq_test_data/sift10M/sift10M_ivfpqdisk_list50k_greedybuild.index";
    // std::string centroid_index_path = "/home/granthe/ivfpq_test_data/sift10M/sift10M_centroid_hnsw_list50k_greedybuild";

    std::string index_store_path =    "/home/granthe/ivfpq_test_data/sift10M/sift10M_r1_list40k_pq168_uint8_greedybuild_slmode_bs";
    std::string disk_store_path =     "/home/granthe/ivfpq_test_data/sift10M/sift10M_r1_ivfpqdisk_list40k_pq168_uint8_greedybuild_slmode_bs.index";
    std::string centroid_index_path = "/home/granthe/ivfpq_test_data/sift10M/sift10M_r1_centroid_hnsw_list40k_pq168_uint8_greedybuild_slmode_bs";
    // centroid hnsw
    
    // 这里的type不起作用..
    if (type == 0) {
        disk_build<uint8_t>(base_filepath, nb, d, ratio, index_store_path, disk_store_path, centroid_index_path);
    } else if (type == 1) {
        search<uint8_t>(query_filepath, ground_truth_filepath, nq, d, k, disk_store_path, centroid_index_path);
    }

    return 0;
}



