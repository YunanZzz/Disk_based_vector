/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQDisk.h>

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
int main() {
    double t0 = elapsed();
    int d = 128;      // dimension
    int nb = 1000000; // database size
    int nq = 100;   // nb of queries
    char* base_filepath ="/mnt/d/VectorDB/sift/sift/sift_base.fvecs";
    char* query_filepath ="/mnt/d/VectorDB/sift/sift/sift_query.fvecs";
    char* ground_truth_filepath="/mnt/d/VectorDB/sift/sift/sift_groundtruth.ivecs";
    
    std::string index_store_path  = "/mnt/d/VectorDB/sift/sift1m_test/sift1M";
    //std::string result_path       = "/mnt/d/VectorDB/RESULT_SET/ivf_pq_disk_wsl.txt";

    size_t dd; // dimension
    size_t nt; // the number of vectors
    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    size_t dd2; // dimension
    size_t nt2; // the number of query
    xb = fvecs_read(base_filepath, &dd, &nt);
    xq = fvecs_read(query_filepath, &dd2, &nt2);

    int ratio = 50;
    {
        srand((int)time(0));
        std::vector<float> trainvecs(nb / ratio * d);
        for (int i = 0; i < nb / ratio; i++) {
            int rng = (rand() % (nb + 1));
            // rng=random number in nb

            for (int j = 0; j < d; j++) {
                // printf(" setting %d vector's %d data, trianvecs[%d]=xb[%d]
                // \n",i,j,d * i + j,rng * d + j);
                trainvecs[d * i + j] = xb[rng * d + j];
            }
        }
        int nlist = 1000;
        int k = 100;
        // load ground-truth and convert int to long
        size_t nq2;
        size_t kk;
        int* gt_int = ivecs_read(ground_truth_filepath, &kk, &nq2);

        //pq parameters
        int m = 32;
        int nbits = 8;
        // Disk parameters
        int top_clusters = 5;
        float estimate_factor = 1.2;
        float prune_factor = 1.9;

        faiss::IndexFlatL2 quantizer(d);
        faiss::IndexIVFPQDisk index(&quantizer, d, nlist, m, nbits, top_clusters, estimate_factor, prune_factor,index_store_path);   
        index.set_assign_replicas(3);
        index.verbose = true;

        printf("[%.3f s] IndexIVFPQ_disk start to train\n", elapsed() - t0);
        index.train(nb / ratio, trainvecs.data());
        printf("[%.3f s] IndexIVFPQ_disk train finished\n", elapsed() - t0);

        printf("[%.3f s] IndexIVFPQ_disk start to add\n", elapsed() - t0);
        index.add(nb, xb);
        printf("[%.3f s] IndexIVFPQ_disk add finished\n", elapsed() - t0);

        printf("[%.3f s] IndexIVFPQ_disk start to reorg\n", elapsed() - t0);
        index.initial_location(xb);
        printf("[%.3f s] IndexIVFPQ_disk reorg finished\n", elapsed() - t0);

        std::vector<double> search_times;
        std::vector<double> recalls;
        omp_set_num_threads(1);
        { // search xq

            idx_t* I = new idx_t[k * nq];
            float* D = new float[k * nq];
            faiss::idx_t* gt;
            gt = new faiss::idx_t[kk * nq];
            for (int i = 0; i < kk * nq; i++) {
                gt[i] = gt_int[i]; // long int / int
            }
            delete[] gt_int;
            //int arr[] = {4, 5, 6, 7, 8, 10};
            //int arr[] = {5, 7, 9, 11, 13};
            int arr[] = {5,10,15,20,25,27};

            // write_index(index, "hnsw.index");
            // how to get length of array

            for (int i : arr) {
                printf("-----------now, the efs is=%d---------\n", i);
                
                index.nprobe = i;
                printf("[%.3f s] IndexIVFPQ_disk start to search\n", elapsed() - t0);
                double t1 = elapsed();
                index.search(nq, xq, k, D, I);
                double t2 = elapsed();
                double search_time = t2 - t1;
                printf("[%.3f s] IndexIVFPQ_disk search finished\n", elapsed() - t0);

                size_t fcc = faiss::indexIVFPQDisk_stats.full_cluster_compare;
                size_t fcr = faiss::indexIVFPQDisk_stats.full_cluster_rerank;
                size_t pcc = faiss::indexIVFPQDisk_stats.partial_cluster_compare;
                size_t pcr = faiss::indexIVFPQDisk_stats.partial_cluster_rerank;

                size_t sl  = faiss::indexIVFPQDisk_stats.pruned;

                int repeated = 0;
                for (int i = 0; i < nq; i++)
                {
                    std::unordered_set<size_t> seen_ids;

                    for (int j = 0; j < k; j++)
                    {
                        size_t temp_id = I[i * k + j];  // 获取第 i 个查询的第 j 个结果的 ID
                        
                        if (seen_ids.find(temp_id) != seen_ids.end())
                        {
                            repeated++;
                        }
                        else
                        {
                            seen_ids.insert(temp_id);
                        }
                    }
                }
                std::cout << "repeated = " << repeated << std::endl;


                std::cout << "full_cluster_compare      :"<<fcc << std::endl;
                std::cout << "full_cluster_rerank       :"<<fcr << std::endl;
                std::cout << "partial_cluster_compare   :"<<pcc << std::endl;
                std::cout << "partial_cluster_rerank    :"<<pcr << std::endl;

                std::cout << "AVG rerank ratio(full)    :" << (double)fcr/(double)fcc << std::endl;
                std::cout << "AVG rerank ratio(partital):" << (double)pcr/(double)pcc << std::endl;

                std::cout << "Scanned lists total       :" << sl << std::endl;
                std::cout << "Scanned lists per query   :" << (double)sl/(double)nq << std::endl;
                std::cout << "Scanned lists account for :" << (double)sl /((double)nq*i) << std::endl;
                

                faiss::indexIVFPQDisk_stats.reset();
                // evaluate result
                int n2_100 = 0;
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
                    umap.clear();
                }
                printf("Intersection R@100 = %.4f\n", n2_100 / float(nq * k));
                double recall = n2_100 / float(nq * k);
                search_times.push_back(search_time/nq*1000);
                recalls.push_back(recall);
            }

            delete[] I;
            delete[] D;
        }

        delete[] xb;
        delete[] xq;
        // Print the search times and recalls
        std::cout << "Search times: [";
        for (size_t i = 0; i < search_times.size(); i++) {
            std::cout << search_times[i];
            if (i < search_times.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "QPS: [";
        for (size_t i = 0; i < search_times.size(); i++) {
            std::cout << (1000.0/search_times[i]);
            if (i < search_times.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "Recalls: [";
        for (size_t i = 0; i < recalls.size(); i++) {
            std::cout << recalls[i];
            if (i < recalls.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        return 0;
    }
}