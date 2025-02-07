
#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexDisk.h>
#include <faiss/IndexRefine.h>

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

void disk_build(const char* base_filepath, int nb, int d, int ratio, const std::string& index_write_path, const std::string& disk_data_store_path, int type) {
    double t0 = elapsed();
    size_t dd; // dimension
    size_t nt; // number of vectors

    float* xb = fvecs_read(base_filepath, &dd, &nt);

    std::vector<float> trainvecs(nb / ratio * d);
    srand(static_cast<int>(time(0)));
    for (int i = 0; i < nb / ratio; i++) {
        int rng = (rand() % (nb + 1));
        for (int j = 0; j < d; j++) {
            trainvecs[d * i + j] = xb[rng * d + j];
        }
    }

    int nlist = 2000;
    int m = 32;
    int nbits = 4;

    faiss::IndexFlatL2 quantizer(d);
    
    // Declare pointers to the base index and refine index
    faiss::IndexIVFPQ* index_pq = nullptr;
    faiss::IndexIVFPQFastScan* index_pqfs = nullptr;
    faiss::IndexRefine* index_refine = nullptr;
    
    faiss::IndexDisk index_disk(d, disk_data_store_path);

    // Determine which base index to use based on `type`
    if (type == 0) {
        index_pq = new faiss::IndexIVFPQ(&quantizer, d, nlist, m, nbits);
        index_refine = new faiss::IndexRefine(index_pq, &index_disk);
        printf("Using IndexIVFPQ as base index.\n");
    } else if (type == 1) {
        index_pqfs = new faiss::IndexIVFPQFastScan(&quantizer, d, nlist, m, nbits);
        index_refine = new faiss::IndexRefine(index_pqfs, &index_disk);
        printf("Using IndexIVFPQFastScan as base index.\n");
    } else {
        throw std::invalid_argument("Invalid type specified. Use 0 for IndexIVFPQ, 1 for IndexIVFPQFastScan.");
    }

    index_refine->k_factor = 3;

    printf("[%.3f s] IndexIVFPQ_Refine start to train\n", elapsed() - t0);
    index_refine->train(nb / ratio, trainvecs.data());
    printf("[%.3f s] IndexIVFPQ_Refine train finished\n", elapsed() - t0);

    printf("[%.3f s] IndexIVFPQ_Refine start to add\n", elapsed() - t0);
    index_refine->add(nb, xb);
    printf("[%.3f s] IndexIVFPQ_Refine add finished\n", elapsed() - t0);
    // Write index to disk
    faiss::write_index(index_refine, index_write_path.c_str());
    printf("[%.3f s] IndexIVFPQ_Refine written to disk\n", elapsed() - t0);

    // Clean up
    delete[] xb;
    delete index_refine;
    if (index_pq) {
        delete index_pq;
    }
    if (index_pqfs) {
        delete index_pqfs;
    }
}

void search(const char* query_filepath, const char* ground_truth_filepath, int nq, int d, int k, const std::string& disk_store_path) {
    double t0 = elapsed();
    size_t dd2; // dimension
    size_t nt2; // number of queries

    float* xq = fvecs_read(query_filepath, &dd2, &nt2);

    size_t nq2;
    size_t kk;
    int* gt_int = ivecs_read(ground_truth_filepath, &kk, &nq2);
    faiss::IndexRefine* index = dynamic_cast<faiss::IndexRefine*>(faiss::read_index(disk_store_path.c_str()));
    std::cout << " write IndexRefine" << std::endl;

    faiss::IndexIVFPQ* index_ivfpq = nullptr;
    faiss::IndexIVFPQFastScan* index_ivfpqfs = nullptr;
    
    if ((index_ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(index->base_index))) {
        printf("Base index is of type IndexIVFPQ\n");
    } else if ((index_ivfpqfs = dynamic_cast<faiss::IndexIVFPQFastScan*>(index->base_index))) {
        printf("Base index is of type IndexIVFPQFastScan\n");
    } else {
        throw std::runtime_error("Base index is neither IndexIVFPQ nor IndexIVFPQFastScan");
    }
    faiss::IndexDisk* index_ds =  dynamic_cast<faiss::IndexDisk*>(index->refine_index);
    std::cout << " index_ds->disk_path :" << index_ds->disk_path << std::endl;
    index->k_factor = 7;
    std::vector<double> search_times;
    std::vector<double> recalls;
    omp_set_num_threads(1);

    idx_t* I = new idx_t[k * nq];
    float* D = new float[k * nq];
    faiss::idx_t* gt = new faiss::idx_t[kk * nq];
    for (int i = 0; i < kk * nq; i++) {
        gt[i] = gt_int[i];
    }
    delete[] gt_int;

    int arr[] = {150};

    for (int i : arr) {
        printf("-----------now, the nprobe is=%d---------\n", i);

        if (index_ivfpq) {
            index_ivfpq->nprobe = i;
        } else if (index_ivfpqfs) {
            index_ivfpqfs->nprobe = i;
        }
        printf("[%.3f s] IndexIVFPQ_disk start to search\n", elapsed() - t0);
        double t1 = elapsed();
        index->search(nq, xq, k, D, I);
        double t2 = elapsed();
        double search_time = t2 - t1;
        printf("[%.3f s] IndexIVFPQ_disk search finished\n", elapsed() - t0);

        size_t re = faiss::indexDisk_stats.rerank;
        double de = faiss::indexDisk_stats.disk_elapsed.count() / 1000; 
        double me = faiss::indexDisk_stats.memory_elapsed.count() / 1000; 
            
        std::cout << "rerank:               :"<< re/nq << std::endl;
        std::cout << "disk_elapsed          :" << de/nq << std::endl;
        std::cout << "memory_elapsed        :" << me/nq << std::endl;

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
                    std::cout << "nq:" << i << "  nr:" << j << std::endl; 
                }
                else
                {
                    seen_ids.insert(temp_id);
                }
            }
        }
        std::cout << "repeated = " << repeated << std::endl;


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
        search_times.push_back(search_time / nq * 1000);
        recalls.push_back(recall);
        if(recall > 0.993)
            break;
    }

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

    std::cout << "QPS: [";
    for (size_t i = 0; i < search_times.size(); i++) {
        std::cout << (1000.0 / search_times[i]);
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

    delete index;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <build|search> <pq_common|pq_fastscan>" << std::endl;
        return 1;
    }

    int op_type = -1;
    int pq_type = -1;
    
    // Determine operation type (build or search)
    std::string operation = argv[1];
    if (operation == "build") {
        op_type = 0;
    } else if (operation == "search") {
        op_type = 1;
    } else {
        std::cerr << "Invalid operation type: " << operation << ". Use 'build' or 'search'." << std::endl;
        return 1;
    }

    // Determine pq_type (ivfpq or ivfpq_fastscan)
    std::string pq_type_str = argv[2];
    if (pq_type_str == "pq_common") {
        pq_type = 0;
    } else if (pq_type_str == "pq_fastscan") {
        pq_type = 1;
    } else {
        std::cerr << "Invalid pq type: " << pq_type_str << ". Use 'pq_common' or 'pq_fastscan'." << std::endl;
        return 1;
    }

    // Set parameters
    int d = 128;      // dimension
    int nb = 1000000; // database size
    int nq = 100;   // number of queries
    int ratio = 50;   // ratio for training
    int k = 100;      // number of nearest neighbors to search

    const char* base_filepath = "/mnt/d/VectorDB/sift/sift/sift_base.fvecs";
    const char* query_filepath = "/mnt/d/VectorDB/sift/sift/sift_query.fvecs";
    const char* ground_truth_filepath = "/mnt/d/VectorDB/sift/sift/sift_groundtruth.ivecs";

    // Adjust index store path based on pq type
    std::string index_write_path = "/home/granthe/faiss/faiss/build/data/sift1M/sift1M_refine_pq32_8_nlist8000_" + pq_type_str;
    std::string disk_data_store_path = "/home/granthe/data/sift1M/sift_base.fvecs_" + pq_type_str;
 
    // Perform the specified operation
    if (op_type == 0) {
        disk_build(base_filepath, nb, d, ratio, index_write_path, disk_data_store_path, pq_type);
    } else if (op_type == 1) {
        search(query_filepath, ground_truth_filepath, nq, d, k, index_write_path);
    }

    return 0;
}
