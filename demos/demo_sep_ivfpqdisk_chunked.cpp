
#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQDisk.h>

#include <faiss/index_io.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
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
void load_data_float(const char* filename, float*& data, size_t num, int dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Error opening file" << std::endl;
        exit(-1);
    }

    // Allocate memory for the data
    data = new float[num * dim];

    // Buffer to hold the entire vector (dimension field + actual vector data)
    unsigned char* buffer = new unsigned char[4 + dim * sizeof(float)];

    // Read each vector
    for (size_t i = 0; i < num; i++) {
        // Read the entire vector (4 bytes for dimension, followed by dim bytes
        // for vector data)
        in.read(reinterpret_cast<char*>(buffer), 4 + dim * sizeof(float));

        // Skip the first 4 bytes (dimension), directly copy the float data
        std::memcpy(data + i * dim, buffer + 4, dim * sizeof(float));
    }

    // Clean up
    delete[] buffer;
    in.close();
}
void load_data_to_float(
        const char* filename,
        float*& data,
        size_t num,
        int dim) {
    std::ifstream in(filename, std::ios::binary); // open file in binary mode
    if (!in.is_open()) {
        std::cout << "Error opening file" << std::endl;
        exit(-1);
    }

    // Allocate memory for the float array (num * dim floats)
    data = new float[num * dim];

    // Buffer to hold the entire vector (dimension field + actual vector data as
    // unsigned char)
    unsigned char* buffer = new unsigned char[4 + dim];

    // Read each vector
    for (size_t i = 0; i < num; i++) {
        // Read the entire vector (4 bytes for dimension, followed by dim bytes
        // for vector data)
        in.read(reinterpret_cast<char*>(buffer), 4 + dim);

        // Convert the unsigned char data to float and store in the float array
        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = static_cast<float>(buffer[4 + j]);
        }
    }

    // Clean up
    delete[] buffer;
    in.close();
}
void load_data(
        const char* filename,
        unsigned char*& data,
        size_t num,
        int dim) {
    std::ifstream in(filename, std::ios::binary); // open file in binary mode
    if (!in.is_open()) {
        std::cout << "Error opening file" << std::endl;
        exit(-1);
    }

    // Allocate memory for the data using size_t to avoid overflow
    data = new unsigned char[num * dim];

    // Buffer to hold the entire vector (dimension field + actual vector data)
    unsigned char* buffer = new unsigned char[4 + dim];

    // Read each vector
    for (size_t i = 0; i < num; i++) {
        // Read the entire vector (4 bytes for dimension, followed by dim bytes
        // for vector data)
        in.read((char*)buffer, 4 + dim);

        // Copy the actual vector data (skipping the first 4 bytes, which
        // represent the dimension)
        std::memcpy(data + i * dim, buffer + 4, dim);
    }

    // Clean up
    delete[] buffer;
    in.close();
}
void load_data_chunk(
        const char* filename,
        unsigned char* data,
        size_t start,
        size_t num,
        int dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(-1);
    }

    // Move file pointer to the start of the chunk
    in.seekg(start * (4 + dim), std::ios::beg);

    unsigned char* buffer = new unsigned char[4 + dim];

    // Read each vector in the chunk
    for (size_t i = 0; i < num; i++) {
        in.read(reinterpret_cast<char*>(buffer), 4 + dim);
        std::memcpy(data + i * dim, buffer + 4, dim);
    }

    delete[] buffer;
    in.close();
}
void load_data_chunk_float(
        const char* filename,
        float* data,
        size_t start,
        size_t num,
        int dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(-1);
    }

    // Move file pointer to the start of the chunk
    in.seekg(start * (4 + dim * sizeof(float)), std::ios::beg);

    // Allocate a buffer for one vector (4 bytes for an index and dim *
    // sizeof(float) bytes for the vector)
    unsigned char* buffer = new unsigned char[4 + dim * sizeof(float)];

    // Read each vector in the chunk
    for (size_t i = 0; i < num; i++) {
        // Read the entire chunk for one vector (4 bytes for index + dim *
        // sizeof(float) for the data)
        in.read(reinterpret_cast<char*>(buffer), 4 + dim * sizeof(float));

        if (in.gcount() != 4 + dim * sizeof(float)) {
            std::cerr << "Error reading data for vector " << i << std::endl;
            break; // Or handle the error as needed
        }

        // Copy only the vector data (skip the first 4 bytes)
        std::memcpy(data + i * dim, buffer + 4, dim * sizeof(float));
    }

    delete[] buffer;
    in.close();
}

void add_chunk_to_index(
        faiss::IndexIVFPQDisk& index,
        unsigned char* chunk_data,
        size_t num_vectors,
        int dim) {
    float* float_data = new float[num_vectors * dim];

    // Convert chunk data to floats
    for (size_t i = 0; i < num_vectors * dim; i++) {
        float_data[i] = static_cast<float>(chunk_data[i]);
    }

    // Add the chunk to the index
    index.add(num_vectors, float_data);

    // Clean up
    delete[] float_data;
}

// void disk_build(const char* base_filepath, int nb, int d, int ratio, const
// std::string& index_store_path, const std::string& disk_store_path) {
//     double t0 = elapsed();
//     size_t dd; // dimension
//     size_t nt; // number of vectors
//     size_t chunk_size = 1000000;  // Load 1 million vectors at a time
//     size_t num_chunks = nb / chunk_size;
//     // float* xb = fvecs_read(base_filepath, &dd, &nt);
//     // float* xb=NULL;
//     //  load_data_float(base_filepath,xb,nb,d);
//     //  printf("base set load done\n");
//             srand(static_cast<int>(time(0)));
//     std::vector<float> trainvecs(nb / 50 * d);
//         for (size_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
//         std::cout << "[Chunk " << chunk_id + 1 << "/" << num_chunks << "]
//         Loading chunk..." << std::endl; unsigned char* xb_chunk = nullptr;
//         load_data_chunk(base_filepath, xb_chunk, chunk_id * chunk_size,
//         chunk_size, d);

//         // Fill the training vectors from this chunk
//         for (int i = 0; i < chunk_size / 50; i++) {  // Assuming you want to
//         sample 2% of vectors
//             int rng = rand() % chunk_size;
//             for (int j = 0; j < d; j++) {
//                 trainvecs[chunk_id * (chunk_size / 50) * d + i * d + j] =
//                 static_cast<float>(xb_chunk[rng * d + j]);
//             }
//         }
//     }

//     // for (int i=0;i<10;i++){
//     //      printf("this is :%d\n",1000000*100*128-1-i);
//     //     printf("Check:%f\n",xb[1000000*100*128-1-i]);
//     // }
//     //     srand(static_cast<int>(time(0)));
//     //     std::vector<float> trainvecs(nb / ratio * d);
//     // for (int i = 0; i < nb / ratio; i++) {
//     //     int rng = (rand() % (nb+1));
//     //     for (int j = 0; j < d; j++) {
//     //         trainvecs[d * i + j] = xb[rng * d + j];
//     //     }
//     // }
//     // printf("check 2\n");
//     int nlist = 500000;
//     int m = 16;
//     int nbits = 8;
//     int top_clusters = 80;
//     float estimate_factor = 1.2;
//     float prune_factor = 5;

//     faiss::IndexFlatL2 quantizer(d);
//     faiss::IndexIVFPQDisk index(&quantizer, d, nlist, m, nbits, top_clusters,
//     estimate_factor, prune_factor, index_store_path);
//     index.set_assign_replicas(3);
//     index.set_estimate_factor_partial(1.1);
//     index.verbose = false;
//     omp_set_num_threads(64);
//     printf("[%.3f s] IndexIVFPQ_disk start to train\n", elapsed() - t0);
//     index.train(nb / ratio, trainvecs.data());
//     printf("[%.3f s] IndexIVFPQ_disk train finished\n", elapsed() - t0);

//     printf("[%.3f s] IndexIVFPQ_disk start to add\n", elapsed() - t0);
//     // index.add(nb, xb);
//         for (size_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
//         std::cout << "[Chunk " << chunk_id + 1 << "/" << num_chunks << "]
//         Adding to index..." << std::endl; unsigned char* xb_chunk = nullptr;
//         load_data_chunk(base_filepath, xb_chunk, chunk_id * chunk_size,
//         chunk_size, d);

//         // Add chunk to index
//         add_chunk_to_index(index, xb_chunk, chunk_size, d);

//         // Clean up chunk memory
//         delete[] xb_chunk;
//     }
//     printf("[%.3f s] IndexIVFPQ_disk add finished\n", elapsed() - t0);

//     printf("[%.3f s] IndexIVFPQ_disk start to reorg\n", elapsed() - t0);
//     index.initial_location(xb);
//     printf("[%.3f s] IndexIVFPQ_disk reorg finished\n", elapsed() - t0);

//     // Write index to disk
//     faiss::write_index(&index, disk_store_path.c_str());
//     printf("[%.3f s] IndexIVFPQ_disk written to disk\n", elapsed() - t0);

//     delete[] xb;
// }
void disk_build(
        const char* base_filepath,
        int nb,
        int d,
        int ratio,
        const std::string& index_store_path,
        const std::string& disk_store_path) {
    double t0 = elapsed();
    size_t chunk_size = 1000000; // Load 1 million vectors at a time
    size_t num_chunks = nb / chunk_size;

    // Prepare the index
    int nlist = 50000;
    int m = 16;
    int nbits = 8;
    int top_clusters = 80;
    float estimate_factor = 1.2;
    float prune_factor = 5;

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFPQDisk index(
            &quantizer,
            d,
            nlist,
            m,
            nbits,
            top_clusters,
            estimate_factor,
            prune_factor,
            index_store_path);
    index.set_assign_replicas(3);
    index.set_estimate_factor_partial(1.1);
    // std::string full_path = std::string("build") + "/" + "testfile";
    std::string centroid_path =
            "build/demos/index_folder/sift10m/centroid_graph";
    // index.set_centroid_index_path(centroid_path);
    index.verbose = false;
    omp_set_num_threads(16);

    srand(static_cast<int>(time(0)));
    std::vector<float> trainvecs(nb / ratio * d);
    std::vector<std::vector<unsigned char>> all_chunks(num_chunks);
    for (size_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        std::cout << "[Chunk " << chunk_id + 1 << "/" << num_chunks
                  << "] Loading chunk..." << std::endl;
        all_chunks[chunk_id].resize(chunk_size * d);

        // Load the data into the vector
        load_data_chunk(
                base_filepath,
                all_chunks[chunk_id].data(),
                chunk_id * chunk_size,
                chunk_size,
                d);
        // Fill the training vectors from this chunk
        for (int i = 0; i < chunk_size / ratio;
             i++) { // Assuming you want to sample 2% of vectors
            int rng = rand() % chunk_size;
            for (int j = 0; j < d; j++) {
                trainvecs[chunk_id * (chunk_size / ratio) * d + i * d + j] =
                        static_cast<float>(all_chunks[chunk_id][rng * d + j]);
            }
        }
    }

    // Step 3: Train the index once
    printf("[%.3f s] ivfflat start to train\n", elapsed() - t0);
    index.train(nb / ratio, trainvecs.data());
    printf("[%.3f s] ivfflat train finished\n", elapsed() - t0);
    // index.load_hnsw_centroid_index();
    // Step 4: Add vectors from the loaded chunks to the index
    std::string disk_output_path = disk_store_path + ".clustered";
    index.set_disk_write(disk_output_path); // Open file for appending

    for (size_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        std::cout << "[Chunk " << chunk_id + 1 << "/" << num_chunks
                  << "] Adding to index..." << std::endl;

        // Use the stored chunk for adding vectors to the index
        add_chunk_to_index(index, all_chunks[chunk_id].data(), chunk_size, d);
    }

    printf("[%.3f s] IndexIVFPQ_disk adding finished\n", elapsed() - t0);
    printf("[%.3f s] IndexIVFPQ_disk start to reorg\n", elapsed() - t0);
    // index.initial_location_chunked(all_chunks);
    printf("[%.3f s] IndexIVFPQ_disk reorg finished\n", elapsed() - t0);
    // Write index to disk
    faiss::write_index(&index, disk_store_path.c_str());
    printf("[%.3f s] IndexIVFPQ_disk written to disk\n", elapsed() - t0);
}

void search(
        const char* query_filepath,
        const char* ground_truth_filepath,
        int nq,
        int d,
        int k,
        const std::string& disk_store_path) {
    double t0 = elapsed();
    size_t dd2; // dimension
    size_t nt2; // number of queries

    float* xq = NULL;
    load_data_to_float(query_filepath, xq, nq, d);

    size_t nq2;
    size_t kk;
    int* gt_int = ivecs_read(ground_truth_filepath, &kk, &nq2);

    faiss::IndexIVFPQDisk* index = dynamic_cast<faiss::IndexIVFPQDisk*>(
    faiss::read_index(disk_store_path.c_str()));

    std::vector<double> search_times;
    std::vector<double> recalls;
    omp_set_num_threads(16);

    idx_t* I = new idx_t[k * nq];
    float* D = new float[k * nq];
    faiss::idx_t* gt;
    gt = new faiss::idx_t[k * nq];
    int jj = 0;
    for (int i = 0; i < kk * nq; i++) {
        gt[jj] = gt_int[i]; // long int / int
        jj += 1;
        if ((i + 1) % 100 == 0) {
            i += 900;
        }
    }
    delete[] gt_int;

    // int arr[] = {60, 80, 100, 120, 140, 160, 180, 200};
    int arr[] = {20, 30, 40, 60, 80, 100, 120, 140, 160, 180};

    for (int i : arr) {
        printf("-----------now, the nprobe is=%d---------\n", i);

        index->nprobe = i;
        printf("[%.3f s] IndexIVFPQ_disk start to search\n", elapsed() - t0);
        double t1 = elapsed();
        index->search(nq, xq, k, D, I);
        double t2 = elapsed();
        double search_time = t2 - t1;
        printf("[%.3f s] IndexIVFPQ_disk search finished\n", elapsed() - t0);

        size_t fcc = faiss::indexIVFPQDisk_stats.full_cluster_compare;
        size_t fcr = faiss::indexIVFPQDisk_stats.full_cluster_rerank;
        size_t pcc = faiss::indexIVFPQDisk_stats.partial_cluster_compare;
        size_t pcr = faiss::indexIVFPQDisk_stats.partial_cluster_rerank;
        size_t sl = faiss::indexIVFPQDisk_stats.pruned;
        double m1t = faiss::indexIVFPQDisk_stats.memory_1_elapsed.count();
        double m2t = faiss::indexIVFPQDisk_stats.memory_2_elapsed.count();
        double ft = faiss::indexIVFPQDisk_stats.disk_full_elapsed.count();
        double pt = faiss::indexIVFPQDisk_stats.disk_partial_elapsed.count();
        double pqt = faiss::indexIVFPQDisk_stats.others_elapsed.count();
        double nt = faiss::indexIVFPQDisk_stats.coarse_elapsed.count();
        double rt = faiss::indexIVFPQDisk_stats.rank_elapsed.count();
        std::cout << "full_cluster_compare      :" << fcc << std::endl;
        std::cout << "full_cluster_rerank       :" << fcr << std::endl;
        std::cout << "partial_cluster_compare   :" << pcc << std::endl;
        std::cout << "partial_cluster_rerank    :" << pcr << std::endl;
        std::cout << "memory_1_elapsed    :" << m1t << std::endl;
        std::cout << "memory_2_elapsed :" << m2t << std::endl;
        std::cout << "disk_full_elapsed    :" << ft << std::endl;
        std::cout << "disk_partial_elapsed :" << pt << std::endl;
        std::cout << "rank_elapsed    :" << rt << std::endl;
        std::cout << "coarse_elapsed    :" << nt << std::endl;
        std::cout << "others_elapsed    :" << pqt << std::endl;
        std::cout << "AVG rerank ratio(full)    :"
                  << static_cast<double>(fcr) / fcc << std::endl;
        std::cout << "AVG rerank ratio(partial) :"
                  << static_cast<double>(pcr) / pcc << std::endl;

        std::cout << "Scanned lists total       :" << sl << std::endl;
        std::cout << "Scanned lists per query   :"
                  << static_cast<double>(sl) / nq << std::endl;
        std::cout << "Scanned lists account for :"
                  << static_cast<double>(sl) / (nq * i) << std::endl;

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
        search_times.push_back(search_time / nq * 1000);
        recalls.push_back(recall);
        if (recall > 0.99)
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

int main(int argc, char* argv[]) {
    int type = atoi(argv[1]); // Set to 0 for build, 1 for search

    int d = 128;           // dimension
    int nb = 1000000 * 10; // database size
    int nq = 10000;        // number of queries
    int ratio = 50;        // ratio for training
    int k = 100;           // number of nearest neighbors to search

    const char* base_filepath =
            "/ssd_root/zhan4404/dataset/dataset/sift1B/bigann_10m_base.bvecs";
    const char* query_filepath =
            "/ssd_root/zhan4404/dataset/dataset/sift1B/bigann_query.bvecs";
    const char* ground_truth_filepath =
            "/ssd_root/zhan4404/dataset/dataset/sift1B/gnd/idx_10M.ivecs";
    std::string index_store_path =
            "/home/zhan4404/disk-based/newfaiss_disk/faiss4_disk_vector/build/demos/index_folder/sift10m/sift10M.index";
    std::string disk_store_path =
            "/home/zhan4404/disk-based/newfaiss_disk/faiss4_disk_vector/build/demos/index_folder/sift10m/sift10M_ivfpqdisk.index";

    if (type == 0) {
        disk_build(
                base_filepath, nb, d, ratio, index_store_path, disk_store_path);
    } else if (type == 1) {
        search(query_filepath,
               ground_truth_filepath,
               nq,
               d,
               k,
               disk_store_path);
    }

    return 0;
}