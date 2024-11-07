/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <map>
#include <sys/stat.h>
#include <sys/time.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <iostream>
#include <faiss/index_io.h>
#include <fstream>
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
void load_data_float(const char* filename, float*& data, size_t num, int dim) {
    std::ifstream in(filename, std::ios::binary); // open file in binary mode
    if (!in.is_open()) {
        std::cout << "Error opening file" << std::endl;
        exit(-1);
    }

    // Allocate memory for the data as float to avoid overflow
    data = new float[num * dim];

    // Buffer to hold the entire vector (dimension field + actual vector data)
    unsigned char* buffer = new unsigned char[4 + dim];

    // Read each vector
    for (size_t i = 0; i < num; i++) {
        // Read the entire vector (4 bytes for dimension, followed by dim bytes for vector data)
        in.read((char*)buffer, 4 + dim);

        // Convert and copy the actual vector data (skipping the first 4 bytes, which represent the dimension)
        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = static_cast<float>(buffer[4 + j]);
        }
    }

    // Clean up
    delete[] buffer;
    in.close();
}
void load_data(const char* filename, unsigned char*& data, size_t num, int dim) {
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
        // Read the entire vector (4 bytes for dimension, followed by dim bytes for vector data)
        in.read((char*)buffer, 4 + dim);

        // Copy the actual vector data (skipping the first 4 bytes, which represent the dimension)
        std::memcpy(data + i * dim, buffer + 4, dim);
    }

    // Clean up
    delete[] buffer;
    in.close();
}
void load_data_chunk(const char* filename, unsigned char* data, size_t start, size_t num, int dim) {
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

void add_chunk_to_index(faiss::IndexIVFFlat& index, unsigned char* chunk_data, size_t num_vectors, int dim) {
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
int main() {
    double t0 = elapsed();
    int d = 128; // dimension
    size_t nb = 1000000*10; 
    size_t nq = 10000; // 10,000 query vectors
        size_t chunk_size = 100000;  // Load 1 million vectors at a time
    size_t num_chunks = nb / chunk_size;
    // File paths
    const char* base_filepath = "/ssd_root/zhan4404/dataset/dataset/sift1B/bigann_10m_base.bvecs";
    const char* query_filepath = "/ssd_root/zhan4404/dataset/dataset/sift1B/bigann_query.bvecs";
     float* xq=NULL;
     load_data_float(query_filepath,xq,nq,d);

    {
        srand(static_cast<int>(time(0)));
    std::vector<float> trainvecs(nb / 50 * d);
    std::vector<std::vector<unsigned char>> all_chunks(num_chunks);
        for (size_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        std::cout << "[Chunk " << chunk_id + 1 << "/" << num_chunks << "] Loading chunk..." << std::endl;
    all_chunks[chunk_id].resize(chunk_size * d);

    // Load the data into the vector
    load_data_chunk(base_filepath, all_chunks[chunk_id].data(), chunk_id * chunk_size, chunk_size, d);
        // Fill the training vectors from this chunk
        for (int i = 0; i < chunk_size / 50; i++) {  // Assuming you want to sample 2% of vectors
            int rng = rand() % chunk_size;
            for (int j = 0; j < d; j++) {
                trainvecs[chunk_id * (chunk_size / 50) * d + i * d + j] = static_cast<float>(all_chunks[chunk_id][rng * d + j]);
            }
        }
    }
       
    int nlist = 1000;
    int k = 100;
            // load ground-truth and convert int to long
    size_t nq2;
    size_t kk; 
    int* gt_int = ivecs_read("/ssd_root/zhan4404/dataset/dataset/sift1B/gnd/idx_10M.ivecs", &kk, &nq2);
    faiss::IndexFlatL2 quantizer(d); // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    assert(!index.is_trained);
    printf("[%.3f s] ivfflat start to train\n",elapsed() - t0);
    index.train(nb/50, trainvecs.data());
    printf("[%.3f s] ivfflat train finished\n",elapsed() - t0);
    assert(index.is_trained);
    printf("[%.3f s] ivfflat start to add\n",elapsed() - t0);

    for (size_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
    std::cout << "[Chunk " << chunk_id + 1 << "/" << num_chunks << "] Adding to index..." << std::endl;

    // Use the stored chunk for adding vectors to the index
    add_chunk_to_index(index, all_chunks[chunk_id].data(), chunk_size, d);
}
    // write_index(&index, "large.index");
    printf("[%.3f s] ivfflat add finished\n",elapsed() - t0);
     omp_set_num_threads(64);
    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
                faiss::idx_t* gt;
                        gt = new faiss::idx_t[100 * nq];
                        int jj=0;
        for (int i = 0; i < kk * nq; i++) {
            gt[jj] = gt_int[i]; //long int / int
            jj+=1;
            if((i+1)%100==0){
                i+=900;
            }

        }
        delete[] gt_int;
        int arr[] = {3, 5, 7, 10, 20,30,40,50, 70, 90, 110};
        //how to get length of array 
        for (int i : arr) {
        //print current i   
        printf("-----------now, the nprobe is=%d---------\n",i);
        index.nprobe = i;
        printf("[%.3f s] ivfflat start to search\n",elapsed() - t0);
        index.search(nq, xq, k, D, I);
        // //iterative query
        // for (int i = 0; i < nq; i++) {
        //     index.search(1, xq + i * d, k, D + i * k, I + i * k);
        // }
        printf("[%.3f s] ivfflat search finished\n",elapsed() - t0);
        
    // evaluate result

        int n2_100=0;
        for (int i = 0; i < nq; i++) {
            std::map<float, int> umap;
            for (int j = 0; j < k; j++) {              
                umap.insert({gt[i*k+j], 0});
            }
            for (int l = 0; l < k; l++) {
                
                if (umap.find(I[i*k+l])!= umap.end()){
                    n2_100++;                 
                }
            }
            umap.clear();

        }
        printf("Intersection R@100 = %.4f\n", n2_100 / float(nq*k));
        
    }


        // printf("I=\n");
        // for (int i = nq - 5; i < nq; i++) {
        //     for (int j = 0; j < k; j++)
        //         printf("%5zd ", I[i * k + j]);
        //     printf("\n");
        // }

        delete[] I;
        delete[] D;
    }

    // delete[] xb;
    delete[] xq;

    return 0;
}
}
