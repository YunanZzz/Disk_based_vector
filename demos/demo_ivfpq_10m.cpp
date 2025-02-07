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
#include <faiss/IndexIVFPQDisk.h>
#include <faiss/IndexRefine.h>
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
        // Read the entire vector (4 bytes for dimension, followed by dim bytes for vector data)
        in.read(reinterpret_cast<char*>(buffer), 4 + dim * sizeof(float));

        // Skip the first 4 bytes (dimension), directly copy the float data
        std::memcpy(data + i * dim, buffer + 4, dim * sizeof(float));
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
void load_data_chunk_float(const char* filename, float* data, size_t start, size_t num, int dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(-1);
    }

    // Move file pointer to the start of the chunk
    in.seekg(start * (4 + dim * sizeof(float)), std::ios::beg);

    // Allocate a buffer for one vector (4 bytes for an index and dim * sizeof(float) bytes for the vector)
    unsigned char* buffer = new unsigned char[4 + dim * sizeof(float)];

    // Read each vector in the chunk
    for (size_t i = 0; i < num; i++) {
        // Read the entire chunk for one vector (4 bytes for index + dim * sizeof(float) for the data)
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

void add_chunk_to_index(faiss::IndexIVFPQDisk& index, float* chunk_data, size_t num_vectors, int dim) {
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

int main() {
    double t0 = elapsed();
    int d = 128; // dimension
    size_t nb = 1000000*10; 
    size_t nq = 100; // 10,000 query vectors
    size_t chunk_size = 1000000;  // Load 1 million vectors at a time
    size_t num_chunks = nb / chunk_size;
    // File paths
    const char* base_filepath = "/mnt/d/VectorDB/sift/sift10m_uint8/bigann_10m_base.bvecs";
    const char* query_filepath = "/mnt/d/VectorDB/sift/sift10m_uint8/bigann_query.bvecs";
    const char* ground_truth_filepath = "/mnt/d/VectorDB/sift/sift10m_uint8/idx_10M.ivecs";

    // const char* base_filepath = "/mnt/d/VectorDB/sift/sift_10M/sift10m/sift10m_base.fvecs";
    // const char* query_filepath = "/mnt/d/VectorDB/sift/sift_10M/sift10m/sift10m_query.fvecs";
    // char* ground_truth_filepath="/mnt/d/VectorDB/sift/sift_10M/sift10m/sift10m_groundtruth.ivecs";
    std::string index_store_path  = "/home/granthe/faiss/faiss/build/sift10M";

    // char* base_filepath ="/mnt/d/VectorDB/sift/sift/sift_base.fvecs";
    // char* query_filepath ="/mnt/d/VectorDB/sift/sift/sift_query.fvecs";
    // char* ground_truth_filepath="/mnt/d/VectorDB/sift/sift/sift_groundtruth.ivecs";
    
    // std::string index_store_path  = "/home/granthe/faiss/faiss/build/sift1M";
    //float* xq=NULL;
    //load_data_float(query_filepath,xq,nq,d);

    // {
    // srand(static_cast<int>(time(0)));
    // std::vector<float> trainvecs(nb / 50 * d);
    // std::vector<std::vector<float>> all_chunks(num_chunks);
    // for (size_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
    //     std::cout << "[Chunk " << chunk_id + 1 << "/" << num_chunks << "] Loading chunk..." << std::endl;
    //     all_chunks[chunk_id].resize(chunk_size * d);

    //     // Load the data into the vector
    //     load_data_chunk_float(base_filepath, all_chunks[chunk_id].data(), chunk_id * chunk_size, chunk_size, d);
    //     // Fill the training vectors from this chunk
    //     for (int i = 0; i < chunk_size / 50; i++) {  // Assuming you want to sample 2% of vectors
    //         int rng = rand() % chunk_size;
    //         for (int j = 0; j < d; j++) {
    //             trainvecs[chunk_id * (chunk_size / 50) * d + i * d + j] = static_cast<float>(all_chunks[chunk_id][rng * d + j]);
    //         }
    //     }
    // }
    size_t dd; // dimension
    size_t nt; // number of vectors
    int ratio = 50;

    size_t dd2; // dimension
    size_t nt2; // number of vectors

    float* xb = load_and_convert_to_float(base_filepath, &dd, &nt);
    float* xq = load_and_convert_to_float(query_filepath, &dd2, &nt2);

    std::vector<float> trainvecs(nb / ratio * d);
    srand(static_cast<int>(time(0)));
    for (int i = 0; i < nb / ratio; i++) {
        int rng = (rand() % (nb + 1));
        for (int j = 0; j < d; j++) {
            trainvecs[d * i + j] = xb[rng * d + j];
        }
    }

    int nlist = 2000;
    int k = 100;
            // load ground-truth and convert int to long
    size_t nq2;
    size_t kk; 
    int* gt_int = ivecs_read(ground_truth_filepath, &kk, &nq2);

    std::cout << "kk :" << kk << std::endl;
    for (int i = 0; i < kk * nq; i++) {
       // std::cout<< " gt["<< i <<"]:" <<  gt_int[i] << std::endl;
            
    }
    int m = 16;
    int nbits = 8;
    // Disk parameters
    int top_clusters = 5;
    float estimate_factor = 1.2;
    float estimate_factor_partial =1.3;
    float prune_factor = 10;
    faiss::IndexFlatL2 quantizer(d); // the other index

    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, nbits); 

    faiss::IndexRefineFlat refine_index(&index);
    refine_index.k_factor = 5;
    //faiss::IndexIVFPQDisk index(&quantizer, d, nlist, m, nbits, top_clusters, estimate_factor, prune_factor,index_store_path); 
    index.assign_replicas = 1;
    // faiss::IndexIVFFlat index(&quantizer, d, nlist);
    assert(!index.is_trained);
    printf("[%.3f s] ivf start to train\n",elapsed() - t0);
    refine_index.train(nb/ratio, trainvecs.data());
    printf("[%.3f s] ivf train finished\n",elapsed() - t0);
    assert(index.is_trained);
    printf("[%.3f s] ivf start to add\n",elapsed() - t0);

    // for (size_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
    //     std::cout << "[Chunk " << chunk_id + 1 << "/" << num_chunks << "] Adding to index..." << std::endl;

    //     // Use the stored chunk for adding vectors to the index
    //     add_chunk_to_index(index, all_chunks[chunk_id].data(), chunk_size, d);
    // }
    // int nadd = 4;
    // index.add_batch_num = nadd;
    // for(int i = 0; i < nadd; i++){
    //     index.add(nb/nadd, xb + (nb/nadd * i) * d);
    // }
    refine_index.add(nb, xb);

    // write_index(&index, "large.index");
    printf("[%.3f s] ivf add finished\n",elapsed() - t0);
    omp_set_num_threads(1);
    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        faiss::idx_t* gt;
        gt = new faiss::idx_t[k * nq];
        int jj=0;
        for (int i = 0; i < nq; i++) {
            for(int j = 0; j < k; j++){
                gt[jj] =  static_cast<faiss::idx_t>(gt_int[i*kk+j]); //long int / int
                jj+=1;
            }
            
        }
        delete[] gt_int;
        int arr[] = {3, 5, 7, 10, 20, 30, 40, 50, 70, 90, 110, 130, 150};
        //how to get length of array 
        for (int i : arr) {
            //print current i   
            printf("-----------now, the nprobe is=%d---------\n",i);
            index.nprobe = i;
            printf("[%.3f s] ivfflat start to search\n",elapsed() - t0);

            double t1 = elapsed();
            refine_index.search(nq, xq, k, D, I);
            double t2 = elapsed();
            double search_time = t2 - t1;

            printf("[%.3f s] ivfflat search finished\n",elapsed() - t0);
            
        // evaluate result
            
            int n2_100=0;
            for (int i = 0; i < nq; i++) {
                std::map<float, int> umap;
                for (int j = 0; j < k; j++) {              
                    umap.insert({gt[i*k+j], 0});
                }
                for (int l = 0; l < k; l++) {
                    //std::cout<< "I[" << i << "*" << k << "+" << l <<"]:" <<  I[i*k+l];
                    //std::cout<< " gt["<< i << "*" << k << "+" << l <<"]:" <<  gt[i*k+l] << std::endl;
                    if (umap.find(I[i*k+l])!= umap.end()){
                        n2_100++;                 
                    }
                }
                umap.clear();

            }
            printf("Intersection R@%d = %.4f\n", k, n2_100 / float(nq*k));
            printf("search time = %.4f\n",            search_time/nq*1000);
            printf("QPS         = %.4f\n",            1000.0/(search_time/nq*1000));
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

