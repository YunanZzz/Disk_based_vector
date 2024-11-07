#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <fstream>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

using namespace std::chrono;
using namespace faiss;


void read_data(
        float* data,
        const std::string data_file_path,
        size_t n,
        size_t d) {
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error:" << data_file_path << std::endl;
        exit(-1);
    }
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * d), d * sizeof(float));
    }
    in.close();
}

void read_groundtruth(
        int* data,
        const std::string data_file_path,
        size_t n,
        size_t d) {
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error:" << data_file_path << std::endl;
        exit(-1);
    }
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * d), d * sizeof(int));
    }
    in.close();
}



int main() {

     // dimension of the vectors to index
    int d = 128;
    int M = 32;
    faiss::IndexHNSWFlat index(d, M, faiss::METRIC_L2);
    index.hnsw.efConstruction = 100; 
    

    size_t nb = 1000000;
    int nq = 10000;
    std::vector<float> database(nb * d);
    std::vector<float> queries(nq * d);
    std::vector<int> gt_nns(nq * 100);

    //std::string data_path         = "D:\\VectorDB\\WEAVESS_data\\siftsmall\\siftsmall_base.fvecs";
    //std::string query_path        = "D:\\VectorDB\\WEAVESS_data\\siftsmall\\siftsmall_query.fvecs";
    //std::string groundtruth_path  = "D:\\VectorDB\\WEAVESS_data\\siftsmall\\siftsmall_groundtruth.ivecs";
    //std::string result_path       = "D:\\VectorDB\\WEAVESS_data\\siftsmall\\siftsmall_result.txt";

    //std::string data_path         = "/mnt/d/VectorDB/WEAVESS_data/siftsmall/siftsmall_base.fvecs";
    //std::string query_path        = "/mnt/d/VectorDB/WEAVESS_data/siftsmall/siftsmall_query.fvecs";
    //std::string groundtruth_path  = "/mnt/d/VectorDB/WEAVESS_data/siftsmall/siftsmall_groundtruth.ivecs";
    //std::string result_path       = "/mnt/d/VectorDB/WEAVESS_data/siftsmall/siftsmall_result_linux.txt";
    
    std::string data_path         = "/mnt/d/VectorDB/sift/sift/sift_base.fvecs";
    std::string query_path        = "/mnt/d/VectorDB/sift/sift/sift_query.fvecs";
    std::string groundtruth_path  = "/mnt/d/VectorDB/sift/sift/sift_groundtruth.ivecs";
    std::string result_path       = "/mnt/d/VectorDB/sift/sift/sift_hnsw1M_result_linux.txt";

    read_data(database.data(), data_path, nb, d);
    read_data(queries.data(), query_path, nq, d);
    read_groundtruth(gt_nns.data(), groundtruth_path, nq, 100);

    std::cout << "Adding" << std::endl;
    auto start_add = high_resolution_clock::now();
    index.add(nb, database.data());
    auto end_add = high_resolution_clock::now();
    auto t_add = duration_cast<microseconds>(end_add - start_add).count();

    faiss::write_index(&index, "D:\\VectorDB\\index_file\\hnsw_small.index");

    std::ofstream outFile(result_path, std::ios::app);
    std::cout << "Build: " << t_add << std::endl;
    outFile << "Build: " << t_add << std::endl;

     // searching the database
    printf("Searching ...\n");
    index.hnsw.efSearch = 128;
    int k = 1;
    for (int efS_factor = 1; efS_factor < 12; efS_factor++)
    {
        index.hnsw.efSearch = 16 * efS_factor;

        std::cout << "Searching " << index.hnsw.efSearch << std::endl;
        outFile << "Searching " << index.hnsw.efSearch << std::endl;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());
        auto end = high_resolution_clock::now();

        int recalls = 0;
        for (size_t i = 0; i < nq; ++i) {
            for (int n = 0; n < k; n++) {
                for (int m = 0; m < k; m++) {
                    if (nns[i * k + n] == gt_nns[i * 100 + m]) {
                        recalls += 1;
                    }
                }
            }
        }
        float recall = 1.0f * recalls / (k * nq);
        auto t = duration_cast<microseconds>(end - start).count();
        int qps = nq * 1.0f * 1000 * 1000 / t;

        std::cout << "Recall@" << k << ": " << recall << ", QPS: " << qps << std::endl;
        outFile << "Recall@" << k << ": " << recall << ", QPS: " << qps << std::endl;
    }
    k = 5;
    for (int efS_factor = 1; efS_factor < 12; efS_factor++) {
        index.hnsw.efSearch = 16 * efS_factor;

        std::cout << "Searching " << index.hnsw.efSearch << std::endl;
        outFile << "Searching " << index.hnsw.efSearch << std::endl;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());
        auto end = high_resolution_clock::now();

        int recalls = 0;
        for (size_t i = 0; i < nq; ++i) {
            for (int n = 0; n < k; n++) {
                for (int m = 0; m < k; m++) {
                    if (nns[i * k + n] == gt_nns[i * 100 + m]) {
                        recalls += 1;
                    }
                }
            }
        }
        float recall = 1.0f * recalls / (k * nq);
        auto t = duration_cast<microseconds>(end - start).count();
        int qps = nq * 1.0f * 1000 * 1000 / t;

        std::cout << "Recall@" << k << ": " << recall << ", QPS: " << qps
                  << std::endl;
        outFile << "Recall@" << k << ": " << recall << ", QPS: " << qps
                << std::endl;
    }

    k = 10;
    for (int efS_factor = 1; efS_factor < 12; efS_factor++) {
        index.hnsw.efSearch = 16 * efS_factor;

        std::cout << "Searching " << index.hnsw.efSearch << std::endl;
        outFile << "Searching " << index.hnsw.efSearch << std::endl;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());
        auto end = high_resolution_clock::now();

        int recalls = 0;
        for (size_t i = 0; i < nq; ++i) {
            for (int n = 0; n < k; n++) {
                for (int m = 0; m < k; m++) {
                    if (nns[i * k + n] == gt_nns[i * 100 + m]) {
                        recalls += 1;
                    }
                }
            }
        }
        float recall = 1.0f * recalls / (k * nq);
        auto t = duration_cast<microseconds>(end - start).count();
        int qps = nq * 1.0f * 1000 * 1000 / t;

        std::cout << "Recall@" << k << ": " << recall << ", QPS: " << qps
                  << std::endl;
        outFile << "Recall@" << k << ": " << recall << ", QPS: " << qps
                << std::endl;
    }
    k = 20;
    for (int efS_factor = 1; efS_factor < 12; efS_factor++) {
        index.hnsw.efSearch = 16 * efS_factor;

        std::cout << "Searching " << index.hnsw.efSearch << std::endl;
        outFile << "Searching " << index.hnsw.efSearch << std::endl;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());
        auto end = high_resolution_clock::now();

        int recalls = 0;
        for (size_t i = 0; i < nq; ++i) {
            for (int n = 0; n < k; n++) {
                for (int m = 0; m < k; m++) {
                    if (nns[i * k + n] == gt_nns[i * 100 + m]) {
                        recalls += 1;
                    }
                }
            }
        }
        float recall = 1.0f * recalls / (k * nq);
        auto t = duration_cast<microseconds>(end - start).count();
        int qps = nq * 1.0f * 1000 * 1000 / t;

        std::cout << "Recall@" << k << ": " << recall << ", QPS: " << qps
                  << std::endl;
        outFile << "Recall@" << k << ": " << recall << ", QPS: " << qps
                << std::endl;
    }

    k = 50;
    for (int efS_factor = 1; efS_factor < 12; efS_factor++) {
        index.hnsw.efSearch = 16 * efS_factor;

        std::cout << "Searching " << index.hnsw.efSearch << std::endl;
        outFile << "Searching " << index.hnsw.efSearch << std::endl;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());
        auto end = high_resolution_clock::now();

        int recalls = 0;
        for (size_t i = 0; i < nq; ++i) {
            for (int n = 0; n < k; n++) {
                for (int m = 0; m < k; m++) {
                    if (nns[i * k + n] == gt_nns[i * 100 + m]) {
                        recalls += 1;
                    }
                }
            }
        }
        float recall = 1.0f * recalls / (k * nq);
        auto t = duration_cast<microseconds>(end - start).count();
        int qps = nq * 1.0f * 1000 * 1000 / t;

        std::cout << "Recall@" << k << ": " << recall << ", QPS: " << qps
                  << std::endl;
        outFile << "Recall@" << k << ": " << recall << ", QPS: " << qps
                << std::endl;
    }

    k = 100;
    for (int efS_factor = 1; efS_factor < 12; efS_factor++) {
        index.hnsw.efSearch = 16 * efS_factor;

        std::cout << "Searching " << index.hnsw.efSearch << std::endl;
        outFile << "Searching " << index.hnsw.efSearch << std::endl;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());
        auto end = high_resolution_clock::now();

        int recalls = 0;
        for (size_t i = 0; i < nq; ++i) {
            for (int n = 0; n < k; n++) {
                for (int m = 0; m < k; m++) {
                    if (nns[i * k + n] == gt_nns[i * 100 + m]) {
                        recalls += 1;
                    }
                }
            }
        }
        float recall = 1.0f * recalls / (k * nq);
        auto t = duration_cast<microseconds>(end - start).count();
        int qps = nq * 1.0f * 1000 * 1000 / t;

        std::cout << "Recall@" << k << ": " << recall << ", QPS: " << qps << std::endl;
        outFile << "Recall@" << k << ": " << recall << ", QPS: " << qps << std::endl;
    }
    outFile.close();
    
    return 0;
}