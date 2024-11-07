#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

using namespace std::chrono;


using idx_t = faiss::idx_t;

void read_data(float* data, const std::string data_file_path, size_t n, size_t d) {
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

void read_groundtruth(int* data, const std::string data_file_path, size_t n, size_t d) {
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

    
    int d = 128;
    size_t nb = 1000000;
    int nq = 10000;

    std::vector<float> database(nb * d);
    std::vector<float> query(nq * d);
    std::vector<int> gt_nns(nq * 100);

    // std::string data_path         = "D:\\VectorDB\\sift\\sift\\sift_base.fvecs";
    // std::string query_path        = "D:\\VectorDB\\sift\\sift\\sift_query.fvecs";
    // std::string groundtruth_path  = "D:\\VectorDB\\sift\\sift\\sift_groundtruth.ivecs";
    // std::string result_path       = "D:\\VectorDB\\RESULT_SET\\ivf_pq_wsl.txt";

    std::string data_path         = "/mnt/d/VectorDB/sift/sift/sift_base.fvecs";
    std::string query_path        = "/mnt/d/VectorDB/sift/sift/sift_query.fvecs";
    std::string groundtruth_path  = "/mnt/d/VectorDB/sift/sift/sift_groundtruth.ivecs";
    std::string result_path       = "/mnt/d/VectorDB/RESULT_SET/ivf_pq_wsl.txt";

    /*
    //std::string data_path         = "D:\\VectorDB\\WEAVESS_data\\siftsmall\\siftsmall_base.fvecs";
    //std::string query_path        = "D:\\VectorDB\\WEAVESS_data\\siftsmall\\siftsmall_query.fvecs";
    //std::string groundtruth_path  = "D:\\VectorDB\\WEAVESS_data\\siftsmall\\siftsmall_groundtruth.ivecs";
    //std::string result_path       = "D:\\VectorDB\\WEAVESS_data\\siftsmall\\siftsmall_result.txt";


    */

    read_data(database.data(), data_path, nb, d);
    read_data(query.data(), query_path, nq, d);
    read_groundtruth(gt_nns.data(), groundtruth_path, nq, 100);

    int nlist =4096;
    int m = 64;
    int nbits = 4;


    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, nbits);

    assert(!index.is_trained);
    std::cout << "Training" << std::endl;
    index.train(nb, database.data());
    assert(index.is_trained);
    std::cout << "Adding" << std::endl;
    index.add(nb, database.data());

    std::ofstream outFile(result_path, std::ios::app);
    if (!outFile)
    {
        std::cout << "error open!" << std::endl;
        return 0;
    }

    std::cout << m << " subvector " << nbits << " pq_nbits"
              << " PQ equals " << m * nbits / d << " Bits" << std::endl;
    outFile << m << " subvector " << nbits << " pq_nbits"
              << " PQ equals " << m * nbits / d << " Bits" << std::endl;

    //omp_set_num_threads(1);
    int k = 1;
    std::cout << "pq parameters: m = " << m << " nbits = " << nbits << std::endl;
    outFile << "pq parameters: m = " << m << " nbits = " << nbits << std::endl;
    for (int i = 1; i < 10; i++) {
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);
        index.nprobe = 3 * i;
        std::cout << "Searching probe:" << index.nprobe << std::endl;
        outFile << "Searching probe:" << index.nprobe << std::endl;
        auto start = high_resolution_clock::now();
        index.search(nq, query.data(), k, dis.data(), nns.data());
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

    k = 100;
    std::cout << "pq parameters: m = " << m << " nbits = " << nbits << std::endl;
    for (int i = 1; i < 10; i++) {
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);
        index.nprobe = 10 * i;
        std::cout << "Searching probe:" << index.nprobe << std::endl;
        outFile << "Searching probe:" << index.nprobe << std::endl;
        auto start = high_resolution_clock::now();
        index.search(nq, query.data(), k, dis.data(), nns.data());
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
}
