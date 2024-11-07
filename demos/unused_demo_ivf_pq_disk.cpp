#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <fstream>
#include <iostream>
#include <chrono>
#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQDisk.h>

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
    // std::string data_path         = "D:\\VectorDB\\sift\\sift\\sift_base.fvecs";
    // std::string query_path        = "D:\\VectorDB\\sift\\sift\\sift_query.fvecs";
    // std::string groundtruth_path  = "D:\\VectorDB\\sift\\sift\\sift_groundtruth.ivecs";
    // std::string index_store_path  = "D:\\VectorDB\\sift\\sift1m_test\\sift1M";
    // std::string result_path       = "D:\\VectorDB\\RESULT_SET\\ivf_pq_disk.txt";

    std::string data_path         = "/mnt/d/VectorDB/sift/sift/sift_base.fvecs";
    std::string query_path        = "/mnt/d/VectorDB/sift/sift/sift_query.fvecs";
    std::string groundtruth_path  = "/mnt/d/VectorDB/sift/sift/sift_groundtruth.ivecs";
    std::string index_store_path  = "/mnt/d/VectorDB/sift/sift1m_test/sift1M";
    std::string result_path       = "/mnt/d/VectorDB/RESULT_SET/ivf_pq_disk_wsl.txt";


    std::vector<float> database(nb * d);
    std::vector<float> query(nq * d);
    std::vector<int> gt_nns(nq * 100);
    read_data(database.data(), data_path, nb, d);
    read_data(query.data(), query_path, nq, d);
    read_groundtruth(gt_nns.data(), groundtruth_path, nq, 100);

    // IVF inverted file list
    int nlist =1024;

    // PQ parameters
    int m = 16;
    int nbits = 8;

    // Disk parameters
    int top_clusters = 5;
    float estimate_factor = 1.2;

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFPQDisk index(&quantizer, d, nlist, m, nbits, top_clusters, estimate_factor, index_store_path);

    assert(!index.is_trained);
    std::cout << "Training" << std::endl;
    index.train(nb, database.data());
    assert(index.is_trained);
    std::cout << "Adding" << std::endl;
    index.add(nb, database.data());

    index.initial_location(database.data());

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
     
     std::cout << "pq parameters: m = " << m << " nbits = " << nbits << std::endl;
     outFile << "pq parameters: m = " << m << " nbits = " << nbits << std::endl;
     int k = 1;
     for (int t = 0; t < 6; t++) {
         index.set_top(t);
         for (float f = 1.0; f < 1.8; f += 0.1) {
             index.set_estimate_factor(f);
             
             std::cout << "Top clusters:" << index.get_top()<< "  estimate factor:" << index.get_estimate_factor();
             outFile << "Top clusters:" << index.get_top()<< "  estimate factor:" << index.get_estimate_factor();

             for (int i = 1; i < 10; i++) {
                 std::vector<faiss::idx_t> nns(k * nq);
                 std::vector<float> dis(k * nq);
                 index.nprobe = 3 * i;
                 std::cout << "Searching " << index.nprobe << std::endl;
                 outFile << "Searching " << index.nprobe << std::endl;
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

                 std::cout << "Recall@" << k << ": " << recall
                           << ", QPS: " << qps << std::endl;
                 outFile << "Recall@" << k << ": " << recall << ", QPS: " << qps
                         << std::endl;
             }
         }
     }


     k = 100;
     for (int t = 0; t < 6; t++) {
         index.set_top(t);
         for (float f = 1.0; f < 1.8; f += 0.1) {
             index.set_estimate_factor(f);
             std::cout << "Top clusters:" << index.get_top()<< "  estimate factor:" << index.get_estimate_factor();
             outFile << "Top clusters:" << index.get_top()<< "  estimate factor:" << index.get_estimate_factor();
             for (int i = 1; i < 10; i++) {
                 std::vector<faiss::idx_t> nns(k * nq);
                 std::vector<float> dis(k * nq);
                 index.nprobe = 10 * i;
                 std::cout << "Searching " << index.nprobe << std::endl;
                 outFile << "Searching " << index.nprobe << std::endl;
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

                 std::cout << "Recall@" << k << ": " << recall
                           << ", QPS: " << qps << std::endl;
                 outFile << "Recall@" << k << ": " << recall << ", QPS: " << qps
                         << std::endl;
             }

         }
     }
    outFile.close();
}
