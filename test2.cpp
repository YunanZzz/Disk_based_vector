#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cstdlib>  // For malloc and free
#include <immintrin.h> // For AVX2 intrinsics

int main() {
    size_t table_size = 16 * 256;
    float* sim_table = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * table_size));
    // Initialize the sim_table with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < table_size; ++i) {
        sim_table[i] = dis(gen);
    }
    // Generate a set of random indexes
    std::uniform_int_distribution<size_t> index_dist(0, table_size - 1);
    std::vector<size_t> index(table_size * 2); 
    for (size_t i = 0; i < table_size * 2; ++i) {
        index[i] = index_dist(gen);
    }

    float sum = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();   
    for (size_t repeat = 0; repeat < 20000 * 10000; ++repeat) {
        for (size_t i = 0; i < 16; i++) {
            //create random index
            size_t index1 = index[i];
            size_t index2 = index[(i + 3) % (table_size * 2)];
            size_t index3 = index[(i + 100) % (table_size * 2)];
            size_t index4 = index[std::max(i * 2 - 1, static_cast<size_t>(0)) % (table_size * 2)];
            sum += sim_table[index1];
            sum += sim_table[index2];
            sum += sim_table[index3];
            sum += sim_table[index4];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Total time " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Sum (to prevent optimization): " << sum << std::endl;

    // Free allocated memory
    std::free(sim_table);

    return 0;
}