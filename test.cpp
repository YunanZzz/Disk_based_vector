#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cstdlib>  // For malloc and free
#include <immintrin.h> // For AVX2 intrinsics

int main() {
    // Define the table size
    size_t table_size = 16 * 256;
    float* sim_table = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * table_size));

    // Initialize the sim_table with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < table_size; ++i) {
        sim_table[i] = dis(gen);
    }

    // Generate a set of random indices
    std::uniform_int_distribution<size_t> index_dist(0, table_size - 1);
    std::vector<size_t> index(table_size * 2); 
    for (size_t i = 0; i < table_size * 2; ++i) {
        index[i] = index_dist(gen);
    }

    float sum = 0.0f;
    __m256 vec_sum = _mm256_setzero_ps();  // AVX register to hold partial sums

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t repeat = 0; repeat < 20000 * 10000; ++repeat) {
        for (size_t i = 0; i < 16; i += 8) {  // Unroll loop in steps of 8 for AVX
            // Prefetching for the next access to sim_table elements
            if (i + 16 < 16) {
                _mm_prefetch(reinterpret_cast<const char*>(&sim_table[index[i + 16]]), _MM_HINT_T0);
            }

            // Create random indices
            size_t idx1 = index[i];
            size_t idx2 = index[(i + 3) % (table_size * 2)];
            size_t idx3 = index[(i + 100) % (table_size * 2)];
            size_t idx4 = index[std::max(i * 2 - 1, static_cast<size_t>(0)) % (table_size * 2)];
            
            // Load the values using AVX
            __m256 val1 = _mm256_set_ps(sim_table[idx1], sim_table[idx2], sim_table[idx3], sim_table[idx4], 
                                        sim_table[idx1], sim_table[idx2], sim_table[idx3], sim_table[idx4]);
            
            // Add values to the running sum
            vec_sum = _mm256_add_ps(vec_sum, val1);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    
    // Horizontal addition of elements in vec_sum to get the final scalar sum
    float temp[8];
    _mm256_store_ps(temp, vec_sum);
    for (int i = 0; i < 8; i++) {
        sum += temp[i];
    }

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Total time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Sum (to prevent optimization): " << sum << std::endl;

    // Free allocated memory
    std::free(sim_table);

    return 0;
}