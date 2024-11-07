#include <iostream>
#include <chrono>
#include <random>
#include <immintrin.h> // For SIMD instructions
// First version (non-SIMD)
float fvec_L2sqr(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

// SIMD version
float fvec_L2sqr_simd(const float* x, const float* y, const size_t L) {
    alignas(32) float TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // Use alignas(32) instead of PORTABLE_ALIGN32
    uint32_t num_blk16 = L >> 4;
    uint32_t l = L & 0b1111;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    for (uint32_t i = 0; i < num_blk16; i++) {
        v1 = _mm256_loadu_ps(x);
        v2 = _mm256_loadu_ps(y);
        x += 8;
        y += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(x);
        v2 = _mm256_loadu_ps(y);
        x += 8;
        y += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    for (uint32_t i = 0; i < l / 8; i++) {
        v1 = _mm256_loadu_ps(x);
        v2 = _mm256_loadu_ps(y);
        x += 8;
        y += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(TmpRes, sum);

    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    for (uint32_t i = 0; i < l % 8; i++) {
        float tmp = (*x) - (*y);
        ret += tmp * tmp;
        x++;
        y++;
    }
    return ret;
}
float fvec_L2sqr_simd2(const float* x, const float* y, const size_t L) {
    alignas(32) float TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // Ensuring alignment
    uint32_t num_blk16 = L >> 4; // Number of 16-element blocks
    uint32_t remainder = L & 0b1111; // Remainder for processing

    __m256 sum = _mm256_setzero_ps(); // Initialize the sum to zero
    __m256 v1, v2, diff;

    // Main loop to process in blocks of 16 elements (2x 8 floats per iteration)
    for (uint32_t i = 0; i < num_blk16; i++) {
        // Load 8 floats from both vectors, subtract, and accumulate squared difference
        v1 = _mm256_loadu_ps(x); // Load unaligned 8 floats from x
        v2 = _mm256_loadu_ps(y); // Load unaligned 8 floats from y
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_fmadd_ps(diff, diff, sum); // Fused multiply-add

        x += 8; // Move pointers ahead by 8 floats
        y += 8;

        // Repeat for the next 8 floats in the same loop iteration
        v1 = _mm256_loadu_ps(x);
        v2 = _mm256_loadu_ps(y);
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_fmadd_ps(diff, diff, sum);

        x += 8;
        y += 8;
    }

    // Handle the remaining elements that are not multiples of 16
    if (remainder >= 8) {
        v1 = _mm256_loadu_ps(x);
        v2 = _mm256_loadu_ps(y);
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_fmadd_ps(diff, diff, sum);

        x += 8;
        y += 8;
        remainder -= 8;
    }

    // Process any remaining elements that don't fit into a full 8-float SIMD block
    for (uint32_t i = 0; i < remainder; i++) {
        float tmp = (*x) - (*y);
        TmpRes[0] += tmp * tmp;
        x++;
        y++;
    }

    // Sum up the values in the SIMD register
    _mm256_store_ps(TmpRes, sum);

    // Reduce the results from the SIMD register to a single float value
    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] +
                TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    return ret;
}
int main() {
    const size_t dimension = 128;
    const size_t num_vectors = 10000;

    // Allocate memory for query vector and vector set
    float query_vector[dimension];
    float vector_set[num_vectors][dimension];

    // Random number generator for vectors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Fill the query vector and vector set with random values
    for (size_t i = 0; i < dimension; i++) {
        query_vector[i] = static_cast<float>(dis(gen));
    }

    for (size_t i = 0; i < num_vectors; i++) {
        for (size_t j = 0; j < dimension; j++) {
            vector_set[i][j] = static_cast<float>(dis(gen));
        }
    }

    // Test L2 distance using the regular function
    auto start = std::chrono::high_resolution_clock::now();
    float result_non_simd = 0;
    for (size_t i = 0; i < num_vectors; i++) {
        result_non_simd += fvec_L2sqr(query_vector, vector_set[i], dimension);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_non_simd = end - start;
    std::cout << "Time for non-SIMD version: " << elapsed_non_simd.count() << " seconds" << std::endl;
    std::cout << "Result from non-SIMD version: " << result_non_simd << std::endl;

    // Test L2 distance using the SIMD function
    start = std::chrono::high_resolution_clock::now();
    float result_simd = 0;
    for (size_t i = 0; i < num_vectors; i++) {
        result_simd += fvec_L2sqr_simd(query_vector, vector_set[i], dimension);
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_simd = end - start;
    std::cout << "Time for SIMD version: " << elapsed_simd.count() << " seconds" << std::endl;
    std::cout << "Result from SIMD version: " << result_simd << std::endl;

        // Test L2 distance using the SIMD2 function
    start = std::chrono::high_resolution_clock::now();
    float result_simd2 = 0;
    for (size_t i = 0; i < num_vectors; i++) {
        result_simd2 += fvec_L2sqr_simd2(query_vector, vector_set[i], dimension);
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_simd2 = end - start;
    std::cout << "Time for SIMD2 version: " << elapsed_simd2.count() << " seconds" << std::endl;
    std::cout << "Result from SIMD2 version: " << result_simd2 << std::endl;

    return 0;
}
