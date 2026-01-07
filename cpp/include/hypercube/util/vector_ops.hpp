/**
 * Vector Operations - SIMD-accelerated parallel vector normalization
 *
 * Provides high-performance vector normalization for batches of embeddings
 * with support for L2 and L1 norms, SIMD acceleration, and thread pool integration.
 */

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "hypercube/thread_pool.hpp"
#include "hypercube/cpu_features.hpp"

namespace hypercube {
namespace util {

/**
 * Normalization types for vector operations
 */
enum class NormalizationType {
    L2,  // Euclidean norm: sqrt(sum(x_i^2))
    L1   // Manhattan norm: sum(|x_i|)
};

/**
 * SIMD-accelerated vector normalization utilities
 */
class VectorOps {
public:
    /**
     * Compute L2 norm (Euclidean) of a vector using SIMD
     */
    static float compute_l2_norm_simd(const std::vector<float>& vec) {
        size_t n = vec.size();
        if (n == 0) return 0.0f;

        // Use runtime AVX2 detection if available
        if (hypercube::cpu_features::has_avx2()) {
#if defined(__AVX2__)
            __m256 sum_vec = _mm256_setzero_ps();
            size_t i = 0;

            // Process 8 floats at a time
            for (; i + 8 <= n; i += 8) {
                __m256 v = _mm256_loadu_ps(&vec[i]);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(v, v));
            }

            // Horizontal sum of the 8 values
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

            float sum = _mm256_cvtss_f32(sum_vec) + _mm256_cvtss_f32(_mm256_permute2f128_ps(sum_vec, sum_vec, 1));

            // Handle remaining elements
            for (; i < n; ++i) {
                sum += vec[i] * vec[i];
            }

            return std::sqrt(sum);
#else
            // Fallback if AVX2 not compiled in
#endif
        }

        // Fallback to scalar computation
        float sum = 0.0f;
        for (float v : vec) {
            sum += v * v;
        }
        return std::sqrt(sum);
    }

    /**
     * Compute L1 norm (Manhattan) of a vector using SIMD
     */
    static float compute_l1_norm_simd(const std::vector<float>& vec) {
        size_t n = vec.size();
        if (n == 0) return 0.0f;

        // Use runtime AVX2 detection if available
        if (hypercube::cpu_features::has_avx2()) {
#if defined(__AVX2__)
            __m256 sum_vec = _mm256_setzero_ps();
            size_t i = 0;

            // Process 8 floats at a time
            for (; i + 8 <= n; i += 8) {
                __m256 v = _mm256_loadu_ps(&vec[i]);
                // Compute absolute values
                __m256 abs_v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
                sum_vec = _mm256_add_ps(sum_vec, abs_v);
            }

            // Horizontal sum
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

            float sum = _mm256_cvtss_f32(sum_vec) + _mm256_cvtss_f32(_mm256_permute2f128_ps(sum_vec, sum_vec, 1));

            // Handle remaining elements
            for (; i < n; ++i) {
                sum += std::abs(vec[i]);
            }

            return sum;
#else
            // Fallback if AVX2 not compiled in
#endif
        }

        // Fallback to scalar computation
        float sum = 0.0f;
        for (float v : vec) {
            sum += std::abs(v);
        }
        return sum;
    }

    /**
     * Normalize a single vector in-place
     */
    static void normalize_vector(std::vector<float>& vec, NormalizationType norm_type) {
        if (vec.empty()) return;

        float norm;
        if (norm_type == NormalizationType::L2) {
            norm = compute_l2_norm_simd(vec);
        } else {
            norm = compute_l1_norm_simd(vec);
        }

        if (norm == 0.0f) return;  // Avoid division by zero

        // Normalize using SIMD for better performance
        normalize_vector_by_factor(vec, 1.0f / norm);
    }

    /**
     * Normalize a vector by dividing by a scalar factor using SIMD
     */
    static void normalize_vector_by_factor(std::vector<float>& vec, float factor) {
        size_t n = vec.size();

        // Use runtime AVX2 detection if available
        if (hypercube::cpu_features::has_avx2()) {
#if defined(__AVX2__)
            __m256 factor_vec = _mm256_set1_ps(factor);
            size_t i = 0;

            for (; i + 8 <= n; i += 8) {
                __m256 v = _mm256_loadu_ps(&vec[i]);
                v = _mm256_mul_ps(v, factor_vec);
                _mm256_storeu_ps(&vec[i], v);
            }

            // Handle remaining elements
            for (; i < n; ++i) {
                vec[i] *= factor;
            }
            return;
#else
            // Fallback if AVX2 not compiled in
#endif
        }

        // Scalar fallback
        for (size_t i = 0; i < n; ++i) {
            vec[i] *= factor;
        }
    }
};

/**
 * Parallel batch vector normalization
 *
 * Normalizes a batch of high-dimensional vectors using thread pool parallelism
 * and SIMD acceleration for optimal performance.
 *
 * @param vectors Batch of vectors to normalize (modified in-place)
 * @param norm_type Type of normalization (L2 or L1)
 */
inline void normalize_vectors_parallel(std::vector<std::vector<float>>& vectors,
                                      NormalizationType norm_type = NormalizationType::L2) {
    if (vectors.empty()) return;

    // Use thread pool for parallel processing
    auto& pool = hypercube::ThreadPool::instance();

    pool.parallel_for(0, vectors.size(), [&](size_t i) {
        VectorOps::normalize_vector(vectors[i], norm_type);
    });
}

} // namespace util
} // namespace hypercube