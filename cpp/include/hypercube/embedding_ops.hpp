/**
 * Embedding Operations for Generative Engine
 * 
 * SIMD-optimized vector operations for:
 * - Cosine similarity
 * - L2 distance
 * - Vector arithmetic (for analogies)
 * - Batch KNN search
 * 
 * These are the heavy-lift operations called from SQL.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <span>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace hypercube {
namespace embedding {

// =============================================================================
// SIMD Cosine Similarity
// =============================================================================

#if defined(__AVX2__)
inline double cosine_similarity_avx2(const float* a, const float* b, size_t n) noexcept {
    __m256 dot_sum = _mm256_setzero_ps();
    __m256 norm_a_sum = _mm256_setzero_ps();
    __m256 norm_b_sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        
        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
    }
    
    // Horizontal sum of 8 floats
    auto hsum = [](const __m256 v) -> float {
        __m128 lo = _mm256_extractf128_ps(v, 0);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    };
    
    float dot = hsum(dot_sum);
    float norm_a = hsum(norm_a_sum);
    float norm_b = hsum(norm_b_sum);
    
    // Handle remainder
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0 || norm_b == 0) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}
#endif

inline double cosine_similarity_portable(const float* a, const float* b, size_t n) noexcept {
    double dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }
    if (norm_a == 0 || norm_b == 0) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

inline double cosine_similarity(const float* a, const float* b, size_t n) noexcept {
#if defined(__AVX2__)
    return cosine_similarity_avx2(a, b, n);
#else
    return cosine_similarity_portable(a, b, n);
#endif
}

// =============================================================================
// SIMD L2 Distance
// =============================================================================

#if defined(__AVX2__)
inline double l2_distance_avx2(const float* a, const float* b, size_t n) noexcept {
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal sum
    __m128 lo = _mm256_extractf128_ps(sum, 0);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 total = _mm_add_ps(lo, hi);
    total = _mm_hadd_ps(total, total);
    total = _mm_hadd_ps(total, total);
    float result = _mm_cvtss_f32(total);
    
    // Handle remainder
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    
    return std::sqrt(result);
}
#endif

inline double l2_distance_portable(const float* a, const float* b, size_t n) noexcept {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline double l2_distance(const float* a, const float* b, size_t n) noexcept {
#if defined(__AVX2__)
    return l2_distance_avx2(a, b, n);
#else
    return l2_distance_portable(a, b, n);
#endif
}

// =============================================================================
// Vector Arithmetic
// =============================================================================

// result = a + b
inline void vector_add(const float* a, const float* b, float* result, size_t n) noexcept {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&result[i], _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) result[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < n; ++i) result[i] = a[i] + b[i];
#endif
}

// result = a - b
inline void vector_sub(const float* a, const float* b, float* result, size_t n) noexcept {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&result[i], _mm256_sub_ps(va, vb));
    }
    for (; i < n; ++i) result[i] = a[i] - b[i];
#else
    for (size_t i = 0; i < n; ++i) result[i] = a[i] - b[i];
#endif
}

// =============================================================================
// Batch KNN Search
// =============================================================================

struct SimilarityResult {
    size_t index;
    double similarity;
};

// Find k most similar vectors to query
inline std::vector<SimilarityResult> find_top_k_cosine(
    const float* query,
    const float* candidates,  // n * dim floats, row-major
    size_t n,
    size_t dim,
    size_t k
) {
    std::vector<SimilarityResult> results;
    results.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        double sim = cosine_similarity(query, &candidates[i * dim], dim);
        results.push_back({i, sim});
    }
    
    // Partial sort for top k
    if (k < n) {
        std::partial_sort(results.begin(), results.begin() + k, results.end(),
            [](const SimilarityResult& a, const SimilarityResult& b) {
                return a.similarity > b.similarity;  // Descending
            });
        results.resize(k);
    } else {
        std::sort(results.begin(), results.end(),
            [](const SimilarityResult& a, const SimilarityResult& b) {
                return a.similarity > b.similarity;
            });
    }
    
    return results;
}

// =============================================================================
// Analogy Vector Computation
// =============================================================================

// Compute target = c + (a - b) for word2vec-style analogy
// "king - man + woman = queen"
inline void analogy_target(
    const float* a,  // positive anchor (e.g., king)
    const float* b,  // negative anchor (e.g., man)
    const float* c,  // query (e.g., woman)
    float* result,
    size_t dim
) noexcept {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 res = _mm256_add_ps(vc, diff);
        _mm256_storeu_ps(&result[i], res);
    }
    for (; i < dim; ++i) {
        result[i] = c[i] + (a[i] - b[i]);
    }
#else
    for (size_t i = 0; i < dim; ++i) {
        result[i] = c[i] + (a[i] - b[i]);
    }
#endif
}

} // namespace embedding
} // namespace hypercube
