/**
 * Embedding Operations for Generative Engine
 * 
 * SIMD-optimized vector operations with runtime dispatch:
 * - AVX-512: 16 floats per operation (Intel 12th gen+)
 * - AVX2+FMA: 8 floats per operation
 * - SSE4.2: 4 floats per operation  
 * - Scalar fallback
 * 
 * Operations:
 * - Cosine similarity
 * - L2 distance
 * - Vector arithmetic (for analogies)
 * - Batch KNN search
 */

#pragma once

#include "hypercube/backend.hpp"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace hypercube {
namespace embedding {

// =============================================================================
// Horizontal Sum Helpers
// =============================================================================

#if defined(HAS_AVX512F) && HAS_AVX512F
inline float hsum_avx512(__m512 v) noexcept {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(lo, hi);
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif

#if defined(__AVX__)
inline float hsum_avx(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

#if defined(__SSE__)
inline float hsum_sse(__m128 v) noexcept {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

// =============================================================================
// AVX-512 Implementations (16 floats/op)
// =============================================================================

#if defined(HAS_AVX512F) && HAS_AVX512F
inline double cosine_similarity_avx512(const float* a, const float* b, size_t n) noexcept {
    __m512 dot_sum = _mm512_setzero_ps();
    __m512 norm_a_sum = _mm512_setzero_ps();
    __m512 norm_b_sum = _mm512_setzero_ps();
    
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        
        dot_sum = _mm512_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm512_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm512_fmadd_ps(vb, vb, norm_b_sum);
    }
    
    float dot = hsum_avx512(dot_sum);
    float norm_a = hsum_avx512(norm_a_sum);
    float norm_b = hsum_avx512(norm_b_sum);
    
    // Remainder
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0 || norm_b == 0) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

inline double l2_distance_avx512(const float* a, const float* b, size_t n) noexcept {
    __m512 sum = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    float result = hsum_avx512(sum);

    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return std::sqrt(result);
}

inline void vector_add_avx512(const float* a, const float* b, float* result, size_t n) noexcept {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&result[i], _mm512_add_ps(va, vb));
    }
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

inline void vector_sub_avx512(const float* a, const float* b, float* result, size_t n) noexcept {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&result[i], _mm512_sub_ps(va, vb));
    }
    for (; i < n; ++i) result[i] = a[i] - b[i];
}

inline void analogy_target_avx512(const float* a, const float* b, const float* c,
                                   float* result, size_t n) noexcept {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 vc = _mm512_loadu_ps(&c[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        _mm512_storeu_ps(&result[i], _mm512_add_ps(vc, diff));
    }
    for (; i < n; ++i) result[i] = c[i] + (a[i] - b[i]);
}
#endif

// =============================================================================
// AVX2+FMA Implementations (8 floats/op)
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
    
    float dot = hsum_avx(dot_sum);
    float norm_a = hsum_avx(norm_a_sum);
    float norm_b = hsum_avx(norm_b_sum);
    
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0 || norm_b == 0) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

inline double l2_distance_avx2(const float* a, const float* b, size_t n) noexcept {
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    float result = hsum_avx(sum);
    
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    
    return std::sqrt(result);
}

inline void vector_add_avx2(const float* a, const float* b, float* result, size_t n) noexcept {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&result[i], _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

inline void vector_sub_avx2(const float* a, const float* b, float* result, size_t n) noexcept {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&result[i], _mm256_sub_ps(va, vb));
    }
    for (; i < n; ++i) result[i] = a[i] - b[i];
}

inline void analogy_target_avx2(const float* a, const float* b, const float* c, 
                                 float* result, size_t n) noexcept {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(&result[i], _mm256_add_ps(vc, diff));
    }
    for (; i < n; ++i) result[i] = c[i] + (a[i] - b[i]);
}
#endif

// =============================================================================
// SSE4.2 Implementations (4 floats/op)
// =============================================================================

#if defined(__SSE4_2__) || defined(__SSE__)
inline double cosine_similarity_sse(const float* a, const float* b, size_t n) noexcept {
    __m128 dot_sum = _mm_setzero_ps();
    __m128 norm_a_sum = _mm_setzero_ps();
    __m128 norm_b_sum = _mm_setzero_ps();
    
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        
        dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
        norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
        norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));
    }
    
    float dot = hsum_sse(dot_sum);
    float norm_a = hsum_sse(norm_a_sum);
    float norm_b = hsum_sse(norm_b_sum);
    
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0 || norm_b == 0) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

inline double l2_distance_sse(const float* a, const float* b, size_t n) noexcept {
    __m128 sum = _mm_setzero_ps();
    
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    
    float result = hsum_sse(sum);
    
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    
    return std::sqrt(result);
}

inline void vector_add_sse(const float* a, const float* b, float* result, size_t n) noexcept {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&result[i], _mm_add_ps(va, vb));
    }
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

inline void vector_sub_sse(const float* a, const float* b, float* result, size_t n) noexcept {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&result[i], _mm_sub_ps(va, vb));
    }
    for (; i < n; ++i) result[i] = a[i] - b[i];
}

inline void analogy_target_sse(const float* a, const float* b, const float* c, 
                                float* result, size_t n) noexcept {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vc = _mm_loadu_ps(&c[i]);
        __m128 diff = _mm_sub_ps(va, vb);
        _mm_storeu_ps(&result[i], _mm_add_ps(vc, diff));
    }
    for (; i < n; ++i) result[i] = c[i] + (a[i] - b[i]);
}
#endif

// =============================================================================
// Scalar Fallback Implementations
// =============================================================================

inline double cosine_similarity_scalar(const float* a, const float* b, size_t n) noexcept {
    double dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }
    if (norm_a == 0 || norm_b == 0) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

inline double l2_distance_scalar(const float* a, const float* b, size_t n) noexcept {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline void vector_add_scalar(const float* a, const float* b, float* result, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) result[i] = a[i] + b[i];
}

inline void vector_sub_scalar(const float* a, const float* b, float* result, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) result[i] = a[i] - b[i];
}

inline void analogy_target_scalar(const float* a, const float* b, const float* c, 
                                   float* result, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) result[i] = c[i] + (a[i] - b[i]);
}

// =============================================================================
// Runtime-Dispatched Public API
// =============================================================================

/**
 * Compute cosine similarity with automatic SIMD dispatch
 */
inline double cosine_similarity(const float* a, const float* b, size_t n) noexcept {
    const auto level = Backend::simd_level();
    
#if defined(HAS_AVX512F) && HAS_AVX512F
    if (level >= SIMDLevel::AVX512) {
        return cosine_similarity_avx512(a, b, n);
    }
#endif
#if defined(__AVX2__)
    if (level >= SIMDLevel::AVX2) {
        return cosine_similarity_avx2(a, b, n);
    }
#endif
#if defined(__SSE4_2__) || defined(__SSE__)
    if (level >= SIMDLevel::SSE2) {
        return cosine_similarity_sse(a, b, n);
    }
#endif
    return cosine_similarity_scalar(a, b, n);
}

/**
 * Compute L2 distance with automatic SIMD dispatch
 */
inline double l2_distance(const float* a, const float* b, size_t n) noexcept {
    const auto level = Backend::simd_level();
    
#if defined(HAS_AVX512F) && HAS_AVX512F
    if (level >= SIMDLevel::AVX512) {
        return l2_distance_avx512(a, b, n);
    }
#endif
#if defined(__AVX2__)
    if (level >= SIMDLevel::AVX2) {
        return l2_distance_avx2(a, b, n);
    }
#endif
#if defined(__SSE4_2__) || defined(__SSE__)
    if (level >= SIMDLevel::SSE2) {
        return l2_distance_sse(a, b, n);
    }
#endif
    return l2_distance_scalar(a, b, n);
}

/**
 * Vector addition with automatic SIMD dispatch
 */
inline void vector_add(const float* a, const float* b, float* result, size_t n) noexcept {
    const auto level = Backend::simd_level();
    
#if defined(HAS_AVX512F) && HAS_AVX512F
    if (level >= SIMDLevel::AVX512) {
        vector_add_avx512(a, b, result, n);
        return;
    }
#endif
#if defined(__AVX2__)
    if (level >= SIMDLevel::AVX2) {
        vector_add_avx2(a, b, result, n);
        return;
    }
#endif
#if defined(__SSE4_2__) || defined(__SSE__)
    if (level >= SIMDLevel::SSE2) {
        vector_add_sse(a, b, result, n);
        return;
    }
#endif
    vector_add_scalar(a, b, result, n);
}

/**
 * Vector subtraction with automatic SIMD dispatch
 */
inline void vector_sub(const float* a, const float* b, float* result, size_t n) noexcept {
    const auto level = Backend::simd_level();
    
#if defined(HAS_AVX512F) && HAS_AVX512F
    if (level >= SIMDLevel::AVX512) {
        vector_sub_avx512(a, b, result, n);
        return;
    }
#endif
#if defined(__AVX2__)
    if (level >= SIMDLevel::AVX2) {
        vector_sub_avx2(a, b, result, n);
        return;
    }
#endif
#if defined(__SSE4_2__) || defined(__SSE__)
    if (level >= SIMDLevel::SSE2) {
        vector_sub_sse(a, b, result, n);
        return;
    }
#endif
    vector_sub_scalar(a, b, result, n);
}

/**
 * Analogy computation: result = c + (a - b)
 * Used for word2vec-style analogies ("king - man + woman = queen")
 */
inline void analogy_target(const float* a, const float* b, const float* c, 
                            float* result, size_t n) noexcept {
    const auto level = Backend::simd_level();
    
#if defined(HAS_AVX512F) && HAS_AVX512F
    if (level >= SIMDLevel::AVX512) {
        analogy_target_avx512(a, b, c, result, n);
        return;
    }
#endif
#if defined(__AVX2__)
    if (level >= SIMDLevel::AVX2) {
        analogy_target_avx2(a, b, c, result, n);
        return;
    }
#endif
#if defined(__SSE4_2__) || defined(__SSE__)
    if (level >= SIMDLevel::SSE2) {
        analogy_target_sse(a, b, c, result, n);
        return;
    }
#endif
    analogy_target_scalar(a, b, c, result, n);
}

// =============================================================================
// Batch KNN Search
// =============================================================================

struct SimilarityResult {
    size_t index;
    double similarity;
};

/**
 * Find k most similar vectors to query using cosine similarity
 * Uses automatic SIMD dispatch for distance computation
 */
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

/**
 * Find k nearest vectors using L2 distance
 */
inline std::vector<SimilarityResult> find_top_k_l2(
    const float* query,
    const float* candidates,
    size_t n,
    size_t dim,
    size_t k
) {
    std::vector<SimilarityResult> results;
    results.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        double dist = l2_distance(query, &candidates[i * dim], dim);
        results.push_back({i, dist});
    }
    
    // Partial sort for closest k (ascending distance)
    if (k < n) {
        std::partial_sort(results.begin(), results.begin() + k, results.end(),
            [](const SimilarityResult& a, const SimilarityResult& b) {
                return a.similarity < b.similarity;  // Ascending (closest first)
            });
        results.resize(k);
    } else {
        std::sort(results.begin(), results.end(),
            [](const SimilarityResult& a, const SimilarityResult& b) {
                return a.similarity < b.similarity;
            });
    }
    
    return results;
}

/**
 * Get the name of the currently active SIMD implementation
 */
inline const char* active_simd_implementation() {
    return simd_level_name(Backend::simd_level());
}

} // namespace embedding
} // namespace hypercube
