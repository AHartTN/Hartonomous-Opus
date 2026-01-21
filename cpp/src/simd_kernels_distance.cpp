#include "hypercube/simd_intrinsics.hpp"
#include "hypercube/cpu_features.hpp"
#include <immintrin.h>
#include <cmath>
#include <stdexcept>

// Unified SIMD Distance Kernel Implementations with Runtime Dispatch
// This file contains all distance-related kernels with runtime capability checks.

namespace hypercube {
namespace kernels {
namespace impl {

// =============================================================================
// Distance L2 (Euclidean) implementations
// =============================================================================

double distance_l2_baseline(const float* a, const float* b, size_t n) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

double distance_l2_sse42(const float* a, const float* b, size_t n) {
    // SSE4.2 implementation with manual horizontal sum
    size_t i = 0;
    double sum_sq = 0.0;

    // Process 4 floats at a time
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 sq = _mm_mul_ps(diff, diff);

        // Manual horizontal sum (more compatible than _mm_dp_ps)
        __m128 shuf = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(sq, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        sum_sq += _mm_cvtss_f32(sums);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq);
}

double distance_l2_avx2(const float* a, const float* b, size_t n) {
    if (!cpu_features::check_os_avx_support()) {
        throw cpu_features::CpuFeatureException("AVX not safely supported by OS");
    }

    // AVX2 implementation
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum_vec = _mm256_add_ps(sum_vec, sq);
    }

    // Horizontal sum
    sum_vec = _mm256_add_ps(sum_vec, _mm256_permute2f128_ps(sum_vec, sum_vec, 1));
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    double sum_sq = _mm256_cvtss_f32(sum_vec);

    // Handle remainder
    for (; i < n; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq);
}

#ifdef __AVX512F__
double distance_l2_avx512(const float* a, const float* b, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    double sum_sq = 0.0;
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        __m512 sq = _mm512_mul_ps(diff, diff);
        sum_sq += _mm512_reduce_add_ps(sq);
    }

    // Handle remainder
    for (; i < n; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq);
}
#endif

double distance_l2_amx(const float* a, const float* b, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    // For now, fall back to AVX-512 implementation
#ifdef __AVX512F__
    return distance_l2_avx512(a, b, n);
#else
    // Fallback to AVX2 if AVX512 not available
    return distance_l2_avx2(a, b, n);
#endif
}

// Distance IP implementations are now in the vector file

// Distance IP (Inner Product) implementations
double distance_ip_baseline(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

double distance_ip_sse42(const float* a, const float* b, size_t n) {
    size_t i = 0;
    double sum = 0.0;

    // Process 4 floats at a time with manual horizontal sum
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 prod = _mm_mul_ps(va, vb);

        // Manual horizontal sum
        __m128 shuf = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sums = _mm_add_ps(prod, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        sum += _mm_cvtss_f32(sums);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    return sum;
}

double distance_ip_avx2(const float* a, const float* b, size_t n) {
    if (!cpu_features::check_os_avx_support()) {
        throw cpu_features::CpuFeatureException("AVX not safely supported by OS");
    }

    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
    }

    // Horizontal sum
    sum_vec = _mm256_add_ps(sum_vec, _mm256_permute2f128_ps(sum_vec, sum_vec, 1));
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    double sum = _mm256_cvtss_f32(sum_vec);

    // Handle remainder
    for (; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    return sum;
}

#ifdef __AVX512F__
double distance_ip_avx512(const float* a, const float* b, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    double sum = 0.0;
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        sum += _mm512_reduce_add_ps(_mm512_mul_ps(va, vb));
    }

    // Handle remainder
    for (; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    return sum;
}
#endif

double distance_ip_amx(const float* a, const float* b, size_t n) {
    if (!cpu_features::check_os_amx_support()) {
        throw cpu_features::CpuFeatureException("AMX not safely supported by OS");
    }

    // For now, fall back to AVX-512 implementation
#ifdef __AVX512F__
    return distance_ip_avx512(a, b, n);
#else
    // Fallback to AVX2 if AVX512 not available
    return distance_ip_avx2(a, b, n);
#endif
}

// GEMM implementations
void gemm_f32_baseline(float alpha, const float* A, size_t m, size_t k,
                      const float* B, size_t n, float beta, float* C) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

void gemm_f32_sse42(float alpha, const float* A, size_t m, size_t k,
                   const float* B, size_t n, float beta, float* C) {
    // SSE4.2 matrix multiplication - vectorized over inner dimension
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            __m128 sum_vec = _mm_setzero_ps();
            size_t p = 0;
            for (; p + 4 <= k; p += 4) {
                __m128 va = _mm_loadu_ps(&A[i * k + p]);
                __m128 vb = _mm_setr_ps(B[p * n + j], B[(p+1) * n + j], B[(p+2) * n + j], B[(p+3) * n + j]);
                sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(va, vb));
            }

            // Horizontal sum
            __m128 shuf = _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(sum_vec, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            float sum = _mm_cvtss_f32(sums);

            // Handle remainder
            for (; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

void gemm_f32_avx2(float alpha, const float* A, size_t m, size_t k,
                  const float* B, size_t n, float beta, float* C) {
    if (!cpu_features::check_os_avx_support()) {
        throw cpu_features::CpuFeatureException("AVX not safely supported by OS");
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            __m256 sum_vec = _mm256_setzero_ps();
            size_t p = 0;
            for (; p + 8 <= k; p += 8) {
                __m256 va = _mm256_loadu_ps(&A[i * k + p]);
                __m256 vb = _mm256_set_ps(
                    B[(p + 7) * n + j], B[(p + 6) * n + j], B[(p + 5) * n + j], B[(p + 4) * n + j],
                    B[(p + 3) * n + j], B[(p + 2) * n + j], B[(p + 1) * n + j], B[p * n + j]
                );
                sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
            }

            // Horizontal sum
            sum_vec = _mm256_add_ps(sum_vec, _mm256_permute2f128_ps(sum_vec, sum_vec, 1));
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            float sum = _mm256_cvtss_f32(sum_vec);

            // Handle remainder
            for (; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

#ifdef __AVX512F__
void gemm_f32_avx512(float alpha, const float* A, size_t m, size_t k,
                    const float* B, size_t n, float beta, float* C) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            __m512 sum_vec = _mm512_setzero_ps();
            size_t p = 0;
            for (; p + 16 <= k; p += 16) {
                __m512 a_vec = _mm512_loadu_ps(&A[i * k + p]);
                // For AVX512, we can use gather or set_ps. Using set_ps for consistency
                __m512 b_vec = _mm512_set_ps(
                    B[(p + 15) * n + j], B[(p + 14) * n + j], B[(p + 13) * n + j], B[(p + 12) * n + j],
                    B[(p + 11) * n + j], B[(p + 10) * n + j], B[(p + 9) * n + j], B[(p + 8) * n + j],
                    B[(p + 7) * n + j], B[(p + 6) * n + j], B[(p + 5) * n + j], B[(p + 4) * n + j],
                    B[(p + 3) * n + j], B[(p + 2) * n + j], B[(p + 1) * n + j], B[p * n + j]
                );
                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            float sum = _mm512_reduce_add_ps(sum_vec);

            // Handle remainder
            for (; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}
#endif

void gemm_f32_amx(float alpha, const float* A, size_t m, size_t k,
                 const float* B, size_t n, float beta, float* C) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    // Simplified AMX GEMM - real implementation would use tile operations
#ifdef __AVX512F__
    gemm_f32_avx512(alpha, A, m, k, B, n, beta, C);
#else
    // Fallback to AVX2 if AVX512 not available
    gemm_f32_avx2(alpha, A, m, k, B, n, beta, C);
#endif
}

// =============================================================================
// Distance IP (Inner Product) Kernel
// =============================================================================

struct DistanceIP {
    static double compute(const float* a, const float* b, size_t n,
                         const cpu_features::RuntimeCpuFeatures& features) {
        // Use capability-based dispatch for optimal kernel selection
        auto preference = cpu_features::select_kernel_implementation(features, "distance_ip");

        try {
            // Try primary implementation
            switch (preference.primary) {
                case cpu_features::KernelCapability::AVX512F:
                case cpu_features::KernelCapability::AVX512_VNNI:
                    if (features.safe_avx512_execution) return avx512_impl(a, b, n);
                    break;
                case cpu_features::KernelCapability::AVX2:
                case cpu_features::KernelCapability::AVX2_VNNI:
                    if (features.safe_avx_execution && features.avx2) return avx2_impl(a, b, n);
                    break;
                case cpu_features::KernelCapability::SSE42:
                    if (features.sse4_2) return sse42_impl(a, b, n);
                    break;
                case cpu_features::KernelCapability::AMX:
                    if (features.safe_amx_execution) return amx_impl(a, b, n);
                    break;
                default:
                    break;
            }
        } catch (const cpu_features::CpuFeatureException&) {
            // Primary failed, try fallback
        }

        try {
            // Try fallback implementation
            switch (preference.fallback) {
                case cpu_features::KernelCapability::AVX512F:
                case cpu_features::KernelCapability::AVX512_VNNI:
                    if (features.safe_avx512_execution) return avx512_impl(a, b, n);
                    break;
                case cpu_features::KernelCapability::AVX2:
                case cpu_features::KernelCapability::AVX2_VNNI:
                    if (features.safe_avx_execution && features.avx2) return avx2_impl(a, b, n);
                    break;
                case cpu_features::KernelCapability::SSE42:
                    if (features.sse4_2) return sse42_impl(a, b, n);
                    break;
                case cpu_features::KernelCapability::AMX:
                    if (features.safe_amx_execution) return amx_impl(a, b, n);
                    break;
                default:
                    break;
            }
        } catch (const cpu_features::CpuFeatureException&) {
            // Fallback failed, use emergency
        }

        // Emergency fallback - always BASELINE
        return baseline_impl(a, b, n);
    }

private:
    static double baseline_impl(const float* a, const float* b, size_t n) {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
        return sum;
    }

    static double sse42_impl(const float* a, const float* b, size_t n) {
        size_t i = 0;
        double sum = 0.0;

        // Process 4 floats at a time with manual horizontal sum
        for (; i + 3 < n; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 prod = _mm_mul_ps(va, vb);

            // Manual horizontal sum
            __m128 shuf = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(prod, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            sum += _mm_cvtss_f32(sums);
        }

        // Handle remaining elements
        for (; i < n; ++i) {
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }

        return sum;
    }

    static double avx2_impl(const float* a, const float* b, size_t n) {
        if (!cpu_features::check_os_avx_support()) {
            throw cpu_features::CpuFeatureException("AVX not safely supported by OS");
        }

        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
        }

        // Horizontal sum
        sum_vec = _mm256_add_ps(sum_vec, _mm256_permute2f128_ps(sum_vec, sum_vec, 1));
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        double sum = _mm256_cvtss_f32(sum_vec);

        // Handle remainder
        for (; i < n; ++i) {
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }

        return sum;
    }

    static double avx512_impl(const float* a, const float* b, size_t n) {
#ifdef __AVX512F__
        if (!cpu_features::check_os_avx512_support()) {
            throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
        }

        double sum = 0.0;
        size_t i = 0;
        for (; i + 15 < n; i += 16) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            sum += _mm512_reduce_add_ps(_mm512_mul_ps(va, vb));
        }

        // Handle remainder
        for (; i < n; ++i) {
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }

        return sum;
#else
        // AVX512 not supported by compiler
        throw cpu_features::CpuFeatureException("AVX-512 not supported by compiler");
#endif
    }

    static double amx_impl(const float* a, const float* b, size_t n) {
        if (!cpu_features::check_os_amx_support()) {
            throw cpu_features::CpuFeatureException("AMX not safely supported by OS");
        }

        // For now, fall back to AVX-512 implementation
#ifdef __AVX512F__
        return distance_ip_avx512(a, b, n);
#else
        // Fallback to AVX2 if AVX512 not available
        return avx2_impl(a, b, n);
#endif
    }
};

} // namespace impl
} // namespace kernels
} // namespace hypercube