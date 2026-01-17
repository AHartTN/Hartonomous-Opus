#include "hypercube/simd_intrinsics.hpp"
#include "hypercube/dispatch.h"
#include <immintrin.h>
#include <cmath>

// AVX512_VNNI implementations of SIMD kernels

namespace hypercube {

// Define global function pointers for dispatch system
DistanceL2Fn distance_l2_avx512_vnni = nullptr;
DistanceIPFn distance_ip_avx512_vnni = nullptr;
GemmF32Fn gemm_f32_avx512_vnni = nullptr;
DotProductDFn dot_product_d_avx512_vnni = nullptr;
DotProductFFn dot_product_f_avx512_vnni = nullptr;
ScaleInplaceDFn scale_inplace_d_avx512_vnni = nullptr;
SubtractScaledDFn subtract_scaled_d_avx512_vnni = nullptr;
NormDFn norm_d_avx512_vnni = nullptr;

namespace simd {

float dot_product(const float* a, const float* b, size_t n) {
    __m512 sum_vec = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
    }
    float sum = _mm512_reduce_add_ps(sum_vec);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double dot_product_d(const double* a, const double* b, size_t n) {
    __m512d sum_vec = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        sum_vec = _mm512_fmadd_pd(va, vb, sum_vec);
    }
    double sum = _mm512_reduce_add_pd(sum_vec);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float cosine_similarity(const float* a, const float* b, size_t n) {
    __m512 dot_vec = _mm512_setzero_ps();
    __m512 norm_a_vec = _mm512_setzero_ps();
    __m512 norm_b_vec = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        dot_vec = _mm512_fmadd_ps(va, vb, dot_vec);
        norm_a_vec = _mm512_fmadd_ps(va, va, norm_a_vec);
        norm_b_vec = _mm512_fmadd_ps(vb, vb, norm_b_vec);
    }
    float dot = _mm512_reduce_add_ps(dot_vec);
    float norm_a = _mm512_reduce_add_ps(norm_a_vec);
    float norm_b = _mm512_reduce_add_ps(norm_b_vec);
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    return (norm_a > 1e-10f && norm_b > 1e-10f) ? dot / (norm_a * norm_b) : 0.0f;
}

void scale_inplace(double* v, double s, size_t n) {
    __m512d s_vec = _mm512_set1_pd(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d vv = _mm512_loadu_pd(&v[i]);
        vv = _mm512_mul_pd(vv, s_vec);
        _mm512_storeu_pd(&v[i], vv);
    }
    for (; i < n; ++i) {
        v[i] *= s;
    }
}

void subtract_scaled(double* a, const double* b, double s, size_t n) {
    __m512d s_vec = _mm512_set1_pd(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        va = _mm512_fnmadd_pd(s_vec, vb, va);
        _mm512_storeu_pd(&a[i], va);
    }
    for (; i < n; ++i) {
        a[i] -= s * b[i];
    }
}

double norm(const double* v, size_t n) {
    __m512d sum_vec = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d vv = _mm512_loadu_pd(&v[i]);
        sum_vec = _mm512_fmadd_pd(vv, vv, sum_vec);
    }
    double sum = _mm512_reduce_add_pd(sum_vec);
    for (; i < n; ++i) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

void normalize(double* v, size_t n) {
    double nrm = norm(v, n);
    if (nrm > 1e-12) {
        scale_inplace(v, 1.0 / nrm, n);
    }
}

// VNNI-specific Int8 dot product using _mm512_dpbusd_epi32 (16 int8 elements per operation)
int32_t dot_product_int8(const int8_t* a, const int8_t* b, size_t n) {
    __m512i sum_vec = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {  // 512 bits / 8 bits = 64 int8 elements
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&a[i]));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&b[i]));
        sum_vec = _mm512_dpbusd_epi32(sum_vec, va, vb);
    }
    int32_t sum = _mm512_reduce_add_epi32(sum_vec);
    // Handle remainder
    for (; i < n; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return sum;
}

// Quantized dot product combining Int8 dot product with scaling/dequantization
float quantized_dot_product(const int8_t* a, const int8_t* b, float scale_a, float scale_b, size_t n) {
    int32_t raw_dot = dot_product_int8(a, b, n);
    return static_cast<float>(raw_dot) * scale_a * scale_b;
}

double distance_l2(const float* a, const float* b, size_t n) {
    double sum_sq = 0.0;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
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

double distance_ip(const float* a, const float* b, size_t n) {
    return static_cast<double>(dot_product(a, b, n));
}

void gemm_f32(float alpha, const float* A, size_t m, size_t k,
              const float* B, size_t n, float beta, float* C) {
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

} // namespace simd

// Initialize dispatch function pointers
struct Avx512VnniInit {
    Avx512VnniInit() {
        distance_l2_avx512_vnni = [](const float* a, const float* b, size_t n) -> double {
            return simd::distance_l2(a, b, n);
        };
        distance_ip_avx512_vnni = [](const float* a, const float* b, size_t n) -> double {
            return simd::distance_ip(a, b, n);
        };
        gemm_f32_avx512_vnni = [](float alpha, const float* A, size_t m, size_t k,
                                 const float* B, size_t n, float beta, float* C) -> void {
            simd::gemm_f32(alpha, A, m, k, B, n, beta, C);
        };
        dot_product_d_avx512_vnni = [](const double* a, const double* b, size_t n) -> double {
            return simd::dot_product_d(a, b, n);
        };
        dot_product_f_avx512_vnni = [](const float* a, const float* b, size_t n) -> float {
            return simd::dot_product(a, b, n);
        };
        scale_inplace_d_avx512_vnni = [](double* v, double s, size_t n) -> void {
            simd::scale_inplace(v, s, n);
        };
        subtract_scaled_d_avx512_vnni = [](double* a, const double* b, double s, size_t n) -> void {
            simd::subtract_scaled(a, b, s, n);
        };
        norm_d_avx512_vnni = [](const double* v, size_t n) -> double {
            return simd::norm(v, n);
        };
    }
} avx512_vnni_init;

} // namespace hypercube