#include "hypercube/simd_intrinsics.hpp"
#include "hypercube/dispatch.h"
#include <immintrin.h>
#include <cmath>

// AVX2_VNNI implementations of SIMD kernels

namespace hypercube {

// Define global function pointers for dispatch system
DistanceL2Fn distance_l2_avx2_vnni = nullptr;
DistanceIPFn distance_ip_avx2_vnni = nullptr;
GemmF32Fn gemm_f32_avx2_vnni = nullptr;
DotProductDFn dot_product_d_avx2_vnni = nullptr;
DotProductFFn dot_product_f_avx2_vnni = nullptr;
ScaleInplaceDFn scale_inplace_d_avx2_vnni = nullptr;
SubtractScaledDFn subtract_scaled_d_avx2_vnni = nullptr;
NormDFn norm_d_avx2_vnni = nullptr;

namespace simd {

float dot_product(const float* a, const float* b, size_t n) {
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
    float sum = _mm256_cvtss_f32(sum_vec);
    // Handle remainder
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double dot_product_d(const double* a, const double* b, size_t n) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        sum_vec = _mm256_fmadd_pd(va, vb, sum_vec);
    }
    // Horizontal sum
    sum_vec = _mm256_add_pd(sum_vec, _mm256_permute2f128_pd(sum_vec, sum_vec, 1));
    sum_vec = _mm256_hadd_pd(sum_vec, sum_vec);
    double sum = _mm256_cvtsd_f64(sum_vec);
    // Handle remainder
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// VNNI-specific Int8 dot product using _mm256_dpbusd_epi32
int32_t dot_product_int8(const int8_t* a, const int8_t* b, size_t n) {
    __m256i sum_vec = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 31 < n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
        sum_vec = _mm256_dpbusd_epi32(sum_vec, va, vb);
    }
    // Horizontal sum of the 8 int32 values in sum_vec
    int32_t sum_array[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_array), sum_vec);
    int32_t sum = 0;
    for (int j = 0; j < 8; ++j) {
        sum += sum_array[j];
    }
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

float cosine_similarity(const float* a, const float* b, size_t n) {
    __m256 dot_vec = _mm256_setzero_ps();
    __m256 norm_a_vec = _mm256_setzero_ps();
    __m256 norm_b_vec = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);
        norm_a_vec = _mm256_fmadd_ps(va, va, norm_a_vec);
        norm_b_vec = _mm256_fmadd_ps(vb, vb, norm_b_vec);
    }
    // Horizontal sums
    dot_vec = _mm256_add_ps(dot_vec, _mm256_permute2f128_ps(dot_vec, dot_vec, 1));
    dot_vec = _mm256_hadd_ps(dot_vec, dot_vec);
    dot_vec = _mm256_hadd_ps(dot_vec, dot_vec);
    float dot = _mm256_cvtss_f32(dot_vec);
    norm_a_vec = _mm256_add_ps(norm_a_vec, _mm256_permute2f128_ps(norm_a_vec, norm_a_vec, 1));
    norm_a_vec = _mm256_hadd_ps(norm_a_vec, norm_a_vec);
    norm_a_vec = _mm256_hadd_ps(norm_a_vec, norm_a_vec);
    float norm_a_sq = _mm256_cvtss_f32(norm_a_vec);
    norm_b_vec = _mm256_add_ps(norm_b_vec, _mm256_permute2f128_ps(norm_b_vec, norm_b_vec, 1));
    norm_b_vec = _mm256_hadd_ps(norm_b_vec, norm_b_vec);
    norm_b_vec = _mm256_hadd_ps(norm_b_vec, norm_b_vec);
    float norm_b_sq = _mm256_cvtss_f32(norm_b_vec);
    // Handle remainder
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    float norm_a = sqrt(norm_a_sq);
    float norm_b = sqrt(norm_b_sq);
    return (norm_a > 1e-10f && norm_b > 1e-10f) ? dot / (norm_a * norm_b) : 0.0f;
}

void scale_inplace(double* v, double s, size_t n) {
    __m256d s_vec = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d vv = _mm256_loadu_pd(&v[i]);
        vv = _mm256_mul_pd(vv, s_vec);
        _mm256_storeu_pd(&v[i], vv);
    }
    // Handle remainder
    for (; i < n; ++i) {
        v[i] *= s;
    }
}

void subtract_scaled(double* a, const double* b, double s, size_t n) {
    __m256d s_vec = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        va = _mm256_fnmadd_pd(vb, s_vec, va);  // va - s*vb
        _mm256_storeu_pd(&a[i], va);
    }
    // Handle remainder
    for (; i < n; ++i) {
        a[i] -= s * b[i];
    }
}

double norm(const double* v, size_t n) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d vv = _mm256_loadu_pd(&v[i]);
        sum_vec = _mm256_fmadd_pd(vv, vv, sum_vec);
    }
    // Horizontal sum
    sum_vec = _mm256_add_pd(sum_vec, _mm256_permute2f128_pd(sum_vec, sum_vec, 1));
    sum_vec = _mm256_hadd_pd(sum_vec, sum_vec);
    double sum = _mm256_cvtsd_f64(sum_vec);
    // Handle remainder
    for (; i < n; ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

void normalize(double* v, size_t n) {
    double nrm = norm(v, n);
    if (nrm > 1e-12) {
        scale_inplace(v, 1.0 / nrm, n);
    }
}

double distance_l2(const float* a, const float* b, size_t n) {
    double sum_sq = 0.0;
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        // Reduce to double
        sq = _mm256_add_ps(sq, _mm256_permute2f128_ps(sq, sq, 1));
        sq = _mm256_hadd_ps(sq, sq);
        sq = _mm256_hadd_ps(sq, sq);
        float partial = _mm256_cvtss_f32(sq);
        sum_sq += static_cast<double>(partial);
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
            __m256 sum_vec = _mm256_setzero_ps();
            size_t p = 0;
            for (; p + 7 < k; p += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i * k + p]);
                __m256 b_vec = _mm256_set_ps(
                    B[(p + 7) * n + j], B[(p + 6) * n + j], B[(p + 5) * n + j], B[(p + 4) * n + j],
                    B[(p + 3) * n + j], B[(p + 2) * n + j], B[(p + 1) * n + j], B[p * n + j]
                );
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            // Reduce sum_vec to float
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

} // namespace simd

// Initialize dispatch function pointers
struct Avx2VnniInit {
    Avx2VnniInit() {
        distance_l2_avx2_vnni = [](const float* a, const float* b, size_t n) -> double {
            return simd::distance_l2(a, b, n);
        };
        distance_ip_avx2_vnni = [](const float* a, const float* b, size_t n) -> double {
            return simd::distance_ip(a, b, n);
        };
        gemm_f32_avx2_vnni = [](float alpha, const float* A, size_t m, size_t k,
                               const float* B, size_t n, float beta, float* C) -> void {
            simd::gemm_f32(alpha, A, m, k, B, n, beta, C);
        };
        dot_product_d_avx2_vnni = [](const double* a, const double* b, size_t n) -> double {
            return simd::dot_product_d(a, b, n);
        };
        dot_product_f_avx2_vnni = [](const float* a, const float* b, size_t n) -> float {
            return simd::dot_product(a, b, n);
        };
        scale_inplace_d_avx2_vnni = [](double* v, double s, size_t n) -> void {
            simd::scale_inplace(v, s, n);
        };
        subtract_scaled_d_avx2_vnni = [](double* a, const double* b, double s, size_t n) -> void {
            simd::subtract_scaled(a, b, s, n);
        };
        norm_d_avx2_vnni = [](const double* v, size_t n) -> double {
            return simd::norm(v, n);
        };
    }
} avx2_vnni_init;

} // namespace hypercube