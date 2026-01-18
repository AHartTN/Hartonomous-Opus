#include "hypercube/simd_intrinsics.hpp"
#include "hypercube/dispatch.h"
#include <cmath>

// SSE42 implementations of SIMD kernels

namespace hypercube {

namespace sse42 {

// Horizontal sum helper for __m128 (float)
inline float hsum_sse_ps(__m128 v) noexcept {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Horizontal sum helper for __m128d (double)
inline double hsum_sse_pd(__m128d v) noexcept {
    __m128d sum = _mm_add_pd(v, _mm_shuffle_pd(v, v, 1));
    return _mm_cvtsd_f64(sum);
}

float dot_product(const float* a, const float* b, size_t n) {
    __m128 sum_vec = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(va, vb));
    }
    float sum = hsum_sse_ps(sum_vec);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double dot_product_d(const double* a, const double* b, size_t n) {
    __m128d sum_vec = _mm_setzero_pd();
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(va, vb));
    }
    double sum = hsum_sse_pd(sum_vec);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double distance_l2(const float* a, const float* b, size_t n) {
    __m128 sum_vec = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 diff = _mm_sub_ps(va, vb);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(diff, diff));
    }
    float sum_sq = hsum_sse_ps(sum_vec);
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

double distance_ip(const float* a, const float* b, size_t n) {
    return static_cast<double>(dot_product(a, b, n));
}

void gemm_f32(float alpha, const float* A, size_t m, size_t k,
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
            float sum = hsum_sse_ps(sum_vec);
            for (; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

float cosine_similarity(const float* a, const float* b, size_t n) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    return (norm_a > 1e-10f && norm_b > 1e-10f) ? dot / (norm_a * norm_b) : 0.0f;
}

void scale_inplace(double* v, double s, size_t n) {
    __m128d s_vec = _mm_set1_pd(s);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d vv = _mm_loadu_pd(&v[i]);
        vv = _mm_mul_pd(vv, s_vec);
        _mm_storeu_pd(&v[i], vv);
    }
    for (; i < n; ++i) {
        v[i] *= s;
    }
}

void subtract_scaled(double* a, const double* b, double s, size_t n) {
    __m128d s_vec = _mm_set1_pd(s);
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        va = _mm_sub_pd(va, _mm_mul_pd(s_vec, vb));
        _mm_storeu_pd(&a[i], va);
    }
    for (; i < n; ++i) {
        a[i] -= s * b[i];
    }
}

double norm(const double* v, size_t n) {
    __m128d sum_vec = _mm_setzero_pd();
    size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d vv = _mm_loadu_pd(&v[i]);
        sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(vv, vv));
    }
    double sum = hsum_sse_pd(sum_vec);
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

} // namespace sse42
} // namespace hypercube