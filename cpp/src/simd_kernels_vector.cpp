#include "hypercube/simd_intrinsics.hpp"
#include "hypercube/cpu_features.hpp"
#include <immintrin.h>
#include <cmath>
#include <stdexcept>

// Unified SIMD Vector Kernel Implementations with Runtime Dispatch
// This file consolidates all ISA-specific vector kernels into unified implementations
// with runtime capability checks instead of compile-time guards.

namespace hypercube {
namespace kernels {
namespace impl {

// Dot product double precision implementations
double dot_product_d_baseline(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double dot_product_d_sse42(const double* a, const double* b, size_t n) {
    // SSE4.2 handles 2 doubles at a time
    size_t i = 0;
    double sum = 0.0;

    for (; i + 1 < n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d prod = _mm_mul_pd(va, vb);
        __m128d shuf = _mm_shuffle_pd(prod, prod, 0x1);
        prod = _mm_add_pd(prod, shuf);
        sum += _mm_cvtsd_f64(prod);
    }

    // Handle remaining element
    if (i < n) {
        sum += a[i] * b[i];
    }

    return sum;
}

double dot_product_d_avx2(const double* a, const double* b, size_t n) {
    if (!cpu_features::check_os_avx_support()) {
        throw cpu_features::CpuFeatureException("AVX not safely supported by OS");
    }

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

#ifdef __AVX512F__
double dot_product_d_avx512(const double* a, const double* b, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    __m512d sum_vec = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        sum_vec = _mm512_fmadd_pd(va, vb, sum_vec);
    }
    double sum = _mm512_reduce_add_pd(sum_vec);

    // Handle remainder
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}
#endif

double dot_product_d_amx(const double* a, const double* b, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    // Fallback to AVX-512 for now
#ifdef __AVX512F__
    return dot_product_d_avx512(a, b, n);
#else
    // Fallback to AVX2 if AVX512 not available
    return dot_product_d_avx2(a, b, n);
#endif
}

// Dot product float precision implementations
float dot_product_f_baseline(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float dot_product_f_sse42(const float* a, const float* b, size_t n) {
    __m128 sum_vec = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(va, vb));
    }

    // Horizontal sum
    __m128 shuf = _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(sum_vec, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float sum = _mm_cvtss_f32(sums);

    // Handle remainder
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

float dot_product_f_avx2(const float* a, const float* b, size_t n) {
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
    float sum = _mm256_cvtss_f32(sum_vec);

    // Handle remainder
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

#ifdef __AVX512F__
float dot_product_f_avx512(const float* a, const float* b, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    __m512 sum_vec = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
    }
    float sum = _mm512_reduce_add_ps(sum_vec);

    // Handle remainder
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}
#endif

float dot_product_f_amx(const float* a, const float* b, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    // Simplified AMX implementation
#ifdef __AVX512F__
    return dot_product_f_avx512(a, b, n);
#else
    // Fallback to AVX2 if AVX512 not available
    return dot_product_f_avx2(a, b, n);
#endif
}

// Scale inplace double precision implementations
void scale_inplace_d_baseline(double* v, double s, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        v[i] *= s;
    }
}

void scale_inplace_d_sse42(double* v, double s, size_t n) {
    __m128d s_vec = _mm_set1_pd(s);
    size_t i = 0;

    for (; i + 1 < n; i += 2) {
        __m128d vec = _mm_loadu_pd(v + i);
        vec = _mm_mul_pd(vec, s_vec);
        _mm_storeu_pd(v + i, vec);
    }

    // Handle remaining element
    if (i < n) {
        v[i] *= s;
    }
}

void scale_inplace_d_avx2(double* v, double s, size_t n) {
    if (!cpu_features::check_os_avx_support()) {
        throw cpu_features::CpuFeatureException("AVX not safely supported by OS");
    }

    __m256d s_vec = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d vec = _mm256_loadu_pd(v + i);
        vec = _mm256_mul_pd(vec, s_vec);
        _mm256_storeu_pd(v + i, vec);
    }

    // Handle remainder
    for (; i < n; ++i) {
        v[i] *= s;
    }
}

#ifdef __AVX512F__
void scale_inplace_d_avx512(double* v, double s, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    __m512d s_vec = _mm512_set1_pd(s);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m512d vec = _mm512_loadu_pd(v + i);
        vec = _mm512_mul_pd(vec, s_vec);
        _mm512_storeu_pd(v + i, vec);
    }

    // Handle remainder
    for (; i < n; ++i) {
        v[i] *= s;
    }
}
#endif

void scale_inplace_d_amx(double* v, double s, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    // Fallback to AVX-512
#ifdef __AVX512F__
    scale_inplace_d_avx512(v, s, n);
#else
    // Fallback to AVX2 if AVX512 not available
    scale_inplace_d_avx2(v, s, n);
#endif
}

// Subtract scaled double precision implementations
void subtract_scaled_d_baseline(double* a, const double* b, double s, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] -= s * b[i];
    }
}

void subtract_scaled_d_sse42(double* a, const double* b, double s, size_t n) {
    __m128d s_vec = _mm_set1_pd(s);
    size_t i = 0;

    for (; i + 1 < n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d scaled_b = _mm_mul_pd(s_vec, vb);
        va = _mm_sub_pd(va, scaled_b);
        _mm_storeu_pd(a + i, va);
    }

    // Handle remaining element
    if (i < n) {
        a[i] -= s * b[i];
    }
}

void subtract_scaled_d_avx2(double* a, const double* b, double s, size_t n) {
    if (!cpu_features::check_os_avx_support()) {
        throw cpu_features::CpuFeatureException("AVX not safely supported by OS");
    }

    __m256d s_vec = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        va = _mm256_fnmadd_pd(s_vec, vb, va);  // va - s*vb
        _mm256_storeu_pd(&a[i], va);
    }

    // Handle remainder
    for (; i < n; ++i) {
        a[i] -= s * b[i];
    }
}

#ifdef __AVX512F__
void subtract_scaled_d_avx512(double* a, const double* b, double s, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    __m512d s_vec = _mm512_set1_pd(s);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        va = _mm512_fnmadd_pd(s_vec, vb, va);  // va - s*vb
        _mm512_storeu_pd(&a[i], va);
    }

    // Handle remainder
    for (; i < n; ++i) {
        a[i] -= s * b[i];
    }
}
#endif

void subtract_scaled_d_amx(double* a, const double* b, double s, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    // Fallback to AVX-512
#ifdef __AVX512F__
    subtract_scaled_d_avx512(a, b, s, n);
#else
    // Fallback to AVX2 if AVX512 not available
    subtract_scaled_d_avx2(a, b, s, n);
#endif
}

// Norm double precision implementations
double norm_d_baseline(const double* v, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

double norm_d_sse42(const double* v, size_t n) {
    double sum = 0.0;
    size_t i = 0;

    for (; i + 1 < n; i += 2) {
        __m128d vec = _mm_loadu_pd(v + i);
        __m128d sq = _mm_mul_pd(vec, vec);
        __m128d shuf = _mm_shuffle_pd(sq, sq, 0x1);
        sq = _mm_add_pd(sq, shuf);
        sum += _mm_cvtsd_f64(sq);
    }

    // Handle remaining element
    if (i < n) {
        sum += v[i] * v[i];
    }

    return std::sqrt(sum);
}

double norm_d_avx2(const double* v, size_t n) {
    if (!cpu_features::check_os_avx_support()) {
        throw cpu_features::CpuFeatureException("AVX not safely supported by OS");
    }

    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d vec = _mm256_loadu_pd(v + i);
        sum_vec = _mm256_fmadd_pd(vec, vec, sum_vec);
    }

    // Horizontal sum
    sum_vec = _mm256_add_pd(sum_vec, _mm256_permute2f128_pd(sum_vec, sum_vec, 1));
    sum_vec = _mm256_hadd_pd(sum_vec, sum_vec);
    double sum = _mm256_cvtsd_f64(sum_vec);

    // Handle remainder
    for (; i < n; ++i) {
        sum += v[i] * v[i];
    }

    return std::sqrt(sum);
}

#ifdef __AVX512F__
double norm_d_avx512(const double* v, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    __m512d sum_vec = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m512d vec = _mm512_loadu_pd(v + i);
        sum_vec = _mm512_fmadd_pd(vec, vec, sum_vec);
    }
    double sum = _mm512_reduce_add_pd(sum_vec);

    // Handle remainder
    for (; i < n; ++i) {
        sum += v[i] * v[i];
    }

    return std::sqrt(sum);
}
#endif

double norm_d_amx(const double* v, size_t n) {
    if (!cpu_features::check_os_avx512_support()) {
        throw cpu_features::CpuFeatureException("AVX-512 not safely supported by OS");
    }

    // Fallback to AVX-512
#ifdef __AVX512F__
    return norm_d_avx512(v, n);
#else
    // Fallback to AVX2 if AVX512 not available
    return norm_d_avx2(v, n);
#endif
}

} // namespace impl
} // namespace kernels
} // namespace hypercube