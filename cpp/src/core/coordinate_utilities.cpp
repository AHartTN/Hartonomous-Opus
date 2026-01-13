#include "hypercube/coordinate_utilities.hpp"
#include "hypercube/simd_intrinsics.hpp"
#include <cmath>
#include <limits>

#ifdef HAS_AVX
#include <immintrin.h>
#endif

namespace hypercube {

// Safe quantization with rounding and clamping
uint32_t CoordinateUtilities::quantize_unit_to_u32(double v) noexcept
{
    // Expect v in [-1.0, 1.0]. Clamp defensively.
    if (v <= -1.0)
        return 0u;
    if (v >= 1.0)
        return UINT32_MAX;

    // Map [-1,1] -> [0, UINT32_MAX] using 64-bit intermediate to avoid precision loss.
    // Use floor(x + 0.5) for rounding to nearest integer deterministically.
    const long double scaled = (static_cast<long double>(v) + 1.0L) * 0.5L * static_cast<long double>(UINT32_MAX);
    uint64_t rounded = static_cast<uint64_t>(std::floor(scaled + 0.5L));
    if (rounded > UINT32_MAX)
        rounded = UINT32_MAX;
    return static_cast<uint32_t>(rounded);
}

#if HAS_AVX
// AVX-optimized quantization for 4 components
void CoordinateUtilities::avx_quantize_point4f_to_point4d(const Point4F& src, Point4D& dst) noexcept
{
    // Process components individually since we need uint32 conversion
    // This is still faster than scalar due to vectorized operations
    __m256d vec = _mm256_set_pd(src.m, src.z, src.y, src.x);

    // Clamp to [-1, 1]
    __m256d min_val = _mm256_set1_pd(-1.0);
    __m256d max_val = _mm256_set1_pd(1.0);
    vec = _mm256_max_pd(vec, min_val);
    vec = _mm256_min_pd(vec, max_val);

    // Map [-1,1] -> [0, UINT32_MAX]
    __m256d add_one = _mm256_add_pd(vec, _mm256_set1_pd(1.0));
    __m256d mul_half = _mm256_mul_pd(add_one, _mm256_set1_pd(0.5));
    __m256d scale = _mm256_mul_pd(mul_half, _mm256_set1_pd(static_cast<double>(UINT32_MAX)));

    // Add 0.5 for rounding
    __m256d half = _mm256_add_pd(scale, _mm256_set1_pd(0.5));

    // Convert to integers using proper AVX instructions
    __m128d low_half = _mm256_castpd256_pd128(half);
    __m128d high_half = _mm256_extractf128_pd(half, 1);

    // Use _mm_cvttpd_epi32 for each 128-bit lane
    __m128i int_low = _mm_cvttpd_epi32(low_half);   // x, y as int32
    __m128i int_high = _mm_cvttpd_epi32(high_half); // z, m as int32

    // Extract individual 32-bit values
    dst.x = _mm_cvtsi128_si32(int_low);
    dst.y = _mm_cvtsi128_si32(_mm_srli_si128(int_low, 4));
    dst.z = _mm_cvtsi128_si32(int_high);
    dst.m = _mm_cvtsi128_si32(_mm_srli_si128(int_high, 4));
}
#endif

// Safe inverse power to prevent gradient blowups
double CoordinateUtilities::safe_pow_inv(double r, double p, double eps)
{
    double rclamped = std::max(r, eps);
    return 1.0 / std::pow(rclamped, p);
}

#if HAS_AVX
// AVX-optimized Euclidean distance for 4D points
double CoordinateUtilities::avx_distance(const Point4F& a, const Point4F& b) noexcept
{
    // Load points into AVX registers
    __m256d a_vec = _mm256_set_pd(a.m, a.z, a.y, a.x);
    __m256d b_vec = _mm256_set_pd(b.m, b.z, b.y, b.x);

    // Compute difference
    __m256d diff = _mm256_sub_pd(a_vec, b_vec);

    // Compute squared difference
    __m256d sq_diff = _mm256_mul_pd(diff, diff);

    // Sum all components: sq_diff[0] + sq_diff[1] + sq_diff[2] + sq_diff[3]
    __m128d sum_high = _mm256_extractf128_pd(sq_diff, 1); // sq_diff[2], sq_diff[3]
    __m128d sum_low = _mm256_castpd256_pd128(sq_diff);    // sq_diff[0], sq_diff[1]

    __m128d sum = _mm_add_pd(sum_low, sum_high); // [sum0+sum2, sum1+sum3]
    sum = _mm_hadd_pd(sum, sum);                 // [sum0+sum1+sum2+sum3, duplicate]

    // Extract and sqrt
    double sum_sq = _mm_cvtsd_f64(sum);
    return std::sqrt(sum_sq);
}
#endif

// Select distance function based on available optimizations
double CoordinateUtilities::optimized_distance(const Point4F& a, const Point4F& b) noexcept
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    double dm = a.m - b.m;
    return std::sqrt(dx * dx + dy * dy + dz * dz + dm * dm);
}

// Select dot product function
double CoordinateUtilities::optimized_dot(const Point4F& a, const Point4F& b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.m * b.m;
}

} // namespace hypercube