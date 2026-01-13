#pragma once

#include "hypercube/types.hpp"
#include <cstdint>

namespace hypercube {

/**
 * Coordinate utility functions for quantization and optimization
 */
class CoordinateUtilities {
public:
    /**
     * Safe quantization with rounding and clamping
     * Maps [-1,1] to [0, UINT32_MAX]
     */
    static uint32_t quantize_unit_to_u32(double v) noexcept;

#if HAS_AVX
    /**
     * AVX-optimized quantization for 4 components
     */
    static void avx_quantize_point4f_to_point4d(const Point4F& src, Point4D& dst) noexcept;
#endif

    /**
     * Safe inverse power to prevent gradient blowups
     */
    static double safe_pow_inv(double r, double p, double eps = 1e-8);

#if HAS_AVX
    /**
     * AVX-optimized Euclidean distance for 4D points
     */
    static double avx_distance(const Point4F& a, const Point4F& b) noexcept;
#endif

    /**
     * Select distance function based on available optimizations
     */
    static double optimized_distance(const Point4F& a, const Point4F& b) noexcept;

    /**
     * Select dot product function
     */
    static double optimized_dot(const Point4F& a, const Point4F& b) noexcept;
};

} // namespace hypercube