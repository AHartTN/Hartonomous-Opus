#pragma once

#include "hypercube/types.hpp"

namespace hypercube {

/**
 * 4D Hilbert Curve implementation
 *
 * Maps 4D coordinates (each 32-bit) to a 128-bit Hilbert index and vice versa.
 * The Hilbert curve preserves locality: nearby points in 4D space map to
 * nearby indices on the curve.
 *
 * Implementation based on:
 * - Butz, A.R. "Alternative Algorithm for Hilbert's Space-Filling Curve" (1971)
 * - Hamilton, C.H. "Compact Hilbert Indices" (2006)
 * - Skilling, J. "Programming the Hilbert curve" (2004)
 */
class HilbertCurve {
public:
    static constexpr uint32_t DIMS = 4;
    static constexpr uint32_t BITS = 32;

    /**
     * Convert 4D coordinates to Hilbert index
     * @param point 4D point with 64-bit coordinates per dimension
     * @return 256-bit Hilbert index as four 64-bit integers
     */
    static HilbertIndex coords_to_index(const Point4D& point) noexcept;

    /**
     * Convert Hilbert index to 4D coordinates
     * @param index 256-bit Hilbert index
     * @return 4D point with 64-bit coordinates per dimension
     */
    static Point4D index_to_coords(const HilbertIndex& index) noexcept;
    
    /**
     * Compute absolute distance between two Hilbert indices
     */
    static HilbertIndex distance(const HilbertIndex& a, const HilbertIndex& b) noexcept;
    
    /**
     * Check if index b is within range of index a
     */
    static bool in_range(const HilbertIndex& center, const HilbertIndex& point, 
                         const HilbertIndex& range) noexcept;

    /**
     * Convert Hilbert index to raw 4D coordinates (corner-origin, no CENTER adjustment)
     * @param index 256-bit Hilbert index
     * @return 4D point in [0, UINT64_MAX]^4 corner-origin space
     *
     * This is used for semantic coordinate mapping where we need the raw
     * Hilbert decode without the CENTER-origin transformation.
     */
    static Point4D index_to_raw_coords(const HilbertIndex& index) noexcept;

    /**
     * Batch coordinate to index conversion using AVX512 SIMD (16 coords at once)
     * @param points Input array of 4D points
     * @param count Number of points to process
     * @param indices Output array for Hilbert indices
     */
    static void coords_to_indices_batch_avx512(const Point4D* points,
                                               size_t count,
                                               HilbertIndex* indices) noexcept;

    /**
     * Batch index to coordinate conversion using AVX512 SIMD (16 indices at once)
     * @param indices Input array of Hilbert indices
     * @param count Number of indices to process
     * @param points Output array for 4D points
     */
    static void indices_to_coords_batch_avx512(const HilbertIndex* indices,
                                               size_t count,
                                               Point4D* points) noexcept;

    /**
     * Batch coordinate to index conversion using AVX2 SIMD (8 coords at once)
     * @param points Input array of 4D points
     * @param count Number of points to process
     * @param indices Output array for Hilbert indices
     */
    static void coords_to_indices_batch_avx2(const Point4D* points,
                                             size_t count,
                                             HilbertIndex* indices) noexcept;

    // Transpose operations for Hilbert curve (public for specialized use cases)
    static void transpose_to_axes(uint32_t* x, uint32_t n, uint32_t bits) noexcept;
    static void axes_to_transpose(uint32_t* x, uint32_t n, uint32_t bits) noexcept;
};

} // namespace hypercube
