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
     * @param point 4D point with 32-bit coordinates per dimension
     * @return 128-bit Hilbert index as two 64-bit integers
     */
    static HilbertIndex coords_to_index(const Point4D& point) noexcept;
    
    /**
     * Convert Hilbert index to 4D coordinates
     * @param index 128-bit Hilbert index
     * @return 4D point with 32-bit coordinates per dimension
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

private:
    // Transpose operations for Hilbert curve
    static void transpose_to_axes(uint32_t* x, uint32_t n, uint32_t bits) noexcept;
    static void axes_to_transpose(uint32_t* x, uint32_t n, uint32_t bits) noexcept;
};

} // namespace hypercube
