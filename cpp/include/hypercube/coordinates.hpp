#pragma once

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include <vector>

namespace hypercube {

/**
 * Combined result of codepoint mapping - coords AND hilbert index
 * Avoids redundant Hilbert encode/decode roundtrip
 */
struct CodepointMapping {
    Point4D coords;
    HilbertIndex hilbert;
};

/**
 * 4D Fibonacci Lattice Coordinate Mapper for Unicode Atoms
 * 
 * Maps all Unicode codepoints onto the surface of a 3-sphere (hypersphere in 4D)
 * using the Fibonacci lattice algorithm extended to 4D.
 * 
 * Key properties:
 * - ALL atoms are evenly distributed on the 3-sphere surface (Dyson sphere)
 * - Semantically related codepoints (A, a, Ä) are placed nearby via sort order
 * - 32 bits per dimension = lossless, collision-free coordinates
 * - Hilbert index is derived from coords for spatial indexing
 * - Compositions (n-grams, words, etc.) have centroids INSIDE the sphere
 */
class CoordinateMapper {
public:
    /**
     * Map a Unicode codepoint to its 4D coordinates on the 3-sphere surface
     * Uses Fibonacci lattice for even distribution with semantic clustering
     * @param codepoint Unicode codepoint (0 to 0x10FFFF)
     * @return 4D point on 3-sphere surface (32 bits per dimension)
     */
    static Point4D map_codepoint(uint32_t codepoint) noexcept;
    
    /**
     * Map a Unicode codepoint to BOTH coords AND hilbert index
     * This is the efficient version - computes hilbert once during mapping
     * @param codepoint Unicode codepoint (0 to 0x10FFFF)
     * @return Combined coords + hilbert index
     */
    static CodepointMapping map_codepoint_full(uint32_t codepoint) noexcept;
    
    /**
     * Determine category of a Unicode codepoint
     * Uses Unicode general categories for classification
     * @param codepoint Unicode codepoint
     * @return Category enum value
     */
    static AtomCategory categorize(uint32_t codepoint) noexcept;
    
    /**
     * Calculate centroid of multiple points
     * Used for composition coordinates - will be INSIDE the sphere
     * @param points Vector of 4D points
     * @return Centroid point (interior for surface points)
     */
    static Point4D centroid(const std::vector<Point4D>& points) noexcept;
    
    /**
     * Calculate weighted centroid
     * @param points Vector of 4D points
     * @param weights Weights for each point (must be same size as points)
     * @return Weighted centroid
     */
    static Point4D weighted_centroid(const std::vector<Point4D>& points,
                                      const std::vector<double>& weights) noexcept;
    
    /**
     * Check if a point is on the 3-sphere surface
     * @param point 4D point to check
     * @return true if on surface (r² ≈ 1 within tolerance)
     */
    static bool is_on_surface(const Point4D& point) noexcept;
    
    /**
     * Calculate Euclidean distance in 4D space
     */
    static double euclidean_distance(const Point4D& a, const Point4D& b) noexcept;
    
    /**
     * Get the count of codepoints in a category
     */
    static uint32_t get_category_count(AtomCategory cat) noexcept;
};

} // namespace hypercube
