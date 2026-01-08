#pragma once

/**
 * Normalization Helpers
 *
 * Provides coordinate normalization for hypercube and hypersphere projections.
 * Maps eigenvector coordinates to uint32 hypercube space and optionally projects to sphere surface.
 */

#include <vector>
#include <array>
#include <cstdint>

namespace hypercube {
namespace helpers {

/**
 * Normalize eigenvectors to [0, 2^32-1]^4 hypercube coordinates
 *
 * Maps k eigenvectors (k=4) of length n to n 4D points in uint32 space.
 * Each dimension is independently normalized to span full uint32 range.
 *
 * @param U Column eigenvectors (k vectors of length n)
 * @param num_threads Number of threads for parallelization (0 = auto)
 * @param verbose Enable diagnostic output
 * @return Vector of n 4D coordinates
 */
std::vector<std::array<uint32_t, 4>> normalize_to_hypercube(
    const std::vector<std::vector<double>>& U,
    int num_threads = 0,
    bool verbose = false
);

/**
 * Project hypercube coordinates onto 4D hypersphere surface
 *
 * Converts coordinates from hypercube [0, 2^32-1]^4 to unit hypersphere
 * with center at (2^31, 2^31, 2^31, 2^31).
 *
 * COORDINATE CONVENTION:
 * - CENTER = 2^31 = 2147483648 (origin of hypercube)
 * - Unit sphere [-1, 1] maps to [0, 2^32-1]
 * - Radius = sphere_radius (default 1.0)
 *
 * @param coords Hypercube coordinates (modified in-place)
 * @param sphere_radius Target radius (default 1.0)
 * @param num_threads Number of threads for parallelization (0 = auto)
 * @param verbose Enable diagnostic output
 */
void project_to_sphere(
    std::vector<std::array<uint32_t, 4>>& coords,
    double sphere_radius = 1.0,
    int num_threads = 0,
    bool verbose = false
);

/**
 * Convert uint32 coordinate to centered unit coordinate [-1, 1]
 * Used for sphere projection
 */
inline double uint32_to_unit(uint32_t val) {
    constexpr double CENTER = 2147483648.0;  // 2^31
    constexpr double SCALE = 2147483647.0;   // max radius
    return (static_cast<double>(val) - CENTER) / SCALE;
}

/**
 * Convert centered unit coordinate [-1, 1] to uint32
 * Used for sphere projection
 */
inline uint32_t unit_to_uint32(double v) {
    constexpr double CENTER = 2147483648.0;  // 2^31
    constexpr double SCALE = 2147483647.0;   // max radius
    double scaled = CENTER + v * SCALE;
    if (scaled < 0.0) scaled = 0.0;
    if (scaled > 4294967295.0) scaled = 4294967295.0;
    return static_cast<uint32_t>(std::round(scaled));
}

/**
 * Compute 4D Euclidean distance between two points
 */
double distance_4d(
    const std::array<uint32_t, 4>& a,
    const std::array<uint32_t, 4>& b
);

/**
 * Check if point is on sphere surface (within tolerance)
 */
bool is_on_sphere(
    const std::array<uint32_t, 4>& coord,
    double sphere_radius = 1.0,
    double tolerance = 0.01
);

} // namespace helpers
} // namespace hypercube
