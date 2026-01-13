#pragma once

#include "hypercube/types.hpp"
#include <vector>
#include <array>

namespace hypercube {

/**
 * @brief Super-Fibonacci integration for uniform S³ point distribution
 *
 * Provides interface to the external Super-Fibonacci library for generating
 * low-discrepancy quaternion samples on the 3-sphere.
 */
class SuperFibonacci {
public:
    /**
     * @brief Generate N uniform points on S³ using Super-Fibonacci spirals
     *
     * @param n Number of points to generate
     * @return Vector of quaternions (w,x,y,z) representing S³ points
     */
    static std::vector<Point4F> generate_points(size_t n);

    /**
     * @brief Get quaternion for specific index in Super-Fibonacci sequence
     *
     * @param index Index in the sequence (0-based)
     * @param total_n Total number of points in the sequence
     * @return Quaternion representing S³ point
     */
    static Point4F get_point(size_t index, size_t total_n);

private:
    // Mathematical constants from Super-Fibonacci paper
    static constexpr double PHI = 1.533751168755204288118041; // ψ
    static constexpr double PSI = 1.41421356237309504880168872420969807856967187537694807317667973799; // sqrt(2.0)

    /**
     * @brief Core Super-Fibonacci algorithm implementation
     *
     * Based on Algorithm 1 from the Super-Fibonacci paper
     */
    static Point4F compute_point(size_t i, size_t n);
};

} // namespace hypercube