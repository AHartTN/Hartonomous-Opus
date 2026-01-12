#include "hypercube/superfibonacci.hpp"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace hypercube {

// Mathematical constants from Super-Fibonacci paper
// φ² = 2, ψ⁴ = ψ + 4
constexpr double SuperFibonacci::PHI;
constexpr double SuperFibonacci::PSI;

std::vector<Point4F> SuperFibonacci::generate_points(size_t n) {
    std::vector<Point4F> points;
    points.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        points.push_back(compute_point(i, n));
    }

    return points;
}

Point4F SuperFibonacci::get_point(size_t index, size_t total_n) {
    return compute_point(index, total_n);
}

Point4F SuperFibonacci::compute_point(size_t i, size_t n) {
    // Algorithm 1 from Super-Fibonacci paper
    // s ← i + 1/2
    double s = static_cast<double>(i) + 0.5;

    // t ← s/n
    double t = s / static_cast<double>(n);

    // d ← 2π * s
    double d = 2.0 * M_PI * s;

    // r ← √t, R ← √(1-t)
    double r = std::sqrt(t);
    double R = std::sqrt(1.0 - t);

    // α ← d/φ, β ← d/ψ
    double alpha = d / PHI;
    double beta = d / PSI;

    // q_i ← (r*sin(α), r*cos(α), R*sin(β), R*cos(β))
    double w = r * std::sin(alpha);
    double x = r * std::cos(alpha);
    double y = R * std::sin(beta);
    double z = R * std::cos(beta);

    return Point4F(w, x, y, z);
}

} // namespace hypercube