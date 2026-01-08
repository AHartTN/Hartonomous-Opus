#pragma once

/**
 * Orthogonalization Helpers
 *
 * Provides Gram-Schmidt orthonormalization and related vector operations.
 * Used by Laplacian eigenmap projection to ensure orthogonal eigenvector basis.
 */

#include <vector>
#include <cstdint>

namespace hypercube {
namespace helpers {

/**
 * Robust Double-Pass Gram-Schmidt Orthonormalization
 *
 * Applies Modified Gram-Schmidt twice ("Twice is enough" - Kahan/Parlett)
 * to correct round-off errors and ensure orthogonal basis.
 *
 * @param Y Column vectors to orthonormalize (modified in-place)
 * @param verbose Enable diagnostic output
 *
 * Handles degenerate cases:
 * - Collapsed dimensions (norm < 1e-10) are regenerated as random vectors
 * - Re-orthogonalization after regeneration
 * - Final verification of orthogonality
 */
void gram_schmidt_orthonormalize(
    std::vector<std::vector<double>>& Y,
    bool verbose = false
);

/**
 * Compute dot product between two vectors
 * Uses SIMD if available
 */
double dot_product(const double* a, const double* b, size_t n);

/**
 * Compute L2 norm of vector
 * Uses SIMD if available
 */
double norm(const double* v, size_t n);

/**
 * Normalize vector to unit length in-place
 */
void normalize_vector(double* v, size_t n);

/**
 * Subtract scaled vector: a = a - s*b
 * Uses SIMD if available
 */
void subtract_scaled(double* a, const double* b, double s, size_t n);

/**
 * Scale vector in-place: v = s*v
 * Uses SIMD if available
 */
void scale_inplace(double* v, double s, size_t n);

/**
 * Verify orthonormality of basis
 * Returns maximum off-diagonal dot product (should be << 1)
 */
double verify_orthonormality(
    const std::vector<std::vector<double>>& Y,
    bool verbose = false
);

} // namespace helpers
} // namespace hypercube
