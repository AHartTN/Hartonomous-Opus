#pragma once

/**
 * Linear Algebra Wrappers with Backend Priority
 *
 * Provides a unified interface for BLAS/LAPACK operations with automatic
 * backend selection: MKL -> Eigen -> Scalar fallback.
 *
 * Priority order:
 * 1. Intel MKL (fastest, most optimized)
 * 2. Eigen library (portable, good performance)
 * 3. Scalar implementations (always available, slowest)
 */

#include <vector>
#include <cstdint>
#include <memory>

namespace hypercube {
namespace la {

// =============================================================================
// BLAS Level 1 (Vector operations)
// =============================================================================

/**
 * Dot product: result = x · y
 */
double dot(const std::vector<double>& x, const std::vector<double>& y);
double dot(const double* x, const double* y, size_t n);

/**
 * AXPY: y = a*x + y
 */
void axpy(double a, const std::vector<double>& x, std::vector<double>& y);
void axpy(double a, const double* x, double* y, size_t n);

/**
 * SCAL: x = a*x (scale vector)
 */
void scal(double a, std::vector<double>& x);
void scal(double a, double* x, size_t n);

/**
 * COPY: y = x
 */
void copy(const std::vector<double>& x, std::vector<double>& y);
void copy(const double* x, double* y, size_t n);

/**
 * NRM2: Euclidean norm ||x||_2
 */
double nrm2(const std::vector<double>& x);
double nrm2(const double* x, size_t n);

// =============================================================================
// BLAS Level 2 (Matrix-vector operations)
// =============================================================================

/**
 * GEMV: y = alpha*A*x + beta*y (general matrix-vector multiply)
 * A is m x n stored in row-major order
 */
void gemv(double alpha, const std::vector<double>& A, const std::vector<double>& x,
          double beta, std::vector<double>& y, bool transpose = false);
void gemv(double alpha, const double* A, size_t m, size_t n, const double* x,
          double beta, double* y, bool transpose = false);

// =============================================================================
// BLAS Level 3 (Matrix-matrix operations)
// =============================================================================

/**
 * GEMM: C = alpha*A*B + beta*C (general matrix-matrix multiply)
 */
void gemm(double alpha, const std::vector<double>& A, const std::vector<double>& B,
          double beta, std::vector<double>& C,
          bool transpose_A = false, bool transpose_B = false);
void gemm(double alpha, const double* A, size_t m, size_t k, const double* B, size_t n,
          double beta, double* C, bool transpose_A = false, bool transpose_B = false);

// =============================================================================
// LAPACK (Linear algebra)
// =============================================================================

/**
 * SYEV: Solve symmetric eigenvalue problem A*x = lambda*x
 * Returns eigenvalues in ascending order, eigenvectors as columns
 */
struct SyevResult {
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;  // Column-major storage
    bool success = false;
};

SyevResult syev(const std::vector<double>& A, size_t n);
SyevResult syev(const double* A, size_t n);

/**
 * SYEVR: Solve symmetric eigenvalue problem with range selection
 * Finds eigenvalues il through iu (1-based indexing)
 */
struct SyevrResult {
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;  // Column-major storage
    int found_count = 0;              // Number of eigenvalues found
    bool success = false;
};

SyevrResult syevr(const std::vector<double>& A, size_t n, int il, int iu);
SyevrResult syevr(const double* A, size_t n, int il, int iu);

} // namespace la
} // namespace hypercube