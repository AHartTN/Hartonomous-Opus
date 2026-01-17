/**
 * Linear Algebra Wrappers Implementation
 *
 * Backend priority: MKL -> Eigen -> Scalar
 */

#include "hypercube/linear_algebra.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

// MKL backend (highest priority)
#if defined(HAS_MKL) && HAS_MKL
#include <mkl.h>
#include <mkl_lapacke.h>
#define LA_USE_MKL 1
#else
#define LA_USE_MKL 0
#endif

// Eigen backend (fallback)
#if defined(HAS_EIGEN) && HAS_EIGEN && !LA_USE_MKL
#include <Eigen/Dense>
#define LA_USE_EIGEN 1
#else
#define LA_USE_EIGEN 0
#endif

// Scalar backend (always available)
#define LA_USE_SCALAR 1

namespace hypercube {
namespace la {

// =============================================================================
// BLAS Level 1 Implementations
// =============================================================================

double dot(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = std::min(x.size(), y.size());
    return dot(x.data(), y.data(), n);
}

double dot(const double* x, const double* y, size_t n) {
#if LA_USE_MKL
    return cblas_ddot(static_cast<MKL_INT>(n), x, 1, y, 1);
#elif LA_USE_EIGEN
    // Eigen doesn't have a direct dot product wrapper, use Eigen::Map
    Eigen::Map<const Eigen::VectorXd> vx(x, n);
    Eigen::Map<const Eigen::VectorXd> vy(y, n);
    return vx.dot(vy);
#else // Scalar fallback
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
#endif
}

void axpy(double a, const std::vector<double>& x, std::vector<double>& y) {
    size_t n = std::min(x.size(), y.size());
    axpy(a, x.data(), y.data(), n);
}

void axpy(double a, const double* x, double* y, size_t n) {
#if LA_USE_MKL
    cblas_daxpy(static_cast<MKL_INT>(n), a, x, 1, y, 1);
#elif LA_USE_EIGEN
    Eigen::Map<const Eigen::VectorXd> vx(x, n);
    Eigen::Map<Eigen::VectorXd> vy(y, n);
    vy += a * vx;
#else // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
#endif
}

void scal(double a, std::vector<double>& x) {
    scal(a, x.data(), x.size());
}

void scal(double a, double* x, size_t n) {
#if LA_USE_MKL
    cblas_dscal(static_cast<MKL_INT>(n), a, x, 1);
#elif LA_USE_EIGEN
    Eigen::Map<Eigen::VectorXd> vx(x, n);
    vx *= a;
#else // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        x[i] *= a;
    }
#endif
}

void copy(const std::vector<double>& x, std::vector<double>& y) {
    size_t n = std::min(x.size(), y.size());
    copy(x.data(), y.data(), n);
}

void copy(const double* x, double* y, size_t n) {
#if LA_USE_MKL
    cblas_dcopy(static_cast<MKL_INT>(n), x, 1, y, 1);
#elif LA_USE_EIGEN
    Eigen::Map<const Eigen::VectorXd> vx(x, n);
    Eigen::Map<Eigen::VectorXd> vy(y, n);
    vy = vx;
#else // Scalar fallback
    std::memcpy(y, x, n * sizeof(double));
#endif
}

double nrm2(const std::vector<double>& x) {
    return nrm2(x.data(), x.size());
}

double nrm2(const double* x, size_t n) {
#if LA_USE_MKL
    return cblas_dnrm2(static_cast<MKL_INT>(n), x, 1);
#elif LA_USE_EIGEN
    Eigen::Map<const Eigen::VectorXd> vx(x, n);
    return vx.norm();
#else // Scalar fallback
    return std::sqrt(dot(x, x, n));
#endif
}

// =============================================================================
// BLAS Level 2 Implementations
// =============================================================================

void gemv(double alpha, const std::vector<double>& A, const std::vector<double>& x,
          double beta, std::vector<double>& y, bool transpose) {
    size_t m = A.size() / x.size();  // Assume square for simplicity
    size_t n = x.size();
    gemv(alpha, A.data(), m, n, x.data(), beta, y.data(), transpose);
}

void gemv(double alpha, const double* A, size_t m, size_t n, const double* x,
          double beta, double* y, bool transpose) {
#if LA_USE_MKL
    CBLAS_TRANSPOSE trans = transpose ? CblasTrans : CblasNoTrans;
    cblas_dgemv(CblasRowMajor, trans, static_cast<MKL_INT>(m), static_cast<MKL_INT>(n),
                alpha, A, static_cast<MKL_INT>(n), x, 1, beta, y, 1);
#elif LA_USE_EIGEN
    Eigen::Map<const Eigen::MatrixXd> mA(A, m, n);
    Eigen::Map<const Eigen::VectorXd> vx(x, n);
    Eigen::Map<Eigen::VectorXd> vy(y, transpose ? n : m);

    if (transpose) {
        vy = alpha * mA.transpose() * vx + beta * vy;
    } else {
        vy = alpha * mA * vx + beta * vy;
    }
#else // Scalar fallback
    // Simple implementation - could be optimized with SIMD
    if (transpose) {
        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < m; ++j) {
                sum += A[j * n + i] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                sum += A[i * n + j] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    }
#endif
}

// =============================================================================
// BLAS Level 3 Implementations
// =============================================================================

void gemm(double alpha, const std::vector<double>& A, const std::vector<double>& B,
          double beta, std::vector<double>& C,
          bool transpose_A, bool transpose_B) {
    // For simplicity, assume square matrices
    size_t n = static_cast<size_t>(std::sqrt(A.size()));
    gemm(alpha, A.data(), n, n, B.data(), n, beta, C.data(), transpose_A, transpose_B);
}

void gemm(double alpha, const double* A, size_t m, size_t k, const double* B, size_t n,
          double beta, double* C, bool transpose_A, bool transpose_B) {
#if LA_USE_MKL
    CBLAS_TRANSPOSE transA = transpose_A ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = transpose_B ? CblasTrans : CblasNoTrans;
    cblas_dgemm(CblasRowMajor, transA, transB,
                static_cast<MKL_INT>(m), static_cast<MKL_INT>(n), static_cast<MKL_INT>(k),
                alpha, A, static_cast<MKL_INT>(transpose_A ? m : k),
                B, static_cast<MKL_INT>(transpose_B ? k : n),
                beta, C, static_cast<MKL_INT>(n));
#elif LA_USE_EIGEN
    size_t rows_A = transpose_A ? k : m;
    size_t cols_A = transpose_A ? m : k;
    size_t rows_B = transpose_B ? n : k;
    size_t cols_B = transpose_B ? k : n;

    Eigen::Map<const Eigen::MatrixXd> mA(A, rows_A, cols_A);
    Eigen::Map<const Eigen::MatrixXd> mB(B, rows_B, cols_B);
    Eigen::Map<Eigen::MatrixXd> mC(C, m, n);

    Eigen::MatrixXd temp;
    if (transpose_A && transpose_B) {
        temp = mA.transpose() * mB.transpose();
    } else if (transpose_A) {
        temp = mA.transpose() * mB;
    } else if (transpose_B) {
        temp = mA * mB.transpose();
    } else {
        temp = mA * mB;
    }

    mC = alpha * temp + beta * mC;
#else // Scalar fallback
    // Very basic implementation - not optimized
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t l = 0; l < k; ++l) {
                double a_val = transpose_A ? A[l * m + i] : A[i * k + l];
                double b_val = transpose_B ? B[j * k + l] : B[l * n + j];
                sum += a_val * b_val;
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
#endif
}

// =============================================================================
// LAPACK Implementations
// =============================================================================

SyevResult syev(const std::vector<double>& A, size_t n) {
    return syev(A.data(), n);
}

SyevResult syev(const double* A, size_t n) {
    SyevResult result;
    result.eigenvalues.resize(n);
    result.eigenvectors.resize(n * n);

#if LA_USE_MKL
    // Copy input matrix since LAPACK overwrites it
    std::vector<double> A_copy(A, A + n * n);
    std::copy(A, A + n * n, A_copy.begin());

    char jobz = 'V';  // Compute eigenvalues and eigenvectors
    char uplo = 'U';  // Upper triangle
    MKL_INT mkl_n = static_cast<MKL_INT>(n);
    MKL_INT lda = mkl_n;
    MKL_INT info = 0;

    // Query workspace size
    MKL_INT lwork = -1;
    double work_query;
    dsyev(&jobz, &uplo, &mkl_n, A_copy.data(), &lda,
          result.eigenvalues.data(), &work_query, &lwork, &info);

    lwork = static_cast<MKL_INT>(work_query);
    std::vector<double> work(lwork);

    // Compute eigenvalues and eigenvectors
    dsyev(&jobz, &uplo, &mkl_n, A_copy.data(), &lda,
          result.eigenvalues.data(), work.data(), &lwork, &info);

    if (info == 0) {
        result.success = true;
        std::copy(A_copy.begin(), A_copy.end(), result.eigenvectors.begin());
    }

#elif LA_USE_EIGEN
    Eigen::Map<const Eigen::MatrixXd> mA(A, n, n);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mA);

    if (solver.info() == Eigen::Success) {
        result.success = true;
        // Copy eigenvalues
        for (size_t i = 0; i < n; ++i) {
            result.eigenvalues[i] = solver.eigenvalues()(i);
        }
        // Copy eigenvectors (Eigen stores in column-major)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                result.eigenvectors[i * n + j] = solver.eigenvectors()(j, i);
            }
        }
    }

#else // Scalar fallback - Jacobi algorithm
    // Copy input matrix
    result.eigenvectors.assign(A, A + n * n);

    // Initialize eigenvalues (diagonal)
    for (size_t i = 0; i < n; ++i) {
        result.eigenvalues[i] = result.eigenvectors[i * n + i];
    }

    // Jacobi iterations for diagonalization
    const int max_sweeps = 50;
    const double eps = 1e-12;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_off = 0.0;

        for (size_t p = 0; p < n - 1; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                double apq = result.eigenvectors[p * n + q];
                if (std::abs(apq) < eps) continue;
                if (std::abs(apq) > max_off) max_off = std::abs(apq);

                double app = result.eigenvectors[p * n + p];
                double aqq = result.eigenvectors[q * n + q];
                double theta = 0.5 * (aqq - app) / apq;

                double t = 1.0 / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                if (theta < 0) t = -t;

                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                // Update matrix
                result.eigenvectors[p * n + p] = app - t * apq;
                result.eigenvectors[q * n + q] = aqq + t * apq;
                result.eigenvectors[p * n + q] = 0.0;
                result.eigenvectors[q * n + p] = 0.0;

                for (size_t r = 0; r < n; ++r) {
                    if (r != p && r != q) {
                        double arp = result.eigenvectors[r * n + p];
                        double arq = result.eigenvectors[r * n + q];
                        result.eigenvectors[r * n + p] = result.eigenvectors[p * n + r] = c * arp - s * arq;
                        result.eigenvectors[r * n + q] = result.eigenvectors[q * n + r] = s * arp + c * arq;
                    }
                }

                // Update eigenvalues
                result.eigenvalues[p] = result.eigenvectors[p * n + p];
                result.eigenvalues[q] = result.eigenvectors[q * n + q];
            }
        }

        if (max_off < eps) {
            result.success = true;
            break;
        }
    }
#endif

    return result;
}

SyevrResult syevr(const std::vector<double>& A, size_t n, int il, int iu) {
    return syevr(A.data(), n, il, iu);
}

SyevrResult syevr(const double* A, size_t n, int il, int iu) {
    SyevrResult result;

#if LA_USE_MKL
    // Copy input matrix
    std::vector<double> A_copy(A, A + n * n);

    char jobz = 'V';  // Compute eigenvalues and eigenvectors
    char range = 'I'; // Index range
    char uplo = 'U';  // Upper triangle
    MKL_INT mkl_n = static_cast<MKL_INT>(n);
    MKL_INT lda = mkl_n;
    double vl = 0.0, vu = 0.0;  // Not used for range='I'
    MKL_INT mkl_il = il;
    MKL_INT mkl_iu = iu;
    double abstol = 0.0;  // Use default
    MKL_INT m_found = 0;

    // Size eigenvectors array for worst case
    result.eigenvalues.resize(iu - il + 1);
    result.eigenvectors.resize(n * (iu - il + 1));

    MKL_INT ldz = mkl_n;
    std::vector<MKL_INT> isuppz(2 * (iu - il + 1));

    // Compute using LAPACKE (workspace handled internally)
    int info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, jobz, range, uplo, mkl_n, A_copy.data(), lda,
                              vl, vu, mkl_il, mkl_iu, abstol, &m_found,
                              result.eigenvalues.data(), result.eigenvectors.data(), ldz, isuppz.data());

    if (info == 0) {
        result.success = true;
        result.found_count = static_cast<int>(m_found);
    }

#else
    // For non-MKL backends, fall back to syev and extract the requested range
    auto full_result = syev(A, n);
    if (full_result.success) {
        result.success = true;
        // Copy requested eigenvalues (adjusting for 0-based indexing)
        int start_idx = il - 1;  // Convert to 0-based
        int end_idx = iu - 1;    // Convert to 0-based
        int count = end_idx - start_idx + 1;

        if (start_idx >= 0 && end_idx < static_cast<int>(n) && count > 0) {
            result.eigenvalues.resize(count);
            result.eigenvectors.resize(n * count);
            result.found_count = count;

            for (int i = 0; i < count; ++i) {
                result.eigenvalues[i] = full_result.eigenvalues[start_idx + i];
                // Copy eigenvector column
                for (size_t j = 0; j < n; ++j) {
                    result.eigenvectors[i * n + j] = full_result.eigenvectors[(start_idx + i) * n + j];
                }
            }
        }
    }
#endif

    return result;
}

} // namespace la
} // namespace hypercube