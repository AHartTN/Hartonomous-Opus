/**
 * Lanczos Eigensolver with Full Reorthogonalization
 * 
 * Finds the k smallest non-zero eigenpairs of a sparse symmetric matrix.
 * This is the math-correct, from-scratch implementation - no external libraries.
 * 
 * Algorithm:
 * 1. Lanczos iteration builds orthonormal Krylov basis Q_m
 * 2. Projects L onto tridiagonal T_m = Q_m^T L Q_m
 * 3. QR algorithm finds eigenpairs of T_m (Ritz values/vectors)
 * 4. Ritz vectors are mapped back: v_j = Q_m y_j
 * 5. Optional shift-invert with CG for faster convergence
 * 
 * Key differences from power iteration:
 * - Power iteration finds LARGEST eigenvalues
 * - Lanczos finds EXTREMAL eigenvalues (both ends)
 * - With shift-invert, we target the SMALLEST
 */

#pragma once

#include "hypercube/laplacian_4d.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>

namespace hypercube {
namespace lanczos {

/**
 * Configuration for Lanczos solver
 */
struct LanczosConfig {
    int max_iterations = 300;       // Max Lanczos iterations (m)
    int num_eigenpairs = 4;         // k smallest non-zero eigenpairs wanted
    double convergence_tol = 1e-10; // Ritz residual tolerance
    double reorth_tol = 1e-8;       // Reorthogonalization threshold
    bool use_shift_invert = true;   // Use shift-invert for smallest eigenvalues
    double shift_sigma = 1e-6;      // Shift for (L - σI)^-1
    int cg_max_iterations = 200;    // CG iterations for shift-invert
    double cg_tolerance = 1e-12;    // CG convergence tolerance
    int num_threads = 0;            // 0 = auto
};

/**
 * Result of Lanczos computation
 */
struct LanczosResult {
    std::vector<double> eigenvalues;            // The k eigenvalues (ascending)
    std::vector<std::vector<double>> eigenvectors;  // The k eigenvectors (each length n)
    std::vector<double> residuals;              // Residual norms for convergence check
    int iterations_used;                        // Actual Lanczos iterations
    bool converged;                             // All residuals below tolerance
};

/**
 * Tridiagonal matrix storage
 */
struct TridiagonalMatrix {
    std::vector<double> alpha;  // Diagonal elements
    std::vector<double> beta;   // Off-diagonal elements (beta[i] = T[i,i+1] = T[i+1,i])
    
    size_t size() const { return alpha.size(); }
};

/**
 * Dense eigensolver for tridiagonal matrices using QR algorithm
 * 
 * Finds ALL eigenpairs of the small tridiagonal T_m matrix.
 * Uses implicit QR with Wilkinson shifts for stability.
 */
class TridiagonalEigensolver {
public:
    struct EigenPair {
        double eigenvalue;
        std::vector<double> eigenvector;
    };
    
    /**
     * Compute all eigenpairs of tridiagonal matrix
     * Returns sorted by eigenvalue (ascending)
     */
    static std::vector<EigenPair> solve(const TridiagonalMatrix& T);
    
private:
    // Implicit QR step with Wilkinson shift
    static void implicit_qr_step(std::vector<double>& diag, 
                                  std::vector<double>& offdiag,
                                  std::vector<std::vector<double>>& Q,
                                  size_t lo, size_t hi);
    
    // Wilkinson shift
    static double wilkinson_shift(double a, double b, double c);
    
    // Givens rotation
    static void givens_rotation(double a, double b, double& c, double& s);
};

/**
 * Conjugate Gradient solver for (L - σI)x = b
 * 
 * Used in shift-invert Lanczos for targeting smallest eigenvalues.
 */
class ConjugateGradient {
public:
    /**
     * Solve (A - σI)x = b where A is the sparse Laplacian
     * 
     * @param L      The sparse Laplacian matrix
     * @param sigma  The shift value
     * @param b      Right-hand side vector
     * @param x0     Initial guess (modified in place to solution)
     * @param tol    Convergence tolerance
     * @param maxiter Maximum iterations
     * @return Number of iterations used (-1 if failed)
     */
    static int solve(const SparseSymmetricMatrix& L,
                     double sigma,
                     const std::vector<double>& b,
                     std::vector<double>& x,
                     double tol = 1e-12,
                     int maxiter = 200,
                     bool verbose = false);
};

/**
 * Main Lanczos eigensolver
 */
class LanczosSolver {
public:
    explicit LanczosSolver(const LanczosConfig& config = LanczosConfig{});
    
    /**
     * Find k smallest non-zero eigenpairs of sparse symmetric matrix L
     * 
     * @param L  The sparse Laplacian (symmetric, positive semi-definite)
     * @return LanczosResult with eigenvalues, eigenvectors, residuals
     */
    LanczosResult solve(const SparseSymmetricMatrix& L);
    
    /**
     * Set progress callback
     */
    using ProgressCallback = std::function<void(const std::string&, int, int)>;
    void set_progress_callback(ProgressCallback cb) { progress_callback_ = std::move(cb); }
    
private:
    LanczosConfig config_;
    ProgressCallback progress_callback_;
    
    // Core Lanczos iteration (builds Q_m and T_m)
    void lanczos_iteration(
        const SparseSymmetricMatrix& L,
        std::vector<std::vector<double>>& Q,  // Lanczos vectors
        TridiagonalMatrix& T,                  // Tridiagonal matrix
        int m                                  // Number of iterations
    );
    
    // Shift-invert Lanczos iteration
    void shift_invert_lanczos(
        const SparseSymmetricMatrix& L,
        std::vector<std::vector<double>>& Q,
        TridiagonalMatrix& T,
        int m
    );
    
    // Full reorthogonalization against all previous Q vectors
    void reorthogonalize(std::vector<double>& w, 
                         const std::vector<std::vector<double>>& Q,
                         size_t num_vecs);
    
    // Extract Ritz pairs from T_m and Q_m
    void extract_ritz_pairs(
        const std::vector<std::vector<double>>& Q,
        const TridiagonalMatrix& T,
        std::vector<double>& eigenvalues,
        std::vector<std::vector<double>>& eigenvectors,
        std::vector<double>& residuals
    );
    
    // Check convergence of Ritz pairs
    bool check_convergence(const std::vector<double>& residuals, int k);
    
    // Report progress
    void report_progress(const std::string& stage, int current, int total);
};

// =============================================================================
// SIMD-optimized vector operations (from laplacian_4d but exposed here)
// =============================================================================

namespace vec {
    // Dot product
    double dot(const double* a, const double* b, size_t n);
    double dot(const std::vector<double>& a, const std::vector<double>& b);
    
    // Vector norm
    double norm(const double* v, size_t n);
    double norm(const std::vector<double>& v);
    
    // Normalize in place
    void normalize(std::vector<double>& v);
    
    // AXPY: y = a*x + y
    void axpy(double a, const double* x, double* y, size_t n);
    void axpy(double a, const std::vector<double>& x, std::vector<double>& y);
    
    // Scale: v = a*v
    void scale(double a, std::vector<double>& v);
    
    // Copy
    void copy(const std::vector<double>& src, std::vector<double>& dst);
}

} // namespace lanczos
} // namespace hypercube
