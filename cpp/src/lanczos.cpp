/**
 * Lanczos Eigensolver Implementation
 * 
 * Complete from-scratch implementation of Lanczos algorithm
 * for finding smallest non-zero eigenvalues of sparse symmetric matrices.
 * 
 * Key components:
 * 1. SIMD-optimized vector operations (AVX2/AVX512)
 * 2. Lanczos iteration with full reorthogonalization
 * 3. Shift-invert with Conjugate Gradient
 * 4. Tridiagonal QR eigensolver (implicit QR with Wilkinson shifts)
 * 5. Ritz value/vector extraction and residual estimation
 */

#include "hypercube/lanczos.hpp"
#include "hypercube/thread_pool.hpp"
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iostream>
#include <iomanip>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace hypercube {
namespace lanczos {

// =============================================================================
// SIMD Vector Operations
// =============================================================================

namespace vec {

double dot(const double* a, const double* b, size_t n) {
#ifdef __AVX512F__
    __m512d sum = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        sum = _mm512_fmadd_pd(va, vb, sum);
    }
    double result = _mm512_reduce_add_pd(sum);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
#elif defined(__AVX2__)
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    size_t i = 0;
    
    // Process 8 doubles per iteration (2 x AVX2 registers)
    for (; i + 8 <= n; i += 8) {
        __m256d va0 = _mm256_loadu_pd(a + i);
        __m256d vb0 = _mm256_loadu_pd(b + i);
        __m256d va1 = _mm256_loadu_pd(a + i + 4);
        __m256d vb1 = _mm256_loadu_pd(b + i + 4);
        sum0 = _mm256_fmadd_pd(va0, vb0, sum0);
        sum1 = _mm256_fmadd_pd(va1, vb1, sum1);
    }
    
    // Process remaining 4 doubles
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        sum0 = _mm256_fmadd_pd(va, vb, sum0);
    }
    
    // Horizontal sum
    sum0 = _mm256_add_pd(sum0, sum1);
    __m128d low = _mm256_castpd256_pd128(sum0);
    __m128d high = _mm256_extractf128_pd(sum0, 1);
    low = _mm_add_pd(low, high);
    low = _mm_hadd_pd(low, low);
    double result = _mm_cvtsd_f64(low);
    
    // Scalar tail
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
#else
    double result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
#endif
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return dot(a.data(), b.data(), std::min(a.size(), b.size()));
}

double norm(const double* v, size_t n) {
    return std::sqrt(dot(v, v, n));
}

double norm(const std::vector<double>& v) {
    return norm(v.data(), v.size());
}

void normalize(std::vector<double>& v) {
    double n = norm(v);
    if (n > std::numeric_limits<double>::epsilon()) {
        scale(1.0 / n, v);
    }
}

void axpy(double a, const double* x, double* y, size_t n) {
#ifdef __AVX512F__
    __m512d va = _mm512_set1_pd(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d vx = _mm512_loadu_pd(x + i);
        __m512d vy = _mm512_loadu_pd(y + i);
        vy = _mm512_fmadd_pd(va, vx, vy);
        _mm512_storeu_pd(y + i, vy);
    }
    for (; i < n; ++i) {
        y[i] += a * x[i];
    }
#elif defined(__AVX2__)
    __m256d va = _mm256_set1_pd(a);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_loadu_pd(x + i);
        __m256d vy = _mm256_loadu_pd(y + i);
        vy = _mm256_fmadd_pd(va, vx, vy);
        _mm256_storeu_pd(y + i, vy);
    }
    for (; i < n; ++i) {
        y[i] += a * x[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
#endif
}

void axpy(double a, const std::vector<double>& x, std::vector<double>& y) {
    axpy(a, x.data(), y.data(), std::min(x.size(), y.size()));
}

void scale(double a, std::vector<double>& v) {
#ifdef __AVX512F__
    __m512d va = _mm512_set1_pd(a);
    size_t i = 0;
    for (; i + 8 <= v.size(); i += 8) {
        __m512d vx = _mm512_loadu_pd(v.data() + i);
        vx = _mm512_mul_pd(va, vx);
        _mm512_storeu_pd(v.data() + i, vx);
    }
    for (; i < v.size(); ++i) {
        v[i] *= a;
    }
#elif defined(__AVX2__)
    __m256d va = _mm256_set1_pd(a);
    size_t i = 0;
    for (; i + 4 <= v.size(); i += 4) {
        __m256d vx = _mm256_loadu_pd(v.data() + i);
        vx = _mm256_mul_pd(va, vx);
        _mm256_storeu_pd(v.data() + i, vx);
    }
    for (; i < v.size(); ++i) {
        v[i] *= a;
    }
#else
    for (auto& x : v) {
        x *= a;
    }
#endif
}

void copy(const std::vector<double>& src, std::vector<double>& dst) {
    dst.resize(src.size());
    std::memcpy(dst.data(), src.data(), src.size() * sizeof(double));
}

} // namespace vec

// =============================================================================
// Conjugate Gradient Solver
// =============================================================================

int ConjugateGradient::solve(
    const SparseSymmetricMatrix& L,
    double sigma,
    const std::vector<double>& b,
    std::vector<double>& x,
    double tol,
    int maxiter
) {
    const size_t n = b.size();
    if (x.size() != n) {
        x.resize(n, 0.0);
    }
    
    // r = b - (L - σI)x = b - Lx + σx
    std::vector<double> r(n);
    std::vector<double> Lx(n);
    L.matvec(x.data(), Lx.data());  // Lx = L * x
    
    for (size_t i = 0; i < n; ++i) {
        r[i] = b[i] - Lx[i] + sigma * x[i];
    }
    
    // p = r (direction)
    std::vector<double> p(r);
    
    double rs_old = vec::dot(r, r);
    double b_norm = vec::norm(b);
    if (b_norm < std::numeric_limits<double>::epsilon()) {
        b_norm = 1.0;
    }
    
    std::vector<double> Ap(n);
    
    for (int iter = 0; iter < maxiter; ++iter) {
        // Check convergence
        double r_norm = std::sqrt(rs_old);
        if (r_norm / b_norm < tol) {
            return iter;
        }
        
        // Ap = (L - σI)p = Lp - σp
        L.matvec(p.data(), Ap.data());
        for (size_t i = 0; i < n; ++i) {
            Ap[i] -= sigma * p[i];
        }
        
        // α = r^T r / p^T Ap
        double pAp = vec::dot(p, Ap);
        if (std::abs(pAp) < std::numeric_limits<double>::epsilon()) {
            // Breakdown - matrix might be singular
            return -1;
        }
        double alpha = rs_old / pAp;
        
        // x = x + α*p
        vec::axpy(alpha, p, x);
        
        // r = r - α*Ap
        vec::axpy(-alpha, Ap, r);
        
        // β = r_new^T r_new / r_old^T r_old
        double rs_new = vec::dot(r, r);
        double beta = rs_new / rs_old;
        
        // p = r + β*p
        for (size_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        
        rs_old = rs_new;
    }
    
    return maxiter;  // Did not converge
}

// =============================================================================
// Tridiagonal QR Eigensolver
// =============================================================================

void TridiagonalEigensolver::givens_rotation(double a, double b, double& c, double& s) {
    if (std::abs(b) < std::numeric_limits<double>::epsilon()) {
        c = 1.0;
        s = 0.0;
    } else if (std::abs(b) > std::abs(a)) {
        double t = -a / b;
        s = 1.0 / std::sqrt(1.0 + t * t);
        c = s * t;
    } else {
        double t = -b / a;
        c = 1.0 / std::sqrt(1.0 + t * t);
        s = c * t;
    }
}

double TridiagonalEigensolver::wilkinson_shift(double a, double b, double c) {
    // For 2x2 trailing submatrix:
    // | a   b |
    // | b   c |
    // Wilkinson shift is eigenvalue of this 2x2 closest to c
    double delta = (a - c) / 2.0;
    double sign = (delta >= 0) ? 1.0 : -1.0;
    return c - sign * b * b / (std::abs(delta) + std::sqrt(delta * delta + b * b));
}

void TridiagonalEigensolver::implicit_qr_step(
    std::vector<double>& diag,
    std::vector<double>& offdiag,
    std::vector<std::vector<double>>& Q,
    size_t lo,
    size_t hi
) {
    size_t n = diag.size();
    
    // Wilkinson shift
    double shift = wilkinson_shift(
        diag[hi - 1],
        offdiag[hi - 1],
        diag[hi]
    );
    
    double x = diag[lo] - shift;
    double z = offdiag[lo];
    
    for (size_t k = lo; k < hi; ++k) {
        // Compute Givens rotation to zero out z
        double c, s;
        givens_rotation(x, z, c, s);
        
        // Apply rotation to tridiagonal matrix
        if (k > lo) {
            offdiag[k - 1] = c * offdiag[k - 1] - s * z;
        }
        
        double d1 = diag[k];
        double d2 = diag[k + 1];
        double e = offdiag[k];
        
        diag[k] = c * c * d1 + s * s * d2 - 2.0 * c * s * e;
        diag[k + 1] = s * s * d1 + c * c * d2 + 2.0 * c * s * e;
        offdiag[k] = c * s * (d1 - d2) + (c * c - s * s) * e;
        
        // Update eigenvector matrix Q
        for (size_t i = 0; i < n; ++i) {
            double q1 = Q[i][k];
            double q2 = Q[i][k + 1];
            Q[i][k] = c * q1 - s * q2;
            Q[i][k + 1] = s * q1 + c * q2;
        }
        
        // Prepare for next iteration
        if (k < hi - 1) {
            x = offdiag[k];
            z = -s * offdiag[k + 1];
            offdiag[k + 1] *= c;
        }
    }
}

std::vector<TridiagonalEigensolver::EigenPair> TridiagonalEigensolver::solve(
    const TridiagonalMatrix& T
) {
    const size_t n = T.size();
    if (n == 0) {
        return {};
    }
    
    // Copy diagonal and off-diagonal
    std::vector<double> diag = T.alpha;
    std::vector<double> offdiag = T.beta;
    offdiag.resize(n - 1);
    
    // Initialize Q = I (eigenvector accumulator)
    std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        Q[i][i] = 1.0;
    }
    
    // Implicit QR iteration
    const double tol = std::numeric_limits<double>::epsilon() * 
                       *std::max_element(diag.begin(), diag.end());
    const int max_iter = 30 * static_cast<int>(n);
    
    size_t hi = n - 1;
    int iter = 0;
    
    while (hi > 0 && iter < max_iter) {
        // Find largest unreduced submatrix
        size_t lo = hi;
        while (lo > 0 && std::abs(offdiag[lo - 1]) > tol) {
            --lo;
        }
        
        if (lo == hi) {
            // Eigenvalue converged
            --hi;
        } else {
            // Check for splits
            bool found_split = false;
            for (size_t k = hi; k > lo; --k) {
                if (std::abs(offdiag[k - 1]) <= tol * (std::abs(diag[k - 1]) + std::abs(diag[k]))) {
                    offdiag[k - 1] = 0.0;
                    if (k == hi) {
                        --hi;
                        found_split = true;
                        break;
                    }
                }
            }
            
            if (!found_split) {
                // Perform QR step
                implicit_qr_step(diag, offdiag, Q, lo, hi);
            }
        }
        ++iter;
    }
    
    // Build result
    std::vector<EigenPair> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i].eigenvalue = diag[i];
        result[i].eigenvector.resize(n);
        for (size_t j = 0; j < n; ++j) {
            result[i].eigenvector[j] = Q[j][i];
        }
    }
    
    // Sort by eigenvalue ascending
    std::sort(result.begin(), result.end(),
        [](const EigenPair& a, const EigenPair& b) {
            return a.eigenvalue < b.eigenvalue;
        });
    
    return result;
}

// =============================================================================
// Lanczos Solver
// =============================================================================

LanczosSolver::LanczosSolver(const LanczosConfig& config)
    : config_(config)
{}

void LanczosSolver::report_progress(const std::string& stage, int current, int total) {
    if (progress_callback_) {
        progress_callback_(stage, current, total);
    }
}

void LanczosSolver::reorthogonalize(
    std::vector<double>& w,
    const std::vector<std::vector<double>>& Q,
    size_t num_vecs
) {
    // Full reorthogonalization with two passes for numerical stability
    // (Modified Gram-Schmidt twice)
    
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t j = 0; j < num_vecs; ++j) {
            double h = vec::dot(w, Q[j]);
            vec::axpy(-h, Q[j], w);
        }
    }
}

void LanczosSolver::lanczos_iteration(
    const SparseSymmetricMatrix& L,
    std::vector<std::vector<double>>& Q,
    TridiagonalMatrix& T,
    int m
) {
    const size_t n = L.dimension();
    
    // Reserve space
    Q.resize(m);
    T.alpha.resize(m);
    T.beta.resize(m);
    
    // Random starting vector
    std::mt19937_64 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    Q[0].resize(n);
    for (size_t i = 0; i < n; ++i) {
        Q[0][i] = dist(rng);
    }
    vec::normalize(Q[0]);
    
    std::vector<double> w(n);
    double beta_prev = 0.0;
    
    for (int j = 0; j < m; ++j) {
        report_progress("Lanczos iteration", j + 1, m);
        
        // w = L * q_j
        L.matvec(Q[j].data(), w.data());
        
        // α_j = q_j^T * w
        double alpha = vec::dot(Q[j], w);
        T.alpha[j] = alpha;
        
        // w = w - α_j * q_j
        vec::axpy(-alpha, Q[j], w);
        
        // w = w - β_{j-1} * q_{j-1} (if j > 0)
        if (j > 0) {
            vec::axpy(-beta_prev, Q[j - 1], w);
        }
        
        // Full reorthogonalization
        reorthogonalize(w, Q, j + 1);
        
        // β_j = ||w||
        double beta = vec::norm(w);
        T.beta[j] = beta;
        
        // Check for breakdown (lucky breakdown = invariant subspace found)
        if (beta < config_.reorth_tol) {
            T.alpha.resize(j + 1);
            T.beta.resize(j + 1);
            Q.resize(j + 1);
            return;
        }
        
        // q_{j+1} = w / β_j
        if (j + 1 < m) {
            Q[j + 1].resize(n);
            for (size_t i = 0; i < n; ++i) {
                Q[j + 1][i] = w[i] / beta;
            }
        }
        
        beta_prev = beta;
    }
}

void LanczosSolver::shift_invert_lanczos(
    const SparseSymmetricMatrix& L,
    std::vector<std::vector<double>>& Q,
    TridiagonalMatrix& T,
    int m
) {
    const size_t n = L.dimension();
    
    // Reserve space
    Q.resize(m);
    T.alpha.resize(m);
    T.beta.resize(m);
    
    // Random starting vector
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    Q[0].resize(n);
    for (size_t i = 0; i < n; ++i) {
        Q[0][i] = dist(rng);
    }
    vec::normalize(Q[0]);
    
    std::vector<double> w(n);
    std::vector<double> cg_x(n);
    double beta_prev = 0.0;
    
    for (int j = 0; j < m; ++j) {
        report_progress("Shift-invert Lanczos", j + 1, m);
        
        // w = (L - σI)^{-1} * q_j via CG
        std::fill(cg_x.begin(), cg_x.end(), 0.0);
        int cg_iters = ConjugateGradient::solve(
            L, config_.shift_sigma, Q[j], cg_x,
            config_.cg_tolerance, config_.cg_max_iterations
        );
        
        if (cg_iters < 0) {
            // CG failed - fall back to standard Lanczos
            std::cerr << "Warning: CG failed at iteration " << j 
                      << ", falling back to direct Lanczos\n";
            lanczos_iteration(L, Q, T, m);
            return;
        }
        
        vec::copy(cg_x, w);
        
        // α_j = q_j^T * w
        double alpha = vec::dot(Q[j], w);
        T.alpha[j] = alpha;
        
        // w = w - α_j * q_j
        vec::axpy(-alpha, Q[j], w);
        
        // w = w - β_{j-1} * q_{j-1}
        if (j > 0) {
            vec::axpy(-beta_prev, Q[j - 1], w);
        }
        
        // Full reorthogonalization
        reorthogonalize(w, Q, j + 1);
        
        // β_j = ||w||
        double beta = vec::norm(w);
        T.beta[j] = beta;
        
        if (beta < config_.reorth_tol) {
            T.alpha.resize(j + 1);
            T.beta.resize(j + 1);
            Q.resize(j + 1);
            return;
        }
        
        if (j + 1 < m) {
            Q[j + 1].resize(n);
            for (size_t i = 0; i < n; ++i) {
                Q[j + 1][i] = w[i] / beta;
            }
        }
        
        beta_prev = beta;
    }
}

void LanczosSolver::extract_ritz_pairs(
    const std::vector<std::vector<double>>& Q,
    const TridiagonalMatrix& T,
    std::vector<double>& eigenvalues,
    std::vector<std::vector<double>>& eigenvectors,
    std::vector<double>& residuals
) {
    const size_t m = Q.size();
    const size_t n = Q[0].size();
    const int k = config_.num_eigenpairs;
    
    // Solve tridiagonal eigenproblem
    auto ritz_pairs = TridiagonalEigensolver::solve(T);
    
    // For shift-invert: eigenvalues of (L-σI)^{-1} are 1/(λ-σ)
    // So we convert back: λ = σ + 1/θ where θ is Ritz value
    if (config_.use_shift_invert) {
        for (auto& pair : ritz_pairs) {
            if (std::abs(pair.eigenvalue) > std::numeric_limits<double>::epsilon()) {
                pair.eigenvalue = config_.shift_sigma + 1.0 / pair.eigenvalue;
            } else {
                pair.eigenvalue = 0.0;  // Near-zero eigenvalue (null space)
            }
        }
        // Re-sort after transform
        std::sort(ritz_pairs.begin(), ritz_pairs.end(),
            [](const auto& a, const auto& b) {
                return a.eigenvalue < b.eigenvalue;
            });
    }
    
    // Skip the zero eigenvalue (first one for Laplacian)
    // Take k smallest non-zero eigenvalues
    int start_idx = 0;
    while (start_idx < static_cast<int>(ritz_pairs.size()) &&
           std::abs(ritz_pairs[start_idx].eigenvalue) < 1e-10) {
        ++start_idx;
    }
    
    eigenvalues.clear();
    eigenvectors.clear();
    residuals.clear();
    
    auto& pool = ThreadPool::instance();
    
    for (int i = start_idx; i < start_idx + k && i < static_cast<int>(ritz_pairs.size()); ++i) {
        eigenvalues.push_back(ritz_pairs[i].eigenvalue);
        
        // Compute eigenvector: v = Q_m * y
        std::vector<double> v(n, 0.0);
        const auto& y = ritz_pairs[i].eigenvector;
        
        // Parallel matrix-vector: v = sum_j y[j] * Q[j]
        for (size_t j = 0; j < m; ++j) {
            vec::axpy(y[j], Q[j], v);
        }
        
        eigenvectors.push_back(std::move(v));
        
        // Residual estimate: |β_m * y_m|
        double res = std::abs(T.beta.back() * y.back());
        residuals.push_back(res);
    }
}

bool LanczosSolver::check_convergence(const std::vector<double>& residuals, int k) {
    if (static_cast<int>(residuals.size()) < k) {
        return false;
    }
    
    for (int i = 0; i < k; ++i) {
        if (residuals[i] > config_.convergence_tol) {
            return false;
        }
    }
    return true;
}

LanczosResult LanczosSolver::solve(const SparseSymmetricMatrix& L) {
    LanczosResult result;
    result.converged = false;
    result.iterations_used = 0;
    
    const size_t n = L.dimension();
    if (n == 0) {
        return result;
    }
    
    // Adaptive: start with smaller Krylov subspace, expand if needed
    int m = std::min(std::max(2 * config_.num_eigenpairs + 10, 30), config_.max_iterations);
    
    std::vector<std::vector<double>> Q;
    TridiagonalMatrix T;
    
    while (m <= config_.max_iterations) {
        report_progress("Building Krylov subspace", m, config_.max_iterations);
        
        if (config_.use_shift_invert) {
            shift_invert_lanczos(L, Q, T, m);
        } else {
            lanczos_iteration(L, Q, T, m);
        }
        
        result.iterations_used = static_cast<int>(Q.size());
        
        // Extract Ritz pairs
        extract_ritz_pairs(Q, T, result.eigenvalues, result.eigenvectors, result.residuals);
        
        // Check convergence
        if (check_convergence(result.residuals, config_.num_eigenpairs)) {
            result.converged = true;
            report_progress("Converged", m, m);
            break;
        }
        
        // Expand subspace
        m = std::min(m + 20, config_.max_iterations);
    }
    
    return result;
}

} // namespace lanczos
} // namespace hypercube
