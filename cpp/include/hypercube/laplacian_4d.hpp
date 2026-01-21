#pragma once

/**
 * Laplacian Eigenmaps + Gram-Schmidt for 4D Hypercube Projection
 *
 * Projects high-dimensional embeddings into the 4D hypercube coordinate space.
 * This module prioritizes industrial libraries (MKL, HNSWLib, Eigen) over custom numerics.
 *
 * Algorithm:
 * 1. Build k-NN similarity graph using HNSWLib (or brute force fallback)
 * 2. Compute unnormalized graph Laplacian L = D - W
 * 3. Solve for 4 smallest non-zero eigenvectors using MKL DSYEV (or Eigen fallback)
 * 4. Apply Gram-Schmidt orthonormalization
 * 5. Normalize to [0, 2^32-1]^4 hypercube coordinates
 *
 * Note: This module is not bit-deterministic and favors speed over reproducibility.
 * Use only for approximate projections where industrial library performance matters.
 *
 * References:
 * - Belkin & Niyogi, "Laplacian Eigenmaps for Dimensionality Reduction", 2003
 */

#include "hypercube/types.hpp"
#include "hypercube/dispatch.h"
#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <functional>


namespace hypercube {

// Forward declarations
struct LaplacianConfig;
struct ProjectionResult;

// Eigenvector computation: MKL_DENSE if available, EIGEN_DENSE fallback

// No runtime choice for neighbor search: always HNSW if available, brute force fallback

/**
 * Anchor point for constrained Laplacian projection
 * Used to align new embeddings with existing 4D coordinate system
 */
struct AnchorPoint {
    size_t token_index;                 // Index in the embedding matrix
    std::array<double, 4> coords_4d;    // Known 4D coordinates from database
    double weight = 1.0;                // Constraint weight (higher = stricter)
};

/**
 * Configuration for Laplacian eigenmap projection
 *
 * Key backends:
 * - NeighborSearchBackend: HNSW (fast, approximate) vs BRUTE_FORCE (exact but slow)
 * - EigenBackend: MKL_DENSE (fastest), EIGEN_DENSE (portable), LANCZOS_SPARSE (large matrices)
 *
 * Policy: Use MKL_DENSE for small matrices (< dense_threshold), LANCZOS_SPARSE for large ones.
 */
struct LaplacianConfig {
    // Graph construction parameters
    int k_neighbors = 50;               // k for k-NN graph construction (increased for better graphs)
    float similarity_threshold = 0.0f;  // Minimum similarity for edges (negative = include all)

    // HNSW parameters (always used for neighbor search)
    size_t hnsw_M = 16;                 // Max connections per layer (higher = more accurate, slower)
    size_t hnsw_ef_construction = 200;  // Construction-time parameter (higher = more accurate)

    // Eigenvalue solver parameters (MKL_DENSE first, Eigen fallback)
    size_t dense_threshold = 20000;     // Matrix size threshold (always dense for now)
    int power_iterations = 200;         // Iterations for inverse power method (unused)
    int num_threads = 0;                // 0 = auto-detect

    // Convergence tolerance (unused in dense solvers)
    double convergence_tol = 1e-4;      // Tolerance for any iterative methods
    int max_deflation_iterations = 50;  // Unused now

    // Projection parameters
    bool project_to_sphere = true;      // Project final coords onto hypersphere
    double sphere_radius = 1.0;         // Radius of target hypersphere (before scaling)
    bool verbose = false;               // Enable verbose debug output

    // Anchor constraints for aligning with existing 4D space
    double anchor_weight = 10.0;        // Weight for anchor constraints (higher = stricter alignment)
};

/**
 * Result of projecting embeddings to 4D
 */
struct ProjectionResult {
    std::vector<std::array<uint32_t, 4>> coords;  // 4D hypercube coordinates
    std::vector<int64_t> hilbert_lo;              // Lower 64 bits of Hilbert index
    std::vector<int64_t> hilbert_hi;              // Upper 64 bits of Hilbert index

    // Statistics
    std::array<double, 4> eigenvalues;            // The 4 eigenvalues used
    double total_variance_explained;              // Sum of eigenvalues / total
    size_t edge_count;                            // Number of edges in similarity graph
    bool converged;                               // Whether the eigensolver converged
};

/**
 * Sparse symmetric matrix in CSR format for Laplacian computation
 */
class SparseSymmetricMatrix {
public:
    SparseSymmetricMatrix() = default;
    explicit SparseSymmetricMatrix(size_t n);
    ~SparseSymmetricMatrix();  // Destroy MKL resources

    // Add edge (symmetric: adds both i->j and j->i)
    void add_edge(size_t i, size_t j, double weight);
    
    // Finalize matrix structure for efficient access
    void finalize();
    
    // Matrix-vector product: y = A * x
    void multiply(const std::vector<double>& x, std::vector<double>& y) const;
    
    // Matrix-vector product (pointer version for Lanczos compatibility)
    void matvec(const double* x, double* y) const;

    // Fallback matrix-vector product implementation
    void fallback_matvec(const double* x, double* y) const;
    
    // Get diagonal element
    double get_diagonal(size_t i) const;
    
    // Set diagonal element
    void set_diagonal(size_t i, double value);
    
    // Number of rows/columns
    size_t size() const { return n_; }
    size_t dimension() const { return n_; }  // Alias for Lanczos compatibility
    
    // Get row degree (sum of off-diagonal weights)
    double get_degree(size_t i) const;
    
    // Validate CSR structure (for debugging)
    bool validate_csr() const;

    // Check if matrix is symmetric
    bool is_symmetric() const;

    // Direct access to CSR structure for MKL/external solvers
    const std::vector<size_t>& row_ptr() const { return row_ptr_; }
    const std::vector<size_t>& col_idx() const { return col_idx_; }
    const std::vector<double>& values() const { return values_; }
    const std::vector<double>& diagonal() const { return diagonal_; }
    
    // Iterate over non-zero entries (works before or after finalization)
    template<typename Func>
    void for_each_edge(Func&& func) const {
        if (finalized_) {
            // Use CSR structure
            for (size_t i = 0; i < n_; ++i) {
                for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
                    func(i, col_idx_[k], values_[k]);
                }
            }
        } else {
            // Use adjacency list structure
            for (size_t i = 0; i < n_; ++i) {
                for (const auto& [j, w] : adj_[i]) {
                    func(i, j, w);
                }
            }
        }
    }
    
private:
    size_t n_ = 0;
    std::vector<size_t> row_ptr_;
    std::vector<size_t> col_idx_;
    std::vector<double> values_;
    std::vector<double> diagonal_;

    // Temporary storage during construction
    std::vector<std::vector<std::pair<size_t, double>>> adj_;
    bool finalized_ = false;

};

/**
 * Main class for Laplacian eigenmap projection to 4D
 */
class LaplacianProjector {
public:
    explicit LaplacianProjector(const LaplacianConfig& config = LaplacianConfig{});
    
    /**
     * Project embeddings to 4D hypercube coordinates
     *
     * @param embeddings  n x d matrix of embeddings (n tokens, d dimensions)
     * @param labels      Optional labels for each token (for progress reporting)
     * @param anchors     Optional anchor points with known 4D coordinates (for alignment)
     * @return ProjectionResult with 4D coordinates and Hilbert indices
     */
    ProjectionResult project(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<std::string>& labels = {},
        const std::vector<AnchorPoint>& anchors = {}
    );
    
    /**
     * Set progress callback for long operations
     */
    using ProgressCallback = std::function<void(const std::string& stage, size_t current, size_t total)>;
    void set_progress_callback(ProgressCallback callback) { progress_callback_ = std::move(callback); }
    
private:
    LaplacianConfig config_;
    ProgressCallback progress_callback_;
    
    // Build k-NN similarity graph
    SparseSymmetricMatrix build_similarity_graph(
        const std::vector<std::vector<float>>& embeddings
    );
    
    // Ensure the similarity graph is connected (add edges if needed)
    void ensure_connectivity(
        SparseSymmetricMatrix& W,
        const std::vector<std::vector<float>>& embeddings
    );
    
    // Compute unnormalized Laplacian L = D - W
    SparseSymmetricMatrix build_laplacian(const SparseSymmetricMatrix& W);
    
    // Solve for 4 smallest non-zero eigenvectors using MKL or Eigen dense solver
    std::vector<std::vector<double>> solve_eigenvectors(
        SparseSymmetricMatrix& L,
        int k,
        std::array<double, 4>& eigenvalues_out
    );

    // MKL dense solver for eigenvectors
    std::vector<std::vector<double>> solve_eigenvectors_mkl_dense(
        SparseSymmetricMatrix& L,
        int k,
        std::array<double, 4>& eigenvalues_out
    );

    // Eigen dense solver for eigenvectors
    std::vector<std::vector<double>> solve_eigenvectors_eigen_dense(
        SparseSymmetricMatrix& L,
        int k,
        std::array<double, 4>& eigenvalues_out
    );

    // Gram-Schmidt orthonormalization on columns
    void gram_schmidt_columns(std::vector<std::vector<double>>& Y);
    
    // Normalize to hypercube [0, 2^32-1]^4
    std::vector<std::array<uint32_t, 4>> normalize_to_hypercube(
        const std::vector<std::vector<double>>& U
    );
    
    // Optional: project to hypersphere
    void project_to_sphere(std::vector<std::array<uint32_t, 4>>& coords);
    
    // Report progress
    void report_progress(const std::string& stage, size_t current, size_t total);
};

// SIMD-optimized vector operations (dispatch-based)
namespace simd {
    inline float dot_product(const float* a, const float* b, size_t n) {
        return get_kernels().dot_product_f(a, b, n);
    }
    inline double dot_product_d(const double* a, const double* b, size_t n) {
        return get_kernels().dot_product_d(a, b, n);
    }
    inline void scale_inplace(double* v, double s, size_t n) {
        get_kernels().scale_inplace_d(v, s, n);
    }
    inline void subtract_scaled(double* a, const double* b, double s, size_t n) {
        get_kernels().subtract_scaled_d(a, b, s, n);
    }
    inline double norm(const double* v, size_t n) {
        return get_kernels().norm_d(v, n);
    }
    inline void normalize(double* v, size_t n) {
        double nrm = norm(v, n);
        if (nrm > 1e-12) {
            scale_inplace(v, 1.0 / nrm, n);
        }
    }
}

} // namespace hypercube
