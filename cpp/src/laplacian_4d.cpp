/**
 * Laplacian Eigenmaps + Gram-Schmidt for 4D Hypercube Projection
 * 
 * Implementation of spectral dimensionality reduction for projecting
 * high-dimensional model embeddings into the 4D hypercube coordinate space.
 * 
 * Key differences from manifold_4d.cpp:
 * - Uses UNNORMALIZED Laplacian (L = D - W) per spec
 * - Applies Gram-Schmidt to COLUMNS of eigenvector matrix
 * - Direct integration with safetensor ingestion (no shape table)
 * - Outputs directly to atom/composition tables
 * - Uses LANCZOS algorithm for smallest eigenvalues (not power iteration!)
 * 
 * Backend Priority (eigensolvers):
 * 1. Intel MKL DSYEVR - Fastest on Intel CPUs, AVX-512 optimized
 * 2. Eigen SelfAdjointEigenSolver - Good performance, portable
 * 3. Custom Jacobi - Fallback, always available
 */

#include "hypercube/laplacian_4d.hpp"
#include "hypercube/lanczos.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/embedding_ops.hpp"  // Centralized SIMD operations

#include <algorithm>
#include <cmath>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <limits>
#include <queue>

// Intel MKL for optimized eigensolvers (highest priority)
#if defined(HAS_MKL) && HAS_MKL
#include <mkl.h>
#define USE_MKL_SOLVER 1
#define USE_EIGEN_SOLVER 0
#elif defined(HAS_EIGEN) && HAS_EIGEN
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#define USE_MKL_SOLVER 0
#define USE_EIGEN_SOLVER 1
#else
#define USE_MKL_SOLVER 0
#define USE_EIGEN_SOLVER 0
#endif

// HNSWLIB for fast k-NN (O(n log n) vs O(n²) brute force)
#if defined(HAS_HNSWLIB) && HAS_HNSWLIB
#include <hnswlib/hnswlib.h>
#define USE_HNSWLIB 1
#else
#define USE_HNSWLIB 0
#endif

namespace hypercube {

// Use centralized SIMD implementations from embedding_ops.hpp
using embedding::cosine_similarity;
using embedding::l2_distance;

// =============================================================================
// Local SIMD helpers for double-precision (Gram-Schmidt, eigenvector ops)
// =============================================================================

namespace simd {

double dot_product_d(const double* a, const double* b, size_t n) {
#if defined(__AVX2__) || defined(__AVX__)
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        sum_vec = _mm256_fmadd_pd(va, vb, sum_vec);
    }
    
    // Horizontal sum
    __m128d hi = _mm256_extractf128_pd(sum_vec, 1);
    __m128d lo = _mm256_castpd256_pd128(sum_vec);
    __m128d sum128 = _mm_add_pd(hi, lo);
    sum128 = _mm_hadd_pd(sum128, sum128);
    double sum = _mm_cvtsd_f64(sum128);
    
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

void scale_inplace(double* v, double s, size_t n) {
#if defined(__AVX2__) || defined(__AVX__)
    __m256d s_vec = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vv = _mm256_loadu_pd(v + i);
        vv = _mm256_mul_pd(vv, s_vec);
        _mm256_storeu_pd(v + i, vv);
    }
    for (; i < n; ++i) v[i] *= s;
#else
    for (size_t i = 0; i < n; ++i) v[i] *= s;
#endif
}

void subtract_scaled(double* a, const double* b, double s, size_t n) {
#if defined(__AVX2__) || defined(__AVX__)
    __m256d s_vec = _mm256_set1_pd(s);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        va = _mm256_fnmadd_pd(vb, s_vec, va);  // a - s*b
        _mm256_storeu_pd(a + i, va);
    }
    for (; i < n; ++i) a[i] -= s * b[i];
#else
    for (size_t i = 0; i < n; ++i) a[i] -= s * b[i];
#endif
}

double norm(const double* v, size_t n) {
    return std::sqrt(dot_product_d(v, v, n));
}

void normalize(double* v, size_t n) {
    double nrm = norm(v, n);
    if (nrm > 1e-12) {
        scale_inplace(v, 1.0 / nrm, n);
    }
}

} // namespace simd

// =============================================================================
// SparseSymmetricMatrix Implementation
// =============================================================================

SparseSymmetricMatrix::SparseSymmetricMatrix(size_t n)
    : n_(n), diagonal_(n, 0.0), adj_(n), finalized_(false) {}

void SparseSymmetricMatrix::add_edge(size_t i, size_t j, double weight) {
    if (finalized_) return;
    if (i >= n_ || j >= n_) return;
    
    // Symmetric: add both directions
    adj_[i].emplace_back(j, weight);
    if (i != j) {
        adj_[j].emplace_back(i, weight);
    }
}

void SparseSymmetricMatrix::finalize() {
    if (finalized_) return;
    
    // Sort and deduplicate adjacency lists
    for (size_t i = 0; i < n_; ++i) {
        auto& list = adj_[i];
        std::sort(list.begin(), list.end(), 
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Merge duplicates by averaging
        if (!list.empty()) {
            std::vector<std::pair<size_t, double>> merged;
            merged.reserve(list.size());
            merged.push_back(list[0]);
            for (size_t k = 1; k < list.size(); ++k) {
                if (list[k].first == merged.back().first) {
                    // Average duplicates
                    merged.back().second = (merged.back().second + list[k].second) / 2.0;
                } else {
                    merged.push_back(list[k]);
                }
            }
            list = std::move(merged);
        }
    }
    
    // Convert to CSR
    row_ptr_.resize(n_ + 1);
    row_ptr_[0] = 0;
    
    size_t total_nnz = 0;
    for (size_t i = 0; i < n_; ++i) {
        total_nnz += adj_[i].size();
        row_ptr_[i + 1] = total_nnz;
    }
    
    col_idx_.resize(total_nnz);
    values_.resize(total_nnz);
    
    size_t idx = 0;
    for (size_t i = 0; i < n_; ++i) {
        for (const auto& [j, w] : adj_[i]) {
            col_idx_[idx] = j;
            values_[idx] = w;
            ++idx;
        }
    }
    
    // Clear temporary storage
    adj_.clear();
    adj_.shrink_to_fit();
    finalized_ = true;
}

void SparseSymmetricMatrix::multiply(const std::vector<double>& x, std::vector<double>& y) const {
    if (x.size() != n_ || y.size() != n_) return;
    matvec(x.data(), y.data());
}

void SparseSymmetricMatrix::matvec(const double* x, double* y) const {
    if (!finalized_) return;
    
    for (size_t i = 0; i < n_; ++i) {
        double sum = diagonal_[i] * x[i];
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            sum += values_[k] * x[col_idx_[k]];
        }
        y[i] = sum;
    }
}

double SparseSymmetricMatrix::get_diagonal(size_t i) const {
    return (i < n_) ? diagonal_[i] : 0.0;
}

void SparseSymmetricMatrix::set_diagonal(size_t i, double value) {
    if (i < n_) diagonal_[i] = value;
}

double SparseSymmetricMatrix::get_degree(size_t i) const {
    if (i >= n_ || !finalized_) return 0.0;
    double sum = 0.0;
    for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
        sum += values_[k];
    }
    return sum;
}

// =============================================================================
// LaplacianProjector Implementation
// =============================================================================

LaplacianProjector::LaplacianProjector(const LaplacianConfig& config)
    : config_(config) {}

void LaplacianProjector::report_progress(const std::string& stage, size_t current, size_t total) {
    if (progress_callback_) {
        progress_callback_(stage, current, total);
    }
}

SparseSymmetricMatrix LaplacianProjector::build_similarity_graph(
    const std::vector<std::vector<float>>& embeddings
) {
    const size_t n = embeddings.size();
    SparseSymmetricMatrix W(n);
    
    if (n == 0) return W;
    
    const size_t dim = embeddings[0].size();
    const int k = config_.k_neighbors;
    const float threshold = config_.similarity_threshold;
    
    // Determine thread count
    int num_threads = config_.num_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads == 0) num_threads = 4;
    }

#if USE_HNSWLIB
    // =========================================================================
    // HNSWLIB PATH: O(n log n) approximate k-NN using HNSW index
    // Much faster than brute-force O(n²) for large datasets
    // =========================================================================
    std::cerr << "[HNSWLIB] Building HNSW index for " << n << " points, dim=" << dim << "\n";
    auto hnsw_start = std::chrono::steady_clock::now();
    
    // Use inner product space (for cosine similarity, normalize vectors first)
    hnswlib::InnerProductSpace space(dim);
    
    // HNSW parameters
    size_t M = 16;         // Max connections per layer (higher = more accurate, slower)
    size_t ef_construction = 200;  // Construction-time parameter (higher = more accurate)
    
    hnswlib::HierarchicalNSW<float> index(&space, n, M, ef_construction);
    
    // Normalize embeddings for cosine similarity via inner product
    std::vector<std::vector<float>> normalized(n);
    {
        // Parallel normalization using std::thread
        std::vector<std::thread> norm_threads;
        std::atomic<size_t> norm_idx{0};
        
        auto normalize_worker = [&]() {
            while (true) {
                size_t i = norm_idx.fetch_add(1);
                if (i >= n) break;
                
                normalized[i].resize(dim);
                float norm_sq = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    norm_sq += embeddings[i][d] * embeddings[i][d];
                }
                float norm = std::sqrt(norm_sq);
                if (norm > 1e-10f) {
                    for (size_t d = 0; d < dim; ++d) {
                        normalized[i][d] = embeddings[i][d] / norm;
                    }
                } else {
                    std::copy(embeddings[i].begin(), embeddings[i].end(), normalized[i].begin());
                }
            }
        };
        
        for (int t = 0; t < num_threads; ++t) {
            norm_threads.emplace_back(normalize_worker);
        }
        for (auto& t : norm_threads) t.join();
    }
    
    // Add points to index (must be sequential for HNSWLIB)
    for (size_t i = 0; i < n; ++i) {
        index.addPoint(normalized[i].data(), i);
    }
    
    auto build_end = std::chrono::steady_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - hnsw_start).count();
    std::cerr << "[HNSWLIB] Index built in " << build_ms << " ms\n";
    
    // Query k-NN for each point using std::thread pool
    index.setEf(std::max(static_cast<size_t>(k * 2), static_cast<size_t>(50)));  // Query-time parameter
    
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> thread_edges(num_threads);
    std::atomic<size_t> progress{0};
    std::atomic<size_t> query_idx{0};
    
    auto query_worker = [&](int tid) {
        auto& local_edges = thread_edges[tid];
        
        while (true) {
            size_t i = query_idx.fetch_add(1);
            if (i >= n) break;
            
            auto result = index.searchKnn(normalized[i].data(), k + 1);  // +1 to skip self
            
            while (!result.empty()) {
                auto [dist, j] = result.top();
                result.pop();
                
                if (j == i) continue;  // Skip self
                
                // Inner product of normalized vectors = cosine similarity
                float sim = 1.0f - dist;  // Convert distance to similarity
                
                if (sim > threshold && i < j) {
                    local_edges.emplace_back(i, j, static_cast<double>(sim));
                }
            }
            
            progress.fetch_add(1);
        }
    };
    
    std::vector<std::thread> query_threads;
    for (int t = 0; t < num_threads; ++t) {
        query_threads.emplace_back(query_worker, t);
    }
    for (auto& t : query_threads) t.join();
    
    auto query_end = std::chrono::steady_clock::now();
    auto query_ms = std::chrono::duration_cast<std::chrono::milliseconds>(query_end - build_end).count();
    std::cerr << "[HNSWLIB] k-NN queries completed in " << query_ms << " ms\n";
    
    // Merge edges
    size_t total_edges = 0;
    for (const auto& te : thread_edges) {
        for (const auto& [i, j, w] : te) {
            W.add_edge(i, j, w);
            ++total_edges;
        }
    }
    
    W.finalize();
    std::cerr << "[HNSWLIB] Built k-NN graph with " << total_edges << " edges\n";
    return W;
    
#else
    // =========================================================================
    // FALLBACK: Brute-force O(n²) k-NN (for systems without HNSWLIB)
    // =========================================================================
    std::cerr << "[BRUTEFORCE] Building k-NN graph for " << n << " points (O(n²))\n";
    
    // Thread-local edge buffers to avoid locking
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> thread_edges(num_threads);
    for (auto& te : thread_edges) te.reserve(n * k / num_threads);
    
    std::atomic<size_t> progress{0};
    
    auto worker = [&](int tid) {
        auto& local_edges = thread_edges[tid];
        
        // Priority queue for top-k neighbors: (similarity, index)
        using SimPair = std::pair<float, size_t>;
        
        for (size_t i = tid; i < n; i += num_threads) {
            const float* emb_i = embeddings[i].data();
            
            // Find k nearest neighbors by cosine similarity
            std::priority_queue<SimPair, std::vector<SimPair>, std::greater<SimPair>> top_k;
            
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                
                float sim = embedding::cosine_similarity(emb_i, embeddings[j].data(), dim);
                
                // Only consider positive similarities above threshold
                if (sim > threshold) {
                    if (top_k.size() < static_cast<size_t>(k)) {
                        top_k.push({sim, j});
                    } else if (sim > top_k.top().first) {
                        top_k.pop();
                        top_k.push({sim, j});
                    }
                }
            }
            
            // Add edges for this node's k-NN
            while (!top_k.empty()) {
                auto [sim, j] = top_k.top();
                top_k.pop();
                // Only add edge from lower to higher index to avoid duplicates
                if (i < j) {
                    local_edges.emplace_back(i, j, static_cast<double>(sim));
                }
            }
            
            progress.fetch_add(1);
        }
    };
    
    // Launch workers
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    // Progress reporting
    while (progress.load() < n) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        report_progress("Building k-NN graph", progress.load(), n);
    }
    
    for (auto& t : threads) t.join();
    report_progress("Building k-NN graph", n, n);
    
    // Merge thread-local edges into global matrix
    size_t total_edges = 0;
    for (const auto& te : thread_edges) {
        for (const auto& [i, j, w] : te) {
            W.add_edge(i, j, w);
            ++total_edges;
        }
    }
    
    W.finalize();
    
    std::cerr << "[LAPLACIAN] Built k-NN graph with " << total_edges << " edges\n";
    return W;
#endif  // USE_HNSWLIB
}

SparseSymmetricMatrix LaplacianProjector::build_laplacian(const SparseSymmetricMatrix& W) {
    const size_t n = W.size();
    SparseSymmetricMatrix L(n);
    
    // Copy structure from W but negate values for off-diagonal
    W.for_each_edge([&](size_t i, size_t j, double w) {
        if (i < j) {
            L.add_edge(i, j, -w);  // Off-diagonal: -W_ij
        }
    });
    
    L.finalize();
    
    // Set diagonal to degree: D_ii = sum_j W_ij
    for (size_t i = 0; i < n; ++i) {
        double degree = W.get_degree(i);
        L.set_diagonal(i, degree);
    }
    
    std::cerr << "[LAPLACIAN] Built unnormalized Laplacian L = D - W\n";
    return L;
}

std::vector<std::vector<double>> LaplacianProjector::find_smallest_eigenvectors(
    SparseSymmetricMatrix& L,
    int k,
    std::array<double, 4>& eigenvalues_out
) {
    const size_t n = L.size();
    
    // ==========================================================================
    // FOR SMALL MATRICES: Use dense eigendecomposition (much more reliable)
    // Threshold: 2000 elements means 2000*2000*8 = 32MB dense matrix
    // ==========================================================================
    if (n <= 2000) {
        std::cerr << "[EIGEN] Using dense eigendecomposition for " << n << " points\n";
        
        // Convert sparse Laplacian to dense
        std::vector<double> L_dense(n * n, 0.0);
        L.for_each_edge([&](size_t i, size_t j, double w) {
            L_dense[i * n + j] = w;
        });
        // Add diagonals
        for (size_t i = 0; i < n; ++i) {
            L_dense[i * n + i] = L.get_diagonal(i);
        }
        
#if USE_MKL_SOLVER
        // ==================================================================
        // INTEL MKL PATH: DSYEVR - RRR algorithm for symmetric eigenvalue
        // Fastest eigensolver available, exploits Intel CPU features
        // ==================================================================
        std::cerr << "[MKL] Using Intel MKL DSYEVR (optimized for Intel CPUs)\n";
        auto mkl_start = std::chrono::steady_clock::now();
        
        // MKL DSYEVR parameters
        char jobz = 'V';  // Compute eigenvalues and eigenvectors
        char range = 'I'; // Index range (smallest k+1 eigenvalues)
        char uplo = 'U';  // Upper triangle stored
        MKL_INT mkl_n = static_cast<MKL_INT>(n);
        MKL_INT lda = mkl_n;
        double vl = 0.0, vu = 0.0;  // Not used for range='I'
        MKL_INT il = 1;  // First eigenvalue index (1-based)
        MKL_INT iu = std::min(static_cast<MKL_INT>(k + 1), mkl_n);  // Last index (+1 to skip null)
        double abstol = 0.0;  // Use default (safe) tolerance
        MKL_INT m_found = 0;  // Number of eigenvalues found
        
        std::vector<double> eigenvalues(iu);
        std::vector<double> Z(n * iu);  // Eigenvector matrix
        MKL_INT ldz = mkl_n;
        std::vector<MKL_INT> isuppz(2 * iu);
        
        MKL_INT info = 0;
        
        // Query workspace size
        MKL_INT lwork = -1, liwork = -1;
        double work_query;
        MKL_INT iwork_query;
        dsyevr(&jobz, &range, &uplo, &mkl_n, L_dense.data(), &lda,
               &vl, &vu, &il, &iu, &abstol, &m_found,
               eigenvalues.data(), Z.data(), &ldz, isuppz.data(),
               &work_query, &lwork, &iwork_query, &liwork, &info);
        
        lwork = static_cast<MKL_INT>(work_query);
        liwork = iwork_query;
        std::vector<double> work(lwork);
        std::vector<MKL_INT> iwork(liwork);
        
        // Compute eigendecomposition
        dsyevr(&jobz, &range, &uplo, &mkl_n, L_dense.data(), &lda,
               &vl, &vu, &il, &iu, &abstol, &m_found,
               eigenvalues.data(), Z.data(), &ldz, isuppz.data(),
               work.data(), &lwork, iwork.data(), &liwork, &info);
        
        if (info != 0) {
            std::cerr << "[MKL] ERROR: DSYEVR failed with info=" << info << "\n";
            return {};
        }
        
        auto mkl_end = std::chrono::steady_clock::now();
        auto mkl_ms = std::chrono::duration_cast<std::chrono::milliseconds>(mkl_end - mkl_start).count();
        std::cerr << "[MKL] Eigendecomposition completed in " << mkl_ms << " ms\n";
        std::cerr << "[MKL] Found " << m_found << " eigenvalues\n";
        
        // Log eigenvalues
        std::cerr << "[MKL] Eigenvalues: ";
        for (MKL_INT i = 0; i < std::min(m_found, (MKL_INT)8); ++i) {
            std::cerr << eigenvalues[i] << " ";
        }
        std::cerr << "...\n";
        
        // Skip index 0 (null space at λ≈0), take next k eigenvectors
        std::vector<std::vector<double>> result;
        for (MKL_INT i = 1; i < m_found && static_cast<int>(i) <= k; ++i) {
            eigenvalues_out[i - 1] = eigenvalues[i];
            
            // Copy eigenvector column from Z matrix
            std::vector<double> ev(n);
            for (size_t r = 0; r < n; ++r) {
                ev[r] = Z[r + i * n];  // Column-major storage
            }
            result.push_back(std::move(ev));
        }
        
        return result;
        
#elif USE_EIGEN_SOLVER
        // ==================================================================
        // EIGEN3 OPTIMIZED PATH: SelfAdjointEigenSolver
        // Uses LAPACK-level optimizations, cache-efficient, vectorized
        // ==================================================================
        std::cerr << "[EIGEN] Using Eigen3 SelfAdjointEigenSolver (optimized)\n";
        auto eigen_start = std::chrono::steady_clock::now();
        
        // Map our data to Eigen matrix (no copy)
        Eigen::Map<Eigen::MatrixXd> L_eigen(L_dense.data(), n, n);
        
        // Compute eigenvalues and eigenvectors
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L_eigen);
        
        if (solver.info() != Eigen::Success) {
            std::cerr << "[EIGEN] ERROR: Eigendecomposition failed!\n";
            return {};
        }
        
        const auto& eigenvalues = solver.eigenvalues();
        const auto& eigenvectors = solver.eigenvectors();
        
        auto eigen_end = std::chrono::steady_clock::now();
        auto eigen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(eigen_end - eigen_start).count();
        std::cerr << "[EIGEN] Eigendecomposition completed in " << eigen_ms << " ms\n";
        
        // Eigenvalues are already sorted ascending by Eigen
        std::cerr << "[EIGEN] Eigenvalues: ";
        for (int i = 0; i < std::min(8, (int)n); ++i) {
            std::cerr << eigenvalues(i) << " ";
        }
        std::cerr << "...\n";
        
        // Skip index 0 (null space at λ≈0), take next k eigenvectors
        std::vector<std::vector<double>> result;
        for (int i = 1; i <= k && i < static_cast<int>(n); ++i) {
            eigenvalues_out[i - 1] = eigenvalues(i);
            
            // Copy eigenvector column
            std::vector<double> ev(n);
            for (size_t r = 0; r < n; ++r) {
                ev[r] = eigenvectors(r, i);
            }
            result.push_back(std::move(ev));
        }
        
        return result;
        
#else
        // ==================================================================
        // FALLBACK: Jacobi eigenvalue algorithm for dense symmetric matrices
        // O(n³) but reliable for small matrices
        // ==================================================================
        std::cerr << "[JACOBI] Using fallback Jacobi eigensolver\n";
        std::vector<double> eigenvalues(n);
        std::vector<std::vector<double>> eigenvectors(n, std::vector<double>(n, 0.0));
        
        // Initialize eigenvectors as identity
        for (size_t i = 0; i < n; ++i) {
            eigenvectors[i][i] = 1.0;
        }
        
        // Jacobi iteration
        const int max_sweeps = 50;
        const double eps = 1e-10;
        
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            double max_off = 0.0;
            
            for (size_t p = 0; p < n - 1; ++p) {
                for (size_t q = p + 1; q < n; ++q) {
                    double apq = L_dense[p * n + q];
                    if (std::abs(apq) < eps) continue;
                    if (std::abs(apq) > max_off) max_off = std::abs(apq);
                    
                    double app = L_dense[p * n + p];
                    double aqq = L_dense[q * n + q];
                    double theta = 0.5 * (aqq - app) / apq;
                    
                    double t = 1.0 / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                    if (theta < 0) t = -t;
                    
                    double c = 1.0 / std::sqrt(1.0 + t * t);
                    double s = t * c;
                    
                    // Update matrix
                    L_dense[p * n + p] = app - t * apq;
                    L_dense[q * n + q] = aqq + t * apq;
                    L_dense[p * n + q] = 0.0;
                    L_dense[q * n + p] = 0.0;
                    
                    for (size_t r = 0; r < n; ++r) {
                        if (r != p && r != q) {
                            double arp = L_dense[r * n + p];
                            double arq = L_dense[r * n + q];
                            L_dense[r * n + p] = L_dense[p * n + r] = c * arp - s * arq;
                            L_dense[r * n + q] = L_dense[q * n + r] = s * arp + c * arq;
                        }
                    }
                    
                    // Update eigenvectors
                    for (size_t r = 0; r < n; ++r) {
                        double vrp = eigenvectors[r][p];
                        double vrq = eigenvectors[r][q];
                        eigenvectors[r][p] = c * vrp - s * vrq;
                        eigenvectors[r][q] = s * vrp + c * vrq;
                    }
                }
            }
            
            if (max_off < eps) {
                std::cerr << "[JACOBI] Converged after " << (sweep + 1) << " sweeps\n";
                break;
            }
        }
        
        // Extract eigenvalues (diagonal of L_dense)
        for (size_t i = 0; i < n; ++i) {
            eigenvalues[i] = L_dense[i * n + i];
        }
        
        // Sort eigenvalues and eigenvectors (ascending)
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return eigenvalues[a] < eigenvalues[b];
        });
        
        // Log eigenvalues
        std::cerr << "[JACOBI] Eigenvalues: ";
        for (int i = 0; i < std::min(8, (int)n); ++i) {
            std::cerr << eigenvalues[indices[i]] << " ";
        }
        std::cerr << "...\n";
        
        // Skip index 0 (null space), take next k eigenvectors
        std::vector<std::vector<double>> result;
        for (int i = 1; i <= k && i < static_cast<int>(n); ++i) {
            size_t idx = indices[i];
            eigenvalues_out[i - 1] = eigenvalues[idx];
            
            // Copy eigenvector (column idx becomes row in result)
            std::vector<double> ev(n);
            for (size_t r = 0; r < n; ++r) {
                ev[r] = eigenvectors[r][idx];
            }
            result.push_back(std::move(ev));
        }
        
        return result;
#endif  // USE_EIGEN_SOLVER
    }
    
    // ==========================================================================
    // FOR LARGE MATRICES: Use Lanczos
    // ==========================================================================
    
    std::cerr << "[LANCZOS] Finding " << k << " smallest non-zero eigenvectors using Lanczos algorithm\n";
    
    // Configure Lanczos solver
    lanczos::LanczosConfig lanczos_config;
    // Request k+1 eigenpairs to skip the null space (constant eigenvector at λ=0)
    lanczos_config.num_eigenpairs = k + 1;
    lanczos_config.max_iterations = std::min(300, static_cast<int>(L.size()) / 2);
    lanczos_config.convergence_tol = config_.convergence_tol;
    lanczos_config.use_shift_invert = true;  // Crucial for finding smallest eigenvalues
    // Use POSITIVE shift just above 0: targets eigenvalues near 0
    // For shift-invert: θ = 1/(λ-σ), largest θ → smallest λ
    // σ = 1e-4 means we target eigenvalues just above 0 (skipping null space)
    lanczos_config.shift_sigma = 1e-4;
    lanczos_config.cg_max_iterations = 500;  // Dense MiniLM manifold needs more iterations
    lanczos_config.cg_tolerance = 1e-12;
    lanczos_config.num_threads = config_.num_threads;
    
    lanczos::LanczosSolver solver(lanczos_config);
    
    // Set up progress callback
    solver.set_progress_callback([this](const std::string& stage, int current, int total) {
        report_progress(stage, static_cast<size_t>(current), static_cast<size_t>(total));
    });
    
    // Run Lanczos
    lanczos::LanczosResult result = solver.solve(L);
    
    // Report convergence status
    std::cerr << "[LANCZOS] " << (result.converged ? "Converged" : "Did not fully converge")
              << " in " << result.iterations_used << " iterations\n";
    
    if (!result.converged) {
        std::cerr << "[LANCZOS] Warning: residuals = [";
        for (size_t i = 0; i < result.residuals.size() && i < 4; ++i) {
            std::cerr << result.residuals[i];
            if (i + 1 < result.residuals.size() && i < 3) std::cerr << ", ";
        }
        std::cerr << "]\n";
    }
    
    // Log ALL eigenvalues found (for debugging)
    std::cerr << "[LANCZOS] All eigenvalues (raw): ";
    for (size_t i = 0; i < result.eigenvalues.size(); ++i) {
        std::cerr << result.eigenvalues[i] << " ";
    }
    std::cerr << "\n";
    
    // Copy eigenvalues SKIPPING index 0 (null space)
    // eigenvalues_out[0..3] = result.eigenvalues[1..4]
    for (int i = 0; i < 4 && (i + 1) < static_cast<int>(result.eigenvalues.size()); ++i) {
        eigenvalues_out[i] = result.eigenvalues[i + 1];  // Skip index 0
    }
    
    std::cerr << "[LANCZOS] Semantic eigenvalues (skipped λ_0): ";
    for (int i = 0; i < 4; ++i) {
        std::cerr << eigenvalues_out[i] << " ";
    }
    std::cerr << "\n";
    
    // Return eigenvectors, SKIPPING index 0 (null space constant vector)
    // The Lanczos solver returns eigenvalues sorted smallest-to-largest
    // For Laplacian: λ_0 = 0 (constant), λ_1..λ_k are the semantic eigenvectors
    std::vector<std::vector<double>> eigenvectors;
    eigenvectors.reserve(k);
    
    // Skip first eigenvector (null space) and take the next k
    int start_idx = 1;  // Skip index 0
    for (int i = start_idx; i < start_idx + k && i < static_cast<int>(result.eigenvectors.size()); ++i) {
        eigenvectors.push_back(std::move(result.eigenvectors[i]));
    }
    
    std::cerr << "[LANCZOS] Returning " << eigenvectors.size() << " eigenvectors (skipped null space)\n";
    return eigenvectors;
}

void LaplacianProjector::gram_schmidt_columns(std::vector<std::vector<double>>& Y) {
    // Y is a list of column vectors, each of length n
    // We orthonormalize them in-place
    
    const int k = static_cast<int>(Y.size());
    if (k == 0) return;
    
    const size_t n = Y[0].size();
    
    std::cerr << "[GS] Gram-Schmidt orthonormalization on " << k << " columns of length " << n << "\n";
    
    for (int j = 0; j < k; ++j) {
        // Subtract projections onto previous columns
        for (int i = 0; i < j; ++i) {
            double proj = simd::dot_product_d(Y[j].data(), Y[i].data(), n);
            simd::subtract_scaled(Y[j].data(), Y[i].data(), proj, n);
        }
        
        // Normalize
        simd::normalize(Y[j].data(), n);
        
        report_progress("Gram-Schmidt", j + 1, k);
    }
    
    // Verify orthonormality
    double max_off_diag = 0.0;
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            double dot = simd::dot_product_d(Y[i].data(), Y[j].data(), n);
            max_off_diag = std::max(max_off_diag, std::abs(dot));
        }
    }
    std::cerr << "[GS] Max off-diagonal dot product: " << max_off_diag << "\n";
}

std::vector<std::array<uint32_t, 4>> LaplacianProjector::normalize_to_hypercube(
    const std::vector<std::vector<double>>& U
) {
    // U is k column vectors (k=4), each of length n
    // We need to transpose: row i becomes the 4D coordinate for token i
    
    if (U.empty() || U.size() < 4) {
        std::cerr << "[NORM] Error: need 4 eigenvectors, got " << U.size() << "\n";
        return {};
    }
    
    const size_t n = U[0].size();
    const int k = 4;
    
    // Find min/max per dimension
    std::array<double, 4> minv, maxv;
    for (int d = 0; d < k; ++d) {
        minv[d] = std::numeric_limits<double>::infinity();
        maxv[d] = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n; ++i) {
            double v = U[d][i];
            if (v < minv[d]) minv[d] = v;
            if (v > maxv[d]) maxv[d] = v;
        }
    }
    
    std::cerr << "[NORM] Coordinate ranges before normalization:\n";
    for (int d = 0; d < k; ++d) {
        std::cerr << "  dim " << d << ": [" << minv[d] << ", " << maxv[d] << "]\n";
    }
    
    const double EPS = 1e-12;
    const double M = 4294967295.0;  // 2^32 - 1
    
    std::vector<std::array<uint32_t, 4>> coords(n);
    
    // Parallelized coordinate normalization
    int num_threads = config_.num_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads == 0) num_threads = 4;
    }
    
    auto norm_worker = [&](int tid) {
        for (size_t i = tid; i < n; i += num_threads) {
            for (int d = 0; d < k; ++d) {
                double range = maxv[d] - minv[d];
                double x = (U[d][i] - minv[d]) / (range + EPS);  // Normalize to [0, 1]
                
                // Scale to uint32 range
                uint64_t val = static_cast<uint64_t>(std::llround(x * M));
                if (val > 0xFFFFFFFFULL) val = 0xFFFFFFFFULL;
                
                coords[i][d] = static_cast<uint32_t>(val);
            }
        }
    };
    
    std::vector<std::thread> norm_threads;
    for (int t = 0; t < num_threads; ++t) {
        norm_threads.emplace_back(norm_worker, t);
    }
    for (auto& t : norm_threads) t.join();
    
    std::cerr << "[NORM] Normalized " << n << " points to [0, 2^32-1]^4\n";
    return coords;
}

void LaplacianProjector::project_to_sphere(std::vector<std::array<uint32_t, 4>>& coords) {
    if (!config_.project_to_sphere) return;
    
    const size_t n = coords.size();
    if (n == 0) return;
    
    std::cerr << "[SPHERE] Projecting " << n << " points onto hypersphere (parallel)...\n";
    
    // COORDINATE CONVENTION: uint32 with CENTER at 2^31 = 2147483648
    // Unit sphere [-1, 1] maps to [1, 2^32-1] with CENTER at 2^31
    constexpr double CENTER = 2147483648.0;  // 2^31 - origin of hypercube
    constexpr double SCALE = 2147483647.0;   // radius to reach [1, 2^32-1]
    const double sphere_radius = config_.sphere_radius;
    
    // Parallelized sphere projection
    int num_threads = config_.num_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads == 0) num_threads = 4;
    }
    
    auto sphere_worker = [&](int tid) {
        for (size_t i = tid; i < n; i += num_threads) {
            // Convert to centered unit coordinates
            double x = (static_cast<double>(coords[i][0]) - CENTER) / SCALE;
            double y = (static_cast<double>(coords[i][1]) - CENTER) / SCALE;
            double z = (static_cast<double>(coords[i][2]) - CENTER) / SCALE;
            double m = (static_cast<double>(coords[i][3]) - CENTER) / SCALE;
            
            // Compute radius and normalize to sphere surface
            double r = std::sqrt(x*x + y*y + z*z + m*m);
            if (r > 1e-12) {
                double s = sphere_radius / r;
                x *= s;
                y *= s;
                z *= s;
                m *= s;
            }
            
            // Convert back to uint32 with CENTER at 2^31
            auto to_uint32 = [CENTER, SCALE](double v) -> uint32_t {
                double scaled = CENTER + v * SCALE;
                if (scaled < 0.0) scaled = 0.0;
                if (scaled > 4294967295.0) scaled = 4294967295.0;
                return static_cast<uint32_t>(std::round(scaled));
            };
            
            coords[i][0] = to_uint32(x);
            coords[i][1] = to_uint32(y);
            coords[i][2] = to_uint32(z);
            coords[i][3] = to_uint32(m);
        }
    };
    
    std::vector<std::thread> sphere_threads;
    for (int t = 0; t < num_threads; ++t) {
        sphere_threads.emplace_back(sphere_worker, t);
    }
    for (auto& t : sphere_threads) t.join();
    
    std::cerr << "[SPHERE] Projection complete\n";
}

ProjectionResult LaplacianProjector::project(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<std::string>& labels
) {
    ProjectionResult result;
    const size_t n = embeddings.size();
    
    if (n == 0) {
        std::cerr << "[PROJECT] No embeddings to project\n";
        return result;
    }
    
    std::cerr << "\n=== Laplacian Eigenmap Projection to 4D ===\n";
    std::cerr << "Tokens: " << n << ", Embedding dim: " << embeddings[0].size() << "\n\n";
    
    auto start = std::chrono::steady_clock::now();
    
    // Step 1: Build k-NN similarity graph
    std::cerr << "[1] Building k-NN similarity graph (k=" << config_.k_neighbors << ")...\n";
    auto W = build_similarity_graph(embeddings);
    result.edge_count = 0;
    W.for_each_edge([&](size_t, size_t, double) { ++result.edge_count; });
    result.edge_count /= 2;  // Each edge counted twice
    
    // Step 2: Build unnormalized Laplacian
    std::cerr << "\n[2] Building unnormalized Laplacian L = D - W...\n";
    auto L = build_laplacian(W);
    
    // Step 3: Find 4 smallest non-zero eigenvectors
    std::cerr << "\n[3] Finding 4 smallest non-zero eigenvectors...\n";
    auto eigenvectors = find_smallest_eigenvectors(L, 4, result.eigenvalues);
    
    if (eigenvectors.size() < 4) {
        std::cerr << "[ERROR] Could not find 4 eigenvectors\n";
        return result;
    }
    
    // Step 4: Gram-Schmidt orthonormalization on columns
    std::cerr << "\n[4] Gram-Schmidt orthonormalization...\n";
    gram_schmidt_columns(eigenvectors);
    
    // Step 5: Normalize to hypercube
    std::cerr << "\n[5] Normalizing to [0, 2^32-1]^4...\n";
    result.coords = normalize_to_hypercube(eigenvectors);
    
    // Step 6: Optional sphere projection
    if (config_.project_to_sphere) {
        std::cerr << "\n[6] Projecting to hypersphere...\n";
        project_to_sphere(result.coords);
    }
    
    // Step 7: Compute Hilbert indices (parallelized)
    std::cerr << "\n[7] Computing Hilbert indices (parallel)...\n";
    result.hilbert_lo.resize(n);
    result.hilbert_hi.resize(n);
    
    // Determine thread count
    int num_threads = config_.num_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads == 0) num_threads = 4;
    }
    
    std::atomic<size_t> hilbert_progress{0};
    auto hilbert_worker = [&](int tid) {
        for (size_t i = tid; i < n; i += num_threads) {
            Point4D pt(result.coords[i][0], result.coords[i][1], 
                       result.coords[i][2], result.coords[i][3]);
            HilbertIndex idx = HilbertCurve::coords_to_index(pt);
            result.hilbert_lo[i] = static_cast<int64_t>(idx.lo);
            result.hilbert_hi[i] = static_cast<int64_t>(idx.hi);
            hilbert_progress.fetch_add(1);
        }
    };
    
    std::vector<std::thread> hilbert_threads;
    for (int t = 0; t < num_threads; ++t) {
        hilbert_threads.emplace_back(hilbert_worker, t);
    }
    for (auto& t : hilbert_threads) t.join();
    
    auto end = std::chrono::steady_clock::now();
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cerr << "\n=== Projection Complete in " << secs << " seconds ===\n";
    std::cerr << "Projected " << n << " tokens to 4D hypercube\n";
    std::cerr << "Similarity graph edges: " << result.edge_count << "\n";
    
    return result;
}

} // namespace hypercube
