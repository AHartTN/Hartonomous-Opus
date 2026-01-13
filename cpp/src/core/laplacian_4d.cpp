/**
 * Laplacian Eigenmaps + Gram-Schmidt for 4D Hypercube Projection
 * 
 * Implementation of spectral dimensionality reduction for projecting
 * high-dimensional model embeddings into the 4D hypercube coordinate space.
 * 
 * Key differences from manifold_4d.cpp:
 * - Uses NORMALIZED Laplacian (L_sym = I - D^(-1/2) * W * D^(-1/2)) per spec
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
#include <unordered_map>
#include <unordered_set>
#include <functional>

// Intel MKL for optimized eigensolvers (highest priority)
#if defined(HAS_MKL) && HAS_MKL
#include <mkl.h>
// #include <mkl_solvers_ee.h>  // FEAST eigensolver disabled
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include <hnswlib/hnswlib.h>
#pragma GCC diagnostic pop
#define USE_HNSWLIB 1
#else
#define USE_HNSWLIB 0
#endif

namespace hypercube {

// Forward declaration
std::vector<std::vector<double>> solve_eigenvectors_cg(
    SparseSymmetricMatrix& L,
    int k,
    std::array<double, 4>& eigenvalues_out,
    const LaplacianConfig& config
);

// Use centralized SIMD implementations from embedding_ops.hpp
using embedding::cosine_similarity;
using embedding::l2_distance;

// NaN/inf guard functions
inline bool has_nan_or_inf(const double* v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(v[i]) || std::isinf(v[i])) {
            return true;
        }
    }
    return false;
}

inline void check_nan_inf(const double* v, size_t n, const char* context) {
    if (has_nan_or_inf(v, n)) {
        std::cerr << "[NAN_INF] Detected NaN or Inf in " << context << "\n";
    }
}

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
    : n_(n), diagonal_(n, 0.0), adj_(n), finalized_(false)
{}

SparseSymmetricMatrix::~SparseSymmetricMatrix() {
}

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
    if (!finalized_) {
        std::cerr << "[MATVEC] ERROR: matrix not finalized!\n";
        return;
    }

    // Use CPU implementation (MKL sparse has bugs)
    fallback_matvec(x, y);
}

// Fallback matrix-vector multiplication (original implementation)
void SparseSymmetricMatrix::fallback_matvec(const double* x, double* y) const {
    // NaN/inf guard on input
    check_nan_inf(x, n_, "fallback_matvec input x");

    // Parallel sparse matrix-vector multiply
    // Each row is independent, perfect for parallelization
    const int64_t n = static_cast<int64_t>(n_);
    const size_t* row_ptr = row_ptr_.data();
    const size_t* col_idx = col_idx_.data();
    const double* values = values_.data();
    const double* diag = diagonal_.data();

    #pragma omp parallel for schedule(static) if(n > 1000)
    for (int64_t i = 0; i < n; ++i) {
        double sum = diag[i] * x[i];
        for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            sum += values[k] * x[col_idx[k]];
        }
        y[i] = sum;
    }

    // NaN/inf guard on output
    check_nan_inf(y, n_, "fallback_matvec output y");
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


// Validate CSR structure (for debugging)
bool SparseSymmetricMatrix::validate_csr() const {
    if (!finalized_) return false;

    // Check row_ptr strictly increasing and within bounds
    if (row_ptr_.size() != n_ + 1) return false;
    if (row_ptr_[0] != 0) return false;
    for (size_t i = 1; i <= n_; ++i) {
        if (row_ptr_[i] < row_ptr_[i - 1]) return false;
        if (row_ptr_[i] > values_.size()) return false;
    }

    // Check col_idx in range and sorted per row
    for (size_t i = 0; i < n_; ++i) {
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            if (col_idx_[k] >= n_) return false;
            if (k > row_ptr_[i] && col_idx_[k] <= col_idx_[k - 1]) return false;
        }
    }

    // Check nnz consistency
    if (col_idx_.size() != values_.size()) return false;
    if (row_ptr_[n_] != values_.size()) return false;

    return true;
}

// Check if matrix is symmetric
bool SparseSymmetricMatrix::is_symmetric() const {
    if (!finalized_) return false;

    for (size_t i = 0; i < n_; ++i) {
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            size_t j = col_idx_[k];
            double w_ij = values_[k];

            // Find corresponding entry j->i
            bool found = false;
            for (size_t l = row_ptr_[j]; l < row_ptr_[j + 1]; ++l) {
                if (col_idx_[l] == i) {
                    if (std::abs(values_[l] - w_ij) > 1e-10) return false;
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    return true;
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
    
    // Normalize embeddings for cosine similarity via inner product
    std::vector<std::vector<float>> normalized(n);
    {
        // Parallel normalization
        auto norm_worker = [&](int tid) {
            for (size_t i = tid; i < n; i += num_threads) {
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

        std::vector<std::thread> norm_threads;
        for (int t = 0; t < num_threads; ++t) {
            norm_threads.emplace_back(norm_worker, t);
        }
        for (auto& t : norm_threads) t.join();
    }

    // Check for degenerate embeddings (all nearly identical after normalization)
    // This happens when embeddings collapse to a single point in high-D space
    {
        size_t samples_to_check = std::min(size_t(100), n);
        float max_diff = 0.0f;
        for (size_t i = 1; i < samples_to_check; ++i) {
            float diff = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float delta = normalized[i][d] - normalized[0][d];
                diff += delta * delta;
            }
            diff = std::sqrt(diff);
            if (diff > max_diff) max_diff = diff;
        }

        float degeneracy_threshold = 0.01f;  // Very small differences indicate collapse
        if (max_diff < degeneracy_threshold) {
            std::cerr << "[HNSWLIB] WARNING: Embeddings are degenerate (max_diff=" << max_diff
                      << "). Adding random noise to break symmetry.\n";

            // Add small random noise to break symmetry
            std::mt19937 rng(42);  // Fixed seed for reproducibility
            std::normal_distribution<float> noise(0.0f, 0.001f);  // Very small noise

            for (size_t i = 0; i < n; ++i) {
                for (size_t d = 0; d < dim; ++d) {
                    normalized[i][d] += noise(rng);
                }
                // Re-normalize after adding noise
                float norm_sq = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    norm_sq += normalized[i][d] * normalized[i][d];
                }
                float norm = std::sqrt(norm_sq);
                if (norm > 1e-10f) {
                    for (size_t d = 0; d < dim; ++d) {
                        normalized[i][d] /= norm;
                    }
                }
            }
        }
    }
    
    // =========================================================================
    // PARALLEL PARTITION STRATEGY for large vocabs (>500K)
    // Build multiple smaller indices in parallel, then query all in parallel
    // Trades slight accuracy for massive speedup (10-20x faster for 1M vocab)
    // =========================================================================
    constexpr size_t PARTITION_THRESHOLD = 500000;  // Increased to handle larger monolithic indices
    constexpr size_t PARTITION_SIZE = 200000;       // Larger partitions for efficiency

    std::vector<std::unique_ptr<hnswlib::HierarchicalNSW<float>>> indices;
    std::vector<std::pair<size_t, size_t>> partition_ranges;  // [start, end) for each partition

    if (n > PARTITION_THRESHOLD) {
        std::cerr << "[HNSWLIB] Using PARALLEL PARTITION strategy for " << n << " points\n";
        std::cerr << "[HNSWLIB] Building " << ((n + PARTITION_SIZE - 1) / PARTITION_SIZE) << " sub-indices in parallel...\n";

        // Create partitions
        for (size_t start = 0; start < n; start += PARTITION_SIZE) {
            size_t end = std::min(start + PARTITION_SIZE, n);
            partition_ranges.emplace_back(start, end);
        }

        indices.resize(partition_ranges.size());

        // Build indices in parallel
        std::vector<std::thread> build_threads;
        std::atomic<size_t> completed_partitions{0};

        for (size_t p = 0; p < partition_ranges.size(); ++p) {
            build_threads.emplace_back([&, p]() {
                auto [start, end] = partition_ranges[p];
                size_t partition_n = end - start;

                // Create index for this partition
                indices[p] = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                    &space, partition_n, M, ef_construction);

                // Add points from this partition
                for (size_t i = start; i < end; ++i) {
                    indices[p]->addPoint(normalized[i].data(), i - start);  // Local index
                }

                size_t done = ++completed_partitions;
                std::cerr << "[HNSWLIB]   Partition " << (p + 1) << "/" << partition_ranges.size()
                          << " complete (" << partition_n << " points)"
                          << " | Total: " << done << "/" << partition_ranges.size() << "\n";
            });
        }

        for (auto& t : build_threads) t.join();

        std::cerr << "[HNSWLIB] All partitions built, ready for k-NN queries\n";
    } else {
        // Standard sequential build for smaller vocabs
        std::cerr << "[HNSWLIB] Using SEQUENTIAL strategy for " << n << " points\n";
        std::cerr << "[HNSWLIB] Adding points to index...\n";

        // Create single index for all points
        indices.push_back(std::make_unique<hnswlib::HierarchicalNSW<float>>(
            &space, n, M, ef_construction));
        partition_ranges.emplace_back(0, n);

        size_t progress_interval = n / 20;  // Report every 5%
        if (progress_interval < 1000) progress_interval = 1000;

        auto add_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n; ++i) {
            indices[0]->addPoint(normalized[i].data(), i);

            // Progress reporting
            if ((i + 1) % progress_interval == 0 || i + 1 == n) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - add_start).count();
                double pct = 100.0 * (i + 1) / n;
                double rate = (i + 1) * 1000.0 / (elapsed_ms + 1);
                double eta_ms = (n - i - 1) * elapsed_ms / (i + 1);
                int eta_min = static_cast<int>(eta_ms / 60000);

                std::cerr << "[HNSWLIB]   Progress: " << (i + 1) << "/" << n
                          << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                          << " | " << std::setprecision(0) << rate << " pts/sec";
                if (i + 1 < n) {
                    std::cerr << " | ETA: " << eta_min << "m";
                }
                std::cerr << "\n";
            }
        }
    }
    
    auto build_end = std::chrono::steady_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - hnsw_start).count();
    std::cerr << "[HNSWLIB] Index built in " << build_ms << " ms\n";
    
    // Query k-NN for each point using work-stealing thread pool
    // For partitioned indices, query ALL partitions and merge results
    std::cerr << "[HNSWLIB] Querying k-NN across " << indices.size() << " partition(s)...\n";

    for (auto& idx : indices) {
        idx->setEf(std::max(static_cast<size_t>(k * 2), static_cast<size_t>(50)));
    }

    std::vector<std::vector<std::tuple<size_t, size_t, double>>> thread_edges(num_threads);
    std::atomic<size_t> progress{0};

    auto knn_worker = [&](int tid) {
        auto& local_edges = thread_edges[tid];

        for (size_t i = tid; i < n; i += num_threads) {
            // Collect candidates from ALL partitions
            std::vector<std::pair<float, size_t>> all_neighbors;

            for (size_t p = 0; p < indices.size(); ++p) {
                auto [start, end] = partition_ranges[p];

                // Check if query point is in this partition
                if (i >= start && i < end) {
                    // Query within partition (use local index)
        
                    auto result = indices[p]->searchKnn(normalized[i].data(), k + 1);

                    while (!result.empty()) {
                        auto [dist, local_j] = result.top();
                        result.pop();

                        size_t global_j = local_j + start;
                        if (global_j != i) {  // Skip self
                            float sim = 1.0f - dist;
                            all_neighbors.emplace_back(sim, global_j);
                        }
                    }
                } else {
                    // Query cross-partition (point i not in this partition)
                    // Search this partition for nearest neighbors to point i
                    auto result = indices[p]->searchKnn(normalized[i].data(), k);

                    while (!result.empty()) {
                        auto [dist, local_j] = result.top();
                        result.pop();

                        size_t global_j = local_j + start;
                        float sim = 1.0f - dist;
                        all_neighbors.emplace_back(sim, global_j);
                    }
                }
            }

            // Sort by similarity and take top-k
            std::sort(all_neighbors.begin(), all_neighbors.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

            // Add top-k edges (only if i < j to avoid duplicates)
            size_t added = 0;
            for (const auto& [sim, j] : all_neighbors) {
                if (added >= static_cast<size_t>(k)) break;
                if (sim > threshold && i < j) {
                    local_edges.emplace_back(i, j, static_cast<double>(sim));
                    ++added;
                }
            }

            progress.fetch_add(1);
        }
    };

    std::vector<std::thread> knn_threads;
    for (int t = 0; t < num_threads; ++t) {
        knn_threads.emplace_back(knn_worker, t);
    }

    // Progress reporting
    while (progress.load() < n) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        report_progress("k-NN queries", progress.load(), n);
    }

    for (auto& t : knn_threads) t.join();
    report_progress("k-NN queries", n, n);
    
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
    
    // NOTE: Do NOT finalize here - ensure_connectivity will add more edges first
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
    
    // NOTE: Do NOT finalize here - ensure_connectivity will add more edges first
    
    std::cerr << "[LAPLACIAN] Built k-NN graph with " << total_edges << " edges\n";
    return W;
#endif  // USE_HNSWLIB
}

// =============================================================================
// Connectivity Enforcement via Union-Find + Cross-Component Edges
// =============================================================================

/**
 * Ensures the similarity graph is connected by detecting components and
 * adding edges between them. Uses Union-Find for O(n α(n)) component detection.
 * 
 * OPTIMIZED: Parallel search, early termination, limited iterations.
 */
void LaplacianProjector::ensure_connectivity(
    SparseSymmetricMatrix& W,
    const std::vector<std::vector<float>>& embeddings
) {
    const size_t n = W.size();
    if (n <= 1) return;

    // Union-Find with path compression and union by rank
    std::vector<size_t> parent(n), rank_uf(n, 0);
    std::iota(parent.begin(), parent.end(), 0);
    
    std::function<size_t(size_t)> find = [&](size_t x) -> size_t {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    };
    
    auto unite = [&](size_t x, size_t y) -> bool {
        size_t rx = find(x), ry = find(y);
        if (rx == ry) return false;
        if (rank_uf[rx] < rank_uf[ry]) std::swap(rx, ry);
        parent[ry] = rx;
        if (rank_uf[rx] == rank_uf[ry]) rank_uf[rx]++;
        return true;
    };
    
    // Build components from existing edges
    W.for_each_edge([&](size_t i, size_t j, double) {
        unite(i, j);
    });
    
    // Count components
    std::unordered_set<size_t> roots;
    for (size_t i = 0; i < n; ++i) {
        roots.insert(find(i));
    }

    size_t num_components = roots.size();
    if (num_components == 1) {
        return;
    }

    std::cerr << "[CONNECT] Found " << num_components << " disconnected components - adding edges\n";
    
    const size_t dim = embeddings[0].size();
    size_t edges_added = 0;
    
    // Get thread count
    int num_threads = config_.num_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads == 0) num_threads = 4;
    }
    
    // Fast path: just connect components with random representatives
    // This is O(k) instead of O(k² * sample²) where k = num_components
    // The edges may not be optimal but they'll connect the graph
    
    // Build component representatives (just pick first member of each)
    std::vector<std::pair<size_t, size_t>> comp_reps;  // (root, representative)
    {
        std::unordered_map<size_t, size_t> first_member;
        for (size_t i = 0; i < n; ++i) {
            size_t root = find(i);
            if (first_member.find(root) == first_member.end()) {
                first_member[root] = i;
            }
        }
        for (auto& [root, rep] : first_member) {
            comp_reps.emplace_back(root, rep);
        }
    }
    
    // Connect all components to the first component's representative
    // This creates a star topology - not optimal but guaranteed connected in O(k)
    size_t main_rep = comp_reps[0].second;

    for (size_t c = 1; c < comp_reps.size(); ++c) {
        size_t other_rep = comp_reps[c].second;

        // Compute similarity for weight - use actual similarity for better conditioning
        float sim = embedding::cosine_similarity(
            embeddings[main_rep].data(),
            embeddings[other_rep].data(),
            dim);

        // Use high weight for connectivity (but not too high to avoid conditioning issues)
        double weight = std::max(0.1, static_cast<double>(sim));  // At least 0.1, preferably the actual similarity
        W.add_edge(main_rep, other_rep, weight);
        unite(main_rep, other_rep);
        ++edges_added;
    }

    // Verify connectivity after enforcement
    std::unordered_set<size_t> final_roots;
    for (size_t i = 0; i < n; ++i) {
        final_roots.insert(find(i));
    }
    size_t final_components = final_roots.size();

    if (final_components > 1) {
        std::cerr << "[CONNECT] WARNING: Still " << final_components << " components after connectivity enforcement!\n";
        std::cerr << "[CONNECT] This will cause zero eigenvalues. Laplacian eigenmaps requires connected graph.\n";
    } else {
        std::cerr << "[CONNECT] Graph is now connected (" << final_components << " component)\n";
    }

    std::cerr << "[CONNECT] Added " << edges_added << " inter-component edges\n";
}

SparseSymmetricMatrix LaplacianProjector::build_laplacian(const SparseSymmetricMatrix& W) {
    const size_t n = W.size();
    SparseSymmetricMatrix L(n);
    
    // 1. Compute Degrees
    std::vector<double> inv_sqrt_degree(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double deg = W.get_degree(i);
        if (deg > 1e-12) {
            inv_sqrt_degree[i] = 1.0 / std::sqrt(deg);
        }
    }

    // 2. Build Normalized Laplacian: L_sym = I - D^(-1/2) * W * D^(-1/2)
    // Diagonal is 1.0 (unless degree is 0)
    for (size_t i = 0; i < n; ++i) {
        if (W.get_degree(i) > 1e-12) {
            L.set_diagonal(i, 1.0);
        } else {
            L.set_diagonal(i, 0.0); // Disconnected node handling
        }
    }

    // Off-diagonals: -W_ij / sqrt(d_i * d_j)
    W.for_each_edge([&](size_t i, size_t j, double w) {
        double val = -w * inv_sqrt_degree[i] * inv_sqrt_degree[j];
        L.add_edge(i, j, val);
    });

    // Add ridge regularization to stabilize the solver (prevent singular matrices)
    // Use larger ridge for better numerical stability on large matrices
    double ridge = 1e-3;
    for (size_t i = 0; i < n; ++i) {
        L.set_diagonal(i, L.get_diagonal(i) + ridge);
    }

    L.finalize();
    return L;
}



std::vector<std::vector<double>> LaplacianProjector::find_smallest_eigenvectors(
    SparseSymmetricMatrix& L,
    int k,
    std::array<double, 4>& eigenvalues_out,
    bool& converged_out
) {
    const size_t n = L.size();

    // FEAST DISABLED: for Laplacians of this size (n≈30k, k=4)
    // classic sparse Lanczos is faster and simpler.
    // FEAST is designed for many eigenvalues in an interval, not a few near zero.

    // ==========================================================================
    // USE MKL DENSE EIGENDECOMPOSITION WHEN AVAILABLE (fastest and most accurate)
    // Lanczos fallback only if MKL not available
    // ==========================================================================

    // Convert sparse Laplacian to dense for MKL/Eigen solvers
    std::vector<double> L_dense(n * n, 0.0);
    L.for_each_edge([&](size_t i, size_t j, double w) {
        L_dense[i * n + j] = w;
    });
    // Add diagonals
    for (size_t i = 0; i < n; ++i) {
        L_dense[i * n + i] = L.get_diagonal(i);
    }

    #if USE_MKL_SOLVER
        std::cerr << "[EIGEN] Using MKL dense eigendecomposition for " << n << " points\n";
        converged_out = true;  // Dense eigendecomposition always succeeds
        
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
    }

#endif  // USE_EIGEN_SOLVER

    // ==========================================================================
    // FOR LARGE MATRICES: Use Lanczos eigensolver on -L to find largest eigenvalues
    // ==========================================================================

    // To find smallest eigenvalues of L, find largest eigenvalues of -L
    // Since L has eigenvalues 0 < λ1 ≤ λ2 ≤ ...
    // Then -L has eigenvalues 0 > -λ1 ≥ -λ2 ≥ ...
    // So largest eigenvalues of -L are 0, -λ1, -λ2, ...
    // Skip 0, then eigenvalues of L are λi = - (eigenvalue of -L)

    std::cerr << "[LANCZOS] Finding " << k << " smallest non-zero eigenvectors using Lanczos on -L\n";
    std::cerr << "[LANCZOS] config_.convergence_tol = " << config_.convergence_tol << "\n";

    // Create -L matrix
    SparseSymmetricMatrix neg_L(L.size());
    L.for_each_edge([&](size_t i, size_t j, double w) {
        neg_L.add_edge(i, j, -w);
    });
    for (size_t i = 0; i < L.size(); ++i) {
        neg_L.set_diagonal(i, -L.get_diagonal(i));
    }
    neg_L.finalize();

    // Configure Lanczos solver for -L
    lanczos::LanczosConfig lanczos_config;
    lanczos_config.num_eigenpairs = k + 1;  // k+1 to skip null space
    // Scale max_iterations with graph size: large graphs need more iterations
    // For small graphs (<10k): 500 iterations
    // For medium graphs (10k-50k): 1000 iterations
    // For large graphs (>50k): 2000 iterations
    size_t graph_size = L.size();
    if (graph_size > 50000) {
        lanczos_config.max_iterations = 2000;
    } else if (graph_size > 10000) {
        lanczos_config.max_iterations = 1000;
    } else {
        lanczos_config.max_iterations = 500;
    }
    lanczos_config.convergence_tol = 1e-8;
    lanczos_config.use_shift_invert = true;  // Use shift-invert Lanczos
    lanczos_config.shift_sigma = -0.1;
    lanczos_config.num_threads = config_.num_threads;

    lanczos::LanczosSolver lanczos_solver(lanczos_config);

    // Set up progress callback
    lanczos_solver.set_progress_callback([this](const std::string& stage, int current, int total) {
        report_progress(stage, static_cast<size_t>(current), static_cast<size_t>(total));
    });

    // Run Lanczos on -L
    lanczos::LanczosResult lanczos_result = lanczos_solver.solve(neg_L);

    // Report convergence status
    std::cerr << "[LANCZOS] " << (lanczos_result.converged ? "Converged" : "Did not fully converge")
              << " in " << lanczos_result.iterations_used << " iterations\n";
    converged_out = lanczos_result.converged;

    if (!lanczos_result.converged) {
        std::cerr << "[LANCZOS] Warning: residuals = [";
        for (size_t i = 0; i < lanczos_result.residuals.size() && i < 4; ++i) {
            std::cerr << lanczos_result.residuals[i];
            if (i + 1 < lanczos_result.residuals.size() && i < 3) std::cerr << ", ";
        }
        std::cerr << "]\n";
    }

    // Log ALL eigenvalues found (for debugging)
    std::cerr << "[LANCZOS] All eigenvalues (raw): ";
    for (size_t i = 0; i < lanczos_result.eigenvalues.size(); ++i) {
        std::cerr << lanczos_result.eigenvalues[i] << " ";
    }
    std::cerr << "\n";

    // Convert eigenvalues back to L eigenvalues and skip null space
    // lanczos_result.eigenvalues are for -L: [0, -λ1, -λ2, -λ3, -λ4, ...]
    // So L eigenvalues are: [0, λ1, λ2, λ3, λ4, ...]
    for (int i = 0; i < 4 && (i + 1) < static_cast<int>(lanczos_result.eigenvalues.size()); ++i) {
        eigenvalues_out[i] = -lanczos_result.eigenvalues[i + 1];  // Negate and skip index 0
    }

    std::cerr << "[LANCZOS] Semantic eigenvalues (skipped λ_0): ";
    for (int i = 0; i < 4; ++i) {
        std::cerr << eigenvalues_out[i] << " ";
    }
    std::cerr << "\n";

    // Return eigenvectors, SKIPPING index 0 (null space constant vector)
    // The Lanczos solver returns eigenvalues sorted smallest-to-largest
    // For Laplacian: λ_0 = 0 (constant), λ_1..λ_k are the semantic eigenvectors
    std::vector<std::vector<double>> lanczos_eigenvectors;
    lanczos_eigenvectors.reserve(k);

    // Skip first eigenvector (null space) and take the next k
    int start_idx = 1;  // Skip index 0
    for (int i = start_idx; i < start_idx + k && i < static_cast<int>(lanczos_result.eigenvectors.size()); ++i) {
        lanczos_eigenvectors.push_back(std::move(lanczos_result.eigenvectors[i]));
    }

    std::cerr << "[LANCZOS] Returning " << lanczos_eigenvectors.size() << " eigenvectors (skipped null space)\n";
    return lanczos_eigenvectors;
}

void LaplacianProjector::gram_schmidt_columns(std::vector<std::vector<double>>& Y) {
    const int k = static_cast<int>(Y.size());
    if (k == 0) return;
    
    const size_t n = Y[0].size();
    std::cerr << "[GS] Robust Double-Pass Gram-Schmidt on " << k << " columns\n";

    // Random generator for recovering collapsed dimensions
    std::mt19937 rng(1337);
    std::normal_distribution<double> dist(0.0, 1.0);

    // "Twice is enough" - Kahan/Parlett
    // We run the orthonormalization twice to correct round-off errors
    for (int pass = 0; pass < 2; ++pass) {
        for (int j = 0; j < k; ++j) {
            // Modified Gram-Schmidt: Project onto current j against all previous i
            for (int i = 0; i < j; ++i) {
                double dot = simd::dot_product_d(Y[j].data(), Y[i].data(), n);
                simd::subtract_scaled(Y[j].data(), Y[i].data(), dot, n);
            }

            // Normalize
            double nrm = simd::norm(Y[j].data(), n);

            // Handle collapsed dimensions (linear dependence)
            if (nrm < 1e-10) {
                if (pass == 0) {
                    std::cerr << "[GS] Warning: Vector " << j << " collapsed (norm=" << nrm 
                              << "). Regenerating random vector.\n";
                    // Regenerate as random vector
                    for (size_t x = 0; x < n; ++x) Y[j][x] = dist(rng);
                    
                    // Re-orthogonalize against previous immediately
                    for (int i = 0; i < j; ++i) {
                        double dot = simd::dot_product_d(Y[j].data(), Y[i].data(), n);
                        simd::subtract_scaled(Y[j].data(), Y[i].data(), dot, n);
                    }
                    nrm = simd::norm(Y[j].data(), n);
                }
            }

            // Safe normalize
            if (nrm > 1e-12) {
                simd::scale_inplace(Y[j].data(), 1.0 / nrm, n);
            } else {
                // Should not happen after regeneration logic, but as failsafe:
                std::fill(Y[j].begin(), Y[j].end(), 0.0);
            }
        }
    }

    // Verification
    double max_off_diag = 0.0;
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            double dot = std::abs(simd::dot_product_d(Y[i].data(), Y[j].data(), n));
            if (dot > max_off_diag) max_off_diag = dot;
        }
    }
    std::cerr << "[GS] Max off-diagonal dot product after correction: " << max_off_diag << "\n";
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
        minv[d] = 1e308;
        maxv[d] = -1e308;
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
            auto to_uint32 = [](double v) -> uint32_t {
                constexpr double CENTER = 2147483648.0;
                constexpr double SCALE = 2147483647.0;
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
    const std::vector<std::string>& labels,
    const std::vector<AnchorPoint>& anchors
) {
    ProjectionResult result;
    const size_t n = embeddings.size();

    if (n == 0) {
        std::cerr << "[PROJECT] No embeddings to project\n";
        return result;
    }

    std::cerr << "\n=== Laplacian Eigenmap Projection to 4D ===\n";
    std::cerr << "Tokens: " << n << ", Embedding dim: " << embeddings[0].size() << "\n";
    if (!anchors.empty()) {
        std::cerr << "Anchors: " << anchors.size() << " (aligning with existing 4D coordinates)\n";
    }
    std::cerr << "\n";

    auto start = std::chrono::steady_clock::now();
    
    // Step 1: Build k-NN similarity graph
    std::cerr << "[1] Building k-NN similarity graph (k=" << config_.k_neighbors << ")...\n";
    auto W = build_similarity_graph(embeddings);
    result.edge_count = 0;
    W.for_each_edge([&](size_t, size_t, double) { ++result.edge_count; });
    result.edge_count /= 2;  // Each edge counted twice
    
    // Step 1.5: Ensure graph connectivity (fix disconnected components)
    std::cerr << "\n[1.5] Ensuring graph connectivity...\n";
    ensure_connectivity(W, embeddings);

    // Step 1.6: Finalize the similarity graph (convert adj list to CSR)
    W.finalize();

    // DEBUG: Check for degree-0 vertices in similarity graph AFTER connectivity
    size_t degree_zero_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (W.get_degree(i) == 0.0) {
            ++degree_zero_count;
        }
    }
    if (degree_zero_count > 0) {
        std::cerr << "[ERROR] After connectivity+finalize: " << degree_zero_count
                  << " vertices STILL have degree 0 in similarity graph!\n";
        std::cerr << "[ERROR] This means connectivity enforcement FAILED!\n";
    }

    // Step 2: Build normalized Laplacian
    std::cerr << "\n[2] Building normalized Laplacian...\n";
    auto L = build_laplacian(W);
    
    // Step 3: Find 4 smallest non-zero eigenvectors
    std::cerr << "\n[3] Finding 4 smallest non-zero eigenvectors...\n";

    // DEBUG: Check Laplacian properties before eigensolver
    {
        size_t n = L.size();
        std::cerr << "[DEBUG] Laplacian matrix size: " << n << "x" << n << "\n";

        // Check degrees and Laplacian diagonal
        double min_degree = 1e308, max_degree = -1e308;
        double min_diag = 1e308, max_diag = -1e308;
        size_t zero_degree_count = 0;

        for (size_t i = 0; i < n; ++i) {
            double degree = W.get_degree(i);
            double diag = L.get_diagonal(i);

            if (degree < 1e-12) zero_degree_count++;
            min_degree = std::min(min_degree, degree);
            max_degree = std::max(max_degree, degree);
            min_diag = std::min(min_diag, diag);
            max_diag = std::max(max_diag, diag);
        }

        std::cerr << "[DEBUG] Degrees: min=" << min_degree << ", max=" << max_degree
                  << ", zero_count=" << zero_degree_count << "\n";
        std::cerr << "[DEBUG] Laplacian diagonal: min=" << min_diag << ", max=" << max_diag << "\n";

        // Check if Laplacian is symmetric
        bool is_sym = L.is_symmetric();
        std::cerr << "[DEBUG] Laplacian is symmetric: " << (is_sym ? "YES" : "NO") << "\n";

        // Check connectivity
        std::vector<bool> visited(n, false);
        std::queue<size_t> q;
        q.push(0);
        visited[0] = true;
        size_t visited_count = 1;

        while (!q.empty()) {
            q.pop();
            L.for_each_edge([&](size_t i, size_t j, double w) {
                if (!visited[j]) {
                    visited[j] = true;
                    visited_count++;
                    q.push(j);
                }
            });
        }

        std::cerr << "[DEBUG] Graph connectivity: " << visited_count << "/" << n
                  << " nodes reachable from node 0\n";
    }

    bool converged = false;
    auto eigenvectors = find_smallest_eigenvectors(L, 4, result.eigenvalues, converged);
    result.converged = converged && (eigenvectors.size() == 4);

    // Check if eigenvectors are degenerate (solver failure)
    bool degenerate = true;
    if (eigenvectors.size() >= 4) {
        // Check if all eigenvectors are identical (constant vector)
        for (size_t i = 1; i < eigenvectors.size(); ++i) {
            bool identical = true;
            for (size_t j = 0; j < eigenvectors[0].size(); ++j) {
                if (std::abs(eigenvectors[i][j] - eigenvectors[0][j]) > 1e-10) {
                    identical = false;
                    break;
                }
            }
            if (!identical) {
                degenerate = false;
                break;
            }
        }
    }

    if (degenerate || eigenvectors.size() < 4) {
        std::cerr << "[EIGEN] Lanczos failed (degenerate eigenvectors), trying CG solver\n";
        auto cg_eigenvectors = solve_eigenvectors_cg(L, 4, result.eigenvalues, config_);
        if (cg_eigenvectors.size() == 4) {
            eigenvectors = std::move(cg_eigenvectors);
            result.converged = true;
            std::cerr << "[EIGEN] CG solver succeeded\n";
        } else {
            std::cerr << "[ERROR] Both Lanczos and CG failed\n";
            result.converged = false;
            return result;
        }
    }
    
    // Step 4: Gram-Schmidt orthonormalization on columns
    std::cerr << "\n[4] Gram-Schmidt orthonormalization...\n";
    gram_schmidt_columns(eigenvectors);

    // Step 4.5: Apply Procrustes alignment if anchors provided
    if (!anchors.empty()) {
        std::cerr << "\n[4.5] Applying Procrustes alignment to match anchor coordinates...\n";

        // Extract anchor positions from eigenvectors (projected positions)
        std::vector<std::array<double, 4>> projected_anchors;
        std::vector<std::array<double, 4>> target_anchors;

        for (const auto& anchor : anchors) {
            if (anchor.token_index >= n) continue;

            // Projected position (from eigenvectors, transposed: row i = token i)
            std::array<double, 4> proj_pos;
            for (int d = 0; d < 4; ++d) {
                proj_pos[d] = eigenvectors[d][anchor.token_index];
            }
            projected_anchors.push_back(proj_pos);

            // Target position (known 4D coordinates from database)
            target_anchors.push_back(anchor.coords_4d);
        }

        if (projected_anchors.size() >= 3) {  // Need at least 3 points for meaningful alignment
            // Compute centroids
            std::array<double, 4> proj_centroid = {0, 0, 0, 0};
            std::array<double, 4> target_centroid = {0, 0, 0, 0};

            for (size_t i = 0; i < projected_anchors.size(); ++i) {
                for (int d = 0; d < 4; ++d) {
                    proj_centroid[d] += projected_anchors[i][d];
                    target_centroid[d] += target_anchors[i][d];
                }
            }

            for (int d = 0; d < 4; ++d) {
                proj_centroid[d] /= projected_anchors.size();
                target_centroid[d] /= target_anchors.size();
            }

            // Center both point sets
            for (auto& p : projected_anchors) {
                for (int d = 0; d < 4; ++d) p[d] -= proj_centroid[d];
            }
            for (auto& p : target_anchors) {
                for (int d = 0; d < 4; ++d) p[d] -= target_centroid[d];
            }

            // Compute scale factor (isotropic scaling)
            double proj_scale = 0, target_scale = 0;
            for (const auto& p : projected_anchors) {
                for (int d = 0; d < 4; ++d) proj_scale += p[d] * p[d];
            }
            for (const auto& p : target_anchors) {
                for (int d = 0; d < 4; ++d) target_scale += p[d] * p[d];
            }

            double scale = std::sqrt(target_scale / (proj_scale + 1e-12));

            std::cerr << "[PROCRUSTES] Scale factor: " << scale << "\n";
            std::cerr << "[PROCRUSTES] Proj centroid: [" << proj_centroid[0] << ", "
                      << proj_centroid[1] << ", " << proj_centroid[2] << ", "
                      << proj_centroid[3] << "]\n";
            std::cerr << "[PROCRUSTES] Target centroid: [" << target_centroid[0] << ", "
                      << target_centroid[1] << ", " << target_centroid[2] << ", "
                      << target_centroid[3] << "]\n";

            // Apply transformation to ALL eigenvectors: scale + translate
            // Transform: Y_new = scale * (Y - proj_centroid) + target_centroid
            for (size_t i = 0; i < n; ++i) {
                for (int d = 0; d < 4; ++d) {
                    eigenvectors[d][i] = scale * (eigenvectors[d][i] - proj_centroid[d]) + target_centroid[d];
                }
            }

            // Verify alignment quality
            double alignment_error = 0;
            for (const auto& anchor : anchors) {
                if (anchor.token_index >= n) continue;

                double error = 0;
                for (int d = 0; d < 4; ++d) {
                    double diff = eigenvectors[d][anchor.token_index] - anchor.coords_4d[d];
                    error += diff * diff;
                }
                alignment_error += std::sqrt(error);
            }
            alignment_error /= anchors.size();

            std::cerr << "[PROCRUSTES] Mean anchor alignment error: " << alignment_error << "\n";
        } else {
            std::cerr << "[PROCRUSTES] WARNING: Only " << projected_anchors.size()
                      << " valid anchors, skipping alignment (need >=3)\n";
        }
    }

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
    
    // Compute variance explained: ratio of eigenvalues used to theoretical maximum
    // For normalized Laplacian, eigenvalues range from 0 to 2
    // We use 4 eigenvectors, so theoretical maximum is 4 * 2 = 8
    double sum_eigenvalues = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum_eigenvalues += result.eigenvalues[i];
    }
    result.total_variance_explained = sum_eigenvalues / 8.0;  // Normalize to [0, 1]

    std::cerr << "\n=== Projection Complete in " << secs << " seconds ===\n";
    std::cerr << "Projected " << n << " tokens to 4D hypercube\n";
    std::cerr << "Similarity graph edges: " << result.edge_count << "\n";

    return result;
}

/**
 * @brief Conjugate Gradient solver for sparse symmetric positive definite systems
 * Solves Ax = b using the Conjugate Gradient method with SIMD acceleration
 */
class ConjugateGradientSolver {
private:
    const SparseSymmetricMatrix& A;
    const size_t n;
    const int max_iterations;
    const double tolerance;
    const int num_threads;

    // SIMD-accelerated operations
    void matvec(const double* x, double* y) const {
        A.matvec(x, y);
    }

    double dot_product(const double* a, const double* b) const {
        return simd::dot_product_d(a, b, n);
    }

    void scale_inplace(double* v, double s) const {
        simd::scale_inplace(v, s, n);
    }

    void add_scaled(double* a, const double* b, double s) const {
        for (size_t i = 0; i < n; ++i) {
            a[i] += s * b[i];
        }
    }

    void copy(const double* src, double* dst) const {
        std::memcpy(dst, src, n * sizeof(double));
    }

    double norm(const double* v) const {
        return std::sqrt(dot_product(v, v));
    }

public:
    ConjugateGradientSolver(const SparseSymmetricMatrix& matrix,
                           int max_iter = 1000,
                           double tol = 1e-8,
                           int threads = 0)
        : A(matrix), n(matrix.size()), max_iterations(max_iter),
          tolerance(tol), num_threads(threads > 0 ? threads :
              std::max(1, static_cast<int>(std::thread::hardware_concurrency()))) {}

    /**
     * @brief Solve Ax = b using Conjugate Gradient
     * @return true if converged within tolerance
     */
    bool solve(const double* b, double* x, double shift = 0.0) const {
        // Working vectors
        std::vector<double> r(n), p(n), Ap(n);

        // Initialize
        matvec(x, Ap.data());
        for (size_t i = 0; i < n; ++i) {
            r[i] = b[i] - (Ap[i] + shift * x[i]);  // (A + shift*I)x - b
        }

        double r_norm_sq = dot_product(r.data(), r.data());
        const double b_norm = norm(b);
        const double tol_sq = tolerance * tolerance * b_norm * b_norm;

        if (r_norm_sq < tol_sq) {
            return true;  // Already converged
        }

        copy(r.data(), p.data());

        for (int iter = 0; iter < max_iterations; ++iter) {
            // Ap = A*p + shift*p
            matvec(p.data(), Ap.data());
            add_scaled(Ap.data(), p.data(), shift);

            double alpha = r_norm_sq / dot_product(p.data(), Ap.data());

            // x = x + alpha*p
            add_scaled(x, p.data(), alpha);

            // r = r - alpha*Ap
            add_scaled(r.data(), Ap.data(), -alpha);

            double r_norm_sq_new = dot_product(r.data(), r.data());

            if (r_norm_sq_new < tol_sq) {
                return true;  // Converged
            }

            double beta = r_norm_sq_new / r_norm_sq;
            r_norm_sq = r_norm_sq_new;

            // p = r + beta*p
            scale_inplace(p.data(), beta);
            add_scaled(p.data(), r.data(), 1.0);
        }

        return false;  // Did not converge
    }
};

/**
 * @brief Solve for smallest eigenvectors using inverse iteration with CG
 *
 * For graph Laplacians, we use the method of solving (L + σI)v = random_vector
 * where σ is a small shift to avoid the null space. This gives us approximations
 * to the eigenvectors corresponding to the smallest eigenvalues.
 */
std::vector<std::vector<double>> solve_eigenvectors_cg(
    SparseSymmetricMatrix& L,
    int k,
    std::array<double, 4>& eigenvalues_out,
    const LaplacianConfig& config
) {
    const size_t n = L.size();
    const double shift = 1e-6;  // Small shift to avoid null space

    std::cerr << "[CG] Using inverse iteration with CG for " << k << " eigenvectors\n";

    ConjugateGradientSolver cg_solver(L, 500, 1e-10, config.num_threads);

    std::vector<std::vector<double>> eigenvectors;
    eigenvectors.reserve(k);

    // Random number generator for initial vectors
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<double> normal(0.0, 1.0);

    // For each eigenvector we want to find
    for (int i = 0; i < k; ++i) {
        // Create random initial vector
        std::vector<double> x(n);
        for (size_t j = 0; j < n; ++j) {
            x[j] = normal(rng);
        }

        // Orthogonalize against previously found eigenvectors
        for (const auto& prev_ev : eigenvectors) {
            double dot = simd::dot_product_d(x.data(), prev_ev.data(), n);
            for (size_t j = 0; j < n; ++j) {
                x[j] -= dot * prev_ev[j];
            }
        }

        // Normalize
        double norm_x = std::sqrt(simd::dot_product_d(x.data(), x.data(), n));
        if (norm_x > 1e-10) {
            for (auto& val : x) val /= norm_x;
        }

        // Solve (L + shift*I)v = x using CG
        // This gives us v ≈ eigenvector corresponding to eigenvalue closest to -shift
        std::vector<double> b = x;  // Right-hand side is our random vector
        bool converged = cg_solver.solve(b.data(), x.data(), shift);

        if (!converged) {
            std::cerr << "[CG] Failed to converge for eigenvector " << i << "\n";
            return {};  // Return empty to fall back to Lanczos
        }

        // Normalize the result
        norm_x = std::sqrt(simd::dot_product_d(x.data(), x.data(), n));
        if (norm_x > 1e-10) {
            for (auto& val : x) val /= norm_x;
        }

        // Rayleigh quotient to estimate eigenvalue: λ ≈ v^T L v / v^T v
        std::vector<double> Lx(n);
        L.matvec(x.data(), Lx.data());
        double rayleigh = simd::dot_product_d(x.data(), Lx.data(), n);

        // The shift gives us λ ≈ rayleigh - shift
        double eigenvalue = rayleigh;

        // Store eigenvalue
        if (i < 4) {
            eigenvalues_out[i] = eigenvalue;
        }

        eigenvectors.push_back(std::move(x));

        std::cerr << "[CG] Eigenvector " << i << " converged, λ ≈ " << eigenvalue << "\n";
    }

    std::cerr << "[CG] Found " << eigenvectors.size() << " eigenvectors\n";
    return eigenvectors;
}

} // namespace hypercube
