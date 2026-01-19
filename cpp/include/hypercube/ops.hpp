#pragma once
/**
 * Optimized Hypercube Operations
 * 
 * High-performance batch operations with:
 * - SIMD/AVX2 distance calculations
 * - Thread pool for parallel processing
 * - Connection pooling for database access
 * - Hilbert partitioning for locality
 * - In-memory graph algorithms
 */

#include "hypercube/types.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/cpu_features.hpp"
#include "hypercube/thread_config.hpp"

#include <vector>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>

namespace hypercube {
namespace ops {

// =============================================================================
// Atom Data Structures (in-memory representation)
// =============================================================================

struct AtomData {
    Blake3Hash id;
    std::vector<Blake3Hash> children;
    std::vector<uint8_t> value;  // For leaves: UTF-8 bytes
    int32_t depth = 0;
    int64_t atom_count = 1;
    int32_t codepoint = -1;      // For leaves: Unicode codepoint
    double centroid[4] = {0};    // X, Y, Z, M
    HilbertIndex hilbert;
    bool is_leaf = false;
};

struct SemanticEdge {
    Blake3Hash from;
    Blake3Hash to;
    double weight;
};

// =============================================================================
// SIMD-Optimized Distance Calculations
// =============================================================================

/**
 * Batch 4D Euclidean distance calculation with SIMD
 * Computes distances from one point to many points
 * Returns indices sorted by distance
 */
struct DistanceResult {
    size_t index;
    double distance;
};

#if defined(__AVX2__)
// AVX2: Process 4 distance calculations in parallel
void batch_distances_avx2(
    const double* target,           // 4 doubles: target centroid
    const double* points,           // N*4 doubles: point centroids (interleaved XYZM)
    size_t count,                   // Number of points
    double* distances_out           // N doubles: output distances
) noexcept;
#endif

// Portable fallback
void batch_distances_portable(
    const double* target,
    const double* points,
    size_t count,
    double* distances_out
) noexcept;

// Auto-select best implementation
inline void batch_distances(
    const double* target,
    const double* points,
    size_t count,
    double* distances_out
) noexcept {
    // Use runtime AVX2 detection for double precision operations
    if (hypercube::cpu_features::has_avx2()) {
#if defined(__AVX2__)
        batch_distances_avx2(target, points, count, distances_out);
        return;
#endif
    }

    // Fallback to portable implementation
    batch_distances_portable(target, points, count, distances_out);
}

/**
 * Find k-nearest neighbors from distance array
 * Uses partial sort for O(n + k log k) complexity
 */
std::vector<DistanceResult> find_knn(
    const double* distances,
    size_t count,
    size_t k
);

// =============================================================================
// Thread Pool for Parallel Operations
// =============================================================================

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);  // 0 = auto-detect from ThreadConfig
    explicit ThreadPool(WorkloadType workload_type);  // Workload-aware construction
    ~ThreadPool();
    
    // Submit work and get future
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>>;
    
    // Parallel for loop
    template<typename Iter, typename Func>
    void parallel_for(Iter begin, Iter end, Func&& func);
    
    // Parallel for with index
    void parallel_for_index(size_t start, size_t end, 
                           std::function<void(size_t)> func);
    
    size_t size() const { return workers_.size(); }
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
};

// Global thread pool singleton
ThreadPool& get_thread_pool();

// =============================================================================
// Atom Cache (In-Memory Graph)
// =============================================================================

class AtomCache {
public:
    // Load atoms from database
    void load_by_depth_range(int min_depth, int max_depth);
    void load_by_ids(std::span<const Blake3Hash> ids);
    void load_all_leaves();
    void load_semantic_edges(size_t limit = 100000);
    
    // Lookup
    const AtomData* get(const Blake3Hash& id) const;
    bool contains(const Blake3Hash& id) const;
    size_t size() const { return atoms_.size(); }
    
    // Graph operations (in-memory, no DB calls)
    std::string reconstruct_text(const Blake3Hash& root) const;
    std::vector<Blake3Hash> get_descendants(const Blake3Hash& root, int max_depth = 100) const;
    
    // Semantic graph
    std::vector<std::pair<Blake3Hash, double>> get_neighbors(const Blake3Hash& id) const;
    std::vector<Blake3Hash> shortest_path(const Blake3Hash& from, const Blake3Hash& to, 
                                          int max_depth = 6) const;
    std::vector<std::pair<Blake3Hash, double>> random_walk(const Blake3Hash& seed, 
                                                           int steps = 10) const;
    
    // Batch operations
    std::vector<std::pair<Blake3Hash, std::string>> batch_reconstruct(
        std::span<const Blake3Hash> ids) const;
    
    std::vector<DistanceResult> knn_by_centroid(
        const Blake3Hash& target, size_t k) const;
    
    std::vector<DistanceResult> knn_by_hilbert(
        const Blake3Hash& target, size_t k) const;
    
    // Clear cache
    void clear();

    // Iterators for partitioning and batch operations
    using iterator = std::unordered_map<Blake3Hash, AtomData, Blake3HashHasher>::iterator;
    using const_iterator = std::unordered_map<Blake3Hash, AtomData, Blake3HashHasher>::const_iterator;

    const_iterator begin() const { return atoms_.begin(); }
    const_iterator end() const { return atoms_.end(); }
    iterator begin() { return atoms_.begin(); }
    iterator end() { return atoms_.end(); }

    // Get all Hilbert indices sorted (for partitioning)
    std::vector<std::pair<HilbertIndex, Blake3Hash>> get_sorted_hilbert_indices() const;

private:
    std::unordered_map<Blake3Hash, AtomData, Blake3HashHasher> atoms_;
    std::unordered_map<Blake3Hash, std::vector<std::pair<Blake3Hash, double>>, 
                       Blake3HashHasher> edges_;
    
    // Cached centroid array for SIMD operations
    mutable std::vector<double> centroid_cache_;  // Interleaved XYZM
    mutable std::vector<Blake3Hash> id_cache_;    // Corresponding IDs
    mutable bool centroid_cache_valid_ = false;
    
    void rebuild_centroid_cache() const;
    
    // Allow analogy_knn to access cache
    friend std::vector<DistanceResult> analogy_knn(
        const Blake3Hash& a, const Blake3Hash& b, const Blake3Hash& c,
        size_t k, const AtomCache& cache);
};

// =============================================================================
// Hilbert Partitioning for Parallel Processing
// =============================================================================

struct HilbertPartition {
    HilbertIndex lo;
    HilbertIndex hi;
    size_t count;
};

/**
 * Partition atoms into ranges for parallel processing
 * Uses Hilbert curve for locality-preserving partitioning
 */
std::vector<HilbertPartition> partition_by_hilbert(
    const AtomCache& cache,
    size_t num_partitions
);

/**
 * Process partitions in parallel with work stealing
 */
template<typename Func>
void process_partitions(
    const std::vector<HilbertPartition>& partitions,
    Func&& func  // void(const HilbertPartition&)
);

// =============================================================================
// Batch Composition Operations
// =============================================================================

/**
 * Compute content hash for text (CPE cascade)
 * Uses SIMD-optimized BLAKE3
 */
Blake3Hash compute_content_hash(
    std::string_view text,
    const AtomCache& cache  // For leaf atom lookups
);

/**
 * Batch compute content hashes with threading
 */
std::vector<Blake3Hash> batch_content_hash(
    std::span<const std::string_view> texts,
    const AtomCache& cache
);

// =============================================================================
// Fréchet Distance (SIMD-optimized)
// =============================================================================

/**
 * Discrete Fréchet distance between two trajectories
 * Uses dynamic programming with SIMD for distance matrix
 */
double frechet_distance(
    std::span<const double> traj1,  // Interleaved XYZM
    std::span<const double> traj2
);

/**
 * Batch Fréchet: find k most similar trajectories
 */
struct FrechetResult {
    Blake3Hash id;
    double distance;
};

std::vector<FrechetResult> batch_frechet_knn(
    const Blake3Hash& query,
    size_t k,
    const AtomCache& cache
);

// =============================================================================
// Jaccard Similarity (Parallel)
// =============================================================================

double semantic_jaccard(
    const Blake3Hash& a,
    const Blake3Hash& b,
    const AtomCache& cache
);

std::vector<std::pair<Blake3Hash, double>> batch_jaccard(
    const Blake3Hash& target,
    std::span<const Blake3Hash> candidates,
    const AtomCache& cache
);

// =============================================================================
// Analogy Operations (Vector Arithmetic)
// =============================================================================

/**
 * A:B :: C:D  (find D given A, B, C)
 * Uses centroid arithmetic: D = C + B - A
 */
std::vector<DistanceResult> analogy_knn(
    const Blake3Hash& a,
    const Blake3Hash& b,
    const Blake3Hash& c,
    size_t k,
    const AtomCache& cache
);

} // namespace ops
} // namespace hypercube
