/**
 * Optimized Hypercube Operations Implementation
 * 
 * SIMD distance calculations, thread pool, and in-memory graph algorithms.
 */

#include "hypercube/ops.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/thread_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <future>
#include <stack>
#include <queue>
#include <unordered_set>
#include <iostream>

#include "hypercube/simd_intrinsics.hpp"

// SIMD intrinsics are included via the wrapper header
// AVX512 support is controlled at the compiler flag level

// CPU feature validation at runtime
static void validate_cpu_features() {
    static bool validated = false;
    if (validated) return;

    // Cross-platform CPUID detection
    [[maybe_unused]] uint32_t eax, ebx, ecx, edx;
#ifdef _MSC_VER
    // MSVC intrinsics for CPUID
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    eax = cpu_info[0];
    ebx = cpu_info[1];
    ecx = cpu_info[2];
    edx = cpu_info[3];
#else
    // GCC-style inline assembly
    __asm__ volatile(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(1), "c"(0)
    );
#endif
    bool cpuid_supported = (edx & (1 << 21)) != 0;

    if (!cpuid_supported) {
        std::cerr << "WARNING: CPUID not supported - cannot validate CPU features at runtime" << std::endl;
        validated = true;
        return;
    }

    // Check AVX512 support at runtime
#ifdef _MSC_VER
    __cpuid(cpu_info, 7);
    eax = cpu_info[0];
    ebx = cpu_info[1];
    ecx = cpu_info[2];
    edx = cpu_info[3];
#else
    __asm__ volatile(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );
#endif
    bool runtime_avx512f = (ebx & (1 << 16)) != 0;
    bool runtime_avx2 = (ebx & (1 << 5)) != 0;

#if defined(HAS_AVX512F) && HAS_AVX512F
    // Compile-time says AVX512F is supported
    if (!runtime_avx512f) {
        std::cerr << "ERROR: Compile-time AVX512F detection mismatch! "
                  << "Compiled with AVX512F support but CPU does not have it. "
                  << "This may cause runtime crashes or incorrect results." << std::endl;
    }
#else
    // Compile-time says AVX512F is NOT supported
    if (runtime_avx512f) {
        std::cerr << "INFO: CPU supports AVX512F but compiled without it. "
                  << "Consider recompiling for optimal performance." << std::endl;
    }
#endif

    // Also validate AVX2
#if defined(HAS_AVX2) && HAS_AVX2
    if (!runtime_avx2) {
        std::cerr << "ERROR: Compile-time AVX2 detection mismatch! "
                  << "Compiled with AVX2 support but CPU does not have it." << std::endl;
    }
#endif

    validated = true;
}

// Initialize validation on first use
struct CpuFeatureValidator {
    CpuFeatureValidator() { validate_cpu_features(); }
} g_validator;

namespace hypercube {
namespace ops {

// =============================================================================
// SIMD Distance Calculations
// =============================================================================

#if defined(__AVX2__)
void batch_distances_avx2(
    const double* target,
    const double* points,
    size_t count,
    double* distances_out
) noexcept {
    // Load target into AVX register (4 doubles)
    __m256d t = _mm256_loadu_pd(target);
    
    // Process 4 points at a time (each point has 4 coords)
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        // Load 4 points (16 doubles total, 4 per point)
        __m256d p0 = _mm256_loadu_pd(&points[(i + 0) * 4]);
        __m256d p1 = _mm256_loadu_pd(&points[(i + 1) * 4]);
        __m256d p2 = _mm256_loadu_pd(&points[(i + 2) * 4]);
        __m256d p3 = _mm256_loadu_pd(&points[(i + 3) * 4]);
        
        // Compute differences
        __m256d d0 = _mm256_sub_pd(p0, t);
        __m256d d1 = _mm256_sub_pd(p1, t);
        __m256d d2 = _mm256_sub_pd(p2, t);
        __m256d d3 = _mm256_sub_pd(p3, t);
        
        // Square
        d0 = _mm256_mul_pd(d0, d0);
        d1 = _mm256_mul_pd(d1, d1);
        d2 = _mm256_mul_pd(d2, d2);
        d3 = _mm256_mul_pd(d3, d3);
        
        // Horizontal sum (x + y + z + m for each point)
        // hadd: [a0+a1, b0+b1, a2+a3, b2+b3]
        __m256d s01 = _mm256_hadd_pd(d0, d1);  // [d0.x+d0.y, d1.x+d1.y, d0.z+d0.m, d1.z+d1.m]
        __m256d s23 = _mm256_hadd_pd(d2, d3);
        
        // Permute to get proper order and add remaining
        __m256d lo = _mm256_permute2f128_pd(s01, s23, 0x20);  // Lower halves
        __m256d hi = _mm256_permute2f128_pd(s01, s23, 0x31);  // Upper halves
        __m256d sums = _mm256_add_pd(lo, hi);
        
        // Square root
        __m256d result = _mm256_sqrt_pd(sums);
        
        // Store
        _mm256_storeu_pd(&distances_out[i], result);
    }
    
    // Handle remainder with portable code
    for (; i < count; ++i) {
        double sum = 0;
        for (int d = 0; d < 4; ++d) {
            double diff = points[i * 4 + d] - target[d];
            sum += diff * diff;
        }
        distances_out[i] = std::sqrt(sum);
    }
}
#endif

void batch_distances_portable(
    const double* target,
    const double* points,
    size_t count,
    double* distances_out
) noexcept {
    for (size_t i = 0; i < count; ++i) {
        double sum = 0;
        for (int d = 0; d < 4; ++d) {
            double diff = points[i * 4 + d] - target[d];
            sum += diff * diff;
        }
        distances_out[i] = std::sqrt(sum);
    }
}

std::vector<DistanceResult> find_knn(
    const double* distances,
    size_t count,
    size_t k
) {
    std::vector<DistanceResult> results;
    results.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        results.push_back({i, distances[i]});
    }
    
    // Partial sort for k smallest
    if (k < count) {
        std::partial_sort(results.begin(), results.begin() + k, results.end(),
            [](const DistanceResult& a, const DistanceResult& b) {
                return a.distance < b.distance;
            });
        results.resize(k);
    } else {
        std::sort(results.begin(), results.end(),
            [](const DistanceResult& a, const DistanceResult& b) {
                return a.distance < b.distance;
            });
    }
    
    return results;
}

// =============================================================================
// Thread Pool Implementation
// =============================================================================

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        // Use ThreadConfig for workload-appropriate thread allocation
        num_threads = ThreadConfig::instance().get_thread_count(WorkloadType::HYBRID);
    }

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                    if (stop_ && tasks_.empty()) return;

                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::ThreadPool(WorkloadType workload_type) {
    size_t num_threads = ThreadConfig::instance().get_thread_count(workload_type);

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                    if (stop_ && tasks_.empty()) return;

                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop_ = true;
    cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void ThreadPool::parallel_for_index(size_t start, size_t end, 
                                    std::function<void(size_t)> func) {
    if (start >= end) return;
    
    size_t count = end - start;
    size_t chunk_size = (count + workers_.size() - 1) / workers_.size();
    
    std::atomic<size_t> completed{0};
    size_t num_chunks = (count + chunk_size - 1) / chunk_size;
    
    for (size_t t = 0; t < num_chunks; ++t) {
        size_t chunk_start = start + t * chunk_size;
        size_t chunk_end = std::min(chunk_start + chunk_size, end);
        
        if (chunk_start >= end) break;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push([func, chunk_start, chunk_end, &completed] {
                for (size_t i = chunk_start; i < chunk_end; ++i) {
                    func(i);
                }
                completed.fetch_add(1, std::memory_order_release);
            });
        }
        cv_.notify_one();
    }
    
    // Wait for all chunks to complete
    while (completed.load(std::memory_order_acquire) < num_chunks) {
        std::this_thread::yield();
    }
}

// Global thread pool
static std::unique_ptr<ThreadPool> g_thread_pool;
static std::once_flag g_pool_init_flag;

ThreadPool& get_thread_pool() {
    std::call_once(g_pool_init_flag, [] {
        g_thread_pool = std::make_unique<ThreadPool>();
    });
    return *g_thread_pool;
}

// =============================================================================
// Atom Cache Implementation
// =============================================================================

const AtomData* AtomCache::get(const Blake3Hash& id) const {
    auto it = atoms_.find(id);
    return it != atoms_.end() ? &it->second : nullptr;
}

bool AtomCache::contains(const Blake3Hash& id) const {
    return atoms_.find(id) != atoms_.end();
}

void AtomCache::clear() {
    atoms_.clear();
    edges_.clear();
    centroid_cache_.clear();
    id_cache_.clear();
    centroid_cache_valid_ = false;
}

std::vector<std::pair<HilbertIndex, Blake3Hash>> AtomCache::get_sorted_hilbert_indices() const {
    std::vector<std::pair<HilbertIndex, Blake3Hash>> indices;
    indices.reserve(atoms_.size());

    for (const auto& [id, atom] : atoms_) {
        indices.emplace_back(atom.hilbert, id);
    }

    // Sort by Hilbert index for locality-preserving partitioning
    std::sort(indices.begin(), indices.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    return indices;
}

void AtomCache::rebuild_centroid_cache() const {
    if (centroid_cache_valid_) return;
    
    centroid_cache_.clear();
    id_cache_.clear();
    centroid_cache_.reserve(atoms_.size() * 4);
    id_cache_.reserve(atoms_.size());
    
    for (const auto& [id, atom] : atoms_) {
        id_cache_.push_back(id);
        centroid_cache_.push_back(atom.centroid[0]);
        centroid_cache_.push_back(atom.centroid[1]);
        centroid_cache_.push_back(atom.centroid[2]);
        centroid_cache_.push_back(atom.centroid[3]);
    }
    
    centroid_cache_valid_ = true;
}

std::string AtomCache::reconstruct_text(const Blake3Hash& root) const {
    std::string result;
    result.reserve(4096);
    
    struct StackEntry {
        Blake3Hash id;
        size_t child_idx;
    };
    
    std::vector<StackEntry> stack;
    stack.push_back({root, 0});
    
    while (!stack.empty()) {
        auto& current = stack.back();
        
        const AtomData* atom = get(current.id);
        if (!atom) {
            stack.pop_back();
            continue;
        }
        
        if (atom->is_leaf) {
            result.append(reinterpret_cast<const char*>(atom->value.data()), 
                         atom->value.size());
            stack.pop_back();
            continue;
        }
        
        if (current.child_idx < atom->children.size()) {
            Blake3Hash child_id = atom->children[current.child_idx];
            current.child_idx++;
            stack.push_back({child_id, 0});
        } else {
            stack.pop_back();
        }
    }
    
    return result;
}

std::vector<Blake3Hash> AtomCache::get_descendants(const Blake3Hash& root, 
                                                    int max_depth) const {
    std::vector<Blake3Hash> result;
    
    struct StackEntry {
        Blake3Hash id;
        int depth;
    };
    
    std::vector<StackEntry> stack;
    stack.push_back({root, 0});
    
    while (!stack.empty()) {
        auto [id, depth] = stack.back();
        stack.pop_back();
        
        result.push_back(id);
        
        if (depth >= max_depth) continue;
        
        const AtomData* atom = get(id);
        if (!atom || atom->is_leaf) continue;
        
        for (const auto& child : atom->children) {
            stack.push_back({child, depth + 1});
        }
    }
    
    return result;
}

std::vector<std::pair<Blake3Hash, double>> AtomCache::get_neighbors(
    const Blake3Hash& id
) const {
    auto it = edges_.find(id);
    if (it == edges_.end()) return {};
    return it->second;
}

std::vector<Blake3Hash> AtomCache::shortest_path(
    const Blake3Hash& from,
    const Blake3Hash& to,
    int max_depth
) const {
    std::unordered_map<Blake3Hash, Blake3Hash, Blake3HashHasher> parent;
    std::queue<std::pair<Blake3Hash, int>> queue;
    
    queue.push({from, 0});
    parent[from] = from;
    
    bool found = false;
    while (!queue.empty() && !found) {
        auto [current, depth] = queue.front();
        queue.pop();
        
        if (depth >= max_depth) continue;
        
        auto neighbors = get_neighbors(current);
        for (const auto& [neighbor, weight] : neighbors) {
            if (parent.find(neighbor) != parent.end()) continue;
            
            parent[neighbor] = current;
            
            if (neighbor == to) {
                found = true;
                break;
            }
            
            queue.push({neighbor, depth + 1});
        }
    }
    
    if (!found) return {};
    
    std::vector<Blake3Hash> path;
    Blake3Hash current = to;
    while (!(current == from)) {
        path.push_back(current);
        current = parent[current];
    }
    path.push_back(from);
    std::reverse(path.begin(), path.end());
    
    return path;
}

std::vector<std::pair<Blake3Hash, double>> AtomCache::random_walk(
    const Blake3Hash& seed,
    int steps
) const {
    std::vector<std::pair<Blake3Hash, double>> path;
    path.reserve(steps + 1);
    
    // Deterministic seed derived from input hash for reproducibility
    // XOR all 8 32-bit chunks of the 32-byte hash, mix with steps
    uint32_t rng_seed = 0;
    for (size_t i = 0; i < 32; i += 4) {
        uint32_t chunk = 0;
        for (size_t j = 0; j < 4; ++j) {
            chunk |= static_cast<uint32_t>(seed.bytes[i + j]) << (j * 8);
        }
        rng_seed ^= chunk;
    }
    rng_seed ^= static_cast<uint32_t>(steps) * 2654435761u;  // golden ratio mixing
    std::mt19937 rng(rng_seed);
    
    Blake3Hash current = seed;
    std::unordered_set<Blake3Hash, Blake3HashHasher> visited;
    
    for (int i = 0; i <= steps; ++i) {
        visited.insert(current);
        
        auto neighbors = get_neighbors(current);
        
        // Filter to unvisited
        std::vector<std::pair<Blake3Hash, double>> available;
        for (const auto& [n, w] : neighbors) {
            if (visited.find(n) == visited.end()) {
                available.push_back({n, w});
            }
        }
        
        if (available.empty()) {
            path.push_back({current, 0.0});
            break;
        }
        
        // Weighted random selection
        double total_weight = 0;
        for (const auto& [n, w] : available) {
            total_weight += w;
        }
        
        std::uniform_real_distribution<double> dist(0, total_weight);
        double r = dist(rng);
        
        double cumulative = 0;
        Blake3Hash next = available[0].first;
        double edge_weight = available[0].second;
        
        for (const auto& [n, w] : available) {
            cumulative += w;
            if (r <= cumulative) {
                next = n;
                edge_weight = w;
                break;
            }
        }
        
        path.push_back({current, edge_weight});
        current = next;
    }
    
    return path;
}

std::vector<std::pair<Blake3Hash, std::string>> AtomCache::batch_reconstruct(
    std::span<const Blake3Hash> ids
) const {
    std::vector<std::pair<Blake3Hash, std::string>> results;
    results.resize(ids.size());
    
    // Parallel reconstruction
    get_thread_pool().parallel_for_index(0, ids.size(), [&](size_t i) {
        results[i] = {ids[i], reconstruct_text(ids[i])};
    });
    
    return results;
}

std::vector<DistanceResult> AtomCache::knn_by_centroid(
    const Blake3Hash& target,
    size_t k
) const {
    const AtomData* target_atom = get(target);
    if (!target_atom) return {};
    
    rebuild_centroid_cache();
    
    if (centroid_cache_.empty()) return {};
    
    // SIMD batch distance calculation
    std::vector<double> distances(id_cache_.size());
    batch_distances(target_atom->centroid, centroid_cache_.data(), 
                   id_cache_.size(), distances.data());
    
    // Find k-nearest
    auto knn = find_knn(distances.data(), distances.size(), k + 1);  // +1 to exclude self
    
    // Filter out self and map indices to IDs
    std::vector<DistanceResult> results;
    results.reserve(k);
    
    for (const auto& r : knn) {
        if (id_cache_[r.index] == target) continue;
        results.push_back({r.index, r.distance});
        if (results.size() >= k) break;
    }
    
    return results;
}

std::vector<DistanceResult> AtomCache::knn_by_hilbert(
    const Blake3Hash& target,
    size_t k
) const {
    const AtomData* target_atom = get(target);
    if (!target_atom) return {};
    
    // Compute Hilbert distances and sort
    std::vector<std::pair<Blake3Hash, HilbertIndex>> with_dist;
    with_dist.reserve(atoms_.size());
    
    for (const auto& [id, atom] : atoms_) {
        if (id == target) continue;
        HilbertIndex dist = HilbertCurve::distance(target_atom->hilbert, atom.hilbert);
        with_dist.push_back({id, dist});
    }
    
    // Partial sort by Hilbert distance
    if (k < with_dist.size()) {
        std::partial_sort(with_dist.begin(), with_dist.begin() + k, with_dist.end(),
            [](const auto& a, const auto& b) {
                // Compare 128-bit Hilbert distance
                if (a.second.hi != b.second.hi) return a.second.hi < b.second.hi;
                return a.second.lo < b.second.lo;
            });
        with_dist.resize(k);
    }
    
    std::vector<DistanceResult> results;
    results.reserve(k);
    
    for (size_t i = 0; i < with_dist.size(); ++i) {
        // Convert Hilbert distance to double (approximate)
        double dist = static_cast<double>(with_dist[i].second.lo) + 
                     static_cast<double>(with_dist[i].second.hi) * 1e19;
        results.push_back({i, dist});
    }
    
    return results;
}

// =============================================================================
// Hilbert Partitioning
// =============================================================================

std::vector<HilbertPartition> partition_by_hilbert(
    const AtomCache& cache,
    size_t num_partitions
) {
    std::vector<HilbertPartition> result;

    if (cache.size() == 0 || num_partitions == 0) {
        return result;
    }

    // Get sorted Hilbert indices
    auto indices = cache.get_sorted_hilbert_indices();

    if (indices.empty()) {
        return result;
    }

    // Handle case where we have fewer atoms than partitions
    if (indices.size() <= num_partitions) {
        // One partition per atom (or fewer partitions than requested)
        for (size_t i = 0; i < indices.size(); ++i) {
            result.push_back({indices[i].first, indices[i].first, 1});
        }
        return result;
    }

    // Divide atoms into roughly equal partitions
    size_t atoms_per_partition = indices.size() / num_partitions;
    size_t remainder = indices.size() % num_partitions;

    size_t start_idx = 0;
    for (size_t p = 0; p < num_partitions; ++p) {
        // Distribute remainder among first partitions
        size_t partition_size = atoms_per_partition + (p < remainder ? 1 : 0);
        size_t end_idx = start_idx + partition_size - 1;

        if (end_idx >= indices.size()) {
            end_idx = indices.size() - 1;
        }

        result.push_back({
            indices[start_idx].first,  // lo: first Hilbert index in partition
            indices[end_idx].first,    // hi: last Hilbert index in partition
            partition_size
        });

        start_idx = end_idx + 1;
        if (start_idx >= indices.size()) break;
    }

    return result;
}

// =============================================================================
// Fréchet Distance (SIMD-optimized DP)
// =============================================================================

double frechet_distance(
    std::span<const double> traj1,
    std::span<const double> traj2
) {
    if (traj1.empty() || traj2.empty()) return 0.0;
    
    size_t n = traj1.size() / 4;
    size_t m = traj2.size() / 4;
    
    // DP table
    std::vector<double> dp(n * m, std::numeric_limits<double>::max());
    
    // Base case
    auto point_dist = [&](size_t i, size_t j) {
        double sum = 0;
        for (int d = 0; d < 4; ++d) {
            double diff = traj1[i * 4 + d] - traj2[j * 4 + d];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    };
    
    dp[0] = point_dist(0, 0);
    
    // First row
    for (size_t j = 1; j < m; ++j) {
        dp[j] = std::max(dp[j - 1], point_dist(0, j));
    }
    
    // First column
    for (size_t i = 1; i < n; ++i) {
        dp[i * m] = std::max(dp[(i - 1) * m], point_dist(i, 0));
    }
    
    // Fill rest
    for (size_t i = 1; i < n; ++i) {
        for (size_t j = 1; j < m; ++j) {
            double d = point_dist(i, j);
            double prev = std::min({
                dp[(i - 1) * m + j],
                dp[i * m + j - 1],
                dp[(i - 1) * m + j - 1]
            });
            dp[i * m + j] = std::max(d, prev);
        }
    }
    
    return dp[n * m - 1];
}

// =============================================================================
// Jaccard Similarity
// =============================================================================

double semantic_jaccard(
    const Blake3Hash& a,
    const Blake3Hash& b,
    const AtomCache& cache
) {
    auto neighbors_a = cache.get_neighbors(a);
    auto neighbors_b = cache.get_neighbors(b);
    
    if (neighbors_a.empty() || neighbors_b.empty()) return 0.0;
    
    std::unordered_set<Blake3Hash, Blake3HashHasher> set_a;
    for (const auto& [n, w] : neighbors_a) {
        set_a.insert(n);
    }
    
    size_t intersection = 0;
    for (const auto& [n, w] : neighbors_b) {
        if (set_a.count(n)) ++intersection;
    }
    
    size_t union_size = set_a.size() + neighbors_b.size() - intersection;
    
    return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
}

std::vector<std::pair<Blake3Hash, double>> batch_jaccard(
    const Blake3Hash& target,
    std::span<const Blake3Hash> candidates,
    const AtomCache& cache
) {
    std::vector<std::pair<Blake3Hash, double>> results;
    results.resize(candidates.size());
    
    get_thread_pool().parallel_for_index(0, candidates.size(), [&](size_t i) {
        results[i] = {candidates[i], semantic_jaccard(target, candidates[i], cache)};
    });
    
    return results;
}

// =============================================================================
// Analogy Operations
// =============================================================================

std::vector<DistanceResult> analogy_knn(
    const Blake3Hash& a,
    const Blake3Hash& b,
    const Blake3Hash& c,
    size_t k,
    const AtomCache& cache
) {
    const AtomData* atom_a = cache.get(a);
    const AtomData* atom_b = cache.get(b);
    const AtomData* atom_c = cache.get(c);
    
    if (!atom_a || !atom_b || !atom_c) return {};
    
    // Compute target: D = C + B - A
    double target[4];
    for (int d = 0; d < 4; ++d) {
        target[d] = atom_c->centroid[d] + atom_b->centroid[d] - atom_a->centroid[d];
    }
    
    // Build cache and compute distances
    cache.rebuild_centroid_cache();
    
    std::vector<double> distances(cache.size());
    batch_distances(target, cache.centroid_cache_.data(), cache.size(), distances.data());
    
    return find_knn(distances.data(), distances.size(), k);
}

} // namespace ops
} // namespace hypercube
