#pragma once

/**
 * HNSW (Hierarchical Navigable Small World) Engine for Approximate KNN
 *
 * Provides high-performance approximate nearest neighbor search for high-dimensional vectors
 * using the HNSW algorithm. Supports configurable parameters and integrates with existing
 * distance metrics and threading infrastructure.
 */

#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <unordered_map>

// HNSWLib disabled

#include "hypercube/thread_pool.hpp"
#include "hypercube/util/vector_ops.hpp"

namespace hypercube {
namespace knn {

/**
 * Distance metric types supported by HNSW
 */
enum class DistanceMetric {
    L2,      // Euclidean distance (default)
    IP,      // Inner product (cosine similarity)
    COSINE   // Cosine similarity
};

/**
 * Configuration parameters for HNSW index
 */
struct HNSWConfig {
    size_t dimensions = 128;           // Vector dimensionality
    size_t max_elements = 1000000;     // Maximum number of elements in index
    size_t M = 16;                     // Number of bi-directional links per element
    size_t ef_construction = 200;      // Size of dynamic candidate list during construction
    size_t ef_search = 64;             // Size of dynamic candidate list during search
    DistanceMetric metric = DistanceMetric::L2;

    // Threading configuration
    size_t num_threads = 0;            // 0 = use hardware concurrency
};

/**
 * Result of a KNN search operation
 */
struct KNNResult {
    size_t id;         // Index/ID of the neighbor
    float distance;    // Distance to the query vector

    KNNResult(size_t id_, float dist) : id(id_), distance(dist) {}
};

/**
 * HNSW-based approximate KNN search engine
 *
 * Provides efficient approximate nearest neighbor search for high-dimensional vectors.
 * Supports concurrent indexing and querying using the project's ThreadPool.
 */
class HNSWEngine {
public:
    /**
     * Constructor with configuration
     */
    explicit HNSWEngine(const HNSWConfig& config = HNSWConfig());

    /**
     * Destructor
     */
    ~HNSWEngine();

    // Non-copyable, non-movable
    HNSWEngine(const HNSWEngine&) = delete;
    HNSWEngine& operator=(const HNSWEngine&) = delete;
    HNSWEngine(HNSWEngine&&) = delete;
    HNSWEngine& operator=(HNSWEngine&&) = delete;

    /**
     * Initialize the index with given configuration
     * Must be called before adding vectors
     */
    void initialize(const HNSWConfig& config);

    /**
     * Add a single vector to the index
     * Thread-safe for concurrent additions
     */
    void add_vector(const std::vector<float>& vector, size_t id);

    /**
     * Add multiple vectors to the index in batch
     * Uses thread pool for parallel processing
     */
    void add_vectors(const std::vector<std::vector<float>>& vectors,
                    const std::vector<size_t>& ids);

    /**
     * Perform approximate KNN search
     * Returns k nearest neighbors sorted by distance
     */
    std::vector<KNNResult> search_knn(const std::vector<float>& query, size_t k) const;

    /**
     * Perform batch KNN search for multiple queries
     * Uses thread pool for parallel processing
     */
    std::vector<std::vector<KNNResult>> search_knn_batch(
        const std::vector<std::vector<float>>& queries, size_t k) const;

    /**
     * Get the current number of vectors in the index
     */
    size_t size() const;

    /**
     * Check if the index is empty
     */
    bool empty() const;

    /**
     * Get the dimensionality of vectors in the index
     */
    size_t dimensions() const;

    /**
     * Clear all vectors from the index
     */
    void clear();

    /**
     * Get current configuration
     */
    const HNSWConfig& config() const { return config_; }

    /**
     * Update search parameters at runtime
     */
    void set_ef_search(size_t ef_search);

    /**
     * Get current memory usage estimate (bytes)
     */
    size_t memory_usage() const;

private:
    /**
     * Create HNSWLIB space object based on distance metric
     */
    std::unique_ptr<hnswlib::SpaceInterface<float>> create_space(const HNSWConfig& config);

    /**
     * Convert distance metric enum to HNSWLIB space
     */
    hnswlib::SpaceInterface<float>* get_space() const;

    // Configuration
    HNSWConfig config_;

    // HNSWLIB components
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;

    // Thread pool for parallel operations
    std::shared_ptr<ThreadPool> thread_pool_;
};

} // namespace knn
} // namespace hypercube