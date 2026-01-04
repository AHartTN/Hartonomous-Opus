/**
 * PMI-Based Geometric Contraction Header
 * 
 * Uses Pointwise Mutual Information instead of raw frequency counting.
 * Uses geodesic midpoints on 4D hypersphere for composite positioning.
 */

#pragma once

#include "hypercube/ingest/cpe.hpp"  // For CompositionRecord, ChildInfo
#include "hypercube/blake3.hpp"

#include <string>
#include <vector>
#include <memory>

namespace hypercube::ingest {

/**
 * PMI-based ingester that contracts pairs based on information density,
 * not raw frequency. Uses geodesic midpoints for geometric positioning.
 */
class PMIIngester {
public:
    /**
     * @param num_threads Number of parallel threads (0 = auto)
     * @param pmi_threshold Minimum PMI score to merge (default 0.0)
     */
    explicit PMIIngester(size_t num_threads = 0, double pmi_threshold = 0.0);
    ~PMIIngester();
    
    /**
     * Ingest text using PMI-based contraction.
     * Returns root hashes for each chunk.
     */
    std::vector<Blake3Hash> ingest(
        const std::string& text,
        std::vector<CompositionRecord>& new_compositions
    );
    
    size_t composition_count() const;
    void clear();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hypercube::ingest
