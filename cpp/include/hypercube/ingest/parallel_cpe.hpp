/**
 * Parallel CPE (Cascading Pair Encoding) Ingester
 * 
 * ZERO DATABASE ROUNDTRIPS until final insert.
 * Everything computed client-side. Fully parallelized.
 */

#pragma once

#include "hypercube/ingest/cpe.hpp"  // CompositionRecord
#include <memory>
#include <vector>
#include <string>

namespace hypercube::ingest {

/**
 * Parallel CPE Ingester
 * 
 * Splits text into chunks, processes in parallel using thread pool,
 * merges results, deduplicates compositions.
 * 
 * All computation uses AtomCalculator - ZERO database calls until
 * the final batch insert.
 */
class ParallelCPEIngester {
public:
    /**
     * @param num_threads Number of worker threads (0 = auto-detect)
     */
    explicit ParallelCPEIngester(size_t num_threads = 0);
    ~ParallelCPEIngester();
    
    /**
     * Ingest text and discover patterns via CPE.
     * 
     * @param text UTF-8 encoded text
     * @param new_compositions Output: all discovered compositions
     * @return Root hash for each chunk
     */
    std::vector<Blake3Hash> ingest(
        const std::string& text,
        std::vector<CompositionRecord>& new_compositions
    );
    
    /**
     * Get count of discovered compositions
     */
    size_t composition_count() const;
    
    /**
     * Clear internal state
     */
    void clear();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hypercube::ingest
