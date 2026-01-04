/**
 * Sequitur Algorithm for Hypercube
 * 
 * Implements the Nevill-Manning & Witten (1997) algorithm for
 * grammar-based compression with content-addressing.
 * 
 * Constraints:
 * - Digram uniqueness: No digram appears more than once in the grammar
 * - Rule utility: Every rule is used more than once
 * 
 * This produces compositions that are:
 * - Lossless (grammar reconstructs exact input)
 * - Maximally deduplicated (no redundant compositions)
 * - Content-addressed (BLAKE3 hash = identity)
 */

#pragma once

#include "hypercube/ingest/cpe.hpp"  // CompositionRecord, ChildInfo
#include "hypercube/types.hpp"
#include <memory>
#include <vector>
#include <string>

namespace hypercube::ingest {

/**
 * Sequitur-based Ingester
 * 
 * Processes text using the Sequitur algorithm to discover
 * repeated patterns and create compositions.
 * 
 * Key properties:
 * - Every composition appears 2+ times in the corpus
 * - No digram is duplicated within the grammar
 * - Lossless: original text can be reconstructed from root hash
 */
class SequiturIngester {
public:
    explicit SequiturIngester(size_t num_threads = 0);
    ~SequiturIngester();
    
    /**
     * Ingest text using Sequitur algorithm.
     * 
     * @param text UTF-8 encoded text
     * @param new_compositions Output: discovered compositions
     * @return Root hashes for each chunk
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
