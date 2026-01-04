/**
 * Universal Substrate Ingester
 * 
 * Language-agnostic, content-agnostic pattern discovery via sliding window.
 * Works identically on text, code, numbers, binary data - all are just
 * sequences of integer tokens (atoms).
 * 
 * "Hello world" = [H,e,l,l,o, ,w,o,r,l,d]
 * "public class" = [p,u,b,l,i,c, ,c,l,a,s,s]
 * "0.987" = [0,.,9,8,7]
 * 
 * No linguistic rules. No special cases. Pure pattern discovery.
 * 
 * The grammar trie builds organically through usage:
 * 1. Sliding window discovers patterns
 * 2. Common patterns get compositions  
 * 3. Next ingest matches known patterns, creates new for unknown
 * 4. Over time vocabulary grows, compression improves
 */

#pragma once

#include "hypercube/ingest/cpe.hpp"  // Re-use ChildInfo and CompositionRecord types
#include <memory>
#include <optional>

namespace hypercube::ingest {

// ============================================================================
// VOCABULARY TRIE
// ============================================================================

/**
 * Trie for O(n) longest-match lookups against existing compositions.
 * 
 * Stores compositions by their child hash sequence. Given a sequence of
 * atoms/compositions, finds the longest prefix that matches an existing
 * composition.
 * 
 * Grows organically as patterns are discovered and ingested.
 */
class VocabularyTrie {
public:
    VocabularyTrie();
    ~VocabularyTrie();
    
    // Insert a composition into the trie (keyed by child hashes)
    void insert(const std::vector<Blake3Hash>& children, const Blake3Hash& composition_hash);
    
    // Find longest matching composition for a sequence starting at position
    // Returns (composition_hash, match_length) or (nullopt, 0) if no match
    std::pair<std::optional<Blake3Hash>, size_t> longest_match(
        const std::vector<Blake3Hash>& sequence,
        size_t start_pos
    ) const;
    
    // Load existing compositions from database
    void load_from_db(const std::vector<CompositionRecord>& compositions);
    
    // Statistics
    size_t size() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// UNIVERSAL INGESTER
// ============================================================================

/**
 * Ingests any sequence of tokens (atoms) using pure sliding window pattern discovery.
 * 
 * Algorithm:
 * 1. Convert input tokens to their atom hashes
 * 2. Greedy longest-match against vocabulary trie
 * 3. For unmatched segments: create sliding window n-grams
 * 4. Return root composition hash
 * 
 * The vocabulary trie is updated with newly discovered patterns.
 */
class UniversalIngester {
public:
    explicit UniversalIngester(VocabularyTrie& vocab);
    ~UniversalIngester();
    
    /**
     * Ingest a sequence of tokens (e.g., Unicode codepoints).
     * 
     * @param tokens Sequence of token IDs (codepoints, byte values, etc.)
     * @param atom_cache Map from token ID to atom info (hash, coordinates)
     * @param new_compositions Output: newly created compositions
     * @return Root composition hash for the entire sequence
     */
    Blake3Hash ingest(
        const std::vector<uint32_t>& tokens,
        const std::unordered_map<uint32_t, db::AtomInfo>& atom_cache,
        std::vector<CompositionRecord>& new_compositions
    );
    
    /**
     * Ingest with explicit atom hashes (for hierarchical ingestion).
     * 
     * @param children Sequence of child infos (hash + coordinates)
     * @param new_compositions Output: newly created compositions
     * @return Root composition hash
     */
    Blake3Hash ingest_hashes(
        const std::vector<ChildInfo>& children,
        std::vector<CompositionRecord>& new_compositions
    );
    
    // Configuration
    void set_min_ngram(size_t n);
    void set_max_ngram(size_t n);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * Compute composition hash from ordered children.
 * hash = BLAKE3(ord_0 || hash_0 || ord_1 || hash_1 || ... || ord_N-1 || hash_N-1)
 */
Blake3Hash compute_composition_hash(const std::vector<Blake3Hash>& children);

} // namespace hypercube::ingest
