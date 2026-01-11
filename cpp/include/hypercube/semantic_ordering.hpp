#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace hypercube {

/**
 * Semantic Ordering System
 * =========================
 *
 * Maps Unicode codepoints to semantic ranks that preserve linguistic relationships.
 * Adjacent ranks correspond to semantically related characters (A/a/Ä grouped together).
 *
 * Key Features:
 * - Dense ranking: 0 to N-1 for all valid Unicode codepoints
 * - Semantic clustering: Related characters get adjacent ranks
 * - Thread-safe initialization
 * - Deterministic ordering
 */

class SemanticOrdering {
public:
    /**
     * Get the dense semantic rank for a codepoint (0 to N-1)
     * @param codepoint Unicode codepoint
     * @return Dense rank, or 0 for invalid codepoints
     */
    static uint32_t get_rank(uint32_t codepoint);

    /**
     * Get the codepoint for a given rank
     * @param rank Dense rank (0 to N-1)
     * @return Unicode codepoint, or 0 if rank is invalid
     */
    static uint32_t get_codepoint(uint32_t rank);

    /**
     * Get total number of valid Unicode codepoints
     * @return Total count of codepoints with assigned ranks
     */
    static uint32_t total_codepoints();

    /**
     * Check if a codepoint has a valid semantic rank
     * @param codepoint Unicode codepoint
     * @return True if codepoint is valid and ranked
     */
    static bool is_valid(uint32_t codepoint);

    /**
     * Get semantic key for advanced ordering (64-bit packed)
     * Used for multi-level sorting: script > category > base > variant
     * @param codepoint Unicode codepoint
     * @return 64-bit semantic key
     */
    static uint64_t get_semantic_key(uint32_t codepoint);

private:
    static std::unordered_map<uint32_t, uint32_t> codepoint_to_rank_;
    static std::vector<uint32_t> rank_to_codepoint_;
    static bool initialized_;
    static std::mutex init_mutex_;

    static void initialize();
    static uint64_t compute_semantic_key(uint32_t codepoint);
};

/**
 * Legacy function for backward compatibility
 * @deprecated Use SemanticOrdering::get_semantic_key() instead
 */
uint64_t get_semantic_key(uint32_t cp);

} // namespace hypercube</xai:function_call">The file cpp/include/hypercube/semantic_ordering.hpp was created successfully.