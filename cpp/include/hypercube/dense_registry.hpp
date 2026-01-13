#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace hypercube {

/**
 * Dense registry for mapping codepoints to dense ranks
 * Provides efficient bidirectional mapping between Unicode codepoints and dense sequential ranks
 */
class DenseRegistry {
private:
    static std::unordered_map<uint32_t, uint32_t> codepoint_to_rank;
    static std::vector<uint32_t> rank_to_codepoint;
    static bool initialized;
    static std::mutex init_mutex;

    static void initialize();

public:
    /**
     * Get the dense rank for a codepoint
     * @param cp Unicode codepoint
     * @return Dense rank (0-based sequential index)
     */
    static uint32_t get_rank(uint32_t cp);

    /**
     * Get total number of active codepoints
     * @return Total count of codepoints in the registry
     */
    static uint32_t total_active();

    /**
     * Get codepoint from dense rank
     * @param rank Dense rank (0-based)
     * @return Unicode codepoint
     */
    static uint32_t get_codepoint(uint32_t rank);
};

} // namespace hypercube