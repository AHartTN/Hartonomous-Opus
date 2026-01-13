#include "hypercube/dense_registry.hpp"
#include "hypercube/types.hpp"
#include <algorithm>

namespace hypercube {

// Static member definitions
std::unordered_map<uint32_t, uint32_t> DenseRegistry::codepoint_to_rank;
std::vector<uint32_t> DenseRegistry::rank_to_codepoint;
bool DenseRegistry::initialized = false;
std::mutex DenseRegistry::init_mutex;

void DenseRegistry::initialize()
{
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized)
        return;

    // Collect all valid codepoints
    // Include all codepoints including surrogates as valid atoms
    for (uint32_t cp = 0; cp <= hypercube::constants::MAX_CODEPOINT; ++cp) {
        // Skip surrogates and non-characters
        if (cp >= 0xD800 && cp <= 0xDFFF)
            continue; // Surrogates
        if ((cp & 0xFFFF) >= 0xFFFE)
            continue; // Non-characters

        uint32_t rank = static_cast<uint32_t>(rank_to_codepoint.size());
        rank_to_codepoint.push_back(cp);
        codepoint_to_rank[cp] = rank;
    }

    initialized = true;
}

uint32_t DenseRegistry::get_rank(uint32_t cp)
{
    if (!initialized)
        initialize();
    auto it = codepoint_to_rank.find(cp);
    return (it != codepoint_to_rank.end()) ? it->second : 0;
}

uint32_t DenseRegistry::total_active()
{
    if (!initialized)
        initialize();
    return static_cast<uint32_t>(rank_to_codepoint.size());
}

uint32_t DenseRegistry::get_codepoint(uint32_t rank)
{
    if (!initialized)
        initialize();
    return (rank < rank_to_codepoint.size()) ? rank_to_codepoint[rank] : 0;
}

} // namespace hypercube