#pragma once

#include "hypercube/types.hpp"
#include "hypercube/dense_registry.hpp"
#include "hypercube/semantic_ordering.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/unicode_categorization.hpp"
#include <vector>
#include <cstdint>

namespace hypercube {

/**
 * Atom Entry - Contiguous registry entry for each atom
 */
struct AtomEntry {
    uint32_t codepoint;
    uint32_t dense_rank;
    uint32_t semantic_rank;
    uint8_t category;
    Point4F base;
    Point4F jittered;
};

/**
 * Atom Registry - Deterministic Atom Collection
 */
class AtomRegistry {
public:
    std::vector<AtomEntry> atoms;

    /**
     * Build the registry deterministically
     */
    static AtomRegistry build();
};

} // namespace hypercube