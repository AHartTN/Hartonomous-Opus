#include "hypercube/atom_registry.hpp"

namespace hypercube {

AtomRegistry AtomRegistry::build() {
    std::vector<AtomEntry> atoms;
    atoms.reserve(DenseRegistry::total_active());

    for (uint32_t dense_rank = 0; dense_rank < DenseRegistry::total_active(); ++dense_rank) {
        uint32_t codepoint = DenseRegistry::get_codepoint(dense_rank);
        uint32_t semantic_rank = SemanticOrdering::get_rank(codepoint);
        uint8_t category = static_cast<uint8_t>(UnicodeCategorizer::categorize(codepoint));
        Point4F base = CoordinateMapper::map_codepoint_float(codepoint);
        Point4F jittered = base;

        AtomEntry entry{codepoint, dense_rank, semantic_rank, category, base, jittered};
        atoms.push_back(entry);
    }

    return AtomRegistry{std::move(atoms)};
}

} // namespace hypercube