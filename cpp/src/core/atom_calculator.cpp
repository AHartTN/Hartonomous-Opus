/**
 * Atom Calculator - Zero-Roundtrip Implementation
 * 
 * All computation is local. No database access.
 */

#include "hypercube/atom_calculator.hpp"
#include "hypercube/util/utf8.hpp"
#include <cstring>

namespace hypercube {

AtomRecord AtomCalculator::compute_atom(uint32_t codepoint) noexcept {
    AtomRecord rec;
    rec.codepoint = codepoint;
    rec.coords = CoordinateMapper::map_codepoint(codepoint);
    rec.hash = Blake3Hasher::hash_codepoint(codepoint);
    rec.hilbert = HilbertCurve::coords_to_index(rec.coords);
    rec.depth = 0;
    rec.atom_count = 1;
    return rec;
}

CompositionRecord AtomCalculator::compute_composition(
    const std::vector<uint32_t>& codepoints
) noexcept {
    CompositionRecord rec;
    
    if (codepoints.empty()) {
        return rec;
    }
    
    // Build child info from atoms
    std::vector<ChildInfo> children;
    children.reserve(codepoints.size());
    
    for (uint32_t cp : codepoints) {
        AtomRecord atom = compute_atom(cp);
        ChildInfo ci;
        ci.hash = atom.hash;
        ci.coords = atom.coords;
        ci.depth = atom.depth;
        ci.atom_count = atom.atom_count;
        children.push_back(ci);
    }
    
    return compute_composition(children);
}

CompositionRecord AtomCalculator::compute_composition(
    const std::vector<ChildInfo>& children
) noexcept {
    CompositionRecord rec;
    
    if (children.empty()) {
        return rec;
    }
    
    if (children.size() == 1) {
        // Single child - this IS that child (no new composition)
        rec.hash = children[0].hash;
        rec.centroid = children[0].coords;
        rec.hilbert = HilbertCurve::coords_to_index(rec.centroid);
        rec.children.push_back(children[0].hash);
        rec.child_coords.push_back(children[0].coords);
        rec.child_depths.push_back(children[0].depth);  // Track child type
        rec.depth = children[0].depth;
        rec.atom_count = children[0].atom_count;
        return rec;
    }
    
    // Collect child hashes, coords, and depths
    rec.children.reserve(children.size());
    rec.child_coords.reserve(children.size());
    rec.child_depths.reserve(children.size());
    
    for (const auto& child : children) {
        rec.children.push_back(child.hash);
        rec.child_coords.push_back(child.coords);
        rec.child_depths.push_back(child.depth);  // Track child type: 0=atom, >0=composition
    }
    
    // Compute composition hash: BLAKE3(ord_0 || hash_0 || ord_1 || hash_1 || ...)
    rec.hash = Blake3Hasher::hash_children_ordered(std::span<const Blake3Hash>(rec.children.data(), rec.children.size()));
    
    // Compute centroid as average of child coordinates
    rec.centroid = CoordinateMapper::centroid(rec.child_coords);
    
    // Compute Hilbert index from centroid
    rec.hilbert = HilbertCurve::coords_to_index(rec.centroid);
    
    // Compute depth = max(child depths) + 1
    uint32_t max_depth = 0;
    rec.atom_count = 0;
    for (const auto& child : children) {
        max_depth = std::max(max_depth, child.depth);
        rec.atom_count += child.atom_count;
    }
    rec.depth = max_depth + 1;
    
    return rec;
}

std::vector<uint32_t> AtomCalculator::decode_utf8(const std::string& text) noexcept {
    return util::decode_utf8(text);
}

CompositionRecord AtomCalculator::compute_vocab_token(
    const std::string& token_text
) noexcept {
    auto codepoints = decode_utf8(token_text);
    return compute_composition(codepoints);
}

} // namespace hypercube
