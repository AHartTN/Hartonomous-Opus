/**
 * Atom Calculator - Zero-Roundtrip Deterministic Computation
 * 
 * Computes ALL atom and composition properties client-side:
 * - Coordinates (from codepoint via CoordinateMapper)
 * - Hash (BLAKE3 of UTF-8 bytes for atoms, ordered children for compositions)
 * - Hilbert index (from coordinates)
 * - Centroid (average of child coordinates)
 * - Depth and atom_count
 * 
 * NO DATABASE ACCESS REQUIRED.
 * All properties are deterministic from the input codepoints/children.
 */

#pragma once

#include "hypercube/types.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/hilbert.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace hypercube {

/**
 * Complete atom record with all computed properties
 */
struct AtomRecord {
    uint32_t codepoint;
    Blake3Hash hash;
    Point4D coords;
    HilbertIndex hilbert;
    
    // Atoms are always depth 0, atom_count 1
    static constexpr uint32_t depth = 0;
    static constexpr uint64_t atom_count = 1;
};

/**
 * Complete composition record with all computed properties
 */
struct CompositionRecord {
    Blake3Hash hash;
    Point4D centroid;
    HilbertIndex hilbert;
    std::vector<Blake3Hash> children;   // Child hashes in order
    std::vector<Point4D> child_coords;  // Child coordinates for LINESTRINGZM
    uint32_t depth;
    uint64_t atom_count;
};

/**
 * AtomCalculator - Deterministic property computation
 * 
 * All methods are static. No state, no database, no network.
 * Pure functions from input to output.
 */
class AtomCalculator {
public:
    /**
     * Compute all properties for a single atom (codepoint)
     */
    static AtomRecord compute_atom(uint32_t codepoint) noexcept;
    
    /**
     * Compute all properties for a composition from codepoint sequence
     * This is for creating a "word" composition like "captain" = [c,a,p,t,a,i,n]
     */
    static CompositionRecord compute_composition(
        const std::vector<uint32_t>& codepoints
    ) noexcept;
    
    /**
     * Compute all properties for a composition from child records
     * Children can be atoms or other compositions
     * This enables hierarchical composition without DB lookups
     */
    struct ChildInfo {
        Blake3Hash hash;
        Point4D coords;
        uint32_t depth;
        uint64_t atom_count;
    };
    
    static CompositionRecord compute_composition(
        const std::vector<ChildInfo>& children
    ) noexcept;
    
    /**
     * Decode a UTF-8 string to codepoints
     */
    static std::vector<uint32_t> decode_utf8(const std::string& text) noexcept;
    
    /**
     * Compute composition for a vocab token string
     * Convenience method: decode UTF-8 then compute composition
     */
    static CompositionRecord compute_vocab_token(
        const std::string& token_text
    ) noexcept;
};

} // namespace hypercube
