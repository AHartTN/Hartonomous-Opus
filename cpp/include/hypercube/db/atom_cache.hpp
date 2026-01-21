#pragma once

#include "hypercube/types.hpp"
#include <libpq-fe.h>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>

namespace hypercube::db {

// Cached atom information for fast lookup during ingestion
// Coordinates are unsigned 32-bit values from YOUR Hilbert mapping
struct AtomInfo {
    Blake3Hash hash;
    double coord_x, coord_y, coord_z, coord_m;  // BIGINT from DB (unsigned 32-bit fits in signed 64-bit)
};

// Load atoms for specific codepoints only (fast - queries only needed atoms)
bool load_atoms_for_codepoints(
    PGconn* conn,
    const std::unordered_set<uint32_t>& codepoints,
    std::unordered_map<uint32_t, AtomInfo>& cache
);

// Get atom from cache (returns nullptr if not found)
const AtomInfo* get_cached_atom(
    const std::unordered_map<uint32_t, AtomInfo>& cache,
    uint32_t codepoint
);

} // namespace hypercube::db
