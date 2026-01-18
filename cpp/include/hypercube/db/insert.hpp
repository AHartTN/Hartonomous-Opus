#pragma once

#include "hypercube/ingest/cpe.hpp"
#include <libpq-fe.h>
#include <vector>

namespace hypercube::db {

// ============================================================================
// ARCHITECTURE: C++ computes everything, SQL just stores
// - Labels computed in C++ during PMI contraction (not SQL UPDATE)
// - No round-trips: deduplication via ON CONFLICT DO NOTHING
// - All data written in single batch COPY → temp table → INSERT
// ============================================================================

// Batch insert compositions using COPY protocol with ON CONFLICT deduplication
// Labels must be pre-computed in C++ (CompositionRecord.label field)
// Returns true on success
bool insert_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps);

// Insert compositions - deduplication handled by DB via ON CONFLICT DO NOTHING
// No round-trip query to check existence - just insert and let DB handle conflicts
// Returns number of compositions passed in (actual inserted count logged to stderr)
size_t insert_new_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps);

} // namespace hypercube::db
