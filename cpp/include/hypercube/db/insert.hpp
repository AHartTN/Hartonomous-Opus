#pragma once

#include "hypercube/ingest/cpe.hpp"
#include <libpq-fe.h>
#include <vector>
#include <unordered_set>

namespace hypercube::db {

// Check which composition hashes already exist in the database
// Returns a set of hex strings for hashes that exist
std::unordered_set<std::string> check_existing_compositions(
    PGconn* conn,
    const std::vector<ingest::CompositionRecord>& comps);

// Filter compositions to only include new ones (not already in database)
// This provides cross-session deduplication
std::vector<ingest::CompositionRecord> filter_new_compositions(
    PGconn* conn,
    const std::vector<ingest::CompositionRecord>& comps);

// Batch insert compositions using COPY protocol
// Returns true on success
bool insert_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps);

// Insert only new compositions (with cross-session deduplication)
// This checks which compositions exist first, then inserts only new ones
// Returns number of new compositions inserted
size_t insert_new_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps);

} // namespace hypercube::db
