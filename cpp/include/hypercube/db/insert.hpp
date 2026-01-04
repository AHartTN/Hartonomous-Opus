#pragma once

#include "hypercube/ingest/cpe.hpp"
#include <libpq-fe.h>
#include <vector>

namespace hypercube::db {

// Batch insert compositions using COPY protocol
// Returns true on success
bool insert_compositions(PGconn* conn, const std::vector<ingest::CompositionRecord>& comps);

} // namespace hypercube::db
