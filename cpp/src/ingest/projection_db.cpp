/**
 * Database persistence for 4D Laplacian projections
 * 
 * Implements bulk upsert operations for atom and composition tables
 * using PostgreSQL COPY for high-throughput insertion.
 */

#include "hypercube/ingest/projection_db.hpp"

#include <libpq-fe.h>
#include <iostream>
#include <sstream>
#include <cstring>

namespace hypercube {
namespace db {

// =============================================================================
// SQL Templates
// =============================================================================

namespace sql {

std::string atom_temp_table_sql() {
    return R"(
        CREATE TEMP TABLE tmp_atom_proj (
            id BYTEA PRIMARY KEY,
            label TEXT,
            geom_ewkb TEXT,
            hilbert_lo BIGINT,
            hilbert_hi BIGINT
        ) ON COMMIT DROP
    )";
}

std::string composition_temp_table_sql() {
    return R"(
        CREATE TEMP TABLE tmp_comp_proj (
            id BYTEA PRIMARY KEY,
            label TEXT,
            centroid_ewkb TEXT,
            hilbert_lo BIGINT,
            hilbert_hi BIGINT
        ) ON COMMIT DROP
    )";
}

std::string atom_merge_sql() {
    return R"(
        INSERT INTO atom (id, label, geom, hilbert_lo, hilbert_hi)
        SELECT 
            t.id,
            t.label,
            ST_GeomFromEWKB(decode(t.geom_ewkb, 'hex')),
            t.hilbert_lo,
            t.hilbert_hi
        FROM tmp_atom_proj t
        ON CONFLICT (id) DO UPDATE SET
            geom = EXCLUDED.geom,
            hilbert_lo = EXCLUDED.hilbert_lo,
            hilbert_hi = EXCLUDED.hilbert_hi
        WHERE atom.geom IS NULL OR $1
    )";
}

std::string composition_merge_sql() {
    return R"(
        INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi)
        SELECT 
            t.id,
            t.label,
            1,
            1,
            1,
            ST_GeomFromEWKB(decode(t.centroid_ewkb, 'hex')),
            t.hilbert_lo,
            t.hilbert_hi
        FROM tmp_comp_proj t
        ON CONFLICT (id) DO UPDATE SET
            centroid = EXCLUDED.centroid,
            hilbert_lo = EXCLUDED.hilbert_lo,
            hilbert_hi = EXCLUDED.hilbert_hi
        WHERE composition.centroid IS NULL OR $1
    )";
}

std::string atom_upsert_sql() {
    return R"(
        INSERT INTO atom (id, label, geom, hilbert_lo, hilbert_hi)
        VALUES ($1, $2, ST_GeomFromEWKB(decode($3, 'hex')), $4, $5)
        ON CONFLICT (id) DO UPDATE SET
            geom = EXCLUDED.geom,
            hilbert_lo = EXCLUDED.hilbert_lo,
            hilbert_hi = EXCLUDED.hilbert_hi
        RETURNING id
    )";
}

std::string composition_upsert_sql() {
    return R"(
        INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi)
        VALUES ($1, $2, 1, 1, 1, ST_GeomFromEWKB(decode($3, 'hex')), $4, $5)
        ON CONFLICT (id) DO UPDATE SET
            centroid = EXCLUDED.centroid,
            hilbert_lo = EXCLUDED.hilbert_lo,
            hilbert_hi = EXCLUDED.hilbert_hi
        RETURNING id
    )";
}

} // namespace sql

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Write a double as little-endian hex
void write_double_hex(std::string& out, double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
        out += hex[(bits >> (i * 8 + 4)) & 0xF];
        out += hex[(bits >> (i * 8)) & 0xF];
    }
}

// Escape text for COPY
std::string escape_for_copy(const std::string& s) {
    std::string result;
    result.reserve(s.size() * 2);
    for (char c : s) {
        switch (c) {
            case '\t': result += "\\t"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\\': result += "\\\\"; break;
            default: result += c; break;
        }
    }
    return result;
}

} // anonymous namespace

// =============================================================================
// ProjectionPersister Implementation
// =============================================================================

ProjectionPersister::ProjectionPersister(PGconn* conn, const PersistConfig& config)
    : conn_(conn), config_(config) {}

std::string ProjectionPersister::build_pointzm_ewkb(const std::array<uint32_t, 4>& coords) {
    std::string ewkb;
    ewkb.reserve(74);
    
    // EWKB header: little-endian (01), POINTZM with SRID (010000e0), SRID=0
    ewkb += "01";           // Little-endian
    ewkb += "010000e0";     // POINTZM with SRID flag
    ewkb += "00000000";     // SRID = 0
    
    // Coordinates as doubles (uint32 stored directly in double for lossless storage)
    write_double_hex(ewkb, static_cast<double>(coords[0]));
    write_double_hex(ewkb, static_cast<double>(coords[1]));
    write_double_hex(ewkb, static_cast<double>(coords[2]));
    write_double_hex(ewkb, static_cast<double>(coords[3]));
    
    return ewkb;
}

size_t ProjectionPersister::persist_atoms(const std::vector<TokenData>& atoms) {
    if (atoms.empty()) return 0;
    
    PGresult* res = PQexec(conn_, "BEGIN");
    PQclear(res);
    
    // Create temp table
    res = PQexec(conn_, sql::atom_temp_table_sql().c_str());
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[DB] Create atom temp table failed: " << PQerrorMessage(conn_) << "\n";
        PQclear(res);
        PQexec(conn_, "ROLLBACK");
        return 0;
    }
    PQclear(res);
    
    // COPY data to temp table
    res = PQexec(conn_, "COPY tmp_atom_proj FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "[DB] COPY atom start failed: " << PQerrorMessage(conn_) << "\n";
        PQclear(res);
        PQexec(conn_, "ROLLBACK");
        return 0;
    }
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 20);
    
    for (const auto& token : atoms) {
        // id (BYTEA as hex)
        batch += "\\\\x";
        batch += token.hash.to_hex();
        batch += "\t";
        
        // label
        batch += escape_for_copy(token.label);
        batch += "\t";
        
        // geom_ewkb
        batch += build_pointzm_ewkb(token.coords);
        batch += "\t";
        
        // hilbert_lo, hilbert_hi
        batch += std::to_string(token.hilbert_lo);
        batch += "\t";
        batch += std::to_string(token.hilbert_hi);
        batch += "\n";
        
        if (batch.size() > (1 << 20)) {
            PQputCopyData(conn_, batch.c_str(), static_cast<int>(batch.size()));
            batch.clear();
        }
    }
    
    if (!batch.empty()) {
        PQputCopyData(conn_, batch.c_str(), static_cast<int>(batch.size()));
    }
    PQputCopyEnd(conn_, nullptr);
    res = PQgetResult(conn_);
    PQclear(res);
    
    // Merge into atom table
    std::string merge_sql = sql::atom_merge_sql();
    // Replace $1 with boolean for update_existing
    size_t pos = merge_sql.find("$1");
    if (pos != std::string::npos) {
        merge_sql.replace(pos, 2, config_.update_existing ? "true" : "false");
    }
    
    res = PQexec(conn_, merge_sql.c_str());
    size_t updated = 0;
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
        updated = static_cast<size_t>(atoi(PQcmdTuples(res)));
    } else {
        std::cerr << "[DB] Atom merge failed: " << PQerrorMessage(conn_) << "\n";
    }
    PQclear(res);
    
    res = PQexec(conn_, "COMMIT");
    PQclear(res);
    
    return updated;
}

size_t ProjectionPersister::persist_compositions(const std::vector<TokenData>& compositions) {
    if (compositions.empty()) return 0;
    
    PGresult* res = PQexec(conn_, "BEGIN");
    PQclear(res);
    
    // Create temp table
    res = PQexec(conn_, sql::composition_temp_table_sql().c_str());
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[DB] Create comp temp table failed: " << PQerrorMessage(conn_) << "\n";
        PQclear(res);
        PQexec(conn_, "ROLLBACK");
        return 0;
    }
    PQclear(res);
    
    // COPY data to temp table
    res = PQexec(conn_, "COPY tmp_comp_proj FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "[DB] COPY comp start failed: " << PQerrorMessage(conn_) << "\n";
        PQclear(res);
        PQexec(conn_, "ROLLBACK");
        return 0;
    }
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 20);
    
    for (const auto& token : compositions) {
        // id (BYTEA as hex)
        batch += "\\\\x";
        batch += token.hash.to_hex();
        batch += "\t";
        
        // label
        batch += escape_for_copy(token.label);
        batch += "\t";
        
        // centroid_ewkb
        batch += build_pointzm_ewkb(token.coords);
        batch += "\t";
        
        // hilbert_lo, hilbert_hi
        batch += std::to_string(token.hilbert_lo);
        batch += "\t";
        batch += std::to_string(token.hilbert_hi);
        batch += "\n";
        
        if (batch.size() > (1 << 20)) {
            PQputCopyData(conn_, batch.c_str(), static_cast<int>(batch.size()));
            batch.clear();
        }
    }
    
    if (!batch.empty()) {
        PQputCopyData(conn_, batch.c_str(), static_cast<int>(batch.size()));
    }
    PQputCopyEnd(conn_, nullptr);
    res = PQgetResult(conn_);
    PQclear(res);
    
    // Merge into composition table
    std::string merge_sql = sql::composition_merge_sql();
    size_t pos = merge_sql.find("$1");
    if (pos != std::string::npos) {
        merge_sql.replace(pos, 2, config_.update_existing ? "true" : "false");
    }
    
    res = PQexec(conn_, merge_sql.c_str());
    size_t updated = 0;
    if (PQresultStatus(res) == PGRES_COMMAND_OK) {
        updated = static_cast<size_t>(atoi(PQcmdTuples(res)));
    } else {
        std::cerr << "[DB] Composition merge failed: " << PQerrorMessage(conn_) << "\n";
    }
    PQclear(res);
    
    res = PQexec(conn_, "COMMIT");
    PQclear(res);
    
    return updated;
}

size_t ProjectionPersister::persist(const std::vector<TokenData>& tokens) {
    // Separate atoms and compositions
    std::vector<TokenData> atoms, compositions;
    atoms.reserve(tokens.size());
    compositions.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        if (token.is_atom) {
            atoms.push_back(token);
        } else {
            compositions.push_back(token);
        }
    }
    
    std::cerr << "[DB] Persisting " << atoms.size() << " atoms, " 
              << compositions.size() << " compositions...\n";
    
    size_t atom_count = persist_atoms(atoms);
    size_t comp_count = persist_compositions(compositions);
    
    std::cerr << "[DB] Updated " << atom_count << " atoms, " << comp_count << " compositions\n";
    
    return atom_count + comp_count;
}

size_t ProjectionPersister::persist(
    const std::vector<std::string>& labels,
    const std::vector<Blake3Hash>& hashes,
    const std::vector<bool>& is_atom,
    const ProjectionResult& result
) {
    const size_t n = labels.size();
    if (n != hashes.size() || n != is_atom.size() || n != result.coords.size()) {
        std::cerr << "[DB] Mismatched array sizes\n";
        return 0;
    }
    
    std::vector<TokenData> tokens(n);
    for (size_t i = 0; i < n; ++i) {
        tokens[i].label = labels[i];
        tokens[i].hash = hashes[i];
        tokens[i].is_atom = is_atom[i];
        tokens[i].coords = result.coords[i];
        tokens[i].hilbert_lo = result.hilbert_lo[i];
        tokens[i].hilbert_hi = result.hilbert_hi[i];
    }
    
    return persist(tokens);
}

} // namespace db
} // namespace hypercube
