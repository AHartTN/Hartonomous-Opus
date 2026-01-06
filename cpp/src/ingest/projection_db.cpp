/**
 * Database persistence for 4D compositions with PHYSICAL-FIRST centroids
 * 
 * Implements bulk upsert operations for composition tables using PostgreSQL COPY.
 * 
 * KEY PRINCIPLE: Centroids are computed deterministically in C++ from Unicode
 * codepoints - NO database lookups required. SQL is just the orchestrator.
 * 
 * Algorithm:
 * 1. Parse token label → UTF-32 codepoints
 * 2. For each codepoint: CoordinateMapper::map_codepoint() → Point4D
 * 3. Average Point4D values → composition centroid (computed in C++)
 * 4. Insert composition_child entries linking to atoms
 * 5. Persist to database
 */

#include "hypercube/ingest/projection_db.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/helpers.hpp"

#include <libpq-fe.h>
#include <iostream>
#include <sstream>
#include <cstring>
#include <codecvt>
#include <locale>

namespace hypercube {
namespace db {

using namespace hypercube::db;

// =============================================================================
// SQL Templates
// =============================================================================

namespace sql {

std::string atom_temp_table_sql() {
    return R"(
        CREATE TEMP TABLE tmp_atom_proj (
            id BYTEA PRIMARY KEY,
            geom_ewkb TEXT,
            hilbert_lo BIGINT,
            hilbert_hi BIGINT
        ) ON COMMIT DROP
    )";
}

std::string composition_temp_table_sql() {
    // Centroid computed in C++ from codepoints, stored here for bulk insert
    return R"(
        CREATE TEMP TABLE tmp_comp_proj (
            id BYTEA PRIMARY KEY,
            label TEXT,
            centroid_ewkb TEXT,
            hilbert_lo BIGINT,
            hilbert_hi BIGINT,
            child_count INTEGER,
            atom_count INTEGER
        ) ON COMMIT DROP
    )";
}

std::string atom_merge_sql() {
    return R"(
        INSERT INTO atom (id, geom, hilbert_lo, hilbert_hi)
        SELECT 
            t.id,
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
    // Centroid already computed in C++ from codepoints - just insert it
    return R"(
        INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi)
        SELECT 
            t.id,
            t.label,
            1,
            t.child_count,
            t.atom_count,
            ST_GeomFromEWKB(decode(t.centroid_ewkb, 'hex')),
            t.hilbert_lo,
            t.hilbert_hi
        FROM tmp_comp_proj t
        ON CONFLICT (id) DO UPDATE SET
            label = COALESCE(EXCLUDED.label, composition.label),
            child_count = EXCLUDED.child_count,
            atom_count = EXCLUDED.atom_count,
            centroid = EXCLUDED.centroid,
            hilbert_lo = EXCLUDED.hilbert_lo,
            hilbert_hi = EXCLUDED.hilbert_hi
    )";
}

// NEW: Create composition_child entries linking compositions to their Unicode atoms
std::string composition_child_temp_table_sql() {
    return R"(
        CREATE TEMP TABLE tmp_comp_child (
            composition_id BYTEA,
            child_id BYTEA,
            child_type CHAR(1),
            ordinal SMALLINT
        ) ON COMMIT DROP
    )";
}

// NEW: Merge composition_child entries
std::string composition_child_merge_sql() {
    return R"(
        INSERT INTO composition_child (composition_id, ordinal, child_type, child_id)
        SELECT t.composition_id, t.ordinal, t.child_type, t.child_id
        FROM tmp_comp_child t
        ON CONFLICT (composition_id, ordinal) DO NOTHING
    )";
}

// NOTE: compute_centroids_sql removed - centroids are now computed
// deterministically in C++ using CoordinateMapper::map_codepoint() and centroid()

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

// NOTE: escape_for_copy removed - use copy_escape() from db/operations.hpp

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
    
    Transaction tx(conn_);
    
    // Create temp table
    Result res = exec(conn_, sql::atom_temp_table_sql());
    if (!res.ok()) {
        std::cerr << "[DB] Create atom temp table failed: " << res.error_message() << "\n";
        return 0;
    }
    
    // COPY data to temp table using CopyStream
    {
        CopyStream copy(conn_, "COPY tmp_atom_proj FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        if (!copy.ok()) {
            std::cerr << "[DB] COPY atom start failed: " << copy.error() << "\n";
            return 0;
        }
        
        std::string batch;
        batch.reserve(1 << 20);
        
        for (const auto& token : atoms) {
            // id (BYTEA as hex)
            copy_bytea(batch, token.hash);
            copy_tab(batch);
            
            // geom_ewkb (no label - atoms identified by codepoint, not label)
            batch += build_pointzm_ewkb(token.coords);
            copy_tab(batch);
            
            // hilbert_lo, hilbert_hi
            batch += std::to_string(token.hilbert_lo);
            copy_tab(batch);
            batch += std::to_string(token.hilbert_hi);
            copy_newline(batch);
            
            if (batch.size() > (1 << 20)) {
                if (!copy.put(batch)) {
                    std::cerr << "[DB] COPY atom failed: " << copy.error() << "\n";
                    return 0;
                }
                batch.clear();
            }
        }
        
        if (!batch.empty()) {
            if (!copy.put(batch)) {
                std::cerr << "[DB] COPY atom failed: " << copy.error() << "\n";
                return 0;
            }
        }
        
        if (!copy.end()) {
            std::cerr << "[DB] COPY atom end failed: " << copy.error() << "\n";
            return 0;
        }
    }
    
    // Merge into atom table
    std::string merge_sql = sql::atom_merge_sql();
    // Replace $1 with boolean for update_existing
    size_t pos = merge_sql.find("$1");
    if (pos != std::string::npos) {
        merge_sql.replace(pos, 2, config_.update_existing ? "true" : "false");
    }
    
    res = exec(conn_, merge_sql);
    size_t updated = 0;
    if (res.ok()) {
        updated = static_cast<size_t>(cmd_tuples(res));
    } else {
        std::cerr << "[DB] Atom merge failed: " << res.error_message() << "\n";
    }
    
    tx.commit();
    return updated;
}

// Helper: Parse UTF-8 string into codepoints
static std::vector<uint32_t> utf8_to_codepoints(const std::string& label) {
    std::vector<uint32_t> codepoints;
    size_t i = 0;
    while (i < label.size()) {
        uint32_t cp = 0;
        unsigned char c = label[i];
        
        if ((c & 0x80) == 0) {
            cp = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = (c & 0x1F) << 6;
            if (i + 1 < label.size()) cp |= (label[i+1] & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = (c & 0x0F) << 12;
            if (i + 1 < label.size()) cp |= (label[i+1] & 0x3F) << 6;
            if (i + 2 < label.size()) cp |= (label[i+2] & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = (c & 0x07) << 18;
            if (i + 1 < label.size()) cp |= (label[i+1] & 0x3F) << 12;
            if (i + 2 < label.size()) cp |= (label[i+2] & 0x3F) << 6;
            if (i + 3 < label.size()) cp |= (label[i+3] & 0x3F);
            i += 4;
        } else {
            i += 1;
            continue;
        }
        codepoints.push_back(cp);
    }
    return codepoints;
}

size_t ProjectionPersister::persist_compositions(const std::vector<TokenData>& compositions) {
    if (compositions.empty()) return 0;
    
    // ==========================================================================
    // PHYSICAL-FIRST: Compute centroids deterministically in C++ from codepoints
    // ==========================================================================
    
    struct CompositionWithCentroid {
        Blake3Hash hash;
        std::string label;
        Point4D centroid;
        HilbertIndex hilbert;
        std::vector<std::pair<Blake3Hash, uint32_t>> atoms;  // (hash, codepoint)
    };
    
    std::vector<CompositionWithCentroid> computed;
    computed.reserve(compositions.size());
    
    for (const auto& token : compositions) {
        CompositionWithCentroid comp;
        comp.hash = token.hash;
        comp.label = token.label;
        
        // Parse label into codepoints
        std::vector<uint32_t> codepoints = utf8_to_codepoints(token.label);
        
        if (codepoints.empty()) {
            // Empty label - skip
            continue;
        }
        
        // Compute atom coordinates and centroid
        std::vector<Point4D> atom_coords;
        atom_coords.reserve(codepoints.size());
        
        for (uint32_t cp : codepoints) {
            // Deterministic coordinate from codepoint
            Point4D coord = CoordinateMapper::map_codepoint(cp);
            atom_coords.push_back(coord);
            
            // Compute atom hash for composition_child
            Blake3Hash atom_hash = Blake3Hasher::hash_codepoint(cp);
            comp.atoms.emplace_back(atom_hash, cp);
        }
        
        // Compute centroid as average of atom coordinates
        comp.centroid = CoordinateMapper::centroid(atom_coords);
        
        // Compute Hilbert index for the centroid
        comp.hilbert = HilbertCurve::coords_to_index(comp.centroid);
        
        computed.push_back(std::move(comp));
    }
    
    std::cerr << "[DB] Computed " << computed.size() << " composition centroids from atoms\n";
    
    // ==========================================================================
    // Database persistence with Transaction RAII
    // ==========================================================================
    
    Transaction tx(conn_);
    
    // Step 1: Create temp table for compositions
    Result res = exec(conn_, sql::composition_temp_table_sql());
    if (!res.ok()) {
        std::cerr << "[DB] Create comp temp table failed: " << res.error_message() << "\n";
        return 0;
    }
    
    // Step 2: Create temp table for composition_child
    res = exec(conn_, sql::composition_child_temp_table_sql());
    if (!res.ok()) {
        std::cerr << "[DB] Create comp_child temp table failed: " << res.error_message() << "\n";
        return 0;
    }
    
    // Step 3: COPY composition data with pre-computed centroids using CopyStream
    {
        CopyStream copy(conn_, "COPY tmp_comp_proj (id, label, centroid_ewkb, hilbert_lo, hilbert_hi, child_count, atom_count) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        if (!copy.ok()) {
            std::cerr << "[DB] COPY comp start failed: " << copy.error() << "\n";
            return 0;
        }
        
        std::string batch;
        batch.reserve(1 << 20);
        
        for (const auto& comp : computed) {
            // id (BYTEA as hex)
            copy_bytea(batch, comp.hash);
            copy_tab(batch);
            
            // label
            copy_escape(batch, comp.label);
            copy_tab(batch);
            
            // centroid_ewkb (computed in C++)
            std::array<uint32_t, 4> coords = {comp.centroid.x, comp.centroid.y, comp.centroid.z, comp.centroid.m};
            batch += build_pointzm_ewkb(coords);
            copy_tab(batch);
            
            // hilbert_lo, hilbert_hi (computed in C++)
            // CAST TO SIGNED INT64 FOR POSTGRESQL (bit-preserving reinterpretation)
            int64_t h_lo = static_cast<int64_t>(comp.hilbert.lo);
            int64_t h_hi = static_cast<int64_t>(comp.hilbert.hi);
            batch += std::to_string(h_lo);
            copy_tab(batch);
            batch += std::to_string(h_hi);
            copy_tab(batch);
            
            // child_count, atom_count
            batch += std::to_string(comp.atoms.size());
            copy_tab(batch);
            batch += std::to_string(comp.atoms.size());
            copy_newline(batch);
            
            if (batch.size() > (1 << 20)) {
                if (!copy.put(batch)) {
                    std::cerr << "[DB] COPY comp failed: " << copy.error() << "\n";
                    return 0;
                }
                batch.clear();
            }
        }
        
        if (!batch.empty()) {
            if (!copy.put(batch)) {
                std::cerr << "[DB] COPY comp failed: " << copy.error() << "\n";
                return 0;
            }
        }
        
        if (!copy.end()) {
            std::cerr << "[DB] COPY comp end failed: " << copy.error() << "\n";
            return 0;
        }
    }
    
    std::cerr << "[DB] Copied " << computed.size() << " compositions to temp table\n";
    
    // Step 4: COPY composition_child data (using pre-computed atom hashes) with CopyStream
    size_t child_count = 0;
    {
        CopyStream copy(conn_, "COPY tmp_comp_child FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        if (!copy.ok()) {
            std::cerr << "[DB] COPY comp_child start failed: " << copy.error() << "\n";
            return 0;
        }
        
        std::string batch;
        batch.reserve(1 << 20);
        
        for (const auto& comp : computed) {
            int ordinal = 0;
            for (const auto& [atom_hash, codepoint] : comp.atoms) {
                // composition_id, child_id, child_type, ordinal
                copy_bytea(batch, comp.hash);
                copy_tab(batch);
                copy_bytea(batch, atom_hash);
                batch += "\tA\t";  // 'A' for Atom
                batch += std::to_string(ordinal);
                copy_newline(batch);
                
                ++ordinal;
                ++child_count;
                
                if (batch.size() > (1 << 20)) {
                    if (!copy.put(batch)) {
                        std::cerr << "[DB] COPY comp_child failed: " << copy.error() << "\n";
                        return 0;
                    }
                    batch.clear();
                }
            }
        }
        
        if (!batch.empty()) {
            if (!copy.put(batch)) {
                std::cerr << "[DB] COPY comp_child failed: " << copy.error() << "\n";
                return 0;
            }
        }
        
        if (!copy.end()) {
            std::cerr << "[DB] COPY comp_child end failed: " << copy.error() << "\n";
            return 0;
        }
    }
    
    std::cerr << "[DB] Created " << child_count << " composition_child entries\n";
    
    // Step 5: Merge compositions into table (with pre-computed centroids)
    res = exec(conn_, sql::composition_merge_sql());
    size_t updated = 0;
    if (res.ok()) {
        updated = static_cast<size_t>(cmd_tuples(res));
    } else {
        std::cerr << "[DB] Composition merge failed: " << res.error_message() << "\n";
    }
    
    // Step 6: Merge composition_child entries
    res = exec(conn_, sql::composition_child_merge_sql());
    if (!res.ok()) {
        std::cerr << "[DB] Composition_child merge failed: " << res.error_message() << "\n";
    }
    
    tx.commit();
    
    std::cerr << "[DB] Persisted " << updated << " compositions with physical-first centroids\n";
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
