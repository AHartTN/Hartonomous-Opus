#pragma once

/**
 * Database persistence for 4D Laplacian projections
 * 
 * Handles inserting/updating atom and composition tables with
 * 4D coordinates and Hilbert indices computed from embedding projection.
 */

#include "hypercube/types.hpp"
#include "hypercube/laplacian_4d.hpp"
#include <string>
#include <vector>
#include <array>
#include <cstdint>

// Forward declare libpq types
struct pg_conn;
typedef pg_conn PGconn;

namespace hypercube {
namespace db {

/**
 * Configuration for database persistence
 */
struct PersistConfig {
    std::string model_name;
    int batch_size = 10000;
    bool update_existing = true;  // If false, skip tokens that already have coordinates
    bool verbose = false;
};

/**
 * Token data for persistence
 */
struct TokenData {
    std::string label;
    Blake3Hash hash;
    bool is_atom;  // true = single char, false = composition
    std::array<uint32_t, 4> coords;
    int64_t hilbert_lo;
    int64_t hilbert_hi;
};

/**
 * Persist 4D projections to atom and composition tables
 * 
 * For atoms (single characters):
 *   UPDATE atom SET geom = POINTZM, hilbert_lo = ?, hilbert_hi = ?
 * 
 * For compositions (multi-char tokens):
 *   UPDATE composition SET centroid = POINTZM, hilbert_lo = ?, hilbert_hi = ?
 */
class ProjectionPersister {
public:
    explicit ProjectionPersister(PGconn* conn, const PersistConfig& config = PersistConfig{});
    
    /**
     * Persist a batch of token projections
     * @param tokens Vector of TokenData with coordinates
     * @return Number of rows updated
     */
    size_t persist(const std::vector<TokenData>& tokens);
    
    /**
     * Persist projection results directly from LaplacianProjector output
     * @param labels Token labels
     * @param hashes Token hashes
     * @param is_atom Whether each token is an atom (single char)
     * @param result ProjectionResult from LaplacianProjector
     * @return Number of rows updated
     */
    size_t persist(
        const std::vector<std::string>& labels,
        const std::vector<Blake3Hash>& hashes,
        const std::vector<bool>& is_atom,
        const ProjectionResult& result
    );
    
private:
    PGconn* conn_;
    PersistConfig config_;
    
    // Build POINTZM EWKB hex string
    static std::string build_pointzm_ewkb(const std::array<uint32_t, 4>& coords);
    
    // Persist atoms using COPY
    size_t persist_atoms(const std::vector<TokenData>& atoms);
    
    // Persist compositions using COPY
    size_t persist_compositions(const std::vector<TokenData>& compositions);
};

/**
 * SQL helper functions for upserts
 */
namespace sql {

/**
 * Upsert atom with 4D coordinates
 */
std::string atom_upsert_sql();

/**
 * Upsert composition with 4D centroid
 */
std::string composition_upsert_sql();

/**
 * Create temp table for atom bulk upsert
 */
std::string atom_temp_table_sql();

/**
 * Create temp table for composition bulk upsert
 */
std::string composition_temp_table_sql();

/**
 * Merge atoms from temp table
 */
std::string atom_merge_sql();

/**
 * Merge compositions from temp table
 */
std::string composition_merge_sql();

} // namespace sql

} // namespace db
} // namespace hypercube
