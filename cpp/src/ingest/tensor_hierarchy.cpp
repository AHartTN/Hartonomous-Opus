/**
 * @file tensor_hierarchy.cpp
 * @brief Build and insert tensor path hierarchy as compositions
 * 
 * Parses tensor names into hierarchical path components and inserts them
 * as composition records with atom and composition children.
 */

#include "hypercube/ingest/db_operations.hpp"

namespace hypercube {
namespace ingest {
namespace db {

bool insert_tensor_hierarchy(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    if (ctx.tensors.empty()) return true;
    
    std::cerr << "[HIER] Building tensor hierarchy from " << ctx.tensors.size() << " tensors\n";
    
    // Collect all unique path components across all tensors
    std::unordered_map<std::string, int> path_to_depth;  // path -> depth
    std::unordered_map<std::string, std::string> path_to_parent;  // path -> parent path
    
    for (const auto& [tensor_name, meta] : ctx.tensors) {
        std::vector<std::string> components = split_tensor_path(tensor_name);
        
        std::string parent_path;
        for (size_t i = 0; i < components.size(); ++i) {
            const std::string& path = components[i];
            int depth = static_cast<int>(i) + 1;
            
            // Record this path if not seen or update depth if deeper
            auto it = path_to_depth.find(path);
            if (it == path_to_depth.end()) {
                path_to_depth[path] = depth;
                if (!parent_path.empty()) {
                    path_to_parent[path] = parent_path;
                }
            }
            
            parent_path = path;
        }
    }
    
    std::cerr << "[HIER] Found " << path_to_depth.size() << " unique hierarchy nodes\n";
    
    // Build composition batch
    std::string comp_batch;
    comp_batch.reserve(path_to_depth.size() * 256);
    
    // Build parent->child edge batch
    std::string child_batch;
    child_batch.reserve(path_to_depth.size() * 128);
    
    // Track full composition records for linking and atom children
    std::unordered_map<std::string, CompositionRecord> path_to_comp;
    
    // Batch for atom children (the characters of each path)
    std::string atom_child_batch;
    atom_child_batch.reserve(path_to_depth.size() * 256);  // Avg ~20 chars per path
    
    for (const auto& [path, depth] : path_to_depth) {
        // Compute FULL composition from path - includes atom children
        CompositionRecord comp = AtomCalculator::compute_vocab_token(path);
        path_to_comp[path] = comp;
        
        // Build composition row with geometry:
        // id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi
        comp_batch += "\\\\x";
        comp_batch += comp.hash.to_hex();
        comp_batch += "\t";
        
        // Escape path for COPY
        for (char ch : path) {
            if (ch == '\t') comp_batch += "\\t";
            else if (ch == '\n') comp_batch += "\\n";
            else if (ch == '\\') comp_batch += "\\\\";
            else comp_batch += ch;
        }
        comp_batch += "\t";
        comp_batch += std::to_string(depth);
        comp_batch += "\t";
        comp_batch += std::to_string(comp.children.size());  // child_count = atom count
        comp_batch += "\t";
        comp_batch += std::to_string(comp.atom_count);
        comp_batch += "\t";
        
        // geom (LINESTRINGZM from child coordinates)
        std::string geom_ewkb = build_composition_linestringzm_ewkb(comp.child_coords);
        if (!geom_ewkb.empty()) {
            comp_batch += geom_ewkb;
        } else {
            comp_batch += "\\N";
        }
        comp_batch += "\t";
        
        // centroid (POINTZM)
        std::string centroid_ewkb = build_composition_pointzm_ewkb(comp.centroid);
        comp_batch += centroid_ewkb;
        comp_batch += "\t";
        
        // hilbert_lo, hilbert_hi
        comp_batch += std::to_string(static_cast<int64_t>(comp.hilbert.lo));
        comp_batch += "\t";
        comp_batch += std::to_string(static_cast<int64_t>(comp.hilbert.hi));
        comp_batch += "\n";
        
        // Build atom children - each character in the path is an atom child
        for (size_t j = 0; j < comp.children.size(); ++j) {
            atom_child_batch += "\\\\x";
            atom_child_batch += comp.hash.to_hex();
            atom_child_batch += "\t";
            atom_child_batch += std::to_string(j);
            atom_child_batch += "\tA\t\\\\x";  // 'A' = atom child
            atom_child_batch += comp.children[j].to_hex();
            atom_child_batch += "\n";
        }
    }
    
    std::cerr << "[HIER] Built " << path_to_comp.size() << " compositions with atom children\n";
    
    // Track how many composition children each parent has (for ordinal tracking)
    std::unordered_map<std::string, size_t> parent_child_count;
    
    // Now build parent->child edges (composition -> composition)
    size_t edge_count = 0;
    for (const auto& [path, parent_path] : path_to_parent) {
        auto child_it = path_to_comp.find(path);
        auto parent_it = path_to_comp.find(parent_path);
        
        if (child_it != path_to_comp.end() && parent_it != path_to_comp.end()) {
            // Parent composition -> child composition
            // Format: composition_id, ordinal, child_type, child_id
            // Ordinal offset by parent's atom count so it doesn't collide with atom children
            size_t atom_count = parent_it->second.children.size();  // Number of atoms in parent's path
            size_t comp_ordinal = atom_count + parent_child_count[parent_path];
            parent_child_count[parent_path]++;
            
            child_batch += "\\\\x";
            child_batch += parent_it->second.hash.to_hex();  // parent's id
            child_batch += "\t";
            child_batch += std::to_string(comp_ordinal);  // ordinal offset past atoms
            child_batch += "\tC\t\\\\x";  // 'C' = composition child
            child_batch += child_it->second.hash.to_hex();  // child's id
            child_batch += "\n";
            edge_count++;
        }
    }
    
    std::cerr << "[HIER] Built " << edge_count << " composition->composition edges\n";
    
    // Stream to database
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Create temp table for compositions WITH GEOMETRY
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_hier_comp ("
        "  id BYTEA,"
        "  label TEXT,"
        "  depth INTEGER,"
        "  child_count INTEGER,"
        "  atom_count BIGINT,"
        "  geom GEOMETRY(LINESTRINGZM, 0),"
        "  centroid GEOMETRY(POINTZM, 0),"
        "  hilbert_lo BIGINT,"
        "  hilbert_hi BIGINT"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[HIER] Create temp table failed: " << PQerrorMessage(conn) << "\n";
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // COPY compositions
    res = PQexec(conn, "COPY tmp_hier_comp FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "[HIER] COPY start failed: " << PQerrorMessage(conn) << "\n";
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    if (!comp_batch.empty()) {
        PQputCopyData(conn, comp_batch.c_str(), static_cast<int>(comp_batch.size()));
    }
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);
    
    // Insert into composition table WITH GEOMETRY
    res = PQexec(conn,
        "INSERT INTO composition (id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi) "
        "SELECT id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi FROM tmp_hier_comp "
        "ON CONFLICT (id) DO UPDATE SET "
        "  label = EXCLUDED.label, "
        "  depth = GREATEST(composition.depth, EXCLUDED.depth), "
        "  geom = COALESCE(EXCLUDED.geom, composition.geom), "
        "  centroid = COALESCE(EXCLUDED.centroid, composition.centroid), "
        "  hilbert_lo = COALESCE(EXCLUDED.hilbert_lo, composition.hilbert_lo), "
        "  hilbert_hi = COALESCE(EXCLUDED.hilbert_hi, composition.hilbert_hi)");
    
    int comp_inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[HIER] Insert compositions failed: " << PQerrorMessage(conn) << "\n";
    }
    PQclear(res);
    
    std::cerr << "[HIER] Inserted/updated " << comp_inserted << " hierarchy compositions\n";
    
    // Insert ATOM children FIRST (the characters that make up each hierarchy path)
    // These get ordinals 0..N-1 where N is the path length
    if (!atom_child_batch.empty()) {
        res = PQexec(conn,
            "CREATE TEMP TABLE tmp_hier_atom_child ("
            "  composition_id BYTEA,"
            "  ordinal SMALLINT,"
            "  child_type CHAR(1),"
            "  child_id BYTEA"
            ") ON COMMIT DROP");
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Create atom child temp table failed: " << PQerrorMessage(conn) << "\n";
        }
        PQclear(res);
        
        res = PQexec(conn, "COPY tmp_hier_atom_child FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        PQclear(res);
        
        PQputCopyData(conn, atom_child_batch.c_str(), static_cast<int>(atom_child_batch.size()));
        PQputCopyEnd(conn, nullptr);
        res = PQgetResult(conn);
        PQclear(res);
        
        // Insert atom children
        res = PQexec(conn,
            "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
            "SELECT composition_id, ordinal, child_type, child_id FROM tmp_hier_atom_child "
            "ON CONFLICT (composition_id, ordinal) DO NOTHING");
        
        int atom_edges = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Insert atom children failed: " << PQerrorMessage(conn) << "\n";
        }
        PQclear(res);
        
        std::cerr << "[HIER] Inserted " << atom_edges << " atom children\n";
    }
    
    // Now insert parent->child composition edges
    // These get ordinals N..N+M-1 where N is path length and M is number of sub-compositions
    if (!child_batch.empty()) {
        res = PQexec(conn,
            "CREATE TEMP TABLE tmp_hier_child ("
            "  composition_id BYTEA,"
            "  ordinal SMALLINT,"
            "  child_type CHAR(1),"
            "  child_id BYTEA"
            ") ON COMMIT DROP");
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Create child temp table failed: " << PQerrorMessage(conn) << "\n";
        }
        PQclear(res);
        
        res = PQexec(conn, "COPY tmp_hier_child FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        PQclear(res);
        
        PQputCopyData(conn, child_batch.c_str(), static_cast<int>(child_batch.size()));
        PQputCopyEnd(conn, nullptr);
        res = PQgetResult(conn);
        PQclear(res);
        
        // Insert composition->composition edges using pre-computed ordinals (offset past atoms)
        res = PQexec(conn,
            "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
            "SELECT composition_id, ordinal, child_type, child_id "
            "FROM tmp_hier_child "
            "ON CONFLICT (composition_id, ordinal) DO NOTHING");
        
        int edges_inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Insert composition edges failed: " << PQerrorMessage(conn) << "\n";
        }
        PQclear(res);
        
        std::cerr << "[HIER] Inserted " << edges_inserted << " composition->composition edges\n";
    }
    
    // Update child counts on parent compositions
    res = PQexec(conn,
        "UPDATE composition c SET child_count = sub.cnt "
        "FROM (SELECT composition_id, COUNT(*) as cnt FROM composition_child GROUP BY composition_id) sub "
        "WHERE c.id = sub.composition_id");
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    return true;
}

} // namespace db
} // namespace ingest
} // namespace hypercube
