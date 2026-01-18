/**
 * @file tensor_hierarchy.cpp
 * @brief Build and insert tensor path hierarchy as compositions
 *
 * Parses tensor names into hierarchical path components and inserts them
 * as composition records with atom and composition children.
 */

#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/blake3.hpp"

namespace hypercube {
namespace ingest {
namespace db {

// Helper to escape SQL string literal using libpq (SECURITY: prevents SQL injection)
static std::string escape_sql_literal(PGconn* conn, const std::string& str) {
    char* escaped = PQescapeLiteral(conn, str.c_str(), str.length());
    if (!escaped) {
        // Fallback: double single quotes (basic SQL escaping)
        std::string result = "'";
        for (char c : str) {
            if (c == '\'') result += "''";
            else result += c;
        }
        result += "'";
        return result;
    }
    std::string result(escaped);
    PQfreemem(escaped);
    return result;
}

bool insert_tensor_hierarchy(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    using namespace hypercube::db;

    if (ctx.tensors.empty()) return true;

    std::cerr << "[HIER] Building tensor hierarchy from " << ctx.tensors.size() << " tensors\n";

    // Collect all unique codepoints from tensor names to ensure atoms exist
    std::unordered_set<uint32_t> unique_codepoints;
    for (const auto& [tensor_name, meta] : ctx.tensors) {
        auto codepoints = AtomCalculator::decode_utf8(tensor_name);
        unique_codepoints.insert(codepoints.begin(), codepoints.end());
    }

    std::cerr << "[HIER] Found " << unique_codepoints.size() << " unique codepoints in tensor names\n";

    // VALIDATE atoms exist - FAIL FAST if not seeded
    if (!unique_codepoints.empty()) {
        // Build list of codepoint hashes to check
        std::string hash_list;
        for (uint32_t cp : unique_codepoints) {
            AtomRecord atom = AtomCalculator::compute_atom(cp);
            if (!hash_list.empty()) hash_list += ", ";
            hash_list += "'\\x" + atom.hash.to_hex() + "'";
        }

        std::string check_sql = "SELECT COUNT(*) FROM atom WHERE id IN (" + hash_list + ")";
        Result check_res = exec(conn, check_sql);

        if (!check_res.ok()) {
            std::cerr << "[HIER] Failed to validate atoms: " << check_res.error_message() << "\n";
            return false;
        }

        int64_t found_count = 0;
        if (PQntuples(check_res.get()) > 0) {
            const char* val = PQgetvalue(check_res.get(), 0, 0);
            if (val) found_count = std::strtoll(val, nullptr, 10);
        }

        if (found_count != static_cast<int64_t>(unique_codepoints.size())) {
            std::cerr << "[HIER] FATAL: Only " << found_count << "/" << unique_codepoints.size()
                      << " atoms exist in database!\n";
            std::cerr << "[HIER] Run seed_atoms_parallel before model ingestion.\n";
            return false;
        }

        std::cerr << "[HIER] Validated " << found_count << " atoms exist\n";
    }
    
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
    
    // Direct bulk insert to database with Transaction RAII
    Transaction tx(conn);

    // Parse comp_batch and build direct INSERT for compositions
    std::string insert_comp_sql = "INSERT INTO composition (id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi) VALUES ";
    std::vector<std::string> comp_values;
    std::istringstream comp_iss(comp_batch);
    std::string comp_line;
    while (std::getline(comp_iss, comp_line)) {
        if (comp_line.empty()) continue;
        std::istringstream line_ss(comp_line);
        std::string id_hex, label, depth_str, child_count_str, atom_count_str, geom_ewkb, centroid_ewkb, hilbert_lo_str, hilbert_hi_str;
        if (!(line_ss >> id_hex >> label >> depth_str >> child_count_str >> atom_count_str >> geom_ewkb >> centroid_ewkb >> hilbert_lo_str >> hilbert_hi_str)) continue;

        // Fix double-escaped hex: \\x -> \x for SQL INSERT
        if (id_hex.substr(0, 3) == "\\\\x") id_hex = "\\" + id_hex.substr(2);
        if (geom_ewkb.substr(0, 3) == "\\\\x") geom_ewkb = "\\" + geom_ewkb.substr(2);
        if (centroid_ewkb.substr(0, 3) == "\\\\x") centroid_ewkb = "\\" + centroid_ewkb.substr(2);

        // Handle \N for null geom
        std::string geom_val = (geom_ewkb == "\\N") ? "NULL" : ("'" + geom_ewkb + "'");

        // SECURITY FIX: Escape label to prevent SQL injection
        // Labels are tensor paths which should only contain [a-zA-Z0-9._] but we escape for safety
        std::string escaped_label = escape_sql_literal(conn, label);

        std::string val = "('" + id_hex + "', " + escaped_label + ", " + depth_str + ", " + child_count_str + ", " + atom_count_str +
                          ", " + geom_val + ", '" + centroid_ewkb + "', " + hilbert_lo_str + ", " + hilbert_hi_str + ")";
        comp_values.push_back(val);
    }

    // Batch compositions insert
    const size_t batch_size = 100;
    int comp_inserted = 0;
    for (size_t i = 0; i < comp_values.size(); i += batch_size) {
        std::string batch_sql = insert_comp_sql;
        for (size_t j = i; j < std::min(i + batch_size, comp_values.size()); ++j) {
            if (j > i) batch_sql += ", ";
            batch_sql += comp_values[j];
        }
        batch_sql += " ON CONFLICT (id) DO UPDATE SET "
                    "  label = EXCLUDED.label, "
                    "  depth = GREATEST(composition.depth, EXCLUDED.depth), "
                    "  geom = COALESCE(EXCLUDED.geom, composition.geom), "
                    "  centroid = COALESCE(EXCLUDED.centroid, composition.centroid), "
                    "  hilbert_lo = COALESCE(EXCLUDED.hilbert_lo, composition.hilbert_lo), "
                    "  hilbert_hi = COALESCE(EXCLUDED.hilbert_hi, composition.hilbert_hi)";

        Result res = exec(conn, batch_sql);
        if (res.ok()) {
            comp_inserted += cmd_tuples(res);
        } else {
            std::cerr << "[HIER] Batch insert compositions failed: " << res.error_message() << "\n";
        }
    }
    
    std::cerr << "[HIER] Inserted/updated " << comp_inserted << " hierarchy compositions\n";
    
    // COPY atom children to temp table, then INSERT with existence checks
    if (!atom_child_batch.empty()) {
        Result drop_res = exec(conn, "DROP TABLE IF EXISTS tmp_hier_atom_child");
        if (!drop_res.ok()) {
            std::cerr << "[HIER] Failed to drop tmp_hier_atom_child: " << drop_res.error_message() << "\n";
        }
        Result create_res = exec(conn, "CREATE TEMP TABLE tmp_hier_atom_child ("
                   "composition_id BYTEA, ordinal SMALLINT, child_type CHAR(1), child_id BYTEA)");
        if (!create_res.ok()) {
            std::cerr << "[HIER] Failed to create tmp_hier_atom_child: " << create_res.error_message() << "\n";
            return false;
        }

        Result copy_res = exec(conn, "COPY tmp_hier_atom_child FROM STDIN");
        if (!copy_res.ok()) {
            std::cerr << "[HIER] COPY start failed: " << copy_res.error_message() << "\n";
            return false;
        }
        PQputCopyData(conn, atom_child_batch.c_str(), static_cast<int>(atom_child_batch.size()));
        int copy_end_result = PQputCopyEnd(conn, nullptr);
        if (copy_end_result != 1) {
            std::cerr << "[HIER] PQputCopyEnd failed: " << PQerrorMessage(conn) << "\n";
            return false;
        }
        PGresult* r = PQgetResult(conn);
        if (PQresultStatus(r) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Atom child COPY failed: " << PQerrorMessage(conn) << "\n";
            PQclear(r);
            return false;
        }
        PQclear(r);

        Result ins_res = exec(conn,
            "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
            "SELECT composition_id, ordinal, child_type, child_id FROM tmp_hier_atom_child t "
            "WHERE EXISTS (SELECT 1 FROM composition WHERE id = t.composition_id) "
            "AND EXISTS (SELECT 1 FROM atom WHERE id = t.child_id) "
            "ON CONFLICT (composition_id, ordinal) DO NOTHING");

        int atom_edges = ins_res.ok() ? cmd_tuples(ins_res) : 0;
        if (!ins_res.ok()) {
            std::cerr << "[HIER] Atom children insert failed: " << ins_res.error_message() << "\n";
            return false;
        }
        // Calculate expected atom children count
        size_t expected_atom_edges = 0;
        for (const auto& [path, comp] : path_to_comp) {
            expected_atom_edges += comp.children.size();
        }
        if (atom_edges == 0 && expected_atom_edges > 0) {
            std::cerr << "[HIER] WARNING: No atom children inserted (expected " << expected_atom_edges << "), possible data integrity issue\n";
        } else {
            std::cerr << "[HIER] Inserted " << atom_edges << " atom children\n";
        }
    }

    // COPY composition children to temp table, then INSERT with existence checks
    if (!child_batch.empty()) {
        Result drop_res = exec(conn, "DROP TABLE IF EXISTS tmp_hier_comp_child");
        if (!drop_res.ok()) {
            std::cerr << "[HIER] Failed to drop tmp_hier_comp_child: " << drop_res.error_message() << "\n";
        }
        Result create_res = exec(conn, "CREATE TEMP TABLE tmp_hier_comp_child ("
                   "composition_id BYTEA, ordinal SMALLINT, child_type CHAR(1), child_id BYTEA)");
        if (!create_res.ok()) {
            std::cerr << "[HIER] Failed to create tmp_hier_comp_child: " << create_res.error_message() << "\n";
            return false;
        }

        Result copy_res = exec(conn, "COPY tmp_hier_comp_child FROM STDIN");
        if (!copy_res.ok()) {
            std::cerr << "[HIER] COPY start failed: " << copy_res.error_message() << "\n";
            return false;
        }
        PQputCopyData(conn, child_batch.c_str(), static_cast<int>(child_batch.size()));
        int copy_end_result = PQputCopyEnd(conn, nullptr);
        if (copy_end_result != 1) {
            std::cerr << "[HIER] PQputCopyEnd failed: " << PQerrorMessage(conn) << "\n";
            return false;
        }
        PGresult* r = PQgetResult(conn);
        if (PQresultStatus(r) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Comp child COPY failed: " << PQerrorMessage(conn) << "\n";
            PQclear(r);
            return false;
        }
        PQclear(r);

        Result ins_res = exec(conn,
            "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
            "SELECT composition_id, ordinal, child_type, child_id FROM tmp_hier_comp_child t "
            "WHERE EXISTS (SELECT 1 FROM composition WHERE id = t.composition_id) "
            "AND EXISTS (SELECT 1 FROM composition WHERE id = t.child_id) "
            "ON CONFLICT (composition_id, ordinal) DO NOTHING");

        int edges_inserted = ins_res.ok() ? cmd_tuples(ins_res) : 0;
        if (!ins_res.ok()) {
            std::cerr << "[HIER] Comp children insert failed: " << ins_res.error_message() << "\n";
            return false;
        }
        // Expected edges is edge_count calculated earlier
        if (edges_inserted == 0 && edge_count > 0) {
            std::cerr << "[HIER] WARNING: No composition->composition edges inserted (expected " << edge_count << "), possible data integrity issue\n";
        } else {
            std::cerr << "[HIER] Inserted " << edges_inserted << " composition->composition edges\n";
        }
    }
    
    // Update child counts ONLY for compositions modified in this batch
    // This is needed because tensor hierarchy adds both atom children AND composition children,
    // and the initial child_count only accounts for atom children.
    // Using temp table to limit scope rather than scanning all composition_child.
    if (!path_to_comp.empty()) {
        // Build list of composition IDs we just modified
        std::string id_list;
        for (const auto& [path, comp] : path_to_comp) {
            if (!id_list.empty()) id_list += ", ";
            id_list += "'\\x" + comp.hash.to_hex() + "'";
        }

        std::string update_sql =
            "UPDATE composition c SET child_count = sub.cnt "
            "FROM (SELECT composition_id, COUNT(*) as cnt FROM composition_child "
            "      WHERE composition_id IN (" + id_list + ") "
            "      GROUP BY composition_id) sub "
            "WHERE c.id = sub.composition_id AND c.child_count != sub.cnt";

        Result update_res = exec(conn, update_sql);
        if (update_res.ok()) {
            int updated = cmd_tuples(update_res);
            if (updated > 0) {
                std::cerr << "[HIER] Updated child_count for " << updated << " compositions\n";
            }
        }
    }

    tx.commit();
    
    return true;
}

} // namespace db
} // namespace ingest
} // namespace hypercube