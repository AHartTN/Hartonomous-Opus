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

namespace hypercube {
namespace ingest {
namespace db {

bool insert_tensor_hierarchy(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    using namespace hypercube::db;
    
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

        // Handle \N for null geom
        std::string geom_val = (geom_ewkb == "\\N") ? "NULL" : ("'" + geom_ewkb + "'");

        std::string val = "('" + id_hex + "', '" + label + "', " + depth_str + ", " + child_count_str + ", " + atom_count_str +
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
    
    // Direct insert ATOM children
    if (!atom_child_batch.empty()) {
        std::string insert_atom_sql = "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) VALUES ";
        std::vector<std::string> atom_values;
        std::istringstream atom_iss(atom_child_batch);
        std::string atom_line;
        while (std::getline(atom_iss, atom_line)) {
            if (atom_line.empty()) continue;
            std::istringstream line_ss(atom_line);
            std::string comp_id_hex, ordinal_str, child_type, child_id_hex;
            if (!(line_ss >> comp_id_hex >> ordinal_str >> child_type >> child_id_hex)) continue;

            std::string val = "('" + comp_id_hex + "', " + ordinal_str + ", '" + child_type + "', '" + child_id_hex + "')";
            atom_values.push_back(val);
        }

        // Batch atom children insert
        int atom_edges = 0;
        for (size_t i = 0; i < atom_values.size(); i += batch_size) {
            std::string batch_sql = insert_atom_sql;
            for (size_t j = i; j < std::min(i + batch_size, atom_values.size()); ++j) {
                if (j > i) batch_sql += ", ";
                batch_sql += atom_values[j];
            }
            batch_sql += " ON CONFLICT (composition_id, ordinal) DO NOTHING";

            Result res = exec(conn, batch_sql);
            if (res.ok()) {
                atom_edges += cmd_tuples(res);
            } else {
                std::cerr << "[HIER] Batch insert atom children failed: " << res.error_message() << "\n";
            }
        }

        std::cerr << "[HIER] Inserted " << atom_edges << " atom children\n";
    }

    // Direct insert parent->child composition edges
    if (!child_batch.empty()) {
        std::string insert_child_sql = "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) VALUES ";
        std::vector<std::string> child_values;
        std::istringstream child_iss(child_batch);
        std::string child_line;
        while (std::getline(child_iss, child_line)) {
            if (child_line.empty()) continue;
            std::istringstream line_ss(child_line);
            std::string comp_id_hex, ordinal_str, child_type, child_id_hex;
            if (!(line_ss >> comp_id_hex >> ordinal_str >> child_type >> child_id_hex)) continue;

            std::string val = "('" + comp_id_hex + "', " + ordinal_str + ", '" + child_type + "', '" + child_id_hex + "')";
            child_values.push_back(val);
        }

        // Batch composition edges insert
        int edges_inserted = 0;
        for (size_t i = 0; i < child_values.size(); i += batch_size) {
            std::string batch_sql = insert_child_sql;
            for (size_t j = i; j < std::min(i + batch_size, child_values.size()); ++j) {
                if (j > i) batch_sql += ", ";
                batch_sql += child_values[j];
            }
            batch_sql += " ON CONFLICT (composition_id, ordinal) DO NOTHING";

            Result res = exec(conn, batch_sql);
            if (res.ok()) {
                edges_inserted += cmd_tuples(res);
            } else {
                std::cerr << "[HIER] Batch insert composition edges failed: " << res.error_message() << "\n";
            }
        }

        std::cerr << "[HIER] Inserted " << edges_inserted << " composition->composition edges\n";
    }
    
    // Update child counts on parent compositions
    exec(conn,
        "UPDATE composition c SET child_count = sub.cnt "
        "FROM (SELECT composition_id, COUNT(*) as cnt FROM composition_child GROUP BY composition_id) sub "
        "WHERE c.id = sub.composition_id");
    
    tx.commit();
    
    return true;
}

} // namespace db
} // namespace ingest
} // namespace hypercube