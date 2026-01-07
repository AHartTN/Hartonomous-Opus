// =============================================================================
// metadata_db.hpp - Database Insertion for Model Metadata
// =============================================================================
// Ingests model metadata as SEMANTIC STRUCTURE - Merkle DAG + AST.
//
// Core principle: Structure IS semantics. A config file is a TREE, not a
// flattened list of key-value pairs with dot-notation paths.
//
// Config JSON → AST where each node is a composition:
//   - Object nodes contain key nodes (H-relation: hierarchy/contains)
//   - Key nodes link to value nodes (V-relation: has_value)
//   - Array nodes contain indexed children (H-relation with ordinal)
//   - Leaf values are compositions from their UTF-8 content
//
// Tokenizer → BPE merge tree (M-relation: merge/compose):
//   - Each token is a composition from its UTF-8 codepoints
//   - BPE merges define parent→child composition relations
//   - This IS the cascading n-gram Merkle DAG
//
// Future: TreeSitter-style parsing for everything - English grammar,
// German cases, mathematical notation, physics equations, etc.
// =============================================================================

#pragma once

#include <libpq-fe.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cstring>

#include "hypercube/ingest/metadata.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/atom_cache.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/blake3.hpp"

namespace hypercube {
namespace ingest {
namespace metadata {

// =============================================================================
// Relation Types for Semantic Structure
// =============================================================================
// H = Hierarchy (parent contains child - AST structure)
// V = Value (key has value)
// M = Merge (BPE: components merge to form token)
// S = Sequence (ordered adjacency)
// T = Type (node has type annotation)

constexpr char REL_HIERARCHY = 'H';
constexpr char REL_VALUE = 'V';
constexpr char REL_MERGE = 'M';
constexpr char REL_SEQUENCE = 'S';
constexpr char REL_TYPE = 'T';

// =============================================================================
// Hash Helpers - Using Blake3Hasher class
// =============================================================================

inline Blake3Hash hash_string(const std::string& s) {
    return Blake3Hasher::hash(s);
}

inline Blake3Hash hash_composition(const std::vector<Blake3Hash>& children) {
    // MUST use ordered hash for consistency with AtomCalculator::compute_composition
    return Blake3Hasher::hash_children_ordered(std::span<const Blake3Hash>(children));
}

// =============================================================================
// Extract codepoints from UTF-8 string
// =============================================================================

inline std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> result;
    result.reserve(s.size());
    
    for (size_t i = 0; i < s.size(); ) {
        uint32_t cp = 0;
        unsigned char c = s[i];
        
        if ((c & 0x80) == 0) {
            cp = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = (c & 0x1F) << 6;
            if (i + 1 < s.size()) cp |= (s[i + 1] & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = (c & 0x0F) << 12;
            if (i + 1 < s.size()) cp |= (s[i + 1] & 0x3F) << 6;
            if (i + 2 < s.size()) cp |= (s[i + 2] & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = (c & 0x07) << 18;
            if (i + 1 < s.size()) cp |= (s[i + 1] & 0x3F) << 12;
            if (i + 2 < s.size()) cp |= (s[i + 2] & 0x3F) << 6;
            if (i + 3 < s.size()) cp |= (s[i + 3] & 0x3F);
            i += 4;
        } else {
            i += 1;  // Invalid UTF-8, skip
            continue;
        }
        result.push_back(cp);
    }
    return result;
}

// =============================================================================
// CompositionData - Internal tracking for built compositions
// =============================================================================

struct CompositionData {
    Blake3Hash hash;
    std::string label;
    int depth = 1;
    std::vector<Blake3Hash> child_hashes;
    std::vector<char> child_types;  // 'A' = atom, 'C' = composition
    size_t atom_count = 0;
    double cx = 0, cy = 0, cz = 0, cm = 0;
};

// =============================================================================
// CompositionBuilder - Build compositions with geometry from atom cache
// =============================================================================

class CompositionBuilder {
public:
    CompositionBuilder(const std::unordered_map<uint32_t, hypercube::db::AtomInfo>& atom_cache)
        : atom_cache_(atom_cache) {}
    
    // Build a composition from a string, returns hash
    Blake3Hash build_from_string(const std::string& text, int depth = 1) {
        std::vector<uint32_t> cps = utf8_to_codepoints(text);
        if (cps.empty()) return Blake3Hash{};
        
        std::vector<Blake3Hash> child_hashes;
        double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
        int valid_coords = 0;
        
        for (uint32_t cp : cps) {
            auto it = atom_cache_.find(cp);
            if (it != atom_cache_.end()) {
                child_hashes.push_back(it->second.hash);
                sum_x += it->second.coord_x;
                sum_y += it->second.coord_y;
                sum_z += it->second.coord_z;
                sum_m += it->second.coord_m;
                valid_coords++;
            }
        }
        
        if (child_hashes.empty()) return Blake3Hash{};
        
        Blake3Hash hash = hash_composition(child_hashes);
        
        // Check if already built
        if (built_compositions_.count(hash)) {
            return hash;
        }
        
        CompositionData data;
        data.hash = hash;
        data.label = text;
        data.depth = depth;
        data.child_hashes = std::move(child_hashes);
        data.child_types.resize(data.child_hashes.size(), 'A');  // All atoms
        data.atom_count = cps.size();
        data.cx = valid_coords > 0 ? sum_x / valid_coords : 0;
        data.cy = valid_coords > 0 ? sum_y / valid_coords : 0;
        data.cz = valid_coords > 0 ? sum_z / valid_coords : 0;
        data.cm = valid_coords > 0 ? sum_m / valid_coords : 0;
        
        built_compositions_[hash] = std::move(data);
        return hash;
    }
    
    // Build a composition from child compositions (for AST nodes)
    Blake3Hash build_from_children(const std::string& label,
                                   const std::vector<Blake3Hash>& children,
                                   const std::vector<char>& child_types,
                                   int depth) {
        if (children.empty()) return Blake3Hash{};
        
        Blake3Hash hash = hash_composition(children);
        
        if (built_compositions_.count(hash)) {
            return hash;
        }
        
        // Compute centroid from child compositions
        double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
        int valid = 0;
        size_t total_atoms = 0;
        
        for (const auto& child_hash : children) {
            auto it = built_compositions_.find(child_hash);
            if (it != built_compositions_.end()) {
                sum_x += it->second.cx;
                sum_y += it->second.cy;
                sum_z += it->second.cz;
                sum_m += it->second.cm;
                total_atoms += it->second.atom_count;
                valid++;
            }
        }
        
        CompositionData data;
        data.hash = hash;
        data.label = label;
        data.depth = depth;
        data.child_hashes = children;
        data.child_types = child_types;
        data.atom_count = total_atoms;
        data.cx = valid > 0 ? sum_x / valid : 0;
        data.cy = valid > 0 ? sum_y / valid : 0;
        data.cz = valid > 0 ? sum_z / valid : 0;
        data.cm = valid > 0 ? sum_m / valid : 0;
        
        built_compositions_[hash] = std::move(data);
        return hash;
    }
    
    // Get composition data by hash
    const CompositionData* get(const Blake3Hash& hash) const {
        auto it = built_compositions_.find(hash);
        return it != built_compositions_.end() ? &it->second : nullptr;
    }
    
    bool has(const Blake3Hash& hash) const {
        return built_compositions_.count(hash) > 0;
    }
    
    const auto& all() const { return built_compositions_; }
    size_t count() const { return built_compositions_.size(); }
    
private:
    const std::unordered_map<uint32_t, hypercube::db::AtomInfo>& atom_cache_;
    std::unordered_map<Blake3Hash, CompositionData, Blake3HashHasher> built_compositions_;
};

// =============================================================================
// RelationData - Internal tracking for built relations
// =============================================================================

struct RelationData {
    char source_type;
    Blake3Hash source;
    char target_type;
    Blake3Hash target;
    char relation_type;
    double weight;
    std::string model;
    int layer;
    std::string component;
};

// =============================================================================
// RelationBuilder - Build semantic relations
// =============================================================================

class RelationBuilder {
public:
    void add(char source_type, const Blake3Hash& source,
             char target_type, const Blake3Hash& target,
             char relation_type, double weight,
             const std::string& model, int layer, const std::string& component) {
        relations_.push_back({source_type, source, target_type, target,
                             relation_type, weight, model, layer, component});
    }
    
    // H-relation: parent contains child (AST hierarchy)
    void add_hierarchy(const Blake3Hash& parent, const Blake3Hash& child,
                       const std::string& model) {
        add('C', parent, 'C', child, REL_HIERARCHY, 1.0, model, -1, "config");
    }
    
    // V-relation: key has value
    void add_value(const Blake3Hash& key, const Blake3Hash& value,
                   const std::string& model) {
        add('C', key, 'C', value, REL_VALUE, 1.0, model, -1, "config");
    }
    
    // M-relation: components merge to form token (BPE)
    void add_merge(const Blake3Hash& component, const Blake3Hash& merged,
                   double priority_weight, const std::string& model) {
        add('C', component, 'C', merged, REL_MERGE, priority_weight, model, -1, "bpe");
    }
    
    // S-relation: sequential adjacency
    void add_sequence(const Blake3Hash& prev, const Blake3Hash& next,
                      const std::string& model) {
        add('C', prev, 'C', next, REL_SEQUENCE, 1.0, model, -1, "sequence");
    }
    
    const std::vector<RelationData>& all() const { return relations_; }
    size_t count() const { return relations_.size(); }
    
private:
    std::vector<RelationData> relations_;
};

// =============================================================================
// Config AST Builder - Parse config atoms into tree structure
// =============================================================================

inline void build_config_ast(const std::vector<MetadataAtom>& atoms,
                            const std::string& model_name,
                            CompositionBuilder& comp_builder,
                            RelationBuilder& rel_builder) {
    // Track path segments we've already built
    std::unordered_map<std::string, Blake3Hash> path_hashes;
    
    // First pass: build all path segment compositions
    for (const auto& atom : atoms) {
        // Split path into segments
        std::vector<std::string> segments;
        std::string current;
        for (char c : atom.path) {
            if (c == '.') {
                if (!current.empty()) {
                    segments.push_back(current);
                    current.clear();
                }
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            segments.push_back(current);
        }
        
        // Build composition for each segment
        std::string accumulated_path;
        for (size_t i = 0; i < segments.size(); ++i) {
            if (i > 0) accumulated_path += ".";
            accumulated_path += segments[i];
            
            if (!path_hashes.count(accumulated_path)) {
                // Build the segment name as a composition
                Blake3Hash seg_hash = comp_builder.build_from_string(segments[i], 1);
                if (seg_hash != Blake3Hash{}) {
                    path_hashes[accumulated_path] = seg_hash;
                }
            }
        }
        
        // Build composition for the value
        if (!atom.value.empty()) {
            Blake3Hash value_hash = comp_builder.build_from_string(atom.value, 1);
            if (value_hash != Blake3Hash{}) {
                path_hashes[atom.path + "=" + atom.value] = value_hash;
            }
        }
    }
    
    // Second pass: build hierarchy relations
    for (const auto& atom : atoms) {
        std::vector<std::string> segments;
        std::string current;
        for (char c : atom.path) {
            if (c == '.') {
                if (!current.empty()) {
                    segments.push_back(current);
                    current.clear();
                }
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            segments.push_back(current);
        }
        
        // Create H-relations between parent and child segments
        std::string parent_path;
        std::string child_path;
        for (size_t i = 0; i < segments.size(); ++i) {
            if (i > 0) child_path += ".";
            child_path += segments[i];
            
            if (i > 0 && path_hashes.count(parent_path) && path_hashes.count(child_path)) {
                rel_builder.add_hierarchy(path_hashes[parent_path], path_hashes[child_path], model_name);
            }
            
            parent_path = child_path;
        }
        
        // Create V-relation from key to value
        if (!atom.value.empty()) {
            std::string value_key = atom.path + "=" + atom.value;
            if (path_hashes.count(atom.path) && path_hashes.count(value_key)) {
                rel_builder.add_value(path_hashes[atom.path], path_hashes[value_key], model_name);
            }
        }
    }
    
    std::cerr << "[CONFIG-AST] Built " << path_hashes.size() << " nodes, "
              << rel_builder.count() << " relations\n";
}

// =============================================================================
// BPE Merge Tree Builder
// =============================================================================

inline void build_bpe_tree(const std::vector<VocabToken>& vocab,
                          const std::vector<BPEMerge>& merges,
                          const std::string& model_name,
                          CompositionBuilder& comp_builder,
                          RelationBuilder& rel_builder) {
    // Build all vocab tokens as compositions
    std::unordered_map<std::string, Blake3Hash> token_hashes;
    
    for (const auto& vt : vocab) {
        Blake3Hash hash = comp_builder.build_from_string(vt.text, 1);
        if (hash != Blake3Hash{}) {
            token_hashes[vt.text] = hash;
        }
    }
    
    std::cerr << "[BPE-TREE] Built " << token_hashes.size() << " token compositions\n";
    
    // Build merge relations
    size_t merge_count = 0;
    for (const auto& merge : merges) {
        auto it_a = token_hashes.find(merge.token_a);
        auto it_b = token_hashes.find(merge.token_b);
        auto it_merged = token_hashes.find(merge.merged);
        
        if (it_a == token_hashes.end() || it_b == token_hashes.end()) continue;
        
        // If merged token doesn't exist yet, create it as a composition of a and b
        Blake3Hash merged_hash;
        if (it_merged == token_hashes.end()) {
            std::vector<Blake3Hash> children = {it_a->second, it_b->second};
            std::vector<char> types = {'C', 'C'};
            merged_hash = comp_builder.build_from_children(merge.merged, children, types, 2);
            if (merged_hash != Blake3Hash{}) {
                token_hashes[merge.merged] = merged_hash;
            }
        } else {
            merged_hash = it_merged->second;
        }
        
        if (merged_hash != Blake3Hash{}) {
            // Weight: higher priority merges (lower number) get higher weight
            double weight = 1.0 / (merge.priority + 1);
            rel_builder.add_merge(it_a->second, merged_hash, weight, model_name);
            rel_builder.add_merge(it_b->second, merged_hash, weight, model_name);
            merge_count += 2;
        }
    }
    
    std::cerr << "[BPE-TREE] Built " << merge_count << " merge relations\n";
}

// =============================================================================
// Special Tokens Builder
// =============================================================================

inline void build_special_tokens(const std::vector<SpecialToken>& tokens,
                                 const std::string& model_name,
                                 CompositionBuilder& comp_builder,
                                 RelationBuilder& rel_builder) {
    for (const auto& st : tokens) {
        Blake3Hash hash = comp_builder.build_from_string(st.content, 1);
        if (hash == Blake3Hash{}) continue;
        
        // Build the role as a composition too
        Blake3Hash role_hash = comp_builder.build_from_string(st.role, 1);
        if (role_hash != Blake3Hash{}) {
            // T-relation: token has type/role
            rel_builder.add('C', hash, 'C', role_hash, REL_TYPE, 1.0, model_name, -1, "special");
        }
    }
    
    std::cerr << "[SPECIAL] Built " << tokens.size() << " special token nodes\n";
}

// =============================================================================
// Stream to Database
// =============================================================================

inline bool stream_to_database(PGconn* conn,
                               const CompositionBuilder& comp_builder,
                               const RelationBuilder& rel_builder) {
    using namespace hypercube::db;

    // Drop problematic index before bulk operations to prevent corruption
    std::cerr << "[STREAM] Dropping idx_comp_label to prevent corruption...\n";
    exec(conn, "DROP INDEX IF EXISTS idx_comp_label");

    // Build COPY buffers
    std::string comp_buffer;
    std::string child_buffer;
    std::string rel_buffer;
    comp_buffer.reserve(32 * 1024 * 1024);
    child_buffer.reserve(64 * 1024 * 1024);
    rel_buffer.reserve(32 * 1024 * 1024);
    
    // Compositions
    for (const auto& pair : comp_builder.all()) {
        const CompositionData& data = pair.second;
        
        // id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi
        copy_bytea(comp_buffer, data.hash);
        copy_tab(comp_buffer);
        copy_escape(comp_buffer, data.label);
        copy_tab(comp_buffer);
        comp_buffer += std::to_string(data.depth);
        copy_tab(comp_buffer);
        comp_buffer += std::to_string(data.child_hashes.size());
        copy_tab(comp_buffer);
        comp_buffer += std::to_string(data.atom_count);
        copy_tab(comp_buffer);
        copy_null(comp_buffer);  // geom
        copy_tab(comp_buffer);
        comp_buffer += "POINT ZM(" + std::to_string(data.cx) + " " + 
                       std::to_string(data.cy) + " " + std::to_string(data.cz) + " " +
                       std::to_string(data.cm) + ")";
        copy_tab(comp_buffer);
        comp_buffer += "0";  // hilbert_lo
        copy_tab(comp_buffer);
        comp_buffer += "0";  // hilbert_hi
        copy_newline(comp_buffer);
        
        // Children
        for (size_t i = 0; i < data.child_hashes.size(); ++i) {
            copy_bytea(child_buffer, data.hash);
            copy_tab(child_buffer);
            child_buffer += std::to_string(i);
            copy_tab(child_buffer);
            child_buffer += data.child_types[i];
            copy_tab(child_buffer);
            copy_bytea(child_buffer, data.child_hashes[i]);
            copy_newline(child_buffer);
        }
    }
    
    // Relations
    for (const auto& rel : rel_builder.all()) {
        rel_buffer += rel.source_type;
        copy_tab(rel_buffer);
        copy_bytea(rel_buffer, rel.source);
        copy_tab(rel_buffer);
        rel_buffer += rel.target_type;
        copy_tab(rel_buffer);
        copy_bytea(rel_buffer, rel.target);
        copy_tab(rel_buffer);
        rel_buffer += rel.relation_type;
        copy_tab(rel_buffer);
        rel_buffer += std::to_string(rel.weight);
        copy_tab(rel_buffer);
        copy_escape(rel_buffer, rel.model);
        copy_tab(rel_buffer);
        rel_buffer += "1";  // source_count
        copy_tab(rel_buffer);
        rel_buffer += std::to_string(rel.layer);
        copy_tab(rel_buffer);
        copy_escape(rel_buffer, rel.component);
        copy_newline(rel_buffer);
    }
    
    std::cerr << "[STREAM] Buffers: comp=" << (comp_buffer.size()/1024) << "KB, "
              << "child=" << (child_buffer.size()/1024) << "KB, "
              << "rel=" << (rel_buffer.size()/1024) << "KB\n";
    
    // Create temp tables
    if (!create_temp_table(conn, "tmp_meta_comp", schema::composition())) {
        std::cerr << "[STREAM] Failed to create tmp_meta_comp\n";
        return false;
    }
    if (!create_temp_table(conn, "tmp_meta_child", schema::composition_child())) {
        std::cerr << "[STREAM] Failed to create tmp_meta_child\n";
        return false;
    }
    if (!create_temp_table(conn, "tmp_meta_rel", schema::relation())) {
        std::cerr << "[STREAM] Failed to create tmp_meta_rel\n";
        return false;
    }
    
    // COPY compositions
    {
        CopyStream copy(conn, "COPY tmp_meta_comp FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        if (!copy.ok()) {
            std::cerr << "[STREAM] COPY comp start failed: " << copy.error() << "\n";
            return false;
        }
        if (!comp_buffer.empty() && !copy.put(comp_buffer)) {
            std::cerr << "[STREAM] COPY comp failed: " << copy.error() << "\n";
            return false;
        }
        if (!copy.end()) {
            std::cerr << "[STREAM] COPY comp end failed: " << copy.error() << "\n";
            return false;
        }
    }
    
    // COPY children
    {
        CopyStream copy(conn, "COPY tmp_meta_child FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        if (!copy.ok()) {
            std::cerr << "[STREAM] COPY child start failed: " << copy.error() << "\n";
            return false;
        }
        if (!child_buffer.empty() && !copy.put(child_buffer)) {
            std::cerr << "[STREAM] COPY child failed: " << copy.error() << "\n";
            return false;
        }
        if (!copy.end()) {
            std::cerr << "[STREAM] COPY child end failed: " << copy.error() << "\n";
            return false;
        }
    }
    
    // COPY relations
    {
        CopyStream copy(conn, "COPY tmp_meta_rel FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        if (!copy.ok()) {
            std::cerr << "[STREAM] COPY rel start failed: " << copy.error() << "\n";
            return false;
        }
        if (!rel_buffer.empty() && !copy.put(rel_buffer)) {
            std::cerr << "[STREAM] COPY rel failed: " << copy.error() << "\n";
            return false;
        }
        if (!copy.end()) {
            std::cerr << "[STREAM] COPY rel end failed: " << copy.error() << "\n";
            return false;
        }
    }
    
    // Merge into main tables
    std::cerr << "[STREAM] Merging into main tables...\n";
    
    Result res = exec(conn,
        "INSERT INTO composition (id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi) "
        "SELECT id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi "
        "FROM tmp_meta_comp "
        "ON CONFLICT (id) DO UPDATE SET "
        "  label = COALESCE(composition.label, EXCLUDED.label), "
        "  centroid = COALESCE(composition.centroid, EXCLUDED.centroid)");
    
    if (!res.ok()) {
        std::cerr << "[STREAM] Composition insert failed: " << res.error_message() << "\n";
        return false;
    }
    int inserted_comps = cmd_tuples(res);
    
    res = exec(conn,
        "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
        "SELECT composition_id, ordinal, child_type, child_id "
        "FROM tmp_meta_child "
        "WHERE EXISTS (SELECT 1 FROM composition WHERE id = tmp_meta_child.composition_id) "
        "ON CONFLICT (composition_id, ordinal) DO NOTHING");
    
    if (!res.ok()) {
        std::cerr << "[STREAM] Child insert failed: " << res.error_message() << "\n";
        return false;
    }
    int inserted_children = cmd_tuples(res);
    
    // Deduplicate relations before insert (BPE merges can produce duplicate A↔B edges)
    res = exec(conn,
        "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
        "SELECT DISTINCT ON (source_id, target_id, relation_type, source_model, layer, component) "
        "       source_type, source_id, target_type, target_id, relation_type, "
        "       AVG(weight) OVER (PARTITION BY source_id, target_id, relation_type, source_model, layer, component), "
        "       source_model, source_count, layer, component "
        "FROM tmp_meta_rel "
        "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
        "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
        "  source_count = relation.source_count + 1");
    
    if (!res.ok()) {
        std::cerr << "[STREAM] Relation insert failed: " << res.error_message() << "\n";
        return false;
    }
    int inserted_rels = cmd_tuples(res);
    
    std::cerr << "[STREAM] Inserted: " << inserted_comps << " compositions, "
              << inserted_children << " children, " << inserted_rels << " relations\n";

    // Recreate the index after bulk operations
    std::cerr << "[STREAM] Recreating idx_comp_label...\n";
    exec(conn, "CREATE INDEX idx_comp_label ON composition(label)");

    return true;
}

// =============================================================================
// Main Entry Point - Insert all metadata for a model
// =============================================================================

inline bool insert_model_metadata(PGconn* conn, const ModelMetadata& meta) {
    using namespace hypercube::db;
    
    std::cerr << "\n=== Inserting Model Metadata: " << meta.model_name << " ===\n";
    std::cerr << "[METADATA] Type: " << meta.model_type << "\n";
    std::cerr << "[METADATA] Config atoms: " << meta.config_atoms.size() << "\n";
    std::cerr << "[METADATA] BPE merges: " << meta.bpe_merges.size() << "\n";
    std::cerr << "[METADATA] Special tokens: " << meta.special_tokens.size() << "\n";
    std::cerr << "[METADATA] Vocab tokens: " << meta.vocab_tokens.size() << "\n";
    
    // Collect all codepoints needed
    std::unordered_set<uint32_t> needed_codepoints;
    
    // From vocab tokens
    for (const auto& vt : meta.vocab_tokens) {
        for (uint32_t cp : utf8_to_codepoints(vt.text)) {
            needed_codepoints.insert(cp);
        }
    }
    // From special tokens
    for (const auto& st : meta.special_tokens) {
        for (uint32_t cp : utf8_to_codepoints(st.content)) {
            needed_codepoints.insert(cp);
        }
        for (uint32_t cp : utf8_to_codepoints(st.role)) {
            needed_codepoints.insert(cp);
        }
    }
    // From config atoms (paths, keys, values)
    for (const auto& atom : meta.config_atoms) {
        for (uint32_t cp : utf8_to_codepoints(atom.path)) {
            needed_codepoints.insert(cp);
        }
        for (uint32_t cp : utf8_to_codepoints(atom.key)) {
            needed_codepoints.insert(cp);
        }
        for (uint32_t cp : utf8_to_codepoints(atom.value)) {
            needed_codepoints.insert(cp);
        }
    }
    
    // Debug: find max codepoint
    uint32_t max_cp = 0;
    for (uint32_t cp : needed_codepoints) {
        if (cp > max_cp) max_cp = cp;
    }
    std::cerr << "[METADATA] Loading " << needed_codepoints.size() << " atoms (max codepoint: " << max_cp << " / 0x" << std::hex << max_cp << std::dec << ")...\n";
    
    // Validate: filter out any codepoints beyond Unicode max (0x10FFFF)
    std::unordered_set<uint32_t> valid_codepoints;
    for (uint32_t cp : needed_codepoints) {
        if (cp <= 0x10FFFF) {
            valid_codepoints.insert(cp);
        } else {
            std::cerr << "[METADATA] WARNING: Invalid codepoint " << cp << " (0x" << std::hex << cp << std::dec << ") skipped\n";
        }
    }
    
    // Load atom cache
    std::unordered_map<uint32_t, AtomInfo> atom_cache;
    if (!load_atoms_for_codepoints(conn, valid_codepoints, atom_cache)) {
        std::cerr << "[METADATA] Warning: Failed to load atom cache\n";
    }
    std::cerr << "[METADATA] Loaded " << atom_cache.size() << " atoms\n";
    
    // Build semantic structure
    CompositionBuilder comp_builder(atom_cache);
    RelationBuilder rel_builder;
    
    // Build config AST
    if (!meta.config_atoms.empty()) {
        build_config_ast(meta.config_atoms, meta.model_name, comp_builder, rel_builder);
    }
    
    // Build BPE merge tree
    if (!meta.vocab_tokens.empty()) {
        build_bpe_tree(meta.vocab_tokens, meta.bpe_merges, meta.model_name, 
                       comp_builder, rel_builder);
    }
    
    // Build special tokens
    if (!meta.special_tokens.empty()) {
        build_special_tokens(meta.special_tokens, meta.model_name, 
                            comp_builder, rel_builder);
    }
    
    std::cerr << "[METADATA] Total: " << comp_builder.count() << " compositions, "
              << rel_builder.count() << " relations\n";
    
    // Start transaction and stream to database
    Transaction tx(conn);
    
    if (!stream_to_database(conn, comp_builder, rel_builder)) {
        return false;
    }
    
    tx.commit();
    return true;
}

} // namespace metadata
} // namespace ingest
} // namespace hypercube
