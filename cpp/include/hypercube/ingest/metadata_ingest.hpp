#pragma once
// =============================================================================
// METADATA INGESTION - Config, Tokenizer, Vocab as First-Class Content
// =============================================================================
//
// Model metadata IS semantic structure:
// - Config values become atoms (with their own 4D coordinates based on meaning)
// - BPE merges become composition edges (token A + token B -> token C)
// - Vocab entries become token atoms with PMI/frequency-based coordinates
// - All metadata is queryable through the same substrate as model weights
//
// This enables queries like:
// - "Find all models with >100k vocab"
// - "Find tokens that compose into 'transformers'"
// - "Find models with similar architectural dimensions"
// =============================================================================

#include <hypercube/ingest/model_manifest.hpp>
#include <hypercube/db/operations.hpp>
#include <hypercube/coordinates.hpp>
#include <libpq-fe.h>
#include <cmath>
#include <sstream>

namespace ingest {
namespace metadata {

// =============================================================================
// Config Atom Generation - Map config values to 4D coordinates
// =============================================================================

// Config values get coordinates based on their semantic type
inline std::tuple<double, double, double, double> 
config_to_coordinates(const std::string& key, const std::string& value) {
    // X: Category (0-1 normalized)
    //   0.0-0.2: Dimension params (hidden_size, d_model, etc.)
    //   0.2-0.4: Layer params (num_layers, num_heads, etc.)
    //   0.4-0.6: Vocab/token params
    //   0.6-0.8: Training params (dropout, etc.)
    //   0.8-1.0: Other
    
    double x = 0.9;  // Default: other
    if (key.find("size") != std::string::npos || 
        key.find("dim") != std::string::npos ||
        key.find("hidden") != std::string::npos) {
        x = 0.1;
    } else if (key.find("layer") != std::string::npos || 
               key.find("head") != std::string::npos ||
               key.find("block") != std::string::npos) {
        x = 0.3;
    } else if (key.find("vocab") != std::string::npos || 
               key.find("token") != std::string::npos ||
               key.find("position") != std::string::npos) {
        x = 0.5;
    } else if (key.find("dropout") != std::string::npos || 
               key.find("eps") != std::string::npos ||
               key.find("init") != std::string::npos) {
        x = 0.7;
    }
    
    // Y: Log-scaled value magnitude
    double numeric_val = 0;
    try {
        numeric_val = std::stod(value);
    } catch (...) {}
    
    double y = 0.5;
    if (numeric_val > 0) {
        y = std::min(1.0, std::log10(numeric_val + 1) / 7.0);  // Log scale, max ~10M
    } else if (numeric_val < 0) {
        y = 0.5 - std::min(0.5, std::log10(-numeric_val + 1) / 7.0);
    }
    
    // Z: Key name hash for uniqueness
    uint32_t hash = 0;
    for (char c : key) hash = hash * 31 + c;
    double z = (hash % 10000) / 10000.0;
    
    // W: Fixed for config atoms
    double w = 0.1;  // Low W = config space
    
    return {x, y, z, w};
}

// =============================================================================
// Token Atom Generation - Map vocab tokens to 4D coordinates
// =============================================================================

inline std::tuple<double, double, double, double>
token_to_coordinates(const std::string& token, int index, int vocab_size) {
    // X: Token type
    //   0.0-0.2: Special tokens (<s>, </s>, <pad>, <mask>)
    //   0.2-0.4: Punctuation/symbols
    //   0.4-0.6: Subword prefixes (##, Ġ)
    //   0.6-0.8: Common words
    //   0.8-1.0: Rare tokens
    
    double x = 0.9;  // Default: rare
    if (token.size() >= 1 && token[0] == '<' && token.back() == '>') {
        x = 0.1;  // Special
    } else if (token.size() == 1 && !std::isalnum(token[0])) {
        x = 0.3;  // Punctuation
    } else if (token.find("Ġ") == 0 || token.find("##") == 0 || token.find("▁") == 0) {
        x = 0.5;  // Subword with prefix
    } else if (index < vocab_size / 10) {
        x = 0.7;  // Common (top 10%)
    }
    
    // Y: Position in vocab (frequency proxy)
    double y = static_cast<double>(index) / vocab_size;
    
    // Z: Token content hash
    uint32_t hash = 0;
    for (char c : token) hash = hash * 31 + c;
    double z = (hash % 10000) / 10000.0;
    
    // W: Token length (normalized)
    double w = std::min(1.0, token.length() / 20.0);
    
    return {x, y, z, w};
}

// =============================================================================
// BPE Merge Composition - Create edges from merge rules
// =============================================================================

struct BPEComposition {
    std::string parent_token;   // Result of merge
    std::string left_token;     // Left part
    std::string right_token;    // Right part
    int merge_rank;             // Order in merge list (lower = more common)
    
    // The merge relationship is directional:
    // left_token + right_token -> parent_token
    // This creates a DAG where common subwords are leaves
};

// =============================================================================
// Database Insertion Functions
// =============================================================================

// Insert config atoms into the atom table
inline int insert_config_atoms(PGconn* conn, const ModelManifest& manifest) {
    if (manifest.config_atoms.empty()) return 0;
    
    hypercube::db::Transaction txn(conn);
    hypercube::db::CopyStream copy(conn, 
        "COPY atom (name, category, x, y, z, w) FROM STDIN WITH (FORMAT text)");
    
    int count = 0;
    for (const auto& [key, value] : manifest.config_atoms) {
        auto [x, y, z, w] = config_to_coordinates(key, value);
        
        // Name format: model_name::config::key=value
        std::string atom_name = manifest.model_name + "::config::" + key + "=" + value;
        
        copy.put_text(atom_name);
        copy.put_text("config");
        copy.put_double(x);
        copy.put_double(y);
        copy.put_double(z);
        copy.put_double(w);
        copy.end_row();
        count++;
    }
    
    if (!copy.finish()) {
        std::cerr << "[METADATA] Config atom insertion failed\n";
        return 0;
    }
    
    txn.commit();
    return count;
}

// Insert vocab tokens as atoms
inline int insert_vocab_atoms(PGconn* conn, const ModelManifest& manifest) {
    if (manifest.vocab.empty()) return 0;
    
    int vocab_size = manifest.vocab.size();
    
    hypercube::db::Transaction txn(conn);
    hypercube::db::CopyStream copy(conn,
        "COPY atom (name, category, x, y, z, w) FROM STDIN WITH (FORMAT text)");
    
    int count = 0;
    for (const auto& [token, index] : manifest.vocab) {
        auto [x, y, z, w] = token_to_coordinates(token, index, vocab_size);
        
        // Name format: model_name::token::escaped_token
        std::string escaped_token = token;
        // Escape special chars for atom name
        for (size_t i = 0; i < escaped_token.size(); ++i) {
            if (escaped_token[i] == ':' || escaped_token[i] == '\n' || 
                escaped_token[i] == '\t' || escaped_token[i] == '\\') {
                escaped_token.insert(i, "\\");
                ++i;
            }
        }
        
        std::string atom_name = manifest.model_name + "::token::" + escaped_token;
        
        copy.put_text(atom_name);
        copy.put_text("token");
        copy.put_double(x);
        copy.put_double(y);
        copy.put_double(z);
        copy.put_double(w);
        copy.end_row();
        count++;
    }
    
    if (!copy.finish()) {
        std::cerr << "[METADATA] Vocab atom insertion failed\n";
        return 0;
    }
    
    txn.commit();
    return count;
}

// Insert BPE merges as composition relations
// This creates a Merkle-DAG structure where:
// - Each merged token is a composition of its parts
// - Common subwords form the leaves
// - Full words are built up through merge chains
inline int insert_bpe_compositions(PGconn* conn, const ModelManifest& manifest) {
    if (manifest.bpe_merges.empty()) return 0;
    
    // First, we need to build the merge graph
    // BPE merge format: "token1 token2" -> merged result
    // The result is token1+token2 (concatenated, possibly with space handling)
    
    // Create compositions for each merge
    hypercube::db::Transaction txn(conn);
    
    // Insert compositions (parent tokens formed by merges)
    std::string sql = R"(
        INSERT INTO composition (name, category, centroid)
        SELECT DISTINCT 
            $1 || '::merge::' || left_tok || '+' || right_tok,
            'bpe_merge',
            NULL
        FROM (VALUES ($2, $3)) AS v(left_tok, right_tok)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
    )";
    
    int count = 0;
    int rank = 0;
    for (const auto& [left, right] : manifest.bpe_merges) {
        std::string merged = left + right;  // BPE merge result
        
        // Skip if we can't find the tokens in vocab
        if (manifest.vocab.find(left) == manifest.vocab.end() ||
            manifest.vocab.find(right) == manifest.vocab.end()) {
            rank++;
            continue;
        }
        
        // Insert the composition
        std::string comp_name = manifest.model_name + "::merge::" + left + "+" + right;
        
        std::string insert_sql = 
            "INSERT INTO composition (name, category) VALUES ($1, 'bpe_merge') "
            "ON CONFLICT (name) DO UPDATE SET name = composition.name RETURNING id";
        
        const char* params[] = { comp_name.c_str() };
        PGresult* res = PQexecParams(conn, insert_sql.c_str(), 1, nullptr, params, nullptr, nullptr, 0);
        
        if (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0) {
            int comp_id = std::stoi(PQgetvalue(res, 0, 0));
            
            // Link to child atoms (left and right tokens)
            std::string left_atom = manifest.model_name + "::token::" + left;
            std::string right_atom = manifest.model_name + "::token::" + right;
            
            std::string link_sql = R"(
                INSERT INTO composition_child (composition_id, child_id, child_type, weight)
                SELECT $1, id, 'A', $4
                FROM atom WHERE name = $2 OR name = $3
                ON CONFLICT DO NOTHING
            )";
            
            std::string weight_str = std::to_string(1.0 / (rank + 1));  // Higher weight for earlier merges
            const char* link_params[] = { 
                std::to_string(comp_id).c_str(),
                left_atom.c_str(), 
                right_atom.c_str(),
                weight_str.c_str()
            };
            PGresult* link_res = PQexecParams(conn, link_sql.c_str(), 4, nullptr, link_params, nullptr, nullptr, 0);
            PQclear(link_res);
            
            count++;
        }
        PQclear(res);
        rank++;
        
        // Progress indicator
        if (rank % 1000 == 0) {
            std::cerr << "\r[BPE] Processed " << rank << "/" << manifest.bpe_merges.size() << " merges";
        }
    }
    
    if (count > 0) {
        std::cerr << "\r[BPE] Inserted " << count << " BPE merge compositions\n";
    }
    
    txn.commit();
    return count;
}

// Insert model as a top-level composition containing all its components
inline int insert_model_composition(PGconn* conn, const ModelManifest& manifest) {
    hypercube::db::Transaction txn(conn);
    
    // Create the model composition
    std::string sql = R"(
        INSERT INTO composition (name, category)
        VALUES ($1, 'model')
        ON CONFLICT (name) DO UPDATE SET name = composition.name
        RETURNING id
    )";
    
    const char* params[] = { manifest.model_name.c_str() };
    PGresult* res = PQexecParams(conn, sql.c_str(), 1, nullptr, params, nullptr, nullptr, 0);
    
    int model_id = 0;
    if (PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) > 0) {
        model_id = std::stoi(PQgetvalue(res, 0, 0));
    }
    PQclear(res);
    
    if (model_id == 0) return 0;
    
    // Link config atoms to model
    std::string link_configs = R"(
        INSERT INTO composition_child (composition_id, child_id, child_type, weight)
        SELECT $1, id, 'A', 1.0
        FROM atom WHERE name LIKE $2 || '::config::%'
        ON CONFLICT DO NOTHING
    )";
    
    const char* config_params[] = { 
        std::to_string(model_id).c_str(), 
        manifest.model_name.c_str() 
    };
    PGresult* config_res = PQexecParams(conn, link_configs.c_str(), 2, nullptr, config_params, nullptr, nullptr, 0);
    int config_count = PQcmdTuples(config_res) ? std::stoi(PQcmdTuples(config_res)) : 0;
    PQclear(config_res);
    
    txn.commit();
    
    std::cerr << "[MODEL] Created model composition '" << manifest.model_name 
              << "' with " << config_count << " config children\n";
    
    return 1;
}

// Main entry point: ingest all metadata for a model
inline void ingest_model_metadata(PGconn* conn, ModelManifest& manifest) {
    std::cerr << "\n[METADATA] Ingesting metadata for " << manifest.model_name << "\n";
    
    // 1. Insert config atoms
    int config_count = insert_config_atoms(conn, manifest);
    std::cerr << "[METADATA] Inserted " << config_count << " config atoms\n";
    
    // 2. Insert vocab tokens as atoms
    int vocab_count = insert_vocab_atoms(conn, manifest);
    std::cerr << "[METADATA] Inserted " << vocab_count << " vocab token atoms\n";
    
    // 3. Insert BPE merges as compositions
    int merge_count = insert_bpe_compositions(conn, manifest);
    std::cerr << "[METADATA] Inserted " << merge_count << " BPE merge compositions\n";
    
    // 4. Create model composition linking everything
    insert_model_composition(conn, manifest);
    
    std::cerr << "[METADATA] Metadata ingestion complete\n";
}

} // namespace metadata
} // namespace ingest
