// Universal Safetensor Package Ingester - Modular Version
// =========================================================
// Ingests ANY HuggingFace model package into the hypercube substrate.
//
// ARCHITECTURE:
//   Atoms:       Unicode codepoints with deterministic 4D coordinates
//   Compositions: Aggregations with centroids = average of atom children
//   Relations:   Edges from multiple sources forming THE KNOWLEDGE GRAPH:
//                - 'E' = Embedding k-NN similarity
//                - 'R' = Router weights (MoE expert routing)
//                - 'W' = Weight similarity (Q/K/V/O/MLP patterns)
//                - 'D' = Dimension activation (token→dimension mappings)
//                - 'C' = BPE composition relations
//
// This refactored version uses modular components from:
//   - hypercube/ingest/context.hpp   : IngestContext, IngestConfig
//   - hypercube/ingest/parsing.hpp   : Safetensor/tokenizer parsing
//   - hypercube/ingest/geometry.hpp  : EWKB geometry builders
//   - hypercube/ingest/db_operations.hpp : Database insertion functions

// Prevent Windows min/max macro issues
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <libpq-fe.h>

// Core hypercube headers
#include "hypercube/types.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/laplacian_4d.hpp"
#include "hypercube/embedding_ops.hpp"

// Modular ingest headers
#include "hypercube/ingest/safetensor.hpp"
#include "hypercube/ingest/context.hpp"
#include "hypercube/ingest/parsing.hpp"
#include "hypercube/ingest/geometry.hpp"
#include "hypercube/ingest/db_operations.hpp"

#ifdef HAS_HNSWLIB
#include "hnswlib/hnswlib/hnswlib.h"
#endif

namespace fs = std::filesystem;
using namespace hypercube;
using namespace hypercube::ingest;

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    IngestConfig config;
    std::string model_dir;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            config.conninfo = "dbname=" + std::string(argv[++i]);
        } else if (arg == "-U" && i + 1 < argc) {
            config.conninfo += " user=" + std::string(argv[++i]);
        } else if (arg == "-h" && i + 1 < argc) {
            config.conninfo += " host=" + std::string(argv[++i]);
        } else if (arg == "-p" && i + 1 < argc) {
            config.conninfo += " port=" + std::string(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            config.model_name = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            config.weight_threshold = std::stof(argv[++i]);
        } else if (arg == "-v") {
            config.verbose = true;
        } else if (arg[0] != '-') {
            model_dir = arg;
        }
    }
    
    if (model_dir.empty()) {
        std::cerr << "Usage: ingest_safetensor [-d db] [-U user] [-h host] [-p port] [-n model_name] [-t threshold] <model_dir>\n";
        std::cerr << "  -n  Model name prefix (e.g. 'minilm', 'llama4')\n";
        std::cerr << "  -t  Weight threshold for attention edges (default 0.5)\n";
        return 1;
    }
    
    fs::path dir(model_dir);
    if (!fs::is_directory(dir)) {
        std::cerr << "Not a directory: " << model_dir << "\n";
        return 1;
    }
    
    // Auto-detect model name from directory if not specified
    if (config.model_name.empty()) {
        config.model_name = dir.filename().string();
    }
    
    // Create ingest context
    IngestContext ctx;
    ctx.model_prefix = config.model_name + ":";
    
    std::cerr << "=== Universal Safetensor Ingester (Modular) ===\n";
    std::cerr << "Directory: " << model_dir << "\n";
    std::cerr << "Model: " << config.model_name << "\n";
    std::cerr << "Threshold: " << config.weight_threshold << "\n\n";
    
    auto total_start = std::chrono::steady_clock::now();
    
    // Find model files
    fs::path vocab_path, tokenizer_path, index_path;
    std::vector<fs::path> safetensor_files;
    
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        std::string path_str = entry.path().string();
        
        // Skip hidden directories and cache folders
        if (path_str.find("\\.") != std::string::npos || 
            path_str.find("/.") != std::string::npos ||
            path_str.find(".cache") != std::string::npos) {
            continue;
        }
        
        if (name == "vocab.txt") vocab_path = entry.path();
        else if (name == "tokenizer.json") tokenizer_path = entry.path();
        else if (name == "model.safetensors.index.json") index_path = entry.path();
        else if (name.ends_with(".safetensors")) {
            safetensor_files.push_back(entry.path());
        }
    }
    
    // Parse tokenizer first (need vocab for token lookups)
    if (!tokenizer_path.empty()) {
        std::cerr << "[1] Parsing tokenizer: " << tokenizer_path << "\n";
        parse_tokenizer(ctx, tokenizer_path);
    }
    
    // Parse vocab.txt if available (BERT-style models)
    if (!vocab_path.empty()) {
        std::cerr << "[2] Parsing vocab: " << vocab_path << "\n";
        parse_vocab(ctx, vocab_path);
    }
    
    // Parse model tensors
    if (!index_path.empty()) {
        std::cerr << "[3] Parsing sharded model index: " << index_path << "\n";
        parse_model_index(ctx, index_path);
    } else if (!safetensor_files.empty()) {
        std::cerr << "[3] Parsing " << safetensor_files.size() << " safetensor files...\n";
        for (const auto& f : safetensor_files) {
            std::cerr << "  Parsing: " << f << "\n";
            if (!parse_safetensor_header(ctx, f)) {
                std::cerr << "  [ERROR] Failed to parse: " << f << "\n";
                return 1;
            }
        }
    }
    
    if (ctx.tensors.empty()) {
        std::cerr << "[ERROR] No tensors found!\n";
        return 1;
    }
    
    std::cerr << "[INFO] Found " << ctx.tensors.size() << " tensors\n";
    
    // Connect to database
    PGconn* conn = PQconnectdb(config.conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }
    
    // === BUILD THE COMPOSITION HIERARCHY ===
    
    // Step 4: Insert tensor hierarchy as compositions
    std::cerr << "\n[4] Building tensor name hierarchy...\n";
    if (!ctx.tensors.empty()) {
        db::insert_tensor_hierarchy(conn, ctx, config);
    }
    
    // Step 5: Insert vocab token compositions (BPE tokens)
    if (!ctx.vocab_tokens.empty()) {
        std::cerr << "\n[5] Inserting token compositions...\n";
        db::insert_compositions(conn, ctx);
    }
    
    // Step 6: Compute composition centroids FROM CHILDREN
    std::cerr << "\n[6] Computing composition centroids hierarchically...\n";
    {
        // First pass: compositions with atom children
        PGresult* res = PQexec(conn, "SELECT recompute_composition_centroids()");
        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            std::cerr << "[CENTROID] Failed: " << PQerrorMessage(conn) << "\n";
        } else {
            int updated = atoi(PQgetvalue(res, 0, 0));
            std::cerr << "[CENTROID] Updated " << updated << " composition centroids from atoms\n";
        }
        PQclear(res);
        
        // Second pass: compositions with composition children (hierarchical)
        res = PQexec(conn,
            "WITH RECURSIVE comp_tree AS ("
            "  SELECT id, centroid, 1 as level FROM composition WHERE centroid IS NOT NULL "
            "  UNION ALL "
            "  SELECT c.id, "
            "    st_centroid_4d(ST_Collect(child.centroid)), "
            "    ct.level + 1 "
            "  FROM composition c "
            "  JOIN composition_child cc ON cc.composition_id = c.id AND cc.child_type = 'C' "
            "  JOIN comp_tree ct ON ct.id = cc.child_id "
            "  JOIN composition child ON child.id = cc.child_id "
            "  WHERE c.centroid IS NULL AND child.centroid IS NOT NULL "
            "  GROUP BY c.id "
            ") "
            "UPDATE composition SET centroid = comp_tree.centroid "
            "FROM comp_tree WHERE composition.id = comp_tree.id");
        
        int hier_updated = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
        if (PQresultStatus(res) == PGRES_COMMAND_OK) {
            std::cerr << "[CENTROID] Updated " << hier_updated << " hierarchical composition centroids\n";
        }
        PQclear(res);
    }
    
    // === EXTRACT RELATIONS (MODEL-SPECIFIC EDGES) ===
    
    // Step 7: Extract k-NN similarity edges from model embeddings as RELATIONS
    std::cerr << "\n[7] Extracting embedding k-NN similarity as relations...\n";
    if (!ctx.tensors.empty()) {
        db::extract_embedding_relations(conn, ctx, config);
    } else {
        std::cerr << "[EMBED] No tensors found, skipping relation extraction\n";
    }
    
    // Step 8: Extract router/attention relations (MoE models, attention weights)
    std::cerr << "\n[8] Extracting weight-based relations (router, attention, MLP)...\n";
    db::insert_attention_relations(conn, ctx, config);
    
    PQfinish(conn);
    
    auto total_end = std::chrono::steady_clock::now();
    auto total_secs = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    
    std::cerr << "\n=== Complete ===\n";
    std::cerr << "Total time: " << total_secs << " seconds\n";
    std::cerr << "Tensors: " << ctx.tensors.size() << "\n";
    std::cerr << "BPE merges: " << ctx.bpe_merges.size() << "\n";
    std::cerr << "Vocab: " << ctx.vocab_tokens.size() << " tokens\n";
    
    return 0;
}
