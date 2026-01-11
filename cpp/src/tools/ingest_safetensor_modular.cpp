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
#include "hypercube/ingest/metadata.hpp"
#include "hypercube/ingest/metadata_db.hpp"
#include "hypercube/ingest/model_manifest.hpp"
#include "hypercube/ingest/multimodal_extraction.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/ingest/db_operations.hpp"

#ifdef HAS_HNSWLIB
#include "hnswlib/hnswlib/hnswlib.h"
#endif

// OpenMP for parallel processing
#ifdef _OPENMP
#include <omp.h>
#endif

// MKL for optimized linear algebra (threaded matrix operations)
#if defined(HAS_MKL) && HAS_MKL
#include <mkl.h>
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
            std::string val = argv[++i];
            if (val.find('=') != std::string::npos) {
                // Full connection string
                config.conninfo = val;
            } else {
                // Just database name
                config.conninfo = "dbname=" + val;
            }
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
        std::cerr << "Usage: ingest_safetensor [-d conninfo|dbname] [-U user] [-h host] [-p port] [-n model_name] [-t threshold] <model_dir>\n";
        std::cerr << "  -d  Database connection string (full conninfo or just dbname)\n";
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
    ctx.verbose = config.verbose;
    
    // ============================================================================
    // CRITICAL: Configure threading BEFORE any parallel operations
    // MKL defaults to 1 thread if not configured - this kills performance
    // Limit to 8 Performance cores on 14900KS to avoid E-core tail latency
    // ============================================================================
    int num_threads = 8;  // 14900KS has 8 Performance cores

    // Set OpenMP threads
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    std::cerr << "[THREADING] OpenMP threads: " << omp_get_max_threads() << "\n";
    #else
    std::cerr << "[THREADING] WARNING: OpenMP not available\n";
    #endif

    // CRITICAL: Configure MKL for multi-threading
    #if defined(HAS_MKL) && HAS_MKL
    mkl_set_num_threads(num_threads);
    mkl_set_dynamic(0);  // Force exact thread count - CRITICAL for performance
    std::cerr << "[THREADING] MKL threads: " << mkl_get_max_threads() << " (dynamic=0)\n";
    #else
    std::cerr << "[THREADING] WARNING: MKL not available - operations will be slow\n";
    #endif

    std::cerr << "=== Universal Safetensor Ingester (Modular) ===\n";
    std::cerr << "Directory: " << model_dir << "\n";
    std::cerr << "Model: " << config.model_name << "\n";
    std::cerr << "Threshold: " << config.weight_threshold << "\n\n";
    
    // Parse model manifest for intelligent routing
    std::cerr << "[0] Parsing model manifest (config.json, architecture detection)...\n";
    ingest::ModelManifest manifest = ingest::parse_model_manifest(dir);
    manifest.model_name = config.model_name;  // Override with user-specified name

    // Store manifest in context for config-driven tensor lookup
    ctx.manifest = manifest;
    
    std::cerr << "\n";
    
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
        else if (name.find(".safetensors") != std::string::npos) {
            safetensor_files.push_back(entry.path());
        }
    }
    
    // Parse tokenizer first (need vocab for token lookups)
    if (!tokenizer_path.empty()) {
        std::cerr << "[1] Parsing tokenizer: " << tokenizer_path << "\n";
        bool tokenizer_ok = parse_tokenizer(ctx, tokenizer_path);
        std::cerr << "[1] Tokenizer parse result: " << (tokenizer_ok ? "OK" : "FAILED") << "\n";
        std::cerr << "[1] BPE merges loaded: " << ctx.bpe_merges.size() << "\n";
        std::cerr << "[1] Vocab entries: " << ctx.vocab.size() << "\n";
    } else {
        std::cerr << "[1] No tokenizer found\n";
    }
    
    // Parse vocab.txt if available (BERT-style models)
    if (!vocab_path.empty()) {
        std::cerr << "[2] Parsing vocab: " << vocab_path << "\n";
        bool vocab_ok = parse_vocab(ctx, vocab_path);
        std::cerr << "[2] Vocab parse result: " << (vocab_ok ? "OK" : "FAILED") << "\n";
        std::cerr << "[2] Vocab tokens loaded: " << ctx.vocab_tokens.size() << "\n";
    } else {
        std::cerr << "[2] No vocab.txt found\n";
    }
    
    // === NEW: Parse ALL model metadata (config, tokenizer, vocab as first-class content) ===
    std::cerr << "\n[2.5] Parsing model metadata (config, tokenizer, special tokens)...\n";
    metadata::ModelMetadata model_meta;
    metadata::parse_model_metadata(dir, model_meta);
    
    // Transfer vocab tokens with compositions to ctx for semantic extraction
    // These are REAL compositions (atom trajectories) that we need for relation building
    if (!model_meta.vocab_tokens.empty() && ctx.vocab_tokens.empty()) {
        ctx.vocab_tokens.resize(model_meta.vocab_tokens.size());
        for (size_t i = 0; i < model_meta.vocab_tokens.size(); ++i) {
            const auto& vt = model_meta.vocab_tokens[i];
            ingest::TokenInfo info;
            info.text = vt.text;
            info.comp = AtomCalculator::compute_vocab_token(vt.text);
            ctx.vocab_tokens[i] = std::move(info);
            ctx.token_to_idx[vt.text] = i;
        }
        std::cerr << "[VOCAB] Transferred " << ctx.vocab_tokens.size() << " token compositions to context\n";
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

    // Log tensor names for debugging
    std::cerr << "[DEBUG] Tensor names:\n";
    for (const auto& [name, meta] : ctx.tensors) {
        std::cerr << "  " << name << " [" << meta.shape.size() << "D";
        for (size_t s : meta.shape) std::cerr << "," << s;
        std::cerr << "] " << meta.dtype << "\n";
    }
    
    // Categorize tensors in manifest for extraction planning
    if (ctx.manifest.has_value()) {
        std::cerr << "[3.1] Categorizing tensors for extraction...\n";
        for (const auto& [name, meta] : ctx.tensors) {
            ctx.manifest->categorize_tensor(name, meta.shape, meta.dtype);
        }
        std::cerr << "[INFO] Created " << ctx.manifest->extraction_plans.size() << " extraction plans\n";

        // Now print the summary with the categorized tensors
        ctx.manifest->print_summary();
    }
    
    // Connect to database
    std::cerr << "[DB] Connecting to database with conninfo: " << config.conninfo << "\n";
    PGconn* conn = PQconnectdb(config.conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "[DB] Connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }
    std::cerr << "[DB] Database connection successful\n";
    
    // === INSERT MODEL METADATA AS FIRST-CLASS CONTENT ===
    // All metadata (config, tokenizer, vocab) becomes atoms/compositions/relations
    std::cerr << "\n[3.5] Inserting model metadata as content...\n";
    if (!model_meta.model_name.empty()) {
        metadata::insert_model_metadata(conn, model_meta);
    }
    
    // === BUILD THE COMPOSITION HIERARCHY ===
    
    // Step 4: Insert tensor hierarchy as compositions
    std::cerr << "\n[4] Building tensor name hierarchy...\n";
    if (!ctx.tensors.empty()) {
        ingest::db::insert_tensor_hierarchy(conn, ctx, config);
    }
    
    // Step 5: Insert vocab token compositions (BPE tokens)
    if (!ctx.vocab_tokens.empty()) {
        std::cerr << "\n[5] Inserting token compositions...\n";
        ingest::db::insert_compositions(conn, ctx);
    }

    // Step 5.5: Project embeddings to 4D using Laplacian eigenmaps and update compositions
    std::cerr << "\n[5.5] Projecting token embeddings to 4D semantic coordinates...\n";
    std::cerr << "[5.5] Vocab tokens available: " << ctx.vocab_tokens.size() << "\n";
    std::cerr << "[5.5] Tensors available: " << ctx.tensors.size() << "\n";

    if (!ctx.vocab_tokens.empty()) {
        std::cerr << "[5.5] Starting Laplacian projection...\n";
        auto projection_start = std::chrono::steady_clock::now();
        ingest::db::project_and_update_embeddings(conn, ctx, config);
        auto projection_end = std::chrono::steady_clock::now();
        auto projection_ms = std::chrono::duration_cast<std::chrono::milliseconds>(projection_end - projection_start).count();
        std::cerr << "[5.5] Laplacian projection completed in " << projection_ms << "ms\n";
    } else {
        std::cerr << "[PROJECTION] No vocab tokens loaded, skipping Laplacian projection\n";
    }

    // Step 6: Compute composition centroids FROM CHILDREN
    std::cerr << "\n[6] Computing composition centroids hierarchically...\n";
    {
        // First pass: compositions with atom children
        hypercube::db::Result res = hypercube::db::exec(conn, "SELECT recompute_composition_centroids()");
        if (!res.ok()) {
            std::cerr << "[CENTROID] Failed: " << res.error_message() << "\n";
        } else {
            int updated = res.integer(0, 0);
            std::cerr << "[CENTROID] Updated " << updated << " composition centroids from atoms\n";
        }
        
        // Second pass: compositions with composition children (hierarchical)
        hypercube::db::Result res2 = hypercube::db::exec(conn,
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
        
        int hier_updated = res2.ok() ? hypercube::db::cmd_tuples(res2) : 0;
        if (res2.ok()) {
            std::cerr << "[CENTROID] Updated " << hier_updated << " hierarchical composition centroids\n";
        }
    }
    
    // === EXTRACT RELATIONS (MODEL-SPECIFIC EDGES) ===
    
    // Step 7: Extract semantic relations from ALL projection matrices
    // This processes: embeddings, Q/K/V projections, FFN layers - everything
    // Each model contributes relations tagged with (source_model, layer, component)
    // Relations ACCUMULATE across models - the semantic substrate grows
    std::cerr << "\n[7] Extracting complete semantic relations (all projections)...\n";
    if (!ctx.tensors.empty()) {
        ingest::db::extract_all_semantic_relations(conn, ctx, config);
    } else {
        std::cerr << "[SEMANTIC] No tensors found, skipping extraction\n";
    }
    
    // Step 8: Extract router/attention relations (MoE models, attention weights)
    std::cerr << "\n[8] Extracting weight-based relations (router, attention, MLP)...\n";
    ingest::db::insert_attention_relations(conn, ctx, config);
    
    // Step 9: Extract multimodal structures (object queries, MoE routers, positional, vision)
    // This extracts semantic structures that make DETR, Florence, MoE models actually work
    std::cerr << "\n[9] Extracting multimodal semantic structures...\n";
    if (ctx.manifest.has_value()) {
        size_t multimodal_relations = ingest::extract_multimodal_structures(
            conn, ctx, *ctx.manifest
        );
        std::cerr << "[MULTIMODAL] Extracted " << multimodal_relations << " semantic relations\n";
    }

    // Step 10: Refresh relation consensus materialized view
    std::cerr << "\n[10] Refreshing relation consensus...\n";
    {
        hypercube::db::Result res = hypercube::db::exec(conn, "SELECT refresh_relation_consensus()");
        if (res.ok()) {
            std::cerr << "[CONSENSUS] Refreshed relation consensus materialized view\n";
        } else {
            std::cerr << "[CONSENSUS] Failed to refresh consensus: " << res.error_message() << "\n";
        }
    }

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
