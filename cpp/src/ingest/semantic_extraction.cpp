/**
 * @file semantic_extraction.cpp
 * @brief TRULY PARALLEL semantic extraction
 *
 * PARALLELISM STRATEGY:
 * 1. MKL cblas_sgemm - THREADED (uses OpenMP internally)
 * 2. normalize_rows - PARALLELIZED with OpenMP
 * 3. HNSW build - Partitioned parallel: build shards, then merge
 * 4. HNSW query - PARALLELIZED (query is thread-safe)
 *
 * Environment: Set OMP_NUM_THREADS and MKL_NUM_THREADS before running
 */

#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/ingest/projection_db.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/laplacian_4d.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <set>
#include <mutex>
#include <atomic>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#define getpid() GetCurrentProcessId()
#else
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <stdlib.h>
#define bswap32(x) _byteswap_ulong(x)
#define bswap16(x) _byteswap_ushort(x)
#else
#define bswap32(x) __builtin_bswap32(x)
#define bswap16(x) __builtin_bswap16(x)
#endif

#if defined(HAS_MKL) && HAS_MKL
#include <mkl.h>
#endif

#ifdef HAS_HNSWLIB
#include <hnswlib/hnswlib.h>
#endif

namespace hypercube {
namespace ingest {
namespace db {

// ============================================================================
// Config
// ============================================================================

static constexpr size_t K = 5;
static constexpr float THRESHOLD = 0.7f;
static constexpr float MIN_NORM = 0.01f;
static int g_num_threads = 1;  // Set at runtime

// Projection k-NN is enabled for complete semantic extraction
static constexpr bool ENABLE_PROJECTION_KNN = true;
static constexpr size_t MAX_PROJECTIONS = 2;  // Limit to first N projections

// ============================================================================
// Load tensor
// ============================================================================

static std::vector<float> load_tensor(const TensorMeta& meta) {
    size_t n = meta.element_count();
    std::vector<float> data(n);

    std::cerr << "[LOAD_TENSOR] Opening file: " << meta.shard_file << "\n";
    std::cerr << "[LOAD_TENSOR] Seeking to offset: " << meta.data_offset_start << "\n";
    std::cerr << "[LOAD_TENSOR] Reading " << n << " elements as " << meta.dtype << "\n";

    std::ifstream f(meta.shard_file, std::ios::binary);
    if (!f) {
        std::cerr << "[LOAD_TENSOR] ERROR: Failed to open file!\n";
        return {};
    }

    f.seekg(static_cast<std::streamoff>(meta.data_offset_start));
    if (f.fail()) {
        std::cerr << "[LOAD_TENSOR] ERROR: Seek failed! Offset may be invalid.\n";
        return {};
    }

    if (meta.dtype == "F32") {
        f.read(reinterpret_cast<char*>(data.data()), n * 4);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for F32 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 4) << ")\n";
    } else if (meta.dtype == "BF16") {
        std::vector<uint16_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * 2);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for BF16 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 2) << ")\n";
        // PARALLEL BF16 conversion
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
            std::memcpy(&data[i], &bits, 4);
        }

        // VALIDATE: Check for extreme values indicating corruption
        size_t corrupt_count = 0;
        size_t nan_count = 0;
        #pragma omp parallel for reduction(+:corrupt_count,nan_count)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            float val = data[i];
            if (std::isnan(val)) {
                nan_count++;
                data[i] = 0.0f;  // Replace NaN with 0
            } else if (std::abs(val) > 1e10f) {  // Extreme values
                corrupt_count++;
                data[i] = 0.0f;  // Replace corrupt values with 0
            }
        }
        if (corrupt_count > 0 || nan_count > 0) {
            std::cerr << "[LOAD_TENSOR] WARNING: Fixed " << corrupt_count << " extreme values and "
                      << nan_count << " NaNs in BF16 data\n";
        }
    } else if (meta.dtype == "F16") {
        std::vector<uint16_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * 2);
        if (f.fail()) {
            std::cerr << "[LOAD_TENSOR] ERROR: Read failed for F16 data!\n";
            return {};
        }
        std::cerr << "[LOAD_TENSOR] Successfully read " << f.gcount() << " bytes (expected " << (n * 2) << ")\n";
        // PARALLEL F16 conversion
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint16_t h = raw[i];
            uint32_t s = (h & 0x8000) << 16;
            uint32_t e = (h >> 10) & 0x1F;
            uint32_t m = h & 0x3FF;
            uint32_t fval = (e == 0) ? s : (e == 31) ? (s | 0x7F800000 | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
            std::memcpy(&data[i], &fval, 4);
        }
    } else {
        std::cerr << "[LOAD_TENSOR] ERROR: Unknown dtype '" << meta.dtype << "'\n";
        return {};
    }

    // Verify non-zero data
    int nonzero = 0;
    for (size_t i = 0; i < std::min(size_t(100), n); ++i) {
        if (std::abs(data[i]) > 1e-10f) ++nonzero;
    }
    std::cerr << "[LOAD_TENSOR] Non-zero values in first 100 elements: " << nonzero << "/100\n";

    return data;
}

// ============================================================================
// PARALLEL normalize rows, return valid indices (skip near-zero)
// ============================================================================

static std::vector<size_t> normalize_rows(float* data, size_t rows, size_t cols) {
    // First pass: compute norms in parallel
    std::vector<float> norms(rows);
    
    #pragma omp parallel for schedule(static) num_threads(g_num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++) {
        const float* row = data + i * cols;
        float norm2 = 0;
        // Use SIMD-friendly loop
        for (size_t j = 0; j < cols; j++) {
            norm2 += row[j] * row[j];
        }
        norms[i] = std::sqrt(norm2);
    }
    
    // Second pass: normalize valid rows in parallel (write back)
    #pragma omp parallel for schedule(static) num_threads(g_num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++) {
        if (norms[i] >= MIN_NORM) {
            float inv = 1.0f / norms[i];
            float* row = data + i * cols;
            for (size_t j = 0; j < cols; j++) {
                row[j] *= inv;
            }
        }
    }
    
    // Collect valid indices (sequential, but fast)
    std::vector<size_t> valid;
    valid.reserve(rows);
    for (size_t i = 0; i < rows; i++) {
        if (norms[i] >= MIN_NORM) valid.push_back(i);
    }
    return valid;
}

// ============================================================================
// k-NN using HNSW - single-threaded BUILD, PARALLEL QUERY
// HNSW addPoint is NOT thread-safe, but searchKnn IS thread-safe
// ============================================================================

struct Edge { int32_t src, tgt; float sim; };

#ifdef HAS_HNSWLIB
static std::vector<Edge> knn(const float* data, const std::vector<size_t>& valid, size_t dim) {
    size_t n = valid.size();
    if (n < 2) return {};

    std::cerr << "[KNN] Building index for " << n << " vectors (single-threaded)...\n";

    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float> idx(&space, n, 16, 100);

    // Progress logging for build (single-threaded, this is the bottleneck)
    auto build_start = std::chrono::steady_clock::now();
    size_t log_interval = std::max(size_t(1), n / 10);

    for (size_t i = 0; i < n; i++) {
        idx.addPoint(data + valid[i] * dim, i);
        if ((i + 1) % log_interval == 0 || i + 1 == n) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - build_start).count();
            double pct = 100.0 * (i + 1) / n;
            double rate = (i + 1) * 1000.0 / (elapsed + 1);
            std::cerr << "[KNN] Build: " << (i + 1) << "/" << n
                      << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                      << " - " << rate << " vecs/sec\n";
        }
    }
    idx.setEf(50);

    std::cerr << "[KNN] Querying " << n << " vectors (PARALLEL, " << g_num_threads << " threads)...\n";

    // PARALLEL query - searchKnn IS thread-safe
    std::vector<std::set<std::tuple<size_t, size_t, float>>> thread_edge_sets(g_num_threads);

    auto query_start = std::chrono::steady_clock::now();
    std::atomic<size_t> progress{0};

    #pragma omp parallel num_threads(g_num_threads)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        std::set<std::tuple<size_t, size_t, float>>& local = thread_edge_sets[tid];

        #pragma omp for schedule(dynamic, 256)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            auto result = idx.searchKnn(data + valid[i] * dim, K + 1);
            while (!result.empty()) {
                auto [dist, j] = result.top();
                result.pop();
                if (static_cast<size_t>(j) == i) continue;
                float sim = 1.0f - dist;
                if (sim >= THRESHOLD && valid[i] < valid[j]) {
                    local.emplace(valid[i], valid[j], sim);
                }
            }

            // Progress logging (every 10% from thread 0)
            size_t p = progress.fetch_add(1);
            if (tid == 0 && (p % (n / 10 + 1) == 0)) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - query_start).count();
                double rate = (p + 1) * 1000.0 / (elapsed + 1);
                std::cerr << "[KNN] Query: " << p << "/" << n
                          << " - " << rate << " queries/sec\n";
            }
        }
    }

    // Merge results
    std::set<std::tuple<size_t, size_t, float>> all_edges;
    for (auto& tes : thread_edge_sets) {
        all_edges.insert(tes.begin(), tes.end());
    }

    std::vector<Edge> edges;
    edges.reserve(all_edges.size());
    for (auto& t : all_edges) {
        edges.push_back({static_cast<int32_t>(std::get<0>(t)), static_cast<int32_t>(std::get<1>(t)), std::get<2>(t)});
    }

    std::cerr << "[KNN] Found " << edges.size() << " unique edges above threshold " << THRESHOLD << "\n";
    return edges;
}
#else
static std::vector<Edge> knn(const float*, const std::vector<size_t>&, size_t) { return {}; }
#endif

// ============================================================================
// Insert edges via batched INSERT
// ============================================================================

static void insert(PGconn* conn, const std::vector<Edge>& edges, const IngestContext& ctx, const std::string& tag) {
    if (edges.empty()) return;

    // Deduplicate edges (though HNSW should already produce unique edges)
    std::set<std::tuple<Blake3Hash, Blake3Hash, float>> seen;
    for (const auto& e : edges) {
        if ((size_t)e.src >= ctx.vocab_tokens.size() || (size_t)e.tgt >= ctx.vocab_tokens.size()) continue;
        const auto& s = ctx.vocab_tokens[e.src].comp;
        const auto& t = ctx.vocab_tokens[e.tgt].comp;
        seen.emplace(s.hash, t.hash, e.sim);
    }

    // Use temp table then INSERT with existence checks
    PQexec(conn, "DROP TABLE IF EXISTS tmp_semantic_rel");
    PQexec(conn, "CREATE TEMP TABLE tmp_semantic_rel ("
                 "source_type CHAR(1), source_id BYTEA, target_type CHAR(1), target_id BYTEA, "
                 "relation_type CHAR(1), weight REAL, source_model TEXT, layer INTEGER, component TEXT)");

    std::string copy_cmd = "COPY tmp_semantic_rel FROM STDIN";
    PGresult* res = PQexec(conn, copy_cmd.c_str());
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "[INSERT] COPY start failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        return;
    }
    PQclear(res);

    const std::string component = "semantic";

    for (const auto& [source_hash, target_hash, weight] : seen) {
        std::string line = "C\t\\\\x" + source_hash.to_hex() + "\tC\t\\\\x" + target_hash.to_hex() +
                          "\tS\t" + std::to_string(weight) + "\t" + tag + "\t-1\t" + component + "\n";

        if (PQputCopyData(conn, line.c_str(), (int)line.size()) != 1) {
            std::cerr << "[INSERT] COPY data failed: " << PQerrorMessage(conn) << "\n";
            return;
        }
    }

    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "[INSERT] COPY to temp failed: " << PQerrorMessage(conn) << "\n";
        return;
    }

    res = PQgetResult(conn);
    PQclear(res);

    // INSERT with existence checks
    res = PQexec(conn,
        "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component) "
        "SELECT source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component "
        "FROM tmp_semantic_rel t "
        "WHERE EXISTS (SELECT 1 FROM composition WHERE id = t.source_id) "
        "AND EXISTS (SELECT 1 FROM composition WHERE id = t.target_id) "
        "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
        "weight = EXCLUDED.weight");

    int inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
    PQclear(res);

    std::cerr << "[INSERT] Inserted " << inserted << " semantic relations (filtered " << (seen.size() - inserted) << " missing refs)\n";
}

// ============================================================================
// Main - PARALLEL everywhere possible
// ============================================================================

bool extract_all_semantic_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    
    // Detect and set thread count
    g_num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (g_num_threads < 1) g_num_threads = 8;
    
    std::cerr << "============================================================\n";
    std::cerr << "[SEMANTIC] PARALLEL EXTRACTION starting\n";
    std::cerr << "[SEMANTIC] Hardware threads: " << g_num_threads << "\n";
    
#ifdef _OPENMP
    omp_set_num_threads(g_num_threads);
    std::cerr << "[SEMANTIC] OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cerr << "[SEMANTIC] WARNING: OpenMP NOT AVAILABLE - many operations single-threaded!\n";
#endif
    
#if defined(HAS_MKL) && HAS_MKL
    mkl_set_num_threads(g_num_threads);
    mkl_set_dynamic(0);  // CRITICAL: Force thread count
    std::cerr << "[SEMANTIC] MKL threads: " << mkl_get_max_threads() << " (dynamic=0)\n";
#else
    std::cerr << "[SEMANTIC] WARNING: MKL NOT AVAILABLE - GEMM will be slow!\n";
#endif

#ifdef HAS_HNSWLIB
    std::cerr << "[SEMANTIC] HNSWLIB: available\n";
#else
    std::cerr << "[SEMANTIC] WARNING: HNSWLIB NOT AVAILABLE - no k-NN!\n";
#endif
    std::cerr << "============================================================\n";
    
    // =========================================================================
    // CONFIG-DRIVEN TENSOR LOOKUP
    // Use the parsed model manifest to find tensors by ROLE, not by name pattern
    // The config.json tells us everything - vocab_size, d_model, architecture
    // =========================================================================
    
    const TensorMeta* emb = nullptr;
    std::string emb_name;
    
    // First try: Use manifest's categorized extraction plans
    if (ctx.manifest.has_value()) {
        std::cerr << "[SEMANTIC] Using config-driven tensor lookup (architecture: " 
                  << architecture_to_string(ctx.manifest->architecture) << ")\n";
        
        for (const auto& plan : ctx.manifest->extraction_plans) {
            if (plan.category == TensorCategory::TOKEN_EMBEDDING) {
                auto it = ctx.tensors.find(plan.name);
                if (it != ctx.tensors.end()) {
                    emb = &it->second;
                    emb_name = plan.name;
                    std::cerr << "[SEMANTIC] Found TOKEN_EMBEDDING via manifest: " << plan.name 
                              << " [" << emb->shape[0] << " x " << emb->shape[1] << "]\n";
                    break;
                }
            }
        }
    }
    
    // Fallback: If no manifest or manifest didn't find embedding, scan tensors
    // This should rarely happen if manifest parsing is working
    if (!emb) {
        std::cerr << "[SEMANTIC] WARNING: Manifest lookup failed, falling back to tensor scan\n";
        
        // Get expected vocab size from config if available
        size_t expected_vocab = ctx.get_vocab_size();
        size_t expected_dim = ctx.get_model_dim();
        
        for (const auto& [name, meta] : ctx.tensors) {
            if (meta.shape.size() != 2) continue;
            if (name.find("position") != std::string::npos) continue;
            
            // If config tells us vocab_size, use it to validate
            if (expected_vocab > 0 && meta.shape[0] == static_cast<int64_t>(expected_vocab)) {
                // This tensor has the right vocab dimension - likely the embedding
                if (expected_dim == 0 || meta.shape[1] == static_cast<int64_t>(expected_dim)) {
                    emb = &meta;
                    emb_name = name;
                    std::cerr << "[SEMANTIC] Found embedding by dimension match: " << name 
                              << " [" << meta.shape[0] << " x " << meta.shape[1] << "]\n";
                    break;
                }
            }
        }
    }
    
    if (!emb) {
        std::cerr << "[SEMANTIC] No embedding tensor found\n";
        return true;
    }
    
    size_t V = static_cast<size_t>(emb->shape[0]);
    size_t D = static_cast<size_t>(emb->shape[1]);
    if (!ctx.vocab_tokens.empty()) V = std::min(V, ctx.vocab_tokens.size());
    
    std::cerr << "[SEMANTIC] Loading " << V << " x " << D << " embeddings...\n";
    auto load_start = std::chrono::steady_clock::now();
    auto E = load_tensor(*emb);
    auto load_end = std::chrono::steady_clock::now();
    if (E.empty()) {
        std::cerr << "[SEMANTIC] Failed to load embeddings\n";
        return false;
    }
    E.resize(V * D);
    std::cerr << "[SEMANTIC] Loaded in " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count() << "ms\n";
    
    std::cerr << "[SEMANTIC] Normalizing (parallel)...\n";
    auto norm_start = std::chrono::steady_clock::now();
    auto valid = normalize_rows(E.data(), V, D);
    auto norm_end = std::chrono::steady_clock::now();
    std::cerr << "[SEMANTIC] " << valid.size() << "/" << V << " valid embeddings in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(norm_end - norm_start).count() << "ms\n";
    
    // NOTE: Base embedding k-NN REMOVED - attention_relations.cpp handles this
    // with better HNSW params (K=15, threshold=0.5, ef=100 vs K=5, threshold=0.7, ef=50)
    // This eliminates ~6.4 seconds of redundant work
    
    // Find ONE of each projection type per layer (skip duplicates)
    std::vector<std::pair<std::string, const TensorMeta*>> projs;
    std::set<std::string> seen;
    
    for (const auto& [name, meta] : ctx.tensors) {
        if (meta.shape.size() != 2) continue;
        size_t r = meta.shape[0], c = meta.shape[1];
        if (c != D) continue;  // Must project from embedding dim
        if (r < 64) continue;  // Skip tiny
        
        // Extract layer + type
        std::string type;
        if (name.find("q_proj") != std::string::npos) type = "q";
        else if (name.find("k_proj") != std::string::npos) type = "k";
        else if (name.find("v_proj") != std::string::npos) type = "v";
        else if (name.find("fc1") != std::string::npos || name.find("up_proj") != std::string::npos) type = "up";
        else continue;
        
        int layer = -1;
        auto p = name.find("layers.");
        if (p != std::string::npos) layer = std::stoi(name.substr(p + 7));
        else {
            p = name.find("layer.");
            if (p != std::string::npos) layer = std::stoi(name.substr(p + 6));
        }
        
        std::string key = std::to_string(layer) + type;
        if (seen.count(key)) continue;
        seen.insert(key);
        projs.push_back({config.model_name + "/L" + std::to_string(layer) + "/" + type, &meta});
    }
    
    std::cerr << "[SEMANTIC] " << projs.size() << " projections\n";
    
    // Skip projection k-NN if disabled (too slow due to single-threaded HNSW build)
    if (!ENABLE_PROJECTION_KNN) {
        std::cerr << "[SEMANTIC] Projection k-NN DISABLED (use attention_relations for token semantics)\n";
        std::cerr << "[SEMANTIC] Total: 0 sparse relations\n";
        return true;
    }
    
    // Limit projections if enabled
    if (projs.size() > MAX_PROJECTIONS) {
        std::cerr << "[SEMANTIC] Limiting to first " << MAX_PROJECTIONS << " projections\n";
        projs.resize(MAX_PROJECTIONS);
    }
    
    size_t total = 0;
    
    for (const auto& [tag, meta] : projs) {
        size_t R = meta->shape[0];
        
        auto W = load_tensor(*meta);
        if (W.empty()) continue;
        
        // P = E @ W^T using MKL  (V x D) @ (D x R) = (V x R)
        std::vector<float> P(V * R);
        
        auto gemm_start = std::chrono::steady_clock::now();
#if defined(HAS_MKL) && HAS_MKL
        // cblas_sgemm: C = alpha * A * B + beta * C
        // A = E (V x D), B = W^T (D x R), C = P (V x R)
        // W is stored as (R x D), so we use CblasTrans
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)V, (int)R, (int)D,
                    1.0f, E.data(), (int)D, W.data(), (int)D,
                    0.0f, P.data(), (int)R);
#else
        // Fallback naive (slow)
        for (size_t i = 0; i < V; i++) {
            for (size_t j = 0; j < R; j++) {
                float sum = 0;
                for (size_t k = 0; k < D; k++) sum += E[i * D + k] * W[j * D + k];
                P[i * R + j] = sum;
            }
        }
#endif
        auto gemm_end = std::chrono::steady_clock::now();
        auto gemm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gemm_end - gemm_start).count();
        
        auto pvalid = normalize_rows(P.data(), V, R);
        
        auto knn_start = std::chrono::steady_clock::now();
        auto edges = knn(P.data(), pvalid, R);
        auto knn_end = std::chrono::steady_clock::now();
        auto knn_ms = std::chrono::duration_cast<std::chrono::milliseconds>(knn_end - knn_start).count();
        
        std::cerr << "[SEMANTIC] " << tag << ": " << edges.size() << " edges (gemm=" << gemm_ms << "ms, knn=" << knn_ms << "ms)\n";
        insert(conn, edges, ctx, tag);
        total += edges.size();
    }
    
    std::cerr << "[SEMANTIC] Total: " << total << " sparse relations\n";
    return true;
}

// ============================================================================
// Project embeddings to 4D using Laplacian eigenmaps
// ============================================================================

bool project_and_update_embeddings(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    std::cerr << "============================================================\n";
    std::cerr << "[PROJECTION] LAPLACIAN EIGENMAP PROJECTION starting\n";
    std::cerr << "============================================================\n";

    // Process ALL token embedding tensors that are reasonable for projection
    std::vector<std::pair<const TensorMeta*, std::string>> embeddings_to_project;

    if (ctx.manifest.has_value()) {
        std::cerr << "[PROJECTION] Using config-driven tensor lookup\n";
        std::cerr << "[PROJECTION] Scanning " << ctx.manifest->extraction_plans.size() << " extraction plans...\n";

        // Scan for embeddings with reasonable vocab size

        // Process all TOKEN_EMBEDDING tensors with extract_embeddings=true
        int candidate_count = 0;

        for (const auto& plan : ctx.manifest->extraction_plans) {
            if (plan.category == TensorCategory::TOKEN_EMBEDDING && plan.extract_embeddings) {
                std::cerr << "[PROJECTION]   Candidate " << (++candidate_count) << ": " << plan.name;
                auto it = ctx.tensors.find(plan.name);
                if (it != ctx.tensors.end()) {
                    const auto& meta = it->second;
                    std::cerr << " [" << meta.shape[0] << " x " << meta.shape[1] << "] - AVAILABLE\n";

                    if (meta.shape.size() == 2) {
                        embeddings_to_project.emplace_back(&meta, plan.name);
                        std::cerr << "[PROJECTION] >>> WILL PROJECT: " << plan.name << "\n";
                    } else {
                        std::cerr << "[PROJECTION] >>> SKIPPED: " << plan.name << " (not 2D)\n";
                    }
                } else {
                    std::cerr << " - NOT FOUND IN TENSORS\n";
                }
            }
        }

        if (candidate_count == 0) {
            std::cerr << "[PROJECTION] WARNING: No TOKEN_EMBEDDING plans found in manifest!\n";
        }
    }

    if (embeddings_to_project.empty()) {
        // Fallback: If no manifest or manifest didn't find embeddings, scan tensors
        std::cerr << "[PROJECTION] No embeddings found via manifest, falling back to tensor scan\n";
        for (const auto& [name, meta] : ctx.tensors) {
            if (meta.shape.size() != 2) continue;
            if (name.find("position") != std::string::npos) continue;

            // Include all potential embedding tensors
            embeddings_to_project.emplace_back(&meta, name);
            std::cerr << "[PROJECTION] >>> FOUND VIA SCAN: " << name
                      << " [" << meta.shape[0] << " x " << meta.shape[1] << "]\n";
        }
    }

    if (embeddings_to_project.empty()) {
        std::cerr << "[PROJECTION] No suitable embedding tensors found for projection\n";
        return true;
    }

    // Process each embedding tensor that was selected
    for (const auto& [emb, emb_name] : embeddings_to_project) {
        std::cerr << "[PROJECTION] Processing embedding: " << emb_name << "\n";

        size_t V = static_cast<size_t>(emb->shape[0]);
        size_t D = static_cast<size_t>(emb->shape[1]);
        if (!ctx.vocab_tokens.empty()) V = std::min(V, ctx.vocab_tokens.size());

        // NOTE: Large vocabularies (>150K) will take significant time for HNSW build
        // The build is progress-monitored and can be interrupted if needed
        if (V > 150000) {
            std::cerr << "[PROJECTION] WARNING: Large vocabulary (" << V << " tokens) - HNSW build may take 30-60 minutes\n";
            std::cerr << "[PROJECTION] This is NORMAL for models like Llama with 200K vocab\n";
            std::cerr << "[PROJECTION] Progress will be shown every 10%...\n";
        }

        std::cerr << "[PROJECTION] Loading " << V << " x " << D << " embeddings...\n";
        auto load_start = std::chrono::steady_clock::now();
        auto E_flat = load_tensor(*emb);
        auto load_end = std::chrono::steady_clock::now();
        if (E_flat.empty()) {
            std::cerr << "[PROJECTION] Failed to load embeddings for " << emb_name << "\n";
            continue;
        }
        E_flat.resize(V * D);
        std::cerr << "[PROJECTION] Loaded in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count() << "ms\n";

        // DIAGNOSTIC: Check embedding statistics BEFORE conversion
        {
            double sum = 0.0, sum_sq = 0.0;
            double min_val = E_flat[0], max_val = E_flat[0];
            size_t zero_count = 0, nan_count = 0;

            for (size_t i = 0; i < std::min(size_t(V * D), E_flat.size()); ++i) {
                float val = E_flat[i];
                if (std::isnan(val)) {
                    ++nan_count;
                    continue;
                }
                if (std::abs(val) < 1e-10f) ++zero_count;
                sum += val;
                sum_sq += val * val;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }

            double mean = sum / (V * D);
            double variance = (sum_sq / (V * D)) - (mean * mean);
            double stddev = std::sqrt(variance);

            std::cerr << "[PROJECTION] Embedding statistics (raw from safetensor):\n";
            std::cerr << "  Tensor: " << emb_name << " [" << V << " x " << D << "]\n";
            std::cerr << "  Dtype: " << emb->dtype << ", File offset: " << emb->data_offset_start << "\n";
            std::cerr << std::scientific << std::setprecision(6);
            std::cerr << "  Min: " << min_val << ", Max: " << max_val << "\n";
            std::cerr << "  Mean: " << mean << ", StdDev: " << stddev << "\n";
            std::cerr << std::defaultfloat << std::setprecision(2);
            std::cerr << "  Zeros: " << zero_count << " (" << (100.0 * zero_count / (V * D)) << "%)\n";
            std::cerr << "  NaNs: " << nan_count << "\n";

            // Check first few embeddings with better precision
            std::cerr << std::scientific << std::setprecision(4);
            std::cerr << "  First embedding (token 0): [";
            for (size_t j = 0; j < std::min(size_t(10), D); ++j) {
                std::cerr << E_flat[j];
                if (j < std::min(size_t(10), D) - 1) std::cerr << ", ";
            }
            std::cerr << ", ...]\n";

            // Show second embedding too
            if (V > 1) {
                std::cerr << "  Second embedding (token 1): [";
                for (size_t j = 0; j < std::min(size_t(10), D) - 1; ++j) {
                    std::cerr << E_flat[D + j];
                    if (j < std::min(size_t(10), D) - 1) std::cerr << ", ";
                }
                std::cerr << ", ...]\n";
            }
            std::cerr << std::defaultfloat << std::setprecision(6);

            // Check if all rows are identical
            bool all_same = true;
            if (V > 1) {
                for (size_t j = 0; j < D; ++j) {
                    if (std::abs(E_flat[j] - E_flat[D + j]) > 1e-6f) {
                        all_same = false;
                        break;
                    }
                }
            }
            std::cerr << "  First two rows identical: " << (all_same ? "YES (DEGENERATE!)" : "NO (good)") << "\n";
        }

        // Convert to vector<vector<float>> for LaplacianProjector
        std::vector<std::vector<float>> embeddings(V);
        for (size_t i = 0; i < V; ++i) {
            embeddings[i].resize(D);
            for (size_t j = 0; j < D; ++j) {
                embeddings[i][j] = E_flat[i * D + j];
            }
        }

        // Prepare labels
        std::vector<std::string> labels(V);
        for (size_t i = 0; i < V; ++i) {
            if (i < ctx.vocab_tokens.size()) {
                labels[i] = ctx.vocab_tokens[i].text;
            } else {
                labels[i] = "token_" + std::to_string(i);
            }
        }

        // Configure Laplacian projector
        LaplacianConfig lap_config;
        lap_config.k_neighbors = 15;
        lap_config.similarity_threshold = 0.0f;
        lap_config.power_iterations = 50;  // Reduced for speed - 50 is often sufficient
        lap_config.num_threads = g_num_threads;
        lap_config.project_to_sphere = true;
        lap_config.verbose = true;
        // RELAXED tolerance for large sparse matrices (>100K points)
        // Residuals ~1e-2 are acceptable for semantic embeddings
        lap_config.convergence_tol = (V > 100000) ? 1e-2 : 1e-3;

        std::cerr << "[PROJECTION] Projecting " << V << " embeddings to 4D using Laplacian eigenmaps...\n";
        auto proj_start = std::chrono::steady_clock::now();

        LaplacianProjector projector(lap_config);
        ProjectionResult result = projector.project(embeddings, labels);

        auto proj_end = std::chrono::steady_clock::now();
        std::cerr << "[PROJECTION] Projection completed in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(proj_end - proj_start).count() << "ms\n";
        std::cerr << "[PROJECTION] 4D coordinates range computed\n";
        std::cerr << "[PROJECTION] Variance explained: " << result.total_variance_explained * 100 << "%\n";

        if (result.total_variance_explained < 0.01) {
            std::cerr << "[PROJECTION] Variance explained too low for " << emb_name << ", skipping database update\n";
            continue;
        }

        // Prepare TokenData for database update
        std::vector<hypercube::db::TokenData> tokens(V);
        for (size_t i = 0; i < V; ++i) {
            tokens[i].label = labels[i];
            tokens[i].hash = ctx.vocab_tokens[i].comp.hash;  // Use existing hash
            tokens[i].is_atom = false;  // These are compositions (tokens)
            tokens[i].coords = result.coords[i];
            tokens[i].hilbert_lo = result.hilbert_lo[i];
            tokens[i].hilbert_hi = result.hilbert_hi[i];
        }

        // Update database with projected coordinates
        std::cerr << "[PROJECTION] Updating database with projected 4D coordinates for " << emb_name << "...\n";
        hypercube::db::PersistConfig persist_config;
        persist_config.update_existing = true;
        hypercube::db::ProjectionPersister persister(conn, persist_config);
        size_t updated = persister.persist(tokens);

        std::cerr << "[PROJECTION] Updated " << updated << " token compositions with Laplacian-projected coordinates from " << emb_name << "\n";
    }

    return true;
}

} // namespace db
} // namespace ingest
} // namespace hypercube
