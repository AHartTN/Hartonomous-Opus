/**
 * @file attention_relations.cpp
 * @brief Extract semantic relations from model embeddings using HNSW
 *
 * CONFIG-DRIVEN: Uses manifest to find tensors by ROLE, not hardcoded patterns.
 * PARALLEL: Uses all available CPU threads via OpenMP.
 * EFFICIENT: Uses HNSW for k-NN instead of O(N²) brute force.
 *
 * The model's embedding matrix encodes TOKEN↔TOKEN semantic relationships
 * learned from training. We extract these as relations between TOKEN COMPOSITIONS
 * which are REAL entities (atom trajectories) that already exist in the database.
 */

#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/helpers.hpp"
#include <thread>
#include <atomic>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <filesystem>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAS_HNSWLIB
#include <hnswlib/hnswlib.h>
#endif

namespace hypercube {
namespace ingest {
namespace db {

using namespace hypercube::db;

// ============================================================================
// HNSW TUNING PARAMETERS
// ============================================================================
// M: Max edges per node. Higher = better recall, more memory, slower build.
//    Typical: 12-48. 16 is a good default.
// ef_construction: Build-time search depth. Higher = better graph quality.
//    Typical: 100-400. 200 is high quality.
// ef_search: Query-time search depth. Higher = better recall, slower queries.
//    Typical: 50-200. 100 gives ~95%+ recall.
// K_NEIGHBORS: How many semantic neighbors to store per token.
// MIN_SIMILARITY: Cosine threshold (normalized IP). 0.5 = 60° angle max.
// ENABLE_HNSW_CACHE: Set to true to cache HNSW indices to disk.
// ============================================================================
static constexpr size_t HNSW_M = 16;
static constexpr size_t HNSW_EF_CONSTRUCTION = 200;
static constexpr size_t HNSW_EF_SEARCH = 100;
static constexpr size_t K_NEIGHBORS = 15;
static constexpr float MIN_SIMILARITY = 0.5f;
static constexpr bool ENABLE_HNSW_CACHE = true;

// ============================================================================
// HNSW CACHE HELPERS
// ============================================================================

static std::string get_cache_path(const std::string& model_name, size_t n, size_t dim) {
    // Create a unique cache key from model name, vector count, and dimension
    std::hash<std::string> hasher;
    size_t h = hasher(model_name) ^ (n * 31) ^ (dim * 17);
    
    // Use system temp directory
    std::filesystem::path cache_dir = std::filesystem::temp_directory_path() / "hypercube_hnsw_cache";
    std::filesystem::create_directories(cache_dir);
    
    return (cache_dir / (std::to_string(h) + ".hnsw")).string();
}

bool insert_attention_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    // =========================================================================
    // CONFIG-DRIVEN TOKEN EMBEDDING EXTRACTION
    // =========================================================================
    
    if (ctx.vocab_tokens.empty()) {
        std::cerr << "[SEMANTIC] No vocabulary tokens loaded, skipping semantic extraction\n";
        return true;
    }
    
    // Detect thread count - use ALL available threads
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads < 1) num_threads = 8;
    
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
    
    // =========================================================================
    // FIND EMBEDDING TENSOR - CONFIG DRIVEN
    // Use manifest's categorized extraction plans instead of hardcoded patterns
    // =========================================================================
    
    const TensorMeta* embed = nullptr;
    std::string embed_name;
    
    // Try manifest-based lookup first - the config tells us the architecture
    if (ctx.manifest.has_value()) {
        std::cerr << "[SEMANTIC] Using config-driven lookup (arch: " 
                  << architecture_to_string(ctx.manifest->architecture) << ")\n";
        
        for (const auto& plan : ctx.manifest->extraction_plans) {
            if (plan.category == TensorCategory::TOKEN_EMBEDDING) {
                auto it = ctx.tensors.find(plan.name);
                if (it != ctx.tensors.end()) {
                    embed = &it->second;
                    embed_name = plan.name;
                    std::cerr << "[SEMANTIC] Found TOKEN_EMBEDDING via manifest: " << plan.name << "\n";
                    break;
                }
            }
        }
    }
    
    // Fallback: dimension-based matching using config vocab_size
    if (!embed) {
        std::cerr << "[SEMANTIC] WARNING: Manifest lookup failed, falling back to dimension match\n";
        size_t expected_vocab = ctx.get_vocab_size();
        for (auto& [name, meta] : ctx.tensors) {
            if (meta.shape.size() != 2) continue;
            if (name.find("position") != std::string::npos) continue;
            
            if (expected_vocab > 0 && meta.shape[0] == static_cast<int64_t>(expected_vocab)) {
                embed = &meta;
                embed_name = name;
                std::cerr << "[SEMANTIC] Found embedding by vocab dimension: " << name << "\n";
                break;
            }
        }
    }
    
    if (!embed) {
        std::cerr << "[SEMANTIC] No token embedding tensor found\n";
        return true;
    }
    
    int64_t model_vocab = embed->shape[0];
    int64_t embed_dim = embed->shape[1];
    int64_t actual_vocab = static_cast<int64_t>(ctx.vocab_tokens.size());
    int64_t vocab_size = std::min(model_vocab, actual_vocab);
    
    std::cerr << "\n[SEMANTIC] Extracting token↔token similarity from " << embed_name << "\n";
    std::cerr << "[SEMANTIC] Vocab: " << vocab_size << " tokens, Embedding dim: " << embed_dim << "\n";
    std::cerr << "[SEMANTIC] Using " << num_threads << " threads\n";
    
    // =========================================================================
    // LOAD AND NORMALIZE EMBEDDINGS
    // =========================================================================
    
    auto load_start = std::chrono::steady_clock::now();
    
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(vocab_size);
    
    std::vector<size_t> valid_indices;
    valid_indices.reserve(vocab_size);
    
    for (int64_t i = 0; i < vocab_size; ++i) {
        auto emb = read_tensor_row(*embed, static_cast<size_t>(i));
        if (!emb.empty() && !ctx.vocab_tokens[i].comp.hash.is_zero()) {
            // Normalize for cosine similarity
            float norm = 0.0f;
            for (float v : emb) norm += v * v;
            norm = std::sqrt(norm);
            if (norm > 1e-6f) {
                for (float& v : emb) v /= norm;
                embeddings.push_back(std::move(emb));
                valid_indices.push_back(static_cast<size_t>(i));
            }
        }
    }
    
    auto load_end = std::chrono::steady_clock::now();
    std::cerr << "[SEMANTIC] Loaded " << embeddings.size() << " token embeddings in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count() << "ms\n";
    
    if (embeddings.size() < 2) {
        std::cerr << "[SEMANTIC] Not enough valid embeddings\n";
        return true;
    }
    
    // =========================================================================
    // BUILD k-NN GRAPH USING HNSW (NOT brute force O(N²))
    // =========================================================================
    
    size_t n = embeddings.size();
    std::vector<std::vector<std::tuple<size_t, size_t, float>>> thread_edges(num_threads);
    
#ifdef HAS_HNSWLIB
    auto hnsw_start = std::chrono::steady_clock::now();
    
    // Use inner product space (normalized vectors = cosine similarity)
    hnswlib::InnerProductSpace space(static_cast<size_t>(embed_dim));
    
    // Check for cached index
    std::string cache_path = get_cache_path(config.model_name, n, static_cast<size_t>(embed_dim));
    bool loaded_from_cache = false;
    
    // Use tunable parameters from top of file
    hnswlib::HierarchicalNSW<float> idx(&space, n, HNSW_M, HNSW_EF_CONSTRUCTION);
    
    if (ENABLE_HNSW_CACHE && std::filesystem::exists(cache_path)) {
        try {
            std::cerr << "[SEMANTIC] Loading cached HNSW index from " << cache_path << "...\n";
            idx.loadIndex(cache_path, &space, n);
            loaded_from_cache = true;
            std::cerr << "[SEMANTIC] Loaded cached index in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - hnsw_start).count() << "ms\n";
        } catch (const std::exception& e) {
            std::cerr << "[SEMANTIC] Cache load failed (" << e.what() << "), rebuilding...\n";
            loaded_from_cache = false;
        }
    }
    
    if (!loaded_from_cache) {
        // Build index (sequential - addPoint is NOT thread-safe)
        std::cerr << "[SEMANTIC] Building HNSW index (" << n << " vectors)...\n";
        size_t progress_interval = std::max<size_t>(n / 10, 1);
        for (size_t i = 0; i < n; ++i) {
            idx.addPoint(embeddings[i].data(), i);
            if ((i + 1) % progress_interval == 0) {
                std::cerr << "  [BUILD] " << (i + 1) << "/" << n << " (" << ((i + 1) * 100 / n) << "%)\n";
            }
        }
        
        // Save to cache for future runs
        if (ENABLE_HNSW_CACHE) {
            try {
                idx.saveIndex(cache_path);
                std::cerr << "[SEMANTIC] Cached index to " << cache_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "[SEMANTIC] Cache save failed: " << e.what() << "\n";
            }
        }
    }
    
    auto hnsw_build = std::chrono::steady_clock::now();
    std::cerr << "[SEMANTIC] HNSW index " << (loaded_from_cache ? "loaded" : "built") << " in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(hnsw_build - hnsw_start).count() << "ms\n";
    
    // Query in parallel - searchKnn IS thread-safe
    idx.setEf(HNSW_EF_SEARCH);
    std::atomic<size_t> progress{0};
    size_t progress_interval = std::max<size_t>(n / 10, 1);
    
    std::cerr << "[SEMANTIC] Querying k-NN (PARALLEL, " << num_threads << " threads)...\n";
    
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& local_edges = thread_edges[tid];
        local_edges.reserve(n * K_NEIGHBORS / num_threads);
        
        #pragma omp for schedule(dynamic, 256)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            auto result = idx.searchKnn(embeddings[i].data(), K_NEIGHBORS + 1);
            
            while (!result.empty()) {
                auto [dist, j] = result.top();
                result.pop();
                
                if (j == static_cast<size_t>(i)) continue;  // Skip self
                
                // Inner product similarity (already normalized)
                float sim = dist;  // For IP space, dist IS the similarity
                if (sim >= MIN_SIMILARITY && static_cast<size_t>(i) < j) {
                    local_edges.emplace_back(static_cast<size_t>(i), j, sim);
                }
            }
            
            size_t done = progress.fetch_add(1) + 1;
            if (done % progress_interval == 0) {
                std::cerr << "  [QUERY] " << done << "/" << n << " (" << (done * 100 / n) << "%)\n";
            }
        }
    }
    
    auto hnsw_query = std::chrono::steady_clock::now();
    std::cerr << "[SEMANTIC] k-NN queries completed in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(hnsw_query - hnsw_build).count() << "ms\n";
    
#else
    // Fallback: brute force if HNSW not available (uses ALL threads, not capped)
    std::cerr << "[SEMANTIC] WARNING: HNSW not available, using brute force (" << num_threads << " threads)\n";
    
    std::atomic<size_t> work_idx{0};
    size_t progress_interval = std::max<size_t>(n / 10, 1);
    
    auto bf_worker = [&](int tid) {
        auto& local_edges = thread_edges[tid];
        std::vector<std::pair<float, size_t>> neighbors;
        neighbors.reserve(n);
        
        while (true) {
            size_t i = work_idx.fetch_add(1);
            if (i >= n) break;
            
            if (i % progress_interval == 0) {
                std::cerr << "  [BF] " << i << "/" << n << " (" << (i * 100 / n) << "%)\n";
            }
            
            neighbors.clear();
            const auto& emb_i = embeddings[i];
            
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                float sim = 0.0f;
                for (size_t d = 0; d < static_cast<size_t>(embed_dim); ++d) {
                    sim += emb_i[d] * embeddings[j][d];
                }
                if (sim >= MIN_SIMILARITY) {
                    neighbors.emplace_back(sim, j);
                }
            }
            
            if (neighbors.empty()) continue;
            
            std::partial_sort(neighbors.begin(),
                              neighbors.begin() + std::min(K_NEIGHBORS, neighbors.size()),
                              neighbors.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            
            for (size_t k = 0; k < std::min(K_NEIGHBORS, neighbors.size()); ++k) {
                size_t j = neighbors[k].second;
                float sim = neighbors[k].first;
                if (i < j) {
                    local_edges.emplace_back(i, j, sim);
                }
            }
        }
    };
    
    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back(bf_worker, t);
    }
    for (auto& th : workers) th.join();
#endif
    
    // Count total edges
    size_t total_edges = 0;
    for (const auto& edges : thread_edges) total_edges += edges.size();
    
    std::cerr << "[SEMANTIC] Found " << total_edges << " semantic similarity edges\n";
    
    if (total_edges == 0) return true;
    
    // =========================================================================
    // INSERT RELATIONS
    // =========================================================================
    
    Transaction tx(conn);
    
    Result res = exec(conn,
        "CREATE TEMP TABLE tmp_semantic ("
        "  source_type CHAR(1), source_id BYTEA,"
        "  target_type CHAR(1), target_id BYTEA,"
        "  weight REAL"
        ") ON COMMIT DROP");
    
    CopyStream copy(conn, "COPY tmp_semantic FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    
    std::string batch;
    batch.reserve(1 << 20);
    
    for (const auto& edges : thread_edges) {
        for (const auto& [i, j, sim] : edges) {
            size_t tok_i = valid_indices[i];
            size_t tok_j = valid_indices[j];
            
            const auto& comp_i = ctx.vocab_tokens[tok_i].comp;
            const auto& comp_j = ctx.vocab_tokens[tok_j].comp;
            
            char type_i = (comp_i.children.size() <= 1) ? 'A' : 'C';
            char type_j = (comp_j.children.size() <= 1) ? 'A' : 'C';
            
            batch += type_i;
            batch += "\t";
            copy_bytea(batch, comp_i.hash);
            batch += "\t";
            batch += type_j;
            batch += "\t";
            copy_bytea(batch, comp_j.hash);
            batch += "\t";
            batch += std::to_string(sim) + "\n";
            
            if (batch.size() > (1 << 19)) {
                copy.put(batch);
                batch.clear();
            }
        }
    }
    
    if (!batch.empty()) copy.put(batch);
    copy.end();
    
    std::string insert_sql = 
        "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
        "SELECT source_type, source_id, target_type, target_id, 'S', weight, '" + config.model_name + "', 1, -1, 'embedding' FROM tmp_semantic "
        "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
        "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
        "  source_count = relation.source_count + 1";
    
    res = exec(conn, insert_sql);
    if (!res.ok()) {
        std::cerr << "[SEMANTIC] Insert failed: " << res.error_message() << "\n";
        return false;
    }
    
    tx.commit();
    
    std::cerr << "[SEMANTIC] Inserted " << total_edges << " token↔token semantic relations\n";
    std::cerr << "[SEMANTIC] These relations give tokens MEANING through semantic proximity\n";
    
    return true;
}

} // namespace db
} // namespace ingest
} // namespace hypercube
