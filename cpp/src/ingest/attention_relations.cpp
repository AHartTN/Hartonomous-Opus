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
#include <sstream>
#include <unordered_map>

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

static bool extract_merge_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);
static bool extract_hierarchy_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);
static bool extract_attention_projection_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);
static bool ensure_dimension_compositions_exist(PGconn* conn, const std::string& component, int layer, const std::vector<int64_t>& dimensions);
static std::unordered_map<std::string, float> fetch_quality_scores(PGconn* conn, const std::string& model_name, const std::vector<std::string>& tensor_names);

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
            if (plan.category == TensorCategory::TOKEN_EMBEDDING ||
                plan.category == TensorCategory::ATTENTION_QUERY ||
                plan.category == TensorCategory::ATTENTION_KEY ||
                plan.category == TensorCategory::ATTENTION_VALUE ||
                plan.category == TensorCategory::ATTENTION_OUTPUT ||
                plan.category == TensorCategory::FFN_UP ||
                plan.category == TensorCategory::FFN_DOWN ||
                plan.category == TensorCategory::FFN_GATE ||
                plan.category == TensorCategory::LAYER_NORM ||
                plan.category == TensorCategory::RMS_NORM ||
                plan.category == TensorCategory::CONV_KERNEL) {
                auto it = ctx.tensors.find(plan.name);
                if (it != ctx.tensors.end()) {
                    embed = &it->second;
                    embed_name = plan.name;
                    std::cerr << "[SEMANTIC] Found " << category_to_string(plan.category) << " via manifest: " << plan.name << "\n";
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

    // Fetch quality scores for the embedding tensor
    std::vector<std::string> tensor_names = {embed_name};
    auto quality_scores = fetch_quality_scores(conn, config.model_name, tensor_names);
    float embed_quality = quality_scores.count(embed_name) ? quality_scores[embed_name] : 1.0f; // Default to 1.0 if not found
    std::cerr << "[SEMANTIC] Embedding quality score: " << embed_quality << "\n";
    
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

    // Direct bulk INSERT into relation_evidence
    std::string insert_sql = R"SQL(
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight)
        VALUES
    )SQL";
    std::vector<std::string> values;

    for (const auto& edges : thread_edges) {
        for (const auto& [i, j, sim] : edges) {
            size_t tok_i = valid_indices[i];
            size_t tok_j = valid_indices[j];

            const auto& comp_i = ctx.vocab_tokens[tok_i].comp;
            const auto& comp_j = ctx.vocab_tokens[tok_j].comp;

            std::string source_hex = "\\x" + comp_i.hash.to_hex();
            std::string target_hex = "\\x" + comp_j.hash.to_hex();

            // Normalize cosine similarity (already -1 to 1 for inner product)
            float normalized = sim;

            // Apply quality weighting: multiply by quality score squared for pairwise relations from same tensor
            normalized *= embed_quality * embed_quality;

            std::string val = "('" + source_hex + "', '" + target_hex + "', 'S', '" +
                              config.model_name + "', -1, 'embedding', 1500.0, 1, " +
                              std::to_string(sim) + ", " + std::to_string(normalized) + ")";
            values.push_back(val);
        }
    }

    // Batch in chunks to avoid query size limits
    const size_t batch_size = 1000;
    for (size_t i = 0; i < values.size(); i += batch_size) {
        std::string batch_sql = insert_sql;
        for (size_t j = i; j < std::min(i + batch_size, values.size()); ++j) {
            if (j > i) batch_sql += ", ";
            batch_sql += values[j];
        }
        batch_sql += R"SQL( ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
        DO UPDATE SET
            rating = relation_evidence.rating +
                     LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                     (
                         (EXCLUDED.normalized_weight + 1.0) / 2.0 -
                         (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))
                     ),
            observation_count = relation_evidence.observation_count + 1,
            raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                         (relation_evidence.observation_count + 1),
            normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                               (relation_evidence.observation_count + 1),
            last_updated = NOW()
    )SQL";

        Result res = exec(conn, batch_sql);
        if (!res.ok()) {
            std::cerr << "[SEMANTIC] Batch insert failed: " << res.error_message() << "\n";
            return false;
        }
    }
    
    tx.commit();
    
    std::cerr << "[SEMANTIC] Inserted " << total_edges << " token↔token semantic relations\n";
    std::cerr << "[SEMANTIC] These relations give tokens MEANING through semantic proximity\n";

    // =========================================================================
    // EXTRACT HIERARCHY RELATIONS FROM TENSOR NAMES
    // =========================================================================

    extract_hierarchy_relations(conn, ctx, config);

    // =========================================================================
    // EXTRACT MERGE RELATIONS FROM BPE/CPE MERGES
    // =========================================================================

    extract_merge_relations(conn, ctx, config);

    // =========================================================================
    // EXTRACT ATTENTION RELATIONS FROM PROJECTION MATRICES
    // =========================================================================

    extract_attention_projection_relations(conn, ctx, config);

    return true;
}

static bool extract_merge_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    std::cerr << "\n[MERGE] Extracting BPE/CPE merge relations...\n";

    if (ctx.bpe_merges.empty()) {
        std::cerr << "[MERGE] No BPE merges found\n";
        return true;
    }

    Transaction tx(conn);

    std::string insert_sql = R"SQL(
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight)
        VALUES
    )SQL";
    std::vector<std::string> values;

    size_t merge_edges = 0;

    // For each BPE merge, create M-relations between parts and merged token
    for (const auto& [left, right] : ctx.bpe_merges) {
        std::string merged = left + right;

        // Create compositions for the parts and merged token if they don't exist
        auto left_hash = AtomCalculator::compute_vocab_token(left).hash;
        auto right_hash = AtomCalculator::compute_vocab_token(right).hash;
        auto merged_hash = AtomCalculator::compute_vocab_token(merged).hash;

        // Create M-relations: left -> merged, right -> merged
        // Weight represents the merge strength (1.0 for direct merges)
        float weight = 1.0f;

        // left -> merged
        {
            std::string source_hex = "\\x" + left_hash.to_hex();
            std::string target_hex = "\\x" + merged_hash.to_hex();

            std::string val = "('" + source_hex + "', '" + target_hex + "', 'M', '" +
                              config.model_name + "', -1, 'merge', 1500.0, 1, " +
                              std::to_string(weight) + ", " + std::to_string(weight) + ")";
            values.push_back(val);
            merge_edges++;
        }

        // right -> merged
        {
            std::string source_hex = "\\x" + right_hash.to_hex();
            std::string target_hex = "\\x" + merged_hash.to_hex();

            std::string val = "('" + source_hex + "', '" + target_hex + "', 'M', '" +
                              config.model_name + "', -1, 'merge', 1500.0, 1, " +
                              std::to_string(weight) + ", " + std::to_string(weight) + ")";
            values.push_back(val);
            merge_edges++;
        }
    }

    // Batch insert merge relations
    if (!values.empty()) {
        const size_t batch_size = 1000;
        for (size_t i = 0; i < values.size(); i += batch_size) {
            std::string batch_sql = insert_sql;
            for (size_t j = i; j < std::min(i + batch_size, values.size()); ++j) {
                if (j > i) batch_sql += ", ";
                batch_sql += values[j];
            }
            batch_sql += R"SQL( ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
            DO UPDATE SET
                rating = relation_evidence.rating +
                         LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                         (
                             (EXCLUDED.normalized_weight + 1.0) / 2.0 -
                             (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))
                         ),
                observation_count = relation_evidence.observation_count + 1,
                raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                             (relation_evidence.observation_count + 1),
                normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                                    (relation_evidence.observation_count + 1),
                last_updated = NOW()
        )SQL";

            Result res = exec(conn, batch_sql);
            if (!res.ok()) {
                std::cerr << "[MERGE] Batch insert failed: " << res.error_message() << "\n";
                return false;
            }
        }
    }

    // Ensure merge token compositions exist
    if (!ctx.bpe_merges.empty()) {
        std::unordered_set<std::string> tokens_to_create;
        for (const auto& [left, right] : ctx.bpe_merges) {
            tokens_to_create.insert(left);
            tokens_to_create.insert(right);
            tokens_to_create.insert(left + right);
        }

        std::string comp_insert = "INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi) VALUES ";
        std::vector<std::string> comp_values;

        for (const std::string& token : tokens_to_create) {
            CompositionRecord comp = AtomCalculator::compute_vocab_token(token);
            std::string hex_id = "\\x" + comp.hash.to_hex();
            std::string centroid = "010100002000000000000000000000000000000000000000000000000000000000000000";  // POINT(0 0 0 0)
            comp_values.push_back("('" + hex_id + "', '" + token + "', 1, 1, 1, '" + centroid + "', 0, 0)");
        }

        if (!comp_values.empty()) {
            for (size_t i = 0; i < comp_values.size(); ++i) {
                if (i > 0) comp_insert += ", ";
                comp_insert += comp_values[i];
            }
            comp_insert += " ON CONFLICT (id) DO NOTHING";

            Result comp_res = exec(conn, comp_insert);
            if (!comp_res.ok()) {
                std::cerr << "[MERGE] Failed to insert merge token compositions: " << comp_res.error_message() << "\n";
                return false;
            }
        }
    }

    tx.commit();

    std::cerr << "[MERGE] Inserted " << merge_edges << " merge relations\n";

    return true;
}

static bool extract_hierarchy_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    std::cerr << "\n[HIERARCHY] Extracting tensor hierarchy relations...\n";

    Transaction tx(conn);

    std::string insert_sql = R"SQL(
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight)
        VALUES
    )SQL";
    // Use map to deduplicate edges (multiple tensors share parent paths)
    struct EdgeKey {
        std::string parent_hex;
        std::string child_hex;
        int layer;
        bool operator<(const EdgeKey& other) const {
            if (parent_hex != other.parent_hex) return parent_hex < other.parent_hex;
            if (child_hex != other.child_hex) return child_hex < other.child_hex;
            return layer < other.layer;
        }
    };
    std::map<EdgeKey, float> unique_edges;

    // Process each tensor to build parent-child relations
    for (const auto& [tensor_name, tensor] : ctx.tensors) {
        // Parse tensor name into components
        std::vector<std::string> parts;
        std::stringstream ss(tensor_name);
        std::string part;
        while (std::getline(ss, part, '.')) {
            parts.push_back(part);
        }

        if (parts.size() < 2) continue;

        // Build hierarchical path
        std::string current_path;
        for (size_t i = 0; i < parts.size(); ++i) {
            std::string parent_path = current_path;
            if (!current_path.empty()) current_path += ".";
            current_path += parts[i];

            if (i > 0) {  // Not the root
                // Create H-relation from parent to child
                auto parent_hash = AtomCalculator::compute_vocab_token(parent_path).hash;
                auto child_hash = AtomCalculator::compute_vocab_token(current_path).hash;

                std::string parent_hex = "\\x" + parent_hash.to_hex();
                std::string child_hex = "\\x" + child_hash.to_hex();

                // Parse layer number if present
                int layer = -1;
                if (parent_path.find("layers.") != std::string::npos) {
                    size_t layer_pos = parent_path.find("layers.");
                    if (layer_pos != std::string::npos) {
                        size_t num_start = layer_pos + 7;
                        size_t num_end = parent_path.find(".", num_start);
                        if (num_end != std::string::npos) {
                            try {
                                layer = std::stoi(parent_path.substr(num_start, num_end - num_start));
                            } catch (...) {}
                        }
                    }
                }

                // Weight = 1.0 for direct hierarchy (strong relation)
                float weight = 1.0f;

                EdgeKey key{parent_hex, child_hex, layer};
                unique_edges[key] = weight;
            }
        }
    }

    // Convert unique edges to values for insertion
    std::vector<std::string> values;
    size_t hierarchy_edges = 0;
    for (const auto& [key, weight] : unique_edges) {
        std::string val = "('" + key.parent_hex + "', '" + key.child_hex + "', 'H', '" +
                          config.model_name + "', " + std::to_string(key.layer) + ", 'hierarchy', 1500.0, 1, " +
                          std::to_string(weight) + ", " + std::to_string(weight) + ")";
        values.push_back(val);
        hierarchy_edges++;
    }

    // Batch insert hierarchy relations
    if (!values.empty()) {
        const size_t batch_size = 1000;
        for (size_t i = 0; i < values.size(); i += batch_size) {
            std::string batch_sql = insert_sql;
            for (size_t j = i; j < std::min(i + batch_size, values.size()); ++j) {
                if (j > i) batch_sql += ", ";
                batch_sql += values[j];
            }
            batch_sql += R"SQL( ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
            DO UPDATE SET
                rating = relation_evidence.rating +
                         LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                         (
                             (EXCLUDED.normalized_weight + 1.0) / 2.0 -
                             (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))
                         ),
                observation_count = relation_evidence.observation_count + 1,
                raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                             (relation_evidence.observation_count + 1),
                normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                                    (relation_evidence.observation_count + 1),
                last_updated = NOW()
        )SQL";

            Result res = exec(conn, batch_sql);
            if (!res.ok()) {
                std::cerr << "[HIERARCHY] Batch insert failed: " << res.error_message() << "\n";
                return false;
            }
        }
    }

    tx.commit();

    std::cerr << "[HIERARCHY] Inserted " << hierarchy_edges << " hierarchy relations\n";

    return true;
}

static bool extract_attention_projection_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    std::cerr << "\n[ATTENTION] Extracting attention projection relations...\n";

    // Find attention projection tensors
    std::vector<std::pair<std::string, const TensorMeta*>> attn_tensors;
    std::vector<std::string> attn_tensor_names;

    for (const auto& [name, meta] : ctx.tensors) {
        if ((name.find("attn.q_proj.weight") != std::string::npos ||
             name.find("attn.k_proj.weight") != std::string::npos ||
             name.find("attn.v_proj.weight") != std::string::npos ||
             name.find("attn.o_proj.weight") != std::string::npos ||
             name.find("self_attn.q_proj.weight") != std::string::npos ||
             name.find("self_attn.k_proj.weight") != std::string::npos ||
             name.find("self_attn.v_proj.weight") != std::string::npos ||
             name.find("self_attn.o_proj.weight") != std::string::npos) &&
            meta.shape.size() == 2) {
            attn_tensors.emplace_back(name, &meta);
        }
    }

    if (attn_tensors.empty()) {
        std::cerr << "[ATTENTION] No attention projection tensors found\n";
        return true;
    }

    std::cerr << "[ATTENTION] Found " << attn_tensors.size() << " attention projection tensors\n";

    size_t total_attn_edges = 0;

    for (const auto& [tensor_name, tensor] : attn_tensors) {
        // Parse layer number from tensor name
        int layer = -1;
        size_t layers_pos = tensor_name.find("layers.");
        if (layers_pos != std::string::npos) {
            size_t num_start = layers_pos + 7;
            size_t num_end = tensor_name.find(".", num_start);
            if (num_end != std::string::npos) {
                layer = std::stoi(tensor_name.substr(num_start, num_end - num_start));
            }
        }

        // Determine component type
        std::string component;
        if (tensor_name.find("q_proj") != std::string::npos) component = "q_proj";
        else if (tensor_name.find("k_proj") != std::string::npos) component = "k_proj";
        else if (tensor_name.find("v_proj") != std::string::npos) component = "v_proj";
        else if (tensor_name.find("o_proj") != std::string::npos) component = "o_proj";
        else component = "attn_proj";

        int64_t out_dim = tensor->shape[0];
        int64_t in_dim = tensor->shape[1];

        // Sample rows to keep computation tractable (attention matrices are large)
        int64_t max_rows = std::min(out_dim, static_cast<int64_t>(512));
        int64_t stride = std::max(static_cast<int64_t>(1), out_dim / max_rows);

        std::cerr << "  " << tensor_name << " [" << out_dim << " x " << in_dim << "] layer=" << layer << " component=" << component;
        if (stride > 1) std::cerr << " (sampling every " << stride << " rows)";
        std::cerr << "\n";

        // Read sampled rows
        std::vector<std::vector<float>> rows;
        std::vector<int64_t> row_indices;
        rows.reserve(max_rows);
        row_indices.reserve(max_rows);

        for (int64_t i = 0; i < out_dim; i += stride) {
            auto row = read_tensor_row(*tensor, static_cast<size_t>(i));
            if (!row.empty()) {
                rows.push_back(std::move(row));
                row_indices.push_back(i);
            }
        }

        if (rows.size() < 2) continue;

        // Build k-NN similarity for attention projection rows
        const int k_neighbors = 8;
        std::vector<std::tuple<size_t, size_t, float>> edges;

        for (size_t i = 0; i < rows.size(); ++i) {
            std::vector<std::pair<float, size_t>> neighbors;
            for (size_t j = 0; j < rows.size(); ++j) {
                if (i == j) continue;
                // Compute cosine similarity
                float sim = 0.0f;
                float norm_i = 0.0f, norm_j = 0.0f;
                for (int64_t d = 0; d < in_dim; ++d) {
                    sim += rows[i][d] * rows[j][d];
                    norm_i += rows[i][d] * rows[i][d];
                    norm_j += rows[j][d] * rows[j][d];
                }
                norm_i = std::sqrt(norm_i);
                norm_j = std::sqrt(norm_j);
                if (norm_i > 1e-6f && norm_j > 1e-6f) {
                    sim /= (norm_i * norm_j);
                }
                neighbors.emplace_back(sim, j);
            }

            std::partial_sort(neighbors.begin(),
                              neighbors.begin() + std::min(static_cast<size_t>(k_neighbors), neighbors.size()),
                              neighbors.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });

            for (size_t k = 0; k < std::min(static_cast<size_t>(k_neighbors), neighbors.size()); ++k) {
                float sim = neighbors[k].first;
                size_t j = neighbors[k].second;
                if (sim >= 0.1f && i < j) {  // Lower threshold for attention projections
                    edges.emplace_back(i, j, sim);
                }
            }
        }

        if (edges.empty()) continue;

        // Insert A-relations for attention projection similarities
        Transaction tx(conn);

        std::string insert_sql = R"SQL(
            INSERT INTO relation_evidence
                (source_id, target_id, relation_type, source_model, layer, component,
                 rating, observation_count, raw_weight, normalized_weight)
            VALUES
        )SQL";
        std::vector<std::string> values;

        for (const auto& [i, j, sim] : edges) {
            // Create dimension composition hashes for attention projection dimensions
            std::string src_key = component + ":" + std::to_string(layer) + ":dim" + std::to_string(row_indices[i]);
            std::string tgt_key = component + ":" + std::to_string(layer) + ":dim" + std::to_string(row_indices[j]);

            auto src_hash = AtomCalculator::compute_vocab_token(src_key).hash;
            auto tgt_hash = AtomCalculator::compute_vocab_token(tgt_key).hash;

            std::string source_hex = "\\x" + src_hash.to_hex();
            std::string target_hex = "\\x" + tgt_hash.to_hex();

            float normalized = sim;  // Cosine similarity

            std::string val = "('" + source_hex + "', '" + target_hex + "', 'A', '" +
                              config.model_name + "', " + std::to_string(layer) + ", '" + component + "', 1500.0, 1, " +
                              std::to_string(sim) + ", " + std::to_string(normalized) + ")";
            values.push_back(val);
        }

        // Batch insert
        const size_t batch_size = 1000;
        for (size_t i = 0; i < values.size(); i += batch_size) {
            std::string batch_sql = insert_sql;
            for (size_t j = i; j < std::min(i + batch_size, values.size()); ++j) {
                if (j > i) batch_sql += ", ";
                batch_sql += values[j];
            }
            batch_sql += R"SQL( ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component)
            DO UPDATE SET
                rating = relation_evidence.rating +
                         LEAST(32.0 * POWER(0.95, relation_evidence.observation_count), 4.0) *
                         (
                             (EXCLUDED.normalized_weight + 1.0) / 2.0 -
                             (1.0 / (1.0 + EXP(-(relation_evidence.rating - 1500.0) / 400.0)))
                         ),
                observation_count = relation_evidence.observation_count + 1,
                raw_weight = (relation_evidence.raw_weight * relation_evidence.observation_count + EXCLUDED.raw_weight) /
                             (relation_evidence.observation_count + 1),
                normalized_weight = (relation_evidence.normalized_weight * relation_evidence.observation_count + EXCLUDED.normalized_weight) /
                                    (relation_evidence.observation_count + 1),
                last_updated = NOW()
        )SQL";

            Result res = exec(conn, batch_sql);
            if (!res.ok()) {
                std::cerr << "[ATTENTION] Batch insert failed: " << res.error_message() << "\n";
                return false;
            }
        }

        tx.commit();

        std::cerr << "    -> " << edges.size() << " attention projection edges\n";
        total_attn_edges += edges.size();

        // Ensure dimension compositions exist
        ensure_dimension_compositions_exist(conn, component, layer, row_indices);
    }

    std::cerr << "[ATTENTION] Total: " << total_attn_edges << " attention projection relations\n";

    return true;
}

static bool ensure_dimension_compositions_exist(PGconn* conn, const std::string& component, int layer, const std::vector<int64_t>& dimensions) {
    // Create compositions for dimension atoms if they don't exist
    Transaction tx(conn);

    std::string insert_sql = "INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi) VALUES ";
    std::vector<std::string> inserts;

    for (int64_t dim : dimensions) {
        std::string key = component + ":" + std::to_string(layer) + ":dim" + std::to_string(dim);
        CompositionRecord comp = AtomCalculator::compute_vocab_token(key);
        std::string hex_id = "\\x" + comp.hash.to_hex();

        // Build centroid EWKB (simplified point)
        std::string centroid = "010100002000000000000000000000000000000000000000000000000000000000000000";  // POINT(0 0 0 0)

        inserts.push_back("('" + hex_id + "', '" + key + "', 1, 1, 1, '" + centroid + "', 0, 0)");
    }

    if (!inserts.empty()) {
        for (size_t i = 0; i < inserts.size(); ++i) {
            if (i > 0) insert_sql += ", ";
            insert_sql += inserts[i];
        }
        insert_sql += " ON CONFLICT (id) DO NOTHING";

        Result ins_res = exec(conn, insert_sql);
        if (!ins_res.ok()) {
            std::cerr << "[ATTENTION] Failed to insert dimension compositions: " << ins_res.error_message() << "\n";
            return false;
        }
    }

    tx.commit();
    return true;
}

static std::unordered_map<std::string, float> fetch_quality_scores(PGconn* conn, const std::string& model_name, const std::vector<std::string>& tensor_names) {
    std::unordered_map<std::string, float> quality_scores;

    if (tensor_names.empty()) return quality_scores;

    // Build IN clause for tensor names (escape single quotes)
    std::string tensor_list;
    for (size_t i = 0; i < tensor_names.size(); ++i) {
        if (i > 0) tensor_list += ",";
        // Escape single quotes by doubling them
        std::string escaped_name = tensor_names[i];
        size_t pos = 0;
        while ((pos = escaped_name.find("'", pos)) != std::string::npos) {
            escaped_name.replace(pos, 1, "''");
            pos += 2;
        }
        tensor_list += "'" + escaped_name + "'";
    }

    // Query projection_metadata for quality scores
    std::string sql = R"SQL(
        SELECT pm.tensor_name, pm.quality_score
        FROM projection_metadata pm
        JOIN model m ON m.id = pm.model_id
        WHERE m.name = ')SQL" + model_name + R"SQL('
        AND pm.tensor_name IN ()SQL" + tensor_list + R"SQL()
        AND pm.quality_score IS NOT NULL
    )SQL";

    Result res = exec(conn, sql);
    if (!res.ok()) {
        std::cerr << "[QUALITY] Failed to fetch quality scores: " << res.error_message() << "\n";
        return quality_scores;
    }

    int ntuples = PQntuples(res.get());
    for (int i = 0; i < ntuples; ++i) {
        std::string tensor_name = PQgetvalue(res.get(), i, 0);
        std::string quality_str = PQgetvalue(res.get(), i, 1);
        try {
            float quality = std::stof(quality_str);
            quality_scores[tensor_name] = quality;
        } catch (const std::exception& e) {
            std::cerr << "[QUALITY] Invalid quality score for " << tensor_name << ": " << quality_str << "\n";
        }
    }

    std::cerr << "[QUALITY] Fetched " << quality_scores.size() << " quality scores for model " << model_name << "\n";
    return quality_scores;
}

} // namespace db
} // namespace ingest
} // namespace hypercube
