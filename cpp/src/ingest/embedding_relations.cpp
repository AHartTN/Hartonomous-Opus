/**
 * @file embedding_relations.cpp
 * @brief Extract k-NN similarity relations from embedding tensors
 *
 * Uses HNSWLIB to build efficient k-NN graphs from model embeddings
 * and inserts similarity edges as relation records.
 *
 * NOW LANGUAGE-AGNOSTIC: Uses tensor shapes and structure instead of English names
 */

#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/tensor_classifier.hpp"
#include <unordered_map>

namespace hypercube {
namespace ingest {
namespace db {

using namespace hypercube::db;

static std::unordered_map<std::string, float> fetch_quality_scores(PGconn* conn, const std::string& model_name, const std::vector<std::string>& tensor_names);

bool extract_embedding_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    // Find embedding tensors using LANGUAGE-AGNOSTIC shape analysis
    std::vector<std::pair<std::string, TensorMeta*>> embed_tensors;

    // First pass: analyze all tensors to determine model statistics
    std::vector<hypercube::ingest::TensorShape> all_shapes;
    for (auto& [name, meta] : ctx.tensors) {
        all_shapes.push_back({meta.shape});
    }
    hypercube::ingest::TensorClassifier classifier;
    classifier.update_model_stats(all_shapes);

    // Second pass: classify embedding tensors by shape
    for (auto& [name, meta] : ctx.tensors) {
        hypercube::ingest::TensorShape shape{meta.shape};
        hypercube::ingest::ClassificationContext context =
            hypercube::ingest::TensorClassifier::analyze_path(name);

        hypercube::ingest::TensorComponent comp_type =
            hypercube::ingest::TensorClassifier::classify(shape, context);

        // Map component types to embedding categories
        switch (comp_type) {
            case hypercube::ingest::TensorComponent::TOKEN_EMBEDDINGS:
                embed_tensors.emplace_back("token", &meta);
                break;
            case hypercube::ingest::TensorComponent::POSITION_EMBEDDINGS:
                embed_tensors.emplace_back("position", &meta);
                break;
            case hypercube::ingest::TensorComponent::PATCH_EMBEDDING:
                embed_tensors.emplace_back("patch", &meta);
                break;
            case hypercube::ingest::TensorComponent::VISUAL_PROJECTION:
                embed_tensors.emplace_back("projection", &meta);
                break;
            default:
                // Keep legacy string matching as fallback for now
                if (name.find("embed_tokens") != std::string::npos ||
                    name.find("word_embeddings") != std::string::npos ||
                    name.find("wte.weight") != std::string::npos ||
                    name.find("token_embedding") != std::string::npos) {
                    embed_tensors.emplace_back("token", &meta);
                }
                else if (name.find("patch_embed") != std::string::npos ||
                         name.find("patch_embedding") != std::string::npos ||
                         (name.find("proj.weight") != std::string::npos && name.find("patch") != std::string::npos)) {
                    embed_tensors.emplace_back("patch", &meta);
                }
                else if (name.find("text_projection") != std::string::npos ||
                         name.find("visual_projection") != std::string::npos) {
                    embed_tensors.emplace_back("projection", &meta);
                }
                else if (name.find("position_embed") != std::string::npos ||
                         name.find("pos_embed") != std::string::npos ||
                         name.find("query_position") != std::string::npos) {
                    embed_tensors.emplace_back("position", &meta);
                }
                break;
        }
    }
    
    if (embed_tensors.empty()) {
        std::cerr << "[EMBED] No embedding tensors found\n";
        return true;
    }
    
    std::cerr << "[EMBED] Found " << embed_tensors.size() << " embedding tensor(s)\n";
    
    // Per-embedding-type thresholds
    auto get_threshold = [&](const std::string& embed_type) -> float {
        if (embed_type == "token") return 0.45f;
        if (embed_type == "patch") return 0.25f;
        if (embed_type == "position") return 0.02f;
        if (embed_type == "projection") return 0.15f;
        return 0.30f;
    };
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 16) num_threads = 16;
    
    auto total_start = std::chrono::steady_clock::now();
    
    // =========================================================================
    // PHASE 1: Process all embedding tensors in parallel, accumulate edges
    // =========================================================================
    std::mutex batch_mutex;
    std::string global_batch;
    global_batch.reserve(16 << 20);  // 16MB initial
    std::atomic<size_t> total_edges{0};
    
    auto process_embedding = [&](const std::string& embed_type, TensorMeta* embed) {
        if (embed->shape.size() < 2) return;

        int64_t num_items = embed->shape[0];
        int64_t embed_dim = embed->shape[1];
        float threshold = get_threshold(embed_type);

        if (embed_type == "token" && !ctx.vocab_tokens.empty()) {
            num_items = std::min(num_items, static_cast<int64_t>(ctx.vocab_tokens.size()));
        }

        std::cerr << "[EMBED] Processing " << embed->name << " [" << embed_type << ", thresh=" << threshold << "]: "
                  << num_items << " x " << embed_dim << " dims\n";

        // Fetch quality score for this embedding tensor
        std::vector<std::string> tensor_names = {embed->name};
        auto quality_scores = fetch_quality_scores(conn, config.model_name, tensor_names);
        float embed_quality = quality_scores.count(embed->name) ? quality_scores[embed->name] : 1.0f;
        std::cerr << "[EMBED] Quality score for " << embed->name << ": " << embed_quality << "\n";
        
        auto start = std::chrono::steady_clock::now();
        
        // Read embeddings
        std::vector<std::vector<float>> embeddings(num_items);
        std::atomic<int64_t> read_idx{0};
        auto read_worker = [&]() {
            while (true) {
                int64_t i = read_idx.fetch_add(1);
                if (i >= num_items) break;
                embeddings[i] = read_tensor_row(*embed, static_cast<size_t>(i));
            }
        };
        std::vector<std::thread> readers;
        for (unsigned t = 0; t < num_threads; ++t) readers.emplace_back(read_worker);
        for (auto& th : readers) th.join();
        
        auto read_end = std::chrono::steady_clock::now();
        auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - start).count();
        std::cerr << "[EMBED] Read embeddings in " << read_ms << "ms\n";
        
        // Build k-NN using HNSWLIB
        const int k_neighbors = 15;
        std::vector<std::vector<std::tuple<size_t, size_t, float>>> thread_edges(num_threads);
        
#ifdef HAS_HNSWLIB
        hnswlib::InnerProductSpace space(embed_dim);
        hnswlib::HierarchicalNSW<float> hnsw(&space, num_items, 16, 200);
        hnsw.setEf(50);
        
        // Normalize for cosine similarity
        std::vector<std::vector<float>> normalized(num_items);
        for (int64_t i = 0; i < num_items; ++i) {
            if (embeddings[i].empty()) continue;
            normalized[i].resize(embed_dim);
            float norm = 0;
            for (size_t d = 0; d < static_cast<size_t>(embed_dim); ++d) 
                norm += embeddings[i][d] * embeddings[i][d];
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (size_t d = 0; d < static_cast<size_t>(embed_dim); ++d) 
                    normalized[i][d] = embeddings[i][d] / norm;
            }
        }
        
        // Parallel index construction
        std::atomic<int64_t> add_idx{0};
        auto add_worker = [&]() {
            while (true) {
                int64_t i = add_idx.fetch_add(1);
                if (i >= num_items) break;
                if (normalized[i].empty()) continue;
                hnsw.addPoint(normalized[i].data(), i);
            }
        };
        std::vector<std::thread> adders;
        for (unsigned t = 0; t < num_threads; ++t) adders.emplace_back(add_worker);
        for (auto& th : adders) th.join();
        
        // Parallel k-NN queries
        std::atomic<int64_t> knn_idx{0};
        auto knn_worker = [&](unsigned tid) {
            auto& local_edges = thread_edges[tid];
            while (true) {
                int64_t i = knn_idx.fetch_add(1);
                if (i >= num_items) break;
                if (normalized[i].empty()) continue;
                
                auto result = hnsw.searchKnn(normalized[i].data(), k_neighbors + 1);
                while (!result.empty()) {
                    auto [dist, j] = result.top();
                    result.pop();
                    if (static_cast<int64_t>(j) == i) continue;
                    float sim = 1.0f - dist;
                    // Apply quality weighting: multiply by quality score squared for pairwise relations from same tensor
                    sim *= embed_quality * embed_quality;
                    if (sim >= threshold && static_cast<size_t>(i) < j) {
                        local_edges.emplace_back(static_cast<size_t>(i), j, sim);
                    }
                }
            }
        };
        std::vector<std::thread> knn_workers;
        for (unsigned t = 0; t < num_threads; ++t) knn_workers.emplace_back(knn_worker, t);
        for (auto& th : knn_workers) th.join();
#endif
        
        auto knn_end = std::chrono::steady_clock::now();
        auto knn_ms = std::chrono::duration_cast<std::chrono::milliseconds>(knn_end - read_end).count();
        
        size_t edge_count = 0;
        for (const auto& edges : thread_edges) edge_count += edges.size();
        std::cerr << "[EMBED] Built k-NN graph: " << edge_count << " edges in " << knn_ms << "ms\n";
        
        if (edge_count == 0) {
            std::cerr << "[EMBED] No edges above threshold " << threshold << " for " << embed_type << "\n";
            return;
        }
        
        // Build batch string for this tensor
        std::string local_batch;
        local_batch.reserve(edge_count * 128);
        
        for (const auto& edges : thread_edges) {
            for (const auto& [i, j, sim] : edges) {
                Blake3Hash source_hash, target_hash;
                char source_type = 'C', target_type = 'C';
                
                if (embed_type == "token" && i < ctx.vocab_tokens.size() && j < ctx.vocab_tokens.size()) {
                    const auto& src = ctx.vocab_tokens[i];
                    const auto& tgt = ctx.vocab_tokens[j];
                    source_hash = src.comp.hash;
                    target_hash = tgt.comp.hash;
                    source_type = (src.comp.children.size() <= 1) ? 'A' : 'C';
                    target_type = (tgt.comp.children.size() <= 1) ? 'A' : 'C';
                } else {
                    std::string src_key = embed_type + ":" + std::to_string(i);
                    std::string tgt_key = embed_type + ":" + std::to_string(j);
                    source_hash = AtomCalculator::compute_vocab_token(src_key).hash;
                    target_hash = AtomCalculator::compute_vocab_token(tgt_key).hash;
                }
                
                local_batch += source_type;
                local_batch += "\t\\\\x" + source_hash.to_hex() + "\t";
                local_batch += target_type;
                local_batch += "\t\\\\x" + target_hash.to_hex() + "\t";
                local_batch += std::to_string(sim) + "\t";
                local_batch += embed_type + "\n";
            }
        }
        
        // Append to global batch under lock
        {
            std::lock_guard<std::mutex> lock(batch_mutex);
            global_batch += local_batch;
        }
        total_edges += edge_count;
    };
    
    // Process all embedding tensors (could parallelize but tensor I/O may bottleneck)
    for (auto& [embed_type, embed] : embed_tensors) {
        process_embedding(embed_type, embed);
    }
    
    auto process_end = std::chrono::steady_clock::now();
    auto process_ms = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - total_start).count();
    std::cerr << "[EMBED] Processed all tensors in " << process_ms << "ms, " << total_edges << " edges\n";
    
    if (total_edges == 0) {
        std::cerr << "[EMBED] No embedding relations to insert\n";
        return true;
    }
    
    // =========================================================================
    // PHASE 2: Single bulk insert of all accumulated edges into relation_evidence
    // =========================================================================
    auto insert_start = std::chrono::steady_clock::now();

    Transaction tx(conn);

    // Parse global_batch and build direct INSERT with VALUES for relation_evidence
    std::string insert_sql = R"SQL(
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight)
        VALUES
    )SQL";
    std::vector<std::string> values;
    std::istringstream iss(global_batch);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        std::istringstream line_ss(line);
        std::string source_type, source_id_hex, target_type, target_id_hex, weight_str, embed_type;
        if (!(line_ss >> source_type >> source_id_hex >> target_type >> target_id_hex >> weight_str >> embed_type)) continue;

        // Remove \\x prefix
        if (source_id_hex.substr(0, 2) == "\\\\x") source_id_hex = source_id_hex.substr(2);
        if (target_id_hex.substr(0, 2) == "\\\\x") target_id_hex = target_id_hex.substr(2);

        std::string val = "('" + source_type + "', '\\x" + source_id_hex + "', '" + target_type + "', '\\x" + target_id_hex +
                          "', 'E', " + weight_str + ", '" + config.model_name + "', 1, -1, '" + embed_type + "')";
        values.push_back(val);
    }

    // Batch in chunks to avoid query size limits
    const size_t batch_size = 1000;
    int total_inserted = 0;
    for (size_t i = 0; i < values.size(); i += batch_size) {
        std::string batch_sql = insert_sql;
        for (size_t j = i; j < std::min(i + batch_size, values.size()); ++j) {
            if (j > i) batch_sql += ", ";
            batch_sql += values[j];
        }
        batch_sql += " ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
                    "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
                    "  source_count = relation.source_count + 1";

        Result res = exec(conn, batch_sql);
        if (res.ok()) {
            total_inserted += cmd_tuples(res);
        } else {
            std::cerr << "[EMBED] Batch insert failed: " << res.error_message() << "\n";
        }
    }

    tx.commit();
    
    auto insert_end = std::chrono::steady_clock::now();
    auto insert_ms = std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - insert_start).count();

    std::cerr << "[EMBED] Bulk inserted " << total_inserted << " relations in " << insert_ms << "ms\n";
    std::cerr << "[EMBED] Total: " << total_edges.load() << " embedding similarity relations\n";
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
        AND pm.tensor_name IN (')SQL" + tensor_list + R"SQL(')
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
