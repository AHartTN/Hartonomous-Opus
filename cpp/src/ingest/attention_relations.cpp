/**
 * @file attention_relations.cpp
 * @brief Extract semantic relations from model embeddings
 * 
 * The model's embedding matrix encodes TOKEN↔TOKEN semantic relationships
 * learned from training. We extract these as relations between TOKEN COMPOSITIONS
 * which are REAL entities (atom trajectories) that already exist in the database.
 * 
 * NO fake dimension pseudo-entities. NO internal model mechanics.
 * Only TOKEN↔TOKEN semantic similarity from the model's learned knowledge.
 * 
 * This gives meaning to compositions by placing them in proximity to
 * semantically related concepts. The Voronoi cells emerge from relationship density.
 */

#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/helpers.hpp"
#include <thread>
#include <atomic>

namespace hypercube {
namespace ingest {
namespace db {

using namespace hypercube::db;

bool insert_attention_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    // =========================================================================
    // EXTRACT TOKEN↔TOKEN SEMANTIC SIMILARITY FROM EMBEDDING MATRIX
    // 
    // The embedding matrix is [vocab_size x embed_dim].
    // Each row is a token's learned semantic position.
    // Cosine similarity between rows = semantic relatedness.
    // 
    // This creates RELATIONS between TOKEN COMPOSITIONS that already exist.
    // Relations give compositions MEANING by establishing semantic proximity.
    // =========================================================================
    
    if (ctx.vocab_tokens.empty()) {
        std::cerr << "[SEMANTIC] No vocabulary tokens loaded, skipping semantic extraction\n";
        return true;
    }
    
    // Find the main token embedding tensor
    TensorMeta* embed = nullptr;
    std::string embed_name;
    
    for (auto& [name, meta] : ctx.tensors) {
        // Look for the token embedding layer
        if ((name.find("embed_tokens") != std::string::npos ||
             name.find("word_embeddings") != std::string::npos ||
             name.find("shared.weight") != std::string::npos ||  // T5/BART style
             name.find("wte.weight") != std::string::npos) &&    // GPT-2 style
            meta.shape.size() == 2) {
            embed = &meta;
            embed_name = name;
            break;
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
    
    // Read all token embeddings into memory
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(vocab_size);
    
    std::vector<size_t> valid_indices;  // Track which tokens have valid embeddings
    valid_indices.reserve(vocab_size);
    
    for (int64_t i = 0; i < vocab_size; ++i) {
        auto emb = read_tensor_row(*embed, static_cast<size_t>(i));
        if (!emb.empty() && !ctx.vocab_tokens[i].comp.hash.is_zero()) {
            embeddings.push_back(std::move(emb));
            valid_indices.push_back(static_cast<size_t>(i));
        }
    }
    
    if (embeddings.size() < 2) {
        std::cerr << "[SEMANTIC] Not enough valid embeddings\n";
        return true;
    }
    
    std::cerr << "[SEMANTIC] Loaded " << embeddings.size() << " token embeddings\n";
    
    // Build k-NN graph: for each token, find its k most similar tokens
    // This captures the model's learned semantic neighborhoods
    const size_t k_neighbors = 15;
    const float min_similarity = 0.3f;  // Only strong semantic connections
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 8) num_threads = 8;
    
    std::vector<std::vector<std::tuple<size_t, size_t, float>>> thread_edges(num_threads);
    std::atomic<size_t> work_idx{0};
    
    auto knn_worker = [&](unsigned tid) {
        auto& local_edges = thread_edges[tid];
        std::vector<std::pair<float, size_t>> neighbors;
        neighbors.reserve(embeddings.size());
        
        while (true) {
            size_t i = work_idx.fetch_add(1);
            if (i >= embeddings.size()) break;
            
            neighbors.clear();
            const auto& emb_i = embeddings[i];
            
            for (size_t j = 0; j < embeddings.size(); ++j) {
                if (i == j) continue;
                float sim = static_cast<float>(
                    embedding::cosine_similarity(emb_i.data(), embeddings[j].data(), embed_dim));
                if (sim >= min_similarity) {
                    neighbors.emplace_back(sim, j);
                }
            }
            
            if (neighbors.empty()) continue;
            
            // Sort by similarity (descending)
            std::partial_sort(neighbors.begin(),
                              neighbors.begin() + std::min(k_neighbors, neighbors.size()),
                              neighbors.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            
            // Add edges (only i < j to avoid duplicates)
            for (size_t k = 0; k < std::min(k_neighbors, neighbors.size()); ++k) {
                size_t j = neighbors[k].second;
                float sim = neighbors[k].first;
                if (i < j) {
                    local_edges.emplace_back(i, j, sim);
                }
            }
        }
    };
    
    std::cerr << "[SEMANTIC] Computing token similarity graph (" << num_threads << " threads)...\n";
    
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back(knn_worker, t);
    }
    for (auto& th : workers) th.join();
    
    // Count total edges
    size_t total_edges = 0;
    for (const auto& edges : thread_edges) total_edges += edges.size();
    
    std::cerr << "[SEMANTIC] Found " << total_edges << " semantic similarity edges\n";
    
    if (total_edges == 0) return true;
    
    // Insert relations between TOKEN COMPOSITIONS
    // These are REAL entities - the token strings decompose into atom trajectories
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
            
            // Determine source/target types
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
    
    // Insert with relation_type='S' for Semantic similarity
    // This is the model's learned knowledge about concept relatedness
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
