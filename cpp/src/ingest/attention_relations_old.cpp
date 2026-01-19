/**
 * @file attention_relations.cpp
 * @brief Extract attention and weight similarity relations
 * 
 * Extracts sparse relations from model weight tensors including:
 *   - Router weights (MoE models) - token-to-expert routing
 *   - Attention projections - row similarity in Q/K/V/O matrices
 *   - Token-to-dimension mappings - which dimensions tokens activate
 */

#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/helpers.hpp"
#include <unordered_set>

namespace hypercube {
namespace ingest {
namespace db {

using namespace hypercube::db;

bool insert_attention_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
    // ==========================================================================
    // EXTRACT SPARSE RELATIONS FROM MODEL TENSORS
    // - Router weights: token -> expert routing (MoE models)
    // - Token relationships emerge from Laplacian projection, not raw weights
    // ==========================================================================
    
    size_t total_edges = 0;
    
    // -------------------------------------------------------------------------
    // PART 1: Router weights for MoE models - sparse token-to-expert routing
    // Only extract router.weight tensors which define which experts handle which tokens
    // -------------------------------------------------------------------------
    std::vector<TensorMeta*> router_tensors;
    for (auto& [name, meta] : ctx.tensors) {
        if (name.find("router.weight") != std::string::npos && meta.shape.size() >= 2) {
            router_tensors.push_back(&meta);
        }
    }
    
    if (!router_tensors.empty()) {
        std::cerr << "[ROUTER] Found " << router_tensors.size() << " router tensors\n";
        
        for (auto* router : router_tensors) {
            // router.weight is [num_experts, hidden_dim]
            // Each row is an expert's routing vector
            int64_t num_experts = router->shape[0];
            int64_t hidden_dim = router->shape[1];
            
            // Parse layer number from tensor name
            int layer = -1;
            size_t layers_pos = router->name.find("layers.");
            if (layers_pos != std::string::npos) {
                size_t num_start = layers_pos + 7;
                size_t num_end = router->name.find(".", num_start);
                if (num_end != std::string::npos) {
                    layer = std::stoi(router->name.substr(num_start, num_end - num_start));
                }
            }
            
            std::cerr << "[ROUTER] " << router->name << " [" << num_experts << " experts x " << hidden_dim << " dims] layer=" << layer << "\n";
            
            // For each expert, find which tokens route to it above threshold
            // This creates expert atoms and token->expert edges
            Transaction tx(conn);
            
            Result res = exec(conn,
                "CREATE TEMP TABLE tmp_router ("
                "  source_type CHAR(1), source_id BYTEA,"
                "  target_type CHAR(1), target_id BYTEA,"
                "  weight REAL, layer SMALLINT, component TEXT"
                ") ON COMMIT DROP");
            
            CopyStream copy(conn, "COPY tmp_router FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
            
            std::string batch;
            batch.reserve(1 << 20);
            size_t router_edges = 0;
            
            for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                auto expert_row = read_tensor_row(*router, static_cast<size_t>(expert_idx));
                if (expert_row.empty()) continue;
                
                // Create expert atom hash
                std::string expert_key = "expert:" + std::to_string(layer) + ":" + std::to_string(expert_idx);
                auto expert_hash = AtomCalculator::compute_vocab_token(expert_key).hash;
                
                // Find significant routing weights (use lower threshold 0.1 for router)
                for (int64_t d = 0; d < hidden_dim && d < static_cast<int64_t>(ctx.vocab_tokens.size()); ++d) {
                    float weight = expert_row[d];
                    if (std::fabs(weight) >= 0.1f) {
                        // Create edge from token to expert
                        const auto& token = ctx.vocab_tokens[d];
                        char token_type = (token.comp.children.size() <= 1) ? 'A' : 'C';
                        
                        batch += token_type;
                        batch += "\t";
                        copy_bytea(batch, token.comp.hash);
                        batch += "\tE\t";
                        copy_bytea(batch, expert_hash);
                        batch += "\t";
                        batch += std::to_string(weight) + "\t";
                        batch += std::to_string(layer) + "\trouter\n";
                        router_edges++;
                        
                        if (batch.size() > (1 << 19)) {
                            copy.put(batch);
                            batch.clear();
                        }
                    }
                }
            }
            
            if (!batch.empty()) copy.put(batch);
            copy.end();
            
            // Insert
            std::string insert_sql = 
                "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
                "SELECT source_type, source_id, target_type, target_id, 'R', weight, '" + config.model_name + "', 1, layer, component FROM tmp_router "
                "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
                "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
                "  source_count = relation.source_count + 1";
            exec(conn, insert_sql);
            
            tx.commit();
            
            std::cerr << "  -> " << router_edges << " routing edges\n";
            total_edges += router_edges;
        }
    } else {
        std::cerr << "[ROUTER] No router tensors found (not an MoE model)\n";
    }
    
    // -------------------------------------------------------------------------
    // PART 2: Attention projections (Q/K/V/O) - token transformation relationships
    // These weights show how tokens relate through the attention mechanism:
    // - Q (query): what a token is looking for
    // - K (key): what a token offers to be found
    // - V (value): what information a token carries
    // - O (output): how attention results are projected back
    // 
    // We extract ROW SIMILARITY: rows that are similar = tokens that transform similarly
    // -------------------------------------------------------------------------
    std::cerr << "\n[ATTN] Extracting attention projection similarities...\n";
    
    struct TensorGroup {
        std::string component;
        std::vector<TensorMeta*> tensors;
    };
    std::vector<TensorGroup> attn_groups = {
        // Attention projections
        {"q_proj", {}}, {"k_proj", {}}, {"v_proj", {}}, {"o_proj", {}},
        {"query", {}}, {"key", {}}, {"value", {}},  // Alternative naming
        {"qkv_proj", {}},  // Fused QKV
        
        // FFN/MLP layers (LLaMA style)
        {"gate_proj", {}}, {"up_proj", {}}, {"down_proj", {}},
        
        // FFN/MLP layers (GPT style)
        {"fc1", {}}, {"fc2", {}}, {"fc_in", {}}, {"fc_out", {}},
        {"c_fc", {}}, {"c_proj", {}},  // GPT-2
        
        // Output projections
        {"lm_head", {}}, {"output", {}}, {"classifier", {}},
        
        // Vision transformers
        {"patch_embed", {}}, {"cls_token", {}},
    };
    
    for (auto& [name, meta] : ctx.tensors) {
        for (auto& group : attn_groups) {
            if (name.find(group.component + ".weight") != std::string::npos && meta.shape.size() == 2) {
                group.tensors.push_back(&meta);
            }
        }
    }
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 8;
    
    for (auto& group : attn_groups) {
        if (group.tensors.empty()) continue;
        
        std::cerr << "[ATTN] Processing " << group.tensors.size() << " " << group.component << " tensors\n";
        
        for (auto* tensor : group.tensors) {
            // Parse layer number
            int layer = -1;
            size_t layers_pos = tensor->name.find("layers.");
            if (layers_pos != std::string::npos) {
                size_t num_start = layers_pos + 7;
                size_t num_end = tensor->name.find(".", num_start);
                if (num_end != std::string::npos) {
                    layer = std::stoi(tensor->name.substr(num_start, num_end - num_start));
                }
            }
            
            int64_t out_dim = tensor->shape[0];
            int64_t in_dim = tensor->shape[1];
            
            // For large tensors, sample rows to keep computation tractable
            // We're looking for which OUTPUT dimensions are related
            int64_t max_rows = std::min(out_dim, static_cast<int64_t>(2048));
            int64_t stride = std::max(static_cast<int64_t>(1), out_dim / max_rows);
            
            std::cerr << "  " << tensor->name << " [" << out_dim << " x " << in_dim << "] layer=" << layer;
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
            
            // Build k-NN similarity for these weight rows
            // Similar rows = dimensions that behave similarly = related features
            const int k_neighbors = 10;
            std::vector<std::vector<std::tuple<size_t, size_t, float>>> thread_edges(num_threads);
            std::atomic<size_t> knn_idx{0};
            
            auto knn_worker = [&](unsigned tid) {
                auto& local_edges = thread_edges[tid];
                std::vector<std::pair<float, size_t>> neighbors;
                neighbors.reserve(rows.size());
                
                while (true) {
                    size_t i = knn_idx.fetch_add(1);
                    if (i >= rows.size()) break;
                    
                    neighbors.clear();
                    for (size_t j = 0; j < rows.size(); ++j) {
                        if (i == j) continue;
                        float sim = static_cast<float>(embedding::cosine_similarity(rows[i].data(), rows[j].data(), in_dim));
                        neighbors.emplace_back(sim, j);
                    }
                    
                    std::partial_sort(neighbors.begin(),
                                      neighbors.begin() + std::min(static_cast<size_t>(k_neighbors), neighbors.size()),
                                      neighbors.end(),
                                      [](auto& a, auto& b) { return a.first > b.first; });
                    
                    for (size_t k = 0; k < std::min(static_cast<size_t>(k_neighbors), neighbors.size()); ++k) {
                        float sim = neighbors[k].first;
                        size_t j = neighbors[k].second;
                        // Use lower threshold for weight matrices (0.15 vs 0.5 for embeddings)
                        if (sim >= 0.15f && i < j) {
                            local_edges.emplace_back(i, j, sim);
                        }
                    }
                }
            };
            
            std::vector<std::thread> workers;
            for (unsigned t = 0; t < num_threads; ++t) {
                workers.emplace_back(knn_worker, t);
            }
            for (auto& th : workers) th.join();
            
            // Count edges
            size_t edge_count = 0;
            for (const auto& edges : thread_edges) edge_count += edges.size();
            
            if (edge_count == 0) continue;
            
            // Insert edges as relations
            // These represent "dimension i and dimension j behave similarly in this layer"
            // This helps identify the "beaten path" - frequently co-activated features
            Transaction tx(conn);
            
            Result res = exec(conn,
                "CREATE TEMP TABLE tmp_attn ("
                "  source_type CHAR(1), source_id BYTEA,"
                "  target_type CHAR(1), target_id BYTEA,"
                "  weight REAL, layer SMALLINT, component TEXT"
                ") ON COMMIT DROP");
            
            CopyStream copy(conn, "COPY tmp_attn FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
            
            std::string batch;
            batch.reserve(1 << 20);
            
            // =========================================================
            // PHASE 1: Create dimension compositions FIRST
            // Each dimension like "q_proj:0:dim42" must exist as a 
            // real composition before relations can reference it
            // =========================================================
            std::unordered_set<std::string> dim_keys_created;
            std::string comp_batch;
            comp_batch.reserve(1 << 18);
            
            for (const auto& edges : thread_edges) {
                for (const auto& [i, j, sim] : edges) {
                    std::string src_key = group.component + ":" + std::to_string(layer) + ":dim" + std::to_string(row_indices[i]);
                    std::string tgt_key = group.component + ":" + std::to_string(layer) + ":dim" + std::to_string(row_indices[j]);
                    dim_keys_created.insert(src_key);
                    dim_keys_created.insert(tgt_key);
                }
            }
            
            // Create compositions for all referenced dimensions
            if (!dim_keys_created.empty()) {
                Result tmp_comp = exec(conn,
                    "CREATE TEMP TABLE IF NOT EXISTS tmp_dim_comp ("
                    "  id BYTEA PRIMARY KEY, label TEXT, depth INT, child_count INT, atom_count INT,"
                    "  centroid GEOMETRY(POINTZM, 0), hilbert_lo BIGINT, hilbert_hi BIGINT"
                    ") ON COMMIT DROP");
                
                CopyStream comp_copy(conn, "COPY tmp_dim_comp FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
                
                for (const std::string& key : dim_keys_created) {
                    CompositionRecord comp = AtomCalculator::compute_vocab_token(key);
                    
                    comp_batch += "\\\\x";
                    comp_batch += comp.hash.to_hex();
                    comp_batch += "\t";
                    for (char ch : key) {
                        if (ch == '\t') comp_batch += "\\t";
                        else if (ch == '\n') comp_batch += "\\n";
                        else if (ch == '\\') comp_batch += "\\\\";
                        else comp_batch += ch;
                    }
                    comp_batch += "\t1\t";
                    comp_batch += std::to_string(comp.children.size());
                    comp_batch += "\t";
                    comp_batch += std::to_string(comp.atom_count);
                    comp_batch += "\t";
                    comp_batch += build_composition_pointzm_ewkb(comp.centroid);
                    comp_batch += "\t";
                    comp_batch += std::to_string(static_cast<int64_t>(comp.hilbert.lo));
                    comp_batch += "\t";
                    comp_batch += std::to_string(static_cast<int64_t>(comp.hilbert.hi));
                    comp_batch += "\n";
                }
                
                comp_copy.put(comp_batch);
                comp_copy.end();
                
                // Insert dimension compositions into main table
                exec(conn,
                    "INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi) "
                    "SELECT id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi FROM tmp_dim_comp "
                    "ON CONFLICT (id) DO NOTHING");
            }
            
            // =========================================================
            // PHASE 2: Now create relations between existing compositions
            // =========================================================
            for (const auto& edges : thread_edges) {
                for (const auto& [i, j, sim] : edges) {
                    std::string src_key = group.component + ":" + std::to_string(layer) + ":dim" + std::to_string(row_indices[i]);
                    std::string tgt_key = group.component + ":" + std::to_string(layer) + ":dim" + std::to_string(row_indices[j]);
                    
                    auto src_hash = AtomCalculator::compute_vocab_token(src_key).hash;
                    auto tgt_hash = AtomCalculator::compute_vocab_token(tgt_key).hash;
                    
                    batch += "C\t";
                    copy_bytea(batch, src_hash);
                    batch += "\tC\t";
                    copy_bytea(batch, tgt_hash);
                    batch += "\t";
                    batch += std::to_string(sim) + "\t";
                    batch += std::to_string(layer) + "\t" + group.component + "\n";
                    
                    if (batch.size() > (1 << 19)) {
                        copy.put(batch);
                        batch.clear();
                    }
                }
            }
            
            if (!batch.empty()) copy.put(batch);
            copy.end();
            
            // Insert with relation_type='W' for weight similarity
            std::string insert_sql = 
                "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
                "SELECT source_type, source_id, target_type, target_id, 'W', weight, '" + config.model_name + "', 1, layer, component FROM tmp_attn "
                "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
                "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
                "  source_count = relation.source_count + 1";
            exec(conn, insert_sql);
            
            tx.commit();
            
            std::cerr << "    -> " << edge_count << " weight similarity edges\n";
            total_edges += edge_count;
        }
    }
    
    // -------------------------------------------------------------------------
    // PART 3: Token-to-dimension mapping via embedding * projection
    // This shows which tokens activate which dimensions most strongly
    // -------------------------------------------------------------------------
    std::cerr << "\n[TOKEN-DIM] Extracting token->dimension activation patterns...\n";
    
    // Find embedding tensor
    TensorMeta* embed = nullptr;
    for (auto& [name, meta] : ctx.tensors) {
        if ((name.find("embed_tokens") != std::string::npos ||
             name.find("word_embeddings") != std::string::npos) && meta.shape.size() == 2) {
            embed = &meta;
            break;
        }
    }
    
    if (embed && !ctx.vocab_tokens.empty()) {
        int64_t vocab_size = std::min(embed->shape[0], static_cast<int64_t>(ctx.vocab_tokens.size()));
        int64_t embed_dim = embed->shape[1];
        
        // For each token, find which dimensions it activates most strongly
        // This creates token->dimension edges
        std::cerr << "[TOKEN-DIM] Processing " << vocab_size << " tokens x " << embed_dim << " dims\n";
        
        Transaction tx(conn);
        
        Result res = exec(conn,
            "CREATE TEMP TABLE tmp_tokdim ("
            "  source_type CHAR(1), source_id BYTEA,"
            "  target_type CHAR(1), target_id BYTEA,"
            "  weight REAL, component TEXT"
            ") ON COMMIT DROP");
        
        CopyStream copy(conn, "COPY tmp_tokdim FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        
        std::string batch;
        batch.reserve(1 << 21);
        size_t tokdim_edges = 0;
        
        // Top-k dimensions per token
        const int top_k = 20;
        
        for (int64_t tok_idx = 0; tok_idx < vocab_size; ++tok_idx) {
            auto emb = read_tensor_row(*embed, static_cast<size_t>(tok_idx));
            if (emb.empty()) continue;
            
            const auto& token = ctx.vocab_tokens[tok_idx];
            if (token.comp.hash.is_zero()) continue;
            
            char token_type = (token.comp.children.size() <= 1) ? 'A' : 'C';
            
            // Find top-k dimensions by absolute value
            std::vector<std::pair<float, int64_t>> dim_vals;
            dim_vals.reserve(embed_dim);
            for (int64_t d = 0; d < embed_dim; ++d) {
                dim_vals.emplace_back(std::fabs(emb[d]), d);
            }
            std::partial_sort(dim_vals.begin(), dim_vals.begin() + std::min(static_cast<int64_t>(top_k), embed_dim),
                              dim_vals.end(), [](auto& a, auto& b) { return a.first > b.first; });
            
            for (int k = 0; k < std::min(static_cast<int64_t>(top_k), embed_dim); ++k) {
                float val = emb[dim_vals[k].second];  // Keep sign
                // Lower threshold for token->dimension activations (0.3)
                if (std::fabs(val) < 0.3f) continue;
                
                int64_t dim_idx = dim_vals[k].second;
                std::string dim_key = "embed:dim" + std::to_string(dim_idx);
                auto dim_hash = AtomCalculator::compute_vocab_token(dim_key).hash;
                
                batch += token_type;
                batch += "\t";
                copy_bytea(batch, token.comp.hash);
                batch += "\tC\t";
                copy_bytea(batch, dim_hash);
                batch += "\t";
                batch += std::to_string(val) + "\tembed\n";
                tokdim_edges++;
                
                if (batch.size() > (1 << 20)) {
                    copy.put(batch);
                    batch.clear();
                }
            }
        }
        
        if (!batch.empty()) copy.put(batch);
        copy.end();
        
        // Insert with relation_type='D' for dimension activation
        std::string insert_sql = 
            "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
            "SELECT source_type, source_id, target_type, target_id, 'D', weight, '" + config.model_name + "', 1, -1, component FROM tmp_tokdim "
            "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
            "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
            "  source_count = relation.source_count + 1";
        exec(conn, insert_sql);
        
        tx.commit();
        
        std::cerr << "[TOKEN-DIM] Created " << tokdim_edges << " token->dimension edges\n";
        total_edges += tokdim_edges;
    }
    
    std::cerr << "\n[EXTRACT] Total: " << total_edges << " relation edges from model weights\n";
    return true;
}

} // namespace db
} // namespace ingest
} // namespace hypercube
