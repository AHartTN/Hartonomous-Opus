/**
 * @file multimodal_extraction.cpp
 * @brief Multimodal relation extraction for DETR, Florence, MoE, vision models
 *
 * Extracts relationships (Relation edges) from multimodal structures:
 *   - Attention patterns from object queries to tokens
 *   - Cross-attention between modalities
 *   - Router relations in MoE models
 *   - Class head relations
 *
 * Populates the Relation table with ELO-averaged weights instead of creating composition nodes.
 */

#include "hypercube/ingest/multimodal_extraction.hpp"
#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/db/operations.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <span>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#define getpid() GetCurrentProcessId()
#else
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAS_HNSWLIB
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include <hnswlib/hnswlib.h>
#pragma GCC diagnostic pop
#endif

namespace hypercube {
namespace ingest {

// ============================================================================
// Configuration
// ============================================================================

static constexpr float MIN_NORM = 0.01f;         // Skip near-zero vectors
// static constexpr size_t BATCH_SIZE = 1000;       // DB batch insert size
static constexpr size_t K_NEIGHBORS = 15;        // Number of similar tokens to link
static constexpr float MIN_SIMILARITY = 0.5f;    // Minimum cosine similarity for relations

// ============================================================================
// Edge structure for relations
// ============================================================================

struct RelationEdge {
    Blake3Hash source_id;
    char source_type;
    Blake3Hash target_id;
    char target_type;
    float weight;
    int layer;
    std::string component;
};

// ============================================================================
// Load tensor data (same pattern as semantic_extraction.cpp)
// ============================================================================

static std::vector<float> load_tensor(const TensorMeta& meta) {
    size_t n = meta.element_count();
    std::vector<float> data(n);

    std::ifstream f(meta.shard_file, std::ios::binary);
    if (!f) return {};

    f.seekg(static_cast<std::streamoff>(meta.data_offset_start));

    if (meta.dtype == "F32") {
        f.read(reinterpret_cast<char*>(data.data()),
               static_cast<std::streamsize>(n * 4));
    } else if (meta.dtype == "BF16") {
        std::vector<uint16_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()),
               static_cast<std::streamsize>(n * 2));
        // PARALLEL BF16 conversion (use int64_t for MSVC OpenMP)
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint32_t bits = static_cast<uint32_t>(raw[static_cast<size_t>(i)]) << 16;
            std::memcpy(&data[static_cast<size_t>(i)], &bits, 4);
        }
    } else if (meta.dtype == "F16") {
        std::vector<uint16_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()),
               static_cast<std::streamsize>(n * 2));
        // PARALLEL F16 conversion
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint16_t h = raw[static_cast<size_t>(i)];
            uint32_t s = (h & 0x8000) << 16;
            uint32_t e = (h >> 10) & 0x1F;
            uint32_t m = h & 0x3FF;
            uint32_t fval = (e == 0) ? s :
                           (e == 31) ? (s | 0x7F800000 | (m << 13)) :
                           (s | ((e + 112) << 23) | (m << 13));
            std::memcpy(&data[static_cast<size_t>(i)], &fval, 4);
        }
    } else if (meta.dtype == "F8_E4M3") {
        std::vector<uint8_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()),
               static_cast<std::streamsize>(n * 1));
        // PARALLEL F8_E4M3FN conversion (E4M3 Finite, no infinities)
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            uint8_t f8 = raw[static_cast<size_t>(i)];
            uint32_t sign = (f8 >> 7) & 0x1;
            uint32_t exp = (f8 >> 3) & 0xF;
            uint32_t mant = f8 & 0x7;
            uint32_t f;
            if (exp == 0) {
                if (mant == 0) {
                    // Zero
                    f = sign << 31;
                } else {
                    // Denormalized number - shift mantissa until leading 1
                    int shift = 0;
                    uint32_t m = mant;
                    while ((m & 0x4) == 0 && shift < 3) {
                        m <<= 1;
                        shift++;
                    }
                    m &= 0x3;  // Remove leading 1
                    // FP32 exponent for subnormal: (127 - 7 - shift)
                    f = (sign << 31) | ((121 - shift) << 23) | (m << 20);
                }
            } else if (exp == 15 && mant == 7) {
                // NaN (the only NaN encoding in E4M3FN)
                f = (sign << 31) | 0x7FC00000;  // Quiet NaN
            } else {
                // Normalized number (including exp==15 with mant!=7, which are valid finite values)
                // FP32 exponent = exp - 7 (E4M3 bias) + 127 (FP32 bias) = exp + 120
                f = (sign << 31) | ((exp + 120) << 23) | (mant << 20);
            }
            std::memcpy(&data[static_cast<size_t>(i)], &f, 4);
        }
    } else if (meta.dtype == "I64") {
        std::vector<int64_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()),
               static_cast<std::streamsize>(n * 8));
        // PARALLEL I64 to float conversion
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            data[static_cast<size_t>(i)] = static_cast<float>(raw[static_cast<size_t>(i)]);
        }
    }
    return data;
}

// ============================================================================
// Escape string for SQL (using libpq)
// ============================================================================

// static std::string escape_sql_string(PGconn* conn, const std::string& str) {
//     char* escaped = PQescapeLiteral(conn, str.c_str(), str.length());
//     if (!escaped) {
//         // Fallback: basic escape
//         std::string result = "'";
//         for (char c : str) {
//             if (c == '\'') result += "''";
//             else if (c == '\\') result += "\\\\";
//             else result += c;
//         }
//         result += "'";
//         return result;
//     }
//     std::string result(escaped);
//     PQfreemem(escaped);
//     return result;
// }

// ============================================================================
// Normalize rows, return valid indices
// ============================================================================

static std::vector<size_t> normalize_and_get_valid(
    float* data, size_t rows, size_t cols
) {
    std::vector<float> norms(rows);

    // Compute norms in parallel
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++) {
        const float* row = data + static_cast<size_t>(i) * cols;
        float norm2 = 0;
        for (size_t j = 0; j < cols; j++) {
            norm2 += row[j] * row[j];
        }
        norms[static_cast<size_t>(i)] = std::sqrt(norm2);
    }

    // Normalize valid rows in parallel
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++) {
        size_t idx = static_cast<size_t>(i);
        if (norms[idx] >= MIN_NORM) {
            float inv = 1.0f / norms[idx];
            float* row = data + idx * cols;
            for (size_t j = 0; j < cols; j++) {
                row[j] *= inv;
            }
        }
    }

    // Collect valid indices
    std::vector<size_t> valid;
    valid.reserve(rows);
    for (size_t i = 0; i < rows; i++) {
        if (norms[i] >= MIN_NORM) valid.push_back(i);
    }
    return valid;
}

// ============================================================================
// Extract Object Queries (DETR, Florence)
// ============================================================================

size_t extract_object_queries(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans,
    void* token_hnsw,
    const std::vector<size_t>& valid_token_indices,
    std::vector<RelationEdge>& edges
) {
    size_t total_edges = 0;

    for (const auto& plan : plans) {
        if (plan.category != TensorCategory::OBJECT_QUERY) continue;

        // Find tensor metadata
        auto it = ctx.tensors.find(plan.name);
        if (it == ctx.tensors.end()) {
            std::cerr << "[MULTIMODAL] Tensor not found: " << plan.name << "\n";
            continue;
        }

        const TensorMeta& meta = it->second;
        if (meta.shape.size() != 2) {
            std::cerr << "[MULTIMODAL] Unexpected shape for " << plan.name << "\n";
            continue;
        }

        size_t num_queries = static_cast<size_t>(meta.shape[0]);
        size_t dim = static_cast<size_t>(meta.shape[1]);

        std::cerr << "[MULTIMODAL] Extracting " << num_queries
                  << " object query relations from " << plan.name << " [" << dim << "d]\n";

        // Load tensor data
        std::vector<float> data = load_tensor(meta);
        if (data.empty()) {
            std::cerr << "[MULTIMODAL] Failed to load tensor: " << plan.name << "\n";
            continue;
        }

        // Normalize
        auto valid = normalize_and_get_valid(data.data(), num_queries, dim);

        for (size_t idx : valid) {
            // Compute hash for the embedding vector
            const float* row = data.data() + idx * dim;
            Blake3Hasher::Incremental hasher;
            hasher.update(std::string_view(reinterpret_cast<const char*>(row), dim * sizeof(float)));
            // NOTE: hash computed but not used (HNSWLib disabled)

#ifdef HAS_HNSWLIB
            // Find similar tokens
            auto result = static_cast<hnswlib::HierarchicalNSW<float>*>(token_hnsw)->searchKnn(row, K_NEIGHBORS + 1);
            while (!result.empty()) {
                auto [dist, j] = result.top();
                result.pop();
                
                float sim = dist;  // Inner product
                if (sim >= MIN_SIMILARITY) {
                    size_t tok_idx = valid_token_indices[j];
                    const auto& comp = ctx.vocab_tokens[tok_idx].comp;
                    char target_type = (comp.children.size() <= 1) ? 'A' : 'C';
                    edges.push_back({hash, 'C', comp.hash, target_type, sim, plan.layer_idx, "OBJECT_QUERY"});
                    total_edges++;
                }
            }
#endif
        }

        std::cerr << "[MULTIMODAL] Extracted " << total_edges << " object query relations\n";
    }

    return total_edges;
}

// ============================================================================
// Extract 2D Positional Encodings
// ============================================================================

size_t extract_positional_encodings(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans,
    void* token_hnsw,
    const std::vector<size_t>& valid_token_indices,
    std::vector<RelationEdge>& edges
) {
    size_t total_edges = 0;

    for (const auto& plan : plans) {
        if (plan.category != TensorCategory::POSITION_EMBEDDING_2D &&
            plan.category != TensorCategory::POSITION_EMBEDDING) continue;

        auto it = ctx.tensors.find(plan.name);
        if (it == ctx.tensors.end()) continue;

        const TensorMeta& meta = it->second;
        if (meta.shape.size() != 2) continue;

        size_t num_positions = static_cast<size_t>(meta.shape[0]);
        size_t dim = static_cast<size_t>(meta.shape[1]);

        bool is_2d = plan.category == TensorCategory::POSITION_EMBEDDING_2D;
        std::string type_str = is_2d ? "position_2d" : "position_1d";

        // Determine axis from name (row vs column)
        std::string axis = "unknown";
        std::string lower_name = plan.name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        if (lower_name.find("row") != std::string::npos) axis = "row";
        else if (lower_name.find("col") != std::string::npos) axis = "column";
        else if (lower_name.find("x_") != std::string::npos) axis = "x";
        else if (lower_name.find("y_") != std::string::npos) axis = "y";

        std::cerr << "[MULTIMODAL] Extracting " << num_positions
                  << " " << type_str << " (" << axis << ") relations from " << plan.name << "\n";

        std::vector<float> data = load_tensor(meta);
        if (data.empty()) continue;

        auto valid = normalize_and_get_valid(data.data(), num_positions, dim);

        for (size_t idx : valid) {
            const float* row = data.data() + idx * dim;
            Blake3Hasher::Incremental hasher;
            hasher.update(std::string_view(reinterpret_cast<const char*>(row), dim * sizeof(float)));
            // NOTE: hash computed but not used (HNSWLib disabled)

#ifdef HAS_HNSWLIB
            auto result = static_cast<hnswlib::HierarchicalNSW<float>*>(token_hnsw)->searchKnn(row, K_NEIGHBORS + 1);
            while (!result.empty()) {
                auto [dist, j] = result.top();
                result.pop();
                
                float sim = dist;
                if (sim >= MIN_SIMILARITY) {
                    size_t tok_idx = valid_token_indices[j];
                    const auto& comp = ctx.vocab_tokens[tok_idx].comp;
                    char target_type = (comp.children.size() <= 1) ? 'A' : 'C';
                    edges.push_back({hash, 'C', comp.hash, target_type, sim, plan.layer_idx, "POSITIONAL"});
                    total_edges++;
                }
            }
#endif
        }
    }

    return total_edges;
}

// ============================================================================
// Extract MoE Routers (THE ROUTING MACHINE)
// ============================================================================

size_t extract_moe_routers(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans,
    void* token_hnsw,
    const std::vector<size_t>& valid_token_indices,
    std::vector<RelationEdge>& edges
) {
    size_t total_edges = 0;

    for (const auto& plan : plans) {
        if (plan.category != TensorCategory::MOE_ROUTER) continue;

        auto it = ctx.tensors.find(plan.name);
        if (it == ctx.tensors.end()) continue;

        const TensorMeta& meta = it->second;
        if (meta.shape.size() != 2) continue;

        // Router shape is [num_experts, d_model]
        size_t num_experts = static_cast<size_t>(meta.shape[0]);
        size_t d_model = static_cast<size_t>(meta.shape[1]);

        std::cerr << "[MULTIMODAL] Extracting MoE router: " << num_experts
                  << " experts x " << d_model << " from " << plan.name << "\n";

        std::vector<float> data = load_tensor(meta);
        if (data.empty()) continue;

        auto valid = normalize_and_get_valid(data.data(), num_experts, d_model);

        for (size_t idx : valid) {
            const float* row = data.data() + idx * d_model;
            Blake3Hasher::Incremental hasher;
            hasher.update(std::string_view(reinterpret_cast<const char*>(row), d_model * sizeof(float)));
            // NOTE: hash computed but not used (HNSWLib disabled)

#ifdef HAS_HNSWLIB
            auto result = static_cast<hnswlib::HierarchicalNSW<float>*>(token_hnsw)->searchKnn(row, K_NEIGHBORS + 1);
            while (!result.empty()) {
                auto [dist, j] = result.top();
                result.pop();
                
                float sim = dist;
                if (sim >= MIN_SIMILARITY) {
                    size_t tok_idx = valid_token_indices[j];
                    const auto& comp = ctx.vocab_tokens[tok_idx].comp;
                    char target_type = (comp.children.size() <= 1) ? 'A' : 'C';
                    edges.push_back({hash, 'C', comp.hash, target_type, sim, plan.layer_idx, "MOE_ROUTER"});
                    total_edges++;
                }
            }
#endif
        }

        std::cerr << "[MULTIMODAL] Extracted " << total_edges << " router relations\n";
    }

    return total_edges;
}

// ============================================================================
// Extract Universal Digital Content Modalities
// ============================================================================

size_t extract_universal_modality(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans,
    void* token_hnsw,
    const std::vector<size_t>& valid_token_indices,
    std::vector<RelationEdge>& edges,
    const std::string& modality_name
) {
    size_t total_edges = 0;

    for (const auto& plan : plans) {
        auto it = ctx.tensors.find(plan.name);
        if (it == ctx.tensors.end()) continue;

        const TensorMeta& meta = it->second;
        if (meta.shape.size() != 2) continue;

        size_t num_items = static_cast<size_t>(meta.shape[0]);
        size_t dim = static_cast<size_t>(meta.shape[1]);

        std::cerr << "[MULTIMODAL] Extracting " << num_items
                  << " " << modality_name << " relations from " << plan.name << " [" << dim << "d]\n";

        std::vector<float> data = load_tensor(meta);
        if (data.empty()) continue;

        auto valid = normalize_and_get_valid(data.data(), num_items, dim);

        for (size_t idx : valid) {
            const float* row = data.data() + idx * dim;
            Blake3Hasher::Incremental hasher;
            hasher.update(std::string_view(reinterpret_cast<const char*>(row), dim * sizeof(float)));
            Blake3Hash hash = hasher.finalize();
            (void)hash; // Suppress unused variable warning

#ifdef HAS_HNSWLIB
            // Find similar tokens using HNSW
            auto result = static_cast<hnswlib::HierarchicalNSW<float>*>(token_hnsw)->searchKnn(row, K_NEIGHBORS + 1);
            while (!result.empty()) {
                auto [dist, j] = result.top();
                result.pop();

                float sim = dist;  // Inner product similarity
                if (sim >= MIN_SIMILARITY) {
                    size_t tok_idx = valid_token_indices[j];
                    const auto& comp = ctx.vocab_tokens[tok_idx].comp;
                    char target_type = (comp.children.size() <= 1) ? 'A' : 'C';
                    edges.push_back({hash, 'C', comp.hash, target_type, sim, plan.layer_idx, modality_name});
                    total_edges++;
                }
            }
#endif
        }

        std::cerr << "[MULTIMODAL] Extracted " << total_edges << " " << modality_name << " relations\n";
    }

    return total_edges;
}

// ============================================================================
// Extract Class Heads (Detection)
// ============================================================================

size_t extract_class_heads(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans,
    void* token_hnsw,
    const std::vector<size_t>& valid_token_indices,
    std::vector<RelationEdge>& edges
) {
    size_t total_edges = 0;

    for (const auto& plan : plans) {
        if (plan.category != TensorCategory::CLASS_HEAD) continue;

        auto it = ctx.tensors.find(plan.name);
        if (it == ctx.tensors.end()) continue;

        const TensorMeta& meta = it->second;
        if (meta.shape.size() != 2) continue;

        size_t num_classes = static_cast<size_t>(meta.shape[0]);
        size_t dim = static_cast<size_t>(meta.shape[1]);

        std::cerr << "[MULTIMODAL] Extracting " << num_classes
                  << " class head relations from " << plan.name << "\n";

        std::vector<float> data = load_tensor(meta);
        if (data.empty()) continue;

        auto valid = normalize_and_get_valid(data.data(), num_classes, dim);

        for (size_t idx : valid) {
            const float* row = data.data() + idx * dim;
            Blake3Hasher::Incremental hasher;
            hasher.update(std::string_view(reinterpret_cast<const char*>(row), dim * sizeof(float)));
            // NOTE: hash computed but not used (HNSWLib disabled)

#ifdef HAS_HNSWLIB
            auto result = static_cast<hnswlib::HierarchicalNSW<float>*>(token_hnsw)->searchKnn(row, K_NEIGHBORS + 1);
            while (!result.empty()) {
                auto [dist, j] = result.top();
                result.pop();

                float sim = dist;
                if (sim >= MIN_SIMILARITY) {
                    size_t tok_idx = valid_token_indices[j];
                    const auto& comp = ctx.vocab_tokens[tok_idx].comp;
                    char target_type = (comp.children.size() <= 1) ? 'A' : 'C';
                    edges.push_back({hash, 'C', comp.hash, target_type, sim, plan.layer_idx, "CLASS_HEAD"});
                    total_edges++;
                }
            }
#endif
        }
    }

    return total_edges;
}

// ============================================================================
// Main Entry Point
// ============================================================================

size_t extract_multimodal_structures(
    PGconn* conn,
    IngestContext& ctx,
    const ModelManifest& manifest
) {
    std::cerr << "\n+--------------------------------------------------------------+\n";
    std::cerr << "|          MULTIMODAL RELATION EXTRACTION                     |\n";
    std::cerr << "+--------------------------------------------------------------+\n";

    size_t total = 0;

    // Find token embedding tensor
    const TensorMeta* token_embed = nullptr;
    std::string token_embed_name;
    for (auto& [name, meta] : ctx.tensors) {
        if (name.find("embed_tokens") != std::string::npos ||
            name.find("word_embeddings") != std::string::npos ||
            name.find("wte.weight") != std::string::npos ||
            name.find("token_embedding") != std::string::npos) {
            token_embed = &meta;
            token_embed_name = name;
            break;
        }
    }
    
    if (!token_embed) {
        std::cerr << "[MULTIMODAL] No token embedding tensor found, skipping relation extraction\n";
        return 0;
    }
    
    size_t vocab_size = ctx.get_vocab_size();
    size_t embed_dim = static_cast<size_t>(token_embed->shape[1]);
    std::vector<float> token_data = load_tensor(*token_embed);
    if (token_data.empty()) {
        std::cerr << "[MULTIMODAL] Failed to load token embeddings\n";
        return 0;
    }
    
    std::vector<size_t> valid_token_indices;
    std::vector<std::vector<float>> normalized_tokens;
    for (size_t i = 0; i < vocab_size; ++i) {
        std::vector<float> emb(embed_dim);
        std::memcpy(emb.data(), token_data.data() + i * embed_dim, embed_dim * sizeof(float));
        float norm = 0;
        for (float v : emb) norm += v * v;
        norm = std::sqrt(norm);
        if (norm >= MIN_NORM) {
            for (float& v : emb) v /= norm;
            normalized_tokens.push_back(std::move(emb));
            valid_token_indices.push_back(i);
        }
    }
    
    if (normalized_tokens.empty()) {
        std::cerr << "[MULTIMODAL] No valid token embeddings\n";
        return 0;
    }
    
    void* token_hnsw_ptr = nullptr;
    #ifdef HAS_HNSWLIB
    hnswlib::InnerProductSpace space(embed_dim);
    hnswlib::HierarchicalNSW<float> token_hnsw(&space, normalized_tokens.size(), 16, 200);
    token_hnsw.setEf(100);
    for (size_t i = 0; i < normalized_tokens.size(); ++i) {
        token_hnsw.addPoint(normalized_tokens[i].data(), i);
    }
    token_hnsw_ptr = &token_hnsw;
    #endif
    
    std::vector<RelationEdge> all_edges;

    // Filter plans by category
    std::vector<TensorExtractionPlan> object_query_plans;
    std::vector<TensorExtractionPlan> positional_plans;
    std::vector<TensorExtractionPlan> moe_router_plans;
    std::vector<TensorExtractionPlan> class_head_plans;
    std::vector<TensorExtractionPlan> chemical_plans;
    std::vector<TensorExtractionPlan> dna_plans;
    std::vector<TensorExtractionPlan> music_notes_plans;
    std::vector<TensorExtractionPlan> music_audio_plans;
    std::vector<TensorExtractionPlan> math_symbols_plans;
    std::vector<TensorExtractionPlan> math_expressions_plans;
    std::vector<TensorExtractionPlan> generic_modality_plans;

    for (const auto& plan : manifest.extraction_plans) {
        switch (plan.category) {
            case TensorCategory::OBJECT_QUERY:
                object_query_plans.push_back(plan);
                break;
            case TensorCategory::POSITION_EMBEDDING:
            case TensorCategory::POSITION_EMBEDDING_2D:
                positional_plans.push_back(plan);
                break;
            case TensorCategory::MOE_ROUTER:
                moe_router_plans.push_back(plan);
                break;
            case TensorCategory::CLASS_HEAD:
                class_head_plans.push_back(plan);
                break;
            case TensorCategory::CHEMICAL_STRUCTURES:
                chemical_plans.push_back(plan);
                break;
            case TensorCategory::DNA_SEQUENCES:
                dna_plans.push_back(plan);
                break;
            case TensorCategory::MUSIC_NOTES:
                music_notes_plans.push_back(plan);
                break;
            case TensorCategory::MUSIC_AUDIO:
                music_audio_plans.push_back(plan);
                break;
            case TensorCategory::MATH_SYMBOLS:
                math_symbols_plans.push_back(plan);
                break;
            case TensorCategory::MATH_EXPRESSIONS:
                math_expressions_plans.push_back(plan);
                break;
            case TensorCategory::GENERIC_MODALITY:
                generic_modality_plans.push_back(plan);
                break;
            default:
                break;
        }
    }

    // Extract each category
    if (!object_query_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << object_query_plans.size()
                  << " object query tensors\n";
        total += extract_object_queries(conn, ctx, object_query_plans, token_hnsw_ptr, valid_token_indices, all_edges);
    }

    if (!positional_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << positional_plans.size()
                  << " positional encoding tensors\n";
        total += extract_positional_encodings(conn, ctx, positional_plans, token_hnsw_ptr, valid_token_indices, all_edges);
    }

    if (!moe_router_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << moe_router_plans.size()
                  << " MoE router tensors\n";
        total += extract_moe_routers(conn, ctx, moe_router_plans, token_hnsw_ptr, valid_token_indices, all_edges);
    }

    if (!class_head_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << class_head_plans.size()
                  << " class head tensors\n";
        total += extract_class_heads(conn, ctx, class_head_plans, token_hnsw_ptr, valid_token_indices, all_edges);
    }

    // Extract universal digital content modalities
    if (!chemical_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << chemical_plans.size()
                  << " chemical structure tensors\n";
        total += extract_universal_modality(conn, ctx, chemical_plans, token_hnsw_ptr, valid_token_indices, all_edges, "CHEMICAL");
    }

    if (!dna_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << dna_plans.size()
                  << " DNA sequence tensors\n";
        total += extract_universal_modality(conn, ctx, dna_plans, token_hnsw_ptr, valid_token_indices, all_edges, "DNA");
    }

    if (!music_notes_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << music_notes_plans.size()
                  << " music notation tensors\n";
        total += extract_universal_modality(conn, ctx, music_notes_plans, token_hnsw_ptr, valid_token_indices, all_edges, "MUSIC_NOTES");
    }

    if (!music_audio_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << music_audio_plans.size()
                  << " music audio tensors\n";
        total += extract_universal_modality(conn, ctx, music_audio_plans, token_hnsw_ptr, valid_token_indices, all_edges, "MUSIC_AUDIO");
    }

    if (!math_symbols_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << math_symbols_plans.size()
                  << " math symbol tensors\n";
        total += extract_universal_modality(conn, ctx, math_symbols_plans, token_hnsw_ptr, valid_token_indices, all_edges, "MATH_SYMBOLS");
    }

    if (!math_expressions_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << math_expressions_plans.size()
                  << " math expression tensors\n";
        total += extract_universal_modality(conn, ctx, math_expressions_plans, token_hnsw_ptr, valid_token_indices, all_edges, "MATH_EXPRESSIONS");
    }

    if (!generic_modality_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << generic_modality_plans.size()
                  << " generic modality tensors\n";
        total += extract_universal_modality(conn, ctx, generic_modality_plans, token_hnsw_ptr, valid_token_indices, all_edges, "GENERIC_MODALITY");
    }

    if (!all_edges.empty()) {
        hypercube::db::Transaction tx(conn);

        // Deduplicate edges using unique constraint key: (source_id, target_id, relation_type, source_model, layer, component)
        struct EdgeKey {
            Blake3Hash source_id, target_id;
            char relation_type;
            std::string source_model;
            int layer;
            std::string component;

            bool operator<(const EdgeKey& other) const {
                int cmp = memcmp(source_id.data(), other.source_id.data(), 32);
                if (cmp != 0) return cmp < 0;
                cmp = memcmp(target_id.data(), other.target_id.data(), 32);
                if (cmp != 0) return cmp < 0;
                if (relation_type != other.relation_type) return relation_type < other.relation_type;
                if (source_model != other.source_model) return source_model < other.source_model;
                if (layer != other.layer) return layer < other.layer;
                return component < other.component;
            }
        };

        std::map<EdgeKey, RelationEdge> deduped_edges;
        for (const auto& edge : all_edges) {
            EdgeKey key{edge.source_id, edge.target_id, 'A', ctx.model_prefix, edge.layer, edge.component};
            auto it = deduped_edges.find(key);
            if (it == deduped_edges.end()) {
                deduped_edges[key] = edge;
            } else {
                // Average weights for duplicates
                it->second.weight = (it->second.weight + edge.weight) / 2.0f;
            }
        }

        std::cerr << "[MULTIMODAL] Deduplicated " << all_edges.size() << " -> " << deduped_edges.size() << " unique edges\n";

        // First, ensure multimodal compositions exist
        std::unordered_map<Blake3Hash, std::string, hypercube::Blake3HashHasher> multimodal_sources;
        for (const auto& [key, edge] : deduped_edges) {
            if (edge.source_type == 'C') {
                std::string label = "multimodal:" + edge.component + ":" + std::to_string(edge.layer);
                multimodal_sources[edge.source_id] = label;
            }
        }

        if (!multimodal_sources.empty()) {
            std::string comp_insert = "INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi) VALUES ";
            std::vector<std::string> comp_values;

            for (const auto& [hash, label] : multimodal_sources) {
                std::string hex_id = "\\x" + hash.to_hex();
                std::string centroid = "ST_SetSRID(ST_MakePoint(0, 0, 0, 0), 0)";  // Dummy centroid

                comp_values.push_back("('" + hex_id + "', '" + label + "', 1, 0, 0, " + centroid + ", 0, 0)");
            }

            if (!comp_values.empty()) {
                for (size_t i = 0; i < comp_values.size(); ++i) {
                    if (i > 0) comp_insert += ", ";
                    comp_insert += comp_values[i];
                }
                comp_insert += " ON CONFLICT (id) DO NOTHING";

                PGresult* comp_res = PQexec(conn, comp_insert.c_str());
                if (PQresultStatus(comp_res) != PGRES_COMMAND_OK) {
                    std::cerr << "[MULTIMODAL] Failed to insert multimodal compositions: " << PQerrorMessage(conn) << "\n";
                } else {
                    int inserted = atoi(PQcmdTuples(comp_res));
                    std::cerr << "[MULTIMODAL] Inserted " << inserted << " multimodal compositions\n";
                }
                PQclear(comp_res);
            }
        }

        // Use temp table then INSERT with existence checks
        PQexec(conn, "DROP TABLE IF EXISTS tmp_multimodal_rel");
        PQexec(conn, "CREATE TEMP TABLE tmp_multimodal_rel ("
                     "source_type CHAR(1), source_id BYTEA, target_type CHAR(1), target_id BYTEA, "
                     "relation_type CHAR(1), weight REAL, source_model TEXT, layer INTEGER, component TEXT)");

        std::string copy_cmd = "COPY tmp_multimodal_rel FROM STDIN WITH (FORMAT text)";
        hypercube::db::CopyStream copy(conn, copy_cmd.c_str());
        if (!copy.ok()) {
            std::cerr << "[MULTIMODAL] COPY start failed: " << copy.error() << "\n";
            return 0;
        }

        for (const auto& [key, edge] : deduped_edges) {
            std::string row;
            row += edge.source_type;
            row += '\t';
            row += "\\\\x";
            row += edge.source_id.to_hex();
            row += '\t';
            row += edge.target_type;
            row += '\t';
            row += "\\\\x";
            row += edge.target_id.to_hex();
            row += '\t';
            row += 'A';
            row += '\t';
            row += std::to_string(edge.weight);
            row += '\t';
            row += ctx.model_prefix;
            row += '\t';
            row += std::to_string(edge.layer);
            row += '\t';
            row += edge.component;
            row += '\n';

            if (!copy.put(row)) {
                std::cerr << "[MULTIMODAL] COPY data failed: " << copy.error() << "\n";
                return 0;
            }
        }

        if (!copy.end()) {
            std::cerr << "[MULTIMODAL] COPY to temp failed: " << copy.error() << "\n";
            return 0;
        }

        // INSERT with existence checks for source and target
        PGresult* ins_res = PQexec(conn,
            "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component) "
            "SELECT source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component "
            "FROM tmp_multimodal_rel t "
            "WHERE ((t.source_type = 'A' AND EXISTS (SELECT 1 FROM atom WHERE id = t.source_id)) "
            "       OR (t.source_type = 'C' AND EXISTS (SELECT 1 FROM composition WHERE id = t.source_id))) "
            "AND ((t.target_type = 'A' AND EXISTS (SELECT 1 FROM atom WHERE id = t.target_id)) "
            "     OR (t.target_type = 'C' AND EXISTS (SELECT 1 FROM composition WHERE id = t.target_id))) "
            "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
            "weight = EXCLUDED.weight");

        int inserted = (PQresultStatus(ins_res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(ins_res)) : 0;
        if (PQresultStatus(ins_res) != PGRES_COMMAND_OK) {
            std::cerr << "[MULTIMODAL] Insert failed: " << PQerrorMessage(conn) << "\n";
        }
        PQclear(ins_res);

        tx.commit();
        std::cerr << "[MULTIMODAL] Inserted " << inserted << " multimodal relations (filtered " << (deduped_edges.size() - inserted) << " missing refs)\n";
    }

    std::cerr << "+--------------------------------------------------------------+\n";
    std::cerr << "|  MULTIMODAL TOTAL: " << std::setw(6) << total << " relations extracted"
              << std::string(18, ' ') << "|\n";
    std::cerr << "+--------------------------------------------------------------+\n\n";

    return total;
}

} // namespace ingest
} // namespace hypercube
