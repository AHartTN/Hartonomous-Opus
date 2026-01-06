/**
 * @file multimodal_extraction.cpp
 * @brief Multimodal semantic extraction for DETR, Florence, MoE, vision models
 *
 * Extracts semantic structures that don't fit the token-embedding paradigm:
 *   - Object queries (DETR slots, visual "tokens")
 *   - 2D positional encodings (row/column embeddings)
 *   - MoE router weights (THE ROUTING MACHINE)
 *   - Class heads (detection prototypes)
 *
 * These become first-class atoms/compositions in the hypercube.
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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hypercube {
namespace ingest {

// ============================================================================
// Configuration
// ============================================================================

static constexpr float MIN_NORM = 0.01f;         // Skip near-zero vectors
static constexpr size_t BATCH_SIZE = 1000;       // DB batch insert size

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
    }
    return data;
}

// ============================================================================
// Escape string for SQL (using libpq)
// ============================================================================

static std::string escape_sql_string(PGconn* conn, const std::string& str) {
    char* escaped = PQescapeLiteral(conn, str.c_str(), str.length());
    if (!escaped) {
        // Fallback: basic escape
        std::string result = "'";
        for (char c : str) {
            if (c == '\'') result += "''";
            else if (c == '\\') result += "\\\\";
            else result += c;
        }
        result += "'";
        return result;
    }
    std::string result(escaped);
    PQfreemem(escaped);
    return result;
}

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
    const std::vector<TensorExtractionPlan>& plans
) {
    size_t total_inserted = 0;

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
                  << " object queries from " << plan.name << " [" << dim << "d]\n";

        // Load tensor data
        std::vector<float> data = load_tensor(meta);
        if (data.empty()) {
            std::cerr << "[MULTIMODAL] Failed to load tensor: " << plan.name << "\n";
            continue;
        }

        // Normalize
        auto valid = normalize_and_get_valid(data.data(), num_queries, dim);

        // Insert as atoms (object queries are semantic anchors)
        std::ostringstream sql;
        sql << "INSERT INTO atoms (id, name, value, source_type, source_info) VALUES ";

        bool first = true;
        size_t batch_count = 0;

        for (size_t idx : valid) {
            // Create atom for this query slot
            std::string name = ctx.model_prefix + "query_" + std::to_string(idx);
            std::string source_info = plan.name + "[" + std::to_string(idx) + "]";

            // Compute hash for the embedding vector
            const float* row = data.data() + idx * dim;
            Blake3Hash hash = Blake3Hash::hash(
                reinterpret_cast<const uint8_t*>(row),
                dim * sizeof(float)
            );

            if (!first) sql << ",";
            first = false;

            sql << "(" << db::to_bytea_literal(hash) << ","
                << escape_sql_string(conn, name) << ","
                << "'object_query',"
                << "'tensor',"
                << escape_sql_string(conn, source_info)
                << ")";

            batch_count++;

            if (batch_count >= BATCH_SIZE) {
                sql << " ON CONFLICT (id) DO NOTHING";
                db::Result res = db::exec(conn, sql.str());
                if (!res.ok()) {
                    std::cerr << "[MULTIMODAL] Insert error: " << res.error_message() << "\n";
                }
                total_inserted += batch_count;

                // Reset
                sql.str("");
                sql << "INSERT INTO atoms (id, name, value, source_type, source_info) VALUES ";
                first = true;
                batch_count = 0;
            }
        }

        // Final batch
        if (batch_count > 0) {
            sql << " ON CONFLICT (id) DO NOTHING";
            db::Result res = db::exec(conn, sql.str());
            if (!res.ok()) {
                std::cerr << "[MULTIMODAL] Insert error: " << res.error_message() << "\n";
            }
            total_inserted += batch_count;
        }

        std::cerr << "[MULTIMODAL] Inserted " << valid.size()
                  << " object query atoms\n";
    }

    return total_inserted;
}

// ============================================================================
// Extract 2D Positional Encodings
// ============================================================================

size_t extract_positional_encodings(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans
) {
    size_t total_inserted = 0;

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
                  << " " << type_str << " (" << axis << ") from " << plan.name << "\n";

        std::vector<float> data = load_tensor(meta);
        if (data.empty()) continue;

        auto valid = normalize_and_get_valid(data.data(), num_positions, dim);

        std::ostringstream sql;
        sql << "INSERT INTO atoms (id, name, value, source_type, source_info) VALUES ";

        bool first = true;
        size_t batch_count = 0;

        for (size_t idx : valid) {
            std::string name = ctx.model_prefix + "pos_" + axis + "_" + std::to_string(idx);
            std::string source_info = plan.name + "[" + std::to_string(idx) + "]";

            const float* row = data.data() + idx * dim;
            Blake3Hash hash = Blake3Hash::hash(
                reinterpret_cast<const uint8_t*>(row),
                dim * sizeof(float)
            );

            if (!first) sql << ",";
            first = false;

            sql << "(" << db::to_bytea_literal(hash) << ","
                << escape_sql_string(conn, name) << ","
                << escape_sql_string(conn, type_str) << ","
                << "'tensor',"
                << escape_sql_string(conn, source_info)
                << ")";

            batch_count++;

            if (batch_count >= BATCH_SIZE) {
                sql << " ON CONFLICT (id) DO NOTHING";
                db::exec(conn, sql.str());
                total_inserted += batch_count;

                sql.str("");
                sql << "INSERT INTO atoms (id, name, value, source_type, source_info) VALUES ";
                first = true;
                batch_count = 0;
            }
        }

        if (batch_count > 0) {
            sql << " ON CONFLICT (id) DO NOTHING";
            db::exec(conn, sql.str());
            total_inserted += batch_count;
        }
    }

    return total_inserted;
}

// ============================================================================
// Extract MoE Routers (THE ROUTING MACHINE)
// ============================================================================

size_t extract_moe_routers(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans
) {
    size_t total_inserted = 0;

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

        // Insert router atoms
        std::ostringstream sql;
        sql << "INSERT INTO atoms (id, name, value, source_type, source_info) VALUES ";

        bool first = true;

        for (size_t idx : valid) {
            std::string name = ctx.model_prefix + "router_layer" +
                              std::to_string(plan.layer_idx) + "_expert" + std::to_string(idx);
            std::string source_info = plan.name + "[" + std::to_string(idx) + "]";

            const float* row = data.data() + idx * d_model;
            Blake3Hash hash = Blake3Hash::hash(
                reinterpret_cast<const uint8_t*>(row),
                d_model * sizeof(float)
            );

            if (!first) sql << ",";
            first = false;

            sql << "(" << db::to_bytea_literal(hash) << ","
                << escape_sql_string(conn, name) << ","
                << "'moe_router',"
                << "'tensor',"
                << escape_sql_string(conn, source_info)
                << ")";
        }

        if (!first) {
            sql << " ON CONFLICT (id) DO NOTHING";
            db::Result res = db::exec(conn, sql.str());
            if (!res.ok()) {
                std::cerr << "[MULTIMODAL] Router insert error: " << res.error_message() << "\n";
            }
            total_inserted += valid.size();
        }

        std::cerr << "[MULTIMODAL] Inserted " << valid.size() << " router atoms\n";
    }

    return total_inserted;
}

// ============================================================================
// Extract Class Heads (Detection)
// ============================================================================

size_t extract_class_heads(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans
) {
    size_t total_inserted = 0;

    for (const auto& plan : plans) {
        if (plan.category != TensorCategory::CLASS_HEAD) continue;

        auto it = ctx.tensors.find(plan.name);
        if (it == ctx.tensors.end()) continue;

        const TensorMeta& meta = it->second;
        if (meta.shape.size() != 2) continue;

        size_t num_classes = static_cast<size_t>(meta.shape[0]);
        size_t dim = static_cast<size_t>(meta.shape[1]);

        std::cerr << "[MULTIMODAL] Extracting " << num_classes
                  << " class prototypes from " << plan.name << "\n";

        std::vector<float> data = load_tensor(meta);
        if (data.empty()) continue;

        auto valid = normalize_and_get_valid(data.data(), num_classes, dim);

        std::ostringstream sql;
        sql << "INSERT INTO atoms (id, name, value, source_type, source_info) VALUES ";

        bool first = true;
        size_t batch_count = 0;

        for (size_t idx : valid) {
            std::string name = ctx.model_prefix + "class_" + std::to_string(idx);
            std::string source_info = plan.name + "[" + std::to_string(idx) + "]";

            const float* row = data.data() + idx * dim;
            Blake3Hash hash = Blake3Hash::hash(
                reinterpret_cast<const uint8_t*>(row),
                dim * sizeof(float)
            );

            if (!first) sql << ",";
            first = false;

            sql << "(" << db::to_bytea_literal(hash) << ","
                << escape_sql_string(conn, name) << ","
                << "'class_prototype',"
                << "'tensor',"
                << escape_sql_string(conn, source_info)
                << ")";

            batch_count++;

            if (batch_count >= BATCH_SIZE) {
                sql << " ON CONFLICT (id) DO NOTHING";
                db::exec(conn, sql.str());
                total_inserted += batch_count;

                sql.str("");
                sql << "INSERT INTO atoms (id, name, value, source_type, source_info) VALUES ";
                first = true;
                batch_count = 0;
            }
        }

        if (batch_count > 0) {
            sql << " ON CONFLICT (id) DO NOTHING";
            db::exec(conn, sql.str());
            total_inserted += batch_count;
        }
    }

    return total_inserted;
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
    std::cerr << "|          MULTIMODAL EXTRACTION                               |\n";
    std::cerr << "+--------------------------------------------------------------+\n";

    size_t total = 0;

    // Filter plans by category
    std::vector<TensorExtractionPlan> object_query_plans;
    std::vector<TensorExtractionPlan> positional_plans;
    std::vector<TensorExtractionPlan> moe_router_plans;
    std::vector<TensorExtractionPlan> class_head_plans;

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
            default:
                break;
        }
    }

    // Extract each category
    if (!object_query_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << object_query_plans.size()
                  << " object query tensors\n";
        total += extract_object_queries(conn, ctx, object_query_plans);
    }

    if (!positional_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << positional_plans.size()
                  << " positional encoding tensors\n";
        total += extract_positional_encodings(conn, ctx, positional_plans);
    }

    if (!moe_router_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << moe_router_plans.size()
                  << " MoE router tensors\n";
        total += extract_moe_routers(conn, ctx, moe_router_plans);
    }

    if (!class_head_plans.empty()) {
        std::cerr << "[MULTIMODAL] Processing " << class_head_plans.size()
                  << " class head tensors\n";
        total += extract_class_heads(conn, ctx, class_head_plans);
    }

    std::cerr << "+--------------------------------------------------------------+\n";
    std::cerr << "|  MULTIMODAL TOTAL: " << std::setw(6) << total << " atoms inserted"
              << std::string(22, ' ') << "|\n";
    std::cerr << "+--------------------------------------------------------------+\n\n";

    return total;
}

} // namespace ingest
} // namespace hypercube
