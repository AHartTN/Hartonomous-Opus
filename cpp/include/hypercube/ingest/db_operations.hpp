/**
 * @file db_operations.hpp
 * @brief Database insertion operations for model ingestion
 * 
 * This header declares the large database insertion functions used during
 * model ingestion. These functions handle bulk COPY operations, temp tables,
 * and transaction management for high-performance data loading.
 * 
 * Functions:
 *   - insert_compositions: Insert vocab token compositions with geometry
 *   - insert_tensor_hierarchy: Build and insert tensor path hierarchy
 *   - extract_embedding_relations: Build k-NN similarity graph from embeddings
 *   - insert_attention_relations: Extract attention/weight similarity relations
 */

#ifndef HYPERCUBE_INGEST_DB_OPERATIONS_HPP
#define HYPERCUBE_INGEST_DB_OPERATIONS_HPP

// Prevent Windows min/max macros from breaking std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <libpq-fe.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

// Internal headers
#include "hypercube/ingest/context.hpp"
#include "hypercube/ingest/geometry.hpp"
#include "hypercube/ingest/parsing.hpp"
#include "hypercube/embedding_ops.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/laplacian_4d.hpp"

// HNSWLIB for k-NN
#ifdef HAS_HNSWLIB
#include <hnswlib/hnswlib.h>
#endif

namespace hypercube {
namespace ingest {
namespace db {

// Use TensorMeta from safetensor namespace
using TensorMeta = safetensor::TensorMeta;

/**
 * @brief Insert vocab token compositions into composition + composition_child tables
 * 
 * Inserts multi-character token compositions with computed geometry:
 *   - geom: LINESTRINGZM from child atom coordinates
 *   - centroid: POINTZM from averaged coordinates
 *   - hilbert_lo/hi: Hilbert curve range for spatial indexing
 * 
 * Uses parallel batch building and COPY streaming for high throughput.
 * 
 * @param conn PostgreSQL connection
 * @param ctx Ingest context containing vocab_tokens
 * @return true on success, false on database error
 */
bool insert_compositions(PGconn* conn, IngestContext& ctx);

/**
 * @brief Build and insert tensor path hierarchy as compositions
 * 
 * Parses tensor names like "model.layers.0.self_attn.q_proj.weight" into
 * hierarchical path components and inserts them as composition records
 * with parent-child relationships.
 * 
 * Creates two types of composition children:
 *   - 'A' (atom): Character atoms that make up the path label
 *   - 'C' (composition): Sub-path compositions forming the hierarchy
 * 
 * @param conn PostgreSQL connection
 * @param ctx Ingest context containing tensors map
 * @param config Ingest configuration (for model name)
 * @return true on success, false on database error
 */
bool insert_tensor_hierarchy(PGconn* conn, IngestContext& ctx, const IngestConfig& config);

/**
 * @brief Extract k-NN similarity relations from embedding tensors
 * 
 * Builds a k-NN similarity graph from model embeddings using HNSWLIB.
 * Supports multiple embedding types with per-type thresholds:
 *   - token embeddings (threshold 0.45)
 *   - patch embeddings for vision models (threshold 0.25)
 *   - position embeddings (threshold 0.02)
 *   - projection embeddings (threshold 0.15)
 * 
 * Inserts edges as relation records with type 'E' (embedding similarity).
 * 
 * @param conn PostgreSQL connection
 * @param ctx Ingest context containing tensors and vocab_tokens
 * @param config Ingest configuration (for model name)
 * @return true on success, false on database error
 */
bool extract_embedding_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);

/**
 * @brief Extract attention and weight similarity relations
 * 
 * Extracts sparse relations from model weight tensors:
 * 
 * Part 1: Router weights (MoE models)
 *   - Token-to-expert routing edges with type 'R'
 * 
 * Part 2: Attention projections (Q/K/V/O, FFN, etc.)
 *   - Row similarity edges showing related output dimensions
 *   - Uses type 'W' for weight similarity
 * 
 * Part 3: Token-to-dimension mappings
 *   - Shows which embedding dimensions each token activates
 *   - Uses type 'D' for dimension activation
 * 
 * @param conn PostgreSQL connection
 * @param ctx Ingest context containing tensors and vocab_tokens
 * @param config Ingest configuration (for model name)
 * @return true on success, false on database error
 */
bool insert_attention_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);

/**
 * @brief Helper to read a single tensor row
 * 
 * Reads one row from a tensor, handling different dtypes and file I/O.
 * 
 * @param meta Tensor metadata including file path and offsets
 * @param row_idx Row index to read
 * @return Vector of float values, empty on error
 */
std::vector<float> read_tensor_row(const TensorMeta& meta, size_t row_idx);

/**
 * @brief Extract semantic relations from ALL projection matrices
 * 
 * Complete multi-model semantic extraction that processes:
 *   - Base embeddings → token similarity in embedding space
 *   - Q projections per layer → query-space similarity
 *   - K projections per layer → key-space similarity
 *   - V projections per layer → value-space similarity
 *   - FFN projections per layer → feed-forward similarity
 * 
 * Each model contributes relations tagged with source_model, layer, component.
 * Relations ACCUMULATE across models - no overwriting.
 * The semantic substrate emerges from consensus across all model perspectives.
 * 
 * @param conn PostgreSQL connection
 * @param ctx Ingest context containing tensors and vocab_tokens
 * @param config Ingest configuration (for model name)
 * @return true on success, false on database error
 */
bool extract_all_semantic_relations(PGconn* conn, IngestContext& ctx, const IngestConfig& config);

/**
 * @brief Project token embeddings to 4D using Laplacian eigenmaps and update database
 *
 * Uses Laplacian eigenmap projection to map high-dimensional token embeddings
 * to 4D hypercube coordinates that preserve semantic relationships.
 * Updates the composition centroids in the database with the projected coordinates.
 *
 * This provides semantically meaningful geometry instead of Unicode-based coordinates.
 *
 * @param conn PostgreSQL connection
 * @param ctx Ingest context containing tensors and vocab_tokens
 * @param config Ingest configuration (for model name)
 * @param anchors Optional anchor points for Procrustes alignment to existing atom coordinates
 * @return true on success, false on database error
 */
bool project_and_update_embeddings(PGconn* conn, IngestContext& ctx, const IngestConfig& config,
                                   const std::vector<AnchorPoint>& anchors = {});

} // namespace db
} // namespace ingest
} // namespace hypercube

#endif // HYPERCUBE_INGEST_DB_OPERATIONS_HPP
