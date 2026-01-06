#pragma once
// =============================================================================
// MULTIMODAL EXTRACTION - Header
// =============================================================================
//
// Extracts semantic structures from all model types:
//   - Object queries (DETR, Florence)
//   - Positional encodings (1D, 2D)
//   - MoE routers and experts
//   - Class heads (detection)
//   - Vision features
// =============================================================================

#include <cstddef>
#include <libpq-fe.h>
#include "hypercube/ingest/context.hpp"
#include "hypercube/ingest/model_manifest.hpp"

namespace hypercube {
namespace ingest {

// Main entry point - extracts all multimodal structures
// Uses IngestContext's tensors map for tensor data access
size_t extract_multimodal_structures(
    PGconn* conn,
    IngestContext& ctx,
    const ModelManifest& manifest
);

// Individual extractors (for fine-grained control)
size_t extract_object_queries(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans
);

size_t extract_positional_encodings(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans
);

size_t extract_moe_routers(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans
);

size_t extract_class_heads(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans
);

} // namespace ingest
} // namespace hypercube
