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
#include "hypercube/ingest/safetensor.hpp"

namespace hypercube {
namespace ingest {

// Main entry point - extracts all multimodal structures
size_t extract_multimodal_structures(
    PGconn* conn,
    IngestContext& ctx,
    const ModelManifest& manifest,
    const SafetensorFile& stfile
);

// Individual extractors (for fine-grained control)
size_t extract_object_queries(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans,
    const SafetensorFile& stfile
);

size_t extract_positional_encodings(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans,
    const SafetensorFile& stfile
);

size_t extract_moe_routers(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans,
    const SafetensorFile& stfile
);

size_t extract_class_heads(
    PGconn* conn,
    IngestContext& ctx,
    const std::vector<TensorExtractionPlan>& plans,
    const SafetensorFile& stfile
);

} // namespace ingest
} // namespace hypercube
