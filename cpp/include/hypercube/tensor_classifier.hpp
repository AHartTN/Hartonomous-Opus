/**
 * @file tensor_classifier.hpp
 * @brief Language-agnostic tensor component classification
 *
 * Classifies neural network tensor components based on mathematical structure
 * rather than English naming conventions. Supports any language/architecture.
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include "hypercube/types.hpp"

namespace hypercube {
namespace ingest {

// ============================================================================
// Tensor Component Types (Language Agnostic)
// ============================================================================

enum class TensorComponent {
    UNKNOWN,

    // Root-level components
    TOKEN_EMBEDDINGS,      // [vocab_size, hidden_dim] - token to vector
    POSITION_EMBEDDINGS,   // [seq_len, hidden_dim] - position to vector
    OUTPUT_PROJECTION,     // [hidden_dim, vocab_size] - logits generation

    // Attention mechanism
    ATTENTION_QUERY,       // [hidden_dim, hidden_dim] - query projection
    ATTENTION_KEY,         // [hidden_dim, hidden_dim] - key projection
    ATTENTION_VALUE,       // [hidden_dim, hidden_dim] - value projection
    ATTENTION_OUTPUT,      // [hidden_dim, hidden_dim] - output projection

    // MLP/Feed-forward layers
    MLP_GATE,             // [hidden_dim, intermediate_dim] - gating mechanism
    MLP_UP,               // [hidden_dim, intermediate_dim] - expansion
    MLP_DOWN,             // [intermediate_dim, hidden_dim] - compression

    // Mixture of Experts
    ROUTER_WEIGHTS,       // [num_experts, hidden_dim] - expert routing

    // Layer normalization
    LAYER_NORM,           // [hidden_dim] - normalization weights

    // Vision components
    PATCH_EMBEDDING,      // [patch_seq, hidden_dim] - patch to vector
    VISUAL_PROJECTION,    // [hidden_dim, hidden_dim] - vision-language alignment
};

// ============================================================================
// Structural Classification Rules
// ============================================================================

struct TensorShape {
    std::vector<int64_t> dims;
    bool is_valid() const { return !dims.empty(); }
    size_t ndims() const { return dims.size(); }
    int64_t operator[](size_t i) const { return dims[i]; }
};

struct ClassificationContext {
    // Global model statistics for relative comparisons
    int64_t typical_hidden_dim = 768;  // Will be updated during analysis
    int64_t typical_vocab_size = 30000;
    int64_t typical_seq_len = 512;
    int64_t num_layers = 12;

    // Hierarchical position info
    std::string path;  // e.g. "layers.0.attention.self.query"
    int layer_idx = -1;
    std::string subcomponent;  // e.g. "attention", "mlp"
};

// ============================================================================
// Language-Agnostic Tensor Classifier
// ============================================================================

class TensorClassifier {
public:
    /**
     * Classify tensor component based on shape and hierarchical position
     */
    static TensorComponent classify(const TensorShape& shape,
                                  const ClassificationContext& ctx);

    /**
     * Extract structural information from tensor path
     */
    static ClassificationContext analyze_path(const std::string& tensor_path);

    /**
     * Update global model statistics from observed tensors
     */
    void update_model_stats(const std::vector<TensorShape>& all_shapes);

    /**
     * Get human-readable description of component type
     */
    static std::string component_name(TensorComponent comp);

private:
    // Shape-based classification rules
    static TensorComponent classify_by_shape(const TensorShape& shape,
                                           const ClassificationContext& ctx);

    // Hierarchy-based refinement
    static TensorComponent refine_by_hierarchy(TensorComponent base_type,
                                             const ClassificationContext& ctx);

    // Helper functions for common patterns
    static bool is_square_matrix(const TensorShape& shape);
    static bool is_expansion_layer(const TensorShape& shape, int64_t hidden_dim);
    static bool is_compression_layer(const TensorShape& shape, int64_t hidden_dim);
    static bool is_embedding_layer(const TensorShape& shape, const ClassificationContext& ctx);
    static bool is_output_projection(const TensorShape& shape, const ClassificationContext& ctx);
    static bool is_router_weights(const TensorShape& shape, const ClassificationContext& ctx);
};

// ============================================================================
// Implementation
// ============================================================================

inline bool TensorClassifier::is_square_matrix(const TensorShape& shape) {
    return shape.ndims() == 2 && shape[0] == shape[1] && shape[0] > 64;
}

inline bool TensorClassifier::is_expansion_layer(const TensorShape& shape,
                                                int64_t hidden_dim) {
    if (shape.ndims() != 2) return false;
    int64_t in_dim = shape[0], out_dim = shape[1];
    return in_dim == hidden_dim && out_dim > in_dim * 2;  // Typically 4x expansion
}

inline bool TensorClassifier::is_compression_layer(const TensorShape& shape,
                                                  int64_t hidden_dim) {
    if (shape.ndims() != 2) return false;
    int64_t in_dim = shape[0], out_dim = shape[1];
    return out_dim == hidden_dim && in_dim > out_dim * 2;  // Typically 4x -> hidden
}

inline bool TensorClassifier::is_embedding_layer(const TensorShape& shape,
                                                const ClassificationContext& ctx) {
    if (shape.ndims() != 2) return false;
    int64_t vocab_or_seq = shape[0], hidden_dim = shape[1];

    // Token embeddings: large vocab, reasonable hidden dim
    if (vocab_or_seq > 1000 && hidden_dim > 64 && hidden_dim < 10000) {
        return true;
    }

    // Position embeddings: sequence length pattern
    if (vocab_or_seq < 10000 && hidden_dim == ctx.typical_hidden_dim) {
        return vocab_or_seq <= ctx.typical_seq_len * 2;  // Allow some margin
    }

    return false;
}

inline bool TensorClassifier::is_output_projection(const TensorShape& shape,
                                                  const ClassificationContext& ctx) {
    if (shape.ndims() != 2) return false;
    int64_t in_dim = shape[0], out_dim = shape[1];
    return in_dim == ctx.typical_hidden_dim && out_dim == ctx.typical_vocab_size;
}

inline bool TensorClassifier::is_router_weights(const TensorShape& shape,
                                              const ClassificationContext& ctx) {
    if (shape.ndims() != 2) return false;
    int64_t num_experts = shape[0], hidden_dim = shape[1];
    return num_experts > 1 && num_experts < 100 && hidden_dim == ctx.typical_hidden_dim;
}

TensorComponent TensorClassifier::classify(const TensorShape& shape,
                                         const ClassificationContext& ctx) {
    TensorComponent base_type = classify_by_shape(shape, ctx);
    return refine_by_hierarchy(base_type, ctx);
}

TensorComponent TensorClassifier::classify_by_shape(const TensorShape& shape,
                                                   const ClassificationContext& ctx) {
    if (!shape.is_valid()) return TensorComponent::UNKNOWN;

    // 2D tensors
    if (shape.ndims() == 2) {
        int64_t d0 = shape[0];
        [[maybe_unused]] int64_t d1 = shape[1];

        // Square matrices - attention projections
        if (is_square_matrix(shape)) {
            return TensorComponent::ATTENTION_QUERY;  // Base type, refined by hierarchy
        }

        // Expansion layers (hidden -> intermediate)
        if (is_expansion_layer(shape, ctx.typical_hidden_dim)) {
            return TensorComponent::MLP_UP;
        }

        // Compression layers (intermediate -> hidden)
        if (is_compression_layer(shape, ctx.typical_hidden_dim)) {
            return TensorComponent::MLP_DOWN;
        }

        // Output projection (hidden -> vocab)
        if (is_output_projection(shape, ctx)) {
            return TensorComponent::OUTPUT_PROJECTION;
        }

        // Router weights (experts -> hidden)
        if (is_router_weights(shape, ctx)) {
            return TensorComponent::ROUTER_WEIGHTS;
        }

        // Embeddings (vocab/seq -> hidden)
        if (is_embedding_layer(shape, ctx)) {
            // Distinguish token vs position by relative dimensions
            if (d0 > ctx.typical_vocab_size / 2) {
                return TensorComponent::TOKEN_EMBEDDINGS;
            } else {
                return TensorComponent::POSITION_EMBEDDINGS;
            }
        }
    }

    // 1D tensors - layer norm weights
    if (shape.ndims() == 1 && shape[0] == ctx.typical_hidden_dim) {
        return TensorComponent::LAYER_NORM;
    }

    return TensorComponent::UNKNOWN;
}

TensorComponent TensorClassifier::refine_by_hierarchy(TensorComponent base_type,
                                                     const ClassificationContext& ctx) {
    // Use hierarchical path to refine base classification
    std::string path = ctx.path;

    // Attention components
    if (path.find("attention") != std::string::npos ||
        path.find("attn") != std::string::npos) {

        if (base_type == TensorComponent::ATTENTION_QUERY) {
            if (path.find("query") != std::string::npos || path.find("q_proj") != std::string::npos) {
                return TensorComponent::ATTENTION_QUERY;
            }
            if (path.find("key") != std::string::npos || path.find("k_proj") != std::string::npos) {
                return TensorComponent::ATTENTION_KEY;
            }
            if (path.find("value") != std::string::npos || path.find("v_proj") != std::string::npos) {
                return TensorComponent::ATTENTION_VALUE;
            }
            if (path.find("output") != std::string::npos || path.find("o_proj") != std::string::npos) {
                return TensorComponent::ATTENTION_OUTPUT;
            }
        }
    }

    // MLP components - but shape already distinguishes these reliably
    // Gate vs Up distinction needs more context, but expansion pattern is clear

    return base_type;  // No refinement needed
}

ClassificationContext TensorClassifier::analyze_path(const std::string& tensor_path) {
    ClassificationContext ctx;
    ctx.path = tensor_path;

    // Extract layer index
    size_t layer_pos = tensor_path.find("layers.");
    if (layer_pos != std::string::npos) {
        size_t num_start = layer_pos + 7;
        size_t num_end = tensor_path.find(".", num_start);
        if (num_end != std::string::npos) {
            try {
                ctx.layer_idx = std::stoi(tensor_path.substr(num_start, num_end - num_start));
            } catch (...) {
                ctx.layer_idx = -1;
            }
        }
    }

    // Extract subcomponent (attention, mlp, etc.)
    if (tensor_path.find("attention") != std::string::npos) {
        ctx.subcomponent = "attention";
    } else if (tensor_path.find("mlp") != std::string::npos ||
               tensor_path.find("ffn") != std::string::npos) {
        ctx.subcomponent = "mlp";
    }

    return ctx;
}

void TensorClassifier::update_model_stats(const std::vector<TensorShape>& all_shapes) {
    // This would analyze all tensor shapes to determine typical dimensions
    // Implementation would find most common dimension patterns
}

std::string TensorClassifier::component_name(TensorComponent comp) {
    switch (comp) {
        case TensorComponent::TOKEN_EMBEDDINGS: return "token_embeddings";
        case TensorComponent::POSITION_EMBEDDINGS: return "position_embeddings";
        case TensorComponent::OUTPUT_PROJECTION: return "output_projection";
        case TensorComponent::ATTENTION_QUERY: return "attention_query";
        case TensorComponent::ATTENTION_KEY: return "attention_key";
        case TensorComponent::ATTENTION_VALUE: return "attention_value";
        case TensorComponent::ATTENTION_OUTPUT: return "attention_output";
        case TensorComponent::MLP_GATE: return "mlp_gate";
        case TensorComponent::MLP_UP: return "mlp_up";
        case TensorComponent::MLP_DOWN: return "mlp_down";
        case TensorComponent::ROUTER_WEIGHTS: return "router_weights";
        case TensorComponent::LAYER_NORM: return "layer_norm";
        case TensorComponent::PATCH_EMBEDDING: return "patch_embedding";
        case TensorComponent::VISUAL_PROJECTION: return "visual_projection";
        default: return "unknown";
    }
}

} // namespace ingest
} // namespace hypercube