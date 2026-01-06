// =============================================================================
// context.hpp - Shared Ingestion Context (State Container)
// =============================================================================
// Replaces global variables with a structured context object that can be
// passed to modular parsing and insertion functions.
// =============================================================================

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#include "hypercube/types.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/ingest/safetensor.hpp"
#include "hypercube/ingest/model_manifest.hpp"

namespace hypercube {
namespace ingest {

// =============================================================================
// Extended TensorMeta for main ingester (adds nothing, base is sufficient)
// =============================================================================

using TensorMeta = safetensor::TensorMeta;

// Alias IngestConfig from safetensor namespace to avoid duplication
using IngestConfig = safetensor::IngestConfig;

// =============================================================================
// TokenInfo - Token with pre-computed composition data
// =============================================================================

struct TokenInfo {
    std::string text;
    CompositionRecord comp;  // Full composition with hash, coords, children
};

// =============================================================================
// IngestContext - All shared state for ingestion pipeline
// =============================================================================

struct IngestContext {
    // Parsed tensor metadata (name -> meta)
    std::unordered_map<std::string, TensorMeta> tensors;
    
    // BPE merges from tokenizer.json [(token1, token2), ...]
    std::vector<std::pair<std::string, std::string>> bpe_merges;
    
    // Vocabulary mapping (token -> index)
    std::unordered_map<std::string, int> vocab;
    
    // Model namespace prefix (e.g., "llama4:")
    std::string model_prefix;
    
    // Processed tokens with composition data
    std::vector<TokenInfo> vocab_tokens;
    
    // Reverse lookup: token text -> index in vocab_tokens
    std::unordered_map<std::string, size_t> token_to_idx;
    
    // Model manifest with architecture info and extraction plans
    std::optional<ModelManifest> manifest;
    
    // =========================================================================
    // Config-Driven Tensor Lookup
    // =========================================================================
    
    // Find tensor by category using manifest, returns nullptr if not found
    const TensorMeta* find_tensor_by_category(TensorCategory category) const {
        if (!manifest) return nullptr;
        
        for (const auto& plan : manifest->extraction_plans) {
            if (plan.category == category) {
                auto it = tensors.find(plan.name);
                if (it != tensors.end()) return &it->second;
            }
        }
        return nullptr;
    }
    
    // Find all tensors of a given category
    std::vector<const TensorMeta*> find_tensors_by_category(TensorCategory category) const {
        std::vector<const TensorMeta*> result;
        if (!manifest) return result;
        
        for (const auto& plan : manifest->extraction_plans) {
            if (plan.category == category) {
                auto it = tensors.find(plan.name);
                if (it != tensors.end()) result.push_back(&it->second);
            }
        }
        return result;
    }
    
    // Get vocab size from config (authoritative) or fallback to parsed vocab
    size_t get_vocab_size() const {
        if (manifest && manifest->dims.vocab_size > 0) {
            return static_cast<size_t>(manifest->dims.vocab_size);
        }
        return vocab_tokens.size();
    }
    
    // Get model dimension from config
    size_t get_model_dim() const {
        if (manifest && manifest->dims.d_model > 0) {
            return static_cast<size_t>(manifest->dims.d_model);
        }
        return 0;
    }
    
    // Clear all state (for reuse)
    void clear() {
        tensors.clear();
        bpe_merges.clear();
        vocab.clear();
        model_prefix.clear();
        vocab_tokens.clear();
        token_to_idx.clear();
        manifest.reset();
    }
};

} // namespace ingest
} // namespace hypercube
