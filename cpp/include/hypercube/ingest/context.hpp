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

#include "hypercube/types.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/ingest/safetensor.hpp"

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
    
    // Clear all state (for reuse)
    void clear() {
        tensors.clear();
        bpe_merges.clear();
        vocab.clear();
        model_prefix.clear();
        vocab_tokens.clear();
        token_to_idx.clear();
    }
};

} // namespace ingest
} // namespace hypercube
