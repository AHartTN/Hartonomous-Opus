/**
 * Generative Engine C API Implementation
 */

#include "hypercube/generative_c.h"
#include "hypercube/generative.hpp"

#include <cstring>
#include <algorithm>

using namespace hypercube::generative;

// Global engine
static GenerativeEngine& engine() {
    return get_engine();
}

// Helper to convert raw bytes to Blake3Hash
static Blake3Hash bytes_to_hash(const uint8_t* bytes) {
    Blake3Hash h;
    std::memcpy(h.data(), bytes, 32);
    return h;
}

extern "C" {

// =============================================================================
// Vocabulary Cache Management
// =============================================================================

GENERATIVE_C_API void gen_vocab_clear(void) {
    engine().vocab.clear();
}

GENERATIVE_C_API int64_t gen_vocab_add(
    const uint8_t* id,
    const char* label,
    int depth,
    double frequency,
    double hilbert
) {
    VocabEntry entry;
    entry.id = bytes_to_hash(id);
    entry.label = label ? label : "";
    entry.depth = depth;
    entry.frequency = frequency;
    entry.hilbert_index = hilbert;
    
    engine().vocab.add_entry(entry);
    return static_cast<int64_t>(engine().vocab.entries.size() - 1);
}

GENERATIVE_C_API int gen_vocab_add_embedding(
    size_t idx,
    const char* model,
    const float* embedding,
    size_t dim
) {
    if (idx >= engine().vocab.entries.size()) {
        return -1;
    }
    
    std::vector<float> emb(embedding, embedding + dim);
    engine().vocab.add_embedding(idx, model ? model : "", emb);
    return 0;
}

GENERATIVE_C_API void gen_vocab_finalize(void) {
    engine().vocab.compute_average_embeddings();
    engine().vocab.build_flat_embeddings();
}

GENERATIVE_C_API size_t gen_vocab_count(void) {
    return engine().vocab.entries.size();
}

GENERATIVE_C_API int64_t gen_vocab_find_label(const char* label) {
    return engine().vocab.find_label(label ? label : "");
}

GENERATIVE_C_API const char* gen_vocab_get_label(size_t idx) {
    auto* entry = engine().vocab.get_entry(idx);
    return entry ? entry->label.c_str() : nullptr;
}

// =============================================================================
// Bigram (PMI) Cache Management
// =============================================================================

GENERATIVE_C_API void gen_bigram_clear(void) {
    engine().bigrams.clear();
}

GENERATIVE_C_API void gen_bigram_add(
    const uint8_t* left_id,
    const uint8_t* right_id,
    double score
) {
    engine().bigrams.add(bytes_to_hash(left_id), bytes_to_hash(right_id), score);
}

GENERATIVE_C_API size_t gen_bigram_count(void) {
    return engine().bigrams.pmi_scores.size();
}

// =============================================================================
// Attention Cache Management
// =============================================================================

GENERATIVE_C_API void gen_attention_clear(void) {
    engine().attention.clear();
}

GENERATIVE_C_API void gen_attention_add(
    const uint8_t* source_id,
    const uint8_t* target_id,
    double weight
) {
    engine().attention.add(bytes_to_hash(source_id), bytes_to_hash(target_id), weight);
}

GENERATIVE_C_API size_t gen_attention_count(void) {
    size_t count = 0;
    for (const auto& [src, targets] : engine().attention.edges) {
        count += targets.size();
    }
    return count;
}

// =============================================================================
// Configuration
// =============================================================================

GENERATIVE_C_API void gen_config_set_weights(
    double w_shape,
    double w_pmi,
    double w_attn,
    double w_global
) {
    engine().config.w_shape = w_shape;
    engine().config.w_pmi = w_pmi;
    engine().config.w_attn = w_attn;
    engine().config.w_global = w_global;
}

GENERATIVE_C_API void gen_config_set_policy(
    int greedy,
    double temperature
) {
    engine().config.greedy = (greedy != 0);
    engine().config.temperature = temperature;
}

GENERATIVE_C_API void gen_config_set_filter(
    size_t max_candidates,
    double hilbert_range
) {
    engine().config.max_candidates = max_candidates;
    engine().config.hilbert_range = hilbert_range;
}

// =============================================================================
// Similarity Search
// =============================================================================

GENERATIVE_C_API size_t gen_find_similar(
    const char* label,
    size_t k,
    GenSimilarResult* results
) {
    auto scored = engine().find_similar(label ? label : "", k);
    
    size_t n = std::min(scored.size(), k);
    for (size_t i = 0; i < n; ++i) {
        results[i].index = scored[i].index;
        results[i].similarity = scored[i].score_total;
    }
    
    return n;
}

// =============================================================================
// Generation
// =============================================================================

GENERATIVE_C_API size_t gen_generate(
    const char* start_label,
    size_t max_tokens,
    GenTokenResult* results
) {
    auto tokens = engine().generate(start_label ? start_label : "", max_tokens);
    
    size_t n = std::min(tokens.size(), max_tokens);
    for (size_t i = 0; i < n; ++i) {
        int64_t idx = engine().vocab.find_label(tokens[i]);
        results[i].token_index = (idx >= 0) ? static_cast<size_t>(idx) : 0;
        // Score components not available from generate() directly
        results[i].score_shape = 0;
        results[i].score_pmi = 0;
        results[i].score_attn = 0;
        results[i].score_global = 0;
        results[i].score_total = 0;
    }
    
    return n;
}

GENERATIVE_C_API size_t gen_score_candidates(
    const char* current_label,
    size_t top_k,
    GenTokenResult* results
) {
    int64_t idx = engine().vocab.find_label(current_label ? current_label : "");
    if (idx < 0) return 0;
    
    TokenState current = engine().make_token_state(idx);
    
    // Get all candidates and score them
    auto candidate_indices = engine().get_all_vocab_candidates();
    
    std::vector<ScoredCandidate> scored;
    scored.reserve(candidate_indices.size());
    
    for (size_t cand_idx : candidate_indices) {
        if (cand_idx == (size_t)idx) continue;  // Skip self
        scored.push_back(engine().score_candidate(current, cand_idx));
    }
    
    // Sort by total score
    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.score_total > b.score_total; });
    
    size_t n = std::min(scored.size(), top_k);
    for (size_t i = 0; i < n; ++i) {
        results[i].token_index = scored[i].index;
        results[i].score_shape = scored[i].score_shape;
        results[i].score_pmi = scored[i].score_pmi;
        results[i].score_attn = scored[i].score_attn;
        results[i].score_global = scored[i].score_global;
        results[i].score_total = scored[i].score_total;
    }
    
    return n;
}

} // extern "C"
