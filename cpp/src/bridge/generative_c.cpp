/**
 * Generative Engine C API Implementation
 */

#include "hypercube/generative_c.h"
#include "hypercube/generative.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/types.hpp"

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

GENERATIVE_C_API int gen_vocab_set_centroid(
    size_t idx,
    double x,
    double y,
    double z,
    double m
) {
    if (idx >= engine().vocab.entries.size()) {
        return -1;
    }
    
    engine().vocab.set_centroid(idx, x, y, z, m);
    return 0;
}

GENERATIVE_C_API void gen_vocab_finalize(void) {
    // No longer needed for embedding computation, but kept for API compat
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

GENERATIVE_C_API double gen_bigram_get(
    const uint8_t* left_id,
    const uint8_t* right_id
) {
    return engine().bigrams.get(bytes_to_hash(left_id), bytes_to_hash(right_id));
}

GENERATIVE_C_API int gen_bigram_debug_find(
    const uint8_t* left_id,
    const uint8_t* right_id,
    double* out_score
) {
    Blake3Hash left = bytes_to_hash(left_id);
    Blake3Hash right = bytes_to_hash(right_id);
    
    // Try to find exact match
    BigramKey key{left, right};
    auto it = engine().bigrams.pmi_scores.find(key);
    if (it != engine().bigrams.pmi_scores.end()) {
        *out_score = it->second;
        return 1;  // Found
    }
    
    // Count how many entries have this left key
    int count = 0;
    for (const auto& [k, v] : engine().bigrams.pmi_scores) {
        if (k.left == left) {
            count++;
            if (count <= 5) {
                // Return first few for debug
            }
        }
    }
    
    *out_score = 0.0;
    return -count;  // Not found, return negative count of left matches
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
    double w_centroid,
    double w_pmi,
    double w_attn,
    double w_global
) {
    engine().config.w_centroid = w_centroid;
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
        results[i].score_centroid = 0;
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
        results[i].score_centroid = scored[i].score_centroid;
        results[i].score_pmi = scored[i].score_pmi;
        results[i].score_attn = scored[i].score_attn;
        results[i].score_global = scored[i].score_global;
        results[i].score_total = scored[i].score_total;
    }

    return n;
}

// =============================================================================
// Geometric Operations (4D Coordinate System)
// =============================================================================

GENERATIVE_C_API void geom_map_codepoint(
    uint32_t codepoint,
    GeomPoint4D* coords
) {
    auto point = hypercube::CoordinateMapper::map_codepoint(codepoint);
    coords->x = point.x;
    coords->y = point.y;
    coords->z = point.z;
    coords->m = point.m;
}

GENERATIVE_C_API double geom_euclidean_distance(
    const GeomPoint4D* a,
    const GeomPoint4D* b
) {
    hypercube::Point4D pa{a->x, a->y, a->z, a->m};
    hypercube::Point4D pb{b->x, b->y, b->z, b->m};
    return hypercube::CoordinateMapper::euclidean_distance(pa, pb);
}

GENERATIVE_C_API void geom_centroid(
    const GeomPoint4D* points,
    size_t count,
    GeomPoint4D* result
) {
    std::vector<hypercube::Point4D> pts;
    pts.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        pts.push_back(hypercube::Point4D{points[i].x, points[i].y, points[i].z, points[i].m});
    }
    auto centroid = hypercube::CoordinateMapper::centroid(pts);
    result->x = centroid.x;
    result->y = centroid.y;
    result->z = centroid.z;
    result->m = centroid.m;
}

GENERATIVE_C_API void geom_weighted_centroid(
    const GeomPoint4D* points,
    const double* weights,
    size_t count,
    GeomPoint4D* result
) {
    std::vector<hypercube::Point4D> pts;
    std::vector<double> wts;
    pts.reserve(count);
    wts.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        pts.push_back(hypercube::Point4D{points[i].x, points[i].y, points[i].z, points[i].m});
        wts.push_back(weights[i]);
    }
    auto centroid = hypercube::CoordinateMapper::weighted_centroid(pts, wts);
    result->x = centroid.x;
    result->y = centroid.y;
    result->z = centroid.z;
    result->m = centroid.m;
}

} // extern "C"
