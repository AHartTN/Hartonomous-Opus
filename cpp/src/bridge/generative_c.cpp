/**
 * Generative Engine C API Implementation
 */

#include "hypercube/generative_c.h"
#include "hypercube/generative.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/types.hpp"
#include "hypercube/atom_calculator.hpp"

#include <cstring>
#include <algorithm>

using namespace hypercube::generative;
using namespace hypercube;

// Global engine
static GenerativeEngine& engine() {
    return get_engine();
}

// Helper to convert raw bytes to Blake3Hash (using generative namespace)
static hypercube::generative::Blake3Hash bytes_to_hash(const uint8_t* bytes) {
    hypercube::generative::Blake3Hash h;
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
    hypercube::generative::Blake3Hash left = bytes_to_hash(left_id);
    hypercube::generative::Blake3Hash right = bytes_to_hash(right_id);
    
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
// Plugin System Integration (TODO: Implement)
// =============================================================================

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
    // Map Unicode codepoint to 4D coordinates using Hilbert curve
    // This creates a deterministic geometric embedding of Unicode
    AtomRecord atom = AtomCalculator::compute_atom(codepoint);
    Point4D p = atom.coords;

    // Convert to uint32 representation for C API
    // Map [-1,1] float range to [0, UINT32_MAX] uint32 range
    auto float_to_uint32 = [](double val) -> uint32_t {
        // Clamp to [-1, 1]
        val = std::max(-1.0, std::min(1.0, val));
        // Map to [0, 1]
        val = (val + 1.0) / 2.0;
        // Scale to uint32
        return static_cast<uint32_t>(val * static_cast<double>(UINT32_MAX));
    };

    coords->x = float_to_uint32(p.x);
    coords->y = float_to_uint32(p.y);
    coords->z = float_to_uint32(p.z);
    coords->m = float_to_uint32(p.m);
}

GENERATIVE_C_API double geom_euclidean_distance(
    const GeomPoint4D* a,
    const GeomPoint4D* b
) {
    // Standard Euclidean distance in 4D space
    double dx = (double)a->x - (double)b->x;
    double dy = (double)a->y - (double)b->y;
    double dz = (double)a->z - (double)b->z;
    double dm = (double)a->m - (double)b->m;
    return std::sqrt(dx*dx + dy*dy + dz*dz + dm*dm);
}

GENERATIVE_C_API void geom_centroid(
    const GeomPoint4D* points,
    size_t count,
    GeomPoint4D* result
) {
    // Compute geometric centroid (arithmetic mean) of 4D points
    if (count == 0) {
        result->x = result->y = result->z = result->m = 0;
        return;
    }

    uint64_t sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    for (size_t i = 0; i < count; ++i) {
        sum_x += points[i].x;
        sum_y += points[i].y;
        sum_z += points[i].z;
        sum_m += points[i].m;
    }

    result->x = (uint32_t)(sum_x / count);
    result->y = (uint32_t)(sum_y / count);
    result->z = (uint32_t)(sum_z / count);
    result->m = (uint32_t)(sum_m / count);
}

GENERATIVE_C_API void geom_weighted_centroid(
    const GeomPoint4D* points,
    const double* weights,
    size_t count,
    GeomPoint4D* result
) {
    // Compute weighted centroid of 4D points
    // Each point is weighted by its importance/confidence
    if (count == 0) {
        result->x = result->y = result->z = result->m = 0;
        return;
    }

    double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    double total_weight = 0;

    for (size_t i = 0; i < count; ++i) {
        double w = weights[i];
        sum_x += (double)points[i].x * w;
        sum_y += (double)points[i].y * w;
        sum_z += (double)points[i].z * w;
        sum_m += (double)points[i].m * w;
        total_weight += w;
    }

    if (total_weight == 0) {
        result->x = result->y = result->z = result->m = 0;
        return;
    }

    result->x = (uint32_t)(sum_x / total_weight);
    result->y = (uint32_t)(sum_y / total_weight);
    result->z = (uint32_t)(sum_z / total_weight);
    result->m = (uint32_t)(sum_m / total_weight);
}

} // extern "C"
