/**
 * Generative Engine C API Bridge
 * 
 * Pure C interface to the C++ generative engine.
 */

#ifndef HYPERCUBE_GENERATIVE_C_H
#define HYPERCUBE_GENERATIVE_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Windows DLL export/import macros */
#if defined(_WIN32) || defined(_MSC_VER)
    #ifdef GENERATIVE_C_EXPORTS
        #define GENERATIVE_C_API __declspec(dllexport)
    #else
        #define GENERATIVE_C_API __declspec(dllimport)
    #endif
#else
    #define GENERATIVE_C_API __attribute__((visibility("default")))
#endif

/* ==========================================================================
 * Geometric Types
 * ========================================================================== */

typedef struct {
    uint32_t x, y, z, m;
} GeomPoint4D;

/* ==========================================================================
 * Vocabulary Cache Management
 * ========================================================================== */

/**
 * Clear the vocabulary cache.
 */
GENERATIVE_C_API void gen_vocab_clear(void);

/**
 * Add a vocabulary entry.
 * 
 * @param id        Entity ID (32 bytes)
 * @param label     Token label string
 * @param depth     Composition depth
 * @param frequency Usage frequency
 * @param hilbert   Hilbert index [0, 1]
 * @return          Index in cache, or -1 on error
 */
GENERATIVE_C_API int64_t gen_vocab_add(
    const uint8_t* id,
    const char* label,
    int depth,
    double frequency,
    double hilbert
);

/**
 * Add a 4D centroid for a vocabulary entry.
 * 
 * @param idx       Entry index
 * @param x         X coordinate
 * @param y         Y coordinate
 * @param z         Z coordinate
 * @param m         M coordinate
 * @return          0 on success, -1 on error
 */
GENERATIVE_C_API int gen_vocab_set_centroid(
    size_t idx,
    double x,
    double y,
    double z,
    double m
);

/**
 * Finalize vocabulary (no longer needed for embeddings, but kept for API compat).
 */
GENERATIVE_C_API void gen_vocab_finalize(void);

/**
 * Get vocabulary count.
 */
GENERATIVE_C_API size_t gen_vocab_count(void);

/**
 * Find entry by label.
 * @return Index or -1 if not found
 */
GENERATIVE_C_API int64_t gen_vocab_find_label(const char* label);

/**
 * Get label for entry.
 * @return Label string (do not free) or NULL
 */
GENERATIVE_C_API const char* gen_vocab_get_label(size_t idx);

/* ==========================================================================
 * Bigram (PMI) Cache Management
 * ========================================================================== */

/**
 * Clear the bigram cache.
 */
GENERATIVE_C_API void gen_bigram_clear(void);

/**
 * Add a bigram PMI score.
 * 
 * @param left_id   Left token ID (32 bytes)
 * @param right_id  Right token ID (32 bytes)
 * @param score     PMI score
 */
GENERATIVE_C_API void gen_bigram_add(
    const uint8_t* left_id,
    const uint8_t* right_id,
    double score
);

/**
 * Get bigram count.
 */
GENERATIVE_C_API size_t gen_bigram_count(void);

/**
 * Lookup a bigram PMI score.
 * 
 * @param left_id   Left token ID (32 bytes)
 * @param right_id  Right token ID (32 bytes)
 * @return          PMI score (0.0 if not found)
 */
GENERATIVE_C_API double gen_bigram_get(
    const uint8_t* left_id,
    const uint8_t* right_id
);

/**
 * Debug function to find a bigram and count left-key matches.
 * 
 * @param left_id   Left token ID (32 bytes)
 * @param right_id  Right token ID (32 bytes)
 * @param out_score Output: the score if found
 * @return          1 if found, or negative count of left-key matches if not
 */
GENERATIVE_C_API int gen_bigram_debug_find(
    const uint8_t* left_id,
    const uint8_t* right_id,
    double* out_score
);

/* ==========================================================================
 * Attention Cache Management
 * ========================================================================== */

/**
 * Clear the attention cache.
 */
GENERATIVE_C_API void gen_attention_clear(void);

/**
 * Add an attention edge.
 * 
 * @param source_id Source token ID (32 bytes)
 * @param target_id Target token ID (32 bytes)
 * @param weight    Attention weight
 */
GENERATIVE_C_API void gen_attention_add(
    const uint8_t* source_id,
    const uint8_t* target_id,
    double weight
);

/**
 * Get attention edge count.
 */
GENERATIVE_C_API size_t gen_attention_count(void);

/* ==========================================================================
 * Configuration
 * ========================================================================== */

/**
 * Set scoring weights.
 */
GENERATIVE_C_API void gen_config_set_weights(
    double w_centroid,
    double w_pmi,
    double w_attn,
    double w_global
);

/**
 * Set selection policy.
 * 
 * @param greedy      1 for greedy, 0 for stochastic
 * @param temperature Temperature for stochastic sampling
 */
GENERATIVE_C_API void gen_config_set_policy(
    int greedy,
    double temperature
);

/**
 * Set candidate filtering.
 * 
 * @param max_candidates Maximum candidates after Hilbert filter
 * @param hilbert_range  Fraction of Hilbert space to search
 */
GENERATIVE_C_API void gen_config_set_filter(
    size_t max_candidates,
    double hilbert_range
);

/* ==========================================================================
 * Similarity Search (uses vocab cache only)
 * ========================================================================== */

typedef struct {
    size_t index;
    double similarity;
} GenSimilarResult;

/**
 * Find similar tokens by shape similarity.
 * 
 * @param label     Query label
 * @param k         Number of results
 * @param results   Pre-allocated result array
 * @return          Number of results written
 */
GENERATIVE_C_API size_t gen_find_similar(
    const char* label,
    size_t k,
    GenSimilarResult* results
);

/* ==========================================================================
 * Generation
 * ========================================================================== */

typedef struct {
    size_t token_index;
    double score_centroid;
    double score_pmi;
    double score_attn;
    double score_global;
    double score_total;
} GenTokenResult;

/**
 * Generate tokens starting from a label.
 * 
 * @param start_label   Starting token label
 * @param max_tokens    Maximum tokens to generate
 * @param results       Pre-allocated result array (size max_tokens)
 * @return              Number of tokens generated
 */
GENERATIVE_C_API size_t gen_generate(
    const char* start_label,
    size_t max_tokens,
    GenTokenResult* results
);

/**
 * Score candidates for next token (for debugging/analysis).
 * 
 * @param current_label Current token label
 * @param top_k         Number of top candidates to return
 * @param results       Pre-allocated result array
 * @return              Number of results written
 */
GENERATIVE_C_API size_t gen_score_candidates(
    const char* current_label,
    size_t top_k,
    GenTokenResult* results
);

/* ==========================================================================
 * Geometric Operations (4D Coordinate System)
 * ========================================================================== */

/**
 * Map a Unicode codepoint to 4D coordinates.
 *
 * @param codepoint Unicode codepoint (0 to 0x10FFFF)
 * @param coords    Output: 4D point on 3-sphere surface
 */
GENERATIVE_C_API void geom_map_codepoint(
    uint32_t codepoint,
    GeomPoint4D* coords
);

/**
 * Calculate Euclidean distance between two 4D points.
 *
 * @param a First point
 * @param b Second point
 * @return Euclidean distance (0.0 to 2.0 in normalized space)
 */
GENERATIVE_C_API double geom_euclidean_distance(
    const GeomPoint4D* a,
    const GeomPoint4D* b
);

/**
 * Calculate centroid of multiple 4D points.
 *
 * @param points Input points array
 * @param count  Number of points
 * @param result Output: centroid point
 */
GENERATIVE_C_API void geom_centroid(
    const GeomPoint4D* points,
    size_t count,
    GeomPoint4D* result
);

/**
 * Calculate weighted centroid of multiple 4D points.
 *
 * @param points  Input points array
 * @param weights Weights for each point
 * @param count   Number of points
 * @param result  Output: weighted centroid point
 */
GENERATIVE_C_API void geom_weighted_centroid(
    const GeomPoint4D* points,
    const double* weights,
    size_t count,
    GeomPoint4D* result
);

#ifdef __cplusplus
}
#endif

#endif /* HYPERCUBE_GENERATIVE_C_H */
