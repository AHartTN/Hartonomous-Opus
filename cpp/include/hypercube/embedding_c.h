/**
 * Embedding Operations C API Bridge
 * 
 * Pure C interface to C++ embedding operations.
 * This allows PostgreSQL extensions (which must be C) to use our C++ SIMD code.
 */

#ifndef HYPERCUBE_EMBEDDING_C_H
#define HYPERCUBE_EMBEDDING_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Windows DLL export/import macros */
#if defined(_WIN32) || defined(_MSC_VER)
    #ifdef EMBEDDING_C_EXPORTS
        #define EMBEDDING_C_API __declspec(dllexport)
    #else
        #define EMBEDDING_C_API __declspec(dllimport)
    #endif
#else
    #define EMBEDDING_C_API __attribute__((visibility("default")))
#endif

/* ==========================================================================
 * Core Vector Operations
 * ========================================================================== */

/**
 * Compute cosine similarity between two embedding vectors.
 * Uses AVX2 SIMD when available.
 * 
 * @param a     First embedding vector
 * @param b     Second embedding vector
 * @param n     Dimension of vectors
 * @return      Cosine similarity in range [-1, 1]
 */
EMBEDDING_C_API double embedding_c_cosine_similarity(
    const float* a,
    const float* b,
    size_t n
);

/**
 * Compute L2 (Euclidean) distance between two embedding vectors.
 * Uses AVX2 SIMD when available.
 * 
 * @param a     First embedding vector
 * @param b     Second embedding vector
 * @param n     Dimension of vectors
 * @return      L2 distance (always >= 0)
 */
EMBEDDING_C_API double embedding_c_l2_distance(
    const float* a,
    const float* b,
    size_t n
);

/**
 * Compute vector sum: result = a + b
 */
EMBEDDING_C_API void embedding_c_vector_add(
    const float* a,
    const float* b,
    float* result,
    size_t n
);

/**
 * Compute vector difference: result = a - b
 */
EMBEDDING_C_API void embedding_c_vector_sub(
    const float* a,
    const float* b,
    float* result,
    size_t n
);

/**
 * Compute analogy target: result = c + (a - b)
 * Used for vector analogies like "king - man + woman = ?"
 */
EMBEDDING_C_API void embedding_c_analogy_target(
    const float* a,
    const float* b,
    const float* c,
    float* result,
    size_t n
);

/* ==========================================================================
 * Similarity Result Structure
 * ========================================================================== */

typedef struct {
    size_t index;       /* Index in the embedding array */
    double similarity;  /* Cosine similarity score */
} EmbeddingSimilarityResult;

/**
 * Find top-k most similar embeddings to a query.
 * Uses AVX2 SIMD for fast batch comparison.
 * 
 * @param query         Query embedding vector
 * @param embeddings    Flat array of embeddings (n_embeddings * dim floats)
 * @param n_embeddings  Number of embeddings in array
 * @param dim           Dimension of each embedding
 * @param k             Number of results to return
 * @param results       Pre-allocated array of k results (caller must free)
 * @return              Number of results written (may be < k)
 */
EMBEDDING_C_API size_t embedding_c_find_top_k(
    const float* query,
    const float* embeddings,
    size_t n_embeddings,
    size_t dim,
    size_t k,
    EmbeddingSimilarityResult* results
);

/* ==========================================================================
 * Cache Management for Fast Batch Operations
 * ========================================================================== */

/**
 * Initialize the embedding cache.
 * Must be called before using cache functions.
 * 
 * @return 0 on success, -1 on error
 */
EMBEDDING_C_API int embedding_c_cache_init(void);

/**
 * Clear and free the embedding cache.
 */
EMBEDDING_C_API void embedding_c_cache_clear(void);

/**
 * Add an embedding to the cache.
 * 
 * @param id        Entity ID (binary, copied)
 * @param id_len    Length of ID
 * @param label     Label string (copied)
 * @param embedding Embedding vector (copied)
 * @param dim       Dimension of embedding
 * @return          Index in cache, or -1 on error
 */
EMBEDDING_C_API int64_t embedding_c_cache_add(
    const uint8_t* id,
    size_t id_len,
    const char* label,
    const float* embedding,
    size_t dim
);

/**
 * Get cache statistics.
 */
EMBEDDING_C_API size_t embedding_c_cache_count(void);
EMBEDDING_C_API size_t embedding_c_cache_dim(void);

/**
 * Find index of label in cache.
 * 
 * @param label     Label to find
 * @return          Index if found, -1 if not found
 */
EMBEDDING_C_API int64_t embedding_c_cache_find_label(const char* label);

/**
 * Get embedding from cache by index.
 * 
 * @param index     Index in cache
 * @return          Pointer to embedding (do not free), or NULL if invalid
 */
EMBEDDING_C_API const float* embedding_c_cache_get_embedding(size_t index);

/**
 * Get label from cache by index.
 * 
 * @param index     Index in cache
 * @return          Label string (do not free), or NULL if invalid
 */
EMBEDDING_C_API const char* embedding_c_cache_get_label(size_t index);

/**
 * Find top-k similar embeddings in cache.
 * 
 * @param query_index   Index of query embedding in cache
 * @param k             Number of results
 * @param results       Pre-allocated array of k results
 * @return              Number of results written
 */
EMBEDDING_C_API size_t embedding_c_cache_similar(
    size_t query_index,
    size_t k,
    EmbeddingSimilarityResult* results
);

/**
 * Find top-k analogy results in cache.
 * Computes: c + (a - b) and finds nearest neighbors.
 * 
 * @param idx_a         Index of positive word (e.g., "king")
 * @param idx_b         Index of negative word (e.g., "man")
 * @param idx_c         Index of query word (e.g., "woman")
 * @param k             Number of results
 * @param results       Pre-allocated array of k results
 * @return              Number of results written
 */
EMBEDDING_C_API size_t embedding_c_cache_analogy(
    size_t idx_a,
    size_t idx_b,
    size_t idx_c,
    size_t k,
    EmbeddingSimilarityResult* results
);

/* ==========================================================================
 * Backend Information
 * ========================================================================== */

/**
 * Get the name of the active SIMD implementation.
 * @return String like "AVX-512", "AVX2", "SSE4.2", or "Scalar"
 */
EMBEDDING_C_API const char* embedding_c_simd_level(void);

/**
 * Get the SIMD register width in floats.
 * @return 16 for AVX-512, 8 for AVX2, 4 for SSE, 1 for Scalar
 */
EMBEDDING_C_API int embedding_c_simd_width(void);

#ifdef __cplusplus
}
#endif

#endif /* HYPERCUBE_EMBEDDING_C_H */
