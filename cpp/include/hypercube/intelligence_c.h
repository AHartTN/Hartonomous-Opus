/**
 * Hartonomous Intelligence C API
 *
 * Complete AI inference engine - replaces llama.cpp/transformers
 * The database IS the model, queries ARE inference
 */

#ifndef HYPERCUBE_INTELLIGENCE_C_H
#define HYPERCUBE_INTELLIGENCE_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef INTELLIGENCE_C_EXPORTS
#    define INTELLIGENCE_API __declspec(dllexport)
#  else
#    define INTELLIGENCE_API __declspec(dllimport)
#  endif
#else
#  define INTELLIGENCE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Core Intelligence Handle
// ============================================================================

typedef struct HartonomousIntelligence HartonomousIntelligence;

/**
 * Initialize Hartonomous intelligence engine
 *
 * @param db_connection PostgreSQL connection string
 * @return Intelligence handle or NULL on error
 */
INTELLIGENCE_API HartonomousIntelligence* hartonomous_init(
    const char* db_connection
);

/**
 * Shutdown and free intelligence handle
 */
INTELLIGENCE_API void hartonomous_free(
    HartonomousIntelligence* intel
);

// ============================================================================
// Text Generation (Database-Native Inference)
// ============================================================================

typedef struct {
    const char* query;              // Input query/prompt
    int max_tokens;                 // Maximum response length
    float temperature;              // Sampling temperature (affects relation selection)
    int top_k;                      // Top-K semantic search results
    int rerank_top_n;               // Rerank to this many
    int max_hops;                   // Relation traversal depth
    float min_relation_rating;      // Minimum ELO rating for relations
    const char* model_filter;       // Optional: filter by model name
    int layer_filter;               // Optional: filter by layer (-1 = no filter)
} HartonomousGenerationParams;

typedef struct {
    char* generated_text;           // Generated response (caller must free)
    int num_compositions_used;      // How many DB compositions contributed
    int num_relations_traversed;    // How many relation hops
    float avg_relevance_score;      // Average semantic similarity
    char** source_ids;              // Composition IDs used (hex strings)
    int num_sources;                // Number of sources
} HartonomousGenerationResult;

/**
 * Generate text response from database intelligence
 *
 * This is the core "inference" function - equivalent to llama.cpp's generate()
 * but using database queries instead of matrix operations.
 *
 * Process:
 * 1. Embed query
 * 2. Semantic search in composition table
 * 3. Expand via relation_evidence (multi-hop)
 * 4. Rank by ELO + relevance
 * 5. Assemble coherent response from text metadata
 *
 * @param intel Intelligence handle
 * @param params Generation parameters
 * @param result Output result structure (caller must free with hartonomous_free_result)
 * @return 0 on success, -1 on error
 */
INTELLIGENCE_API int hartonomous_generate(
    HartonomousIntelligence* intel,
    const HartonomousGenerationParams* params,
    HartonomousGenerationResult* result
);

/**
 * Free generation result
 */
INTELLIGENCE_API void hartonomous_free_result(
    HartonomousGenerationResult* result
);

// ============================================================================
// Embedding Operations
// ============================================================================

typedef struct {
    float* embedding;               // Embedding vector (caller must free)
    int dimension;                  // Vector dimension
    char* composition_id;           // Created composition ID (hex string, caller must free)
} HartonomousEmbeddingResult;

/**
 * Create embedding for text
 *
 * This creates a composition in the database and returns its embedding.
 * The embedding is computed via:
 * 1. Lookup/create in composition table
 * 2. Use existing embedding if available
 * 3. Or compute from related compositions
 *
 * @param intel Intelligence handle
 * @param text Input text
 * @param model_name Model to use for embedding
 * @param result Output embedding (caller must free with hartonomous_free_embedding)
 * @return 0 on success, -1 on error
 */
INTELLIGENCE_API int hartonomous_embed(
    HartonomousIntelligence* intel,
    const char* text,
    const char* model_name,
    HartonomousEmbeddingResult* result
);

/**
 * Free embedding result
 */
INTELLIGENCE_API void hartonomous_free_embedding(
    HartonomousEmbeddingResult* result
);

// ============================================================================
// Semantic Search
// ============================================================================

typedef struct {
    char* composition_id;           // Composition ID (hex)
    char* text;                     // Text content from metadata
    float distance;                 // Semantic distance
    float rating;                   // Relation ELO rating (if applicable)
    char* model;                    // Model name
    int layer;                      // Layer number
    char* component;                // Component name
} HartonomousSearchHit;

typedef struct {
    HartonomousSearchHit* hits;     // Search results (caller must free with hartonomous_free_search)
    int num_hits;                   // Number of results
    float search_time_ms;           // Query execution time
} HartonomousSearchResult;

/**
 * Semantic search in database
 *
 * @param intel Intelligence handle
 * @param query Query text (will be embedded)
 * @param top_k Number of results
 * @param result Output results (caller must free with hartonomous_free_search)
 * @return 0 on success, -1 on error
 */
INTELLIGENCE_API int hartonomous_search(
    HartonomousIntelligence* intel,
    const char* query,
    int top_k,
    HartonomousSearchResult* result
);

/**
 * Free search results
 */
INTELLIGENCE_API void hartonomous_free_search(
    HartonomousSearchResult* result
);

// ============================================================================
// Relation Expansion
// ============================================================================

typedef struct {
    char* source_id;                // Source composition ID
    char* target_id;                // Target composition ID
    char relation_type;             // E/T/S/M
    float rating;                   // ELO rating
    float raw_weight;               // Raw similarity
    int observation_count;          // How many times observed
    char* target_text;              // Text content of target
} HartonomousRelation;

typedef struct {
    HartonomousRelation* relations; // Relations found (caller must free with hartonomous_free_relations)
    int num_relations;              // Number of relations
} HartonomousRelationsResult;

/**
 * Expand relations from a composition
 *
 * @param intel Intelligence handle
 * @param composition_id Starting composition (hex string)
 * @param max_hops Maximum traversal depth
 * @param min_rating Minimum ELO rating threshold
 * @param relation_types Filter by types (e.g., "ET" for embedding+temporal, NULL for all)
 * @param result Output relations (caller must free with hartonomous_free_relations)
 * @return 0 on success, -1 on error
 */
INTELLIGENCE_API int hartonomous_expand_relations(
    HartonomousIntelligence* intel,
    const char* composition_id,
    int max_hops,
    float min_rating,
    const char* relation_types,
    HartonomousRelationsResult* result
);

/**
 * Free relations result
 */
INTELLIGENCE_API void hartonomous_free_relations(
    HartonomousRelationsResult* result
);

// ============================================================================
// Database Statistics
// ============================================================================

typedef struct {
    int64_t total_compositions;     // Total compositions in database
    int64_t total_relations;        // Total relations
    int64_t compositions_with_embeddings;  // Compositions that have embeddings
    float avg_relation_rating;      // Average ELO rating
    char** model_names;             // Distinct model names
    int64_t* model_counts;          // Counts per model
    int num_models;                 // Number of distinct models
} HartonomousStats;

/**
 * Get database statistics
 *
 * @param intel Intelligence handle
 * @param stats Output statistics (caller must free with hartonomous_free_stats)
 * @return 0 on success, -1 on error
 */
INTELLIGENCE_API int hartonomous_get_stats(
    HartonomousIntelligence* intel,
    HartonomousStats* stats
);

/**
 * Free statistics
 */
INTELLIGENCE_API void hartonomous_free_stats(
    HartonomousStats* stats
);

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Get last error message
 *
 * @param intel Intelligence handle
 * @return Error message string (do not free)
 */
INTELLIGENCE_API const char* hartonomous_last_error(
    HartonomousIntelligence* intel
);

#ifdef __cplusplus
}
#endif

#endif // HYPERCUBE_INTELLIGENCE_C_H
