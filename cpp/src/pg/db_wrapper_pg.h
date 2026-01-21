/**
 * db_wrapper_pg.h - Header for Unified PostgreSQL Database Wrapper
 *
 * Exposes the API functions for loading in-memory data structures from the database.
 */

#ifndef DB_WRAPPER_PG_H
#define DB_WRAPPER_PG_H

#include <stdint.h>

#define HASH_SIZE 32

/* ============================================================================
 * Data Structures
 * ============================================================================ */

typedef struct SemanticEdge {
    uint8_t source[HASH_SIZE];
    uint8_t target[HASH_SIZE];
    double  weight;
} SemanticEdge;

typedef struct EdgeCollection {
    SemanticEdge *edges;
    int count;
    int capacity;
} EdgeCollection;

typedef struct CentroidEntry {
    uint8_t id[HASH_SIZE];
    char *label;
    float embedding[4];  // 4D centroid: x,y,z,m
} CentroidEntry;

typedef struct CentroidCollection {
    CentroidEntry *entries;
    int count;
    int capacity;
} CentroidCollection;

typedef struct VocabEntry {
    uint8_t id[HASH_SIZE];
    char *label;
    int depth;
    double frequency;
    double hilbert;
    double centroid_x, centroid_y, centroid_z, centroid_m;
} VocabEntry;

typedef struct VocabCollection {
    VocabEntry *entries;
    int count;
    int capacity;
} VocabCollection;

typedef struct RelationEntry {
    uint8_t source_id[HASH_SIZE];
    uint8_t target_id[HASH_SIZE];
    double weight;
} RelationEntry;

typedef struct RelationCollection {
    RelationEntry *entries;
    int count;
    int capacity;
    const char *relation_type;  // "S" for bigrams, "A"/"W" for attention
} RelationCollection;

/* ============================================================================
 * API Functions
 * ============================================================================ */

/**
 * load_edges() - Load semantic edges from database
 * Returns: EdgeCollection with all semantic edges (bidirectional bigrams)
 * Memory: Allocated in CurrentMemoryContext, caller responsible for cleanup
 */
EdgeCollection *load_edges();

/**
 * load_centroids() - Load 4D centroids from composition table
 * Returns: CentroidCollection with centroids and labels
 * Memory: Allocated in CurrentMemoryContext, caller responsible for cleanup
 */
CentroidCollection *load_centroids();

/**
 * load_vocab() - Load vocabulary from composition table
 * Returns: VocabCollection with vocabulary entries and their metadata
 * Memory: Allocated in CurrentMemoryContext, caller responsible for cleanup
 */
VocabCollection *load_vocab();

/**
 * load_relations() - Load relations from relation table
 * relation_type: "S" for similarity (bigrams), "A" or "W" for attention
 * Returns: RelationCollection with source/target/weight triples
 * Memory: Allocated in CurrentMemoryContext, caller responsible for cleanup
 */
RelationCollection *load_relations(const char *relation_type);

/* ============================================================================
 * Cleanup Functions
 * ============================================================================ */

void free_edge_collection(EdgeCollection *collection);
void free_centroid_collection(CentroidCollection *collection);
void free_vocab_collection(VocabCollection *collection);
void free_relation_collection(RelationCollection *collection);

#endif /* DB_WRAPPER_PG_H */