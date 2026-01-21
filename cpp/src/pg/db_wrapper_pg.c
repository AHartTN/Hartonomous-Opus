/**
 * db_wrapper_pg.c - Unified PostgreSQL Database Wrapper
 *
 * Centralizes all database operations for Hypercube PostgreSQL extensions.
 * This module is the ONLY component that links to PostgreSQL libraries (SPI, pg_types, etc.)
 * and handles all PostgreSQL-specific code: SPI connections, memory context, data types.
 *
 * Provides clean API functions that return in-memory data structures for computation.
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include <string.h>
#include <stdlib.h>

#include "pg_utils.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

/* ============================================================================
 * Data Structures for In-Memory Results
 * ============================================================================ */

#define MAX_EDGES 100000

typedef struct SemanticEdge {
    uint8   source[HASH_SIZE];
    uint8   target[HASH_SIZE];
    double  weight;
} SemanticEdge;

typedef struct EdgeCollection {
    SemanticEdge *edges;
    int count;
    int capacity;
} EdgeCollection;

typedef struct CentroidEntry {
    uint8 id[HASH_SIZE];
    char *label;
    float embedding[4];  // 4D centroid: x,y,z,m
} CentroidEntry;

typedef struct CentroidCollection {
    CentroidEntry *entries;
    int count;
    int capacity;
} CentroidCollection;

typedef struct VocabEntry {
    uint8 id[HASH_SIZE];
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
    uint8 source_id[HASH_SIZE];
    uint8 target_id[HASH_SIZE];
    double weight;
} RelationEntry;

typedef struct RelationCollection {
    RelationEntry *entries;
    int count;
    int capacity;
    const char *relation_type;  // "S" for bigrams, "A"/"W" for attention
} RelationCollection;

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

static void log_db_operation(const char *operation, int result_count, const char *details)
{
    ereport(NOTICE, (errmsg("DB Wrapper: %s completed - loaded %d items%s%s",
                           operation, result_count,
                           details ? " - " : "", details ? details : "")));
}

static void log_db_error(const char *operation, const char *error_msg)
{
    ereport(ERROR, (errmsg("DB Wrapper %s failed: %s", operation, error_msg)));
}

static int safe_spi_connect()
{
    int ret = SPI_connect();
    if (ret != SPI_OK_CONNECT) {
        log_db_error("SPI_connect", "Failed to establish SPI connection");
    }
    return ret;
}

static void safe_spi_finish()
{
    SPI_finish();
}

/* ============================================================================
 * Public API Functions
 * ============================================================================ */

/**
 * load_edges() - Load semantic edges from database
 * Returns: EdgeCollection with all semantic edges (bidirectional bigrams)
 * Memory: Allocated in CurrentMemoryContext, caller responsible for cleanup
 */
EdgeCollection *load_edges()
{
    EdgeCollection *collection = (EdgeCollection *)palloc(sizeof(EdgeCollection));
    collection->capacity = MAX_EDGES;
    collection->edges = (SemanticEdge *)palloc(sizeof(SemanticEdge) * collection->capacity);
    collection->count = 0;

    if (safe_spi_connect() != SPI_OK_CONNECT) {
        return collection;
    }

    /* Query using composition_child table to get bigram children */
    const char *query =
        "SELECT cc1.child_id as child1, cc2.child_id as child2, 1.0::float8 as weight "
        "FROM composition c "
        "JOIN composition_child cc1 ON cc1.composition_id = c.id AND cc1.ordinal = 1 AND cc1.child_type = 'A' "
        "JOIN composition_child cc2 ON cc2.composition_id = c.id AND cc2.ordinal = 2 AND cc2.child_type = 'A' "
        "WHERE c.depth = 1 AND c.atom_count = 2 "
        "LIMIT 50000";

    int ret = SPI_execute(query, true, 0);
    if (ret != SPI_OK_SELECT || SPI_processed == 0) {
        safe_spi_finish();
        log_db_operation("load_edges", 0, "No edges found");
        return collection;
    }

    /* Process each row - create bidirectional edges */
    for (uint64 i = 0; i < SPI_processed && collection->count + 2 < collection->capacity; i++) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool isnull1, isnull2, isnull_w;

        Datum d1 = SPI_getbinval(tuple, tupdesc, 1, &isnull1);
        Datum d2 = SPI_getbinval(tuple, tupdesc, 2, &isnull2);
        Datum dw = SPI_getbinval(tuple, tupdesc, 3, &isnull_w);

        if (isnull1 || isnull2) continue;

        /* Use PG_DETOAST_DATUM_COPY to safely handle TOASTed data */
        bytea *b1 = (bytea *)PG_DETOAST_DATUM_COPY(d1);
        bytea *b2 = (bytea *)PG_DETOAST_DATUM_COPY(d2);

        /* Validate bytea size */
        int len1 = VARSIZE(b1) - VARHDRSZ;
        int len2 = VARSIZE(b2) - VARHDRSZ;
        if (len1 < HASH_SIZE || len2 < HASH_SIZE) {
            pfree(b1);
            pfree(b2);
            continue;
        }

        double weight = isnull_w ? 1.0 : DatumGetFloat8(dw);

        /* Add bidirectional edges */
        memcpy(collection->edges[collection->count].source, VARDATA(b1), HASH_SIZE);
        memcpy(collection->edges[collection->count].target, VARDATA(b2), HASH_SIZE);
        collection->edges[collection->count].weight = weight;
        collection->count++;

        memcpy(collection->edges[collection->count].source, VARDATA(b2), HASH_SIZE);
        memcpy(collection->edges[collection->count].target, VARDATA(b1), HASH_SIZE);
        collection->edges[collection->count].weight = weight;
        collection->count++;

        pfree(b1);
        pfree(b2);
    }

    safe_spi_finish();
    log_db_operation("load_edges", collection->count, NULL);
    return collection;
}

/**
 * load_centroids() - Load 4D centroids from composition table
 * Returns: CentroidCollection with centroids and labels
 * Memory: Allocated in CurrentMemoryContext, caller responsible for cleanup
 */
CentroidCollection *load_centroids()
{
    CentroidCollection *collection = (CentroidCollection *)palloc(sizeof(CentroidCollection));
    collection->capacity = 10000;
    collection->entries = (CentroidEntry *)palloc(sizeof(CentroidEntry) * collection->capacity);
    collection->count = 0;

    if (safe_spi_connect() != SPI_OK_CONNECT) {
        return collection;
    }

    /* Query to extract 4D centroids as float4[] */
    const char *query =
        "SELECT c.id, c.label, "
        "       ARRAY[ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid)]::float4[] AS emb "
        "FROM composition c "
        "WHERE c.centroid IS NOT NULL "
        "  AND c.label IS NOT NULL "
        "  AND c.label NOT LIKE '[%' "
        "ORDER BY c.label";

    int ret = SPI_execute(query, true, 0);
    if (ret != SPI_OK_SELECT) {
        safe_spi_finish();
        log_db_error("load_centroids", "SPI query failed");
        return collection;
    }

    for (uint64 i = 0; i < SPI_processed && collection->count < collection->capacity; i++) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool isnull;

        /* Get ID (bytea) */
        Datum id_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull) continue;
        bytea *id_bytea = DatumGetByteaP(id_datum);
        uint8_t *id_data = (uint8_t *)VARDATA(id_bytea);
        size_t id_len = VARSIZE(id_bytea) - VARHDRSZ;

        /* Get label */
        Datum label_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        if (isnull) continue;
        char *label = TextDatumGetCString(label_datum);

        /* Get 4D centroid array */
        Datum emb_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull);
        if (isnull) {
            pfree(label);
            continue;
        }
        ArrayType *emb_arr = DatumGetArrayTypeP(emb_datum);
        int dim = ArrayGetNItems(ARR_NDIM(emb_arr), ARR_DIMS(emb_arr));
        float *emb = (float *)ARR_DATA_PTR(emb_arr);

        if (dim >= 4) {
            CentroidEntry *entry = &collection->entries[collection->count];
            memcpy(entry->id, id_data, HASH_SIZE);
            entry->label = MemoryContextStrdup(CurrentMemoryContext, label);
            entry->embedding[0] = emb[0];
            entry->embedding[1] = emb[1];
            entry->embedding[2] = emb[2];
            entry->embedding[3] = emb[3];
            collection->count++;
        }

        pfree(label);
    }

    safe_spi_finish();
    log_db_operation("load_centroids", collection->count, NULL);
    return collection;
}

/**
 * load_vocab() - Load vocabulary from composition table
 * Returns: VocabCollection with vocabulary entries and their metadata
 * Memory: Allocated in CurrentMemoryContext, caller responsible for cleanup
 */
VocabCollection *load_vocab()
{
    VocabCollection *collection = (VocabCollection *)palloc(sizeof(VocabCollection));
    collection->capacity = 50000;
    collection->entries = (VocabEntry *)palloc(sizeof(VocabEntry) * collection->capacity);
    collection->count = 0;

    if (safe_spi_connect() != SPI_OK_CONNECT) {
        return collection;
    }

    /* Load all compositions with 4D centroids */
    const char *query =
        "SELECT c.id, c.label, c.depth, "
        "       COALESCE((SELECT COUNT(*) FROM composition_child cc WHERE cc.child_id = c.id), 0) AS freq, "
        "       ST_X(c.centroid) AS cx, ST_Y(c.centroid) AS cy, "
        "       ST_Z(c.centroid) AS cz, ST_M(c.centroid) AS cm, "
        "       (c.hilbert_lo::float8 / 9223372036854775807.0) AS hilbert "
        "FROM composition c "
        "WHERE c.label IS NOT NULL "
        "  AND c.label NOT LIKE '[%' "
        "  AND c.centroid IS NOT NULL "
        "ORDER BY c.label";

    int ret = SPI_execute(query, true, 0);
    if (ret != SPI_OK_SELECT) {
        safe_spi_finish();
        log_db_error("load_vocab", "SPI query failed");
        return collection;
    }

    for (uint64 i = 0; i < SPI_processed && collection->count < collection->capacity; i++) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool isnull;

        /* Get ID (bytea) */
        Datum id_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull) continue;
        bytea *id_bytea = DatumGetByteaP(id_datum);
        uint8_t *id_data = (uint8_t *)VARDATA(id_bytea);

        /* Get label */
        char *label = SPI_getvalue(tuple, tupdesc, 2);
        if (!label) continue;

        /* Get depth */
        Datum depth_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull);
        int depth = isnull ? 1 : DatumGetInt32(depth_datum);

        /* Get frequency */
        Datum freq_datum = SPI_getbinval(tuple, tupdesc, 4, &isnull);
        double freq = isnull ? 1.0 : (double)DatumGetInt64(freq_datum);

        /* Get 4D centroid */
        Datum cx_datum = SPI_getbinval(tuple, tupdesc, 5, &isnull);
        double cx = isnull ? 0.0 : DatumGetFloat8(cx_datum);

        Datum cy_datum = SPI_getbinval(tuple, tupdesc, 6, &isnull);
        double cy = isnull ? 0.0 : DatumGetFloat8(cy_datum);

        Datum cz_datum = SPI_getbinval(tuple, tupdesc, 7, &isnull);
        double cz = isnull ? 0.0 : DatumGetFloat8(cz_datum);

        Datum cm_datum = SPI_getbinval(tuple, tupdesc, 8, &isnull);
        double cm = isnull ? 0.0 : DatumGetFloat8(cm_datum);

        /* Get hilbert index */
        Datum hilbert_datum = SPI_getbinval(tuple, tupdesc, 9, &isnull);
        double hilbert = isnull ? 0.0 : DatumGetFloat8(hilbert_datum);

        VocabEntry *entry = &collection->entries[collection->count];
        memcpy(entry->id, id_data, HASH_SIZE);
        entry->label = MemoryContextStrdup(CurrentMemoryContext, label);
        entry->depth = depth;
        entry->frequency = freq;
        entry->hilbert = hilbert;
        entry->centroid_x = cx;
        entry->centroid_y = cy;
        entry->centroid_z = cz;
        entry->centroid_m = cm;
        collection->count++;

        pfree(label);
    }

    safe_spi_finish();
    log_db_operation("load_vocab", collection->count, NULL);
    return collection;
}

/**
 * load_relations() - Load relations from relation table
 * relation_type: "S" for similarity (bigrams), "A" or "W" for attention
 * Returns: RelationCollection with source/target/weight triples
 * Memory: Allocated in CurrentMemoryContext, caller responsible for cleanup
 */
RelationCollection *load_relations(const char *relation_type)
{
    RelationCollection *collection = (RelationCollection *)palloc(sizeof(RelationCollection));
    collection->capacity = 100000;
    collection->entries = (RelationEntry *)palloc(sizeof(RelationEntry) * collection->capacity);
    collection->count = 0;
    collection->relation_type = relation_type;

    if (safe_spi_connect() != SPI_OK_CONNECT) {
        return collection;
    }

    /* Load from relation table */
    char query[512];
    if (strcmp(relation_type, "S") == 0) {
        /* Similarity edges for bigrams */
        snprintf(query, sizeof(query),
            "SELECT source_id, target_id, weight "
            "FROM relation "
            "WHERE relation_type = 'S' "
            "  AND weight > 0.3");
    } else {
        /* Attention edges (A, W, S) */
        snprintf(query, sizeof(query),
            "SELECT source_id, target_id, weight "
            "FROM relation "
            "WHERE relation_type IN ('A', 'W', 'S') "
            "  AND ABS(weight) > 0.1");
    }

    int ret = SPI_execute(query, true, 0);
    if (ret != SPI_OK_SELECT) {
        safe_spi_finish();
        log_db_error("load_relations", "SPI query failed");
        return collection;
    }

    for (uint64 i = 0; i < SPI_processed && collection->count < collection->capacity; i++) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool isnull;

        Datum left_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull) continue;
        bytea *left_bytea = DatumGetByteaP(left_datum);

        Datum right_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        if (isnull) continue;
        bytea *right_bytea = DatumGetByteaP(right_datum);

        Datum score_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull);
        /* weight is REAL (float4), not FLOAT8 */
        double score = isnull ? 0.0 : (double)DatumGetFloat4(score_datum);

        RelationEntry *entry = &collection->entries[collection->count];
        memcpy(entry->source_id, VARDATA(left_bytea), HASH_SIZE);
        memcpy(entry->target_id, VARDATA(right_bytea), HASH_SIZE);
        entry->weight = score;
        collection->count++;
    }

    safe_spi_finish();
    log_db_operation("load_relations", collection->count, relation_type);
    return collection;
}

/* ============================================================================
 * Cleanup Functions
 * ============================================================================ */

void free_edge_collection(EdgeCollection *collection)
{
    if (collection) {
        if (collection->edges) pfree(collection->edges);
        pfree(collection);
    }
}

void free_centroid_collection(CentroidCollection *collection)
{
    if (collection) {
        if (collection->entries) pfree(collection->entries);
        pfree(collection);
    }
}

void free_vocab_collection(VocabCollection *collection)
{
    if (collection) {
        if (collection->entries) pfree(collection->entries);
        pfree(collection);
    }
}

void free_relation_collection(RelationCollection *collection)
{
    if (collection) {
        if (collection->entries) pfree(collection->entries);
        pfree(collection);
    }
}