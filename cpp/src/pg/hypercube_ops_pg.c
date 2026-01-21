/**
 * hypercube_ops_pg.c - PostgreSQL Extension for High-Performance Batch Operations
 * 
 * Pure C implementation with in-memory arrays for batch operations.
 * NO GLOBAL STATICS - all state is local to function calls.
 * 
 * Key Optimizations:
 *   - Batch loading: One query loads all needed data
 *   - In-memory processing: Graph algorithms run without SPI in loop
 *   - PARALLEL SAFE: Can run on multiple cores
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "access/htup_details.h"
#include "utils/memutils.h"

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "pg_utils.h"
#include "db_wrapper_pg.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

typedef struct WalkStep {
    uint8   id[HASH_SIZE];
    double  weight;
} WalkStep;

/* ============================================================================
 * Load Semantic Edges from Database (ONE query)
 * Uses wrapper for proper edge traversal
 * ============================================================================ */

static SemanticEdge *load_all_edges(int *out_count)
{
    EdgeCollection *collection = load_edges();
    *out_count = collection->count;
    /* Note: collection->edges is allocated in CurrentMemoryContext, so it will be
       automatically freed when the function context ends. We return the array directly
       for backward compatibility with existing code. */
    return collection->edges;
}

/* ============================================================================
 * Random Walk Using Edge Array (all in-memory)
 * ============================================================================ */

static WalkStep *random_walk(const uint8 *seed, int steps, 
                              SemanticEdge *edges, int edge_count,
                              int *out_steps)
{
    WalkStep *path = (WalkStep *)palloc(sizeof(WalkStep) * (steps + 1));
    int path_len = 0;
    
    /* Handle NULL edges */
    if (edges == NULL || edge_count == 0) {
        memcpy(path[0].id, seed, HASH_SIZE);
        path[0].weight = 0;
        *out_steps = 1;
        return path;
    }
    
    /* Simple visited tracking */
    uint8 (*visited)[HASH_SIZE] = (uint8 (*)[HASH_SIZE])palloc(sizeof(uint8[HASH_SIZE]) * (steps + 1));
    int visited_count = 0;
    
    uint8 current[HASH_SIZE];
    memcpy(current, seed, HASH_SIZE);
    
    /* Deterministic seed derived from input hash for reproducibility */
    /* XOR all bytes of seed hash to create a 32-bit seed */
    unsigned int rng_seed = 0;
    for (int i = 0; i < HASH_SIZE; i++) {
        rng_seed ^= ((unsigned int)seed[i]) << ((i % 4) * 8);
    }
    /* Mix in steps count for variation with same seed */
    rng_seed ^= (unsigned int)steps * 2654435761u;
    srand(rng_seed);
    
    for (int step = 0; step <= steps; step++) {
        /* Add current to path */
        memcpy(path[path_len].id, current, HASH_SIZE);
        path[path_len].weight = 0;
        
        /* Mark visited */
        memcpy(visited[visited_count], current, HASH_SIZE);
        visited_count++;
        
        /* Find edges from current node */
        SemanticEdge *candidates = (SemanticEdge *)palloc(sizeof(SemanticEdge) * 100);
        int candidate_count = 0;
        double total_weight = 0;
        
        for (int i = 0; i < edge_count && candidate_count < 100; i++) {
            if (!hash_equals(edges[i].source, current)) continue;
            
            /* Check if target is visited */
            bool is_visited = false;
            for (int j = 0; j < visited_count; j++) {
                if (hash_equals(visited[j], edges[i].target)) {
                    is_visited = true;
                    break;
                }
            }
            if (is_visited) continue;
            
            candidates[candidate_count] = edges[i];
            total_weight += edges[i].weight;
            candidate_count++;
        }
        
        if (candidate_count == 0) {
            pfree(candidates);
            path_len++;
            break;
        }
        
        /* Weighted random selection */
        double r = (double)rand() / RAND_MAX * total_weight;
        double cumulative = 0;
        int chosen = 0;
        
        for (int i = 0; i < candidate_count; i++) {
            cumulative += candidates[i].weight;
            if (r <= cumulative) {
                chosen = i;
                break;
            }
        }
        
        path[path_len].weight = candidates[chosen].weight;
        memcpy(current, candidates[chosen].target, HASH_SIZE);
        
        pfree(candidates);
        path_len++;
    }
    
    pfree(visited);
    *out_steps = path_len;
    return path;
}

/* ============================================================================
 * BFS Shortest Path Using Edge Array (all in-memory)
 * ============================================================================ */

typedef struct {
    uint8 id[HASH_SIZE];
    uint8 parent[HASH_SIZE];
    int depth;
} BFSNode;

static uint8 *shortest_path(const uint8 *from, const uint8 *to,
                            int max_depth,
                            SemanticEdge *edges, int edge_count,
                            int *out_len)
{
    /* Handle NULL edges */
    if (edges == NULL || edge_count == 0) {
        *out_len = 0;
        return NULL;
    }
    
    /* BFS queue */
    int queue_cap = 10000;
    BFSNode *queue = (BFSNode *)palloc(sizeof(BFSNode) * queue_cap);
    int queue_head = 0, queue_tail = 0;
    
    /* Enqueue start */
    memcpy(queue[queue_tail].id, from, HASH_SIZE);
    memcpy(queue[queue_tail].parent, from, HASH_SIZE);
    queue[queue_tail].depth = 0;
    queue_tail++;
    
    bool found = false;
    
    while (queue_head < queue_tail && !found) {
        BFSNode curr = queue[queue_head++];
        
        if (curr.depth >= max_depth) continue;
        
        /* Find neighbors */
        for (int i = 0; i < edge_count && !found; i++) {
            if (!hash_equals(edges[i].source, curr.id)) continue;
            
            /* Check if already visited */
            bool visited = false;
            for (int j = 0; j < queue_tail; j++) {
                if (hash_equals(queue[j].id, edges[i].target)) {
                    visited = true;
                    break;
                }
            }
            if (visited) continue;
            
            /* Check if target found */
            if (hash_equals(edges[i].target, to)) {
                /* Add final node */
                if (queue_tail < queue_cap) {
                    memcpy(queue[queue_tail].id, to, HASH_SIZE);
                    memcpy(queue[queue_tail].parent, curr.id, HASH_SIZE);
                    queue[queue_tail].depth = curr.depth + 1;
                    queue_tail++;
                }
                found = true;
                break;
            }
            
            /* Enqueue neighbor */
            if (queue_tail < queue_cap) {
                memcpy(queue[queue_tail].id, edges[i].target, HASH_SIZE);
                memcpy(queue[queue_tail].parent, curr.id, HASH_SIZE);
                queue[queue_tail].depth = curr.depth + 1;
                queue_tail++;
            }
        }
    }
    
    if (!found) {
        pfree(queue);
        *out_len = 0;
        return NULL;
    }
    
    /* Reconstruct path backwards from target */
    uint8 *path = (uint8 *)palloc(HASH_SIZE * (max_depth + 2));
    int path_len = 0;
    
    uint8 current[HASH_SIZE];
    memcpy(current, to, HASH_SIZE);
    
    while (path_len < max_depth + 2) {
        memcpy(path + path_len * HASH_SIZE, current, HASH_SIZE);
        path_len++;
        
        if (hash_equals(current, from))
            break;
        
        /* Find parent in queue */
        bool found_parent = false;
        for (int i = queue_tail - 1; i >= 0; i--) {
            if (hash_equals(queue[i].id, current)) {
                memcpy(current, queue[i].parent, HASH_SIZE);
                found_parent = true;
                break;
            }
        }
        if (!found_parent) break;
    }
    
    pfree(queue);
    
    /* Reverse path */
    for (int i = 0; i < path_len / 2; i++) {
        uint8 tmp[HASH_SIZE];
        memcpy(tmp, path + i * HASH_SIZE, HASH_SIZE);
        memcpy(path + i * HASH_SIZE, path + (path_len - 1 - i) * HASH_SIZE, HASH_SIZE);
        memcpy(path + (path_len - 1 - i) * HASH_SIZE, tmp, HASH_SIZE);
    }
    
    *out_len = path_len;
    return path;
}

/* ============================================================================
 * PostgreSQL Function: hypercube_semantic_walk
 * ============================================================================ */

typedef struct {
    WalkStep   *steps;
    int         num_steps;
} SemanticWalkState;

PG_FUNCTION_INFO_V1(hypercube_semantic_walk);
Datum hypercube_semantic_walk(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    SemanticWalkState *state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        bytea *seed_bytea = PG_GETARG_BYTEA_PP(0);
        int steps = PG_NARGS() > 1 ? PG_GETARG_INT32(1) : 10;
        
        uint8 seed[HASH_SIZE];
        bytea_to_hash(seed_bytea, seed);
        
        /* Connect and load edges with ONE query */
        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        
        int edge_count = 0;
        SemanticEdge *edges = load_all_edges(&edge_count);
        
        /* Perform walk entirely in memory */
        state = (SemanticWalkState *)palloc0(sizeof(SemanticWalkState));
        state->steps = random_walk(seed, steps, edges, edge_count, &state->num_steps);
        
        if (edges) pfree(edges);
        SPI_finish();
        
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite")));
        }
        
        funcctx->user_fctx = state;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = state->num_steps;
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (SemanticWalkState *)funcctx->user_fctx;
    
    if (funcctx->call_cntr < funcctx->max_calls) {
        int idx = funcctx->call_cntr;
        
        Datum values[3];
        bool isnulls[3] = {false, false, false};
        
        values[0] = Int32GetDatum((int32)idx);
        values[1] = PointerGetDatum(hash_to_bytea(state->steps[idx].id));
        values[2] = Float8GetDatum(state->steps[idx].weight);
        
        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, isnulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    SRF_RETURN_DONE(funcctx);
}

/* ============================================================================
 * PostgreSQL Function: hypercube_semantic_path
 * ============================================================================ */

typedef struct {
    uint8  *path;
    int     path_len;
} SemanticPathState;

PG_FUNCTION_INFO_V1(hypercube_semantic_path);
Datum hypercube_semantic_path(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    SemanticPathState *state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        bytea *from_bytea = PG_GETARG_BYTEA_PP(0);
        bytea *to_bytea = PG_GETARG_BYTEA_PP(1);
        int max_depth = PG_NARGS() > 2 ? PG_GETARG_INT32(2) : 6;
        
        uint8 from[HASH_SIZE], to[HASH_SIZE];
        bytea_to_hash(from_bytea, from);
        bytea_to_hash(to_bytea, to);
        
        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        
        int edge_count = 0;
        SemanticEdge *edges = load_all_edges(&edge_count);
        
        state = (SemanticPathState *)palloc0(sizeof(SemanticPathState));
        state->path = shortest_path(from, to, max_depth, edges, edge_count, &state->path_len);
        
        if (edges) pfree(edges);
        SPI_finish();
        
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite")));
        }
        
        funcctx->user_fctx = state;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = state->path_len;
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (SemanticPathState *)funcctx->user_fctx;
    
    if (funcctx->call_cntr < funcctx->max_calls) {
        int idx = funcctx->call_cntr;
        
        Datum values[2];
        bool isnulls[2] = {false, false};
        
        values[0] = Int32GetDatum((int32)idx);
        values[1] = PointerGetDatum(hash_to_bytea(state->path + idx * HASH_SIZE));
        
        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, isnulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    SRF_RETURN_DONE(funcctx);
}

/* ============================================================================
 * PostgreSQL Function: hypercube_batch_lookup
 * Batch lookup atoms by ID array
 * ============================================================================ */

typedef struct {
    int     current;
    int     total;
    uint8  *ids;
    int    *depths;
    bool   *is_leafs;
    int    *child_counts;
    double *centroid_xs;
} BatchLookupState;

PG_FUNCTION_INFO_V1(hypercube_batch_lookup);
Datum hypercube_batch_lookup(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    BatchLookupState *state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        ArrayType *ids_arr = PG_GETARG_ARRAYTYPE_P(0);
        Datum *elems;
        bool *nulls;
        int nelems;
        deconstruct_array(ids_arr, BYTEAOID, -1, false, TYPALIGN_INT,
                         &elems, &nulls, &nelems);
        
        state = (BatchLookupState *)palloc0(sizeof(BatchLookupState));
        state->total = nelems;
        state->ids = (uint8 *)palloc(nelems * HASH_SIZE);
        state->depths = (int *)palloc0(nelems * sizeof(int));
        state->is_leafs = (bool *)palloc0(nelems * sizeof(bool));
        state->child_counts = (int *)palloc0(nelems * sizeof(int));
        state->centroid_xs = (double *)palloc0(nelems * sizeof(double));
        
        for (int i = 0; i < nelems; i++) {
            if (!nulls[i]) {
                bytea_to_hash(DatumGetByteaP(elems[i]), state->ids + i * HASH_SIZE);
            }
        }
        
        /* Query with composition_child table for child counts */
        StringInfoData query;
        initStringInfo(&query);
        appendStringInfoString(&query, 
            "SELECT a.id, 0 as depth, (a.value IS NOT NULL) as is_leaf, "
            "(SELECT COUNT(*) FROM composition_child cc WHERE cc.composition_id = a.id)::int as child_count, ST_X(a.geom) "
            "FROM atom a WHERE a.id = ANY(ARRAY[");
        
        for (int i = 0; i < nelems; i++) {
            if (i > 0) appendStringInfoChar(&query, ',');
            appendStringInfoString(&query, "'\\x");
            for (int j = 0; j < HASH_SIZE; j++) {
                appendStringInfo(&query, "%02x", state->ids[i * HASH_SIZE + j]);
            }
            appendStringInfoString(&query, "'::bytea");
        }
        appendStringInfoString(&query, "])");
        
        int ret = SPI_execute(query.data, true, 0);
        if (ret == SPI_OK_SELECT) {
            for (uint64 i = 0; i < SPI_processed; i++) {
                HeapTuple tuple = SPI_tuptable->vals[i];
                TupleDesc tupdesc = SPI_tuptable->tupdesc;
                bool isnull;
                
                Datum id_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
                if (isnull) continue;
                
                uint8 id[HASH_SIZE];
                bytea_to_hash(DatumGetByteaP(id_datum), id);
                
                /* Find matching input index */
                for (int j = 0; j < nelems; j++) {
                    if (memcmp(state->ids + j * HASH_SIZE, id, HASH_SIZE) == 0) {
                        Datum d = SPI_getbinval(tuple, tupdesc, 2, &isnull);
                        state->depths[j] = isnull ? 0 : DatumGetInt32(d);
                        
                        d = SPI_getbinval(tuple, tupdesc, 3, &isnull);
                        state->is_leafs[j] = !isnull && DatumGetBool(d);
                        
                        d = SPI_getbinval(tuple, tupdesc, 4, &isnull);
                        state->child_counts[j] = isnull ? 0 : DatumGetInt32(d);
                        
                        d = SPI_getbinval(tuple, tupdesc, 5, &isnull);
                        state->centroid_xs[j] = isnull ? 0 : DatumGetFloat8(d);
                        break;
                    }
                }
            }
        }
        
        pfree(query.data);
        SPI_finish();
        
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite")));
        }
        
        funcctx->user_fctx = state;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = nelems;
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (BatchLookupState *)funcctx->user_fctx;
    
    if (funcctx->call_cntr < funcctx->max_calls) {
        int idx = funcctx->call_cntr;
        
        Datum values[5];
        bool isnulls[5] = {false, false, false, false, false};
        
        values[0] = PointerGetDatum(hash_to_bytea(state->ids + idx * HASH_SIZE));
        values[1] = Int32GetDatum(state->depths[idx]);
        values[2] = BoolGetDatum(state->is_leafs[idx]);
        values[3] = Int32GetDatum(state->child_counts[idx]);
        values[4] = Float8GetDatum(state->centroid_xs[idx]);
        
        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, isnulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    SRF_RETURN_DONE(funcctx);
}

/* ============================================================================
 * PostgreSQL Function: hypercube_batch_reconstruct
 * Batch reconstruct text from multiple atom IDs
 * Uses CTE with RECURSIVE to load all descendants in ONE query
 * ============================================================================ */

typedef struct AtomNode {
    uint8   id[HASH_SIZE];
    uint8  *children;       /* Array of child hashes, N * HASH_SIZE bytes */
    int     child_count;
    char   *value;          /* UTF-8 text for leaves, NULL otherwise */
    int     value_len;
} AtomNode;

/* Hash table for fast atom lookup during reconstruction */
#define ATOM_HASH_SIZE 16384

typedef struct AtomHashEntry {
    uint8       id[HASH_SIZE];
    AtomNode   *node;
    struct AtomHashEntry *next;
} AtomHashEntry;

static unsigned int hash_id(const uint8 *id) {
    /* Use first 4 bytes as hash - BLAKE3 is well-distributed */
    return (id[0] | (id[1] << 8) | (id[2] << 16) | (id[3] << 24)) & (ATOM_HASH_SIZE - 1);
}

static AtomNode *find_atom(AtomHashEntry **table, const uint8 *id) {
    unsigned int h = hash_id(id);
    AtomHashEntry *e = table[h];
    while (e) {
        if (memcmp(e->id, id, HASH_SIZE) == 0) return e->node;
        e = e->next;
    }
    return NULL;
}

static void insert_atom(AtomHashEntry **table, AtomNode *node, MemoryContext ctx) {
    unsigned int h = hash_id(node->id);
    AtomHashEntry *e = MemoryContextAlloc(ctx, sizeof(AtomHashEntry));
    memcpy(e->id, node->id, HASH_SIZE);
    e->node = node;
    e->next = table[h];
    table[h] = e;
}

/* Recursive in-memory text reconstruction */
static void reconstruct_node(AtomHashEntry **table, const uint8 *id,
                             StringInfo result, int max_depth) {
    if (max_depth <= 0) return;

    AtomNode *node = find_atom(table, id);
    if (!node) return;

    if (node->value && node->value_len > 0) {
        /* Leaf node - append value */
        appendBinaryStringInfo(result, node->value, node->value_len);
    } else if (node->children && node->child_count > 0) {
        /* Composition - recurse into children in order */
        for (int i = 0; i < node->child_count; i++) {
            reconstruct_node(table, node->children + i * HASH_SIZE, result, max_depth - 1);
        }
    }
}

typedef struct {
    int     current;
    int     total;
    uint8  *ids;
    text  **texts;
} BatchReconstructState;

PG_FUNCTION_INFO_V1(hypercube_batch_reconstruct);
Datum hypercube_batch_reconstruct(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    BatchReconstructState *state;

    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        ArrayType *ids_arr = PG_GETARG_ARRAYTYPE_P(0);
        Datum *elems;
        bool *nulls;
        int nelems;
        deconstruct_array(ids_arr, BYTEAOID, -1, false, TYPALIGN_INT,
                         &elems, &nulls, &nelems);

        state = (BatchReconstructState *)palloc0(sizeof(BatchReconstructState));
        state->total = nelems;
        state->ids = (uint8 *)palloc(nelems * HASH_SIZE);
        state->texts = (text **)palloc0(nelems * sizeof(text *));

        for (int i = 0; i < nelems; i++) {
            if (!nulls[i]) {
                bytea_to_hash(DatumGetByteaP(elems[i]), state->ids + i * HASH_SIZE);
            }
        }

        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }

        /* Build IN clause for root IDs */
        StringInfoData in_clause;
        initStringInfo(&in_clause);
        appendStringInfoString(&in_clause, "VALUES ");
        for (int i = 0; i < nelems; i++) {
            if (i > 0) appendStringInfoChar(&in_clause, ',');
            appendStringInfoString(&in_clause, "('\\x");
            for (int j = 0; j < HASH_SIZE; j++) {
                appendStringInfo(&in_clause, "%02x", state->ids[i * HASH_SIZE + j]);
            }
            appendStringInfoString(&in_clause, "'::bytea)");
        }

        /* Single recursive CTE query using relation table */
        StringInfoData query;
        initStringInfo(&query);
        appendStringInfo(&query,
            "WITH RECURSIVE roots(id) AS (%s), "
            "tree AS ("
            "  SELECT a.id, a.value, 0 as depth, 'A' as node_type "
            "  FROM atom a JOIN roots r ON a.id = r.id "
            "  UNION ALL "
            "  SELECT c.id, NULL::bytea, c.depth, 'C' as node_type "
            "  FROM composition c JOIN roots r ON c.id = r.id "
            "  UNION ALL "
            "  SELECT COALESCE(a.id, c.id), COALESCE(a.value, NULL), COALESCE(0, c.depth), cc.child_type "
            "  FROM tree t "
            "  JOIN composition_child cc ON cc.composition_id = t.id "
            "  LEFT JOIN atom a ON cc.child_type = 'A' AND a.id = cc.child_id "
            "  LEFT JOIN composition c ON cc.child_type = 'C' AND c.id = cc.child_id "
            "  WHERE t.node_type = 'C'"
            ") "
            "SELECT DISTINCT t.id, t.value, "
            "  (SELECT array_agg(cc.child_id ORDER BY cc.ordinal) FROM composition_child cc WHERE cc.composition_id = t.id) as children "
            "FROM tree t",
            in_clause.data);

        int ret = SPI_execute(query.data, true, 0);

        /* Build in-memory hash table of all atoms */
        AtomHashEntry **atom_table = palloc0(ATOM_HASH_SIZE * sizeof(AtomHashEntry *));

        if (ret == SPI_OK_SELECT) {
            for (uint64 i = 0; i < SPI_processed; i++) {
                HeapTuple tuple = SPI_tuptable->vals[i];
                TupleDesc tupdesc = SPI_tuptable->tupdesc;
                bool isnull;

                Datum id_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
                if (isnull) continue;

                AtomNode *node = MemoryContextAlloc(funcctx->multi_call_memory_ctx,
                                                    sizeof(AtomNode));
                memset(node, 0, sizeof(AtomNode));

                bytea *id_bytea = DatumGetByteaP(id_datum);
                memcpy(node->id, VARDATA_ANY(id_bytea), HASH_SIZE);

                /* Get value (for leaves) - now at column 2 */
                Datum value_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
                if (!isnull) {
                    bytea *value_bytea = DatumGetByteaP(value_datum);
                    int vlen = VARSIZE_ANY_EXHDR(value_bytea);
                    if (vlen > 0) {
                        node->value = MemoryContextAlloc(funcctx->multi_call_memory_ctx, vlen);
                        memcpy(node->value, VARDATA_ANY(value_bytea), vlen);
                        node->value_len = vlen;
                    }
                }

                /* Get children array (now at column 3, from subquery) */
                Datum children_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull);
                if (!isnull) {
                    ArrayType *children_arr = DatumGetArrayTypeP(children_datum);
                    Datum *child_elems;
                    bool *child_nulls;
                    int nchildren;
                    deconstruct_array(children_arr, BYTEAOID, -1, false, TYPALIGN_INT,
                                     &child_elems, &child_nulls, &nchildren);

                    if (nchildren > 0) {
                        node->child_count = nchildren;
                        node->children = MemoryContextAlloc(funcctx->multi_call_memory_ctx,
                                                            nchildren * HASH_SIZE);
                        for (int j = 0; j < nchildren; j++) {
                            if (!child_nulls[j]) {
                                bytea *child_bytea = DatumGetByteaP(child_elems[j]);
                                memcpy(node->children + j * HASH_SIZE,
                                       VARDATA_ANY(child_bytea), HASH_SIZE);
                            }
                        }
                    }
                }

                insert_atom(atom_table, node, funcctx->multi_call_memory_ctx);
            }
        }

        pfree(query.data);
        pfree(in_clause.data);
        SPI_finish();

        /* Reconstruct each requested ID from in-memory data */
        for (int i = 0; i < nelems; i++) {
            StringInfoData result;
            initStringInfo(&result);

            reconstruct_node(atom_table, state->ids + i * HASH_SIZE, &result, 1000);

            if (result.len > 0) {
                state->texts[i] = (text *)MemoryContextAlloc(
                    funcctx->multi_call_memory_ctx,
                    VARHDRSZ + result.len
                );
                SET_VARSIZE(state->texts[i], VARHDRSZ + result.len);
                memcpy(VARDATA(state->texts[i]), result.data, result.len);
            }
            pfree(result.data);
        }

        pfree(atom_table);

        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite")));
        }

        funcctx->user_fctx = state;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = nelems;

        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    state = (BatchReconstructState *)funcctx->user_fctx;

    if (funcctx->call_cntr < funcctx->max_calls) {
        int idx = funcctx->call_cntr;

        Datum values[2];
        bool isnulls[2] = {false, false};

        values[0] = PointerGetDatum(hash_to_bytea(state->ids + idx * HASH_SIZE));

        if (state->texts[idx]) {
            values[1] = PointerGetDatum(state->texts[idx]);
        } else {
            isnulls[1] = true;
        }

        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, isnulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }

    SRF_RETURN_DONE(funcctx);
}
/* ============================================================================
 * PostgreSQL Function: hypercube_knn_batch
 * Fast K-nearest neighbors using in-memory centroid cache
 * ============================================================================ */

typedef struct {
    uint8   id[HASH_SIZE];
    double  cx, cy, cz, cm;
} AtomCentroid;

static double euclidean_4d(double x1, double y1, double z1, double m1,
                           double x2, double y2, double z2, double m2)
{
    double dx = x1 - x2;
    double dy = y1 - y2;
    double dz = z1 - z2;
    double dm = m1 - m2;
    return sqrt(dx*dx + dy*dy + dz*dz + dm*dm);
}

typedef struct {
    uint8   id[HASH_SIZE];
    double  dist;
} KNNResult;

static int knn_compare(const void *a, const void *b)
{
    const KNNResult *ra = (const KNNResult *)a;
    const KNNResult *rb = (const KNNResult *)b;
    if (ra->dist < rb->dist) return -1;
    if (ra->dist > rb->dist) return 1;
    return 0;
}

typedef struct {
    KNNResult *results;
    int        count;
    int        current;
} KNNState;

PG_FUNCTION_INFO_V1(hypercube_knn_batch);
Datum hypercube_knn_batch(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    
    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        bytea *target_bytea = PG_GETARG_BYTEA_PP(0);
        int k = PG_NARGS() > 1 ? PG_GETARG_INT32(1) : 10;
        int depth_filter = PG_NARGS() > 2 ? PG_GETARG_INT32(2) : -1;
        
        uint8 target_id[HASH_SIZE];
        memcpy(target_id, VARDATA_ANY(target_bytea), HASH_SIZE);
        
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed")));
        
        /* Load target centroid from composition table */
        char query[512];
        snprintf(query, sizeof(query),
            "SELECT ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid) "
            "FROM composition WHERE id = '\\x");
        for (int i = 0; i < HASH_SIZE; i++)
            snprintf(query + strlen(query), sizeof(query) - strlen(query), "%02x", target_id[i]);
        strcat(query, "'::bytea");
        
        int ret = SPI_execute(query, true, 1);
        if (ret != SPI_OK_SELECT || SPI_processed == 0)
        {
            SPI_finish();
            MemoryContextSwitchTo(oldcontext);
            funcctx->max_calls = 0;
            SRF_RETURN_DONE(funcctx);
        }
        
        bool isnull;
        double tx = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
        double ty = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull));
        double tz = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3, &isnull));
        double tm = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4, &isnull));
        
        /* Load all compositions and compute distances */
        if (depth_filter >= 0)
            snprintf(query, sizeof(query),
                "SELECT id, ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid) "
                "FROM composition WHERE depth = %d LIMIT 100000", depth_filter);
        else
            snprintf(query, sizeof(query),
                "SELECT id, ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid) "
                "FROM composition LIMIT 100000");
        
        ret = SPI_execute(query, true, 0);
        if (ret != SPI_OK_SELECT)
        {
            SPI_finish();
            MemoryContextSwitchTo(oldcontext);
            funcctx->max_calls = 0;
            SRF_RETURN_DONE(funcctx);
        }
        
        /* SPI_processed is uint64, defensive cap at LIMIT value */
        uint64_t total_rows = SPI_processed;
        if (total_rows > 100000) total_rows = 100000;
        int count = (int)total_rows;
        KNNResult *results = palloc(count * sizeof(KNNResult));
        int result_count = 0;
        
        for (int i = 0; i < count; i++)
        {
            HeapTuple tuple = SPI_tuptable->vals[i];
            TupleDesc tupdesc = SPI_tuptable->tupdesc;
            
            bytea *id = DatumGetByteaP(SPI_getbinval(tuple, tupdesc, 1, &isnull));
            if (isnull) continue;
            
            /* Skip self */
            if (memcmp(VARDATA_ANY(id), target_id, HASH_SIZE) == 0) continue;
            
            double cx = DatumGetFloat8(SPI_getbinval(tuple, tupdesc, 2, &isnull));
            double cy = DatumGetFloat8(SPI_getbinval(tuple, tupdesc, 3, &isnull));
            double cz = DatumGetFloat8(SPI_getbinval(tuple, tupdesc, 4, &isnull));
            double cm = DatumGetFloat8(SPI_getbinval(tuple, tupdesc, 5, &isnull));
            
            memcpy(results[result_count].id, VARDATA_ANY(id), HASH_SIZE);
            results[result_count].dist = euclidean_4d(tx, ty, tz, tm, cx, cy, cz, cm);
            result_count++;
        }
        
        SPI_finish();
        
        /* Sort by distance */
        qsort(results, result_count, sizeof(KNNResult), knn_compare);
        
        /* Keep only k results */
        int final_count = result_count < k ? result_count : k;
        
        KNNState *state = palloc(sizeof(KNNState));
        state->results = results;
        state->count = final_count;
        state->current = 0;
        
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR, (errmsg("function must return composite")));
        
        funcctx->user_fctx = state;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = final_count;
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    KNNState *state = (KNNState *)funcctx->user_fctx;
    
    if (state->current < state->count)
    {
        KNNResult *r = &state->results[state->current];
        state->current++;
        
        Datum values[2];
        bool nulls[2] = {false, false};
        
        bytea *id_bytea = palloc(VARHDRSZ + HASH_SIZE);
        SET_VARSIZE(id_bytea, VARHDRSZ + HASH_SIZE);
        memcpy(VARDATA(id_bytea), r->id, HASH_SIZE);
        
        values[0] = PointerGetDatum(id_bytea);
        values[1] = Float8GetDatum(r->dist);
        
        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    SRF_RETURN_DONE(funcctx);
}

PG_FUNCTION_INFO_V1(hypercube_attention);
Datum hypercube_attention(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    
    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldcontext;
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        bytea *target_bytea = PG_GETARG_BYTEA_PP(0);
        int k = PG_NARGS() > 1 ? PG_GETARG_INT32(1) : 10;
        
        uint8 target_id[HASH_SIZE];
        memcpy(target_id, VARDATA_ANY(target_bytea), HASH_SIZE);
        
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed")));
        
        /* Load target centroid from composition */
        char query[512];
        snprintf(query, sizeof(query),
            "SELECT ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid) "
            "FROM composition WHERE id = '\\x");
        for (int i = 0; i < HASH_SIZE; i++)
            snprintf(query + strlen(query), sizeof(query) - strlen(query), "%02x", target_id[i]);
        strcat(query, "'::bytea");
        
        int ret = SPI_execute(query, true, 1);
        if (ret != SPI_OK_SELECT || SPI_processed == 0)
        {
            SPI_finish();
            MemoryContextSwitchTo(oldcontext);
            funcctx->max_calls = 0;
            SRF_RETURN_DONE(funcctx);
        }
        
        bool isnull;
        double tx = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
        double ty = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull));
        double tz = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3, &isnull));
        double tm = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4, &isnull));
        
        /* Load all compositions and compute attention scores */
        snprintf(query, sizeof(query),
            "SELECT id, ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid) "
            "FROM composition LIMIT 100000");
        
        ret = SPI_execute(query, true, 0);
        if (ret != SPI_OK_SELECT)
        {
            SPI_finish();
            MemoryContextSwitchTo(oldcontext);
            funcctx->max_calls = 0;
            SRF_RETURN_DONE(funcctx);
        }
        
        /* SPI_processed is uint64, defensive cap at LIMIT value */
        uint64_t total_attn = SPI_processed;
        if (total_attn > 100000) total_attn = 100000;
        int count = (int)total_attn;
        KNNResult *results = palloc(count * sizeof(KNNResult));
        int result_count = 0;
        
        for (int i = 0; i < count; i++)
        {
            HeapTuple tuple = SPI_tuptable->vals[i];
            TupleDesc tupdesc = SPI_tuptable->tupdesc;
            
            bytea *id = DatumGetByteaP(SPI_getbinval(tuple, tupdesc, 1, &isnull));
            if (isnull) continue;
            
            /* Skip self */
            if (memcmp(VARDATA_ANY(id), target_id, HASH_SIZE) == 0) continue;
            
            double cx = DatumGetFloat8(SPI_getbinval(tuple, tupdesc, 2, &isnull));
            double cy = DatumGetFloat8(SPI_getbinval(tuple, tupdesc, 3, &isnull));
            double cz = DatumGetFloat8(SPI_getbinval(tuple, tupdesc, 4, &isnull));
            double cm = DatumGetFloat8(SPI_getbinval(tuple, tupdesc, 5, &isnull));
            
            double dist = euclidean_4d(tx, ty, tz, tm, cx, cy, cz, cm);
            
            memcpy(results[result_count].id, VARDATA_ANY(id), HASH_SIZE);
            /* Attention score = 1 / (1 + distance) */
            results[result_count].dist = 1.0 / (1.0 + dist);
            result_count++;
        }
        
        SPI_finish();
        
        /* Sort by attention score (descending - higher is better) */
        /* We negate scores for sorting since knn_compare sorts ascending */
        for (int i = 0; i < result_count; i++)
            results[i].dist = -results[i].dist;
        qsort(results, result_count, sizeof(KNNResult), knn_compare);
        for (int i = 0; i < result_count; i++)
            results[i].dist = -results[i].dist;
        
        int final_count = result_count < k ? result_count : k;
        
        KNNState *state = palloc(sizeof(KNNState));
        state->results = results;
        state->count = final_count;
        state->current = 0;
        
        TupleDesc tupdesc;
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR, (errmsg("function must return composite")));
        
        funcctx->user_fctx = state;
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        funcctx->max_calls = final_count;
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    KNNState *state = (KNNState *)funcctx->user_fctx;
    
    if (state->current < state->count)
    {
        KNNResult *r = &state->results[state->current];
        state->current++;
        
        Datum values[2];
        bool nulls[2] = {false, false};
        
        bytea *id_bytea = palloc(VARHDRSZ + HASH_SIZE);
        SET_VARSIZE(id_bytea, VARHDRSZ + HASH_SIZE);
        memcpy(VARDATA(id_bytea), r->id, HASH_SIZE);
        
        values[0] = PointerGetDatum(id_bytea);
        values[1] = Float8GetDatum(r->dist);  /* This is actually the attention score */
        
        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    SRF_RETURN_DONE(funcctx);
}

/* ============================================================================
 * SEED ATOMS - High-Performance Unicode Atom Seeding
 * Generates all ~1.1M Unicode codepoint atoms using batch INSERT
 * Called as: SELECT seed_atoms();
 * ============================================================================ */

#include "hypercube_c.h"

/* Helper: encode codepoint as UTF-8 bytes, return length */
static int encode_utf8(uint32_t cp, uint8_t *out)
{
    if (cp < 0x80) {
        out[0] = (uint8_t)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = 0xC0 | (cp >> 6);
        out[1] = 0x80 | (cp & 0x3F);
        return 2;
    } else if (cp < 0x10000) {
        out[0] = 0xE0 | (cp >> 12);
        out[1] = 0x80 | ((cp >> 6) & 0x3F);
        out[2] = 0x80 | (cp & 0x3F);
        return 3;
    } else {
        out[0] = 0xF0 | (cp >> 18);
        out[1] = 0x80 | ((cp >> 12) & 0x3F);
        out[2] = 0x80 | ((cp >> 6) & 0x3F);
        out[3] = 0x80 | (cp & 0x3F);
        return 4;
    }
}

/* Helper: convert double to hex string for EWKB */
static void double_to_hex(double val, char* out)
{
    union { double d; uint64_t u; } conv = {val};
    uint64_t bits = conv.u;
    static const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
        uint8_t byte = (bits >> (i * 8)) & 0xFF;
        out[i * 2] = hex_chars[byte >> 4];
        out[i * 2 + 1] = hex_chars[byte & 0x0F];
    }
}

PG_FUNCTION_INFO_V1(seed_atoms);
Datum seed_atoms(PG_FUNCTION_ARGS)
{
    int64 inserted = 0;
    int ret;
    bool isnull;
    int processed_count = 0;
    const int TOTAL_CODEPOINTS = 1114112;

    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("SPI_connect failed")));

    ereport(NOTICE, (errmsg("seed_atoms: starting")));

    /* Check if atoms already exist */
    ret = SPI_execute("SELECT COUNT(*) FROM atom", true, 0);
    if (ret == SPI_OK_SELECT && SPI_processed > 0) {
        int64 existing = DatumGetInt64(SPI_getbinval(
            SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
        if (existing > 1000000) {
            SPI_finish();
            ereport(NOTICE, (errmsg("atoms already seeded: %lld", (long long)existing)));
            PG_RETURN_INT64(existing);
        }
    }

    ereport(WARNING, (errmsg("=== PostgreSQL Extension Atom Seeder ===")));
    ereport(WARNING, (errmsg("Processing ~1.1M Unicode codepoints...")));

    /* Use batch processing for better performance */
    static const int BATCH_SIZE = 1000;
    uint32_t codepoint = 0;
    uint32_t max_codepoint = 0x10FFFF;
    uint32_t surrogate_start = 0xD800;
    uint32_t surrogate_end = 0xDFFF;

    while (codepoint <= max_codepoint) {
        /* Start a batch */
        StringInfoData batch_sql;
        initStringInfo(&batch_sql);
        appendStringInfoString(&batch_sql, "INSERT INTO atom (id, codepoint, value, geom, hilbert_lo, hilbert_hi) VALUES ");

        int batch_count = 0;
        uint32_t batch_start = codepoint;

        /* Build batch of INSERTs */
        for (; codepoint <= max_codepoint && batch_count < BATCH_SIZE; ++codepoint) {
            /* Skip surrogates */
            if (codepoint >= surrogate_start && codepoint <= surrogate_end) {
                continue;
            }

            /* Map codepoint using C API functions */
            hc_point4d_t coords = hc_map_codepoint(codepoint);
            hc_hash_t hash = hc_blake3_codepoint(codepoint);
            hc_hilbert_t hilbert = hc_coords_to_hilbert(coords);

            /* Convert hash to hex */
            char hash_hex[65];
            hc_hash_to_hex(hash, hash_hex);

            /* Encode UTF-8 */
            uint8_t utf8_bytes[4];
            int utf8_len = encode_utf8(codepoint, utf8_bytes);

            /* Build EWKB geometry (POINTZM) */
            char ewkb[75];
            memcpy(ewkb, "01b90b0000", 10);  /* POINTZM little-endian */

            /* Double to hex for EWKB */
            double_to_hex(coords.x, ewkb + 10);
            double_to_hex(coords.y, ewkb + 26);
            double_to_hex(coords.z, ewkb + 42);
            double_to_hex(coords.m, ewkb + 58);

            /* Add to batch */
            if (batch_count > 0) appendStringInfoChar(&batch_sql, ',');

            appendStringInfo(&batch_sql, "('\\x%s'::bytea, %u, '\\x", hash_hex, codepoint);

            /* Add UTF-8 bytes as hex */
            static const char hex_chars[] = "0123456789abcdef";
            for (int i = 0; i < utf8_len; ++i) {
                uint8_t byte = utf8_bytes[i];
                appendStringInfo(&batch_sql, "%c%c", hex_chars[byte >> 4], hex_chars[byte & 0x0F]);
            }

            appendStringInfo(&batch_sql, "'::bytea, '\\x%s'::geometry, %lld, %lld)",
                ewkb, (long long)hilbert.lo, (long long)hilbert.hi);

            batch_count++;
        }

        /* Execute batch if we have items */
        if (batch_count > 0) {
            ret = SPI_execute(batch_sql.data, false, 0);
            if (ret == SPI_OK_INSERT) {
                processed_count += batch_count;
                if (processed_count % 10000 == 0) {
                    ereport(NOTICE, (errmsg("seed_atoms: processed %d/%d atoms", processed_count, TOTAL_CODEPOINTS)));
                }
            } else {
                ereport(WARNING, (errmsg("Failed batch insert for codepoints %u-%u", batch_start, codepoint - 1)));
            }
        }

        pfree(batch_sql.data);
    }

    ereport(NOTICE, (errmsg("seed_atoms: processed %d/%d atoms", processed_count, TOTAL_CODEPOINTS)));

    /* Create indexes after bulk insert */
    SPI_execute("CREATE INDEX IF NOT EXISTS idx_atom_codepoint ON atom(codepoint)", false, 0);
    SPI_execute("CREATE INDEX IF NOT EXISTS idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo)", false, 0);
    SPI_execute("CREATE INDEX IF NOT EXISTS idx_atom_geom ON atom USING GIST(geom)", false, 0);
    SPI_execute("ANALYZE atom", false, 0);

    /* Get final count */
    ret = SPI_execute("SELECT COUNT(*) FROM atom", true, 0);
    if (ret == SPI_OK_SELECT && SPI_processed > 0) {
        inserted = DatumGetInt64(SPI_getbinval(
            SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
    }

    SPI_finish();

    ereport(NOTICE, (errmsg("seeded %lld atoms successfully via extension", (long long)inserted)));
    PG_RETURN_INT64(inserted);
}
