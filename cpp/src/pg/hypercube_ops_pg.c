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

#include "pg_utils.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

#define MAX_EDGES 100000

/* ============================================================================
 * Data Structures (all local, no globals)
 * ============================================================================ */

typedef struct SemanticEdge {
    uint8   source[HASH_SIZE];
    uint8   target[HASH_SIZE];
    double  weight;
} SemanticEdge;

typedef struct WalkStep {
    uint8   id[HASH_SIZE];
    double  weight;
} WalkStep;

/* ============================================================================
 * Load Semantic Edges from Database (ONE query)
 * Uses simple approach - avoid array operations that might crash
 * ============================================================================ */

static SemanticEdge *load_all_edges(int *out_count)
{
    int ret;
    
    /* Query using unnest to get individual child elements */
    const char *query = 
        "SELECT c[1] as child1, c[2] as child2, "
        "COALESCE(ST_M(ST_StartPoint(geom)), 1.0) as weight "
        "FROM atom, LATERAL (SELECT children as c) sub "
        "WHERE depth = 1 AND atom_count = 2 "
        "AND array_length(children, 1) = 2 "
        "LIMIT 50000";
    
    ret = SPI_execute(query, true, 0);
    if (ret != SPI_OK_SELECT || SPI_processed == 0) {
        *out_count = 0;
        return NULL;
    }
    
    /* Allocate edge array - each row creates 2 bidirectional edges */
    uint64 capacity = SPI_processed * 2 + 1;
    SemanticEdge *edges = (SemanticEdge *)palloc0(sizeof(SemanticEdge) * capacity);
    int count = 0;
    
    for (uint64 i = 0; i < SPI_processed && count + 2 < (int)capacity; i++) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool isnull1, isnull2, isnull_w;
        
        Datum d1 = SPI_getbinval(tuple, tupdesc, 1, &isnull1);
        Datum d2 = SPI_getbinval(tuple, tupdesc, 2, &isnull2);
        Datum dw = SPI_getbinval(tuple, tupdesc, 3, &isnull_w);
        
        if (isnull1 || isnull2) continue;
        
        bytea *b1 = DatumGetByteaP(d1);
        bytea *b2 = DatumGetByteaP(d2);
        double weight = isnull_w ? 1.0 : DatumGetFloat8(dw);
        
        uint8 a[HASH_SIZE], b[HASH_SIZE];
        bytea_to_hash(b1, a);
        bytea_to_hash(b2, b);
        
        /* Add bidirectional edges */
        memcpy(edges[count].source, a, HASH_SIZE);
        memcpy(edges[count].target, b, HASH_SIZE);
        edges[count].weight = weight;
        count++;
        
        memcpy(edges[count].source, b, HASH_SIZE);
        memcpy(edges[count].target, a, HASH_SIZE);
        edges[count].weight = weight;
        count++;
    }
    
    *out_count = count;
    return edges;
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
    
    /* Simple visited tracking */
    uint8 (*visited)[HASH_SIZE] = (uint8 (*)[HASH_SIZE])palloc(sizeof(uint8[HASH_SIZE]) * (steps + 1));
    int visited_count = 0;
    
    uint8 current[HASH_SIZE];
    memcpy(current, seed, HASH_SIZE);
    
    /* Seed random */
    srand((unsigned int)time(NULL) ^ (unsigned int)((uintptr_t)seed));
    
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
        
        /* Build IN clause query */
        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        
        /* Query with array */
        StringInfoData query;
        initStringInfo(&query);
        appendStringInfoString(&query, 
            "SELECT id, depth, (value IS NOT NULL) as is_leaf, "
            "array_length(children, 1) as child_count, ST_X(centroid) "
            "FROM atom WHERE id = ANY(ARRAY[");
        
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
 * Uses existing semantic_reconstruct for now (optimization later)
 * ============================================================================ */

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
        
        /* Call semantic_reconstruct for each (TODO: batch optimize later) */
        if (SPI_connect() != SPI_OK_CONNECT) {
            ereport(ERROR, (errmsg("SPI_connect failed")));
        }
        
        for (int i = 0; i < nelems; i++) {
            char query[256];
            char hex[65];
            for (int j = 0; j < HASH_SIZE; j++) {
                snprintf(hex + j*2, 3, "%02x", state->ids[i * HASH_SIZE + j]);
            }
            
            snprintf(query, sizeof(query),
                "SELECT semantic_reconstruct('\\x%s'::bytea)", hex);
            
            int ret = SPI_execute(query, true, 1);
            if (ret == SPI_OK_SELECT && SPI_processed > 0) {
                bool isnull;
                Datum d = SPI_getbinval(SPI_tuptable->vals[0], 
                                        SPI_tuptable->tupdesc, 1, &isnull);
                if (!isnull) {
                    /* Copy to multi_call context */
                    text *t = DatumGetTextP(d);
                    state->texts[i] = (text *)MemoryContextAlloc(
                        funcctx->multi_call_memory_ctx,
                        VARSIZE(t)
                    );
                    memcpy(state->texts[i], t, VARSIZE(t));
                }
            }
        }
        
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
Datum hypercube_knn_batch(PG_FUNCTION_ARGS);

PG_FUNCTION_INFO_V1(hypercube_attention);
Datum hypercube_attention(PG_FUNCTION_ARGS);
