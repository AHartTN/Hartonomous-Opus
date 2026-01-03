/**
 * semantic_ops_pg.c - PostgreSQL Extension for Semantic Operations
 * 
 * Pure C implementation that includes PostgreSQL headers and calls the
 * hypercube C API (hypercube_c.h). Provides high-performance semantic
 * query operations.
 * 
 * Functions:
 *   - semantic_traverse(root_id, max_depth) -> SETOF (id, depth, ordinal, path_len)
 *   - semantic_reconstruct(root_id) -> text
 *   - semantic_hilbert_distance_128(lo1, hi1, lo2, hi2) -> (lo, hi)
 *   - semantic_4d_distance(x1,y1,z1,m1, x2,y2,z2,m2) -> float8
 *   - semantic_centroid_4d(x[], y[], z[], m[]) -> (x, y, z, m)
 *   - semantic_coords_from_hilbert(lo, hi) -> (x, y, z, m)
 *   - semantic_hilbert_from_coords(x, y, z, m) -> (lo, hi)
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

#include "hypercube_c.h"
#include "pg_utils.h"

#include <string.h>
#include <math.h>

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

/* ============================================================================
 * semantic_hilbert_distance_128: 128-bit Hilbert distance
 * ============================================================================ */

PG_FUNCTION_INFO_V1(semantic_hilbert_distance_128);
Datum semantic_hilbert_distance_128(PG_FUNCTION_ARGS)
{
    int64 lo1 = PG_GETARG_INT64(0);
    int64 hi1 = PG_GETARG_INT64(1);
    int64 lo2 = PG_GETARG_INT64(2);
    int64 hi2 = PG_GETARG_INT64(3);
    
    hc_hilbert_t a = {(uint64)lo1, (uint64)hi1};
    hc_hilbert_t b = {(uint64)lo2, (uint64)hi2};
    
    hc_hilbert_t dist = hc_hilbert_distance(a, b);
    
    TupleDesc tupdesc;
    Datum values[2];
    bool nulls[2] = {false, false};
    HeapTuple tuple;
    
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    
    tupdesc = BlessTupleDesc(tupdesc);
    
    values[0] = Int64GetDatum((int64)dist.lo);
    values[1] = Int64GetDatum((int64)dist.hi);
    
    tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/* ============================================================================
 * semantic_4d_distance: 4D Euclidean distance
 * ============================================================================ */

PG_FUNCTION_INFO_V1(semantic_4d_distance);
Datum semantic_4d_distance(PG_FUNCTION_ARGS)
{
    double x1 = PG_GETARG_FLOAT8(0);
    double y1 = PG_GETARG_FLOAT8(1);
    double z1 = PG_GETARG_FLOAT8(2);
    double m1 = PG_GETARG_FLOAT8(3);
    double x2 = PG_GETARG_FLOAT8(4);
    double y2 = PG_GETARG_FLOAT8(5);
    double z2 = PG_GETARG_FLOAT8(6);
    double m2 = PG_GETARG_FLOAT8(7);
    
    hc_point4d_t a = {(uint32)x1, (uint32)y1, (uint32)z1, (uint32)m1};
    hc_point4d_t b = {(uint32)x2, (uint32)y2, (uint32)z2, (uint32)m2};
    
    double dist = hc_euclidean_distance(a, b);
    
    PG_RETURN_FLOAT8(dist);
}

/* ============================================================================
 * semantic_centroid_4d: 4D centroid calculation
 * ============================================================================ */

PG_FUNCTION_INFO_V1(semantic_centroid_4d);
Datum semantic_centroid_4d(PG_FUNCTION_ARGS)
{
    ArrayType *x_arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *y_arr = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *z_arr = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *m_arr = PG_GETARG_ARRAYTYPE_P(3);
    
    int n = ArrayGetNItems(ARR_NDIM(x_arr), ARR_DIMS(x_arr));
    if (n == 0)
        PG_RETURN_NULL();
    
    /* Extract arrays as float8 */
    Datum *x_elems, *y_elems, *z_elems, *m_elems;
    bool *x_nulls, *y_nulls, *z_nulls, *m_nulls;
    int nx, ny, nz, nm;
    
    deconstruct_array(x_arr, FLOAT8OID, 8, true, TYPALIGN_DOUBLE, &x_elems, &x_nulls, &nx);
    deconstruct_array(y_arr, FLOAT8OID, 8, true, TYPALIGN_DOUBLE, &y_elems, &y_nulls, &ny);
    deconstruct_array(z_arr, FLOAT8OID, 8, true, TYPALIGN_DOUBLE, &z_elems, &z_nulls, &nz);
    deconstruct_array(m_arr, FLOAT8OID, 8, true, TYPALIGN_DOUBLE, &m_elems, &m_nulls, &nm);
    
    /* Allocate points and weights */
    hc_point4d_t *points = (hc_point4d_t *)palloc(n * sizeof(hc_point4d_t));
    int count = 0;
    
    for (int i = 0; i < n && i < nx && i < ny && i < nz && i < nm; i++)
    {
        if ((x_nulls && x_nulls[i]) || (y_nulls && y_nulls[i]) ||
            (z_nulls && z_nulls[i]) || (m_nulls && m_nulls[i]))
            continue;
        
        points[count].x = (uint32)DatumGetFloat8(x_elems[i]);
        points[count].y = (uint32)DatumGetFloat8(y_elems[i]);
        points[count].z = (uint32)DatumGetFloat8(z_elems[i]);
        points[count].m = (uint32)DatumGetFloat8(m_elems[i]);
        count++;
    }
    
    if (count == 0)
    {
        pfree(points);
        PG_RETURN_NULL();
    }
    
    hc_point4d_t centroid = hc_centroid(points, count);
    pfree(points);
    
    TupleDesc tupdesc;
    Datum values[4];
    bool nulls[4] = {false, false, false, false};
    HeapTuple tuple;
    
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    
    tupdesc = BlessTupleDesc(tupdesc);
    
    values[0] = Float8GetDatum((double)centroid.x);
    values[1] = Float8GetDatum((double)centroid.y);
    values[2] = Float8GetDatum((double)centroid.z);
    values[3] = Float8GetDatum((double)centroid.m);
    
    tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/* ============================================================================
 * semantic_coords_from_hilbert: Inverse Hilbert mapping
 * ============================================================================ */

PG_FUNCTION_INFO_V1(semantic_coords_from_hilbert);
Datum semantic_coords_from_hilbert(PG_FUNCTION_ARGS)
{
    int64 lo = PG_GETARG_INT64(0);
    int64 hi = PG_GETARG_INT64(1);
    
    hc_hilbert_t idx = {(uint64)lo, (uint64)hi};
    hc_point4d_t point = hc_hilbert_to_coords(idx);
    
    TupleDesc tupdesc;
    Datum values[4];
    bool nulls[4] = {false, false, false, false};
    HeapTuple tuple;
    
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    
    tupdesc = BlessTupleDesc(tupdesc);
    
    values[0] = Float8GetDatum((double)point.x);
    values[1] = Float8GetDatum((double)point.y);
    values[2] = Float8GetDatum((double)point.z);
    values[3] = Float8GetDatum((double)point.m);
    
    tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/* ============================================================================
 * semantic_hilbert_from_coords: Forward Hilbert mapping
 * ============================================================================ */

PG_FUNCTION_INFO_V1(semantic_hilbert_from_coords);
Datum semantic_hilbert_from_coords(PG_FUNCTION_ARGS)
{
    double x = PG_GETARG_FLOAT8(0);
    double y = PG_GETARG_FLOAT8(1);
    double z = PG_GETARG_FLOAT8(2);
    double m = PG_GETARG_FLOAT8(3);
    
    hc_point4d_t point = {(uint32)x, (uint32)y, (uint32)z, (uint32)m};
    hc_hilbert_t idx = hc_coords_to_hilbert(point);
    
    TupleDesc tupdesc;
    Datum values[2];
    bool nulls[2] = {false, false};
    HeapTuple tuple;
    
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    
    tupdesc = BlessTupleDesc(tupdesc);
    
    values[0] = Int64GetDatum((int64)idx.lo);
    values[1] = Int64GetDatum((int64)idx.hi);
    
    tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/* ============================================================================
 * semantic_traverse: DFS traversal of composition DAG
 * This requires SPI for database access - keeping it in C
 * ============================================================================ */

typedef struct {
    uint8 id[32];
    int depth;
    int ordinal;
    int path_len;
} TraverseNode;

typedef struct {
    TraverseNode *nodes;
    int count;
    int capacity;
    int current;
    TupleDesc tupdesc;
} TraverseState;

static void traverse_push(TraverseState *state, TraverseNode node)
{
    if (state->count >= state->capacity) {
        state->capacity *= 2;
        state->nodes = (TraverseNode *)repalloc(state->nodes, 
                                                 state->capacity * sizeof(TraverseNode));
    }
    state->nodes[state->count++] = node;
}

PG_FUNCTION_INFO_V1(semantic_traverse);
Datum semantic_traverse(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    TraverseState *state;
    
    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldcontext;
        bytea *root_id = PG_GETARG_BYTEA_PP(0);
        int max_depth = PG_NARGS() > 1 ? PG_GETARG_INT32(1) : 100;
        TupleDesc tupdesc;
        
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR,
                    (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                     errmsg("function returning record called in context that cannot accept type record")));
        
        state = (TraverseState *)palloc(sizeof(TraverseState));
        state->capacity = 1024;
        state->nodes = (TraverseNode *)palloc(state->capacity * sizeof(TraverseNode));
        state->count = 0;
        state->current = 0;
        state->tupdesc = BlessTupleDesc(tupdesc);
        
        /* Connect to SPI */
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed")));
        
        /* BFS traversal using queue (simplified - using array as queue) */
        TraverseNode *queue = (TraverseNode *)palloc(10000 * sizeof(TraverseNode));
        int queue_head = 0, queue_tail = 0;
        
        /* Hash set for visited - simplified using linear search for now */
        uint8 (*visited)[32] = (uint8 (*)[32])palloc(10000 * 32);
        int visited_count = 0;
        
        /* Add root */
        TraverseNode root;
        bytea_to_hash(root_id, root.id);
        root.depth = 0;
        root.ordinal = 0;
        root.path_len = 0;
        queue[queue_tail++] = root;
        
        while (queue_head < queue_tail)
        {
            TraverseNode node = queue[queue_head++];
            
            /* Check if visited */
            bool found = false;
            for (int i = 0; i < visited_count; i++) {
                if (memcmp(visited[i], node.id, 32) == 0) {
                    found = true;
                    break;
                }
            }
            if (found) continue;
            
            /* Mark visited */
            memcpy(visited[visited_count++], node.id, 32);
            
            /* Add to results */
            traverse_push(state, node);
            
            if (node.depth >= max_depth) continue;
            
            /* Query for children */
            char query[256];
            char id_hex[65];
            for (int i = 0; i < 32; i++)
                snprintf(id_hex + i*2, 3, "%02x", node.id[i]);
            
            snprintf(query, sizeof(query),
                "SELECT unnest(children), generate_subscripts(children, 1) - 1 "
                "FROM atom WHERE id = '\\x%s' AND children IS NOT NULL",
                id_hex);
            
            int ret = SPI_execute(query, true, 0);
            if (ret == SPI_OK_SELECT && SPI_processed > 0)
            {
                for (uint64 i = 0; i < SPI_processed && queue_tail < 10000; i++)
                {
                    HeapTuple tuple = SPI_tuptable->vals[i];
                    
                    bool isnull;
                    Datum child_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &isnull);
                    if (isnull) continue;
                    
                    Datum ord_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 2, &isnull);
                    int ordinal = isnull ? 0 : DatumGetInt32(ord_datum);
                    
                    TraverseNode child;
                    bytea *child_bytea = DatumGetByteaP(child_datum);
                    bytea_to_hash(child_bytea, child.id);
                    child.depth = node.depth + 1;
                    child.ordinal = ordinal;
                    child.path_len = node.path_len + 1;
                    
                    queue[queue_tail++] = child;
                }
            }
        }
        
        SPI_finish();
        pfree(queue);
        pfree(visited);
        
        funcctx->user_fctx = state;
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (TraverseState *)funcctx->user_fctx;
    
    if (state->current < state->count)
    {
        TraverseNode *node = &state->nodes[state->current++];
        
        Datum values[4];
        bool nulls[4] = {false, false, false, false};
        HeapTuple tuple;
        
        values[0] = PointerGetDatum(hash_to_bytea(node->id));
        values[1] = Int32GetDatum(node->depth);
        values[2] = Int32GetDatum(node->ordinal);
        values[3] = Int32GetDatum(node->path_len);
        
        tuple = heap_form_tuple(state->tupdesc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    SRF_RETURN_DONE(funcctx);
}

/* ============================================================================
 * semantic_reconstruct: Text reconstruction from composition
 * ============================================================================ */

PG_FUNCTION_INFO_V1(semantic_reconstruct);
Datum semantic_reconstruct(PG_FUNCTION_ARGS)
{
    bytea *root_id = PG_GETARG_BYTEA_PP(0);
    
    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("SPI_connect failed")));
    
    /* Stack for DFS */
    typedef struct {
        uint8 id[32];
        int child_idx;
        uint8 *child_ids;  /* Array of 32-byte hashes */
        int num_children;
    } StackEntry;
    
    StackEntry *stack = (StackEntry *)palloc(1000 * sizeof(StackEntry));
    int stack_top = 0;
    
    /* Result buffer */
    int result_capacity = 4096;
    char *result = (char *)palloc(result_capacity);
    int result_len = 0;
    
    /* Start with root */
    bytea_to_hash(root_id, stack[0].id);
    stack[0].child_idx = -1;
    stack[0].child_ids = NULL;
    stack[0].num_children = 0;
    stack_top = 1;
    
    while (stack_top > 0)
    {
        StackEntry *current = &stack[stack_top - 1];
        
        /* First visit - query for children or value */
        if (current->child_idx == -1)
        {
            char id_hex[65];
            for (int i = 0; i < 32; i++)
                snprintf(id_hex + i*2, 3, "%02x", current->id[i]);
            
            char query[256];
            snprintf(query, sizeof(query),
                "SELECT value, children FROM atom WHERE id = '\\x%s'", id_hex);
            
            int ret = SPI_execute(query, true, 1);
            if (ret == SPI_OK_SELECT && SPI_processed > 0)
            {
                HeapTuple tuple = SPI_tuptable->vals[0];
                
                bool value_null, children_null;
                Datum value_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &value_null);
                Datum children_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 2, &children_null);
                
                if (!value_null)
                {
                    /* Leaf node - append value to result */
                    bytea *value = DatumGetByteaP(value_datum);
                    int vlen = VARSIZE_ANY_EXHDR(value);
                    
                    /* Grow result buffer if needed */
                    while (result_len + vlen >= result_capacity) {
                        result_capacity *= 2;
                        result = (char *)repalloc(result, result_capacity);
                    }
                    
                    memcpy(result + result_len, VARDATA_ANY(value), vlen);
                    result_len += vlen;
                    stack_top--;
                    continue;
                }
                
                if (!children_null)
                {
                    /* Composition - get children array */
                    ArrayType *arr = DatumGetArrayTypeP(children_datum);
                    int n = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
                    
                    current->child_idx = 0;
                    current->num_children = 0;
                    current->child_ids = (uint8 *)MemoryContextAlloc(
                        fcinfo->flinfo->fn_mcxt, n * 32);
                    
                    Datum *elems;
                    bool *nulls_arr;
                    int nelems;
                    deconstruct_array(arr, BYTEAOID, -1, false, TYPALIGN_INT,
                                     &elems, &nulls_arr, &nelems);
                    
                    for (int i = 0; i < nelems; i++)
                    {
                        if (!nulls_arr[i]) {
                            bytea *child = DatumGetByteaP(elems[i]);
                            bytea_to_hash(child, current->child_ids + current->num_children * 32);
                            current->num_children++;
                        }
                    }
                }
                else
                {
                    /* No value, no children - skip */
                    stack_top--;
                    continue;
                }
            }
            else
            {
                /* Not found */
                stack_top--;
                continue;
            }
        }
        
        /* Process next child */
        if (current->child_idx < current->num_children && stack_top < 999)
        {
            int idx = current->child_idx++;
            StackEntry *child = &stack[stack_top++];
            memcpy(child->id, current->child_ids + idx * 32, 32);
            child->child_idx = -1;
            child->child_ids = NULL;
            child->num_children = 0;
        }
        else
        {
            /* All children processed */
            if (current->child_ids)
                pfree(current->child_ids);
            stack_top--;
        }
    }
    
    SPI_finish();
    pfree(stack);
    
    /* Return as text */
    text *result_text = cstring_to_text_with_len(result, result_len);
    pfree(result);
    
    PG_RETURN_TEXT_P(result_text);
}
