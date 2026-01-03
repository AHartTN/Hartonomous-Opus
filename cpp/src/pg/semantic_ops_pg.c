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
#include "lib/stringinfo.h"

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
        TupleDesc tupdesc;
        
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR,
                    (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                     errmsg("function returning record called in context that cannot accept type record")));
        
        state = (TraverseState *)palloc(sizeof(TraverseState));
        state->capacity = 16;
        state->nodes = (TraverseNode *)palloc(state->capacity * sizeof(TraverseNode));
        state->count = 0;
        state->current = 0;
        state->tupdesc = BlessTupleDesc(tupdesc);
        
        funcctx->user_fctx = state;
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (TraverseState *)funcctx->user_fctx;
    
    /* Just return empty result */
    SRF_RETURN_DONE(funcctx);
}

/* ============================================================================
 * semantic_reconstruct: Text reconstruction from composition
 * 
 * Simple iterative approach using SPI - fetch children on demand.
 * Uses a stack to avoid recursion and palloc in caller's context.
 * ============================================================================ */

PG_FUNCTION_INFO_V1(semantic_reconstruct);
Datum semantic_reconstruct(PG_FUNCTION_ARGS)
{
    bytea *root_arg = PG_GETARG_BYTEA_PP(0);
    if (VARSIZE_ANY_EXHDR(root_arg) < 32) {
        PG_RETURN_TEXT_P(cstring_to_text(""));
    }
    
    /* Copy root hash to local buffer */
    uint8 root_hash[32];
    memcpy(root_hash, VARDATA_ANY(root_arg), 32);
    
    /* Save caller's memory context */
    MemoryContext caller_ctx = CurrentMemoryContext;
    
    /* 
     * Allocate result buffer in CALLER context BEFORE SPI_connect.
     * This ensures the buffer survives SPI_finish.
     */
    #define RESULT_INIT_SIZE 4096
    #define MAX_RESULT_SIZE (64 * 1024 * 1024)  /* 64MB max */
    
    char *result_buf = (char *)palloc(RESULT_INIT_SIZE);
    int result_len = 0;
    int result_cap = RESULT_INIT_SIZE;
    
    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("SPI_connect failed")));
    
    /* Stack for DFS traversal - array of 32-byte hashes */
    #define MAX_STACK_DEPTH 10000
    uint8 *stack = (uint8 *)palloc(MAX_STACK_DEPTH * 32);
    int stack_top = 0;
    
    /* Push root */
    memcpy(stack, root_hash, 32);
    stack_top = 1;
    
    char query[256];
    
    while (stack_top > 0) {
        /* Pop from stack */
        stack_top--;
        uint8 *current_id = stack + stack_top * 32;
        
        /* Build hex string for query */
        char hex[65];
        for (int i = 0; i < 32; i++)
            snprintf(hex + i*2, 3, "%02x", current_id[i]);
        
        /* Query this node */
        snprintf(query, sizeof(query),
            "SELECT value, children FROM atom WHERE id = '\\x%s'", hex);
        
        int ret = SPI_execute(query, true, 1);
        if (ret != SPI_OK_SELECT || SPI_processed == 0)
            continue;
        
        HeapTuple tuple = SPI_tuptable->vals[0];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool value_null, children_null;
        
        Datum value_datum = SPI_getbinval(tuple, tupdesc, 1, &value_null);
        Datum children_datum = SPI_getbinval(tuple, tupdesc, 2, &children_null);
        
        if (!value_null) {
            /* Leaf node - append value to result buffer in caller context */
            bytea *val = DatumGetByteaPP(value_datum);
            int len = VARSIZE_ANY_EXHDR(val);
            
            /* Check capacity, grow if needed */
            while (result_len + len > result_cap) {
                if (result_cap >= MAX_RESULT_SIZE) {
                    SPI_freetuptable(SPI_tuptable);
                    SPI_finish();
                    ereport(ERROR, (errmsg("semantic_reconstruct: result too large")));
                }
                
                /* Grow in caller context */
                MemoryContext old_ctx = MemoryContextSwitchTo(caller_ctx);
                int new_cap = result_cap * 2;
                char *new_buf = (char *)palloc(new_cap);
                memcpy(new_buf, result_buf, result_len);
                pfree(result_buf);
                result_buf = new_buf;
                result_cap = new_cap;
                MemoryContextSwitchTo(old_ctx);
            }
            
            memcpy(result_buf + result_len, VARDATA_ANY(val), len);
            result_len += len;
        }
        else if (!children_null) {
            /* Composition node - push children in REVERSE order for correct DFS */
            ArrayType *arr = DatumGetArrayTypeP(children_datum);
            int nelems = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
            
            if (nelems > 0 && stack_top + nelems <= MAX_STACK_DEPTH) {
                Datum *elems;
                bool *nulls;
                int n;
                deconstruct_array(arr, BYTEAOID, -1, false, TYPALIGN_INT,
                                 &elems, &nulls, &n);
                
                /* Push in reverse order so first child is processed first */
                for (int i = n - 1; i >= 0; i--) {
                    if (!nulls[i]) {
                        bytea *child = DatumGetByteaPP(elems[i]);
                        if (VARSIZE_ANY_EXHDR(child) >= 32) {
                            memcpy(stack + stack_top * 32, VARDATA_ANY(child), 32);
                            stack_top++;
                        }
                    }
                }
            }
        }
        
        /* Free SPI tuple table for next iteration */
        SPI_freetuptable(SPI_tuptable);
    }
    
    pfree(stack);
    SPI_finish();
    
    /* Now we're in caller context with result_buf containing the data */
    if (result_len > 0) {
        text *output = cstring_to_text_with_len(result_buf, result_len);
        pfree(result_buf);
        PG_RETURN_TEXT_P(output);
    }
    
    pfree(result_buf);
    PG_RETURN_TEXT_P(cstring_to_text(""));
}
