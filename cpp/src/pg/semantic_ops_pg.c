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

        /* Get root hash argument */
        bytea *root_arg = PG_GETARG_BYTEA_PP(0);
        if (VARSIZE_ANY_EXHDR(root_arg) < 32) {
            SRF_RETURN_DONE(funcctx);
        }

        char hex[65];
        uint8 *hash_bytes = (uint8 *)VARDATA_ANY(root_arg);
        for (int i = 0; i < 32; i++)
            snprintf(hex + i*2, 3, "%02x", hash_bytes[i]);

        /* Load entire subtree in one query instead of row-by-row */
        state = (TraverseState *)palloc(sizeof(TraverseState));
        state->capacity = 256;
        state->nodes = (TraverseNode *)palloc(state->capacity * sizeof(TraverseNode));
        state->count = 0;
        state->current = 0;
        state->tupdesc = BlessTupleDesc(tupdesc);

        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed")));

        /* Single query to load all reachable nodes with their depths and ordinals */
        char query[4096];
        snprintf(query, sizeof(query),
            "WITH RECURSIVE subtree AS ("
            "  SELECT id, 0 as depth, 0 as ordinal, ARRAY[id] as path "
            "  FROM composition WHERE id = '\\x%s' "
            "  UNION ALL "
            "  SELECT c.id, s.depth + 1, cc.ordinal, s.path || c.id "
            "  FROM subtree s "
            "  JOIN composition_child cc ON cc.composition_id = s.id "
            "  JOIN composition c ON cc.child_type = 'C' AND c.id = cc.child_id "
            ") "
            "SELECT id, depth, ordinal "
            "FROM subtree "
            "ORDER BY depth, ordinal",
            hex);

        int ret = SPI_execute(query, true, 0); /* Get all rows */
        if (ret != SPI_OK_SELECT) {
            SPI_finish();
            ereport(ERROR, (errmsg("SPI_execute failed")));
        }

        /* Process all results into state */
        for (uint64 proc = 0; proc < SPI_processed; proc++) {
            bool isnull_id, isnull_depth, isnull_ordinal;
            Datum id_val = SPI_getbinval(SPI_tuptable->vals[proc], SPI_tuptable->tupdesc, 1, &isnull_id);
            Datum depth_val = SPI_getbinval(SPI_tuptable->vals[proc], SPI_tuptable->tupdesc, 2, &isnull_depth);
            Datum ordinal_val = SPI_getbinval(SPI_tuptable->vals[proc], SPI_tuptable->tupdesc, 3, &isnull_ordinal);

            if (!isnull_id && !isnull_depth && !isnull_ordinal) {
                TraverseNode node;
                bytea *id_bytes = DatumGetByteaPP(id_val);
                memcpy(node.id, VARDATA_ANY(id_bytes), 32);
                node.depth = DatumGetInt32(depth_val);
                node.ordinal = DatumGetInt32(ordinal_val);
                node.path_len = node.depth; /* Approximate path length */

                traverse_push(state, node);
            }
        }

        SPI_freetuptable(SPI_tuptable);
        SPI_finish();

        funcctx->user_fctx = state;
        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    state = (TraverseState *)funcctx->user_fctx;

    /* Return next node */
    if (state->current < state->count) {
        TraverseNode *node = &state->nodes[state->current++];

        Datum values[4];
        bool nulls[4] = {false, false, false, false};
        HeapTuple tuple;

        values[0] = PointerGetDatum(cstring_to_text_with_len((char*)node->id, 32));
        values[1] = Int32GetDatum(node->depth);
        values[2] = Int32GetDatum(node->ordinal);
        values[3] = Int32GetDatum(node->path_len);

        tuple = heap_form_tuple(state->tupdesc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }

    /* All done */
    SRF_RETURN_DONE(funcctx);
}

/* ============================================================================
 * semantic_reconstruct: Text reconstruction from composition
 * 
 * Uses SQL CTE for safety - avoids SPI recursion issues that crash PostgreSQL.
 * ============================================================================ */

PG_FUNCTION_INFO_V1(semantic_reconstruct);
Datum semantic_reconstruct(PG_FUNCTION_ARGS)
{
    bytea *root_arg = PG_GETARG_BYTEA_PP(0);
    if (VARSIZE_ANY_EXHDR(root_arg) < 32) {
        PG_RETURN_TEXT_P(cstring_to_text(""));
    }
    
    char hex[65];
    uint8 *hash_bytes = (uint8 *)VARDATA_ANY(root_arg);
    for (int i = 0; i < 32; i++)
        snprintf(hex + i*2, 3, "%02x", hash_bytes[i]);
    
    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("SPI_connect failed")));
    
    /* Use a CTE with composition_child table for traversal */
    char query[2048];
    snprintf(query, sizeof(query),
        "WITH RECURSIVE tree AS ("
        "  SELECT id, value, 1 as ord, ARRAY[1] as path "
        "  FROM atom WHERE id = '\\x%s' "
        "  UNION ALL "
        "  SELECT COALESCE(a.id, c2.id), COALESCE(a.value, NULL), cc.ordinal, t.path || cc.ordinal "
        "  FROM tree t "
        "  JOIN composition_child cc ON cc.composition_id = t.id "
        "  LEFT JOIN atom a ON cc.child_type = 'A' AND a.id = cc.child_id "
        "  LEFT JOIN composition c2 ON cc.child_type = 'C' AND c2.id = cc.child_id "
        "  WHERE t.value IS NULL "
        ") "
        "SELECT convert_from(string_agg(value, ''::bytea ORDER BY path), 'UTF8') "
        "FROM tree WHERE value IS NOT NULL",
        hex);
    
    int ret = SPI_execute(query, true, 1);
    
    text *result;
    if (ret == SPI_OK_SELECT && SPI_processed > 0) {
        bool isnull;
        Datum val = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
        if (!isnull) {
            text *t = DatumGetTextPP(val);
            result = (text *)SPI_palloc(VARSIZE_ANY(t));
            memcpy(result, t, VARSIZE_ANY(t));
        } else {
            result = cstring_to_text("");
        }
        SPI_freetuptable(SPI_tuptable);
    } else {
        result = cstring_to_text("");
    }
    
    SPI_finish();
    PG_RETURN_TEXT_P(result);
}
