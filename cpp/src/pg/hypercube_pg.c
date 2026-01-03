/**
 * hypercube_pg.c - PostgreSQL Extension for Hypercube
 * 
 * Pure C implementation that includes PostgreSQL headers and calls the
 * hypercube C API (hypercube_c.h). This avoids PostgreSQL header
 * incompatibilities with modern C++ compilers.
 * 
 * Functions:
 *   - hypercube_coords_to_hilbert(x, y, z, m) -> (hilbert_lo, hilbert_hi)
 *   - hypercube_hilbert_to_coords(lo, hi) -> (x, y, z, m)
 *   - hypercube_blake3(data) -> bytea
 *   - hypercube_blake3_codepoint(cp) -> bytea
 *   - hypercube_map_codepoint(cp) -> (x, y, z, m, hilbert_lo, hilbert_hi, hash, category)
 *   - hypercube_categorize(cp) -> text
 *   - hypercube_seed_atoms() -> SETOF (codepoint, x, y, z, m, hilbert_lo, hilbert_hi, hash, category)
 *   - hypercube_centroid(x[], y[], z[], m[]) -> (x, y, z, m)
 *   - hypercube_is_on_surface(x, y, z, m) -> bool
 *   - hypercube_hilbert_distance(lo1, hi1, lo2, hi2) -> (lo, hi)
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"

#include "hypercube_c.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

/* ============================================================================
 * Hilbert Curve Functions
 * ============================================================================ */

PG_FUNCTION_INFO_V1(hypercube_coords_to_hilbert);
Datum hypercube_coords_to_hilbert(PG_FUNCTION_ARGS)
{
    int64 x = PG_GETARG_INT64(0);
    int64 y = PG_GETARG_INT64(1);
    int64 z = PG_GETARG_INT64(2);
    int64 m = PG_GETARG_INT64(3);
    
    hc_point4d_t point;
    point.x = (uint32)x;
    point.y = (uint32)y;
    point.z = (uint32)z;
    point.m = (uint32)m;
    
    hc_hilbert_t hilbert = hc_coords_to_hilbert(point);
    
    TupleDesc tupdesc;
    Datum values[2];
    bool nulls[2] = {false, false};
    HeapTuple tuple;
    
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    
    tupdesc = BlessTupleDesc(tupdesc);
    
    values[0] = Int64GetDatum((int64)hilbert.lo);
    values[1] = Int64GetDatum((int64)hilbert.hi);
    
    tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

PG_FUNCTION_INFO_V1(hypercube_hilbert_to_coords);
Datum hypercube_hilbert_to_coords(PG_FUNCTION_ARGS)
{
    int64 lo = PG_GETARG_INT64(0);
    int64 hi = PG_GETARG_INT64(1);
    
    hc_hilbert_t hilbert;
    hilbert.lo = (uint64)lo;
    hilbert.hi = (uint64)hi;
    
    hc_point4d_t point = hc_hilbert_to_coords(hilbert);
    
    TupleDesc tupdesc;
    Datum values[4];
    bool nulls[4] = {false, false, false, false};
    HeapTuple tuple;
    
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    
    tupdesc = BlessTupleDesc(tupdesc);
    
    values[0] = Int64GetDatum((int64)point.x);
    values[1] = Int64GetDatum((int64)point.y);
    values[2] = Int64GetDatum((int64)point.z);
    values[3] = Int64GetDatum((int64)point.m);
    
    tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

PG_FUNCTION_INFO_V1(hypercube_hilbert_distance);
Datum hypercube_hilbert_distance(PG_FUNCTION_ARGS)
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
 * BLAKE3 Hashing Functions
 * ============================================================================ */

PG_FUNCTION_INFO_V1(hypercube_blake3);
Datum hypercube_blake3(PG_FUNCTION_ARGS)
{
    bytea *input = PG_GETARG_BYTEA_PP(0);
    size_t len = VARSIZE_ANY_EXHDR(input);
    const uint8 *data = (const uint8 *)VARDATA_ANY(input);
    
    hc_hash_t hash = hc_blake3(data, len);
    
    bytea *result = (bytea *)palloc(VARHDRSZ + HC_HASH_SIZE);
    SET_VARSIZE(result, VARHDRSZ + HC_HASH_SIZE);
    memcpy(VARDATA(result), hash.bytes, HC_HASH_SIZE);
    
    PG_RETURN_BYTEA_P(result);
}

PG_FUNCTION_INFO_V1(hypercube_blake3_codepoint);
Datum hypercube_blake3_codepoint(PG_FUNCTION_ARGS)
{
    int32 codepoint = PG_GETARG_INT32(0);
    
    hc_hash_t hash = hc_blake3_codepoint((uint32)codepoint);
    
    bytea *result = (bytea *)palloc(VARHDRSZ + HC_HASH_SIZE);
    SET_VARSIZE(result, VARHDRSZ + HC_HASH_SIZE);
    memcpy(VARDATA(result), hash.bytes, HC_HASH_SIZE);
    
    PG_RETURN_BYTEA_P(result);
}

/* ============================================================================
 * Coordinate Mapping Functions
 * ============================================================================ */

PG_FUNCTION_INFO_V1(hypercube_map_codepoint);
Datum hypercube_map_codepoint(PG_FUNCTION_ARGS)
{
    int32 codepoint = PG_GETARG_INT32(0);
    
    hc_atom_t atom = hc_map_atom((uint32)codepoint);
    
    TupleDesc tupdesc;
    Datum values[8];
    bool nulls[8] = {false, false, false, false, false, false, false, false};
    HeapTuple tuple;
    bytea *hash_bytea;
    
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("function returning record called in context that cannot accept type record")));
    
    tupdesc = BlessTupleDesc(tupdesc);
    
    values[0] = Int64GetDatum((int64)atom.coords.x);
    values[1] = Int64GetDatum((int64)atom.coords.y);
    values[2] = Int64GetDatum((int64)atom.coords.z);
    values[3] = Int64GetDatum((int64)atom.coords.m);
    values[4] = Int64GetDatum((int64)atom.hilbert.lo);
    values[5] = Int64GetDatum((int64)atom.hilbert.hi);
    
    hash_bytea = (bytea *)palloc(VARHDRSZ + HC_HASH_SIZE);
    SET_VARSIZE(hash_bytea, VARHDRSZ + HC_HASH_SIZE);
    memcpy(VARDATA(hash_bytea), atom.hash.bytes, HC_HASH_SIZE);
    values[6] = PointerGetDatum(hash_bytea);
    
    values[7] = CStringGetTextDatum(hc_category_name(atom.category));
    
    tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

PG_FUNCTION_INFO_V1(hypercube_categorize);
Datum hypercube_categorize(PG_FUNCTION_ARGS)
{
    int32 codepoint = PG_GETARG_INT32(0);
    hc_category_t cat = hc_categorize((uint32)codepoint);
    PG_RETURN_TEXT_P(cstring_to_text(hc_category_name(cat)));
}

PG_FUNCTION_INFO_V1(hypercube_is_on_surface);
Datum hypercube_is_on_surface(PG_FUNCTION_ARGS)
{
    int64 x = PG_GETARG_INT64(0);
    int64 y = PG_GETARG_INT64(1);
    int64 z = PG_GETARG_INT64(2);
    int64 m = PG_GETARG_INT64(3);
    
    hc_point4d_t point;
    point.x = (uint32)x;
    point.y = (uint32)y;
    point.z = (uint32)z;
    point.m = (uint32)m;
    
    PG_RETURN_BOOL(hc_is_on_surface(point));
}

/* ============================================================================
 * Set-Returning Function: seed_atoms
 * ============================================================================ */

typedef struct {
    uint32 current_codepoint;
    TupleDesc tupdesc;
} SeedAtomsState;

PG_FUNCTION_INFO_V1(hypercube_seed_atoms);
Datum hypercube_seed_atoms(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    SeedAtomsState *state;
    
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
        
        state = (SeedAtomsState *)palloc(sizeof(SeedAtomsState));
        state->current_codepoint = UINT32_MAX;  /* Will start at 0 */
        state->tupdesc = BlessTupleDesc(tupdesc);
        
        funcctx->user_fctx = state;
        funcctx->max_calls = hc_valid_codepoint_count();
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (SeedAtomsState *)funcctx->user_fctx;
    
    /* Get next valid codepoint */
    state->current_codepoint = hc_next_codepoint(state->current_codepoint);
    
    if (state->current_codepoint != UINT32_MAX)
    {
        hc_atom_t atom = hc_map_atom(state->current_codepoint);
        
        Datum values[9];
        bool nulls[9] = {false, false, false, false, false, false, false, false, false};
        HeapTuple tuple;
        bytea *hash_bytea;
        
        values[0] = Int32GetDatum((int32)atom.codepoint);
        values[1] = Int64GetDatum((int64)atom.coords.x);
        values[2] = Int64GetDatum((int64)atom.coords.y);
        values[3] = Int64GetDatum((int64)atom.coords.z);
        values[4] = Int64GetDatum((int64)atom.coords.m);
        values[5] = Int64GetDatum((int64)atom.hilbert.lo);
        values[6] = Int64GetDatum((int64)atom.hilbert.hi);
        
        hash_bytea = (bytea *)palloc(VARHDRSZ + HC_HASH_SIZE);
        SET_VARSIZE(hash_bytea, VARHDRSZ + HC_HASH_SIZE);
        memcpy(VARDATA(hash_bytea), atom.hash.bytes, HC_HASH_SIZE);
        values[7] = PointerGetDatum(hash_bytea);
        
        values[8] = CStringGetTextDatum(hc_category_name(atom.category));
        
        tuple = heap_form_tuple(state->tupdesc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    
    SRF_RETURN_DONE(funcctx);
}

/* ============================================================================
 * Centroid Function
 * ============================================================================ */

PG_FUNCTION_INFO_V1(hypercube_centroid);
Datum hypercube_centroid(PG_FUNCTION_ARGS)
{
    ArrayType *x_arr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *y_arr = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *z_arr = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *m_arr = PG_GETARG_ARRAYTYPE_P(3);
    
    int n = ArrayGetNItems(ARR_NDIM(x_arr), ARR_DIMS(x_arr));
    
    if (n == 0)
        PG_RETURN_NULL();
    
    int64 *x_data = (int64 *)ARR_DATA_PTR(x_arr);
    int64 *y_data = (int64 *)ARR_DATA_PTR(y_arr);
    int64 *z_data = (int64 *)ARR_DATA_PTR(z_arr);
    int64 *m_data = (int64 *)ARR_DATA_PTR(m_arr);
    
    /* Allocate points array */
    hc_point4d_t *points = (hc_point4d_t *)palloc(n * sizeof(hc_point4d_t));
    for (int i = 0; i < n; ++i)
    {
        points[i].x = (uint32)x_data[i];
        points[i].y = (uint32)y_data[i];
        points[i].z = (uint32)z_data[i];
        points[i].m = (uint32)m_data[i];
    }
    
    hc_point4d_t centroid = hc_centroid(points, n);
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
    
    values[0] = Int64GetDatum((int64)centroid.x);
    values[1] = Int64GetDatum((int64)centroid.y);
    values[2] = Int64GetDatum((int64)centroid.z);
    values[3] = Int64GetDatum((int64)centroid.m);
    
    tuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/* ============================================================================
 * Content Hash (CPE Cascade) - Compute Merkle DAG root hash for text
 * ============================================================================ */

PG_FUNCTION_INFO_V1(hypercube_content_hash);
Datum hypercube_content_hash(PG_FUNCTION_ARGS)
{
    ArrayType *hashes_arr = PG_GETARG_ARRAYTYPE_P(0);
    Datum *elems;
    bool *nulls;
    int nelems;
    hc_hash_t *atom_hashes;
    hc_hash_t result;
    bytea *result_bytea;
    int i;
    
    deconstruct_array(hashes_arr, BYTEAOID, -1, false, TYPALIGN_INT,
                     &elems, &nulls, &nelems);
    
    if (nelems == 0) {
        PG_RETURN_NULL();
    }
    
    /* Convert bytea array to hc_hash_t array */
    atom_hashes = (hc_hash_t *)palloc(nelems * sizeof(hc_hash_t));
    for (i = 0; i < nelems; i++) {
        if (nulls[i]) {
            memset(atom_hashes[i].bytes, 0, HC_HASH_SIZE);
        } else {
            bytea *b = DatumGetByteaP(elems[i]);
            if (VARSIZE(b) - VARHDRSZ >= HC_HASH_SIZE) {
                memcpy(atom_hashes[i].bytes, VARDATA(b), HC_HASH_SIZE);
            } else {
                memset(atom_hashes[i].bytes, 0, HC_HASH_SIZE);
            }
        }
    }
    
    /* Compute CPE cascade */
    result = hc_content_hash_codepoints(NULL, nelems, atom_hashes);
    
    pfree(atom_hashes);
    
    /* Return as bytea */
    result_bytea = (bytea *)palloc(VARHDRSZ + HC_HASH_SIZE);
    SET_VARSIZE(result_bytea, VARHDRSZ + HC_HASH_SIZE);
    memcpy(VARDATA(result_bytea), result.bytes, HC_HASH_SIZE);
    
    PG_RETURN_BYTEA_P(result_bytea);
}
