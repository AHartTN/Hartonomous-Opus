/**
 * Embedding Operations PostgreSQL Extension (Pure C)
 * 
 * Exposes SIMD-accelerated embedding operations to SQL.
 * Uses the C bridge to call C++ implementation.
 */

#include <postgres.h>
#include <fmgr.h>
#include <funcapi.h>
#include <utils/array.h>
#include <utils/builtins.h>
#include <catalog/pg_type.h>
#include <access/htup_details.h>
#include <executor/spi.h>
#include <utils/memutils.h>

#include "hypercube/embedding_c.h"

PG_MODULE_MAGIC;

/* ==========================================================================
 * embedding_cosine_sim(float4[], float4[]) -> float8
 * ========================================================================== */

PG_FUNCTION_INFO_V1(embedding_cosine_sim);

Datum
embedding_cosine_sim(PG_FUNCTION_ARGS)
{
    ArrayType *arr_a, *arr_b;
    float *a, *b;
    int n_a, n_b;
    double result;
    
    if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
        PG_RETURN_NULL();
    
    arr_a = PG_GETARG_ARRAYTYPE_P(0);
    arr_b = PG_GETARG_ARRAYTYPE_P(1);
    
    n_a = ArrayGetNItems(ARR_NDIM(arr_a), ARR_DIMS(arr_a));
    n_b = ArrayGetNItems(ARR_NDIM(arr_b), ARR_DIMS(arr_b));
    
    if (n_a != n_b || n_a == 0)
        PG_RETURN_FLOAT8(0.0);
    
    a = (float *)ARR_DATA_PTR(arr_a);
    b = (float *)ARR_DATA_PTR(arr_b);
    
    result = embedding_c_cosine_similarity(a, b, (size_t)n_a);
    
    PG_RETURN_FLOAT8(result);
}

/* ==========================================================================
 * embedding_l2_dist(float4[], float4[]) -> float8
 * ========================================================================== */

PG_FUNCTION_INFO_V1(embedding_l2_dist);

Datum
embedding_l2_dist(PG_FUNCTION_ARGS)
{
    ArrayType *arr_a, *arr_b;
    float *a, *b;
    int n_a, n_b;
    double result;
    
    if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
        PG_RETURN_NULL();
    
    arr_a = PG_GETARG_ARRAYTYPE_P(0);
    arr_b = PG_GETARG_ARRAYTYPE_P(1);
    
    n_a = ArrayGetNItems(ARR_NDIM(arr_a), ARR_DIMS(arr_a));
    n_b = ArrayGetNItems(ARR_NDIM(arr_b), ARR_DIMS(arr_b));
    
    if (n_a != n_b || n_a == 0)
        PG_RETURN_FLOAT8(0.0);
    
    a = (float *)ARR_DATA_PTR(arr_a);
    b = (float *)ARR_DATA_PTR(arr_b);
    
    result = embedding_c_l2_distance(a, b, (size_t)n_a);
    
    PG_RETURN_FLOAT8(result);
}

/* ==========================================================================
 * embedding_vector_add(float4[], float4[]) -> float4[]
 * ========================================================================== */

PG_FUNCTION_INFO_V1(embedding_vector_add);

Datum
embedding_vector_add(PG_FUNCTION_ARGS)
{
    ArrayType *arr_a, *arr_b, *result_arr;
    float *a, *b, *result;
    int n_a, n_b;
    int dims[1];
    int lbs[1] = {1};
    
    if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
        PG_RETURN_NULL();
    
    arr_a = PG_GETARG_ARRAYTYPE_P(0);
    arr_b = PG_GETARG_ARRAYTYPE_P(1);
    
    n_a = ArrayGetNItems(ARR_NDIM(arr_a), ARR_DIMS(arr_a));
    n_b = ArrayGetNItems(ARR_NDIM(arr_b), ARR_DIMS(arr_b));
    
    if (n_a != n_b || n_a == 0)
        PG_RETURN_NULL();
    
    a = (float *)ARR_DATA_PTR(arr_a);
    b = (float *)ARR_DATA_PTR(arr_b);
    
    result = palloc(n_a * sizeof(float));
    embedding_c_vector_add(a, b, result, (size_t)n_a);
    
    dims[0] = n_a;
    result_arr = construct_md_array((Datum *)result, NULL, 1, dims, lbs,
                                    FLOAT4OID, sizeof(float), true, TYPALIGN_INT);
    
    PG_RETURN_ARRAYTYPE_P(result_arr);
}

/* ==========================================================================
 * embedding_vector_sub(float4[], float4[]) -> float4[]
 * ========================================================================== */

PG_FUNCTION_INFO_V1(embedding_vector_sub);

Datum
embedding_vector_sub(PG_FUNCTION_ARGS)
{
    ArrayType *arr_a, *arr_b, *result_arr;
    float *a, *b, *result;
    int n_a, n_b;
    int dims[1];
    int lbs[1] = {1};
    
    if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
        PG_RETURN_NULL();
    
    arr_a = PG_GETARG_ARRAYTYPE_P(0);
    arr_b = PG_GETARG_ARRAYTYPE_P(1);
    
    n_a = ArrayGetNItems(ARR_NDIM(arr_a), ARR_DIMS(arr_a));
    n_b = ArrayGetNItems(ARR_NDIM(arr_b), ARR_DIMS(arr_b));
    
    if (n_a != n_b || n_a == 0)
        PG_RETURN_NULL();
    
    a = (float *)ARR_DATA_PTR(arr_a);
    b = (float *)ARR_DATA_PTR(arr_b);
    
    result = palloc(n_a * sizeof(float));
    embedding_c_vector_sub(a, b, result, (size_t)n_a);
    
    dims[0] = n_a;
    result_arr = construct_md_array((Datum *)result, NULL, 1, dims, lbs,
                                    FLOAT4OID, sizeof(float), true, TYPALIGN_INT);
    
    PG_RETURN_ARRAYTYPE_P(result_arr);
}

/* ==========================================================================
 * embedding_analogy_vec(float4[], float4[], float4[]) -> float4[]
 * Computes: c + (a - b)
 * ========================================================================== */

PG_FUNCTION_INFO_V1(embedding_analogy_vec);

Datum
embedding_analogy_vec(PG_FUNCTION_ARGS)
{
    ArrayType *arr_a, *arr_b, *arr_c, *result_arr;
    float *a, *b, *c, *result;
    int n_a, n_b, n_c;
    int dims[1];
    int lbs[1] = {1};
    
    if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2))
        PG_RETURN_NULL();
    
    arr_a = PG_GETARG_ARRAYTYPE_P(0);
    arr_b = PG_GETARG_ARRAYTYPE_P(1);
    arr_c = PG_GETARG_ARRAYTYPE_P(2);
    
    n_a = ArrayGetNItems(ARR_NDIM(arr_a), ARR_DIMS(arr_a));
    n_b = ArrayGetNItems(ARR_NDIM(arr_b), ARR_DIMS(arr_b));
    n_c = ArrayGetNItems(ARR_NDIM(arr_c), ARR_DIMS(arr_c));
    
    if (n_a != n_b || n_a != n_c || n_a == 0)
        PG_RETURN_NULL();
    
    a = (float *)ARR_DATA_PTR(arr_a);
    b = (float *)ARR_DATA_PTR(arr_b);
    c = (float *)ARR_DATA_PTR(arr_c);
    
    result = palloc(n_a * sizeof(float));
    embedding_c_analogy_target(a, b, c, result, (size_t)n_a);
    
    dims[0] = n_a;
    result_arr = construct_md_array((Datum *)result, NULL, 1, dims, lbs,
                                    FLOAT4OID, sizeof(float), true, TYPALIGN_INT);
    
    PG_RETURN_ARRAYTYPE_P(result_arr);
}

/* ==========================================================================
 * Cache Management Functions
 * ========================================================================== */

PG_FUNCTION_INFO_V1(embedding_cache_init);

Datum
embedding_cache_init(PG_FUNCTION_ARGS)
{
    int result = embedding_c_cache_init();
    if (result != 0) {
        ereport(ERROR, (errmsg("Failed to initialize embedding cache")));
    }
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(embedding_cache_clear);

Datum
embedding_cache_clear(PG_FUNCTION_ARGS)
{
    embedding_c_cache_clear();
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(embedding_cache_count);

Datum
embedding_cache_count(PG_FUNCTION_ARGS)
{
    PG_RETURN_INT64((int64)embedding_c_cache_count());
}

PG_FUNCTION_INFO_V1(embedding_cache_dim);

Datum
embedding_cache_dim(PG_FUNCTION_ARGS)
{
    PG_RETURN_INT64((int64)embedding_c_cache_dim());
}

/* ==========================================================================
 * embedding_cache_load(model_name TEXT) -> BIGINT
 * Loads all embeddings for a model into C++ cache via SPI
 * ========================================================================== */

PG_FUNCTION_INFO_V1(embedding_cache_load);

Datum
embedding_cache_load(PG_FUNCTION_ARGS)
{
    text *model_arg;
    char *model_name;
    StringInfoData sql;
    int ret;
    uint64 i;
    int64 count = 0;
    
    if (PG_ARGISNULL(0))
        PG_RETURN_INT64(0);
    
    model_arg = PG_GETARG_TEXT_PP(0);
    model_name = text_to_cstring(model_arg);
    
    /* Initialize and clear cache */
    embedding_c_cache_init();
    embedding_c_cache_clear();
    
    /* Connect to SPI */
    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }
    
    /* Query to extract embeddings as float4[] */
    initStringInfo(&sql);
    appendStringInfo(&sql,
        "SELECT c.id, c.label, "
        "       (SELECT array_agg(ST_Y(geom) ORDER BY ST_X(geom))::float4[] "
        "        FROM ST_DumpPoints(s.embedding)) AS emb "
        "FROM shape s "
        "JOIN composition c ON c.id = s.entity_id "
        "WHERE s.model_name = '%s' "
        "  AND c.label IS NOT NULL "
        "  AND c.label NOT LIKE '[%%' "
        "ORDER BY c.label",
        model_name);
    
    ret = SPI_execute(sql.data, true, 0);
    if (ret != SPI_OK_SELECT) {
        SPI_finish();
        ereport(ERROR, (errmsg("SPI query failed")));
    }
    
    for (i = 0; i < SPI_processed; i++) {
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
        
        /* Get embedding array */
        Datum emb_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull);
        if (isnull) continue;
        ArrayType *emb_arr = DatumGetArrayTypeP(emb_datum);
        int dim = ArrayGetNItems(ARR_NDIM(emb_arr), ARR_DIMS(emb_arr));
        float *emb = (float *)ARR_DATA_PTR(emb_arr);
        
        /* Add to cache */
        if (embedding_c_cache_add(id_data, id_len, label, emb, (size_t)dim) >= 0) {
            count++;
        }
        
        pfree(label);
    }
    
    SPI_finish();
    pfree(sql.data);
    pfree(model_name);
    
    ereport(NOTICE, (errmsg("Loaded %lld embeddings into cache", (long long)count)));
    
    PG_RETURN_INT64(count);
}

/* ==========================================================================
 * embedding_similar(label TEXT, k INT) -> SETOF (label TEXT, similarity FLOAT8)
 * ========================================================================== */

typedef struct {
    size_t current;
    size_t count;
    EmbeddingSimilarityResult *results;
} SimilarFuncState;

PG_FUNCTION_INFO_V1(embedding_similar);

Datum
embedding_similar(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    SimilarFuncState *state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        text *label_arg;
        char *label;
        int k;
        int64_t idx;
        size_t n_results;
        TupleDesc tupdesc;
        
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        label_arg = PG_GETARG_TEXT_PP(0);
        label = text_to_cstring(label_arg);
        k = PG_GETARG_INT32(1);
        
        if (embedding_c_cache_count() == 0) {
            ereport(ERROR, (errmsg("Cache empty. Call embedding_cache_load() first.")));
        }
        
        idx = embedding_c_cache_find_label(label);
        if (idx < 0) {
            ereport(ERROR, (errmsg("Label not found in cache: %s", label)));
        }
        
        state = palloc(sizeof(SimilarFuncState));
        state->results = palloc(k * sizeof(EmbeddingSimilarityResult));
        n_results = embedding_c_cache_similar((size_t)idx, (size_t)k, state->results);
        state->count = n_results;
        state->current = 0;
        
        funcctx->user_fctx = state;
        
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite type")));
        }
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        
        pfree(label);
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (SimilarFuncState *)funcctx->user_fctx;
    
    if (state->current < state->count) {
        Datum values[2];
        bool nulls[2] = {false, false};
        HeapTuple tuple;
        const char *result_label;
        
        result_label = embedding_c_cache_get_label(state->results[state->current].index);
        values[0] = CStringGetTextDatum(result_label);
        values[1] = Float8GetDatum(state->results[state->current].similarity);
        
        state->current++;
        
        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    } else {
        SRF_RETURN_DONE(funcctx);
    }
}

/* ==========================================================================
 * embedding_analogy(positive TEXT, negative TEXT, query TEXT, k INT)
 *   -> SETOF (label TEXT, similarity FLOAT8)
 * ========================================================================== */

PG_FUNCTION_INFO_V1(embedding_analogy);

Datum
embedding_analogy(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    SimilarFuncState *state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        text *a_arg, *b_arg, *c_arg;
        char *a_label, *b_label, *c_label;
        int k;
        int64_t idx_a, idx_b, idx_c;
        size_t n_results;
        TupleDesc tupdesc;
        
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        a_arg = PG_GETARG_TEXT_PP(0);
        b_arg = PG_GETARG_TEXT_PP(1);
        c_arg = PG_GETARG_TEXT_PP(2);
        a_label = text_to_cstring(a_arg);
        b_label = text_to_cstring(b_arg);
        c_label = text_to_cstring(c_arg);
        k = PG_GETARG_INT32(3);
        
        if (embedding_c_cache_count() == 0) {
            ereport(ERROR, (errmsg("Cache empty. Call embedding_cache_load() first.")));
        }
        
        idx_a = embedding_c_cache_find_label(a_label);
        idx_b = embedding_c_cache_find_label(b_label);
        idx_c = embedding_c_cache_find_label(c_label);
        
        if (idx_a < 0 || idx_b < 0 || idx_c < 0) {
            ereport(ERROR, (errmsg("One or more labels not found in cache")));
        }
        
        state = palloc(sizeof(SimilarFuncState));
        state->results = palloc(k * sizeof(EmbeddingSimilarityResult));
        n_results = embedding_c_cache_analogy(
            (size_t)idx_a, (size_t)idx_b, (size_t)idx_c,
            (size_t)k, state->results
        );
        state->count = n_results;
        state->current = 0;
        
        funcctx->user_fctx = state;
        
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite type")));
        }
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        
        pfree(a_label);
        pfree(b_label);
        pfree(c_label);
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (SimilarFuncState *)funcctx->user_fctx;
    
    if (state->current < state->count) {
        Datum values[2];
        bool nulls[2] = {false, false};
        HeapTuple tuple;
        const char *result_label;
        
        result_label = embedding_c_cache_get_label(state->results[state->current].index);
        values[0] = CStringGetTextDatum(result_label);
        values[1] = Float8GetDatum(state->results[state->current].similarity);
        
        state->current++;
        
        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    } else {
        SRF_RETURN_DONE(funcctx);
    }
}
