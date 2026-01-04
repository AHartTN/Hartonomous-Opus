/**
 * Generative Engine PostgreSQL Extension (Pure C)
 * 
 * Exposes the generative walk engine to SQL.
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

#include "hypercube/generative_c.h"

PG_MODULE_MAGIC;

/* ==========================================================================
 * Internal helper: load vocabulary
 * ========================================================================== */

static int64 load_vocab_internal(void)
{
    int ret;
    uint64 i;
    int64 count = 0;
    int64 centroid_count = 0;
    
    /* Clear existing cache */
    gen_vocab_clear();
    
    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }
    
    /* Load all compositions with 4D centroids */
    ret = SPI_execute(
        "SELECT c.id, c.label, c.depth, "
        "       COALESCE((SELECT COUNT(*) FROM composition_child cc WHERE cc.child_id = c.id), 0) AS freq, "
        "       ST_X(c.centroid) AS cx, ST_Y(c.centroid) AS cy, "
        "       ST_Z(c.centroid) AS cz, ST_M(c.centroid) AS cm, "
        "       (c.hilbert_lo::float8 / 9223372036854775807.0) AS hilbert "
        "FROM composition c "
        "WHERE c.label IS NOT NULL "
        "  AND c.label NOT LIKE '[%' "
        "  AND c.centroid IS NOT NULL "
        "ORDER BY c.label",
        true, 0
    );
    
    if (ret != SPI_OK_SELECT) {
        SPI_finish();
        ereport(ERROR, (errmsg("Failed to load compositions")));
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
        
        /* Add to vocab */
        int64_t idx = gen_vocab_add(id_data, label, depth, freq, hilbert);
        
        /* Set 4D centroid */
        if (idx >= 0 && (cx != 0.0 || cy != 0.0 || cz != 0.0 || cm != 0.0)) {
            gen_vocab_set_centroid((size_t)idx, cx, cy, cz, cm);
            centroid_count++;
        }
        
        count++;
        pfree(label);
    }
    
    ereport(NOTICE, (errmsg("Loaded %lld vocabulary entries with %lld 4D centroids", 
                            (long long)count, (long long)centroid_count)));
    
    gen_vocab_finalize();
    
    SPI_finish();
    
    return count;
}

/* ==========================================================================
 * gen_load_vocab() -> BIGINT
 * Load vocabulary from composition + shape tables (all models)
 * ========================================================================== */

PG_FUNCTION_INFO_V1(gen_load_vocab);

Datum
gen_load_vocab(PG_FUNCTION_ARGS)
{
    PG_RETURN_INT64(load_vocab_internal());
}

/* ==========================================================================
 * Internal helper: load bigrams from relation table
 * PMI-like scores come from model ingestion similarity edges
 * ========================================================================== */

static int64 load_bigrams_internal(void)
{
    int ret;
    uint64 i;
    int64 count = 0;
    
    gen_bigram_clear();
    
    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }
    
    /* Load from relation table - 'S' similarity edges are PMI-like scores from model */
    ret = SPI_execute(
        "SELECT source_id, target_id, weight "
        "FROM relation "
        "WHERE relation_type = 'S' "
        "  AND weight > 0.3",
        true, 0
    );
    
    if (ret == SPI_OK_SELECT) {
        for (i = 0; i < SPI_processed; i++) {
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
            
            gen_bigram_add(
                (uint8_t *)VARDATA(left_bytea),
                (uint8_t *)VARDATA(right_bytea),
                score
            );
            count++;
        }
    }
    
    SPI_finish();
    
    ereport(NOTICE, (errmsg("Loaded %lld bigram PMI scores", (long long)count)));
    
    return count;
}

/* ==========================================================================
 * gen_load_bigrams() -> BIGINT
 * Load bigram PMI scores into cache
 * ========================================================================== */

PG_FUNCTION_INFO_V1(gen_load_bigrams);

Datum
gen_load_bigrams(PG_FUNCTION_ARGS)
{
    PG_RETURN_INT64(load_bigrams_internal());
}

/* ==========================================================================
 * Internal helper: load attention
 * ========================================================================== */

static int64 load_attention_internal(void)
{
    int ret;
    uint64 i;
    int64 count = 0;
    
    gen_attention_clear();
    
    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }
    
    /* Load attention relations - 'A' explicit attention, 'W' weight edges from model */
    ret = SPI_execute(
        "SELECT source_id, target_id, weight "
        "FROM relation "
        "WHERE relation_type IN ('A', 'W') "
        "  AND ABS(weight) > 0.1",  /* Weight edges can be negative */
        true, 0
    );
    
    if (ret == SPI_OK_SELECT) {
        for (i = 0; i < SPI_processed; i++) {
            HeapTuple tuple = SPI_tuptable->vals[i];
            TupleDesc tupdesc = SPI_tuptable->tupdesc;
            bool isnull;
            
            Datum src_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
            if (isnull) continue;
            bytea *src_bytea = DatumGetByteaP(src_datum);
            
            Datum tgt_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
            if (isnull) continue;
            bytea *tgt_bytea = DatumGetByteaP(tgt_datum);
            
            Datum weight_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull);
            /* weight is REAL (float4), not FLOAT8 */
            double weight = isnull ? 0.0 : (double)DatumGetFloat4(weight_datum);
            
            gen_attention_add(
                (uint8_t *)VARDATA(src_bytea),
                (uint8_t *)VARDATA(tgt_bytea),
                weight
            );
            count++;
        }
    }
    
    SPI_finish();
    
    ereport(NOTICE, (errmsg("Loaded %lld attention edges", (long long)count)));
    
    return count;
}

/* ==========================================================================
 * gen_load_attention() -> BIGINT
 * Load attention relations into cache
 * ========================================================================== */

PG_FUNCTION_INFO_V1(gen_load_attention);

Datum
gen_load_attention(PG_FUNCTION_ARGS)
{
    PG_RETURN_INT64(load_attention_internal());
}

/* ==========================================================================
 * gen_lookup_bigram(left_label TEXT, right_label TEXT) -> FLOAT8
 * Debug function to lookup a specific bigram PMI score
 * ========================================================================== */

PG_FUNCTION_INFO_V1(gen_lookup_bigram);

Datum
gen_lookup_bigram(PG_FUNCTION_ARGS)
{
    text *left_text = PG_GETARG_TEXT_PP(0);
    text *right_text = PG_GETARG_TEXT_PP(1);
    
    char *left_label = text_to_cstring(left_text);
    char *right_label = text_to_cstring(right_text);
    
    /* Find vocab indices by label */
    int64_t left_idx = gen_vocab_find_label(left_label);
    int64_t right_idx = gen_vocab_find_label(right_label);
    
    if (left_idx < 0) {
        ereport(NOTICE, (errmsg("Left label '%s' not found in vocab", left_label)));
        pfree(left_label);
        pfree(right_label);
        PG_RETURN_FLOAT8(0.0);
    }
    
    if (right_idx < 0) {
        ereport(NOTICE, (errmsg("Right label '%s' not found in vocab", right_label)));
        pfree(left_label);
        pfree(right_label);
        PG_RETURN_FLOAT8(0.0);
    }
    
    /* Get IDs from vocab entries */
    /* We need to query the DB for the actual IDs since we don't have a direct API */
    int ret;
    double score = 0.0;
    
    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }
    
    /* Get IDs for both labels from the database */
    char query[512];
    snprintf(query, sizeof(query),
        "SELECT a.id, b.id FROM composition a, composition b "
        "WHERE a.label = '%s' AND b.label = '%s'",
        left_label, right_label);
    
    ret = SPI_execute(query, true, 1);
    
    if (ret == SPI_OK_SELECT && SPI_processed > 0) {
        HeapTuple tuple = SPI_tuptable->vals[0];
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        bool isnull;
        
        Datum left_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (!isnull) {
            bytea *left_bytea = DatumGetByteaP(left_datum);
            
            Datum right_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
            if (!isnull) {
                bytea *right_bytea = DatumGetByteaP(right_datum);
                
                /* Debug: print first 8 bytes of each hash */
                ereport(NOTICE, (errmsg("Left ID hex: %02x%02x%02x%02x%02x%02x%02x%02x", 
                    ((uint8_t *)VARDATA(left_bytea))[0],
                    ((uint8_t *)VARDATA(left_bytea))[1],
                    ((uint8_t *)VARDATA(left_bytea))[2],
                    ((uint8_t *)VARDATA(left_bytea))[3],
                    ((uint8_t *)VARDATA(left_bytea))[4],
                    ((uint8_t *)VARDATA(left_bytea))[5],
                    ((uint8_t *)VARDATA(left_bytea))[6],
                    ((uint8_t *)VARDATA(left_bytea))[7])));
                ereport(NOTICE, (errmsg("Right ID hex: %02x%02x%02x%02x%02x%02x%02x%02x", 
                    ((uint8_t *)VARDATA(right_bytea))[0],
                    ((uint8_t *)VARDATA(right_bytea))[1],
                    ((uint8_t *)VARDATA(right_bytea))[2],
                    ((uint8_t *)VARDATA(right_bytea))[3],
                    ((uint8_t *)VARDATA(right_bytea))[4],
                    ((uint8_t *)VARDATA(right_bytea))[5],
                    ((uint8_t *)VARDATA(right_bytea))[6],
                    ((uint8_t *)VARDATA(right_bytea))[7])));
                
                /* Use debug find to get more info */
                double found_score = 0.0;
                int result = gen_bigram_debug_find(
                    (uint8_t *)VARDATA(left_bytea),
                    (uint8_t *)VARDATA(right_bytea),
                    &found_score
                );
                
                if (result == 1) {
                    score = found_score;
                    ereport(NOTICE, (errmsg("Bigram FOUND '%s' -> '%s': %f", 
                                            left_label, right_label, score)));
                } else {
                    ereport(NOTICE, (errmsg("Bigram NOT FOUND '%s' -> '%s' (left matches: %d)", 
                                            left_label, right_label, -result)));
                }
            }
        }
    }
    
    SPI_finish();
    
    pfree(left_label);
    pfree(right_label);
    
    PG_RETURN_FLOAT8(score);
}

/* ==========================================================================
 * gen_load_all() -> TEXT
 * Load everything in one call
 * ========================================================================== */

PG_FUNCTION_INFO_V1(gen_load_all);

Datum
gen_load_all(PG_FUNCTION_ARGS)
{
    int64 vocab_count, bigram_count, attn_count;
    char result[256];
    
    /* Load vocab */
    vocab_count = load_vocab_internal();
    
    /* Load bigrams */
    bigram_count = load_bigrams_internal();
    
    /* Load attention */
    attn_count = load_attention_internal();
    
    snprintf(result, sizeof(result),
        "Loaded: vocab=%lld, bigrams=%lld, attention=%lld",
        (long long)vocab_count, (long long)bigram_count, (long long)attn_count);
    
    PG_RETURN_TEXT_P(cstring_to_text(result));
}

/* ==========================================================================
 * gen_config(w_shape, w_pmi, w_attn, w_global, greedy, temperature)
 * ========================================================================== */

PG_FUNCTION_INFO_V1(gen_config);

Datum
gen_config(PG_FUNCTION_ARGS)
{
    double w_shape = PG_GETARG_FLOAT8(0);
    double w_pmi = PG_GETARG_FLOAT8(1);
    double w_attn = PG_GETARG_FLOAT8(2);
    double w_global = PG_GETARG_FLOAT8(3);
    int greedy = PG_GETARG_BOOL(4) ? 1 : 0;
    double temperature = PG_GETARG_FLOAT8(5);
    
    gen_config_set_weights(w_shape, w_pmi, w_attn, w_global);
    gen_config_set_policy(greedy, temperature);
    
    PG_RETURN_VOID();
}

/* ==========================================================================
 * gen_config_filter(max_candidates, hilbert_range)
 * ========================================================================== */

PG_FUNCTION_INFO_V1(gen_config_filter);

Datum
gen_config_filter(PG_FUNCTION_ARGS)
{
    int max_candidates = PG_GETARG_INT32(0);
    double hilbert_range = PG_GETARG_FLOAT8(1);
    
    gen_config_set_filter((size_t)max_candidates, hilbert_range);
    
    PG_RETURN_VOID();
}

/* ==========================================================================
 * gen_similar(label TEXT, k INT) -> SETOF (label, similarity)
 * ========================================================================== */

typedef struct {
    size_t current;
    size_t count;
    GenSimilarResult *results;
} GenSimilarState;

PG_FUNCTION_INFO_V1(gen_similar);

Datum
gen_similar(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    GenSimilarState *state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        text *label_arg;
        char *label;
        int k;
        TupleDesc tupdesc;
        
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        label_arg = PG_GETARG_TEXT_PP(0);
        label = text_to_cstring(label_arg);
        k = PG_GETARG_INT32(1);
        
        if (gen_vocab_count() == 0) {
            ereport(ERROR, (errmsg("Vocab empty. Call gen_load_vocab() first.")));
        }
        
        state = palloc(sizeof(GenSimilarState));
        state->results = palloc(k * sizeof(GenSimilarResult));
        state->count = gen_find_similar(label, (size_t)k, state->results);
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
    state = (GenSimilarState *)funcctx->user_fctx;
    
    if (state->current < state->count) {
        Datum values[2];
        bool nulls[2] = {false, false};
        HeapTuple tuple;
        const char *result_label;
        
        result_label = gen_vocab_get_label(state->results[state->current].index);
        values[0] = CStringGetTextDatum(result_label ? result_label : "");
        values[1] = Float8GetDatum(state->results[state->current].similarity);
        
        state->current++;
        
        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    } else {
        SRF_RETURN_DONE(funcctx);
    }
}

/* ==========================================================================
 * gen_next_candidates(label TEXT, k INT) 
 *   -> SETOF (label, score_shape, score_pmi, score_attn, score_global, score_total)
 * ========================================================================== */

typedef struct {
    size_t current;
    size_t count;
    GenTokenResult *results;
} GenCandidatesState;

PG_FUNCTION_INFO_V1(gen_next_candidates);

Datum
gen_next_candidates(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    GenCandidatesState *state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        text *label_arg;
        char *label;
        int k;
        TupleDesc tupdesc;
        
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        label_arg = PG_GETARG_TEXT_PP(0);
        label = text_to_cstring(label_arg);
        k = PG_GETARG_INT32(1);
        
        if (gen_vocab_count() == 0) {
            ereport(ERROR, (errmsg("Vocab empty. Call gen_load_vocab() first.")));
        }
        
        state = palloc(sizeof(GenCandidatesState));
        state->results = palloc(k * sizeof(GenTokenResult));
        state->count = gen_score_candidates(label, (size_t)k, state->results);
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
    state = (GenCandidatesState *)funcctx->user_fctx;
    
    if (state->current < state->count) {
        Datum values[6];
        bool nulls[6] = {false, false, false, false, false, false};
        HeapTuple tuple;
        const char *result_label;
        GenTokenResult *r = &state->results[state->current];
        
        result_label = gen_vocab_get_label(r->token_index);
        values[0] = CStringGetTextDatum(result_label ? result_label : "");
        values[1] = Float8GetDatum(r->score_centroid);
        values[2] = Float8GetDatum(r->score_pmi);
        values[3] = Float8GetDatum(r->score_attn);
        values[4] = Float8GetDatum(r->score_global);
        values[5] = Float8GetDatum(r->score_total);
        
        state->current++;
        
        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    } else {
        SRF_RETURN_DONE(funcctx);
    }
}

/* ==========================================================================
 * gen_walk(start_label TEXT, max_tokens INT) -> SETOF TEXT
 * ========================================================================== */

typedef struct {
    size_t current;
    size_t count;
    GenTokenResult *results;
} GenWalkState;

PG_FUNCTION_INFO_V1(gen_walk);

Datum
gen_walk(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    GenWalkState *state;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        text *label_arg;
        char *label;
        int max_tokens;
        
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        label_arg = PG_GETARG_TEXT_PP(0);
        label = text_to_cstring(label_arg);
        max_tokens = PG_GETARG_INT32(1);
        
        if (gen_vocab_count() == 0) {
            ereport(ERROR, (errmsg("Vocab empty. Call gen_load_vocab() first.")));
        }
        
        state = palloc(sizeof(GenWalkState));
        state->results = palloc(max_tokens * sizeof(GenTokenResult));
        state->count = gen_generate(label, (size_t)max_tokens, state->results);
        state->current = 0;
        
        funcctx->user_fctx = state;
        
        pfree(label);
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    state = (GenWalkState *)funcctx->user_fctx;
    
    if (state->current < state->count) {
        const char *result_label;
        
        result_label = gen_vocab_get_label(state->results[state->current].token_index);
        state->current++;
        
        SRF_RETURN_NEXT(funcctx, CStringGetTextDatum(result_label ? result_label : ""));
    } else {
        SRF_RETURN_DONE(funcctx);
    }
}

/* ==========================================================================
 * gen_complete(start TEXT, max_tokens INT) -> TEXT
 * Convenience function: returns concatenated output
 * ========================================================================== */

PG_FUNCTION_INFO_V1(gen_complete);

Datum
gen_complete(PG_FUNCTION_ARGS)
{
    text *start_arg = PG_GETARG_TEXT_PP(0);
    int max_tokens = PG_GETARG_INT32(1);
    char *start_label = text_to_cstring(start_arg);
    
    GenTokenResult *results;
    size_t count, i;
    StringInfoData buf;
    
    if (gen_vocab_count() == 0) {
        ereport(ERROR, (errmsg("Vocab empty. Call gen_load_vocab() first.")));
    }
    
    results = palloc(max_tokens * sizeof(GenTokenResult));
    count = gen_generate(start_label, (size_t)max_tokens, results);
    
    initStringInfo(&buf);
    
    for (i = 0; i < count; i++) {
        const char *label = gen_vocab_get_label(results[i].token_index);
        if (label) {
            if (i > 0) {
                /* Simple spacing heuristic */
                if (label[0] != '#' && label[0] != ',' && label[0] != '.' && 
                    label[0] != '!' && label[0] != '?') {
                    appendStringInfoChar(&buf, ' ');
                }
            }
            appendStringInfoString(&buf, label);
        }
    }
    
    pfree(results);
    pfree(start_label);
    
    PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

/* ==========================================================================
 * gen_stats() -> TABLE(key TEXT, value BIGINT)
 * ========================================================================== */

PG_FUNCTION_INFO_V1(gen_stats);

Datum
gen_stats(PG_FUNCTION_ARGS)
{
    FuncCallContext *funcctx;
    int call_cntr;
    
    if (SRF_IS_FIRSTCALL()) {
        MemoryContext oldcontext;
        TupleDesc tupdesc;
        
        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
        
        funcctx->max_calls = 3;  /* vocab, bigrams, attention */
        
        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE) {
            ereport(ERROR, (errmsg("function must return composite type")));
        }
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);
        
        MemoryContextSwitchTo(oldcontext);
    }
    
    funcctx = SRF_PERCALL_SETUP();
    call_cntr = funcctx->call_cntr;
    
    if (call_cntr < 3) {
        Datum values[2];
        bool nulls[2] = {false, false};
        HeapTuple tuple;
        
        switch (call_cntr) {
            case 0:
                values[0] = CStringGetTextDatum("vocab_count");
                values[1] = Int64GetDatum((int64)gen_vocab_count());
                break;
            case 1:
                values[0] = CStringGetTextDatum("bigram_count");
                values[1] = Int64GetDatum((int64)gen_bigram_count());
                break;
            case 2:
                values[0] = CStringGetTextDatum("attention_count");
                values[1] = Int64GetDatum((int64)gen_attention_count());
                break;
        }
        
        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    } else {
        SRF_RETURN_DONE(funcctx);
    }
}
