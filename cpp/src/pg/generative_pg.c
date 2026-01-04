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
    int64 emb_count = 0;
    
    /* Clear existing cache */
    gen_vocab_clear();
    
    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }
    
    /* Step 1: Load all compositions with depth and label */
    ret = SPI_execute(
        "SELECT c.id, c.label, c.depth, "
        "       COALESCE((SELECT COUNT(*) FROM composition_child cc WHERE cc.child_id = c.id), 0) AS freq "
        "FROM composition c "
        "WHERE c.label IS NOT NULL "
        "  AND c.label NOT LIKE '[%' "
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
        
        /* Add to vocab (hilbert index = 0 for now, will update) */
        gen_vocab_add(id_data, label, depth, freq, 0.0);
        count++;
        
        pfree(label);
    }
    
    ereport(NOTICE, (errmsg("Loaded %lld vocabulary entries", (long long)count)));
    
    /* Step 2: Load embeddings for each vocab entry (all models) */
    ret = SPI_execute(
        "SELECT c.label, s.model_name, "
        "       (SELECT array_agg(ST_Y(geom) ORDER BY ST_X(geom))::float4[] "
        "        FROM ST_DumpPoints(s.embedding)) AS emb "
        "FROM shape s "
        "JOIN composition c ON c.id = s.entity_id "
        "WHERE c.label IS NOT NULL "
        "  AND c.label NOT LIKE '[%' "
        "ORDER BY c.label, s.model_name",
        true, 0
    );
    
    if (ret == SPI_OK_SELECT) {
        for (i = 0; i < SPI_processed; i++) {
            HeapTuple tuple = SPI_tuptable->vals[i];
            TupleDesc tupdesc = SPI_tuptable->tupdesc;
            bool isnull;
            
            /* Get label */
            char *label = SPI_getvalue(tuple, tupdesc, 1);
            if (!label) continue;
            
            /* Find vocab index */
            int64_t idx = gen_vocab_find_label(label);
            if (idx < 0) {
                pfree(label);
                continue;
            }
            
            /* Get model name */
            char *model = SPI_getvalue(tuple, tupdesc, 2);
            
            /* Get embedding array */
            Datum emb_datum = SPI_getbinval(tuple, tupdesc, 3, &isnull);
            if (isnull) {
                pfree(label);
                if (model) pfree(model);
                continue;
            }
            
            ArrayType *emb_arr = DatumGetArrayTypeP(emb_datum);
            int dim = ArrayGetNItems(ARR_NDIM(emb_arr), ARR_DIMS(emb_arr));
            float *emb = (float *)ARR_DATA_PTR(emb_arr);
            
            /* Add embedding to entry */
            gen_vocab_add_embedding((size_t)idx, model ? model : "default", emb, (size_t)dim);
            emb_count++;
            
            pfree(label);
            if (model) pfree(model);
        }
    }
    
    ereport(NOTICE, (errmsg("Loaded %lld embeddings across all models", (long long)emb_count)));
    
    /* Finalize (compute averages, build flat array) */
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
 * Internal helper: load bigrams
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
    
    ret = SPI_execute(
        "SELECT left_id, right_id, COALESCE(pmi, ln(count + 1)) AS score "
        "FROM bigram_stats "
        "WHERE count > 0",
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
            double score = isnull ? 0.0 : DatumGetFloat8(score_datum);
            
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
    
    /* Load attention relations */
    ret = SPI_execute(
        "SELECT source_id, target_id, weight "
        "FROM relation "
        "WHERE relation_type = 'A' "
        "  AND weight > 0",
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
            double weight = isnull ? 0.0 : DatumGetFloat8(weight_datum);
            
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
        values[1] = Float8GetDatum(r->score_shape);
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
