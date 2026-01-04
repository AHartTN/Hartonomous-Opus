-- ============================================================================
-- Generative Walk Engine PostgreSQL Extension
-- LLM-like generation using the hypercube substrate
-- ============================================================================

-- =============================================================================
-- Type definitions
-- =============================================================================

CREATE TYPE gen_similar_result AS (
    label TEXT,
    similarity DOUBLE PRECISION
);

CREATE TYPE gen_candidate_result AS (
    label TEXT,
    score_shape DOUBLE PRECISION,
    score_pmi DOUBLE PRECISION,
    score_attn DOUBLE PRECISION,
    score_global DOUBLE PRECISION,
    score_total DOUBLE PRECISION
);

CREATE TYPE gen_stats_result AS (
    key TEXT,
    value BIGINT
);

-- =============================================================================
-- Cache Loading Functions
-- =============================================================================

-- Load vocabulary from composition + shape tables (all models)
CREATE FUNCTION gen_load_vocab()
RETURNS BIGINT
AS 'MODULE_PATHNAME', 'gen_load_vocab'
LANGUAGE C VOLATILE;

COMMENT ON FUNCTION gen_load_vocab() IS
'Load vocabulary entries from composition table with embeddings from all models.
This aggregates embeddings across models for each token.';

-- Load bigram PMI scores
CREATE FUNCTION gen_load_bigrams()
RETURNS BIGINT
AS 'MODULE_PATHNAME', 'gen_load_bigrams'
LANGUAGE C VOLATILE;

COMMENT ON FUNCTION gen_load_bigrams() IS
'Load bigram PMI scores from bigram_stats table.
Run extract_bigrams_from_compositions() first if table is empty.';

-- Load attention relations
CREATE FUNCTION gen_load_attention()
RETURNS BIGINT
AS 'MODULE_PATHNAME', 'gen_load_attention'
LANGUAGE C VOLATILE;

COMMENT ON FUNCTION gen_load_attention() IS
'Load attention edges from relation table (type A).';

-- Load everything in one call
CREATE FUNCTION gen_load_all()
RETURNS TEXT
AS 'MODULE_PATHNAME', 'gen_load_all'
LANGUAGE C VOLATILE;

COMMENT ON FUNCTION gen_load_all() IS
'Load vocab, bigrams, and attention in one call.
Returns summary of loaded counts.';

-- =============================================================================
-- Configuration
-- =============================================================================

-- Configure scoring weights and selection policy
CREATE FUNCTION gen_config(
    w_shape DOUBLE PRECISION DEFAULT 0.4,
    w_pmi DOUBLE PRECISION DEFAULT 0.3,
    w_attn DOUBLE PRECISION DEFAULT 0.2,
    w_global DOUBLE PRECISION DEFAULT 0.1,
    greedy BOOLEAN DEFAULT TRUE,
    temperature DOUBLE PRECISION DEFAULT 1.0
)
RETURNS VOID
AS 'MODULE_PATHNAME', 'gen_config'
LANGUAGE C VOLATILE;

COMMENT ON FUNCTION gen_config(DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, BOOLEAN, DOUBLE PRECISION) IS
'Configure the generative engine.
- w_shape: Weight for embedding similarity (default 0.4)
- w_pmi: Weight for PMI/co-occurrence (default 0.3)
- w_attn: Weight for attention relations (default 0.2)
- w_global: Weight for frequency prior (default 0.1)
- greedy: If true, always pick highest score; if false, sample (default true)
- temperature: Softmax temperature for sampling (default 1.0)';

-- Configure candidate filtering
CREATE FUNCTION gen_config_filter(
    max_candidates INTEGER DEFAULT 500,
    hilbert_range DOUBLE PRECISION DEFAULT 0.1
)
RETURNS VOID
AS 'MODULE_PATHNAME', 'gen_config_filter'
LANGUAGE C VOLATILE;

COMMENT ON FUNCTION gen_config_filter(INTEGER, DOUBLE PRECISION) IS
'Configure candidate filtering for generation.
- max_candidates: Max candidates to consider (default 500, set high to disable Hilbert filter)
- hilbert_range: Hilbert distance range for pre-filtering (default 0.1)';

-- =============================================================================
-- Similarity Search
-- =============================================================================

-- Find similar tokens by shape (embedding similarity)
CREATE FUNCTION gen_similar(label TEXT, k INTEGER DEFAULT 10)
RETURNS SETOF gen_similar_result
AS 'MODULE_PATHNAME', 'gen_similar'
LANGUAGE C STABLE STRICT;

COMMENT ON FUNCTION gen_similar(TEXT, INTEGER) IS
'Find K most similar tokens by embedding similarity.
Uses averaged embeddings across all models.
Example: SELECT * FROM gen_similar(''whale'', 10)';

-- =============================================================================
-- Next Token Prediction
-- =============================================================================

-- Score candidates for next token
CREATE FUNCTION gen_next_candidates(label TEXT, k INTEGER DEFAULT 20)
RETURNS SETOF gen_candidate_result
AS 'MODULE_PATHNAME', 'gen_next_candidates'
LANGUAGE C STABLE STRICT;

COMMENT ON FUNCTION gen_next_candidates(TEXT, INTEGER) IS
'Show top K next-token candidates with score breakdown.
Useful for understanding what drives predictions.
Example: SELECT * FROM gen_next_candidates(''the'', 10)';

-- =============================================================================
-- Generation
-- =============================================================================

-- Generate tokens as a set
CREATE FUNCTION gen_walk(start_label TEXT, max_tokens INTEGER DEFAULT 20)
RETURNS SETOF TEXT
AS 'MODULE_PATHNAME', 'gen_walk'
LANGUAGE C STABLE STRICT;

COMMENT ON FUNCTION gen_walk(TEXT, INTEGER) IS
'Generate a sequence of tokens starting from a label.
Returns one row per generated token.
Example: SELECT * FROM gen_walk(''the'', 10)';

-- Generate and concatenate to text
CREATE FUNCTION gen_complete(start_label TEXT, max_tokens INTEGER DEFAULT 20)
RETURNS TEXT
AS 'MODULE_PATHNAME', 'gen_complete'
LANGUAGE C STABLE STRICT;

COMMENT ON FUNCTION gen_complete(TEXT, INTEGER) IS
'Generate text completion starting from a token.
Returns concatenated output as single string.
Example: SELECT gen_complete(''whale'', 15)';

-- =============================================================================
-- Statistics
-- =============================================================================

CREATE FUNCTION gen_stats()
RETURNS SETOF gen_stats_result
AS 'MODULE_PATHNAME', 'gen_stats'
LANGUAGE C STABLE;

COMMENT ON FUNCTION gen_stats() IS
'Show generative engine cache statistics.
Returns vocab_count, bigram_count, attention_count.';

-- =============================================================================
-- Convenience Wrappers
-- =============================================================================

-- Initialize and generate in one call
CREATE OR REPLACE FUNCTION generate(prompt TEXT, max_tokens INTEGER DEFAULT 20)
RETURNS TEXT
LANGUAGE PLPGSQL STABLE AS $$
DECLARE
    result TEXT;
BEGIN
    -- Auto-load if needed
    IF (SELECT COALESCE(SUM(value), 0) FROM gen_stats()) = 0 THEN
        PERFORM gen_load_vocab();
        PERFORM gen_load_bigrams();
        PERFORM gen_load_attention();
    END IF;
    
    -- Generate
    SELECT gen_complete(prompt, max_tokens) INTO result;
    RETURN result;
END;
$$;

COMMENT ON FUNCTION generate(TEXT, INTEGER) IS
'Generate text with auto-initialization.
Example: SELECT generate(''the whale'', 15)';

-- Easy config presets
CREATE OR REPLACE FUNCTION gen_preset(preset_name TEXT)
RETURNS VOID
LANGUAGE PLPGSQL AS $$
BEGIN
    CASE preset_name
        WHEN 'shape_only' THEN
            -- Pure embedding similarity
            PERFORM gen_config(1.0, 0.0, 0.0, 0.0, true, 1.0);
        WHEN 'pmi_only' THEN
            -- Pure co-occurrence
            PERFORM gen_config(0.0, 1.0, 0.0, 0.0, true, 1.0);
        WHEN 'attention_only' THEN
            -- Pure attention
            PERFORM gen_config(0.0, 0.0, 1.0, 0.0, true, 1.0);
        WHEN 'balanced' THEN
            -- Equal weights
            PERFORM gen_config(0.25, 0.25, 0.25, 0.25, true, 1.0);
        WHEN 'creative' THEN
            -- Stochastic with higher temperature
            PERFORM gen_config(0.4, 0.3, 0.2, 0.1, false, 1.5);
        WHEN 'focused' THEN
            -- Greedy with low temperature
            PERFORM gen_config(0.5, 0.3, 0.2, 0.0, true, 0.5);
        ELSE
            RAISE EXCEPTION 'Unknown preset: %. Use: shape_only, pmi_only, attention_only, balanced, creative, focused', preset_name;
    END CASE;
END;
$$;

COMMENT ON FUNCTION gen_preset(TEXT) IS
'Apply a configuration preset.
Presets: shape_only, pmi_only, attention_only, balanced, creative, focused';
