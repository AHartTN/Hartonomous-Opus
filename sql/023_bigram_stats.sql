-- ============================================================================
-- Bigram Statistics for PMI/Co-occurrence Scoring
-- ============================================================================

-- Bigram counts (left token → right token)
CREATE TABLE IF NOT EXISTS bigram_stats (
    left_id  BYTEA NOT NULL,
    right_id BYTEA NOT NULL,
    count    BIGINT NOT NULL DEFAULT 1,
    pmi      DOUBLE PRECISION,  -- Computed PMI score
    PRIMARY KEY (left_id, right_id)
);

-- Indexes for efficient lookup
CREATE INDEX IF NOT EXISTS idx_bigram_left ON bigram_stats(left_id);
CREATE INDEX IF NOT EXISTS idx_bigram_right ON bigram_stats(right_id);

-- Unigram counts (for PMI computation)
CREATE TABLE IF NOT EXISTS unigram_stats (
    token_id BYTEA PRIMARY KEY,
    count    BIGINT NOT NULL DEFAULT 1
);

-- Total token count (for PMI normalization)
CREATE TABLE IF NOT EXISTS token_corpus_stats (
    id INTEGER PRIMARY KEY DEFAULT 1,
    total_tokens BIGINT NOT NULL DEFAULT 0,
    total_bigrams BIGINT NOT NULL DEFAULT 0
);

INSERT INTO token_corpus_stats (id, total_tokens, total_bigrams)
VALUES (1, 0, 0)
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- Functions to update bigram stats during ingestion
-- ============================================================================

-- Increment bigram count
CREATE OR REPLACE FUNCTION increment_bigram(
    p_left_id BYTEA,
    p_right_id BYTEA,
    p_count BIGINT DEFAULT 1
)
RETURNS VOID
LANGUAGE SQL AS $$
    INSERT INTO bigram_stats (left_id, right_id, count)
    VALUES (p_left_id, p_right_id, p_count)
    ON CONFLICT (left_id, right_id)
    DO UPDATE SET count = bigram_stats.count + p_count;
$$;

-- Increment unigram count
CREATE OR REPLACE FUNCTION increment_unigram(
    p_token_id BYTEA,
    p_count BIGINT DEFAULT 1
)
RETURNS VOID
LANGUAGE SQL AS $$
    INSERT INTO unigram_stats (token_id, count)
    VALUES (p_token_id, p_count)
    ON CONFLICT (token_id)
    DO UPDATE SET count = unigram_stats.count + p_count;
$$;

-- ============================================================================
-- Compute PMI scores from counts
-- ============================================================================

-- PMI = log(P(x,y) / (P(x) * P(y)))
--     = log(count(x,y) * N / (count(x) * count(y)))

CREATE OR REPLACE FUNCTION compute_pmi_scores()
RETURNS INTEGER
LANGUAGE PLPGSQL AS $$
DECLARE
    v_total_bigrams BIGINT;
    v_total_tokens BIGINT;
    updated_count INTEGER;
BEGIN
    -- Get totals
    SELECT SUM(count) INTO v_total_bigrams FROM bigram_stats;
    SELECT SUM(count) INTO v_total_tokens FROM unigram_stats;
    
    IF v_total_bigrams IS NULL OR v_total_bigrams = 0 THEN
        RETURN 0;
    END IF;
    
    -- Update corpus stats
    UPDATE token_corpus_stats 
    SET total_tokens = COALESCE(v_total_tokens, 0),
        total_bigrams = COALESCE(v_total_bigrams, 0)
    WHERE id = 1;
    
    -- Compute PMI for each bigram
    UPDATE bigram_stats b
    SET pmi = ln(
        (b.count::DOUBLE PRECISION * v_total_tokens * v_total_tokens) /
        (COALESCE(ul.count, 1)::DOUBLE PRECISION * COALESCE(ur.count, 1)::DOUBLE PRECISION * v_total_bigrams)
    )
    FROM unigram_stats ul, unigram_stats ur
    WHERE ul.token_id = b.left_id
      AND ur.token_id = b.right_id;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    
    RETURN updated_count;
END;
$$;

-- ============================================================================
-- Query functions for scoring
-- ============================================================================

-- Get PMI score for a bigram
CREATE OR REPLACE FUNCTION get_bigram_pmi(
    p_left_id BYTEA,
    p_right_id BYTEA
)
RETURNS DOUBLE PRECISION
LANGUAGE SQL STABLE AS $$
    SELECT COALESCE(pmi, 0.0)
    FROM bigram_stats
    WHERE left_id = p_left_id AND right_id = p_right_id;
$$;

-- Get raw count for a bigram
CREATE OR REPLACE FUNCTION get_bigram_count(
    p_left_id BYTEA,
    p_right_id BYTEA
)
RETURNS BIGINT
LANGUAGE SQL STABLE AS $$
    SELECT COALESCE(count, 0)
    FROM bigram_stats
    WHERE left_id = p_left_id AND right_id = p_right_id;
$$;

-- Get top continuations by PMI
CREATE OR REPLACE FUNCTION top_continuations_pmi(
    p_left_id BYTEA,
    k INTEGER DEFAULT 10
)
RETURNS TABLE (
    token_id BYTEA,
    label TEXT,
    pmi DOUBLE PRECISION,
    count BIGINT
)
LANGUAGE SQL STABLE AS $$
    SELECT 
        b.right_id,
        c.label,
        b.pmi,
        b.count
    FROM bigram_stats b
    JOIN composition c ON c.id = b.right_id
    WHERE b.left_id = p_left_id
      AND b.pmi IS NOT NULL
    ORDER BY b.pmi DESC
    LIMIT k;
$$;

-- ============================================================================
-- Extract bigrams from existing composition_child relationships
-- ============================================================================

CREATE OR REPLACE FUNCTION extract_bigrams_from_compositions()
RETURNS INTEGER
LANGUAGE PLPGSQL AS $$
DECLARE
    inserted_count INTEGER := 0;
    parent_rec RECORD;
    prev_child BYTEA;
    child_rec RECORD;
BEGIN
    -- Clear existing data
    TRUNCATE bigram_stats, unigram_stats;
    
    -- For each depth-2 composition (sentences containing tokens)
    FOR parent_rec IN 
        SELECT DISTINCT c.id as composition_id
        FROM composition c
        WHERE c.depth = 2
    LOOP
        prev_child := NULL;
        
        -- Iterate depth-1 children (tokens with labels) in order
        FOR child_rec IN
            SELECT cc.child_id, c.label
            FROM composition_child cc
            JOIN composition c ON c.id = cc.child_id
            WHERE cc.composition_id = parent_rec.composition_id
              AND cc.child_type = 'C'
              AND c.depth = 1
              AND c.label IS NOT NULL
              AND c.label NOT LIKE '[%'
            ORDER BY cc.ordinal
        LOOP
            IF prev_child IS NOT NULL THEN
                -- Insert bigram between consecutive tokens
                PERFORM increment_bigram(prev_child, child_rec.child_id);
                inserted_count := inserted_count + 1;
            END IF;
            
            -- Update unigram
            PERFORM increment_unigram(child_rec.child_id);
            
            prev_child := child_rec.child_id;
        END LOOP;
    END LOOP;
    
    -- Compute PMI scores
    PERFORM compute_pmi_scores();
    
    RETURN inserted_count;
END;
$$;

COMMENT ON FUNCTION extract_bigrams_from_compositions() IS
'Extract bigram statistics from existing composition_child relationships.
Run this after ingesting content to populate PMI scores.';
