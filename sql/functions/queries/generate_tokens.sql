-- =============================================================================
-- GENERATIVE QUERY FUNCTIONS
-- =============================================================================
-- Token generation and prompting capabilities for the hypercube
-- =============================================================================

-- Main token generation function
CREATE OR REPLACE FUNCTION generate_tokens(
    p_prompt TEXT,
    p_max_tokens INTEGER DEFAULT 20,
    p_temperature REAL DEFAULT 0.7,
    p_top_k INTEGER DEFAULT 40
)
RETURNS TABLE(
    pos INTEGER,
    token TEXT,
    score DOUBLE PRECISION
) AS $$
DECLARE
    v_context_ids BYTEA[] := ARRAY[]::BYTEA[];
    v_next_id BYTEA;
    v_next_label TEXT;
    v_next_score DOUBLE PRECISION;
    v_pos INTEGER := 0;
    v_enc RECORD;
    v_cand RECORD;
BEGIN
    -- Encode prompt to context IDs
    FOR v_enc IN SELECT * FROM encode_prompt(p_prompt) LOOP
        IF v_enc.composition_id IS NOT NULL THEN
            v_context_ids := array_append(v_context_ids, v_enc.composition_id);
        END IF;
    END LOOP;

    IF array_length(v_context_ids, 1) IS NULL OR array_length(v_context_ids, 1) = 0 THEN
        RAISE WARNING 'No tokens found in prompt';
        RETURN;
    END IF;

    -- Generate tokens
    FOR v_pos IN 1..p_max_tokens LOOP
        -- Get top candidates
        SELECT candidate_id, candidate_label, combined_score
        INTO v_next_id, v_next_label, v_next_score
        FROM score_candidates(v_context_ids, p_top_k)
        ORDER BY combined_score DESC
        LIMIT 1;

        EXIT WHEN v_next_id IS NULL;

        -- Return token
        pos := v_pos;
        token := v_next_label;
        score := v_next_score;
        RETURN NEXT;

        -- Update context
        v_context_ids := array_append(v_context_ids, v_next_id);

        -- Stop on sentence-ending punctuation
        EXIT WHEN v_next_label IN ('.', '!', '?', '[SEP]', '[EOS]');
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;