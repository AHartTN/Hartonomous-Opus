-- =============================================================================
-- GENERATIVE QUERY FUNCTIONS
-- =============================================================================
-- Token generation and prompting capabilities for the hypercube
-- =============================================================================

-- Tokenize text into composition IDs
CREATE OR REPLACE FUNCTION encode_prompt(p_text TEXT)
RETURNS TABLE(
    pos INTEGER,
    token TEXT,
    composition_id BYTEA,
    has_centroid BOOLEAN
) AS $$
DECLARE
    v_words TEXT[];
    v_word TEXT;
    v_pos INTEGER := 0;
    v_comp_id BYTEA;
    v_has_centroid BOOLEAN;
BEGIN
    -- Simple word tokenization
    v_words := regexp_split_to_array(lower(p_text), E'[\\s,.!?;:\'"()\\[\\]{}]+');

    FOREACH v_word IN ARRAY v_words LOOP
        IF v_word = '' THEN
            CONTINUE;
        END IF;

        -- Look up in vocabulary
        SELECT c.id, c.centroid IS NOT NULL
        INTO v_comp_id, v_has_centroid
        FROM composition c
        WHERE c.label = v_word
        LIMIT 1;

        pos := v_pos;
        token := v_word;
        composition_id := v_comp_id;
        has_centroid := COALESCE(v_has_centroid, FALSE);
        RETURN NEXT;

        v_pos := v_pos + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- Score candidate next tokens given context (using 4D centroids)
CREATE OR REPLACE FUNCTION score_candidates(
    p_context_ids BYTEA[],
    p_k INTEGER DEFAULT 50
)
RETURNS TABLE(
    candidate_id BYTEA,
    candidate_label TEXT,
    centroid_score DOUBLE PRECISION,   -- 4D proximity to context
    attention_score DOUBLE PRECISION,  -- Attention relation weights
    pmi_score DOUBLE PRECISION,        -- Bigram PMI score
    combined_score DOUBLE PRECISION
) AS $$
    WITH context_centroid AS (
        -- Get centroid of last context token
        SELECT c.centroid, c.id
        FROM composition c
        WHERE c.id = p_context_ids[array_length(p_context_ids, 1)]
          AND c.centroid IS NOT NULL
        LIMIT 1
    ),
    attention_from_context AS (
        -- Sum attention weights from context
        SELECT r.target_id, SUM(r.weight) as total_attention
        FROM relation r
        WHERE r.source_id = ANY(p_context_ids)
          AND r.relation_type = 'A'
        GROUP BY r.target_id
    ),
    pmi_from_last AS (
        -- PMI (sequence relation) from last token
        SELECT r.target_id, r.weight as pmi_weight
        FROM relation r
        CROSS JOIN context_centroid ctx
        WHERE r.source_id = ctx.id
          AND r.relation_type = 'S'
    ),
    candidates AS (
        -- Vocabulary tokens with centroids, not in context
        SELECT c.id, c.label, c.centroid
        FROM composition c
        WHERE c.centroid IS NOT NULL
          AND c.label IS NOT NULL
          AND c.label NOT LIKE '[%'
          AND NOT (c.id = ANY(p_context_ids))
        LIMIT 5000
    )
    SELECT
        cand.id,
        cand.label,
        centroid_similarity(cand.centroid, ctx.centroid) as cent_score,
        COALESCE(attn.total_attention, 0) as attn_score,
        COALESCE(pmi.pmi_weight, 0) as pmi_score,
        -- Combined: 40% centroid, 30% PMI, 30% attention
        centroid_similarity(cand.centroid, ctx.centroid) * 0.4 +
        COALESCE(pmi.pmi_weight, 0) * 0.3 +
        COALESCE(attn.total_attention, 0) * 0.3 as combined
    FROM candidates cand
    CROSS JOIN context_centroid ctx
    LEFT JOIN attention_from_context attn ON attn.target_id = cand.id
    LEFT JOIN pmi_from_last pmi ON pmi.target_id = cand.id
    ORDER BY combined DESC
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Generate token completion as a single string
CREATE OR REPLACE FUNCTION complete(
    p_prompt TEXT,
    p_max_tokens INTEGER DEFAULT 20
)
RETURNS TEXT AS $$
    SELECT string_agg(token, ' ' ORDER BY pos)
    FROM generate_tokens(p_prompt, p_max_tokens, 0.7, 40);
$$ LANGUAGE SQL STABLE;

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

-- Semantic analogy: A:B::C:? using vector arithmetic
CREATE OR REPLACE FUNCTION vector_analogy(
    p_a TEXT,
    p_b TEXT,
    p_c TEXT,
    p_k INTEGER DEFAULT 5
)
RETURNS TABLE(
    label TEXT,
    score DOUBLE PRECISION
) AS $$
DECLARE
    v_a GEOMETRY;
    v_b GEOMETRY;
    v_c GEOMETRY;
    v_target_x DOUBLE PRECISION;
    v_target_y DOUBLE PRECISION;
    v_target_z DOUBLE PRECISION;
    v_target_m DOUBLE PRECISION;
    v_target GEOMETRY;
BEGIN
    -- Get centroids
    SELECT centroid INTO v_a FROM composition WHERE label = p_a AND centroid IS NOT NULL LIMIT 1;
    SELECT centroid INTO v_b FROM composition WHERE label = p_b AND centroid IS NOT NULL LIMIT 1;
    SELECT centroid INTO v_c FROM composition WHERE label = p_c AND centroid IS NOT NULL LIMIT 1;

    IF v_a IS NULL OR v_b IS NULL OR v_c IS NULL THEN
        RAISE WARNING 'One or more tokens not found with centroid';
        RETURN;
    END IF;

    -- Compute target: C + (B - A)
    v_target_x := ST_X(v_c) + (ST_X(v_b) - ST_X(v_a));
    v_target_y := ST_Y(v_c) + (ST_Y(v_b) - ST_Y(v_a));
    v_target_z := ST_Z(v_c) + (ST_Z(v_b) - ST_Z(v_a));
    v_target_m := ST_M(v_c) + (ST_M(v_b) - ST_M(v_a));

    v_target := ST_SetSRID(ST_MakePoint(v_target_x, v_target_y, v_target_z, v_target_m), 0);

    -- Find nearest to target
    RETURN QUERY
    SELECT
        c.label,
        1.0 / (1.0 + centroid_distance(c.centroid, v_target)) as sim
    FROM composition c
    WHERE c.centroid IS NOT NULL
      AND c.label IS NOT NULL
      AND c.label NOT IN (p_a, p_b, p_c)
      AND c.label NOT LIKE '[%'
    ORDER BY centroid_distance(c.centroid, v_target) ASC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;