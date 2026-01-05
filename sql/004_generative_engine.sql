-- =============================================================================
-- GENERATIVE ENGINE FOR 3-TABLE SUBSTRATE
-- =============================================================================
-- LLM-equivalent inference using 4D hypercube geometry:
--   Atom:        Unicode primitives (coordinate system)
--   Composition: Vocabulary + 4D centroids (Laplacian projected)
--   Relation:    Attention/PMI edges (semantic connections)
-- 
-- Note: N-dimensional model embeddings are projected to 4D during ingestion.
-- All similarity uses 4D Euclidean distance on composition.centroid.
-- =============================================================================

-- =============================================================================
-- 4D SIMILARITY OPERATIONS
-- =============================================================================
-- Centroids are POINTZM geometry: X, Y, Z, M = 4D coordinates

-- Euclidean distance between two 4D points
CREATE OR REPLACE FUNCTION centroid_distance(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
    SELECT sqrt(
        power(ST_X(p_a) - ST_X(p_b), 2) +
        power(ST_Y(p_a) - ST_Y(p_b), 2) +
        power(ST_Z(p_a) - ST_Z(p_b), 2) +
        power(ST_M(p_a) - ST_M(p_b), 2)
    )
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

-- Convert distance to similarity (inverse, normalized)
CREATE OR REPLACE FUNCTION centroid_similarity(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
    SELECT 1.0 / (1.0 + centroid_distance(p_a, p_b))
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

-- Hilbert distance (for locality-sensitive pre-filtering)
CREATE OR REPLACE FUNCTION hilbert_distance(
    p_lo_a BIGINT, p_hi_a BIGINT,
    p_lo_b BIGINT, p_hi_b BIGINT
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_diff_lo BIGINT;
    v_diff_hi BIGINT;
BEGIN
    v_diff_lo := abs(p_lo_a - p_lo_b);
    v_diff_hi := abs(p_hi_a - p_hi_b);
    -- Combine as 128-bit distance approximation
    RETURN v_diff_hi::DOUBLE PRECISION * 9223372036854775808.0 + v_diff_lo::DOUBLE PRECISION;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- =============================================================================
-- SEMANTIC SIMILARITY SEARCH (4D Centroid Distance)
-- =============================================================================

-- Find similar tokens by 4D centroid proximity
CREATE OR REPLACE FUNCTION similar_tokens(
    p_label TEXT,
    p_k INTEGER DEFAULT 10
)
RETURNS TABLE(
    label TEXT,
    similarity DOUBLE PRECISION
) AS $$
    WITH query_centroid AS (
        SELECT c.centroid
        FROM composition c
        WHERE c.label = p_label
          AND c.centroid IS NOT NULL
        LIMIT 1
    )
    SELECT 
        c.label,
        centroid_similarity(c.centroid, q.centroid) as sim
    FROM composition c
    CROSS JOIN query_centroid q
    WHERE c.label IS NOT NULL
      AND c.label != p_label
      AND c.centroid IS NOT NULL
      AND c.label NOT LIKE '[%'   -- Skip special tokens
      AND c.label NOT LIKE '##%'  -- Skip subword prefixes
    ORDER BY centroid_distance(c.centroid, q.centroid) ASC
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Find similar tokens with Hilbert pre-filtering (O(log n) range scan)
CREATE OR REPLACE FUNCTION similar_tokens_fast(
    p_label TEXT,
    p_k INTEGER DEFAULT 10,
    p_hilbert_range DOUBLE PRECISION DEFAULT 0.01
)
RETURNS TABLE(
    label TEXT,
    similarity DOUBLE PRECISION
) AS $$
DECLARE
    v_centroid GEOMETRY;
    v_hilbert_lo BIGINT;
    v_hilbert_hi BIGINT;
    v_range BIGINT;
BEGIN
    -- Get query token's centroid and Hilbert index
    SELECT c.centroid, c.hilbert_lo, c.hilbert_hi
    INTO v_centroid, v_hilbert_lo, v_hilbert_hi
    FROM composition c
    WHERE c.label = p_label
      AND c.centroid IS NOT NULL
    LIMIT 1;
    
    IF v_centroid IS NULL THEN
        RETURN;
    END IF;
    
    -- Hilbert range = fraction of 64-bit space
    v_range := (9223372036854775807 * p_hilbert_range)::BIGINT;
    
    -- Use Hilbert index range for pre-filtering, then sort by 4D distance
    RETURN QUERY
    SELECT 
        c.label,
        centroid_similarity(c.centroid, v_centroid) as sim
    FROM composition c
    WHERE c.centroid IS NOT NULL
      AND c.label IS NOT NULL
      AND c.label != p_label
      AND c.label NOT LIKE '[%'
      AND c.hilbert_lo BETWEEN v_hilbert_lo - v_range AND v_hilbert_lo + v_range
    ORDER BY centroid_distance(c.centroid, v_centroid) ASC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- PROMPT ENCODING
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

-- =============================================================================
-- CANDIDATE SCORING
-- =============================================================================

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

-- =============================================================================
-- GENERATIVE WALK (Main Inference Loop)
-- =============================================================================
-- Note: Renamed to generate_sql to avoid conflict with C extension generate()

CREATE OR REPLACE FUNCTION generate_sql(
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

-- =============================================================================
-- CONVENIENCE FUNCTIONS
-- =============================================================================

-- Generate text completion as a single string
CREATE OR REPLACE FUNCTION complete(
    p_prompt TEXT,
    p_max_tokens INTEGER DEFAULT 20
)
RETURNS TEXT AS $$
    SELECT string_agg(token, ' ' ORDER BY pos)
    FROM generate_sql(p_prompt, p_max_tokens, 0.7, 40);
$$ LANGUAGE SQL STABLE;

-- Semantic analogy: A is to B as C is to ?
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

-- Semantic midpoint: find tokens between A and B
CREATE OR REPLACE FUNCTION semantic_midpoint(
    p_a TEXT,
    p_b TEXT,
    p_k INTEGER DEFAULT 5
)
RETURNS TABLE(
    label TEXT,
    dist_to_midpoint DOUBLE PRECISION
) AS $$
DECLARE
    v_a GEOMETRY;
    v_b GEOMETRY;
    v_mid GEOMETRY;
BEGIN
    SELECT centroid INTO v_a FROM composition WHERE label = p_a AND centroid IS NOT NULL LIMIT 1;
    SELECT centroid INTO v_b FROM composition WHERE label = p_b AND centroid IS NOT NULL LIMIT 1;
    
    IF v_a IS NULL OR v_b IS NULL THEN
        RAISE WARNING 'Tokens not found with centroids';
        RETURN;
    END IF;
    
    -- Midpoint
    v_mid := ST_SetSRID(ST_MakePoint(
        (ST_X(v_a) + ST_X(v_b)) / 2,
        (ST_Y(v_a) + ST_Y(v_b)) / 2,
        (ST_Z(v_a) + ST_Z(v_b)) / 2,
        (ST_M(v_a) + ST_M(v_b)) / 2
    ), 0);
    
    RETURN QUERY
    SELECT 
        c.label,
        centroid_distance(c.centroid, v_mid)
    FROM composition c
    WHERE c.centroid IS NOT NULL
      AND c.label IS NOT NULL
      AND c.label NOT IN (p_a, p_b)
      AND c.label NOT LIKE '[%'
    ORDER BY centroid_distance(c.centroid, v_mid) ASC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- DATABASE STATISTICS
-- =============================================================================

CREATE OR REPLACE FUNCTION gen_db_stats()
RETURNS TABLE(
    stat_name TEXT,
    stat_value BIGINT
) AS $$
    SELECT 'atoms'::TEXT, COUNT(*) FROM atom
    UNION ALL
    SELECT 'compositions', COUNT(*) FROM composition
    UNION ALL
    SELECT 'compositions_with_centroid', COUNT(*) FROM composition WHERE centroid IS NOT NULL
    UNION ALL
    SELECT 'relations', COUNT(*) FROM relation
    UNION ALL
    SELECT 'attention_edges', COUNT(*) FROM relation WHERE relation_type = 'A'
    UNION ALL
    SELECT 'pmi_edges', COUNT(*) FROM relation WHERE relation_type = 'S';
$$ LANGUAGE SQL STABLE;
