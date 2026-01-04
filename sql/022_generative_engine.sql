-- =============================================================================
-- GENERATIVE ENGINE FOR 4-TABLE SUBSTRATE
-- =============================================================================
-- This is the LLM-equivalent inference engine using:
--   Atom:        Unicode primitives (coordinate system)
--   Composition: Vocabulary + PMI patterns (grammar)
--   Relation:    Attention edges (directional influence)
--   Shape:       Embeddings (semantic geometry)
-- =============================================================================

BEGIN;

-- =============================================================================
-- VECTOR OPERATIONS ON SHAPE EMBEDDINGS
-- =============================================================================
-- Embeddings are LineStringZM where X=dim_index, Y=value, Z=0, M=0
-- Need proper L2 distance, not bounding box distance

-- Extract embedding as array of values (Y coordinates)
CREATE OR REPLACE FUNCTION embedding_to_array(p_embedding GEOMETRY)
RETURNS REAL[] AS $$
    SELECT array_agg(ST_Y(geom) ORDER BY ST_X(geom))::REAL[]
    FROM ST_DumpPoints(p_embedding) dp(path, geom);
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

-- L2 (Euclidean) distance between two embeddings
CREATE OR REPLACE FUNCTION embedding_l2_distance(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_arr_a REAL[];
    v_arr_b REAL[];
    v_sum DOUBLE PRECISION := 0;
    v_i INTEGER;
BEGIN
    v_arr_a := embedding_to_array(p_a);
    v_arr_b := embedding_to_array(p_b);
    
    IF array_length(v_arr_a, 1) != array_length(v_arr_b, 1) THEN
        RETURN NULL;
    END IF;
    
    FOR v_i IN 1..array_length(v_arr_a, 1) LOOP
        v_sum := v_sum + power(v_arr_a[v_i] - v_arr_b[v_i], 2);
    END LOOP;
    
    RETURN sqrt(v_sum);
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Cosine similarity between two embeddings
CREATE OR REPLACE FUNCTION embedding_cosine_similarity(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_arr_a REAL[];
    v_arr_b REAL[];
    v_dot DOUBLE PRECISION := 0;
    v_norm_a DOUBLE PRECISION := 0;
    v_norm_b DOUBLE PRECISION := 0;
    v_i INTEGER;
BEGIN
    v_arr_a := embedding_to_array(p_a);
    v_arr_b := embedding_to_array(p_b);
    
    IF array_length(v_arr_a, 1) != array_length(v_arr_b, 1) THEN
        RETURN NULL;
    END IF;
    
    FOR v_i IN 1..array_length(v_arr_a, 1) LOOP
        v_dot := v_dot + v_arr_a[v_i] * v_arr_b[v_i];
        v_norm_a := v_norm_a + v_arr_a[v_i] * v_arr_a[v_i];
        v_norm_b := v_norm_b + v_arr_b[v_i] * v_arr_b[v_i];
    END LOOP;
    
    IF v_norm_a = 0 OR v_norm_b = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN v_dot / (sqrt(v_norm_a) * sqrt(v_norm_b));
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- =============================================================================
-- SEMANTIC SIMILARITY SEARCH (Proper Vector Distance)
-- =============================================================================

-- Find similar tokens by embedding cosine similarity
CREATE OR REPLACE FUNCTION similar_tokens(
    p_label TEXT,
    p_k INTEGER DEFAULT 10,
    p_model TEXT DEFAULT 'minilm'
)
RETURNS TABLE(
    label TEXT,
    similarity DOUBLE PRECISION
) AS $$
    WITH query_emb AS (
        SELECT s.embedding
        FROM shape s
        JOIN composition c ON c.id = s.entity_id
        WHERE c.label = p_label AND s.model_name = p_model
        LIMIT 1
    )
    SELECT c.label, embedding_cosine_similarity(s.embedding, q.embedding) as sim
    FROM shape s
    JOIN composition c ON c.id = s.entity_id
    CROSS JOIN query_emb q
    WHERE c.label IS NOT NULL
      AND c.label != p_label
      AND s.model_name = p_model
      AND c.label NOT LIKE '[%'  -- Skip special tokens
      AND c.label NOT LIKE '##%' -- Skip subword tokens for cleaner results
    ORDER BY embedding_cosine_similarity(s.embedding, q.embedding) DESC
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- PROMPT ENCODING
-- =============================================================================

-- Tokenize text into composition IDs (matching MiniLM vocabulary)
CREATE OR REPLACE FUNCTION encode_prompt(p_text TEXT)
RETURNS TABLE(
    pos INTEGER,
    token TEXT,
    composition_id BYTEA,
    has_embedding BOOLEAN
) AS $$
DECLARE
    v_words TEXT[];
    v_word TEXT;
    v_pos INTEGER := 0;
    v_comp_id BYTEA;
    v_has_emb BOOLEAN;
BEGIN
    -- Simple word tokenization (split on spaces and punctuation)
    v_words := regexp_split_to_array(lower(p_text), E'[\\s,.!?;:\'"()\\[\\]{}]+');
    
    FOREACH v_word IN ARRAY v_words LOOP
        IF v_word = '' THEN
            CONTINUE;
        END IF;
        
        -- Look up in vocabulary
        SELECT c.id INTO v_comp_id
        FROM composition c
        WHERE c.label = v_word
        LIMIT 1;
        
        -- Check if it has an embedding
        SELECT EXISTS(
            SELECT 1 FROM shape s WHERE s.entity_id = v_comp_id
        ) INTO v_has_emb;
        
        pos := v_pos;
        token := v_word;
        composition_id := v_comp_id;
        has_embedding := COALESCE(v_has_emb, FALSE);
        RETURN NEXT;
        
        v_pos := v_pos + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- CANDIDATE SCORING
-- =============================================================================

-- Score candidate next tokens given context
CREATE OR REPLACE FUNCTION score_candidates(
    p_context_ids BYTEA[],           -- Previous token IDs
    p_k INTEGER DEFAULT 50,           -- Number of candidates to score
    p_model TEXT DEFAULT 'minilm'
)
RETURNS TABLE(
    candidate_id BYTEA,
    candidate_label TEXT,
    embedding_score DOUBLE PRECISION,  -- Cosine similarity to context centroid
    attention_score DOUBLE PRECISION,  -- Sum of attention weights from context
    combined_score DOUBLE PRECISION
) AS $$
    WITH context_embeddings AS (
        -- Get embeddings for context tokens
        SELECT s.embedding, s.entity_id
        FROM shape s
        WHERE s.entity_id = ANY(p_context_ids)
          AND s.model_name = p_model
    ),
    context_last AS (
        -- Use last token's embedding as primary anchor
        SELECT embedding
        FROM shape s
        WHERE s.entity_id = p_context_ids[array_length(p_context_ids, 1)]
          AND s.model_name = p_model
        LIMIT 1
    ),
    attention_from_context AS (
        -- Sum attention weights from all context tokens to each target
        SELECT r.target_id, SUM(r.weight) as total_attention
        FROM relation r
        WHERE r.source_id = ANY(p_context_ids)
          AND r.relation_type = 'A'
          AND r.source_model = p_model
        GROUP BY r.target_id
    ),
    candidates AS (
        -- Get vocabulary tokens that aren't in context
        SELECT c.id, c.label, s.embedding
        FROM composition c
        JOIN shape s ON s.entity_id = c.id AND s.model_name = p_model
        WHERE c.label IS NOT NULL
          AND c.label NOT LIKE '[%'
          AND NOT (c.id = ANY(p_context_ids))
        LIMIT 5000  -- Pre-filter for performance
    )
    SELECT 
        cand.id,
        cand.label,
        COALESCE(embedding_cosine_similarity(cand.embedding, ctx.embedding), 0) as emb_score,
        COALESCE(attn.total_attention, 0) as attn_score,
        -- Combined score: embedding similarity + attention influence
        COALESCE(embedding_cosine_similarity(cand.embedding, ctx.embedding), 0) * 0.6 +
        COALESCE(attn.total_attention, 0) * 0.4 as combined
    FROM candidates cand
    CROSS JOIN context_last ctx
    LEFT JOIN attention_from_context attn ON attn.target_id = cand.id
    ORDER BY 
        COALESCE(embedding_cosine_similarity(cand.embedding, ctx.embedding), 0) * 0.6 +
        COALESCE(attn.total_attention, 0) * 0.4 DESC
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- GENERATIVE WALK (Main Inference Loop)
-- =============================================================================

CREATE OR REPLACE FUNCTION generate(
    p_prompt TEXT,
    p_max_tokens INTEGER DEFAULT 20,
    p_temperature REAL DEFAULT 0.7,
    p_top_k INTEGER DEFAULT 40,
    p_model TEXT DEFAULT 'minilm'
)
RETURNS TABLE(
    pos INTEGER,
    token TEXT,
    score DOUBLE PRECISION
) AS $$
DECLARE
    v_context_ids BYTEA[] := ARRAY[]::BYTEA[];
    v_pos INTEGER := 0;
    v_next_id BYTEA;
    v_next_label TEXT;
    v_next_score DOUBLE PRECISION;
    v_prompt_token RECORD;
BEGIN
    -- Step 1: Encode prompt
    FOR v_prompt_token IN SELECT * FROM encode_prompt(p_prompt) LOOP
        IF v_prompt_token.composition_id IS NOT NULL THEN
            v_context_ids := array_append(v_context_ids, v_prompt_token.composition_id);
            
            pos := v_pos;
            token := v_prompt_token.token;
            score := 1.0;
            RETURN NEXT;
            
            v_pos := v_pos + 1;
        END IF;
    END LOOP;
    
    -- Step 2: Autoregressive generation
    WHILE v_pos < (SELECT COUNT(*) FROM encode_prompt(p_prompt))::INTEGER + p_max_tokens LOOP
        
        -- Score candidates
        SELECT 
            sc.candidate_id,
            sc.candidate_label,
            sc.combined_score
        INTO v_next_id, v_next_label, v_next_score
        FROM score_candidates(v_context_ids, p_top_k, p_model) sc
        WHERE sc.candidate_id IS NOT NULL
        ORDER BY 
            -- Temperature-scaled sampling: higher temp = more randomness
            (sc.combined_score + random() * p_temperature) DESC
        LIMIT 1;
        
        IF v_next_id IS NULL THEN
            EXIT;  -- No more candidates
        END IF;
        
        -- Append to context
        v_context_ids := array_append(v_context_ids, v_next_id);
        
        -- Return generated token
        pos := v_pos;
        token := v_next_label;
        score := v_next_score;
        RETURN NEXT;
        
        v_pos := v_pos + 1;
        
        -- Stop on special tokens
        IF v_next_label IN ('[SEP]', '[EOS]', '.', '?', '!') THEN
            EXIT;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- COMPLETION API (User-Friendly Interface)
-- =============================================================================

-- Complete a prompt and return as text
CREATE OR REPLACE FUNCTION complete(
    p_prompt TEXT,
    p_max_tokens INTEGER DEFAULT 20,
    p_temperature REAL DEFAULT 0.7
)
RETURNS TEXT AS $$
    SELECT string_agg(token, ' ' ORDER BY pos)
    FROM generate(p_prompt, p_max_tokens, p_temperature);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- SEMANTIC QUERIES (Fixed)
-- =============================================================================

-- Find semantically similar tokens (using proper cosine similarity)
CREATE OR REPLACE FUNCTION semantic_similar(
    p_token TEXT,
    p_k INTEGER DEFAULT 10
)
RETURNS TABLE(token TEXT, similarity DOUBLE PRECISION) AS $$
    SELECT label, similarity FROM similar_tokens(p_token, p_k, 'minilm');
$$ LANGUAGE SQL STABLE;

-- Analogy using vector arithmetic: A is to B as C is to ?
CREATE OR REPLACE FUNCTION vector_analogy(
    p_a TEXT,  -- "king"
    p_b TEXT,  -- "man"  
    p_c TEXT,  -- "queen"
    p_k INTEGER DEFAULT 5
)
RETURNS TABLE(answer TEXT, score DOUBLE PRECISION) AS $$
    WITH emb AS (
        SELECT c.label, embedding_to_array(s.embedding) as vec
        FROM shape s
        JOIN composition c ON c.id = s.entity_id
        WHERE s.model_name = 'minilm'
          AND c.label IN (p_a, p_b, p_c)
    ),
    vectors AS (
        SELECT 
            (SELECT vec FROM emb WHERE label = p_a) as va,
            (SELECT vec FROM emb WHERE label = p_b) as vb,
            (SELECT vec FROM emb WHERE label = p_c) as vc
    ),
    -- Target = C + (A - B) ... find closest to this
    all_tokens AS (
        SELECT c.label, embedding_to_array(s.embedding) as vec
        FROM shape s
        JOIN composition c ON c.id = s.entity_id
        WHERE s.model_name = 'minilm'
          AND c.label IS NOT NULL
          AND c.label NOT IN (p_a, p_b, p_c)
          AND c.label NOT LIKE '[%'
          AND c.label NOT LIKE '##%'
    )
    SELECT 
        t.label,
        -- Score by how well token matches: C + (A - B)
        -- Higher = better match for the analogy pattern
        (
            SELECT SUM(
                (v.vc[i] + v.va[i] - v.vb[i]) * t.vec[i]
            ) / NULLIF(
                sqrt(SUM(power(v.vc[i] + v.va[i] - v.vb[i], 2))) *
                sqrt(SUM(power(t.vec[i], 2)))
            , 0)
            FROM generate_series(1, array_length(t.vec, 1)) i
            CROSS JOIN vectors v
        )::DOUBLE PRECISION as analogy_score
    FROM all_tokens t
    ORDER BY 2 DESC NULLS LAST
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

COMMIT;
