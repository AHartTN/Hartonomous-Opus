-- =============================================================================
-- Model Ingestion Infrastructure
-- Complete package for ingesting AI models and building query capabilities
-- =============================================================================
--
-- This file provides:
-- 1. Model metadata tracking
-- 2. Vocabulary ingestion as compositions
-- 3. BPE merge rules as semantic edges
-- 4. Token relationship queries
-- 5. Model comparison functions
--
-- All coordinates use full 32-bit precision (no normalization/quantization)
-- All geometries use SRID 0 (raw 4D space)

BEGIN;

-- =============================================================================
-- Model Registry: Track ingested models
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    model_type      TEXT,  -- 'transformer', 'embedding', 'tokenizer', etc.
    source_path     TEXT,
    vocab_size      INTEGER,
    embedding_dim   INTEGER,
    hidden_dim      INTEGER,
    num_layers      INTEGER,
    metadata        JSONB,
    ingested_at     TIMESTAMPTZ DEFAULT now(),
    atom_count      BIGINT,  -- Updated after ingestion
    composition_count BIGINT
);

CREATE INDEX IF NOT EXISTS idx_model_registry_name ON model_registry(name);

-- =============================================================================
-- Token Vocabulary: Map model tokens to atom IDs
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_vocabulary (
    model_id        INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    token_index     INTEGER NOT NULL,  -- Original vocab index
    token_text      TEXT NOT NULL,
    atom_id         BYTEA REFERENCES atom(id),
    is_special      BOOLEAN DEFAULT FALSE,  -- [CLS], [SEP], [PAD], etc.
    frequency       BIGINT,  -- If available from training
    PRIMARY KEY (model_id, token_index)
);

CREATE INDEX IF NOT EXISTS idx_model_vocab_atom ON model_vocabulary(atom_id);
CREATE INDEX IF NOT EXISTS idx_model_vocab_text ON model_vocabulary(token_text);

-- =============================================================================
-- BPE Merge Rules: Track tokenizer composition rules
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_bpe_merges (
    model_id        INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    merge_rank      INTEGER NOT NULL,  -- Priority (lower = more common)
    left_token      TEXT NOT NULL,
    right_token     TEXT NOT NULL,
    merged_token    TEXT NOT NULL,
    left_atom_id    BYTEA,
    right_atom_id   BYTEA,
    merged_atom_id  BYTEA,
    PRIMARY KEY (model_id, merge_rank)
);

CREATE INDEX IF NOT EXISTS idx_model_bpe_merged ON model_bpe_merges(merged_atom_id);

-- =============================================================================
-- Ingestion Functions
-- =============================================================================

-- Register a new model
CREATE OR REPLACE FUNCTION register_model(
    p_name TEXT,
    p_model_type TEXT DEFAULT 'unknown',
    p_source_path TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
) RETURNS INTEGER AS $$
DECLARE
    v_id INTEGER;
BEGIN
    INSERT INTO model_registry (name, model_type, source_path, metadata)
    VALUES (p_name, p_model_type, p_source_path, p_metadata)
    ON CONFLICT (name) DO UPDATE SET
        model_type = EXCLUDED.model_type,
        source_path = EXCLUDED.source_path,
        metadata = model_registry.metadata || EXCLUDED.metadata
    RETURNING id INTO v_id;
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Ingest a vocabulary token (creates composition if needed)
CREATE OR REPLACE FUNCTION ingest_vocab_token(
    p_model_id INTEGER,
    p_token_index INTEGER,
    p_token_text TEXT,
    p_is_special BOOLEAN DEFAULT FALSE
) RETURNS BYTEA AS $$
DECLARE
    v_atom_id BYTEA;
    v_existing BYTEA;
BEGIN
    -- Check if this token already exists as an atom
    SELECT id INTO v_existing
    FROM atom
    WHERE depth > 0 AND atom_reconstruct_text(id) = p_token_text
    LIMIT 1;
    
    IF v_existing IS NOT NULL THEN
        v_atom_id := v_existing;
    ELSE
        -- Token doesn't exist - it will be created during text ingestion
        -- For now, just record the mapping
        v_atom_id := NULL;
    END IF;
    
    -- Record in vocabulary table
    INSERT INTO model_vocabulary (model_id, token_index, token_text, atom_id, is_special)
    VALUES (p_model_id, p_token_index, p_token_text, v_atom_id, p_is_special)
    ON CONFLICT (model_id, token_index) DO UPDATE SET
        token_text = EXCLUDED.token_text,
        atom_id = COALESCE(EXCLUDED.atom_id, model_vocabulary.atom_id),
        is_special = EXCLUDED.is_special;
    
    RETURN v_atom_id;
END;
$$ LANGUAGE plpgsql;

-- Update model statistics after ingestion
CREATE OR REPLACE FUNCTION update_model_stats(p_model_id INTEGER) RETURNS VOID AS $$
BEGIN
    UPDATE model_registry SET
        vocab_size = (SELECT COUNT(*) FROM model_vocabulary WHERE model_id = p_model_id),
        atom_count = (SELECT COUNT(*) FROM atom),
        composition_count = (SELECT COUNT(*) FROM atom WHERE depth > 0)
    WHERE id = p_model_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Token Query Functions (for LLM-style operations)
-- =============================================================================

-- Tokenize text using known vocabulary
CREATE OR REPLACE FUNCTION tokenize_text(
    p_model_id INTEGER,
    p_text TEXT
) RETURNS TABLE(tok_position INTEGER, token_text TEXT, token_index INTEGER, atom_id BYTEA) AS $$
DECLARE
    v_pos INTEGER := 0;
    v_remaining TEXT := p_text;
    v_match RECORD;
    v_found BOOLEAN;
BEGIN
    -- Greedy longest-match tokenization
    WHILE length(v_remaining) > 0 LOOP
        v_found := FALSE;
        
        -- Try to find longest matching token
        FOR v_match IN
            SELECT mv.token_text, mv.token_index, mv.atom_id
            FROM model_vocabulary mv
            WHERE mv.model_id = p_model_id
              AND v_remaining LIKE mv.token_text || '%'
            ORDER BY length(mv.token_text) DESC
            LIMIT 1
        LOOP
            tok_position := v_pos;
            token_text := v_match.token_text;
            token_index := v_match.token_index;
            atom_id := v_match.atom_id;
            RETURN NEXT;
            
            v_remaining := substring(v_remaining FROM length(v_match.token_text) + 1);
            v_pos := v_pos + 1;
            v_found := TRUE;
        END LOOP;
        
        -- If no match, take single character (unknown token)
        IF NOT v_found THEN
            tok_position := v_pos;
            token_text := left(v_remaining, 1);
            token_index := -1;  -- Unknown
            atom_id := NULL;
            RETURN NEXT;
            
            v_remaining := substring(v_remaining FROM 2);
            v_pos := v_pos + 1;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- Find similar tokens by Hilbert proximity
CREATE OR REPLACE FUNCTION find_similar_tokens(
    p_model_id INTEGER,
    p_token_text TEXT,
    p_k INTEGER DEFAULT 10
) RETURNS TABLE(token_text TEXT, token_index INTEGER, hilbert_distance NUMERIC) AS $$
    WITH target AS (
        SELECT mv.atom_id
        FROM model_vocabulary mv
        WHERE mv.model_id = p_model_id AND mv.token_text = p_token_text
        LIMIT 1
    )
    SELECT
        mv.token_text,
        mv.token_index,
        ABS(a.hilbert_lo::NUMERIC - t_a.hilbert_lo::NUMERIC) +
        ABS(a.hilbert_hi::NUMERIC - t_a.hilbert_hi::NUMERIC) * 9223372036854775808::NUMERIC
    FROM target t
    JOIN atom t_a ON t_a.id = t.atom_id
    JOIN model_vocabulary mv ON mv.model_id = p_model_id
    JOIN atom a ON a.id = mv.atom_id
    WHERE mv.atom_id != t.atom_id
    ORDER BY
        ABS(a.hilbert_hi - t_a.hilbert_hi),
        ABS(a.hilbert_lo - t_a.hilbert_lo)
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Get token co-occurrence (tokens that appear in same compositions)
CREATE OR REPLACE FUNCTION get_token_cooccurrence(
    p_model_id INTEGER,
    p_token_text TEXT,
    p_limit INTEGER DEFAULT 20
) RETURNS TABLE(co_token TEXT, co_count BIGINT, avg_distance DOUBLE PRECISION) AS $$
    WITH target_atom AS (
        SELECT atom_id FROM model_vocabulary
        WHERE model_id = p_model_id AND token_text = p_token_text
        LIMIT 1
    ),
    containing_compositions AS (
        SELECT DISTINCT a.id as comp_id
        FROM atom a, target_atom t
        WHERE t.atom_id = ANY(a.children)
    ),
    sibling_atoms AS (
        SELECT DISTINCT c.child_id, a.id as comp_id
        FROM containing_compositions cc
        JOIN atom a ON a.id = cc.comp_id
        CROSS JOIN LATERAL unnest(a.children) AS c(child_id)
        WHERE c.child_id != (SELECT atom_id FROM target_atom)
    )
    SELECT
        mv.token_text,
        COUNT(*) as co_count,
        AVG(atom_distance(sa.child_id, (SELECT atom_id FROM target_atom)))
    FROM sibling_atoms sa
    JOIN model_vocabulary mv ON mv.atom_id = sa.child_id AND mv.model_id = p_model_id
    GROUP BY mv.token_text
    ORDER BY COUNT(*) DESC
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Semantic Edge Queries
-- =============================================================================

-- Get all semantic edges for a token (BPE relationships)
CREATE OR REPLACE FUNCTION get_token_edges(
    p_model_id INTEGER,
    p_token_text TEXT
) RETURNS TABLE(
    direction TEXT,
    related_token TEXT,
    merge_rank INTEGER,
    merged_result TEXT
) AS $$
    -- Token appears on left side of merge
    SELECT 
        'left'::TEXT,
        right_token,
        merge_rank,
        merged_token
    FROM model_bpe_merges
    WHERE model_id = p_model_id AND left_token = p_token_text
    
    UNION ALL
    
    -- Token appears on right side of merge
    SELECT 
        'right'::TEXT,
        left_token,
        merge_rank,
        merged_token
    FROM model_bpe_merges
    WHERE model_id = p_model_id AND right_token = p_token_text
    
    UNION ALL
    
    -- Token is result of merge
    SELECT 
        'merged_from'::TEXT,
        left_token || ' + ' || right_token,
        merge_rank,
        merged_token
    FROM model_bpe_merges
    WHERE model_id = p_model_id AND merged_token = p_token_text
    
    ORDER BY merge_rank;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Model Comparison Functions
-- =============================================================================

-- Compare vocabularies between two models
CREATE OR REPLACE FUNCTION compare_vocabularies(
    p_model1_id INTEGER,
    p_model2_id INTEGER
) RETURNS TABLE(
    token_text TEXT,
    in_model1 BOOLEAN,
    in_model2 BOOLEAN,
    model1_index INTEGER,
    model2_index INTEGER
) AS $$
    SELECT
        COALESCE(m1.token_text, m2.token_text),
        m1.token_text IS NOT NULL,
        m2.token_text IS NOT NULL,
        m1.token_index,
        m2.token_index
    FROM model_vocabulary m1
    FULL OUTER JOIN model_vocabulary m2 
        ON m1.token_text = m2.token_text AND m2.model_id = p_model2_id
    WHERE m1.model_id = p_model1_id OR m1.model_id IS NULL
    ORDER BY COALESCE(m1.token_text, m2.token_text);
$$ LANGUAGE SQL STABLE;

-- Find unique tokens in model (not in any other model)
CREATE OR REPLACE FUNCTION find_unique_tokens(p_model_id INTEGER)
RETURNS TABLE(token_text TEXT, token_index INTEGER) AS $$
    SELECT mv.token_text, mv.token_index
    FROM model_vocabulary mv
    WHERE mv.model_id = p_model_id
      AND NOT EXISTS (
          SELECT 1 FROM model_vocabulary other
          WHERE other.model_id != p_model_id
            AND other.token_text = mv.token_text
      )
    ORDER BY mv.token_index;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Views for Model Analysis
-- =============================================================================

CREATE OR REPLACE VIEW v_model_summary AS
SELECT
    mr.id,
    mr.name,
    mr.model_type,
    mr.vocab_size,
    mr.embedding_dim,
    mr.ingested_at,
    COUNT(DISTINCT mv.token_index) as actual_vocab_count,
    COUNT(DISTINCT mbm.merge_rank) as bpe_merge_count,
    COUNT(DISTINCT mv.atom_id) FILTER (WHERE mv.atom_id IS NOT NULL) as linked_atoms
FROM model_registry mr
LEFT JOIN model_vocabulary mv ON mv.model_id = mr.id
LEFT JOIN model_bpe_merges mbm ON mbm.model_id = mr.id
GROUP BY mr.id;

CREATE OR REPLACE VIEW v_vocabulary_coverage AS
SELECT
    mv.model_id,
    mr.name as model_name,
    COUNT(*) as total_tokens,
    COUNT(mv.atom_id) as linked_tokens,
    ROUND(COUNT(mv.atom_id)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2) as coverage_pct
FROM model_vocabulary mv
JOIN model_registry mr ON mr.id = mv.model_id
GROUP BY mv.model_id, mr.name;

-- =============================================================================
-- LLM-Style Query Interface
-- =============================================================================

-- Prompt-style query: find relevant compositions
CREATE OR REPLACE FUNCTION query_semantic(
    p_query_text TEXT,
    p_model_id INTEGER DEFAULT NULL,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    atom_id BYTEA,
    content TEXT,
    relevance_score DOUBLE PRECISION,
    depth INTEGER
) AS $$
DECLARE
    v_query_hash BYTEA;
    v_query_x DOUBLE PRECISION;
    v_query_y DOUBLE PRECISION;
    v_query_z DOUBLE PRECISION;
    v_query_m DOUBLE PRECISION;
BEGIN
    -- Hash the query to get target coordinates
    -- In production, this would use the C++ blake3 function
    v_query_hash := digest(p_query_text, 'sha256');
    
    -- Extract coordinate from hash bytes (first 16 bytes -> 4 floats)
    v_query_x := get_byte(v_query_hash, 0)::DOUBLE PRECISION * 16777216 +
                 get_byte(v_query_hash, 1)::DOUBLE PRECISION * 65536 +
                 get_byte(v_query_hash, 2)::DOUBLE PRECISION * 256 +
                 get_byte(v_query_hash, 3)::DOUBLE PRECISION;
    v_query_y := get_byte(v_query_hash, 4)::DOUBLE PRECISION * 16777216 +
                 get_byte(v_query_hash, 5)::DOUBLE PRECISION * 65536 +
                 get_byte(v_query_hash, 6)::DOUBLE PRECISION * 256 +
                 get_byte(v_query_hash, 7)::DOUBLE PRECISION;
    v_query_z := get_byte(v_query_hash, 8)::DOUBLE PRECISION * 16777216 +
                 get_byte(v_query_hash, 9)::DOUBLE PRECISION * 65536 +
                 get_byte(v_query_hash, 10)::DOUBLE PRECISION * 256 +
                 get_byte(v_query_hash, 11)::DOUBLE PRECISION;
    v_query_m := get_byte(v_query_hash, 12)::DOUBLE PRECISION * 16777216 +
                 get_byte(v_query_hash, 13)::DOUBLE PRECISION * 65536 +
                 get_byte(v_query_hash, 14)::DOUBLE PRECISION * 256 +
                 get_byte(v_query_hash, 15)::DOUBLE PRECISION;
    
    -- Find nearest compositions by spatial distance
    RETURN QUERY
    SELECT
        a.id,
        atom_reconstruct_text(a.id),
        1.0 / (1.0 + sqrt(
            power(ST_X(a.geom) - v_query_x, 2) +
            power(ST_Y(a.geom) - v_query_y, 2) +
            power(ST_Z(a.geom) - v_query_z, 2) +
            power(ST_M(a.geom) - v_query_m, 2)
        )) as relevance,
        a.depth
    FROM atom a
    WHERE a.depth > 0
    ORDER BY sqrt(
        power(ST_X(a.geom) - v_query_x, 2) +
        power(ST_Y(a.geom) - v_query_y, 2) +
        power(ST_Z(a.geom) - v_query_z, 2) +
        power(ST_M(a.geom) - v_query_m, 2)
    )
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

COMMIT;
