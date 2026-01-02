-- Model Ingestion for Hartonomous Hypercube
-- Ingests AI model vocabularies, configs, and metadata into the substrate
-- Each vocab token becomes a composition (trajectory) of atoms
-- Single-char tokens reference atoms directly
-- Multi-char tokens become LINESTRINGZM trajectories

BEGIN;

-- ============================================================================
-- MODEL REGISTRY: Track ingested models
-- ============================================================================
CREATE TABLE IF NOT EXISTS model (
    id              blake3_hash PRIMARY KEY,
    name            TEXT NOT NULL,
    model_type      TEXT NOT NULL,  -- 'wordpiece', 'bpe', 'unigram', 'char', etc.
    vocab_size      INTEGER NOT NULL,
    config          JSONB,
    source_path     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    UNIQUE(name, model_type)
);

-- ============================================================================
-- VOCAB TOKENS: Individual vocabulary entries from a model
-- Each token is a composition that references atoms (for chars) or other tokens
-- ============================================================================
CREATE TABLE IF NOT EXISTS vocab_token (
    id              blake3_hash PRIMARY KEY,
    model_id        blake3_hash NOT NULL REFERENCES model(id) ON DELETE CASCADE,
    token_index     INTEGER NOT NULL,           -- Original index in vocab
    token_text      TEXT NOT NULL,              -- The actual token string
    is_special      BOOLEAN NOT NULL DEFAULT false,  -- [PAD], [CLS], etc.
    is_subword      BOOLEAN NOT NULL DEFAULT false,  -- ##prefix tokens
    
    -- Computed trajectory centroid in 4D space
    coords          GEOMETRY(POINTZM, 0) NOT NULL,
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,
    
    -- Link to composition in relation table
    composition_id  blake3_hash REFERENCES relation(id),
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    UNIQUE(model_id, token_index)
);

-- Indexes for vocab queries
CREATE INDEX IF NOT EXISTS idx_vocab_token_model ON vocab_token(model_id);
CREATE INDEX IF NOT EXISTS idx_vocab_token_text ON vocab_token(token_text);
CREATE INDEX IF NOT EXISTS idx_vocab_token_coords ON vocab_token USING GIST(coords);
CREATE INDEX IF NOT EXISTS idx_vocab_token_hilbert ON vocab_token(hilbert_hi, hilbert_lo);
CREATE INDEX IF NOT EXISTS idx_vocab_token_special ON vocab_token(is_special) WHERE is_special;
CREATE INDEX IF NOT EXISTS idx_vocab_token_subword ON vocab_token(is_subword) WHERE is_subword;

-- ============================================================================
-- TOKEN EDGES: Relationships between tokens (n-gram adjacency, frequency)
-- These are the "beaten paths" - frequently co-occurring token pairs
-- ============================================================================
CREATE TABLE IF NOT EXISTS token_edge (
    id              blake3_hash PRIMARY KEY,
    model_id        blake3_hash NOT NULL REFERENCES model(id) ON DELETE CASCADE,
    
    -- Source and target tokens
    src_token_id    blake3_hash NOT NULL REFERENCES vocab_token(id),
    dst_token_id    blake3_hash NOT NULL REFERENCES vocab_token(id),
    
    -- Edge as LINESTRINGZM (trajectory from src to dst)
    trajectory      GEOMETRY(LINESTRINGZM, 0) NOT NULL,
    
    -- Edge weight/frequency (how often this pair co-occurs)
    weight          DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    frequency       BIGINT NOT NULL DEFAULT 1,
    
    -- Direction matters for language
    ordinal         INTEGER NOT NULL DEFAULT 0,  -- Position in sequence
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    UNIQUE(model_id, src_token_id, dst_token_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_token_edge_model ON token_edge(model_id);
CREATE INDEX IF NOT EXISTS idx_token_edge_src ON token_edge(src_token_id);
CREATE INDEX IF NOT EXISTS idx_token_edge_dst ON token_edge(dst_token_id);
CREATE INDEX IF NOT EXISTS idx_token_edge_trajectory ON token_edge USING GIST(trajectory);
CREATE INDEX IF NOT EXISTS idx_token_edge_weight ON token_edge(weight DESC);

-- ============================================================================
-- FUNCTIONS: Ingest model vocabulary
-- ============================================================================

-- Hash a token string with its model context for unique identification
CREATE OR REPLACE FUNCTION hypercube_hash_token(
    p_model_id bytea,
    p_token_text text,
    p_token_index integer
) RETURNS bytea
AS $$
DECLARE
    v_data bytea;
BEGIN
    -- Combine model_id + token_index + token_text for unique hash
    v_data := p_model_id || int4send(p_token_index) || convert_to(p_token_text, 'UTF8');
    RETURN hypercube_blake3(v_data);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Convert a token string to its trajectory (LINESTRINGZM of atom coords)
-- Single chars become a single point, multi-char become a line
CREATE OR REPLACE FUNCTION hypercube_token_to_trajectory(p_token_text text)
RETURNS geometry
AS $$
DECLARE
    v_coords geometry[];
    v_codepoints integer[];
    v_atom_coord geometry;
    i integer;
    v_clean_text text;
BEGIN
    -- Strip WordPiece prefix (##) if present
    v_clean_text := regexp_replace(p_token_text, '^##', '');
    
    -- Handle empty or special tokens
    IF v_clean_text IS NULL OR v_clean_text = '' THEN
        -- Return origin point for special/empty tokens
        RETURN ST_SetSRID(ST_MakePoint(0.5, 0.5, 0.5, 0.5), 0);
    END IF;
    
    v_coords := ARRAY[]::geometry[];
    
    -- Extract codepoints from token text
    FOR i IN 1..char_length(v_clean_text)
    LOOP
        SELECT coords INTO v_atom_coord
        FROM atom
        WHERE codepoint = ascii(substring(v_clean_text, i, 1));
        
        IF v_atom_coord IS NOT NULL THEN
            v_coords := array_append(v_coords, v_atom_coord);
        END IF;
    END LOOP;
    
    -- Return based on number of points
    IF array_length(v_coords, 1) IS NULL OR array_length(v_coords, 1) = 0 THEN
        -- No valid atoms found, return origin
        RETURN ST_SetSRID(ST_MakePoint(0.5, 0.5, 0.5, 0.5), 0);
    ELSIF array_length(v_coords, 1) = 1 THEN
        -- Single point - return as is
        RETURN v_coords[1];
    ELSE
        -- Multiple points - create LINESTRINGZM
        RETURN ST_MakeLine(v_coords);
    END IF;
END;
$$ LANGUAGE plpgsql STABLE;

-- Compute 4D centroid from a geometry (point or linestring)
CREATE OR REPLACE FUNCTION hypercube_geometry_centroid(p_geom geometry)
RETURNS geometry
AS $$
DECLARE
    v_type text;
    v_point geometry;
    v_sum_x double precision := 0;
    v_sum_y double precision := 0;
    v_sum_z double precision := 0;
    v_sum_m double precision := 0;
    v_count integer := 0;
    i integer;
BEGIN
    v_type := ST_GeometryType(p_geom);
    
    IF v_type = 'ST_Point' THEN
        RETURN p_geom;
    ELSIF v_type IN ('ST_LineString', 'ST_MultiPoint') THEN
        -- Average all points
        FOR i IN 1..ST_NumPoints(p_geom)
        LOOP
            v_point := ST_PointN(p_geom, i);
            v_sum_x := v_sum_x + ST_X(v_point);
            v_sum_y := v_sum_y + ST_Y(v_point);
            v_sum_z := v_sum_z + ST_Z(v_point);
            v_sum_m := v_sum_m + ST_M(v_point);
            v_count := v_count + 1;
        END LOOP;
        
        IF v_count > 0 THEN
            RETURN ST_SetSRID(ST_MakePoint(
                v_sum_x / v_count,
                v_sum_y / v_count,
                v_sum_z / v_count,
                v_sum_m / v_count
            ), 0);
        END IF;
    END IF;
    
    -- Fallback
    RETURN ST_SetSRID(ST_MakePoint(0.5, 0.5, 0.5, 0.5), 0);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Register a new model
CREATE OR REPLACE FUNCTION hypercube_register_model(
    p_name text,
    p_model_type text,
    p_vocab_size integer,
    p_config jsonb DEFAULT NULL,
    p_source_path text DEFAULT NULL
) RETURNS bytea
AS $$
DECLARE
    v_id bytea;
    v_data bytea;
BEGIN
    -- Generate model ID from name + type
    v_data := convert_to(p_name || '|' || p_model_type, 'UTF8');
    v_id := hypercube_blake3(v_data);
    
    -- Insert or update
    INSERT INTO model (id, name, model_type, vocab_size, config, source_path)
    VALUES (v_id, p_name, p_model_type, p_vocab_size, p_config, p_source_path)
    ON CONFLICT (name, model_type) DO UPDATE SET
        vocab_size = p_vocab_size,
        config = COALESCE(p_config, model.config),
        source_path = COALESCE(p_source_path, model.source_path);
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Ingest a single vocab token
CREATE OR REPLACE FUNCTION hypercube_ingest_vocab_token(
    p_model_id bytea,
    p_token_index integer,
    p_token_text text
) RETURNS bytea
AS $$
DECLARE
    v_id bytea;
    v_trajectory geometry;
    v_centroid geometry;
    v_hilbert_lo bigint;
    v_hilbert_hi bigint;
    v_is_special boolean;
    v_is_subword boolean;
    v_composition_id bytea;
    v_atom_ids bytea[];
    v_atom_is_atom boolean[];
    v_clean_text text;
    i integer;
    v_cp integer;
    v_atom_id bytea;
BEGIN
    -- Generate token ID
    v_id := hypercube_hash_token(p_model_id, p_token_text, p_token_index);
    
    -- Check if already exists
    IF EXISTS (SELECT 1 FROM vocab_token WHERE id = v_id) THEN
        RETURN v_id;
    END IF;
    
    -- Detect special tokens
    v_is_special := p_token_text ~ '^\[.+\]$' OR p_token_text ~ '^<.+>$';
    v_is_subword := p_token_text ~ '^##';
    
    -- Get trajectory and centroid
    v_trajectory := hypercube_token_to_trajectory(p_token_text);
    v_centroid := hypercube_geometry_centroid(v_trajectory);
    
    -- Compute Hilbert index from centroid
    SELECT h.hilbert_lo, h.hilbert_hi
    INTO v_hilbert_lo, v_hilbert_hi
    FROM hypercube_coords_to_hilbert(
        (ST_X(v_centroid) * 4294967295)::bigint,
        (ST_Y(v_centroid) * 4294967295)::bigint,
        (ST_Z(v_centroid) * 4294967295)::bigint,
        (ST_M(v_centroid) * 4294967295)::bigint
    ) h;
    
    -- Create composition from atoms (for non-special tokens)
    IF NOT v_is_special THEN
        v_clean_text := regexp_replace(p_token_text, '^##', '');
        v_atom_ids := ARRAY[]::bytea[];
        v_atom_is_atom := ARRAY[]::boolean[];
        
        FOR i IN 1..char_length(v_clean_text)
        LOOP
            v_cp := ascii(substring(v_clean_text, i, 1));
            SELECT id INTO v_atom_id FROM atom WHERE codepoint = v_cp;
            
            IF v_atom_id IS NOT NULL THEN
                v_atom_ids := array_append(v_atom_ids, v_atom_id);
                v_atom_is_atom := array_append(v_atom_is_atom, true);
            END IF;
        END LOOP;
        
        IF array_length(v_atom_ids, 1) > 0 THEN
            v_composition_id := hypercube_insert_composition(v_atom_ids, v_atom_is_atom);
        END IF;
    END IF;
    
    -- Insert vocab token
    INSERT INTO vocab_token (
        id, model_id, token_index, token_text, 
        is_special, is_subword,
        coords, hilbert_lo, hilbert_hi, composition_id
    )
    VALUES (
        v_id, p_model_id, p_token_index, p_token_text,
        v_is_special, v_is_subword,
        v_centroid, COALESCE(v_hilbert_lo, 0), COALESCE(v_hilbert_hi, 0), v_composition_id
    );
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Batch ingest vocab from an array (for performance)
CREATE OR REPLACE FUNCTION hypercube_batch_ingest_vocab(
    p_model_id bytea,
    p_tokens text[]
) RETURNS integer
AS $$
DECLARE
    v_count integer := 0;
    v_idx integer;
    v_token text;
BEGIN
    v_idx := 0;
    FOREACH v_token IN ARRAY p_tokens
    LOOP
        PERFORM hypercube_ingest_vocab_token(p_model_id, v_idx, v_token);
        v_idx := v_idx + 1;
        v_count := v_count + 1;
    END LOOP;
    
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- Create edges between adjacent tokens in a sequence
CREATE OR REPLACE FUNCTION hypercube_create_token_edges(
    p_model_id bytea,
    p_token_ids bytea[],
    p_weight double precision DEFAULT 1.0
) RETURNS integer
AS $$
DECLARE
    v_count integer := 0;
    i integer;
    v_edge_id bytea;
    v_src_coords geometry;
    v_dst_coords geometry;
    v_trajectory geometry;
BEGIN
    IF array_length(p_token_ids, 1) < 2 THEN
        RETURN 0;
    END IF;
    
    FOR i IN 1..(array_length(p_token_ids, 1) - 1)
    LOOP
        -- Get source and dest coordinates
        SELECT coords INTO v_src_coords FROM vocab_token WHERE id = p_token_ids[i];
        SELECT coords INTO v_dst_coords FROM vocab_token WHERE id = p_token_ids[i + 1];
        
        IF v_src_coords IS NOT NULL AND v_dst_coords IS NOT NULL THEN
            -- Create trajectory from src to dst
            v_trajectory := ST_MakeLine(ARRAY[v_src_coords, v_dst_coords]);
            
            -- Generate edge ID
            v_edge_id := hypercube_blake3(p_token_ids[i] || p_token_ids[i + 1] || int4send(i));
            
            -- Insert or update edge
            INSERT INTO token_edge (
                id, model_id, src_token_id, dst_token_id,
                trajectory, weight, frequency, ordinal
            )
            VALUES (
                v_edge_id, p_model_id, p_token_ids[i], p_token_ids[i + 1],
                v_trajectory, p_weight, 1, i - 1
            )
            ON CONFLICT (model_id, src_token_id, dst_token_id, ordinal) DO UPDATE SET
                weight = token_edge.weight + p_weight,
                frequency = token_edge.frequency + 1;
            
            v_count := v_count + 1;
        END IF;
    END LOOP;
    
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS: Model statistics
-- ============================================================================

CREATE OR REPLACE VIEW model_vocab_stats AS
SELECT 
    m.name as model_name,
    m.model_type,
    m.vocab_size as expected_vocab_size,
    COUNT(vt.id) as actual_vocab_size,
    COUNT(*) FILTER (WHERE vt.is_special) as special_tokens,
    COUNT(*) FILTER (WHERE vt.is_subword) as subword_tokens,
    AVG(char_length(vt.token_text)) as avg_token_length
FROM model m
LEFT JOIN vocab_token vt ON vt.model_id = m.id
GROUP BY m.id, m.name, m.model_type, m.vocab_size;

CREATE OR REPLACE VIEW token_edge_stats AS
SELECT 
    m.name as model_name,
    COUNT(te.id) as total_edges,
    AVG(te.weight) as avg_weight,
    SUM(te.frequency) as total_frequency,
    AVG(ST_Length(te.trajectory)) as avg_trajectory_length
FROM model m
LEFT JOIN token_edge te ON te.model_id = m.id
GROUP BY m.id, m.name;

COMMIT;
