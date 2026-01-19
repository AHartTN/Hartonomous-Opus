-- =============================================================================
-- HYPERCUBE DATABASE SCHEMA - CONSOLIDATED DEPLOYMENT FILE
-- =============================================================================
-- Generated: 2026-01-18 14:32:53
-- This file contains the complete schema and can be run directly with psql.
--
-- Usage:
--   psql -h HOST -U USER -d DATABASE -f full_schema.sql
--
-- Prerequisites:
--   - PostgreSQL 14+ with PostGIS extension
--   - Database must exist (CREATE DATABASE hypercube;)
-- =============================================================================

-- Suppress NOTICE messages during deployment
SET client_min_messages = WARNING;

-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;

-- =============================================================================
-- FILE: schema/01_tables.sql
-- =============================================================================
-- =============================================================================
-- CORE TABLES - 3-TABLE HYPERCUBE ARCHITECTURE
-- =============================================================================
-- This file contains ONLY table definitions for the hypercube substrate.
-- Indexes, constraints, and functions are in separate files.
-- =============================================================================

-- =============================================================================
-- 1. ATOM: Unicode codepoints with 4D Laplacian-projected geometry
-- =============================================================================
-- Pure, immutable coordinate system. ~1.1M rows seeded once.
-- BLAKE3(id) = hash of UTF-8 bytes for each codepoint.

CREATE TABLE IF NOT EXISTS atom (
    id              BYTEA PRIMARY KEY,              -- BLAKE3(codepoint bytes)
    codepoint       INTEGER NOT NULL UNIQUE,        -- Unicode codepoint (0-0x10FFFF)
    value           BYTEA NOT NULL,                 -- UTF-8 bytes of the character
    geom            GEOMETRY(POINTZM, 0) NOT NULL,  -- 4D Laplacian coordinates (SRID 0)
    hilbert_lo      NUMERIC(20,0) NOT NULL,         -- Hilbert curve index (low 64 bits)
    hilbert_hi      NUMERIC(20,0) NOT NULL,         -- Hilbert curve index (high 64 bits)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- 2. COMPOSITION: Aggregations with 4D centroids
-- =============================================================================
-- BPE tokens, words, phrases, sentences, paragraphs.
-- id = BLAKE3(ordered child hashes concatenated)

CREATE TABLE IF NOT EXISTS composition (
    id              BYTEA PRIMARY KEY,              -- BLAKE3(child_ids concatenated)
    label           TEXT,                           -- Human-readable (e.g., "whale", "##ing")
    depth           INTEGER NOT NULL DEFAULT 1,     -- Hierarchy depth (1=atom children, 2+=nested)
    child_count     INTEGER NOT NULL,               -- Number of direct children
    atom_count      BIGINT NOT NULL,                -- Total leaf atoms in subtree
    geom            GEOMETRY(LINESTRINGZM, 0),      -- Path through child centroids
    centroid        GEOMETRY(POINTZM, 0),           -- 4D Laplacian centroid
    hilbert_lo      NUMERIC(20,0),                  -- Hilbert index (low 64 bits)
    hilbert_hi      NUMERIC(20,0),                  -- Hilbert index (high 64 bits)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- 3. COMPOSITION_CHILD: Ordered children junction table
-- =============================================================================
-- Maintains parent-child relationships with ordering

CREATE TABLE IF NOT EXISTS composition_child (
    composition_id  BYTEA NOT NULL REFERENCES composition(id) ON DELETE CASCADE,
    ordinal         SMALLINT NOT NULL,              -- Position in sequence (0-based)
    child_type      CHAR(1) NOT NULL,               -- 'A' = atom, 'C' = composition
    child_id        BYTEA NOT NULL,                 -- References atom.id or composition.id
    PRIMARY KEY (composition_id, ordinal)
);

-- =============================================================================
-- 4. RELATION: Semantic edges between entities
-- =============================================================================
-- The knowledge graph. Edges between atoms/compositions.
-- Types: S=sequence, A=attention, P=proximity, T=temporal, etc.

CREATE TABLE IF NOT EXISTS relation (
    id              BIGSERIAL PRIMARY KEY,
    source_type     CHAR(1) NOT NULL,               -- 'A' = atom, 'C' = composition
    source_id       BYTEA NOT NULL,                 -- References atom.id or composition.id
    target_type     CHAR(1) NOT NULL,               -- 'A' = atom, 'C' = composition
    target_id       BYTEA NOT NULL,                 -- References atom.id or composition.id
    relation_type   CHAR(1) NOT NULL,               -- S=sequence, A=attention, P=proximity
    weight          REAL NOT NULL DEFAULT 1.0,      -- Intensity/strength of relation
    source_model    TEXT NOT NULL DEFAULT '',       -- Which model contributed this edge
    source_count    INTEGER NOT NULL DEFAULT 1,     -- How many times seen (for averaging)
    layer           INTEGER NOT NULL DEFAULT -1,    -- Model layer (-1 = N/A)
    component       TEXT NOT NULL DEFAULT '',       -- Model component (attention, mlp, etc.)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (source_id, target_id, relation_type, source_model, layer, component)
);

-- =============================================================================
-- 5. MODEL REGISTRY: Track AI models and embeddings
-- =============================================================================

CREATE TABLE IF NOT EXISTS model (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,           -- Model identifier (e.g., 'bert-base-uncased')
    source          TEXT NOT NULL,                  -- Source (e.g., 'huggingface', 'openai')
    version         TEXT,                           -- Model version
    embedding_dim   INTEGER,                        -- Embedding dimensionality
    config          JSONB,                          -- Model configuration
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- 6. PROJECTION METADATA: Quality tracking for 4D projections
-- =============================================================================

CREATE TABLE IF NOT EXISTS projection_metadata (
    id BIGSERIAL PRIMARY KEY,

    -- Model and tensor identification
    model_id BIGINT NOT NULL REFERENCES model(id) ON DELETE CASCADE,
    tensor_name TEXT NOT NULL,

    UNIQUE (model_id, tensor_name),

    -- Tensor characteristics
    role TEXT NOT NULL CHECK (role IN ('embeddings', 'attention', 'ffn', 'other')),
    dtype TEXT NOT NULL,
    dim INTEGER NOT NULL CHECK (dim > 0),

    -- Projection quality metrics
    variance_explained REAL CHECK (variance_explained >= 0.0 AND variance_explained <= 1.0),
    converged BOOLEAN NOT NULL DEFAULT FALSE,

    -- Policy outcomes
    geom_written BOOLEAN NOT NULL DEFAULT FALSE,
    quality_score REAL,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- 7. BIGRAM STATISTICS: Co-occurrence and PMI scoring
-- =============================================================================

CREATE TABLE IF NOT EXISTS bigram_stats (
    left_id         BYTEA NOT NULL,
    right_id        BYTEA NOT NULL,
    count           BIGINT NOT NULL DEFAULT 1,
    pmi             DOUBLE PRECISION,               -- Computed PMI score
    PRIMARY KEY (left_id, right_id)
);

CREATE TABLE IF NOT EXISTS unigram_stats (
    token_id        BYTEA PRIMARY KEY,
    count           BIGINT NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS token_corpus_stats (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    total_tokens    BIGINT NOT NULL DEFAULT 0,
    total_bigrams   BIGINT NOT NULL DEFAULT 0
);

INSERT INTO token_corpus_stats (id, total_tokens, total_bigrams)
VALUES (1, 0, 0)
ON CONFLICT (id) DO NOTHING;

-- =============================================================================
-- 8. RELATION_EVIDENCE: Detailed evidence for semantic relations
-- =============================================================================

CREATE TABLE IF NOT EXISTS relation_evidence (
    id                  BIGSERIAL PRIMARY KEY,
    source_id           BYTEA NOT NULL,
    target_id           BYTEA NOT NULL,
    relation_type       CHAR(1) NOT NULL,
    source_model        TEXT NOT NULL,
    layer               INTEGER NOT NULL DEFAULT -1,
    component           TEXT NOT NULL DEFAULT '',

    -- Evidence aggregation
    raw_weight          REAL NOT NULL,
    normalized_weight   REAL NOT NULL,
    rating              REAL NOT NULL DEFAULT 1500.0,  -- ELO rating starting point
    observation_count   INTEGER NOT NULL DEFAULT 1,
    last_updated        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (source_id, target_id, relation_type, source_model, layer, component)
);

-- =============================================================================
-- 9. RELATION_CONSENSUS: ELO-aggregated relation consensus
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS relation_consensus AS
SELECT
    source_id,
    target_id,
    relation_type,

    AVG(rating) as avg_rating,
    STDDEV(rating) as rating_stddev,
    TANH((AVG(rating) - 1500.0) / 400.0) as consensus_weight,
    1.0 / (1.0 + COALESCE(STDDEV(rating), 0) / 400.0) as confidence,

    COUNT(*) as num_models,
    SUM(observation_count) as total_observations

FROM relation_evidence
GROUP BY source_id, target_id, relation_type;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE atom IS 'Unicode codepoints with 4D Laplacian-projected coordinates';
COMMENT ON TABLE composition IS 'Token aggregations (BPE, words, phrases) with 4D centroids';
COMMENT ON TABLE composition_child IS 'Ordered parent-child relationships for compositions';
COMMENT ON TABLE relation IS 'Semantic edges forming the knowledge graph';
COMMENT ON TABLE relation_evidence IS 'Detailed evidence and ratings for semantic relations';
COMMENT ON TABLE model IS 'Registry of AI models used for embeddings';
COMMENT ON TABLE bigram_stats IS 'Token co-occurrence counts and PMI scores';
COMMENT ON TABLE unigram_stats IS 'Individual token frequency counts';
COMMENT ON TABLE token_corpus_stats IS 'Corpus-wide token and bigram statistics';

-- =============================================================================
-- FILE: schema/02_indexes.sql
-- =============================================================================
-- =============================================================================
-- DATABASE INDEXES - Performance optimization for hypercube queries
-- =============================================================================
-- Indexes are critical for the hypercube's performance characteristics:
-- - Hilbert indexes enable O(log n) locality queries
-- - GIST indexes enable efficient geometric operations
-- - Foreign key indexes optimize joins
-- =============================================================================

-- =============================================================================
-- ATOM INDEXES
-- =============================================================================

-- Codepoint lookup (frequent for Unicode operations)
CREATE INDEX IF NOT EXISTS idx_atom_codepoint ON atom(codepoint);

-- Hilbert curve ordering for locality-sensitive queries
CREATE INDEX IF NOT EXISTS idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);

-- Spatial operations (KNN, distance, containment)
CREATE INDEX IF NOT EXISTS idx_atom_geom ON atom USING GIST(geom);

-- =============================================================================
-- COMPOSITION INDEXES
-- =============================================================================

-- Centroid similarity search (most important query pattern)
CREATE INDEX IF NOT EXISTS idx_comp_centroid ON composition USING GIST(centroid);

-- Hilbert locality for composition queries
CREATE INDEX IF NOT EXISTS idx_comp_hilbert ON composition(hilbert_hi, hilbert_lo);

-- Label lookups for vocabulary operations
CREATE INDEX IF NOT EXISTS idx_comp_label ON composition(label);

-- Depth-based queries (tree traversal)
CREATE INDEX IF NOT EXISTS idx_comp_depth ON composition(depth);

-- Child count queries (statistics and filtering)
CREATE INDEX IF NOT EXISTS idx_comp_child_count ON composition(child_count);

-- =============================================================================
-- COMPOSITION_CHILD INDEXES
-- =============================================================================

-- Child lookup (finding parents of a given entity)
CREATE INDEX IF NOT EXISTS idx_comp_child_child ON composition_child(child_id);

-- Ordinal ordering for sequence operations
CREATE INDEX IF NOT EXISTS idx_comp_child_ordinal ON composition_child(composition_id, ordinal);

-- =============================================================================
-- RELATION INDEXES
-- =============================================================================

-- Source entity lookups (outgoing edges)
CREATE INDEX IF NOT EXISTS idx_relation_source ON relation(source_id);

-- Target entity lookups (incoming edges)
CREATE INDEX IF NOT EXISTS idx_relation_target ON relation(target_id);

-- Relation type filtering
CREATE INDEX IF NOT EXISTS idx_relation_type ON relation(relation_type);

-- Model attribution (filtering by model)
CREATE INDEX IF NOT EXISTS idx_relation_model ON relation(source_model) WHERE source_model != '';

-- Weight ordering (top-K queries)
CREATE INDEX IF NOT EXISTS idx_relation_weight ON relation(weight DESC);

-- Layer filtering (neural network layer attribution)
CREATE INDEX IF NOT EXISTS idx_relation_layer ON relation(layer) WHERE layer != -1;

-- =============================================================================
-- MODEL REGISTRY INDEXES
-- =============================================================================

-- Model name lookups
CREATE INDEX IF NOT EXISTS idx_model_name ON model(name);

-- Source filtering
CREATE INDEX IF NOT EXISTS idx_model_source ON model(source);

-- =============================================================================
-- BIGRAM STATISTICS INDEXES
-- =============================================================================

-- Left token lookups (continuation queries)
CREATE INDEX IF NOT EXISTS idx_bigram_left ON bigram_stats(left_id);

-- Right token lookups (predecessor queries)
CREATE INDEX IF NOT EXISTS idx_bigram_right ON bigram_stats(right_id);

-- PMI ordering (top continuations)
CREATE INDEX IF NOT EXISTS idx_bigram_pmi ON bigram_stats(pmi DESC) WHERE pmi IS NOT NULL;

-- =============================================================================
-- PROJECTION METADATA INDEXES
-- =============================================================================

-- Model lookups (which models have been projected)
CREATE INDEX IF NOT EXISTS idx_proj_meta_model ON projection_metadata(model_id);

-- Role filtering (embeddings vs attention vs ffn)
CREATE INDEX IF NOT EXISTS idx_proj_meta_role ON projection_metadata(role);

-- Quality-based queries (champion model selection)
CREATE INDEX IF NOT EXISTS idx_proj_meta_quality ON projection_metadata(quality_score DESC);

-- Geometry write status (track which projections define coordinates)
CREATE INDEX IF NOT EXISTS idx_proj_meta_written ON projection_metadata(geom_written);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON INDEX idx_atom_hilbert IS 'Hilbert curve ordering for O(log n) locality queries';
COMMENT ON INDEX idx_atom_geom IS 'GIST index for 4D geometric operations (KNN, distance)';
COMMENT ON INDEX idx_comp_centroid IS 'Critical index for 4D similarity search';
COMMENT ON INDEX idx_comp_hilbert IS 'Hilbert ordering for composition locality queries';
COMMENT ON INDEX idx_relation_source IS 'Outgoing edge lookups from source entities';
COMMENT ON INDEX idx_relation_target IS 'Incoming edge lookups to target entities';
COMMENT ON INDEX idx_relation_weight IS 'Weight-ordered queries for top-K semantic neighbors';

-- =============================================================================
-- FILE: schema/03_constraints.sql
-- =============================================================================
-- Function to validate composition child references
CREATE OR REPLACE FUNCTION validate_composition_child()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure the referenced child exists
    IF NEW.child_type = 'A' THEN
        IF NOT EXISTS (SELECT 1 FROM atom WHERE id = NEW.child_id) THEN
            RAISE EXCEPTION 'Atom child % does not exist', NEW.child_id;
        END IF;
    ELSIF NEW.child_type = 'C' THEN
        IF NOT EXISTS (SELECT 1 FROM composition WHERE id = NEW.child_id) THEN
            RAISE EXCEPTION 'Composition child % does not exist', NEW.child_id;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to refresh relation consensus materialized view
CREATE OR REPLACE FUNCTION refresh_relation_consensus()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW relation_consensus;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- FILE: functions/geometry/distance.sql
-- =============================================================================
-- =============================================================================
-- GEOMETRY DISTANCE FUNCTIONS
-- =============================================================================
-- Core distance and similarity functions for 4D Laplacian-projected coordinates
-- These are the foundation for all similarity operations in the hypercube
-- =============================================================================

-- =============================================================================
-- 4D EUCLIDEAN DISTANCE
-- =============================================================================
-- Computes Euclidean distance between two 4D points (POINTZM geometry)
-- Used for similarity calculations throughout the system

CREATE OR REPLACE FUNCTION centroid_distance(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
    SELECT sqrt(
        power(ST_X(p_a) - ST_X(p_b), 2) +
        power(ST_Y(p_a) - ST_Y(p_b), 2) +
        power(ST_Z(p_a) - ST_Z(p_b), 2) +
        power(ST_M(p_a) - ST_M(p_b), 2)
    )
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

-- =============================================================================
-- 4D SIMILARITY (INVERSE DISTANCE)
-- =============================================================================
-- Converts distance to similarity score using inverse relationship
-- Higher values = more similar (range: 0 to 1)

CREATE OR REPLACE FUNCTION centroid_similarity(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
    SELECT 1.0 / (1.0 + centroid_distance(p_a, p_b))
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

-- =============================================================================
-- HILBERT DISTANCE (APPROXIMATION)
-- =============================================================================
-- Fast approximation using 128-bit Hilbert indices
-- Used for locality-sensitive pre-filtering before exact distance calculation

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
-- 4D CENTROID COMPUTATION
-- =============================================================================
-- Computes the centroid (average) of multiple 4D points
-- Essential for aggregating coordinates in composition hierarchies

CREATE OR REPLACE FUNCTION st_centroid_4d(geom geometry)
RETURNS geometry AS $$
DECLARE
    n integer;
    cx double precision := 0;
    cy double precision := 0;
    cz double precision := 0;
    cm double precision := 0;
    rec record;
BEGIN
    IF ST_GeometryType(geom) = 'ST_Point' THEN
        RETURN geom;
    END IF;

    n := 0;
    FOR rec IN SELECT (ST_DumpPoints(geom)).geom AS pt LOOP
        cx := cx + ST_X(rec.pt);
        cy := cy + ST_Y(rec.pt);
        cz := cz + COALESCE(ST_Z(rec.pt), 0);
        cm := cm + COALESCE(ST_M(rec.pt), 0);
        n := n + 1;
    END LOOP;

    IF n = 0 THEN
        RETURN NULL;
    END IF;

    RETURN ST_SetSRID(ST_MakePoint(cx/n, cy/n, cz/n, cm/n), ST_SRID(geom));
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON FUNCTION centroid_distance IS '4D Euclidean distance between two POINTZM geometries';
COMMENT ON FUNCTION centroid_similarity IS 'Similarity score (0-1) based on inverse distance';
COMMENT ON FUNCTION hilbert_distance IS 'Fast Hilbert index distance approximation for pre-filtering';
COMMENT ON FUNCTION st_centroid_4d IS '4D centroid computation for XYZM coordinate averaging';

-- =============================================================================
-- FILE: functions/atoms/atom_is_leaf.sql
-- =============================================================================
-- =============================================================================
-- ATOM_IS_LEAF
-- =============================================================================
-- Check if entity is a leaf (atom) vs composition
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_is_leaf(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/atoms/atom_centroid.sql
-- =============================================================================
-- =============================================================================
-- ATOM_CENTROID
-- =============================================================================
-- Get 4D centroid from atom or composition
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_centroid(p_id BYTEA)
RETURNS GEOMETRY(POINTZM, 0) AS $$
DECLARE
    v_geom GEOMETRY(POINTZM, 0);
BEGIN
    -- Check atom table first (most common case)
    SELECT geom INTO v_geom FROM atom WHERE id = p_id;
    IF v_geom IS NOT NULL THEN
        RETURN v_geom;
    END IF;

    -- Check composition table
    SELECT centroid INTO v_geom FROM composition WHERE id = p_id;
    IF v_geom IS NOT NULL THEN
        RETURN v_geom;
    END IF;

    -- Entity not found - return NULL (caller should validate with atom_exists())
    RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- FILE: functions/atoms/atom_exists.sql
-- =============================================================================
-- =============================================================================
-- ATOM_EXISTS
-- =============================================================================
-- Check if entity exists (atom or composition)
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_exists(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id)
        OR EXISTS(SELECT 1 FROM composition WHERE id = p_id);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/atoms/atom_text.sql
-- =============================================================================
-- =============================================================================
-- ATOM_TEXT
-- =============================================================================
-- Get text representation of an atom
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_text(p_id BYTEA)
RETURNS TEXT AS $$
    SELECT chr(codepoint) FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/atoms/atom_reconstruct_text.sql
-- =============================================================================
-- =============================================================================
-- ATOM TEXT RECONSTRUCTION FUNCTIONS
-- =============================================================================
-- Functions for reconstructing readable text from atom/composition hierarchies
-- =============================================================================

-- Reconstruct text from a composition by walking its children (recursive)
CREATE OR REPLACE FUNCTION atom_reconstruct_text(p_id BYTEA)
RETURNS TEXT AS $$
DECLARE
    v_result TEXT := '';
    v_is_atom BOOLEAN;
    v_child RECORD;
BEGIN
    -- Check if it's a leaf atom
    SELECT TRUE INTO v_is_atom FROM atom WHERE id = p_id;

    IF v_is_atom THEN
        -- Direct atom: return character
        SELECT chr(codepoint) INTO v_result FROM atom WHERE id = p_id;
        RETURN v_result;
    END IF;

    -- It's a composition: recurse through children
    FOR v_child IN
        SELECT cc.child_id, cc.child_type
        FROM composition_child cc
        WHERE cc.composition_id = p_id
        ORDER BY cc.ordinal
    LOOP
        IF v_child.child_type = 'A' THEN
            -- Child is an atom: append character
            v_result := v_result || COALESCE((SELECT chr(codepoint) FROM atom WHERE id = v_child.child_id), '');
        ELSE
            -- Child is a composition: recurse
            v_result := v_result || COALESCE(atom_reconstruct_text(v_child.child_id), '');
        END IF;
    END LOOP;

    -- Try composition label as fallback
    IF v_result = '' THEN
        SELECT label INTO v_result FROM composition WHERE id = p_id;
    END IF;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- FILE: functions/atoms/atom_knn.sql
-- =============================================================================
CREATE OR REPLACE FUNCTION atom_knn(p_id BYTEA, p_k INTEGER DEFAULT 5)
RETURNS TABLE(neighbor_id BYTEA, distance DOUBLE PRECISION) AS $$
DECLARE
    v_geom GEOMETRY;
BEGIN
    SELECT geom INTO v_geom FROM atom WHERE id = p_id;
    IF v_geom IS NULL THEN
        SELECT centroid INTO v_geom FROM composition WHERE id = p_id;
    END IF;

    IF v_geom IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT a.id, ST_3DDistance(a.geom, v_geom)
    FROM atom a
    WHERE a.id != p_id
    ORDER BY a.geom <-> v_geom
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- FILE: functions/atoms/lookup.sql
-- =============================================================================
-- =============================================================================
-- ATOM LOOKUP FUNCTIONS
-- =============================================================================
-- Functions for finding and retrieving atoms by various criteria
-- =============================================================================

-- Lookup by codepoint
CREATE OR REPLACE FUNCTION atom_by_codepoint(p_cp INTEGER)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE codepoint = p_cp;
$$ LANGUAGE SQL STABLE;

-- Batch lookup atoms by codepoints (for ingestion)
CREATE OR REPLACE FUNCTION get_atoms_by_codepoints(p_codepoints INTEGER[])
RETURNS TABLE(
    codepoint INTEGER,
    id_hex TEXT,
    coord_x BIGINT,
    coord_y BIGINT,
    coord_z BIGINT,
    coord_m BIGINT
) AS $$
    SELECT
        a.codepoint,
        encode(a.id, 'hex'),
        ST_X(a.geom)::BIGINT,
        ST_Y(a.geom)::BIGINT,
        ST_Z(a.geom)::BIGINT,
        ST_M(a.geom)::BIGINT
    FROM atom a
    WHERE a.codepoint = ANY(p_codepoints);
$$ LANGUAGE SQL STABLE;

-- Hilbert range query for atoms
CREATE OR REPLACE FUNCTION atom_hilbert_range(p_hi_lo BIGINT, p_hi_hi BIGINT, p_lo_lo BIGINT, p_lo_hi BIGINT)
RETURNS TABLE(id BYTEA, codepoint INTEGER) AS $$
    SELECT id, codepoint FROM atom
    WHERE hilbert_hi BETWEEN p_hi_lo AND p_hi_hi
      AND hilbert_lo BETWEEN p_lo_lo AND p_lo_hi
    ORDER BY hilbert_hi, hilbert_lo;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/atoms/atom_hilbert_range.sql
-- =============================================================================
-- =============================================================================
-- ATOM_HILBERT_RANGE
-- =============================================================================
-- Hilbert range query for atoms using hilbert_hi and hilbert_lo bounds
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_hilbert_range(p_hi_lo BIGINT, p_hi_hi BIGINT, p_lo_lo BIGINT, p_lo_hi BIGINT)
RETURNS TABLE(id BYTEA, codepoint INTEGER) AS $$
    SELECT id, codepoint FROM atom
    WHERE hilbert_hi BETWEEN p_hi_lo AND p_hi_hi
      AND hilbert_lo BETWEEN p_lo_lo AND p_lo_hi
    ORDER BY hilbert_hi, hilbert_lo;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/atoms/atom_by_codepoint.sql
-- =============================================================================
-- =============================================================================
-- ATOM_BY_CODEPOINT
-- =============================================================================
-- Lookup atom by Unicode codepoint
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_by_codepoint(p_cp INTEGER)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE codepoint = p_cp;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/atoms/get_atoms_by_codepoints.sql
-- =============================================================================
-- =============================================================================
-- GET_ATOMS_BY_CODEPOINTS
-- =============================================================================
-- Batch lookup atoms by Unicode codepoint array
-- =============================================================================

CREATE OR REPLACE FUNCTION get_atoms_by_codepoints(p_codepoints INTEGER[])
RETURNS TABLE(
    codepoint INTEGER,
    id_hex TEXT,
    coord_x BIGINT,
    coord_y BIGINT,
    coord_z BIGINT,
    coord_m BIGINT
) AS $$
    SELECT
        a.codepoint,
        encode(a.id, 'hex'),
        ST_X(a.geom)::BIGINT,
        ST_Y(a.geom)::BIGINT,
        ST_Z(a.geom)::BIGINT,
        ST_M(a.geom)::BIGINT
    FROM atom a
    WHERE a.codepoint = ANY(p_codepoints);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/atoms/atom_distance.sql
-- =============================================================================
CREATE OR REPLACE FUNCTION atom_distance(p_id1 BYTEA, p_id2 BYTEA)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_geom1 GEOMETRY;
    v_geom2 GEOMETRY;
BEGIN
    SELECT COALESCE(
        (SELECT geom FROM atom WHERE id = p_id1),
        (SELECT centroid FROM composition WHERE id = p_id1)
    ) INTO v_geom1;

    SELECT COALESCE(
        (SELECT geom FROM atom WHERE id = p_id2),
        (SELECT centroid FROM composition WHERE id = p_id2)
    ) INTO v_geom2;

    IF v_geom1 IS NULL OR v_geom2 IS NULL THEN
        RETURN NULL;
    END IF;

    RETURN ST_3DDistance(v_geom1, v_geom2);
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- FILE: functions/compositions/atom_children.sql
-- =============================================================================
-- =============================================================================
-- ATOM_CHILDREN
-- =============================================================================
-- Get children of a composition
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_children(p_id BYTEA)
RETURNS TABLE(child_id BYTEA, child_type CHAR(1), ordinal SMALLINT) AS $$
    SELECT cc.child_id, cc.child_type, cc.ordinal
    FROM composition_child cc
    WHERE cc.composition_id = p_id
    ORDER BY cc.ordinal;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/compositions/atom_child_count.sql
-- =============================================================================
CREATE OR REPLACE FUNCTION atom_child_count(p_id BYTEA)
RETURNS INTEGER AS $$
    SELECT COALESCE(
        (SELECT child_count FROM composition WHERE id = p_id),
        0
    );
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/compositions/find_composition.sql
-- =============================================================================
-- =============================================================================
-- FIND_COMPOSITION
-- =============================================================================
-- Find composition by label (vocabulary lookup)
-- =============================================================================

CREATE OR REPLACE FUNCTION find_composition(p_label TEXT)
RETURNS BYTEA AS $$
    SELECT id FROM composition WHERE label = p_label LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/compositions/compute_composition_centroid.sql
-- =============================================================================
-- =============================================================================
-- COMPUTE COMPOSITION CENTROID
-- =============================================================================
-- Computes composition centroid from its atom children
-- =============================================================================

CREATE OR REPLACE FUNCTION compute_composition_centroid(comp_id bytea)
RETURNS geometry AS $$
    SELECT st_centroid_4d(ST_Collect(a.geom))
    FROM composition_child cc
    JOIN atom a ON a.id = cc.child_id
    WHERE cc.composition_id = comp_id;
$$ LANGUAGE sql STABLE PARALLEL SAFE;

-- =============================================================================
-- FILE: functions/compositions/recompute_composition_centroids.sql
-- =============================================================================
-- =============================================================================
-- RECOMPUTE COMPOSITION CENTROIDS
-- =============================================================================
-- Recomputes all composition centroids from their children (atoms AND compositions)
-- Handles hierarchical compositions by propagating from leaves up to root
-- =============================================================================

CREATE OR REPLACE FUNCTION recompute_composition_centroids(batch_size integer DEFAULT 10000)
RETURNS integer AS $$
DECLARE
    updated integer := 0;
    total_updated integer := 0;
    max_depth integer;
    current_depth integer;
BEGIN
    -- First pass: compositions with ATOM children (leaf compositions)
    WITH comp_centroids AS (
        SELECT
            cc.composition_id as id,
            ST_SetSRID(st_centroid_4d(ST_Collect(a.geom)), 0) as new_centroid
        FROM composition_child cc
        JOIN atom a ON a.id = cc.child_id
        WHERE cc.child_type = 'A'
        GROUP BY cc.composition_id
    )
    UPDATE composition c
    SET centroid = comp_centroids.new_centroid
    FROM comp_centroids
    WHERE c.id = comp_centroids.id;

    GET DIAGNOSTICS updated = ROW_COUNT;
    total_updated := total_updated + updated;

    -- Get max depth for iterative propagation
    SELECT MAX(depth) INTO max_depth FROM composition;
    IF max_depth IS NULL THEN
        RETURN total_updated;
    END IF;

    -- Propagate centroids up the tree, from deepest to shallowest
    -- Start from max_depth and work up to 1
    FOR current_depth IN REVERSE max_depth..1 LOOP
        WITH comp_centroids AS (
            SELECT
                cc.composition_id as id,
                ST_SetSRID(st_centroid_4d(ST_Collect(child.centroid)), 0) as new_centroid
            FROM composition_child cc
            JOIN composition child ON child.id = cc.child_id
            JOIN composition parent ON parent.id = cc.composition_id
            WHERE cc.child_type = 'C'
              AND parent.depth = current_depth
              AND child.centroid IS NOT NULL
            GROUP BY cc.composition_id
            HAVING COUNT(*) > 0
        )
        UPDATE composition c
        SET centroid = comp_centroids.new_centroid
        FROM comp_centroids
        WHERE c.id = comp_centroids.id
          AND c.centroid IS NULL;

        GET DIAGNOSTICS updated = ROW_COUNT;
        total_updated := total_updated + updated;
    END LOOP;

    RETURN total_updated;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- FILE: functions/compositions/maintain_child_count_integrity.sql
-- =============================================================================
-- =============================================================================
-- CHILD COUNT INTEGRITY MAINTENANCE
-- =============================================================================
-- Maintains composition.child_count to always match the actual number of
-- children in composition_child.
--
-- DESIGN DECISION:
-- - Bulk insert mode: C++ sets child_count directly, trigger is disabled
-- - Incremental mode: Trigger maintains count on DELETE/UPDATE only
--
-- The INSERT trigger is DISABLED because:
-- 1. Bulk inserts from C++ already set child_count correctly
-- 2. Having the trigger increment on INSERT would double the count
-- 3. Compositions are created with accurate child_count, children added after
--
-- DELETE and UPDATE triggers remain active to handle:
-- - Removing children from existing compositions
-- - Moving children between compositions
-- =============================================================================

-- Drop existing trigger if present (to avoid conflicts)
DROP TRIGGER IF EXISTS trigger_maintain_child_count ON composition_child;

-- Trigger function to maintain child_count (DELETE and UPDATE only)
CREATE OR REPLACE FUNCTION maintain_composition_child_count()
RETURNS TRIGGER AS $$
BEGIN
    -- NOTE: INSERT is NOT handled here - bulk inserts set child_count directly
    -- This avoids double-counting when C++ inserts compositions with pre-computed counts

    IF TG_OP = 'DELETE' THEN
        -- Decrement child_count for the removed composition
        UPDATE composition SET child_count = child_count - 1 WHERE id = OLD.composition_id;
        RETURN OLD;

    ELSIF TG_OP = 'UPDATE' THEN
        -- If composition_id changed, adjust counts accordingly
        IF OLD.composition_id IS DISTINCT FROM NEW.composition_id THEN
            UPDATE composition SET child_count = child_count - 1 WHERE id = OLD.composition_id;
            UPDATE composition SET child_count = child_count + 1 WHERE id = NEW.composition_id;
        END IF;
        RETURN NEW;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger on composition_child table (DELETE and UPDATE only)
-- NOTE: INSERT is intentionally excluded - C++ handles counts during bulk insert
CREATE TRIGGER trigger_maintain_child_count
    AFTER DELETE OR UPDATE OF composition_id ON composition_child
    FOR EACH ROW EXECUTE FUNCTION maintain_composition_child_count();

-- Add check constraint to ensure child_count is never negative (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'check_child_count_non_negative'
        AND conrelid = 'composition'::regclass
    ) THEN
        ALTER TABLE composition ADD CONSTRAINT check_child_count_non_negative CHECK (child_count >= 0);
    END IF;
END $$;

-- =============================================================================
-- REPAIR FUNCTIONS
-- =============================================================================

-- Function to recalculate child_count for all compositions (for data repair)
CREATE OR REPLACE FUNCTION recalculate_all_child_counts()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    WITH updated AS (
        UPDATE composition c
        SET child_count = COALESCE(sub.actual_count, 0)
        FROM (
            SELECT composition_id, COUNT(*) as actual_count
            FROM composition_child
            GROUP BY composition_id
        ) sub
        WHERE c.id = sub.composition_id
          AND c.child_count != sub.actual_count
        RETURNING 1
    )
    SELECT COUNT(*) INTO updated_count FROM updated;

    -- Also handle compositions with no children
    UPDATE composition
    SET child_count = 0
    WHERE child_count != 0
      AND NOT EXISTS (SELECT 1 FROM composition_child WHERE composition_id = composition.id);

    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION recalculate_all_child_counts() IS
    'Repairs child_count for all compositions. Returns number of compositions updated.';

-- Function to find compositions with mismatched child_count
CREATE OR REPLACE FUNCTION find_child_count_mismatches()
RETURNS TABLE (
    composition_id_hex TEXT,
    label TEXT,
    expected_count BIGINT,
    actual_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        encode(c.id, 'hex'),
        c.label,
        COALESCE(cc.cnt, 0),
        c.child_count
    FROM composition c
    LEFT JOIN (
        SELECT composition_id, COUNT(*) as cnt
        FROM composition_child
        GROUP BY composition_id
    ) cc ON c.id = cc.composition_id
    WHERE c.child_count != COALESCE(cc.cnt, 0);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION find_child_count_mismatches() IS
    'Finds compositions where child_count does not match actual children in composition_child';

-- =============================================================================
-- FILE: functions/compositions/validate_child_references.sql
-- =============================================================================
-- =============================================================================
-- CHILD REFERENCE VALIDATION TRIGGER
-- =============================================================================
-- Validates that child_id in composition_child references an existing record
-- in either atom or composition table based on child_type.
--
-- This enforces referential integrity that can't be done with standard FK
-- constraints due to the polymorphic nature of child_id (can reference atom
-- or composition based on child_type).
--
-- CRITICAL: Without this trigger, children can reference non-existent records,
-- causing silent data corruption and broken graph traversals.
-- =============================================================================

-- Drop existing trigger and function if they exist
DROP TRIGGER IF EXISTS trigger_validate_child_references ON composition_child;
DROP FUNCTION IF EXISTS validate_child_references();

-- Validation function
CREATE OR REPLACE FUNCTION validate_child_references()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate based on child_type
    IF NEW.child_type = 'A' THEN
        -- Child must exist in atom table
        IF NOT EXISTS (SELECT 1 FROM atom WHERE id = NEW.child_id) THEN
            RAISE EXCEPTION 'Invalid atom child_id: % (not found in atom table)', encode(NEW.child_id, 'hex')
                USING HINT = 'Ensure atoms are seeded before inserting compositions';
        END IF;
    ELSIF NEW.child_type = 'C' THEN
        -- Child must exist in composition table
        IF NOT EXISTS (SELECT 1 FROM composition WHERE id = NEW.child_id) THEN
            RAISE EXCEPTION 'Invalid composition child_id: % (not found in composition table)', encode(NEW.child_id, 'hex')
                USING HINT = 'Ensure parent compositions are inserted before children';
        END IF;
    ELSE
        RAISE EXCEPTION 'Invalid child_type: % (must be A or C)', NEW.child_type;
    END IF;

    -- Also validate that composition_id exists
    IF NOT EXISTS (SELECT 1 FROM composition WHERE id = NEW.composition_id) THEN
        RAISE EXCEPTION 'Invalid composition_id: % (not found in composition table)', encode(NEW.composition_id, 'hex')
            USING HINT = 'Ensure composition is inserted before its children';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create the BEFORE INSERT trigger (BEFORE so we can reject invalid data)
CREATE TRIGGER trigger_validate_child_references
    BEFORE INSERT OR UPDATE ON composition_child
    FOR EACH ROW EXECUTE FUNCTION validate_child_references();

-- Add helpful comment
COMMENT ON FUNCTION validate_child_references() IS
    'Validates referential integrity of composition_child.child_id against atom/composition tables based on child_type';

-- =============================================================================
-- REPAIR FUNCTION: Find and report orphaned children
-- =============================================================================
-- Use this to diagnose existing data integrity issues

CREATE OR REPLACE FUNCTION find_orphaned_children()
RETURNS TABLE (
    composition_id_hex TEXT,
    ordinal SMALLINT,
    child_type CHAR(1),
    child_id_hex TEXT,
    issue TEXT
) AS $$
BEGIN
    -- Find atom children referencing non-existent atoms
    RETURN QUERY
    SELECT
        encode(cc.composition_id, 'hex'),
        cc.ordinal,
        cc.child_type,
        encode(cc.child_id, 'hex'),
        'Atom child references non-existent atom'::TEXT
    FROM composition_child cc
    WHERE cc.child_type = 'A'
      AND NOT EXISTS (SELECT 1 FROM atom WHERE id = cc.child_id);

    -- Find composition children referencing non-existent compositions
    RETURN QUERY
    SELECT
        encode(cc.composition_id, 'hex'),
        cc.ordinal,
        cc.child_type,
        encode(cc.child_id, 'hex'),
        'Composition child references non-existent composition'::TEXT
    FROM composition_child cc
    WHERE cc.child_type = 'C'
      AND NOT EXISTS (SELECT 1 FROM composition WHERE id = cc.child_id);

    -- Find children with invalid composition_id
    RETURN QUERY
    SELECT
        encode(cc.composition_id, 'hex'),
        cc.ordinal,
        cc.child_type,
        encode(cc.child_id, 'hex'),
        'Child references non-existent parent composition'::TEXT
    FROM composition_child cc
    WHERE NOT EXISTS (SELECT 1 FROM composition WHERE id = cc.composition_id);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION find_orphaned_children() IS
    'Finds composition_child records that reference non-existent atoms or compositions';

-- =============================================================================
-- REPAIR FUNCTION: Delete orphaned children
-- =============================================================================
-- Use with caution - only after reviewing find_orphaned_children() results

CREATE OR REPLACE FUNCTION delete_orphaned_children()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH deleted AS (
        DELETE FROM composition_child cc
        WHERE
            -- Orphaned atom children
            (cc.child_type = 'A' AND NOT EXISTS (SELECT 1 FROM atom WHERE id = cc.child_id))
            -- Orphaned composition children
            OR (cc.child_type = 'C' AND NOT EXISTS (SELECT 1 FROM composition WHERE id = cc.child_id))
            -- Children with missing parent
            OR NOT EXISTS (SELECT 1 FROM composition WHERE id = cc.composition_id)
        RETURNING 1
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION delete_orphaned_children() IS
    'Deletes orphaned composition_child records. Run find_orphaned_children() first to review.';


-- =============================================================================
-- FILE: functions/relations/semantic_neighbors.sql
-- =============================================================================
CREATE OR REPLACE FUNCTION semantic_neighbors(p_id BYTEA, p_limit INTEGER DEFAULT 10)
RETURNS TABLE(neighbor_id BYTEA, weight REAL, relation_type CHAR(1)) AS $$
    SELECT r.target_id, r.weight, r.relation_type
    FROM relation r
    WHERE r.source_id = p_id
    UNION
    SELECT r.source_id, r.weight, r.relation_type
    FROM relation r
    WHERE r.target_id = p_id
    ORDER BY weight DESC
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/relations/attention.sql
-- =============================================================================
-- Attention scores (similarity in 4D space)
CREATE OR REPLACE FUNCTION attention(p_id BYTEA, p_k INTEGER DEFAULT 5)
RETURNS TABLE(target_id BYTEA, score DOUBLE PRECISION) AS $$
DECLARE
    v_geom GEOMETRY;
BEGIN
    -- Get the geometry
    SELECT geom INTO v_geom FROM atom WHERE id = p_id;
    IF v_geom IS NULL THEN
        SELECT centroid INTO v_geom FROM composition WHERE id = p_id;
    END IF;

    IF v_geom IS NULL THEN
        RETURN;
    END IF;

    -- Score = 1 / (1 + distance)
    RETURN QUERY
    SELECT a.id, 1.0 / (1.0 + ST_3DDistance(a.geom, v_geom))
    FROM atom a
    WHERE a.id != p_id
    ORDER BY a.geom <-> v_geom
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- FILE: functions/relations/analogy.sql
-- =============================================================================
-- Analogy: A is to B as C is to ?
CREATE OR REPLACE FUNCTION analogy(p_a BYTEA, p_b BYTEA, p_c BYTEA, p_k INTEGER DEFAULT 3)
RETURNS TABLE(result_id BYTEA, similarity DOUBLE PRECISION) AS $$
DECLARE
    v_a GEOMETRY;
    v_b GEOMETRY;
    v_c GEOMETRY;
    v_target GEOMETRY;
BEGIN
    -- Get geometries
    SELECT COALESCE(
        (SELECT geom FROM atom WHERE id = p_a),
        (SELECT centroid FROM composition WHERE id = p_a)
    ) INTO v_a;

    SELECT COALESCE(
        (SELECT geom FROM atom WHERE id = p_b),
        (SELECT centroid FROM composition WHERE id = p_b)
    ) INTO v_b;

    SELECT COALESCE(
        (SELECT geom FROM atom WHERE id = p_c),
        (SELECT centroid FROM composition WHERE id = p_c)
    ) INTO v_c;

    IF v_a IS NULL OR v_b IS NULL OR v_c IS NULL THEN
        RETURN;
    END IF;

    -- Target = C + (B - A)
    v_target := ST_SetSRID(ST_MakePoint(
        ST_X(v_c) + (ST_X(v_b) - ST_X(v_a)),
        ST_Y(v_c) + (ST_Y(v_b) - ST_Y(v_a)),
        ST_Z(v_c) + (ST_Z(v_b) - ST_Z(v_a)),
        ST_M(v_c) + (ST_M(v_b) - ST_M(v_a))
    ), 0);

    -- Find nearest to target
    RETURN QUERY
    SELECT a.id, 1.0 / (1.0 + ST_3DDistance(a.geom, v_target))
    FROM atom a
    WHERE a.id NOT IN (p_a, p_b, p_c)
    ORDER BY a.geom <-> v_target
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- FILE: functions/relations/upsert_relation.sql
-- =============================================================================
-- Upsert relation with weight averaging
CREATE OR REPLACE FUNCTION upsert_relation(
    p_source_type CHAR(1),
    p_source_id BYTEA,
    p_target_type CHAR(1),
    p_target_id BYTEA,
    p_relation_type CHAR(1),
    p_weight REAL,
    p_source_model TEXT DEFAULT '',
    p_layer INTEGER DEFAULT -1,
    p_component TEXT DEFAULT ''
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component)
    VALUES (p_source_type, p_source_id, p_target_type, p_target_id, p_relation_type, p_weight,
            COALESCE(p_source_model, ''), COALESCE(p_layer, -1), COALESCE(p_component, ''))
    ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET
        weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1),
        source_count = relation.source_count + 1
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- FILE: functions/relations/edges.sql
-- =============================================================================
-- =============================================================================
-- RELATION EDGE FUNCTIONS
-- =============================================================================
-- Functions for managing and generating semantic edges
-- =============================================================================



-- Generate k-NN semantic edges from composition centroids
CREATE OR REPLACE FUNCTION generate_knn_edges(
    p_k integer DEFAULT 10,
    p_model_name text DEFAULT 'centroid_knn'
)
RETURNS integer AS $$
DECLARE
    inserted integer := 0;
BEGIN
    INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component)
    SELECT
        'C', c1.id, 'C', neighbor.id, 'S',
        1.0 / (1.0 + neighbor.dist),
        p_model_name, -1, 'knn'
    FROM composition c1
    CROSS JOIN LATERAL (
        SELECT c2.id, c1.centroid <-> c2.centroid AS dist
        FROM composition c2
        WHERE c2.id != c1.id
          AND c2.centroid IS NOT NULL
        ORDER BY c1.centroid <-> c2.centroid
        LIMIT p_k
    ) neighbor
    WHERE c1.centroid IS NOT NULL
    ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET
        weight = GREATEST(relation.weight, EXCLUDED.weight),
        source_count = relation.source_count + 1;

    GET DIAGNOSTICS inserted = ROW_COUNT;
    RETURN inserted;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- FILE: functions/queries/search_text.sql
-- =============================================================================
-- =============================================================================
-- QUERY SEARCH FUNCTIONS
-- =============================================================================
-- Text search and Q&A capabilities for the hypercube
-- =============================================================================

-- Search compositions by keyword pattern, returns matches with scores
CREATE OR REPLACE FUNCTION search_text(
    query_text TEXT,
    result_limit INTEGER DEFAULT 50
)
RETURNS TABLE(
    id BYTEA,
    text TEXT,
    depth INTEGER,
    atom_count BIGINT,
    score FLOAT
)
AS $$
DECLARE
    pattern TEXT;
BEGIN
    -- Build regex pattern from keywords (words > 2 chars)
    SELECT array_to_string(array_agg(word), '|') INTO pattern
    FROM (
        SELECT unnest(string_to_array(lower(query_text), ' ')) AS word
    ) w
    WHERE length(word) > 2;

    IF pattern IS NULL OR pattern = '' THEN
        pattern := lower(query_text);
    END IF;

    RETURN QUERY
    SELECT
        c.id,
        atom_reconstruct_text(c.id)::TEXT as text,
        c.depth,
        c.atom_count,
        (c.depth::FLOAT * 0.5 + c.atom_count::FLOAT * 0.1) as score
    FROM composition c
    WHERE atom_reconstruct_text(c.id) ~* pattern
      AND c.depth >= 3
      AND length(atom_reconstruct_text(c.id)) > 3
    ORDER BY c.depth DESC, c.atom_count DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- FILE: functions/queries/ask.sql
-- =============================================================================
-- =============================================================================
-- QUERY SEARCH FUNCTIONS
-- =============================================================================
-- Text search and Q&A capabilities for the hypercube
-- =============================================================================

-- Q&A function: takes a natural language question, returns answer with evidence
CREATE OR REPLACE FUNCTION ask(question TEXT)
RETURNS TEXT
AS $$
DECLARE
    answer TEXT := '';
    evidence TEXT[];
    top_results RECORD;
    pattern TEXT;
BEGIN
    -- Extract keywords from question
    SELECT array_to_string(array_agg(word), '|') INTO pattern
    FROM (
        SELECT unnest(string_to_array(lower(question), ' ')) AS word
    ) w
    WHERE length(word) > 2
      AND word NOT IN ('what', 'who', 'where', 'when', 'why', 'how',
                        'the', 'is', 'are', 'was', 'were', 'did', 'does',
                        'can', 'could', 'would', 'should', 'will');

    IF pattern IS NULL OR pattern = '' THEN
        RETURN 'Please provide a more specific question.';
    END IF;

    -- Gather evidence from compositions
    SELECT array_agg(text ORDER BY score DESC)
    INTO evidence
    FROM (
        SELECT DISTINCT ON (text) text, score
        FROM search_text(pattern, 100)
        WHERE length(text) > 5
        ORDER BY text, score DESC
        LIMIT 20
    ) sub;

    IF evidence IS NULL OR array_length(evidence, 1) = 0 THEN
        RETURN 'No information found for: ' || question;
    END IF;

    -- Build answer from evidence
    answer := 'Based on the text, I found:' || E'\n\n';

    FOR i IN 1..LEAST(10, array_length(evidence, 1)) LOOP
        answer := answer || '  - ' || trim(evidence[i]) || E'\n';
    END LOOP;

    RETURN answer;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- FILE: functions/queries/encode_prompt.sql
-- =============================================================================
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

-- =============================================================================
-- FILE: functions/queries/score_candidates.sql
-- =============================================================================
-- =============================================================================
-- GENERATIVE QUERY FUNCTIONS
-- =============================================================================
-- Token generation and prompting capabilities for the hypercube
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
-- FILE: functions/queries/generate_tokens.sql
-- =============================================================================
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

-- =============================================================================
-- FILE: functions/queries/complete.sql
-- =============================================================================
-- =============================================================================
-- GENERATIVE QUERY FUNCTIONS
-- =============================================================================
-- Token generation and prompting capabilities for the hypercube
-- =============================================================================

-- Generate token completion as a single string
CREATE OR REPLACE FUNCTION complete(
    p_prompt TEXT,
    p_max_tokens INTEGER DEFAULT 20
)
RETURNS TEXT AS $$
    SELECT string_agg(token, ' ' ORDER BY pos)
    FROM generate_tokens(p_prompt, p_max_tokens, 0.7, 40);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: functions/queries/vector_analogy.sql
-- =============================================================================
-- =============================================================================
-- GENERATIVE QUERY FUNCTIONS
-- =============================================================================
-- Token generation and prompting capabilities for the hypercube
-- =============================================================================

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

-- =============================================================================
-- FILE: functions/stats/db_stats.sql
-- =============================================================================
-- =============================================================================
-- DB_STATS
-- =============================================================================
-- Get comprehensive database statistics
-- =============================================================================

CREATE OR REPLACE FUNCTION db_stats()
RETURNS TABLE(
    atoms BIGINT,
    compositions BIGINT,
    compositions_with_centroid BIGINT,
    relations BIGINT,
    models TEXT[]
) AS $$
    SELECT
        (SELECT count(*) FROM atom),
        (SELECT count(*) FROM composition),
        (SELECT count(*) FROM composition WHERE centroid IS NOT NULL),
        (SELECT count(*) FROM relation),
        (SELECT array_agg(DISTINCT source_model) FROM relation WHERE source_model IS NOT NULL AND source_model != '');
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- FILE: procedures/ingestion/seed_atoms.sql
-- =============================================================================
-- =============================================================================
-- SEED_ATOMS PROCEDURE
-- =============================================================================
-- Bulk load all Unicode atoms with proper session tuning and indexing
-- =============================================================================

CREATE OR REPLACE PROCEDURE seed_atoms_setup()
LANGUAGE plpgsql AS $$
BEGIN
    -- Session tuning for bulk load
    SET synchronous_commit = off;
    SET maintenance_work_mem = '2GB';
    SET work_mem = '256MB';

    -- Drop indexes for fast bulk insert
    DROP INDEX IF EXISTS idx_atom_geom;
    DROP INDEX IF EXISTS idx_atom_hilbert;
    DROP INDEX IF EXISTS idx_atom_codepoint;

    -- Truncate atom table
    TRUNCATE atom CASCADE;
END;
$$;

CREATE OR REPLACE PROCEDURE seed_atoms_finalize()
LANGUAGE plpgsql AS $$
BEGIN
    -- Restore session settings
    SET synchronous_commit = on;
    SET maintenance_work_mem = '64MB';
    SET work_mem = '4MB';

    -- Rebuild indexes
    SET maintenance_work_mem = '2GB';
    SET max_parallel_maintenance_workers = 4;

    CREATE INDEX IF NOT EXISTS idx_atom_codepoint ON atom(codepoint);
    CREATE INDEX IF NOT EXISTS idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);
    CREATE INDEX IF NOT EXISTS idx_atom_geom ON atom USING GIST(geom);

    -- Analyze for query optimization
    ANALYZE atom;

    -- Reset maintenance settings
    SET maintenance_work_mem = '64MB';
END;
$$;

-- =============================================================================
-- FILE: procedures/maintenance/prune_all.sql
-- =============================================================================
-- Wrapper procedure to run all pruning operations
-- Callable from ingestion pipeline or API

CREATE OR REPLACE PROCEDURE prune_all(relation_threshold REAL DEFAULT 0.1)
LANGUAGE plpgsql AS $$
BEGIN
    RAISE NOTICE 'Starting pruning operations...';

    -- Prune low-quality projections
    CALL prune_projections_quality();

    -- Prune duplicate projections
    CALL prune_projections_deduplication();

    -- Prune low-weight relations
    CALL prune_relations_weight(relation_threshold);

    RAISE NOTICE 'All pruning operations completed.';
END;
$$;

-- =============================================================================
-- FILE: procedures/maintenance/prune_projections_deduplication.sql
-- =============================================================================
-- Procedure to prune duplicate projections, keeping top 3 champion models per role
-- Uses get_champion_models to identify champions and removes non-champion projections

CREATE OR REPLACE PROCEDURE prune_projections_deduplication()
LANGUAGE plpgsql AS $$
DECLARE
    champion RECORD;
    deleted_count INTEGER := 0;
    total_deleted INTEGER := 0;
BEGIN
    -- For each role, delete projections not from champion models
    FOR champion IN
        SELECT DISTINCT role FROM projection_metadata
    LOOP
        -- Delete projections where model_id not in champions for this role
        DELETE FROM projection_metadata pm
        WHERE pm.role = champion.role
          AND pm.model_id NOT IN (
              SELECT model_id
              FROM get_champion_models(champion.role)
          );

        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        total_deleted := total_deleted + deleted_count;

        RAISE NOTICE 'Pruned % duplicate projections for role %', deleted_count, champion.role;
    END LOOP;

    RAISE NOTICE 'Total pruned duplicate projections: %', total_deleted;
END;
$$;

-- =============================================================================
-- FILE: procedures/maintenance/prune_projections_quality.sql
-- =============================================================================
-- Procedure to prune low-quality projections
-- Deletes projections where quality_score < 2.0 or converged = false

CREATE OR REPLACE PROCEDURE prune_projections_quality()
LANGUAGE plpgsql AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM projection_metadata
    WHERE quality_score < 2.0 OR converged = false;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Pruned % low-quality projections', deleted_count;
END;
$$;

-- =============================================================================
-- FILE: procedures/maintenance/prune_relations_weight.sql
-- =============================================================================
-- Procedure to prune relations with weights below threshold (default 0.1)

CREATE OR REPLACE PROCEDURE prune_relations_weight(threshold REAL DEFAULT 0.1)
LANGUAGE plpgsql AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM relation
    WHERE weight < threshold;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Pruned % relations with weight < %', deleted_count, threshold;
END;
$$;

-- =============================================================================
-- DEPLOYMENT COMPLETE
-- =============================================================================
-- To verify: SELECT * FROM db_stats();
-- =============================================================================

-- Restore normal message level
SET client_min_messages = NOTICE;