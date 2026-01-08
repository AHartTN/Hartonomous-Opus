-- =============================================================================
-- THREE TABLES TO CHANGE THE WORLD
-- =============================================================================
-- Atom:        Unicode seeds only, YOUR coordinate system
-- Composition: Aggregations of atoms/compositions (BPE, words, phrases)
--              Centroids are 4D projections via Laplacian eigenmap
-- Relation:    Semantic edges with weights (the knowledge graph)
-- =============================================================================
-- Note: N-dimensional embeddings from AI models are projected to 4D during
-- ingestion via Laplacian Eigenmaps + Gram-Schmidt orthonormalization.
-- Raw embeddings NEVER touch the database - only 4D coordinates.
-- =============================================================================

-- Required extensions
CREATE EXTENSION IF NOT EXISTS postgis;

-- Drop old tables in correct order (foreign keys)
DROP TABLE IF EXISTS composition_child CASCADE;
DROP TABLE IF EXISTS relation CASCADE;
DROP TABLE IF EXISTS composition CASCADE;
DROP TABLE IF EXISTS atom CASCADE;

-- =============================================================================
-- 1. ATOM: Unicode codepoints ONLY
-- =============================================================================
-- Pure, immutable, YOUR coordinate system. Hilbert computed from YOUR formula.
-- ~1.1M rows seeded once.

CREATE TABLE atom (
    id              BYTEA PRIMARY KEY,              -- BLAKE3(codepoint bytes)
    codepoint       INTEGER NOT NULL UNIQUE,        -- Unicode codepoint (0-0x10FFFF)
    value           BYTEA NOT NULL,                 -- UTF-8 bytes of the character
    geom            GEOMETRY(POINTZM, 0) NOT NULL,  -- YOUR 4D coordinate mapping
    hilbert_lo      NUMERIC(20,0) NOT NULL,         -- Hilbert index (low 64 bits)
    hilbert_hi      NUMERIC(20,0) NOT NULL,         -- Hilbert index (high 64 bits)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_atom_codepoint ON atom(codepoint);
CREATE INDEX idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);
CREATE INDEX idx_atom_geom ON atom USING GIST(geom);

-- =============================================================================
-- 2. COMPOSITION: Aggregations of atoms and other compositions
-- =============================================================================
-- BPE tokens, words, phrases, sentences.
-- id = BLAKE3(ordered children hashes concatenated)
-- Geometry: LINESTRINGZM traces path through children, centroid for similarity

CREATE TABLE composition (
    id              BYTEA PRIMARY KEY,              -- BLAKE3(child_ids concatenated)
    label           TEXT,                           -- Human-readable (e.g., "whale", "##ing")
    depth           INTEGER NOT NULL DEFAULT 1,     -- 1 = direct atom children, 2+ = nested
    child_count     INTEGER NOT NULL,               -- Number of direct children
    atom_count      BIGINT NOT NULL,                -- Total leaf atoms in subtree
    geom            GEOMETRY(LINESTRINGZM, 0),      -- Path through child centroids
    centroid        GEOMETRY(POINTZM, 0),           -- 4D centroid for similarity
    hilbert_lo      NUMERIC(20,0),                  -- Hilbert index (low 64 bits)
    hilbert_hi      NUMERIC(20,0),                  -- Hilbert index (high 64 bits)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_comp_centroid ON composition USING GIST(centroid);
CREATE INDEX idx_comp_hilbert ON composition(hilbert_hi, hilbert_lo);
CREATE INDEX idx_comp_label ON composition(label);
CREATE INDEX idx_comp_depth ON composition(depth);

-- Composition children (ordered)
CREATE TABLE composition_child (
    composition_id  BYTEA NOT NULL REFERENCES composition(id) ON DELETE CASCADE,
    ordinal         SMALLINT NOT NULL,              -- Position in sequence (0-based)
    child_type      CHAR(1) NOT NULL,               -- 'A' = atom, 'C' = composition
    child_id        BYTEA NOT NULL,                 -- References atom.id or composition.id
    PRIMARY KEY (composition_id, ordinal)
);

CREATE INDEX idx_comp_child_child ON composition_child(child_id);

-- =============================================================================
-- 3. RELATION: Semantic edges with weights
-- =============================================================================
-- The actual knowledge graph. Edges between atoms/compositions.
-- Types: S=sequence, A=attention, P=proximity, T=temporal, etc.

CREATE TABLE relation (
    id              BIGSERIAL PRIMARY KEY,
    source_type     CHAR(1) NOT NULL,               -- 'A' = atom, 'C' = composition
    source_id       BYTEA NOT NULL,                 -- References atom.id or composition.id
    target_type     CHAR(1) NOT NULL,               -- 'A' = atom, 'C' = composition
    target_id       BYTEA NOT NULL,                 -- References atom.id or composition.id
    relation_type   CHAR(1) NOT NULL,               -- S=sequence, A=attention, P=proximity
    weight          REAL NOT NULL DEFAULT 1.0,      -- Intensity/strength of relation
    source_model    TEXT NOT NULL DEFAULT '',       -- Which model contributed this edge
    source_count    INTEGER NOT NULL DEFAULT 1,     -- How many times seen (for averaging)
    layer           INTEGER NOT NULL DEFAULT -1,    -- Model layer that produced this edge (-1 = N/A)
    component       TEXT NOT NULL DEFAULT '',       -- Model component (attention, mlp, etc.)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    UNIQUE (source_id, target_id, relation_type, source_model, layer, component)
);

CREATE INDEX idx_relation_source ON relation(source_id);
CREATE INDEX idx_relation_target ON relation(target_id);
CREATE INDEX idx_relation_type ON relation(relation_type);
CREATE INDEX idx_relation_model ON relation(source_model) WHERE source_model != '';

-- =============================================================================
-- Helper: Upsert relation with weight averaging
-- =============================================================================
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
) RETURNS BIGINT AS $
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
$ LANGUAGE plpgsql;


