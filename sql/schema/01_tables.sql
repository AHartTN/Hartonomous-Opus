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