-- =============================================================================
-- Hartonomous Hypercube - Unified Schema v5
-- =============================================================================
-- Single source of truth for all atom storage and basic operations.
-- Heavy computation delegated to C extensions for performance.
-- 
-- Node Roles:
--   0 = generic composition (default)
--   1 = unicode atom (leaf)
--   2 = token (word/subword)
--   3 = sentence
--   4 = paragraph
--   5 = document_content_root
--   6 = file_info_root
--   7 = file_metadata_root
--   8 = file_root (parent ingestion node)
--   9 = ast_node
--  10 = kv_pair (metadata key/value)
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

BEGIN;

-- =============================================================================
-- Core Table: Unified Atom Storage
-- =============================================================================
-- POINTZM = leaf atoms (Unicode codepoints)
-- LINESTRINGZM = compositions (trajectory through child centroids)

CREATE TABLE IF NOT EXISTS atom (
    id              BYTEA PRIMARY KEY,              -- BLAKE3 hash (content-addressed)
    geom            GEOMETRY(GEOMETRYZM, 0) NOT NULL, -- 4D geometry (SRID 0)
    centroid        GEOMETRY(POINTZM, 0),           -- Pre-computed 4D centroid
    value           BYTEA,                          -- Raw bytes (leaves only)
    codepoint       INTEGER UNIQUE,                 -- Unicode codepoint (leaves only)
    hilbert_lo      BIGINT NOT NULL,                -- Hilbert index (low 64 bits)
    hilbert_hi      BIGINT NOT NULL,                -- Hilbert index (high 64 bits)
    depth           INTEGER NOT NULL DEFAULT 0,     -- DAG depth (0 = leaf)
    atom_count      BIGINT NOT NULL DEFAULT 1,      -- Total leaves in subtree
    node_role       SMALLINT NOT NULL DEFAULT 0,    -- Semantic role (see header)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    CONSTRAINT ck_leaf CHECK (depth > 0 OR (value IS NOT NULL AND codepoint IS NOT NULL))
);

-- =============================================================================
-- Relation Table: All relationships between atoms
-- =============================================================================
-- Relation types:
--   'C' = composition (binary: ordinal 1=left, 2=right)
--   'S' = sequence (document content in reading order)
--   'M' = metadata (key-value associations)
--   'R' = reference (cross-reference, citation, link)

CREATE TABLE IF NOT EXISTS relation (
    parent_id       BYTEA NOT NULL REFERENCES atom(id) ON DELETE CASCADE,
    child_id        BYTEA NOT NULL REFERENCES atom(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,               -- Position: 1,2 for binary; 1,2,3... for sequence
    relation_type   CHAR(1) NOT NULL DEFAULT 'C',   -- C=composition, S=sequence, M=metadata, R=reference
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    PRIMARY KEY (parent_id, ordinal, relation_type),
    CONSTRAINT ck_ordinal_positive CHECK (ordinal > 0),
    CONSTRAINT ck_relation_type CHECK (relation_type IN ('C', 'S', 'M', 'R'))
);

-- =============================================================================
-- Indexes
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_atom_geom ON atom USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_atom_centroid ON atom USING GIST(centroid);
CREATE INDEX IF NOT EXISTS idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);
CREATE INDEX IF NOT EXISTS idx_atom_depth ON atom(depth);
CREATE INDEX IF NOT EXISTS idx_atom_codepoint ON atom(codepoint) WHERE codepoint IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_atom_node_role ON atom(node_role) WHERE node_role > 0;

-- Relation indexes for efficient traversal
CREATE INDEX IF NOT EXISTS idx_relation_parent ON relation(parent_id);
CREATE INDEX IF NOT EXISTS idx_relation_child ON relation(child_id);
CREATE INDEX IF NOT EXISTS idx_relation_type ON relation(relation_type);
CREATE INDEX IF NOT EXISTS idx_relation_parent_type ON relation(parent_id, relation_type);

-- =============================================================================
-- Trigger: Auto-compute centroid on insert/update
-- =============================================================================

CREATE OR REPLACE FUNCTION trg_compute_centroid() RETURNS TRIGGER AS $$
DECLARE
    v_z DOUBLE PRECISION;
    v_m DOUBLE PRECISION;
BEGIN
    IF ST_GeometryType(NEW.geom) = 'ST_Point' THEN
        NEW.centroid := NEW.geom;
    ELSE
        SELECT AVG(ST_Z(g.geom)), AVG(ST_M(g.geom))
        INTO v_z, v_m
        FROM ST_DumpPoints(NEW.geom) g;
        
        NEW.centroid := ST_SetSRID(ST_MakePoint(
            ST_X(ST_Centroid(NEW.geom)),
            ST_Y(ST_Centroid(NEW.geom)),
            COALESCE(v_z, 0),
            COALESCE(v_m, 0)
        ), 0);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_atom_centroid ON atom;
CREATE TRIGGER trg_atom_centroid
    BEFORE INSERT OR UPDATE OF geom ON atom
    FOR EACH ROW EXECUTE FUNCTION trg_compute_centroid();

COMMIT;
