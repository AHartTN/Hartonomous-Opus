-- =============================================================================
-- Hartonomous Hypercube - Unified Schema v4
-- =============================================================================
-- Single source of truth for all atom storage and basic operations.
-- Heavy computation delegated to C extensions for performance.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

BEGIN;

-- =============================================================================
-- Core Table: Unified Atom Storage
-- =============================================================================
-- POINTZM = leaf atoms (Unicode codepoints)
-- LINESTRINGZM = compositions (trajectory through child centroids)

CREATE TABLE IF NOT EXISTS atom (
    id              BYTEA PRIMARY KEY,              -- BLAKE3 hash
    geom            GEOMETRY(GEOMETRYZM, 0) NOT NULL, -- 4D geometry (SRID 0)
    centroid        GEOMETRY(POINTZM, 0),           -- Pre-computed 4D centroid
    children        BYTEA[],                        -- Child hashes (NULL for leaves)
    value           BYTEA,                          -- Raw bytes (leaves only)
    codepoint       INTEGER UNIQUE,                 -- Unicode codepoint (leaves only)
    hilbert_lo      BIGINT NOT NULL,                -- Hilbert index (low 64 bits)
    hilbert_hi      BIGINT NOT NULL,                -- Hilbert index (high 64 bits)
    depth           INTEGER NOT NULL DEFAULT 0,     -- DAG depth (0 = leaf)
    atom_count      BIGINT NOT NULL DEFAULT 1,      -- Total leaves in subtree
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    CONSTRAINT ck_leaf CHECK (depth > 0 OR (value IS NOT NULL AND codepoint IS NOT NULL)),
    CONSTRAINT ck_comp CHECK (depth = 0 OR children IS NOT NULL)
);

-- =============================================================================
-- Indexes
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_atom_geom ON atom USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_atom_centroid ON atom USING GIST(centroid);
CREATE INDEX IF NOT EXISTS idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);
CREATE INDEX IF NOT EXISTS idx_atom_depth ON atom(depth);
CREATE INDEX IF NOT EXISTS idx_atom_codepoint ON atom(codepoint) WHERE codepoint IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_atom_children_gin ON atom USING GIN(children);

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
