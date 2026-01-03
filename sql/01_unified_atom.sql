-- =============================================================================
-- Unified Atom Table Schema
-- Status: CURRENT (v3) - This is the canonical schema
-- =============================================================================
--
-- Consolidates atom, relation, and relation_edge into a SINGLE table.
--
-- Key insight:
-- - POINTZM = leaf atoms (Unicode codepoints, single bytes)
-- - LINESTRINGZM = compositions (trajectory through child 4D coordinates)
-- - Children stored in `children` array for deterministic reconstruction
-- - ST_Centroid on LINESTRINGZM gives 4D centroid for Hilbert encoding
--
-- This enables:
-- - Universal storage for all content types
-- - GIST index on geom for spatial queries
-- - Hilbert index for range queries and ordering
-- - Bit-perfect reconstruction via children array traversal

-- Extensions must be created outside transaction
CREATE EXTENSION IF NOT EXISTS postgis;

BEGIN;

-- =============================================================================
-- STEP 2: Create the unified atom table
-- =============================================================================

-- Single table for all content: leaves (atoms) and compositions
CREATE TABLE IF NOT EXISTS atom (
    -- Content-addressed identifier (BLAKE3 hash)
    -- For leaves: BLAKE3(canonical bytes)
    -- For compositions: BLAKE3(ordinal||child_hash||ordinal||child_hash||...)
    id              BYTEA PRIMARY KEY,

    -- Geometry: POINTZM for leaves, LINESTRINGZM for compositions
    -- SRID 0 = no projection (raw 4D space)
    -- For compositions: vertices are child centroids in sequence order
    geom            GEOMETRY(GEOMETRYZM, 0) NOT NULL,

    -- Child references for compositions (NULL for leaves)
    -- Array of BLAKE3 hashes in ordinal order
    -- DFS traversal of this array = original content
    children        BYTEA[],

    -- Canonical value for leaf nodes only (UTF-8 bytes of codepoint)
    -- NULL for compositions (their value is derived from children)
    value           BYTEA,

    -- Unicode codepoint for leaf atoms (NULL for compositions)
    -- This is metadata for fast lookup, derived from value
    codepoint       INTEGER UNIQUE,

    -- Hilbert curve index for ordering and range queries
    -- Computed from ST_Centroid(geom) -> 4D coords -> Hilbert
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,

    -- Depth in the DAG (0 = leaf, 1 = pair of atoms, etc.)
    depth           INTEGER NOT NULL DEFAULT 0,

    -- Total leaf atoms in subtree (1 for leaves)
    atom_count      BIGINT NOT NULL DEFAULT 1,

    -- Creation timestamp
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Constraints (relaxed for flexibility)
    -- Leaves: depth=0, value set, codepoint set, children NULL
    -- Compositions: depth>0, children set, value NULL, codepoint NULL
    CONSTRAINT valid_leaf CHECK (depth > 0 OR (value IS NOT NULL AND codepoint IS NOT NULL)),
    CONSTRAINT valid_composition CHECK (depth = 0 OR children IS NOT NULL)
);

-- =============================================================================
-- STEP 3: Create indexes
-- =============================================================================

-- Spatial index on geometry (works for both POINT and LINESTRING)
CREATE INDEX IF NOT EXISTS idx_atom_geom ON atom USING GIST(geom);

-- Hilbert index for range queries
CREATE INDEX IF NOT EXISTS idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);

-- Depth index for level-based queries
CREATE INDEX IF NOT EXISTS idx_atom_depth ON atom(depth);

-- Codepoint index for fast leaf atom lookup (used by CPE ingester)
CREATE INDEX IF NOT EXISTS idx_atom_codepoint ON atom(codepoint) WHERE codepoint IS NOT NULL;

-- =============================================================================
-- STEP 4: Helper functions
-- =============================================================================

-- Check if a node is a leaf (POINT) or composition (LINESTRING)
CREATE OR REPLACE FUNCTION atom_is_leaf(p_id BYTEA) RETURNS BOOLEAN AS $$
    SELECT depth = 0 FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- Get child IDs from a composition (from children array)
CREATE OR REPLACE FUNCTION atom_get_children(p_id BYTEA)
RETURNS TABLE(child_id BYTEA, ordinal INTEGER) AS $$
    SELECT unnest(children), generate_subscripts(children, 1) - 1
    FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- Get 4D centroid of any atom (works for POINT and LINESTRING)
-- For POINTZM (leaves): returns the point coordinates directly
-- For LINESTRINGZM (compositions): computes geometric centroid
CREATE OR REPLACE FUNCTION atom_centroid(p_id BYTEA)
RETURNS TABLE(x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION) AS $$
    SELECT
        ST_X(ST_Centroid(geom)),
        ST_Y(ST_Centroid(geom)),
        ST_Z(ST_Centroid(geom)),
        ST_M(ST_Centroid(geom))
    FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- Reconstruct content from a composition (DFS traversal via children array)
CREATE OR REPLACE FUNCTION atom_reconstruct(p_root_id BYTEA) RETURNS BYTEA AS $$
WITH RECURSIVE tree AS (
    -- Start with the root
    SELECT
        id,
        children,
        value,
        ARRAY[]::BIGINT[] as path
    FROM atom
    WHERE id = p_root_id

    UNION ALL

    -- Recursively get children
    SELECT
        a.id,
        a.children,
        a.value,
        t.path || c.ordinal
    FROM tree t
    CROSS JOIN LATERAL unnest(t.children) WITH ORDINALITY AS c(child_id, ordinal)
    JOIN atom a ON a.id = c.child_id
    WHERE t.children IS NOT NULL
)
SELECT string_agg(value, ''::BYTEA ORDER BY path)
FROM tree
WHERE value IS NOT NULL;
$$ LANGUAGE SQL STABLE;

-- Reconstruct as text (convenience function)
CREATE OR REPLACE FUNCTION atom_reconstruct_text(p_root_id BYTEA) RETURNS TEXT AS $$
    SELECT convert_from(atom_reconstruct(p_root_id), 'UTF8');
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- STEP 5: Stats views
-- =============================================================================

CREATE OR REPLACE VIEW atom_stats AS
SELECT
    depth,
    COUNT(*) as count,
    AVG(atom_count)::NUMERIC(10,2) as avg_atoms,
    SUM(atom_count) as total_atoms,
    pg_size_pretty(pg_total_relation_size('atom')) as table_size
FROM atom
GROUP BY depth
ORDER BY depth;

-- Leaf distribution by geometry type
CREATE OR REPLACE VIEW atom_type_stats AS
SELECT
    ST_GeometryType(geom) as geom_type,
    depth,
    COUNT(*) as count
FROM atom
GROUP BY ST_GeometryType(geom), depth
ORDER BY depth, geom_type;

-- =============================================================================
-- STEP 6: Similarity/distance functions
-- =============================================================================

-- 4D Euclidean distance between two atoms (using centroids)
-- 4D Euclidean distance between two atoms (using all 4 dimensions)
CREATE OR REPLACE FUNCTION atom_distance(p_id1 BYTEA, p_id2 BYTEA) RETURNS DOUBLE PRECISION AS $$
    SELECT sqrt(
        power(ST_X(a1.geom) - ST_X(a2.geom), 2) +
        power(ST_Y(a1.geom) - ST_Y(a2.geom), 2) +
        power(ST_Z(a1.geom) - ST_Z(a2.geom), 2) +
        power(ST_M(a1.geom) - ST_M(a2.geom), 2)
    )
    FROM atom a1, atom a2
    WHERE a1.id = p_id1 AND a2.id = p_id2;
$$ LANGUAGE SQL STABLE;

-- Find nearest neighbors by Hilbert index (fast)
CREATE OR REPLACE FUNCTION atom_nearest_hilbert(p_id BYTEA, p_limit INTEGER DEFAULT 10)
RETURNS TABLE(neighbor_id BYTEA, hilbert_distance NUMERIC) AS $$
    WITH target AS (
        SELECT hilbert_lo, hilbert_hi FROM atom WHERE id = p_id
    )
    SELECT
        a.id,
        ABS(a.hilbert_lo::NUMERIC - t.hilbert_lo::NUMERIC) + 
        ABS(a.hilbert_hi::NUMERIC - t.hilbert_hi::NUMERIC) * 9223372036854775808::NUMERIC
    FROM atom a, target t
    WHERE a.id != p_id
    ORDER BY ABS(a.hilbert_hi::NUMERIC - t.hilbert_hi::NUMERIC), ABS(a.hilbert_lo::NUMERIC - t.hilbert_lo::NUMERIC)
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Find nearest neighbors by spatial distance (accurate but slower)
CREATE OR REPLACE FUNCTION atom_nearest_spatial(p_id BYTEA, p_limit INTEGER DEFAULT 10)
RETURNS TABLE(neighbor_id BYTEA, distance DOUBLE PRECISION) AS $$
    SELECT
        a.id,
        ST_Distance(ST_Centroid(a.geom), ST_Centroid(target.geom))
    FROM atom a, atom target
    WHERE target.id = p_id AND a.id != p_id
    ORDER BY ST_Distance(ST_Centroid(a.geom), ST_Centroid(target.geom))
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

COMMIT;

-- =============================================================================
-- AI/ML Query Functions
-- =============================================================================

-- Search for compositions containing a text pattern
CREATE OR REPLACE FUNCTION atom_search_text(
    p_pattern TEXT,
    p_min_atoms INTEGER DEFAULT 1,
    p_max_atoms INTEGER DEFAULT 100,
    p_limit INTEGER DEFAULT 20
) RETURNS TABLE(
    id BYTEA,
    depth INTEGER,
    atom_count BIGINT,
    content TEXT
) AS $$
    SELECT 
        a.id,
        a.depth,
        a.atom_count,
        atom_reconstruct_text(a.id)
    FROM atom a
    WHERE a.depth > 0 
      AND a.atom_count BETWEEN p_min_atoms AND p_max_atoms
      AND atom_reconstruct_text(a.id) LIKE '%' || p_pattern || '%'
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Find compositions by exact content (content-addressed lookup)
CREATE OR REPLACE FUNCTION atom_find_exact(p_content TEXT)
RETURNS BYTEA AS $$
DECLARE
    v_id BYTEA;
BEGIN
    -- Look for existing composition with this exact content
    SELECT id INTO v_id
    FROM atom
    WHERE depth > 0 AND atom_reconstruct_text(id) = p_content
    LIMIT 1;
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql STABLE;

-- Get semantic neighbors of a composition (compositions with similar centroids)
CREATE OR REPLACE FUNCTION atom_semantic_neighbors(
    p_id BYTEA,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    neighbor_id BYTEA,
    content TEXT,
    distance DOUBLE PRECISION
) AS $$
    WITH target AS (
        SELECT ST_Centroid(geom) as centroid FROM atom WHERE id = p_id
    )
    SELECT 
        a.id,
        atom_reconstruct_text(a.id),
        ST_Distance(ST_Centroid(a.geom), t.centroid)
    FROM atom a, target t
    WHERE a.id != p_id AND a.depth > 0
    ORDER BY ST_Distance(ST_Centroid(a.geom), t.centroid)
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Get all compositions that contain a specific sub-composition
CREATE OR REPLACE FUNCTION atom_find_parents(p_child_id BYTEA, p_limit INTEGER DEFAULT 20)
RETURNS TABLE(parent_id BYTEA, depth INTEGER, atom_count BIGINT) AS $$
    SELECT id, depth, atom_count
    FROM atom
    WHERE p_child_id = ANY(children)
    ORDER BY depth
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Depth-first traversal returning all descendants
CREATE OR REPLACE FUNCTION atom_descendants(p_root_id BYTEA, p_max_depth INTEGER DEFAULT 10)
RETURNS TABLE(id BYTEA, depth INTEGER, level INTEGER, content TEXT) AS $$
    WITH RECURSIVE tree AS (
        SELECT id, depth, 0 as level, children
        FROM atom WHERE id = p_root_id
        
        UNION ALL
        
        SELECT a.id, a.depth, t.level + 1, a.children
        FROM tree t
        CROSS JOIN LATERAL unnest(t.children) AS child_id
        JOIN atom a ON a.id = child_id
        WHERE t.level < p_max_depth AND t.children IS NOT NULL
    )
    SELECT 
        t.id,
        t.depth,
        t.level,
        CASE WHEN t.depth = 0 THEN convert_from(a.value, 'UTF8') ELSE NULL END
    FROM tree t
    JOIN atom a ON a.id = t.id;
$$ LANGUAGE SQL STABLE;

-- Get vocabulary statistics (most common compositions at each depth)
CREATE OR REPLACE FUNCTION atom_vocabulary_stats(p_depth INTEGER DEFAULT 1, p_limit INTEGER DEFAULT 20)
RETURNS TABLE(id BYTEA, content TEXT, parent_count BIGINT) AS $$
    SELECT 
        a.id,
        atom_reconstruct_text(a.id),
        (SELECT COUNT(*) FROM atom p WHERE a.id = ANY(p.children))
    FROM atom a
    WHERE a.depth = p_depth
    ORDER BY (SELECT COUNT(*) FROM atom p WHERE a.id = ANY(p.children)) DESC
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

