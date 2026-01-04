-- =============================================================================
-- FOUR TABLE SCHEMA - Core Functions
-- =============================================================================
-- These functions work with the 4-table schema:
--   atom:             Unicode codepoints only (ALL are leaves)
--   composition:      Aggregations with depth
--   composition_child: Ordered children
--   relation:         Semantic edges
-- =============================================================================

BEGIN;

-- Drop ALL old functions that assume unified schema (different signatures too)
DROP FUNCTION IF EXISTS atom_is_leaf(BYTEA) CASCADE;
DROP FUNCTION IF EXISTS atom_centroid(BYTEA) CASCADE;
DROP FUNCTION IF EXISTS atom_children(BYTEA) CASCADE;
DROP FUNCTION IF EXISTS atom_child_count(BYTEA) CASCADE;
DROP FUNCTION IF EXISTS atom_reconstruct_text(BYTEA) CASCADE;
DROP FUNCTION IF EXISTS atom_stats() CASCADE;
DROP FUNCTION IF EXISTS atom_knn(BYTEA, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS semantic_neighbors(BYTEA, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS attention(BYTEA, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS analogy(BYTEA, BYTEA, BYTEA, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS atom_hilbert_range(BIGINT, BIGINT, BIGINT, BIGINT) CASCADE;
DROP FUNCTION IF EXISTS atom_distance(BYTEA, BYTEA) CASCADE;

-- =============================================================================
-- ATOM FUNCTIONS (leaf-only table)
-- =============================================================================

-- Check if entity is a leaf (atom) vs composition
-- In 4-table schema: if it's in atom table, it's a leaf
CREATE OR REPLACE FUNCTION atom_is_leaf(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id);
$$ LANGUAGE SQL STABLE;

-- Get 4D centroid from atom (geom IS the centroid for atoms)
CREATE OR REPLACE FUNCTION atom_centroid(p_id BYTEA)
RETURNS GEOMETRY(POINTZM, 0) AS $$
    -- Try atom table first
    SELECT geom FROM atom WHERE id = p_id
    UNION ALL
    -- Fall back to composition centroid
    SELECT centroid FROM composition WHERE id = p_id
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Get children of a composition (empty for atoms)
CREATE OR REPLACE FUNCTION atom_children(p_id BYTEA)
RETURNS TABLE(child_id BYTEA, child_type CHAR(1), ordinal SMALLINT) AS $$
    SELECT cc.child_id, cc.child_type, cc.ordinal
    FROM composition_child cc
    WHERE cc.composition_id = p_id
    ORDER BY cc.ordinal;
$$ LANGUAGE SQL STABLE;

-- Get child count (0 for atoms)
CREATE OR REPLACE FUNCTION atom_child_count(p_id BYTEA)
RETURNS INTEGER AS $$
    SELECT COALESCE(
        (SELECT child_count FROM composition WHERE id = p_id),
        0
    );
$$ LANGUAGE SQL STABLE;

-- Lookup by codepoint
CREATE OR REPLACE FUNCTION atom_by_codepoint(p_cp INTEGER)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE codepoint = p_cp;
$$ LANGUAGE SQL STABLE;

-- Check if hash exists
CREATE OR REPLACE FUNCTION atom_exists(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id)
        OR EXISTS(SELECT 1 FROM composition WHERE id = p_id);
$$ LANGUAGE SQL STABLE;

-- Get text from atom (single character)
CREATE OR REPLACE FUNCTION atom_text(p_id BYTEA)
RETURNS TEXT AS $$
    SELECT chr(codepoint) FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- TEXT RECONSTRUCTION (recursive for compositions)
-- =============================================================================

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
-- SPATIAL QUERIES
-- =============================================================================

-- K-nearest neighbors using geometry
CREATE OR REPLACE FUNCTION atom_knn(p_id BYTEA, p_k INTEGER DEFAULT 5)
RETURNS TABLE(neighbor_id BYTEA, distance DOUBLE PRECISION) AS $$
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
    
    -- Return nearest atoms
    RETURN QUERY
    SELECT a.id, ST_3DDistance(a.geom, v_geom)
    FROM atom a
    WHERE a.id != p_id
    ORDER BY a.geom <-> v_geom
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- Hilbert range query
CREATE OR REPLACE FUNCTION atom_hilbert_range(p_hi_lo BIGINT, p_hi_hi BIGINT, p_lo_lo BIGINT, p_lo_hi BIGINT)
RETURNS TABLE(id BYTEA, codepoint INTEGER) AS $$
    SELECT id, codepoint FROM atom
    WHERE hilbert_hi BETWEEN p_hi_lo AND p_hi_hi
      AND hilbert_lo BETWEEN p_lo_lo AND p_lo_hi
    ORDER BY hilbert_hi, hilbert_lo;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- SEMANTIC QUERIES (using relation table)
-- =============================================================================

-- Get semantic neighbors from relation table
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
-- STATISTICS (function and view for compatibility)
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_stats()
RETURNS TABLE(
    atoms BIGINT,
    compositions BIGINT,
    compositions_with_centroid BIGINT,
    relations BIGINT,
    max_depth INTEGER
) AS $$
    SELECT 
        (SELECT COUNT(*) FROM atom),
        (SELECT COUNT(*) FROM composition),
        (SELECT COUNT(*) FROM composition WHERE centroid IS NOT NULL),
        (SELECT COUNT(*) FROM relation),
        (SELECT COALESCE(MAX(depth), 0) FROM composition);
$$ LANGUAGE SQL STABLE;

-- Compatibility view for atom_stats (C++ tests expect a table)
DROP VIEW IF EXISTS atom_stats_view CASCADE;
CREATE VIEW atom_stats_view AS
SELECT * FROM atom_stats();

-- Compatibility: atom_type_stats for C++ tests
DROP VIEW IF EXISTS atom_type_stats CASCADE;
CREATE VIEW atom_type_stats AS
SELECT 
    'atom' AS entity_type,
    COUNT(*) AS count,
    0 AS max_depth
FROM atom
UNION ALL
SELECT 
    'composition' AS entity_type,
    COUNT(*) AS count,
    COALESCE(MAX(depth), 0) AS max_depth
FROM composition
UNION ALL
SELECT 
    'relation' AS entity_type,
    COUNT(*) AS count,
    0 AS max_depth
FROM relation;

-- =============================================================================
-- COMPATIBILITY FUNCTIONS (for C++ tests that expect old schema)
-- =============================================================================

-- atom_distance - compute 3D distance between two atoms
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

-- atom_nearest_spatial - compatibility wrapper
CREATE OR REPLACE FUNCTION atom_nearest_spatial(p_id BYTEA, p_limit INTEGER DEFAULT 5)
RETURNS TABLE(neighbor_id BYTEA, distance DOUBLE PRECISION) AS $$
    SELECT * FROM atom_knn(p_id, p_limit);
$$ LANGUAGE SQL STABLE;

COMMIT;
