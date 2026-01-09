-- =============================================================================
-- ATOM CORE FUNCTIONS
-- =============================================================================
-- Basic atom operations for the 3-table hypercube schema
-- =============================================================================

-- Check if entity is a leaf (atom) vs composition
CREATE OR REPLACE FUNCTION atom_is_leaf(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id);
$$ LANGUAGE SQL STABLE;

-- Get 4D centroid from atom or composition
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

-- Check if hash exists (atom or composition)
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

-- atom_distance - compute 3D distance between two atoms (compatibility)
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