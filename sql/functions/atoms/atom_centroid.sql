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