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