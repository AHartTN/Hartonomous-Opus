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