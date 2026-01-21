-- attention - attention scores (similarity in 4D space)
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