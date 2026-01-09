-- =============================================================================
-- RELATION CORE FUNCTIONS
-- =============================================================================
-- Basic semantic graph operations for the relation table
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