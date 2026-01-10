-- =============================================================================
-- GENERATIVE QUERY FUNCTIONS
-- =============================================================================
-- Token generation and prompting capabilities for the hypercube
-- =============================================================================

-- Semantic analogy: A:B::C:? using vector arithmetic
CREATE OR REPLACE FUNCTION vector_analogy(
    p_a TEXT,
    p_b TEXT,
    p_c TEXT,
    p_k INTEGER DEFAULT 5
)
RETURNS TABLE(
    label TEXT,
    score DOUBLE PRECISION
) AS $$
DECLARE
    v_a GEOMETRY;
    v_b GEOMETRY;
    v_c GEOMETRY;
    v_target_x DOUBLE PRECISION;
    v_target_y DOUBLE PRECISION;
    v_target_z DOUBLE PRECISION;
    v_target_m DOUBLE PRECISION;
    v_target GEOMETRY;
BEGIN
    -- Get centroids
    SELECT centroid INTO v_a FROM composition WHERE label = p_a AND centroid IS NOT NULL LIMIT 1;
    SELECT centroid INTO v_b FROM composition WHERE label = p_b AND centroid IS NOT NULL LIMIT 1;
    SELECT centroid INTO v_c FROM composition WHERE label = p_c AND centroid IS NOT NULL LIMIT 1;

    IF v_a IS NULL OR v_b IS NULL OR v_c IS NULL THEN
        RAISE WARNING 'One or more tokens not found with centroid';
        RETURN;
    END IF;

    -- Compute target: C + (B - A)
    v_target_x := ST_X(v_c) + (ST_X(v_b) - ST_X(v_a));
    v_target_y := ST_Y(v_c) + (ST_Y(v_b) - ST_Y(v_a));
    v_target_z := ST_Z(v_c) + (ST_Z(v_b) - ST_Z(v_a));
    v_target_m := ST_M(v_c) + (ST_M(v_b) - ST_M(v_a));

    v_target := ST_SetSRID(ST_MakePoint(v_target_x, v_target_y, v_target_z, v_target_m), 0);

    -- Find nearest to target
    RETURN QUERY
    SELECT
        c.label,
        1.0 / (1.0 + centroid_distance(c.centroid, v_target)) as sim
    FROM composition c
    WHERE c.centroid IS NOT NULL
      AND c.label IS NOT NULL
      AND c.label NOT IN (p_a, p_b, p_c)
      AND c.label NOT LIKE '[%'
    ORDER BY centroid_distance(c.centroid, v_target) ASC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;