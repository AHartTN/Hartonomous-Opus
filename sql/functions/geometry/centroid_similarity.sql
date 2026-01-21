-- Convert distance to similarity (inverse, normalized)
CREATE OR REPLACE FUNCTION centroid_similarity(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
    SELECT 1.0 / (1.0 + centroid_distance(p_a, p_b))
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;