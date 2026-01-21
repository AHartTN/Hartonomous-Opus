-- Euclidean distance between two 4D points (POINTZM geometry)
CREATE OR REPLACE FUNCTION centroid_distance(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
    SELECT sqrt(
        power(ST_X(p_a) - ST_X(p_b), 2) +
        power(ST_Y(p_a) - ST_Y(p_b), 2) +
        power(ST_Z(p_a) - ST_Z(p_b), 2) +
        power(ST_M(p_a) - ST_M(p_b), 2)
    )
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;