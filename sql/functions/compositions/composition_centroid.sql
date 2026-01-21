-- Get 4D centroid from composition
CREATE OR REPLACE FUNCTION composition_centroid(p_id BYTEA)
RETURNS TABLE(x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION) AS $$
    SELECT ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid)
    FROM composition WHERE id = p_id;
$$ LANGUAGE SQL STABLE;