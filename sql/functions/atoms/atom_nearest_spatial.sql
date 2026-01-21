-- atom_nearest_spatial - compatibility wrapper for spatial KNN queries
-- Returns k-nearest neighbors by spatial distance (for atoms, uses geom)
CREATE OR REPLACE FUNCTION atom_nearest_spatial(p_id BYTEA, p_limit INTEGER DEFAULT 5)
RETURNS TABLE(neighbor_id BYTEA, distance DOUBLE PRECISION) AS $$
    SELECT * FROM atom_knn(p_id, p_limit);
$$ LANGUAGE SQL STABLE;