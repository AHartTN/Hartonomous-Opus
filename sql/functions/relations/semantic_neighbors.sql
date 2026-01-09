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