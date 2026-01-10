-- =============================================================================
-- RELATION EDGE FUNCTIONS
-- =============================================================================
-- Functions for managing and generating semantic edges
-- =============================================================================



-- Generate k-NN semantic edges from composition centroids
CREATE OR REPLACE FUNCTION generate_knn_edges(
    p_k integer DEFAULT 10,
    p_model_name text DEFAULT 'centroid_knn'
)
RETURNS integer AS $$
DECLARE
    inserted integer := 0;
BEGIN
    INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component)
    SELECT
        'C', c1.id, 'C', neighbor.id, 'S',
        1.0 / (1.0 + neighbor.dist),
        p_model_name, -1, 'knn'
    FROM composition c1
    CROSS JOIN LATERAL (
        SELECT c2.id, c1.centroid <-> c2.centroid AS dist
        FROM composition c2
        WHERE c2.id != c1.id
          AND c2.centroid IS NOT NULL
        ORDER BY c1.centroid <-> c2.centroid
        LIMIT p_k
    ) neighbor
    WHERE c1.centroid IS NOT NULL
    ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET
        weight = GREATEST(relation.weight, EXCLUDED.weight),
        source_count = relation.source_count + 1;

    GET DIAGNOSTICS inserted = ROW_COUNT;
    RETURN inserted;
END;
$$ LANGUAGE plpgsql;