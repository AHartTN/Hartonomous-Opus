-- =============================================================================
-- RELATION EDGE FUNCTIONS
-- =============================================================================
-- Functions for managing and generating semantic edges
-- =============================================================================

-- Upsert relation with weight averaging
CREATE OR REPLACE FUNCTION upsert_relation(
    p_source_type CHAR(1),
    p_source_id BYTEA,
    p_target_type CHAR(1),
    p_target_id BYTEA,
    p_relation_type CHAR(1),
    p_weight REAL,
    p_source_model TEXT DEFAULT '',
    p_layer INTEGER DEFAULT -1,
    p_component TEXT DEFAULT ''
) RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component)
    VALUES (p_source_type, p_source_id, p_target_type, p_target_id, p_relation_type, p_weight,
            COALESCE(p_source_model, ''), COALESCE(p_layer, -1), COALESCE(p_component, ''))
    ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET
        weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1),
        source_count = relation.source_count + 1
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

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