-- =============================================================================
-- 4D Centroid Functions for Hypercube
-- PostGIS ST_Centroid only handles 2D, so we need custom 4D centroid
-- =============================================================================

-- Compute 4D centroid (average of all points in XYZM space)
CREATE OR REPLACE FUNCTION st_centroid_4d(geom geometry)
RETURNS geometry AS $$
DECLARE
    n integer;
    cx double precision := 0;
    cy double precision := 0;
    cz double precision := 0;
    cm double precision := 0;
    rec record;
BEGIN
    -- Single point returns itself
    IF ST_GeometryType(geom) = 'ST_Point' THEN
        RETURN geom;
    END IF;
    
    -- Count points and sum coordinates
    n := 0;
    FOR rec IN SELECT (ST_DumpPoints(geom)).geom AS pt LOOP
        cx := cx + ST_X(rec.pt);
        cy := cy + ST_Y(rec.pt);
        cz := cz + COALESCE(ST_Z(rec.pt), 0);
        cm := cm + COALESCE(ST_M(rec.pt), 0);
        n := n + 1;
    END LOOP;
    
    IF n = 0 THEN
        RETURN NULL;
    END IF;
    
    RETURN ST_SetSRID(ST_MakePoint(cx/n, cy/n, cz/n, cm/n), ST_SRID(geom));
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

COMMENT ON FUNCTION st_centroid_4d(geometry) IS 
'Compute true 4D centroid (XYZM) of a geometry. PostGIS ST_Centroid only handles XY.';

-- Compute composition centroid from its atom children
CREATE OR REPLACE FUNCTION compute_composition_centroid(comp_id bytea)
RETURNS geometry AS $$
    SELECT st_centroid_4d(ST_Collect(a.geom))
    FROM composition_child cc
    JOIN atom a ON a.id = cc.child_id
    WHERE cc.composition_id = comp_id;
$$ LANGUAGE sql STABLE PARALLEL SAFE;

COMMENT ON FUNCTION compute_composition_centroid(bytea) IS
'Compute a composition centroid from the 4D coordinates of its atom children.';

-- Update all composition centroids from their atom children
CREATE OR REPLACE FUNCTION recompute_composition_centroids(batch_size integer DEFAULT 10000)
RETURNS integer AS $$
DECLARE
    updated integer := 0;
    batch integer;
    processed_ids bytea[];
BEGIN
    -- Process compositions that have children in composition_child table
    -- Using cursor-style batching with OFFSET would be slow, so we mark progress differently
    
    -- Single pass: update all compositions that have children
    WITH comp_centroids AS (
        SELECT 
            cc.composition_id as id,
            st_centroid_4d(ST_Collect(a.geom)) as new_centroid
        FROM composition_child cc
        JOIN atom a ON a.id = cc.child_id
        GROUP BY cc.composition_id
    )
    UPDATE composition c
    SET centroid = comp_centroids.new_centroid
    FROM comp_centroids
    WHERE c.id = comp_centroids.id;
    
    GET DIAGNOSTICS updated = ROW_COUNT;
    
    RETURN updated;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION recompute_composition_centroids(integer) IS
'Recompute all composition centroids from their atom children. Use after ingestion.';

-- =============================================================================
-- Generate k-NN semantic edges from composition centroids
-- =============================================================================
CREATE OR REPLACE FUNCTION generate_knn_edges(
    p_k integer DEFAULT 10,
    p_model_name text DEFAULT 'centroid_knn'
)
RETURNS integer AS $$
DECLARE
    inserted integer := 0;
BEGIN
    -- Insert k-NN edges for each composition based on centroid distance
    -- Using lateral join for efficient k-NN computation
    INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component)
    SELECT 
        'C', c1.id, 'C', neighbor.id, 'S',
        1.0 / (1.0 + neighbor.dist),  -- Convert distance to similarity weight
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

COMMENT ON FUNCTION generate_knn_edges(integer, text) IS
'Generate semantic similarity edges from composition centroids using k-NN.';

-- Test the function
DO $$
DECLARE
    test_result geometry;
BEGIN
    SELECT st_centroid_4d(ST_Collect(a.geom))
    INTO test_result
    FROM composition c 
    JOIN composition_child cc ON cc.composition_id = c.id 
    JOIN atom a ON a.id = cc.child_id 
    WHERE c.label = 'whale';
    
    RAISE NOTICE 'whale centroid: %', ST_AsText(test_result);
END $$;
