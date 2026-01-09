-- =============================================================================
-- RECOMPUTE COMPOSITION CENTROIDS
-- =============================================================================
-- Recomputes all composition centroids from their children (atoms AND compositions)
-- Handles hierarchical compositions by propagating from leaves up to root
-- =============================================================================

CREATE OR REPLACE FUNCTION recompute_composition_centroids(batch_size integer DEFAULT 10000)
RETURNS integer AS $$
DECLARE
    updated integer := 0;
    total_updated integer := 0;
    max_depth integer;
    current_depth integer;
BEGIN
    -- First pass: compositions with ATOM children (leaf compositions)
    WITH comp_centroids AS (
        SELECT
            cc.composition_id as id,
            ST_SetSRID(st_centroid_4d(ST_Collect(a.geom)), 0) as new_centroid
        FROM composition_child cc
        JOIN atom a ON a.id = cc.child_id
        WHERE cc.child_type = 'A'
        GROUP BY cc.composition_id
    )
    UPDATE composition c
    SET centroid = comp_centroids.new_centroid
    FROM comp_centroids
    WHERE c.id = comp_centroids.id;

    GET DIAGNOSTICS updated = ROW_COUNT;
    total_updated := total_updated + updated;

    -- Get max depth for iterative propagation
    SELECT MAX(depth) INTO max_depth FROM composition;
    IF max_depth IS NULL THEN
        RETURN total_updated;
    END IF;

    -- Propagate centroids up the tree, from deepest to shallowest
    -- Start from max_depth and work up to 1
    FOR current_depth IN REVERSE max_depth..1 LOOP
        WITH comp_centroids AS (
            SELECT
                cc.composition_id as id,
                ST_SetSRID(st_centroid_4d(ST_Collect(child.centroid)), 0) as new_centroid
            FROM composition_child cc
            JOIN composition child ON child.id = cc.child_id
            JOIN composition parent ON parent.id = cc.composition_id
            WHERE cc.child_type = 'C'
              AND parent.depth = current_depth
              AND child.centroid IS NOT NULL
            GROUP BY cc.composition_id
            HAVING COUNT(*) > 0
        )
        UPDATE composition c
        SET centroid = comp_centroids.new_centroid
        FROM comp_centroids
        WHERE c.id = comp_centroids.id
          AND c.centroid IS NULL;

        GET DIAGNOSTICS updated = ROW_COUNT;
        total_updated := total_updated + updated;
    END LOOP;

    RETURN total_updated;
END;
$$ LANGUAGE plpgsql;