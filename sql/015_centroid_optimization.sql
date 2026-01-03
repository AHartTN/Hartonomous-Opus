-- =============================================================================
-- Hartonomous Hypercube - Centroid Optimization v4
-- =============================================================================
-- Ensures centroid column is populated for legacy data.
-- =============================================================================

BEGIN;

-- Maintenance: populate missing centroids in batches
CREATE OR REPLACE FUNCTION maintenance_centroids(p_batch INTEGER DEFAULT 10000)
RETURNS INTEGER AS $$
DECLARE
    v_total INTEGER := 0;
    v_batch INTEGER;
BEGIN
    -- Points: centroid = geom
    LOOP
        UPDATE atom SET centroid = geom
        WHERE id IN (
            SELECT id FROM atom 
            WHERE centroid IS NULL AND ST_GeometryType(geom) = 'ST_Point'
            LIMIT p_batch
        );
        GET DIAGNOSTICS v_batch = ROW_COUNT;
        v_total := v_total + v_batch;
        EXIT WHEN v_batch < p_batch;
    END LOOP;
    
    -- LineStrings: compute 4D centroid
    LOOP
        UPDATE atom SET centroid = (
            SELECT ST_SetSRID(ST_MakePoint(
                ST_X(ST_Centroid(geom)),
                ST_Y(ST_Centroid(geom)),
                (SELECT AVG(ST_Z(g.geom)) FROM ST_DumpPoints(atom.geom) g),
                (SELECT AVG(ST_M(g.geom)) FROM ST_DumpPoints(atom.geom) g)
            ), 0)
        )
        WHERE id IN (
            SELECT id FROM atom 
            WHERE centroid IS NULL AND ST_GeometryType(geom) != 'ST_Point'
            LIMIT p_batch
        );
        GET DIAGNOSTICS v_batch = ROW_COUNT;
        v_total := v_total + v_batch;
        EXIT WHEN v_batch < p_batch;
    END LOOP;
    
    RETURN v_total;
END;
$$ LANGUAGE plpgsql;

-- Stats view
CREATE OR REPLACE VIEW v_centroid_status AS
SELECT 
    CASE WHEN centroid IS NULL THEN 'missing' ELSE 'ok' END AS status,
    COUNT(*) AS count
FROM atom GROUP BY 1;

COMMIT;
