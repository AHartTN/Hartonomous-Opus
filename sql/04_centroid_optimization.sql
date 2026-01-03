-- =============================================================================
-- Centroid Optimization - Pre-computed centroid for fast spatial queries
-- =============================================================================
--
-- Problem: Computing ST_Centroid(geom) on every query is expensive
-- Solution: Store pre-computed centroid as POINTZM, index it
--
-- For leaves (POINTZM): centroid = geom (same point)
-- For compositions (LINESTRINGZM): centroid = geometric center of trajectory

BEGIN;

-- Add centroid column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'atom' AND column_name = 'centroid'
    ) THEN
        ALTER TABLE atom ADD COLUMN centroid GEOMETRY(POINTZM, 0);
    END IF;
END
$$;

-- Create spatial index on centroid
CREATE INDEX IF NOT EXISTS idx_atom_centroid ON atom USING GIST(centroid);

-- NOTE: Centroids are computed on INSERT via trigger below.
-- For existing data without centroids, run maintenance manually:
--   SELECT maintenance_compute_centroids();

-- Create trigger to auto-populate centroid on insert/update
CREATE OR REPLACE FUNCTION atom_compute_centroid() RETURNS TRIGGER AS $$
DECLARE
    v_centroid_2d GEOMETRY;
    v_avg_z DOUBLE PRECISION;
    v_avg_m DOUBLE PRECISION;
BEGIN
    IF ST_GeometryType(NEW.geom) = 'ST_Point' THEN
        -- For POINTZM leaves: centroid is the point itself
        NEW.centroid := NEW.geom;
    ELSE
        -- For LINESTRINGZM compositions: compute 4D centroid
        -- ST_Centroid gives us X,Y; we need to compute Z,M from all vertices
        v_centroid_2d := ST_Centroid(NEW.geom);
        
        SELECT AVG(ST_Z(geom)), AVG(ST_M(geom))
        INTO v_avg_z, v_avg_m
        FROM ST_DumpPoints(NEW.geom);
        
        NEW.centroid := ST_SetSRID(
            ST_MakePoint(ST_X(v_centroid_2d), ST_Y(v_centroid_2d), v_avg_z, v_avg_m),
            0
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_atom_centroid ON atom;
CREATE TRIGGER trg_atom_centroid
    BEFORE INSERT OR UPDATE OF geom ON atom
    FOR EACH ROW
    EXECUTE FUNCTION atom_compute_centroid();

-- =============================================================================
-- Fast spatial query functions using pre-computed centroid
-- =============================================================================

-- Fast 4D distance using centroid
CREATE OR REPLACE FUNCTION atom_distance_fast(p_id1 BYTEA, p_id2 BYTEA) 
RETURNS DOUBLE PRECISION AS $$
    SELECT sqrt(
        power(ST_X(a1.centroid) - ST_X(a2.centroid), 2) +
        power(ST_Y(a1.centroid) - ST_Y(a2.centroid), 2) +
        power(ST_Z(a1.centroid) - ST_Z(a2.centroid), 2) +
        power(ST_M(a1.centroid) - ST_M(a2.centroid), 2)
    )
    FROM atom a1, atom a2
    WHERE a1.id = p_id1 AND a2.id = p_id2;
$$ LANGUAGE SQL STABLE;

-- Fast nearest neighbors using centroid (can use spatial index)
CREATE OR REPLACE FUNCTION atom_nearest_fast(p_id BYTEA, p_limit INTEGER DEFAULT 10)
RETURNS TABLE(neighbor_id BYTEA, distance DOUBLE PRECISION) AS $$
    SELECT
        a.id,
        sqrt(
            power(ST_X(a.centroid) - ST_X(target.centroid), 2) +
            power(ST_Y(a.centroid) - ST_Y(target.centroid), 2) +
            power(ST_Z(a.centroid) - ST_Z(target.centroid), 2) +
            power(ST_M(a.centroid) - ST_M(target.centroid), 2)
        )
    FROM atom a, atom target
    WHERE target.id = p_id AND a.id != p_id
    ORDER BY a.centroid <-> target.centroid  -- Uses GIST index
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Find compositions within radius of a point
CREATE OR REPLACE FUNCTION atom_within_radius(
    p_x DOUBLE PRECISION,
    p_y DOUBLE PRECISION,
    p_z DOUBLE PRECISION,
    p_m DOUBLE PRECISION,
    p_radius DOUBLE PRECISION,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE(id BYTEA, depth INTEGER, atom_count BIGINT, distance DOUBLE PRECISION) AS $$
    WITH target AS (
        SELECT ST_SetSRID(ST_MakePoint(p_x, p_y, p_z, p_m), 0) as pt
    )
    SELECT 
        a.id,
        a.depth,
        a.atom_count,
        sqrt(
            power(ST_X(a.centroid) - p_x, 2) +
            power(ST_Y(a.centroid) - p_y, 2) +
            power(ST_Z(a.centroid) - p_z, 2) +
            power(ST_M(a.centroid) - p_m, 2)
        ) as dist
    FROM atom a, target t
    WHERE ST_DWithin(a.centroid, t.pt, p_radius)
    ORDER BY a.centroid <-> t.pt
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Find compositions with centroids in a bounding box
CREATE OR REPLACE FUNCTION atom_in_bbox(
    p_x_min DOUBLE PRECISION, p_x_max DOUBLE PRECISION,
    p_y_min DOUBLE PRECISION, p_y_max DOUBLE PRECISION,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE(id BYTEA, depth INTEGER, atom_count BIGINT) AS $$
    SELECT id, depth, atom_count
    FROM atom
    WHERE centroid && ST_MakeEnvelope(p_x_min, p_y_min, p_x_max, p_y_max, 0)
    ORDER BY hilbert_hi, hilbert_lo
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

COMMIT;

-- =============================================================================
-- Maintenance Functions (run manually, not during schema setup)
-- =============================================================================

-- Maintenance: Populate centroids for existing data that lacks them
CREATE OR REPLACE FUNCTION maintenance_compute_centroids(p_batch_size INTEGER DEFAULT 10000)
RETURNS INTEGER AS $$
DECLARE
    v_updated INTEGER := 0;
    v_batch INTEGER;
BEGIN
    LOOP
        -- For leaves (POINT): centroid = geom directly
        UPDATE atom 
        SET centroid = geom
        WHERE id IN (
            SELECT id FROM atom 
            WHERE centroid IS NULL AND ST_GeometryType(geom) = 'ST_Point' 
            LIMIT p_batch_size
        );
        GET DIAGNOSTICS v_batch = ROW_COUNT;
        v_updated := v_updated + v_batch;
        
        -- For compositions (LINESTRING): compute proper 4D centroid
        UPDATE atom 
        SET centroid = (
            SELECT ST_SetSRID(
                ST_MakePoint(
                    ST_X(ST_Centroid(geom)),
                    ST_Y(ST_Centroid(geom)),
                    AVG(ST_Z(dp.geom)),
                    AVG(ST_M(dp.geom))
                ),
                0
            )
            FROM ST_DumpPoints(atom.geom) dp
        )
        WHERE id IN (
            SELECT id FROM atom 
            WHERE centroid IS NULL AND ST_GeometryType(geom) = 'ST_LineString' 
            LIMIT p_batch_size
        );
        GET DIAGNOSTICS v_batch = ROW_COUNT;
        v_updated := v_updated + v_batch;
        
        IF v_batch < p_batch_size THEN
            EXIT;
        END IF;
        
        RAISE NOTICE 'Computed % centroids so far...', v_updated;
    END LOOP;
    
    RETURN v_updated;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Statistics View
-- =============================================================================

-- View for centroid coverage
CREATE OR REPLACE VIEW atom_centroid_stats AS
SELECT 
    CASE WHEN centroid IS NOT NULL THEN 'has_centroid' ELSE 'missing' END as status,
    depth,
    COUNT(*) as count
FROM atom
GROUP BY status, depth
ORDER BY depth, status;
