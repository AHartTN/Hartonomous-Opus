-- Spatial Query Examples for Hartonomous Hypercube
-- Demonstrates O(log N) + O(K) geometric semantic queries
--
-- Key concepts:
-- - Atoms are points on S³ (3-sphere surface)
-- - Compositions are trajectories (LINESTRINGZM) through 4D space
-- - Centroids fall interior as hierarchy deepens
-- - Hilbert index preserves spatial locality for range queries
-- - PostGIS functions enable semantic search via geometry

-- ============================================================================
-- CENTROID DISTANCE: Find semantically similar content
-- ============================================================================

-- Find compositions closest to a given text
CREATE OR REPLACE FUNCTION hypercube_find_similar(
    p_text text,
    p_limit integer DEFAULT 10,
    p_min_atoms integer DEFAULT 3,
    p_max_atoms integer DEFAULT 50
)
RETURNS TABLE(
    content text,
    atom_count bigint,
    distance double precision,
    similarity double precision
)
AS $$
DECLARE
    v_target_id bytea;
    v_target_coords geometry;
BEGIN
    -- Ingest (or get existing) target
    v_target_id := hypercube_ingest_text(p_text);
    SELECT coords INTO v_target_coords FROM relation WHERE id = v_target_id;

    RETURN QUERY
    SELECT
        hypercube_retrieve_text(r.id) as content,
        r.atom_count,
        ST_Distance(r.coords, v_target_coords) as distance,
        1.0 / (1.0 + ST_Distance(r.coords, v_target_coords)) as similarity
    FROM relation r
    WHERE r.id != v_target_id
      AND r.atom_count BETWEEN p_min_atoms AND p_max_atoms
    ORDER BY ST_Distance(r.coords, v_target_coords)
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- HILBERT RANGE QUERIES: Find content in spatial region
-- ============================================================================

-- Find compositions in Hilbert index range (efficient locality query)
CREATE OR REPLACE FUNCTION hypercube_hilbert_range(
    p_center_text text,
    p_range bigint DEFAULT 1000000000,
    p_limit integer DEFAULT 20
)
RETURNS TABLE(
    content text,
    atom_count bigint,
    hilbert_lo bigint,
    hilbert_delta bigint
)
AS $$
DECLARE
    v_center_id bytea;
    v_center_lo bigint;
    v_center_hi bigint;
BEGIN
    v_center_id := hypercube_ingest_text(p_center_text);
    SELECT hilbert_lo, hilbert_hi INTO v_center_lo, v_center_hi
    FROM relation WHERE id = v_center_id;

    RETURN QUERY
    SELECT
        hypercube_retrieve_text(r.id) as content,
        r.atom_count,
        r.hilbert_lo,
        ABS(r.hilbert_lo - v_center_lo) as hilbert_delta
    FROM relation r
    WHERE r.hilbert_hi = v_center_hi  -- Same high region
      AND r.hilbert_lo BETWEEN v_center_lo - p_range AND v_center_lo + p_range
      AND r.id != v_center_id
    ORDER BY ABS(r.hilbert_lo - v_center_lo)
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- TRAJECTORY SIMILARITY: Frechet distance for sequence comparison
-- ============================================================================

-- Compare two texts using Frechet distance (trajectory shape similarity)
CREATE OR REPLACE FUNCTION hypercube_frechet_similarity(
    p_text1 text,
    p_text2 text
)
RETURNS TABLE(
    text1 text,
    text2 text,
    centroid_distance double precision,
    frechet_distance double precision,
    trajectory_similarity double precision
)
AS $$
DECLARE
    v_id1 bytea;
    v_id2 bytea;
    v_frechet double precision;
BEGIN
    v_id1 := hypercube_ingest_text(p_text1);
    v_id2 := hypercube_ingest_text(p_text2);
    v_frechet := hypercube_similarity(v_id1, v_id2);

    RETURN QUERY
    SELECT
        p_text1,
        p_text2,
        ST_Distance(r1.coords, r2.coords) as centroid_distance,
        1.0 / v_frechet - 1.0 as frechet_distance,
        v_frechet as trajectory_similarity
    FROM relation r1, relation r2
    WHERE r1.id = v_id1 AND r2.id = v_id2;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- GAP DETECTION: Find underrepresented regions in semantic space
-- ============================================================================

-- Find gaps in Hilbert space (regions with low composition density)
CREATE OR REPLACE FUNCTION hypercube_find_gaps(
    p_bucket_size bigint DEFAULT 100000000000000,
    p_min_gap_size integer DEFAULT 5
)
RETURNS TABLE(
    bucket_start bigint,
    bucket_end bigint,
    composition_count bigint,
    is_gap boolean
)
AS $$
BEGIN
    RETURN QUERY
    WITH buckets AS (
        SELECT
            (hilbert_lo / p_bucket_size) * p_bucket_size as bucket,
            count(*) as cnt
        FROM relation
        GROUP BY (hilbert_lo / p_bucket_size)
    ),
    all_buckets AS (
        SELECT generate_series(
            (SELECT min(bucket) FROM buckets),
            (SELECT max(bucket) FROM buckets),
            p_bucket_size
        ) as bucket
    )
    SELECT
        ab.bucket as bucket_start,
        ab.bucket + p_bucket_size as bucket_end,
        COALESCE(b.cnt, 0) as composition_count,
        COALESCE(b.cnt, 0) < p_min_gap_size as is_gap
    FROM all_buckets ab
    LEFT JOIN buckets b ON b.bucket = ab.bucket
    ORDER BY ab.bucket;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- VORONOI-STYLE NEAREST NEIGHBOR
-- ============================================================================

-- Find the single nearest existing composition to a query
CREATE OR REPLACE FUNCTION hypercube_nearest(p_text text)
RETURNS TABLE(
    content text,
    atom_count bigint,
    distance double precision
)
AS $$
DECLARE
    v_id bytea;
    v_coords geometry;
BEGIN
    v_id := hypercube_ingest_text(p_text);
    SELECT coords INTO v_coords FROM relation WHERE id = v_id;

    RETURN QUERY
    SELECT
        hypercube_retrieve_text(r.id),
        r.atom_count,
        ST_Distance(r.coords, v_coords)
    FROM relation r
    WHERE r.id != v_id
    ORDER BY r.coords <-> v_coords  -- KNN operator
    LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- INTERSECTION QUERIES: Find overlapping content
-- ============================================================================

-- Find compositions whose trajectories intersect spatially
-- (Content that passes through similar regions of semantic space)
CREATE OR REPLACE FUNCTION hypercube_find_intersecting(
    p_text text,
    p_buffer double precision DEFAULT 0.01,
    p_limit integer DEFAULT 20
)
RETURNS TABLE(
    content text,
    atom_count bigint,
    intersection_area double precision
)
AS $$
DECLARE
    v_id bytea;
    v_bbox geometry;
BEGIN
    v_id := hypercube_ingest_text(p_text);

    -- Create bounding box around target centroid
    SELECT ST_Buffer(coords, p_buffer) INTO v_bbox
    FROM relation WHERE id = v_id;

    RETURN QUERY
    SELECT
        hypercube_retrieve_text(r.id),
        r.atom_count,
        ST_Area(ST_Intersection(v_bbox, ST_Buffer(r.coords, p_buffer))) as intersection_area
    FROM relation r
    WHERE r.id != v_id
      AND r.coords && v_bbox  -- Bounding box intersection (fast)
    ORDER BY ST_Distance(r.coords, (SELECT coords FROM relation WHERE id = v_id))
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- CLUSTER ANALYSIS: Group similar compositions
-- ============================================================================

-- Simple clustering by Hilbert regions
CREATE OR REPLACE VIEW composition_clusters AS
WITH cluster_stats AS (
    SELECT
        (hilbert_hi::numeric * 1e18 + hilbert_lo) / 1e15 as cluster_id,
        count(*) as size,
        avg(atom_count) as avg_atoms,
        avg(ST_X(coords)) as centroid_x,
        avg(ST_Y(coords)) as centroid_y,
        avg(ST_Z(coords)) as centroid_z,
        avg(ST_M(coords)) as centroid_m
    FROM relation
    GROUP BY (hilbert_hi::numeric * 1e18 + hilbert_lo) / 1e15
)
SELECT
    cluster_id::bigint,
    size,
    avg_atoms::numeric(10,1),
    centroid_x::numeric(10,4),
    centroid_y::numeric(10,4),
    centroid_z::numeric(10,4),
    centroid_m::numeric(10,4)
FROM cluster_stats
WHERE size > 1
ORDER BY size DESC;

-- ============================================================================
-- STATISTICS VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW spatial_stats AS
SELECT
    'compositions' as entity,
    count(*) as total,
    avg(atom_count)::numeric(10,1) as avg_size,
    min(ST_X(coords))::numeric(10,4) as min_x,
    max(ST_X(coords))::numeric(10,4) as max_x,
    min(ST_Y(coords))::numeric(10,4) as min_y,
    max(ST_Y(coords))::numeric(10,4) as max_y
FROM relation
UNION ALL
SELECT
    'atoms',
    count(*),
    1,
    min(ST_X(coords))::numeric(10,4),
    max(ST_X(coords))::numeric(10,4),
    min(ST_Y(coords))::numeric(10,4),
    max(ST_Y(coords))::numeric(10,4)
FROM atom;

-- ============================================================================
-- USAGE EXAMPLES (as comments)
-- ============================================================================

-- Find content similar to "computer":
-- SELECT * FROM hypercube_find_similar('computer', 10);

-- Compare two texts:
-- SELECT * FROM hypercube_frechet_similarity('hello', 'hallo');

-- Find nearest neighbor:
-- SELECT * FROM hypercube_nearest('algorithm');

-- Hilbert range query:
-- SELECT * FROM hypercube_hilbert_range('programming', 1000000000000);

-- View cluster statistics:
-- SELECT * FROM composition_clusters LIMIT 20;

-- View spatial bounds:
-- SELECT * FROM spatial_stats;
