-- =============================================================================
-- GEOMETRY DISTANCE FUNCTIONS
-- =============================================================================
-- Core distance and similarity functions for 4D Laplacian-projected coordinates
-- These are the foundation for all similarity operations in the hypercube
-- =============================================================================

-- =============================================================================
-- 4D EUCLIDEAN DISTANCE
-- =============================================================================
-- Computes Euclidean distance between two 4D points (POINTZM geometry)
-- Used for similarity calculations throughout the system

CREATE OR REPLACE FUNCTION centroid_distance(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
    SELECT sqrt(
        power(ST_X(p_a) - ST_X(p_b), 2) +
        power(ST_Y(p_a) - ST_Y(p_b), 2) +
        power(ST_Z(p_a) - ST_Z(p_b), 2) +
        power(ST_M(p_a) - ST_M(p_b), 2)
    )
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

-- =============================================================================
-- 4D SIMILARITY (INVERSE DISTANCE)
-- =============================================================================
-- Converts distance to similarity score using inverse relationship
-- Higher values = more similar (range: 0 to 1)

CREATE OR REPLACE FUNCTION centroid_similarity(p_a GEOMETRY, p_b GEOMETRY)
RETURNS DOUBLE PRECISION AS $$
    SELECT 1.0 / (1.0 + centroid_distance(p_a, p_b))
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

-- =============================================================================
-- HILBERT DISTANCE (APPROXIMATION)
-- =============================================================================
-- Fast approximation using 128-bit Hilbert indices
-- Used for locality-sensitive pre-filtering before exact distance calculation

CREATE OR REPLACE FUNCTION hilbert_distance(
    p_lo_a BIGINT, p_hi_a BIGINT,
    p_lo_b BIGINT, p_hi_b BIGINT
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_diff_lo BIGINT;
    v_diff_hi BIGINT;
BEGIN
    v_diff_lo := abs(p_lo_a - p_lo_b);
    v_diff_hi := abs(p_hi_a - p_hi_b);
    -- Combine as 128-bit distance approximation
    RETURN v_diff_hi::DOUBLE PRECISION * 9223372036854775808.0 + v_diff_lo::DOUBLE PRECISION;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- =============================================================================
-- 4D CENTROID COMPUTATION
-- =============================================================================
-- Computes the centroid (average) of multiple 4D points
-- Essential for aggregating coordinates in composition hierarchies

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
    IF ST_GeometryType(geom) = 'ST_Point' THEN
        RETURN geom;
    END IF;

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

-- =============================================================================
-- COMPOSITION CENTROID COMPUTATION
-- =============================================================================
-- Computes centroid of a composition from its atom children
-- Handles hierarchical compositions by aggregating all leaf atoms

CREATE OR REPLACE FUNCTION compute_composition_centroid(comp_id bytea)
RETURNS geometry AS $$
    SELECT st_centroid_4d(ST_Collect(a.geom))
    FROM composition_child cc
    JOIN atom a ON a.id = cc.child_id
    WHERE cc.composition_id = comp_id;
$$ LANGUAGE sql STABLE PARALLEL SAFE;

-- =============================================================================
-- BATCH CENTROID RECOMPUTATION
-- =============================================================================
-- Recomputes centroids for all compositions in hierarchical order
-- Critical for maintaining coordinate consistency after data changes

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
            st_centroid_4d(ST_Collect(a.geom)) as new_centroid
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
                st_centroid_4d(ST_Collect(child.centroid)) as new_centroid
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

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON FUNCTION centroid_distance IS '4D Euclidean distance between two POINTZM geometries';
COMMENT ON FUNCTION centroid_similarity IS 'Similarity score (0-1) based on inverse distance';
COMMENT ON FUNCTION hilbert_distance IS 'Fast Hilbert index distance approximation for pre-filtering';
COMMENT ON FUNCTION st_centroid_4d IS '4D centroid computation for XYZM coordinate averaging';
COMMENT ON FUNCTION compute_composition_centroid IS 'Computes composition centroid from atom children';
COMMENT ON FUNCTION recompute_composition_centroids IS 'Batch centroid recomputation for consistency';