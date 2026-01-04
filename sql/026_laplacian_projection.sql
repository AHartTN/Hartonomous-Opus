-- =============================================================================
-- 4D Laplacian Projection SQL Helpers
-- =============================================================================
-- SQL functions for upserting atoms and compositions with 4D coordinates
-- from the Laplacian eigenmap projection pipeline.
--
-- Usage: Called by ingest_safetensor after computing 4D projections
-- =============================================================================

-- Ensure extensions
CREATE EXTENSION IF NOT EXISTS postgis;

-- -----------------------------------------------------------------------------
-- Atom upsert with 4D geometry
-- -----------------------------------------------------------------------------
-- Upserts an atom with POINTZM geometry and Hilbert index
-- Parameters:
--   p_id: BLAKE3 hash (BYTEA)
--   p_label: Token text
--   p_x, p_y, p_z, p_m: 4D coordinates (stored as doubles in POINTZM)
--   p_hilbert_lo, p_hilbert_hi: 128-bit Hilbert index
-- Returns: atom id

CREATE OR REPLACE FUNCTION upsert_atom_4d(
    p_id BYTEA,
    p_label TEXT,
    p_x DOUBLE PRECISION,
    p_y DOUBLE PRECISION,
    p_z DOUBLE PRECISION,
    p_m DOUBLE PRECISION,
    p_hilbert_lo BIGINT,
    p_hilbert_hi BIGINT
) RETURNS BYTEA
LANGUAGE plpgsql
AS $$
DECLARE
    v_geom GEOMETRY;
BEGIN
    -- Build POINTZM geometry
    v_geom := ST_SetSRID(ST_MakePoint(p_x, p_y, p_z, p_m), 0);
    
    -- Upsert atom
    INSERT INTO atom (id, label, geom, hilbert_lo, hilbert_hi)
    VALUES (p_id, p_label, v_geom, p_hilbert_lo, p_hilbert_hi)
    ON CONFLICT (id) DO UPDATE SET
        geom = EXCLUDED.geom,
        hilbert_lo = EXCLUDED.hilbert_lo,
        hilbert_hi = EXCLUDED.hilbert_hi
    WHERE atom.geom IS NULL;  -- Only update if no existing geometry
    
    RETURN p_id;
END;
$$;

-- -----------------------------------------------------------------------------
-- Composition upsert with 4D centroid
-- -----------------------------------------------------------------------------
-- Upserts a composition (depth-1 token) with centroid and Hilbert index
-- Parameters:
--   p_id: BLAKE3 hash (BYTEA)
--   p_label: Token text
--   p_x, p_y, p_z, p_m: 4D centroid coordinates
--   p_hilbert_lo, p_hilbert_hi: 128-bit Hilbert index
-- Returns: composition id

CREATE OR REPLACE FUNCTION upsert_composition_4d(
    p_id BYTEA,
    p_label TEXT,
    p_x DOUBLE PRECISION,
    p_y DOUBLE PRECISION,
    p_z DOUBLE PRECISION,
    p_m DOUBLE PRECISION,
    p_hilbert_lo BIGINT,
    p_hilbert_hi BIGINT
) RETURNS BYTEA
LANGUAGE plpgsql
AS $$
DECLARE
    v_centroid GEOMETRY;
BEGIN
    -- Build POINTZM centroid
    v_centroid := ST_SetSRID(ST_MakePoint(p_x, p_y, p_z, p_m), 0);
    
    -- Upsert composition
    INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi)
    VALUES (p_id, p_label, 1, 1, 1, v_centroid, p_hilbert_lo, p_hilbert_hi)
    ON CONFLICT (id) DO UPDATE SET
        centroid = EXCLUDED.centroid,
        hilbert_lo = EXCLUDED.hilbert_lo,
        hilbert_hi = EXCLUDED.hilbert_hi
    WHERE composition.centroid IS NULL;  -- Only update if no existing centroid
    
    RETURN p_id;
END;
$$;

-- -----------------------------------------------------------------------------
-- Bulk upsert from temp tables (for COPY-based ingestion)
-- -----------------------------------------------------------------------------

-- Merge atoms from tmp_atom_proj temp table
CREATE OR REPLACE FUNCTION merge_atom_projections(p_update_existing BOOLEAN DEFAULT FALSE)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_count INTEGER;
BEGIN
    INSERT INTO atom (id, label, geom, hilbert_lo, hilbert_hi)
    SELECT 
        id,
        label,
        ST_GeomFromEWKB(decode(geom_ewkb, 'hex')),
        hilbert_lo,
        hilbert_hi
    FROM tmp_atom_proj
    ON CONFLICT (id) DO UPDATE SET
        geom = EXCLUDED.geom,
        hilbert_lo = EXCLUDED.hilbert_lo,
        hilbert_hi = EXCLUDED.hilbert_hi
    WHERE atom.geom IS NULL OR p_update_existing;
    
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$;

-- Merge compositions from tmp_comp_proj temp table
CREATE OR REPLACE FUNCTION merge_composition_projections(p_update_existing BOOLEAN DEFAULT FALSE)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_count INTEGER;
BEGIN
    INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi)
    SELECT 
        id,
        label,
        1,  -- depth for single tokens
        1,  -- child_count
        1,  -- atom_count
        ST_GeomFromEWKB(decode(centroid_ewkb, 'hex')),
        hilbert_lo,
        hilbert_hi
    FROM tmp_comp_proj
    ON CONFLICT (id) DO UPDATE SET
        centroid = EXCLUDED.centroid,
        hilbert_lo = EXCLUDED.hilbert_lo,
        hilbert_hi = EXCLUDED.hilbert_hi
    WHERE composition.centroid IS NULL OR p_update_existing;
    
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$;

-- -----------------------------------------------------------------------------
-- Hilbert-based range query helpers
-- -----------------------------------------------------------------------------

-- Find atoms within Hilbert distance of a point
-- FIXED: Uses proper 128-bit composite (hilbert_hi, hilbert_lo) ordering
CREATE OR REPLACE FUNCTION atoms_near_hilbert(
    p_hilbert_lo BIGINT,
    p_hilbert_hi BIGINT,
    p_range_lo BIGINT,
    p_range_hi BIGINT DEFAULT 1,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE(
    id BYTEA,
    label TEXT,
    geom GEOMETRY,
    hilbert_distance NUMERIC
)
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
    SELECT 
        a.id,
        a.label,
        a.geom,
        -- Proper 128-bit distance: |((hi1 - hi2) << 64) + (lo1 - lo2)|
        ABS(
            (a.hilbert_hi::NUMERIC - p_hilbert_hi::NUMERIC) * 18446744073709551616::NUMERIC +
            (a.hilbert_lo::NUMERIC - p_hilbert_lo::NUMERIC)
        ) AS hilbert_distance
    FROM atom a
    WHERE 
        -- Range check on BOTH hi and lo
        a.hilbert_hi BETWEEN (p_hilbert_hi - p_range_hi) AND (p_hilbert_hi + p_range_hi)
        AND a.hilbert_lo BETWEEN (p_hilbert_lo - p_range_lo) AND (p_hilbert_lo + p_range_lo)
    ORDER BY (a.hilbert_hi, a.hilbert_lo)  -- Composite ordering for correct 128-bit locality
    LIMIT p_limit;
$$;

-- Find compositions within Hilbert distance
-- FIXED: Uses proper 128-bit composite (hilbert_hi, hilbert_lo) ordering
CREATE OR REPLACE FUNCTION compositions_near_hilbert(
    p_hilbert_lo BIGINT,
    p_hilbert_hi BIGINT,
    p_range_lo BIGINT,
    p_range_hi BIGINT DEFAULT 1,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE(
    id BYTEA,
    label TEXT,
    centroid GEOMETRY,
    hilbert_distance NUMERIC
)
LANGUAGE sql
STABLE
PARALLEL SAFE
AS $$
    SELECT 
        c.id,
        c.label,
        c.centroid,
        ABS(
            (c.hilbert_hi::NUMERIC - p_hilbert_hi::NUMERIC) * 18446744073709551616::NUMERIC +
            (c.hilbert_lo::NUMERIC - p_hilbert_lo::NUMERIC)
        ) AS hilbert_distance
    FROM composition c
    WHERE 
        c.hilbert_hi BETWEEN (p_hilbert_hi - p_range_hi) AND (p_hilbert_hi + p_range_hi)
        AND c.hilbert_lo BETWEEN (p_hilbert_lo - p_range_lo) AND (p_hilbert_lo + p_range_lo)
    ORDER BY (c.hilbert_hi, c.hilbert_lo)  -- Composite ordering
    LIMIT p_limit;
$$;

-- -----------------------------------------------------------------------------
-- Statistics and validation
-- -----------------------------------------------------------------------------

-- Get coordinate statistics for atoms
CREATE OR REPLACE FUNCTION atom_coord_stats()
RETURNS TABLE(
    total_count BIGINT,
    with_geom_count BIGINT,
    min_x DOUBLE PRECISION,
    max_x DOUBLE PRECISION,
    min_y DOUBLE PRECISION,
    max_y DOUBLE PRECISION,
    min_z DOUBLE PRECISION,
    max_z DOUBLE PRECISION,
    min_m DOUBLE PRECISION,
    max_m DOUBLE PRECISION
)
LANGUAGE sql
STABLE
AS $$
    SELECT 
        COUNT(*),
        COUNT(geom),
        MIN(ST_X(geom)),
        MAX(ST_X(geom)),
        MIN(ST_Y(geom)),
        MAX(ST_Y(geom)),
        MIN(ST_Z(geom)),
        MAX(ST_Z(geom)),
        MIN(ST_M(geom)),
        MAX(ST_M(geom))
    FROM atom;
$$;

-- Get coordinate statistics for compositions
CREATE OR REPLACE FUNCTION composition_coord_stats()
RETURNS TABLE(
    total_count BIGINT,
    with_centroid_count BIGINT,
    min_x DOUBLE PRECISION,
    max_x DOUBLE PRECISION,
    min_y DOUBLE PRECISION,
    max_y DOUBLE PRECISION,
    min_z DOUBLE PRECISION,
    max_z DOUBLE PRECISION,
    min_m DOUBLE PRECISION,
    max_m DOUBLE PRECISION
)
LANGUAGE sql
STABLE
AS $$
    SELECT 
        COUNT(*),
        COUNT(centroid),
        MIN(ST_X(centroid)),
        MAX(ST_X(centroid)),
        MIN(ST_Y(centroid)),
        MAX(ST_Y(centroid)),
        MIN(ST_Z(centroid)),
        MAX(ST_Z(centroid)),
        MIN(ST_M(centroid)),
        MAX(ST_M(centroid))
    FROM composition;
$$;

-- Validate Hilbert indices match coordinates
CREATE OR REPLACE FUNCTION validate_hilbert_indices(p_sample_size INTEGER DEFAULT 100)
RETURNS TABLE(
    entity_type TEXT,
    id BYTEA,
    label TEXT,
    stored_hilbert_lo BIGINT,
    computed_hilbert_lo BIGINT,
    match BOOLEAN
)
LANGUAGE sql
STABLE
AS $$
    -- This requires the hypercube_coords_to_hilbert function from the extension
    SELECT 
        'atom'::TEXT,
        a.id,
        a.label,
        a.hilbert_lo,
        (hypercube_coords_to_hilbert(
            ST_X(a.geom)::INT,
            ST_Y(a.geom)::INT,
            ST_Z(a.geom)::INT,
            ST_M(a.geom)::INT
        )).lo,
        a.hilbert_lo = (hypercube_coords_to_hilbert(
            ST_X(a.geom)::INT,
            ST_Y(a.geom)::INT,
            ST_Z(a.geom)::INT,
            ST_M(a.geom)::INT
        )).lo
    FROM atom a
    WHERE a.geom IS NOT NULL
    LIMIT p_sample_size;
$$;

COMMENT ON FUNCTION upsert_atom_4d IS 'Insert or update atom with 4D POINTZM geometry and Hilbert index';
COMMENT ON FUNCTION upsert_composition_4d IS 'Insert or update composition with 4D centroid and Hilbert index';
COMMENT ON FUNCTION merge_atom_projections IS 'Bulk merge atoms from tmp_atom_proj temp table';
COMMENT ON FUNCTION merge_composition_projections IS 'Bulk merge compositions from tmp_comp_proj temp table';
