-- Lossless Schema Migration for Hypercube
-- 
-- This migration:
-- 1. Ensures raw 32-bit integer coordinates exist on atom and relation tables
-- 2. Creates helper functions for uint32/int32 conversion
-- 3. For NEW databases: relation table has no PostGIS coords column
-- 4. For EXISTING databases: migrates PostGIS coords to integer, then drops coords column
--
-- Integer storage is lossless - PostgreSQL INTEGER is 32-bit signed,
-- but the bit pattern is identical to uint32 so no precision is lost.

BEGIN;

-- =============================================================================
-- STEP 1: Ensure raw coordinate columns exist on atom table
-- =============================================================================

ALTER TABLE atom ADD COLUMN IF NOT EXISTS coord_x INTEGER;
ALTER TABLE atom ADD COLUMN IF NOT EXISTS coord_y INTEGER;
ALTER TABLE atom ADD COLUMN IF NOT EXISTS coord_z INTEGER;
ALTER TABLE atom ADD COLUMN IF NOT EXISTS coord_m INTEGER;

-- =============================================================================
-- STEP 2: Migrate existing atom coordinates from PostGIS (if needed)
-- =============================================================================

-- Only run if atoms exist with PostGIS coords but no integer coords
UPDATE atom SET
    coord_x = ((ST_X(coords) * 4294967295)::bigint & x'FFFFFFFF'::bigint)::integer,
    coord_y = ((ST_Y(coords) * 4294967295)::bigint & x'FFFFFFFF'::bigint)::integer,
    coord_z = ((ST_Z(coords) * 4294967295)::bigint & x'FFFFFFFF'::bigint)::integer,
    coord_m = ((ST_M(coords) * 4294967295)::bigint & x'FFFFFFFF'::bigint)::integer
WHERE coord_x IS NULL AND coords IS NOT NULL;

-- =============================================================================
-- STEP 3: Create helper functions for coordinate conversion
-- =============================================================================

-- Convert signed INTEGER to unsigned BIGINT (for display/computation)
CREATE OR REPLACE FUNCTION int32_to_uint32(val INTEGER) RETURNS BIGINT AS $$
BEGIN
    IF val >= 0 THEN
        RETURN val::BIGINT;
    ELSE
        RETURN (val::BIGINT + 4294967296);  -- 2^32
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Convert unsigned BIGINT to signed INTEGER (for storage)
CREATE OR REPLACE FUNCTION uint32_to_int32(val BIGINT) RETURNS INTEGER AS $$
BEGIN
    IF val < 2147483648 THEN  -- < 2^31
        RETURN val::INTEGER;
    ELSE
        RETURN (val - 4294967296)::INTEGER;  -- Wrap to negative
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Compute Hilbert index from raw integer coordinates
-- This matches the C++ implementation exactly
CREATE OR REPLACE FUNCTION compute_hilbert_from_coords(
    p_x INTEGER, p_y INTEGER, p_z INTEGER, p_m INTEGER
) RETURNS TABLE(hilbert_lo BIGINT, hilbert_hi BIGINT) AS $$
BEGIN
    -- Convert to unsigned for Hilbert computation
    -- Then call the C++ extension function
    RETURN QUERY
    SELECT h.hilbert_lo, h.hilbert_hi
    FROM hypercube_coords_to_hilbert(
        int32_to_uint32(p_x),
        int32_to_uint32(p_y),
        int32_to_uint32(p_z),
        int32_to_uint32(p_m)
    ) h;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- STEP 4: Create composition centroid function that preserves precision
-- =============================================================================

-- Compute centroid from child coordinates without PostGIS
-- Uses proper integer arithmetic with overflow protection
CREATE OR REPLACE FUNCTION compute_centroid_lossless(
    p_child_coords INTEGER[][]  -- Array of [x,y,z,m] arrays
) RETURNS INTEGER[] AS $$
DECLARE
    sum_x BIGINT := 0;
    sum_y BIGINT := 0;
    sum_z BIGINT := 0;
    sum_m BIGINT := 0;
    n INTEGER;
    child INTEGER[];
BEGIN
    n := array_length(p_child_coords, 1);
    IF n IS NULL OR n = 0 THEN
        RETURN ARRAY[0, 0, 0, 0];
    END IF;
    
    -- Sum as unsigned (convert from two's complement)
    FOREACH child SLICE 1 IN ARRAY p_child_coords LOOP
        sum_x := sum_x + int32_to_uint32(child[1]);
        sum_y := sum_y + int32_to_uint32(child[2]);
        sum_z := sum_z + int32_to_uint32(child[3]);
        sum_m := sum_m + int32_to_uint32(child[4]);
    END LOOP;
    
    -- Average and convert back to signed integer storage
    RETURN ARRAY[
        uint32_to_int32(sum_x / n),
        uint32_to_int32(sum_y / n),
        uint32_to_int32(sum_z / n),
        uint32_to_int32(sum_m / n)
    ];
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- STEP 5: Ensure relation table has integer coordinates (no PostGIS needed)
-- =============================================================================

-- For existing databases with old schema that had coords column:
-- First migrate data, then we can drop the column
DO $$
BEGIN
    -- Check if coords column exists on relation table
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'relation' AND column_name = 'coords'
    ) THEN
        -- Add integer columns if they don't exist
        ALTER TABLE relation ADD COLUMN IF NOT EXISTS coord_x INTEGER;
        ALTER TABLE relation ADD COLUMN IF NOT EXISTS coord_y INTEGER;
        ALTER TABLE relation ADD COLUMN IF NOT EXISTS coord_z INTEGER;
        ALTER TABLE relation ADD COLUMN IF NOT EXISTS coord_m INTEGER;
        
        -- Migrate from PostGIS to integer
        UPDATE relation SET
            coord_x = ((ST_X(coords) * 4294967295)::bigint & x'FFFFFFFF'::bigint)::integer,
            coord_y = ((ST_Y(coords) * 4294967295)::bigint & x'FFFFFFFF'::bigint)::integer,
            coord_z = ((ST_Z(coords) * 4294967295)::bigint & x'FFFFFFFF'::bigint)::integer,
            coord_m = ((ST_M(coords) * 4294967295)::bigint & x'FFFFFFFF'::bigint)::integer
        WHERE coord_x IS NULL AND coords IS NOT NULL;
        
        -- Now drop the PostGIS coords column (no longer needed)
        ALTER TABLE relation DROP COLUMN IF EXISTS coords;
        
        -- Drop the old PostGIS index if it exists
        DROP INDEX IF EXISTS idx_relation_coords;
    END IF;
END $$;

-- =============================================================================
-- STEP 6: Create indexes on integer coordinates
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_atom_coord_x ON atom(coord_x);
CREATE INDEX IF NOT EXISTS idx_atom_coord_y ON atom(coord_y);
CREATE INDEX IF NOT EXISTS idx_atom_coord_z ON atom(coord_z);
CREATE INDEX IF NOT EXISTS idx_atom_coord_m ON atom(coord_m);

-- Composite index for coordinate queries
CREATE INDEX IF NOT EXISTS idx_atom_coords_int ON atom(coord_x, coord_y, coord_z, coord_m);

-- =============================================================================
-- STEP 7: Updated CPE functions using lossless coordinates
-- =============================================================================

-- Lossless pair composition
CREATE OR REPLACE FUNCTION cpe_get_pair_lossless(
    p_left_id BYTEA,
    p_left_x INTEGER, p_left_y INTEGER, p_left_z INTEGER, p_left_m INTEGER,
    p_left_depth INTEGER, p_left_atoms BIGINT, p_left_is_atom BOOLEAN,
    p_right_id BYTEA,
    p_right_x INTEGER, p_right_y INTEGER, p_right_z INTEGER, p_right_m INTEGER,
    p_right_depth INTEGER, p_right_atoms BIGINT, p_right_is_atom BOOLEAN
) RETURNS TABLE(
    id BYTEA,
    cx INTEGER, cy INTEGER, cz INTEGER, cm INTEGER,
    h_lo BIGINT, h_hi BIGINT,
    depth INTEGER,
    atom_count BIGINT
) AS $$
DECLARE
    v_id BYTEA;
    v_concat_data BYTEA;
    v_centroid INTEGER[];
    v_child_coords INTEGER[][];
BEGIN
    -- Build deterministic hash input: ordinal(4 bytes) + hash(32 bytes) for each child
    v_concat_data := int4send(0) || p_left_id || int4send(1) || p_right_id;
    v_id := hypercube_blake3(v_concat_data);
    
    -- Check if already exists (content-addressed deduplication)
    IF EXISTS (SELECT 1 FROM relation WHERE relation.id = v_id) THEN
        RETURN QUERY
        SELECT r.id, r.coord_x, r.coord_y, r.coord_z, r.coord_m,
               r.hilbert_lo, r.hilbert_hi, r.depth, r.atom_count
        FROM relation r WHERE r.id = v_id;
        RETURN;
    END IF;
    
    -- Compute centroid using integer arithmetic
    v_child_coords := ARRAY[
        ARRAY[p_left_x, p_left_y, p_left_z, p_left_m],
        ARRAY[p_right_x, p_right_y, p_right_z, p_right_m]
    ];
    v_centroid := compute_centroid_lossless(v_child_coords);
    
    -- Return result (caller inserts)
    id := v_id;
    cx := v_centroid[1];
    cy := v_centroid[2];
    cz := v_centroid[3];
    cm := v_centroid[4];
    
    -- Compute Hilbert index
    SELECT h.hilbert_lo, h.hilbert_hi INTO h_lo, h_hi
    FROM compute_hilbert_from_coords(cx, cy, cz, cm) h;
    
    depth := GREATEST(p_left_depth, p_right_depth) + 1;
    atom_count := p_left_atoms + p_right_atoms;
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Get atom info with raw coordinates
CREATE OR REPLACE FUNCTION cpe_get_atom_info(p_codepoint INTEGER)
RETURNS TABLE(
    id BYTEA,
    x INTEGER, y INTEGER, z INTEGER, m INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT a.id::BYTEA, a.coord_x, a.coord_y, a.coord_z, a.coord_m
    FROM atom a
    WHERE a.codepoint = p_codepoint;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- STEP 8: Updated insert function using raw coordinates
-- =============================================================================

CREATE OR REPLACE FUNCTION hypercube_insert_composition_lossless(
    p_id BYTEA,
    p_coord_x INTEGER, p_coord_y INTEGER, p_coord_z INTEGER, p_coord_m INTEGER,
    p_hilbert_lo BIGINT, p_hilbert_hi BIGINT,
    p_depth INTEGER,
    p_child_count INTEGER,
    p_atom_count BIGINT,
    p_child_ids BYTEA[],
    p_child_is_atom BOOLEAN[]
) RETURNS BYTEA AS $$
DECLARE
    i INTEGER;
BEGIN
    -- Check if already exists (content-addressed deduplication)
    IF EXISTS (SELECT 1 FROM relation WHERE id = p_id) THEN
        RETURN p_id;
    END IF;
    
    -- Insert relation with raw integer coords only (no PostGIS - lossless)
    INSERT INTO relation (
        id, coord_x, coord_y, coord_z, coord_m,
        hilbert_lo, hilbert_hi, depth, child_count, atom_count
    ) VALUES (
        p_id, p_coord_x, p_coord_y, p_coord_z, p_coord_m,
        p_hilbert_lo, p_hilbert_hi, p_depth, p_child_count, p_atom_count
    );
    
    -- Insert edges
    FOR i IN 1..array_length(p_child_ids, 1) LOOP
        INSERT INTO relation_edge (parent_id, child_id, ordinal, is_atom)
        VALUES (p_id, p_child_ids[i], i - 1, p_child_is_atom[i]);
    END LOOP;
    
    RETURN p_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- STEP 9: Stats view including raw coordinates
-- =============================================================================

CREATE OR REPLACE VIEW atom_coord_stats AS
SELECT 
    category,
    COUNT(*) as count,
    MIN(int32_to_uint32(coord_x)) as min_x,
    MAX(int32_to_uint32(coord_x)) as max_x,
    MIN(int32_to_uint32(coord_y)) as min_y,
    MAX(int32_to_uint32(coord_y)) as max_y,
    MIN(int32_to_uint32(coord_z)) as min_z,
    MAX(int32_to_uint32(coord_z)) as max_z,
    MIN(int32_to_uint32(coord_m)) as min_m,
    MAX(int32_to_uint32(coord_m)) as max_m
FROM atom
GROUP BY category;

COMMIT;
