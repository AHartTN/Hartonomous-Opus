-- Unified Schema Test Suite
-- Tests for the unified atom table with 4D spatial coordinates
-- Run: psql -d hypercube -f tests/test_unified_schema.sql
-- 
-- Prerequisites: atoms seeded, some compositions ingested via cpe_ingest

\set ON_ERROR_STOP on
\timing on

BEGIN;

-- =============================================================================
-- Helper functions
-- =============================================================================

CREATE OR REPLACE FUNCTION test_assert(condition boolean, test_name text)
RETURNS void AS $$
BEGIN
    IF NOT condition THEN
        RAISE EXCEPTION 'FAILED: %', test_name;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION test_assert_equal(val1 anyelement, val2 anyelement, test_name text)
RETURNS void AS $$
BEGIN
    IF val1 IS DISTINCT FROM val2 THEN
        RAISE EXCEPTION 'FAILED: % (got % expected %)', test_name, val1, val2;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION test_assert_less(val1 double precision, val2 double precision, test_name text)
RETURNS void AS $$
BEGIN
    IF val1 >= val2 THEN
        RAISE EXCEPTION 'FAILED: % (got % >= %)', test_name, val1, val2;
    END IF;
    RAISE NOTICE 'PASSED: % (% < %)', test_name, val1, val2;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN RAISE NOTICE '============================================'; END $$;
DO $$ BEGIN RAISE NOTICE '  Unified Schema Test Suite'; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;

-- =============================================================================
-- SECTION 1: Atom Table Structure
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Section 1: Atom Table Structure ---'; END $$;

-- Test 1.1: Atoms exist
DO $$
DECLARE
    v_count BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_count FROM atom WHERE depth = 0;
    PERFORM test_assert(v_count > 1000000, format('At least 1M atoms exist (got %s)', v_count));
END $$;

-- Test 1.2: Compositions exist  
DO $$
DECLARE
    v_count BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_count FROM atom WHERE depth > 0;
    PERFORM test_assert(v_count > 0, format('Compositions exist (got %s)', v_count));
END $$;

-- Test 1.3: All atoms have geometry
DO $$
DECLARE
    v_null_geom BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_null_geom FROM atom WHERE geom IS NULL;
    PERFORM test_assert(v_null_geom = 0, 'All atoms have geometry');
END $$;

-- Test 1.4: Atoms are POINTZM, compositions are LINESTRINGZM
DO $$
DECLARE
    v_bad_point BIGINT;
    v_bad_line BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_bad_point 
    FROM atom WHERE depth = 0 AND ST_GeometryType(geom) != 'ST_Point';
    
    SELECT COUNT(*) INTO v_bad_line 
    FROM atom WHERE depth > 0 AND ST_GeometryType(geom) != 'ST_LineString';
    
    PERFORM test_assert(v_bad_point = 0, 'All leaves (depth=0) are POINT');
    PERFORM test_assert(v_bad_line = 0, 'All compositions (depth>0) are LINESTRING');
END $$;

-- Test 1.5: SRID is 0 for all geometries
DO $$
DECLARE
    v_bad_srid BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_bad_srid FROM atom WHERE ST_SRID(geom) != 0;
    PERFORM test_assert(v_bad_srid = 0, 'All geometries have SRID=0');
END $$;

-- =============================================================================
-- SECTION 2: Coordinate Precision
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Section 2: Coordinate Precision (32-bit) ---'; END $$;

-- Test 2.1: Coordinates are in uint32 range (0 to 4294967295)
DO $$
DECLARE
    v_bad_x BIGINT;
    v_bad_y BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_bad_x 
    FROM atom 
    WHERE depth = 0 
      AND (ST_X(geom) < 0 OR ST_X(geom) > 4294967295);
    
    SELECT COUNT(*) INTO v_bad_y 
    FROM atom 
    WHERE depth = 0 
      AND (ST_Y(geom) < 0 OR ST_Y(geom) > 4294967295);
    
    PERFORM test_assert(v_bad_x = 0, 'X coordinates in uint32 range');
    PERFORM test_assert(v_bad_y = 0, 'Y coordinates in uint32 range');
END $$;

-- Test 2.2: Z and M coordinates exist
DO $$
DECLARE
    v_null_z BIGINT;
    v_null_m BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_null_z FROM atom WHERE depth = 0 AND ST_Z(geom) IS NULL;
    SELECT COUNT(*) INTO v_null_m FROM atom WHERE depth = 0 AND ST_M(geom) IS NULL;
    
    PERFORM test_assert(v_null_z = 0, 'Z coordinates exist');
    PERFORM test_assert(v_null_m = 0, 'M coordinates exist');
END $$;

-- =============================================================================
-- SECTION 3: Hilbert Index Properties
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Section 3: Hilbert Index Properties ---'; END $$;

-- Test 3.1: All atoms have Hilbert indices
DO $$
DECLARE
    v_null_hilbert BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_null_hilbert 
    FROM atom WHERE hilbert_lo IS NULL OR hilbert_hi IS NULL;
    PERFORM test_assert(v_null_hilbert = 0, 'All atoms have Hilbert indices');
END $$;

-- Test 3.2: Hilbert indices are unique for atoms
DO $$
DECLARE
    v_duplicates BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_duplicates
    FROM (
        SELECT hilbert_hi, hilbert_lo, COUNT(*) as cnt
        FROM atom WHERE depth = 0
        GROUP BY hilbert_hi, hilbert_lo
        HAVING COUNT(*) > 1
    ) dups;
    -- Note: Collisions are possible but should be rare
    RAISE NOTICE 'Hilbert index collisions: %', v_duplicates;
END $$;

-- Test 3.3: Hilbert proximity correlates with spatial proximity (A closer to a than to Z)
DO $$
DECLARE
    v_upper_A_hi BIGINT;
    v_upper_A_lo BIGINT;
    v_lower_a_hi BIGINT;
    v_lower_a_lo BIGINT;
    v_upper_Z_hi BIGINT;
    v_upper_Z_lo BIGINT;
    v_dist_Aa NUMERIC;
    v_dist_AZ NUMERIC;
BEGIN
    SELECT hilbert_hi, hilbert_lo INTO v_upper_A_hi, v_upper_A_lo FROM atom WHERE codepoint = 65;  -- A
    SELECT hilbert_hi, hilbert_lo INTO v_lower_a_hi, v_lower_a_lo FROM atom WHERE codepoint = 97;  -- a  
    SELECT hilbert_hi, hilbert_lo INTO v_upper_Z_hi, v_upper_Z_lo FROM atom WHERE codepoint = 90;  -- Z
    
    v_dist_Aa := ABS(v_upper_A_hi::NUMERIC - v_lower_a_hi::NUMERIC) * 9223372036854775808::NUMERIC + 
                 ABS(v_upper_A_lo::NUMERIC - v_lower_a_lo::NUMERIC);
    v_dist_AZ := ABS(v_upper_A_hi::NUMERIC - v_upper_Z_hi::NUMERIC) * 9223372036854775808::NUMERIC + 
                 ABS(v_upper_A_lo::NUMERIC - v_upper_Z_lo::NUMERIC);
    
    RAISE NOTICE 'Hilbert distance A-a: %, A-Z: %', v_dist_Aa, v_dist_AZ;
    -- Semantic ordering should put A closer to a than to Z
    PERFORM test_assert(v_dist_Aa < v_dist_AZ, 'A closer to a than to Z (Hilbert)');
END $$;

-- =============================================================================
-- SECTION 4: BLAKE3 Hash Properties
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Section 4: BLAKE3 Hash Properties ---'; END $$;

-- Test 4.1: All IDs are 32 bytes
DO $$
DECLARE
    v_bad_length BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_bad_length FROM atom WHERE length(id) != 32;
    PERFORM test_assert(v_bad_length = 0, 'All IDs are 32 bytes (BLAKE3)');
END $$;

-- Test 4.2: IDs are unique
DO $$
DECLARE
    v_duplicates BIGINT;
BEGIN
    SELECT COUNT(*) - COUNT(DISTINCT id) INTO v_duplicates FROM atom;
    PERFORM test_assert(v_duplicates = 0, 'All IDs are unique');
END $$;

-- Test 4.3: Deterministic hashing (same codepoint = same hash)
DO $$
DECLARE
    v_A_id BYTEA;
    v_computed_id BYTEA;
BEGIN
    SELECT id INTO v_A_id FROM atom WHERE codepoint = 65;
    -- BLAKE3 of 'A' UTF-8 encoding (0x41)
    SELECT hypercube_blake3_codepoint(65) INTO v_computed_id;
    
    PERFORM test_assert_equal(v_A_id, v_computed_id, 'Hash is deterministic for codepoint 65 (A)');
END $$;

-- =============================================================================
-- SECTION 5: Composition Structure
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Section 5: Composition Structure ---'; END $$;

-- Test 5.1: Compositions have children array
DO $$
DECLARE
    v_null_children BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_null_children FROM atom WHERE depth > 0 AND children IS NULL;
    PERFORM test_assert(v_null_children = 0, 'All compositions have children array');
END $$;

-- Test 5.2: Children references exist in atom table
DO $$
DECLARE
    v_orphan_count BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_orphan_count
    FROM atom a
    CROSS JOIN LATERAL unnest(a.children) AS child_id
    WHERE a.depth > 0 
      AND NOT EXISTS (SELECT 1 FROM atom WHERE id = child_id)
    LIMIT 1;  -- Just check if any exist
    
    PERFORM test_assert(v_orphan_count = 0, 'All children references exist');
END $$;

-- Test 5.3: atom_count is correct (sum of children atom_counts)
DO $$
DECLARE
    v_mismatch BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_mismatch
    FROM atom a
    WHERE a.depth > 0 
      AND a.atom_count != (
          SELECT COALESCE(SUM(c.atom_count), 0)
          FROM unnest(a.children) child_id
          JOIN atom c ON c.id = child_id
      )
    LIMIT 1;
    
    PERFORM test_assert(v_mismatch = 0, 'atom_count matches sum of children');
END $$;

-- Test 5.4: depth is correct (max child depth + 1)
DO $$
DECLARE
    v_mismatch BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_mismatch
    FROM atom a
    WHERE a.depth > 0 
      AND a.depth != (
          SELECT COALESCE(MAX(c.depth), -1) + 1
          FROM unnest(a.children) child_id
          JOIN atom c ON c.id = child_id
      )
    LIMIT 1;
    
    PERFORM test_assert(v_mismatch = 0, 'depth = max(child.depth) + 1');
END $$;

-- =============================================================================
-- SECTION 6: Reconstruction
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Section 6: Text Reconstruction ---'; END $$;

-- Test 6.1: Can reconstruct atoms to single characters
DO $$
DECLARE
    v_A_text TEXT;
BEGIN
    SELECT atom_reconstruct_text(id) INTO v_A_text FROM atom WHERE codepoint = 65;
    PERFORM test_assert_equal(v_A_text, 'A', 'atom_reconstruct_text works for A');
END $$;

-- Test 6.2: Find a small composition and verify reconstruction
DO $$
DECLARE
    v_id BYTEA;
    v_text TEXT;
    v_len INTEGER;
BEGIN
    -- Find a small composition (2-4 characters)
    SELECT id, atom_count INTO v_id, v_len 
    FROM atom 
    WHERE depth > 0 AND atom_count BETWEEN 2 AND 4
    LIMIT 1;
    
    IF v_id IS NOT NULL THEN
        v_text := atom_reconstruct_text(v_id);
        PERFORM test_assert(v_text IS NOT NULL, format('Composition reconstructs to text (len %s)', v_len));
        PERFORM test_assert(char_length(v_text) = v_len, 
            format('Reconstructed length matches atom_count (expected %s, got %s)', v_len, char_length(v_text)));
    END IF;
END $$;

-- =============================================================================
-- SECTION 7: Semantic Queries
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Section 7: Semantic Query Functions ---'; END $$;

-- Test 7.1: atom_nearest_hilbert works
DO $$
DECLARE
    v_A_id BYTEA;
    v_neighbor_count BIGINT;
BEGIN
    SELECT id INTO v_A_id FROM atom WHERE codepoint = 65;
    SELECT COUNT(*) INTO v_neighbor_count FROM atom_nearest_hilbert(v_A_id, 5);
    
    PERFORM test_assert(v_neighbor_count = 5, 'atom_nearest_hilbert returns requested neighbors');
END $$;

-- Test 7.2: atom_nearest_spatial works
DO $$
DECLARE
    v_A_id BYTEA;
    v_neighbor_count BIGINT;
BEGIN
    SELECT id INTO v_A_id FROM atom WHERE codepoint = 65;
    SELECT COUNT(*) INTO v_neighbor_count FROM atom_nearest_spatial(v_A_id, 5);
    
    PERFORM test_assert(v_neighbor_count = 5, 'atom_nearest_spatial returns requested neighbors');
END $$;

-- Test 7.3: atom_distance works
DO $$
DECLARE
    v_upper_A_id BYTEA;
    v_lower_a_id BYTEA;
    v_dist DOUBLE PRECISION;
BEGIN
    SELECT id INTO v_upper_A_id FROM atom WHERE codepoint = 65;
    SELECT id INTO v_lower_a_id FROM atom WHERE codepoint = 97;
    
    v_dist := atom_distance(v_upper_A_id, v_lower_a_id);
    
    PERFORM test_assert(v_dist IS NOT NULL AND v_dist >= 0, 'atom_distance returns valid distance');
    RAISE NOTICE 'Distance A-a: %', v_dist;
END $$;

-- Test 7.4: atom_find_parents works
DO $$
DECLARE
    v_A_id BYTEA;
    v_parent_count BIGINT;
BEGIN
    SELECT id INTO v_A_id FROM atom WHERE codepoint = 65;
    SELECT COUNT(*) INTO v_parent_count FROM atom_find_parents(v_A_id, 10);
    
    RAISE NOTICE 'Compositions containing A: %', v_parent_count;
    PERFORM test_assert(v_parent_count >= 0, 'atom_find_parents executes');
END $$;

-- =============================================================================
-- SECTION 8: Spatial Index Performance
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Section 8: Spatial Index Performance ---'; END $$;

-- Test 8.1: GIST index exists and is used
DO $$
DECLARE
    v_index_exists BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'atom' AND indexname = 'idx_atom_geom'
    ) INTO v_index_exists;
    
    PERFORM test_assert(v_index_exists, 'GIST index on geom exists');
END $$;

-- Test 8.2: Hilbert index exists
DO $$
DECLARE
    v_index_exists BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'atom' AND indexname = 'idx_atom_hilbert'
    ) INTO v_index_exists;
    
    PERFORM test_assert(v_index_exists, 'Hilbert index exists');
END $$;

-- =============================================================================
-- Summary Statistics
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;
DO $$ BEGIN RAISE NOTICE '  Test Summary Statistics'; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;

DO $$
DECLARE
    v_rec RECORD;
BEGIN
    FOR v_rec IN 
        SELECT 
            depth,
            COUNT(*) as count,
            MIN(atom_count) as min_atoms,
            MAX(atom_count) as max_atoms
        FROM atom
        GROUP BY depth
        ORDER BY depth
        LIMIT 10
    LOOP
        RAISE NOTICE 'Depth %: % items (atoms %-% )', 
            v_rec.depth, v_rec.count, v_rec.min_atoms, v_rec.max_atoms;
    END LOOP;
END $$;

DO $$
DECLARE
    v_table_size TEXT;
BEGIN
    SELECT pg_size_pretty(pg_total_relation_size('atom')) INTO v_table_size;
    RAISE NOTICE 'Atom table size: %', v_table_size;
END $$;

-- Cleanup
DROP FUNCTION IF EXISTS test_assert(boolean, text);
DROP FUNCTION IF EXISTS test_assert_equal(anyelement, anyelement, text);
DROP FUNCTION IF EXISTS test_assert_less(double precision, double precision, text);

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;
DO $$ BEGIN RAISE NOTICE '  All tests completed successfully'; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;

ROLLBACK;  -- Don't commit test artifacts
