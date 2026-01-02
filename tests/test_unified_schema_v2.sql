-- =============================================================================
-- Unified Schema Validation Tests
-- Tests the current schema: atom table with geom/centroid/children
-- =============================================================================
--
-- Run: psql -d hypercube -f test_unified_schema_v2.sql

\set ON_ERROR_STOP on
\timing on

BEGIN;

-- Helper functions for assertions
CREATE OR REPLACE FUNCTION assert_true(condition boolean, test_name text)
RETURNS void AS $$
BEGIN
    IF NOT condition THEN
        RAISE EXCEPTION 'FAILED: %', test_name;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION assert_eq(val1 anyelement, val2 anyelement, test_name text)
RETURNS void AS $$
BEGIN
    IF val1 IS DISTINCT FROM val2 THEN
        RAISE EXCEPTION 'FAILED: % (got %, expected %)', test_name, val1, val2;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION assert_gt(val1 double precision, val2 double precision, test_name text)
RETURNS void AS $$
BEGIN
    IF val1 <= val2 THEN
        RAISE EXCEPTION 'FAILED: % (% <= %)', test_name, val1, val2;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '╔══════════════════════════════════════════════════════════════╗'; END $$;
DO $$ BEGIN RAISE NOTICE '║     Hartonomous Hypercube - Unified Schema Tests             ║'; END $$;
DO $$ BEGIN RAISE NOTICE '╚══════════════════════════════════════════════════════════════╝'; END $$;
DO $$ BEGIN RAISE NOTICE ''; END $$;

-- =============================================================================
-- TEST 1: Schema Structure
-- =============================================================================
DO $$ BEGIN RAISE NOTICE '─── Test 1: Schema Structure ───'; END $$;

DO $$
BEGIN
    -- Check required columns exist
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='atom' AND column_name='id'),
        'Column id exists');
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='atom' AND column_name='geom'),
        'Column geom exists');
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='atom' AND column_name='children'),
        'Column children exists');
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='atom' AND column_name='centroid'),
        'Column centroid exists');
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='atom' AND column_name='hilbert_lo'),
        'Column hilbert_lo exists');
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='atom' AND column_name='depth'),
        'Column depth exists');
END $$;

-- =============================================================================
-- TEST 2: Unicode Atom Coverage
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 2: Unicode Atom Coverage ───'; END $$;

DO $$
DECLARE
    leaf_count bigint;
    ascii_count bigint;
    no_surrogates bigint;
BEGIN
    SELECT COUNT(*) INTO leaf_count FROM atom WHERE depth = 0;
    SELECT COUNT(*) INTO ascii_count FROM atom WHERE codepoint BETWEEN 0 AND 127;
    SELECT COUNT(*) INTO no_surrogates FROM atom WHERE codepoint BETWEEN 55296 AND 57343;
    
    RAISE NOTICE 'Leaf atoms: %, ASCII atoms: %', leaf_count, ascii_count;
    
    PERFORM assert_gt(leaf_count::double precision, 1100000.0, 
        'At least 1.1M Unicode codepoints seeded');
    PERFORM assert_eq(ascii_count::integer, 128, 
        'All 128 ASCII codepoints present');
    PERFORM assert_eq(no_surrogates::integer, 0, 
        'No surrogate codepoints stored');
END $$;

-- =============================================================================
-- TEST 3: Geometry Types
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 3: Geometry Types ───'; END $$;

DO $$
DECLARE
    point_count bigint;
    line_count bigint;
    other_count bigint;
BEGIN
    SELECT COUNT(*) INTO point_count FROM atom WHERE ST_GeometryType(geom) = 'ST_Point';
    SELECT COUNT(*) INTO line_count FROM atom WHERE ST_GeometryType(geom) = 'ST_LineString';
    SELECT COUNT(*) INTO other_count FROM atom WHERE ST_GeometryType(geom) NOT IN ('ST_Point', 'ST_LineString');
    
    RAISE NOTICE 'POINT geometries: %, LINESTRING geometries: %, Other: %', 
        point_count, line_count, other_count;
    
    PERFORM assert_gt(point_count::double precision, 0, 'Has POINT geometries (leaves)');
    PERFORM assert_eq(other_count::integer, 0, 'Only POINT and LINESTRING geometries');
END $$;

-- =============================================================================
-- TEST 4: SRID = 0 for All Atoms
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 4: SRID Validation ───'; END $$;

DO $$
DECLARE
    bad_srid_count bigint;
BEGIN
    SELECT COUNT(*) INTO bad_srid_count FROM atom WHERE ST_SRID(geom) != 0;
    
    PERFORM assert_eq(bad_srid_count::integer, 0, 
        'All atoms have SRID = 0 (raw 4D space)');
END $$;

-- =============================================================================
-- TEST 5: Centroid Population
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 5: Centroid Population ───'; END $$;

DO $$
DECLARE
    null_centroid_count bigint;
    total_count bigint;
BEGIN
    SELECT COUNT(*) INTO total_count FROM atom;
    SELECT COUNT(*) INTO null_centroid_count FROM atom WHERE centroid IS NULL;
    
    RAISE NOTICE 'Total atoms: %, Missing centroids: %', total_count, null_centroid_count;
    
    PERFORM assert_eq(null_centroid_count::integer, 0, 
        'All atoms have computed centroids');
END $$;

-- =============================================================================
-- TEST 6: BLAKE3 Hash Uniqueness
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 6: BLAKE3 Hash Uniqueness ───'; END $$;

DO $$
DECLARE
    total_count bigint;
    unique_count bigint;
BEGIN
    SELECT COUNT(*) INTO total_count FROM atom;
    SELECT COUNT(DISTINCT id) INTO unique_count FROM atom;
    
    PERFORM assert_eq(total_count, unique_count, 
        'All BLAKE3 hashes are unique');
END $$;

-- =============================================================================
-- TEST 7: Hilbert Index Validity
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 7: Hilbert Index Validity ───'; END $$;

DO $$
DECLARE
    null_hilbert bigint;
    distinct_hilbert bigint;
    sample_lo bigint;
    sample_hi bigint;
BEGIN
    SELECT COUNT(*) INTO null_hilbert FROM atom WHERE hilbert_lo IS NULL OR hilbert_hi IS NULL;
    SELECT COUNT(DISTINCT (hilbert_hi, hilbert_lo)) INTO distinct_hilbert FROM atom;
    
    SELECT hilbert_lo, hilbert_hi INTO sample_lo, sample_hi FROM atom WHERE codepoint = 65;
    
    RAISE NOTICE 'Sample (A): hilbert_lo=%, hilbert_hi=%', sample_lo, sample_hi;
    RAISE NOTICE 'Distinct Hilbert indices: %', distinct_hilbert;
    
    PERFORM assert_eq(null_hilbert::integer, 0, 'No NULL Hilbert indices');
END $$;

-- =============================================================================
-- TEST 8: Composition Structure
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 8: Composition Structure ───'; END $$;

DO $$
DECLARE
    comp_count bigint;
    bad_comp_count bigint;
    max_depth integer;
BEGIN
    SELECT COUNT(*) INTO comp_count FROM atom WHERE depth > 0;
    SELECT COUNT(*) INTO bad_comp_count FROM atom WHERE depth > 0 AND children IS NULL;
    SELECT MAX(depth) INTO max_depth FROM atom;
    
    RAISE NOTICE 'Compositions: %, Max depth: %', comp_count, max_depth;
    
    IF comp_count > 0 THEN
        PERFORM assert_eq(bad_comp_count::integer, 0, 
            'All compositions have children array');
        PERFORM assert_gt(max_depth::double precision, 0, 
            'Max depth > 0 for compositions');
    ELSE
        RAISE NOTICE 'SKIP: No compositions to test';
    END IF;
END $$;

-- =============================================================================
-- TEST 9: Core SQL Functions
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 9: Core SQL Functions ───'; END $$;

DO $$
DECLARE
    v_id BYTEA;
    v_is_leaf boolean;
    v_centroid record;
    v_dist double precision;
    v_text text;
BEGIN
    -- Get test atom
    SELECT id INTO v_id FROM atom WHERE codepoint = 65;
    
    -- Test atom_is_leaf
    SELECT atom_is_leaf(v_id) INTO v_is_leaf;
    PERFORM assert_eq(v_is_leaf, true, 'atom_is_leaf returns true for codepoint');
    
    -- Test atom_centroid
    SELECT * INTO v_centroid FROM atom_centroid(v_id);
    PERFORM assert_true(v_centroid.x IS NOT NULL, 'atom_centroid returns x');
    PERFORM assert_true(v_centroid.y IS NOT NULL, 'atom_centroid returns y');
    PERFORM assert_true(v_centroid.z IS NOT NULL, 'atom_centroid returns z');
    PERFORM assert_true(v_centroid.m IS NOT NULL, 'atom_centroid returns m');
    
    -- Test atom_distance
    SELECT atom_distance(
        (SELECT id FROM atom WHERE codepoint = 65),
        (SELECT id FROM atom WHERE codepoint = 66)
    ) INTO v_dist;
    PERFORM assert_gt(v_dist, 0, 'atom_distance between different atoms > 0');
    
    -- Test atom_reconstruct_text
    SELECT atom_reconstruct_text(v_id) INTO v_text;
    PERFORM assert_eq(v_text, 'A', 'atom_reconstruct_text returns character');
END $$;

-- =============================================================================
-- TEST 10: Spatial Index Usage
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 10: Spatial Index Usage ───'; END $$;

DO $$
DECLARE
    neighbor_count bigint;
    query_start timestamp;
    query_end timestamp;
BEGIN
    query_start := clock_timestamp();
    
    -- Use centroid-based nearest neighbor (should use GIST index)
    SELECT COUNT(*) INTO neighbor_count
    FROM atom_nearest_fast((SELECT id FROM atom WHERE codepoint = 65), 10);
    
    query_end := clock_timestamp();
    
    RAISE NOTICE 'Found % nearest neighbors in %', neighbor_count, query_end - query_start;
    
    PERFORM assert_eq(neighbor_count::integer, 10, 
        'atom_nearest_fast returns requested count');
END $$;

-- =============================================================================
-- TEST 11: AI/ML Functions Exist
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 11: AI/ML Function Availability ───'; END $$;

DO $$
BEGIN
    -- Check AI operation functions exist
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'attention_self'),
        'attention_self function exists');
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'transform_analogy'),
        'transform_analogy function exists');
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'infer_related'),
        'infer_related function exists');
    PERFORM assert_true(
        EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'generate_random_walk'),
        'generate_random_walk function exists');
END $$;

-- =============================================================================
-- TEST 12: Coordinate Precision (32-bit per dimension)
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '─── Test 12: Coordinate Precision ───'; END $$;

DO $$
DECLARE
    sample_x double precision;
    sample_y double precision;
    max_coord double precision;
BEGIN
    -- Get sample coordinates (should be raw uint32 values stored in double)
    SELECT ST_X(geom), ST_Y(geom) INTO sample_x, sample_y
    FROM atom WHERE codepoint = 65;
    
    -- Max possible uint32: 4294967295
    SELECT MAX(ST_X(centroid)) INTO max_coord FROM atom LIMIT 1;
    
    RAISE NOTICE 'Sample coords for A: x=%, y=%', sample_x, sample_y;
    RAISE NOTICE 'Max X coordinate seen: %', max_coord;
    
    -- Verify coordinates are in uint32 range (not normalized to 0-1)
    PERFORM assert_gt(sample_x, 1.0, 
        'X coordinate is raw uint32 (not normalized to 0-1)');
END $$;

-- =============================================================================
-- Summary
-- =============================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '╔══════════════════════════════════════════════════════════════╗'; END $$;
DO $$ BEGIN RAISE NOTICE '║     All unified schema tests completed successfully          ║'; END $$;
DO $$ BEGIN RAISE NOTICE '╚══════════════════════════════════════════════════════════════╝'; END $$;

-- Cleanup
DROP FUNCTION IF EXISTS assert_true(boolean, text);
DROP FUNCTION IF EXISTS assert_eq(anyelement, anyelement, text);
DROP FUNCTION IF EXISTS assert_gt(double precision, double precision, text);

ROLLBACK;
