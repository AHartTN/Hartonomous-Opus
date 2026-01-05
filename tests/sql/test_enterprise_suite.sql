-- =============================================================================
-- Hypercube Enterprise Test Suite - Comprehensive SQL Validation
-- =============================================================================
-- A comprehensive test suite that validates all SQL functionality including:
-- - Schema integrity
-- - Extension functionality
-- - Core functions (ingest, retrieve, search)
-- - Query API
-- - Generative engine
-- - Data integrity
--
-- Usage: psql -d hypercube -f test_enterprise_suite.sql
-- Exit codes handled via DO blocks raising exceptions on failure
-- =============================================================================

\set ON_ERROR_STOP on
\timing on

-- =============================================================================
-- TEST FRAMEWORK
-- =============================================================================

BEGIN;

-- Create test schema for isolation
CREATE SCHEMA IF NOT EXISTS hc_test;
SET search_path TO hc_test, public;

-- Test results table
CREATE TABLE IF NOT EXISTS hc_test.test_results (
    id SERIAL PRIMARY KEY,
    suite TEXT NOT NULL,
    test_name TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    duration_ms NUMERIC,
    details TEXT,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- Helper functions
CREATE OR REPLACE FUNCTION hc_test.assert_true(condition boolean, test_name text, suite_name text DEFAULT 'GENERAL')
RETURNS void AS $$
DECLARE
    start_time TIMESTAMP;
    duration_ms NUMERIC;
BEGIN
    start_time := clock_timestamp();
    
    IF condition IS NULL THEN
        INSERT INTO hc_test.test_results(suite, test_name, passed, details)
        VALUES (suite_name, test_name, false, 'Condition was NULL');
        RAISE NOTICE '[FAIL] %::%s - Condition was NULL', suite_name, test_name;
        RETURN;
    END IF;
    
    duration_ms := EXTRACT(MILLISECONDS FROM clock_timestamp() - start_time);
    
    IF NOT condition THEN
        INSERT INTO hc_test.test_results(suite, test_name, passed, duration_ms)
        VALUES (suite_name, test_name, false, duration_ms);
        RAISE EXCEPTION '[FAIL] %::%s', suite_name, test_name;
    END IF;
    
    INSERT INTO hc_test.test_results(suite, test_name, passed, duration_ms)
    VALUES (suite_name, test_name, true, duration_ms);
    RAISE NOTICE '[PASS] %::%s (%.2fms)', suite_name, test_name, duration_ms;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hc_test.assert_equal(val1 anyelement, val2 anyelement, test_name text, suite_name text DEFAULT 'GENERAL')
RETURNS void AS $$
BEGIN
    IF val1 IS DISTINCT FROM val2 THEN
        INSERT INTO hc_test.test_results(suite, test_name, passed, details)
        VALUES (suite_name, test_name, false, format('Expected %s, got %s', val2, val1));
        RAISE EXCEPTION '[FAIL] %::%s - Expected %, got %', suite_name, test_name, val2, val1;
    END IF;
    
    INSERT INTO hc_test.test_results(suite, test_name, passed)
    VALUES (suite_name, test_name, true);
    RAISE NOTICE '[PASS] %::%s (value: %)', suite_name, test_name, val1;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hc_test.assert_greater_than(val1 numeric, val2 numeric, test_name text, suite_name text DEFAULT 'GENERAL')
RETURNS void AS $$
BEGIN
    IF val1 <= val2 THEN
        INSERT INTO hc_test.test_results(suite, test_name, passed, details)
        VALUES (suite_name, test_name, false, format('%s <= %s', val1, val2));
        RAISE EXCEPTION '[FAIL] %::%s - % is not greater than %', suite_name, test_name, val1, val2;
    END IF;
    
    INSERT INTO hc_test.test_results(suite, test_name, passed)
    VALUES (suite_name, test_name, true);
    RAISE NOTICE '[PASS] %::%s (% > %)', suite_name, test_name, val1, val2;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '=============================================================================='; END $$;
DO $$ BEGIN RAISE NOTICE '  HYPERCUBE ENTERPRISE TEST SUITE'; END $$;
DO $$ BEGIN RAISE NOTICE '=============================================================================='; END $$;
DO $$ BEGIN RAISE NOTICE ''; END $$;

-- =============================================================================
-- SUITE 1: EXTENSION VALIDATION
-- =============================================================================

DO $$ BEGIN RAISE NOTICE '--- SUITE 1: Extension Validation ---'; END $$;

DO $$
DECLARE
    ext_count INT;
BEGIN
    -- Check hypercube extension
    SELECT COUNT(*) INTO ext_count FROM pg_extension WHERE extname = 'hypercube';
    PERFORM hc_test.assert_equal(ext_count, 1, 'hypercube_extension_installed', 'EXTENSIONS');
    
    -- Check hypercube_ops extension
    SELECT COUNT(*) INTO ext_count FROM pg_extension WHERE extname = 'hypercube_ops';
    PERFORM hc_test.assert_equal(ext_count, 1, 'hypercube_ops_extension_installed', 'EXTENSIONS');
    
    -- Check generative extension
    SELECT COUNT(*) INTO ext_count FROM pg_extension WHERE extname = 'generative';
    PERFORM hc_test.assert_equal(ext_count, 1, 'generative_extension_installed', 'EXTENSIONS');
    
    -- Check semantic_ops extension
    SELECT COUNT(*) INTO ext_count FROM pg_extension WHERE extname = 'semantic_ops';
    PERFORM hc_test.assert_equal(ext_count, 1, 'semantic_ops_extension_installed', 'EXTENSIONS');
    
    -- Check postgis
    SELECT COUNT(*) INTO ext_count FROM pg_extension WHERE extname = 'postgis';
    PERFORM hc_test.assert_equal(ext_count, 1, 'postgis_extension_installed', 'EXTENSIONS');
END $$;

-- =============================================================================
-- SUITE 2: THREE-TABLE SCHEMA VALIDATION
-- =============================================================================
-- Schema: atom (Unicode leaves), composition (aggregations), 
--         composition_child (junction), relation (edges)
-- =============================================================================

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- SUITE 2: Three-Table Schema Validation ---'; END $$;

DO $$
DECLARE
    tbl_count INT;
    col_count INT;
    idx_count INT;
BEGIN
    -- Check atom table exists (Unicode codepoints ONLY - all are leaves)
    SELECT COUNT(*) INTO tbl_count FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'atom';
    PERFORM hc_test.assert_equal(tbl_count, 1, 'atom_table_exists', 'SCHEMA');
    
    -- Check composition table exists (aggregations with 4D centroids)
    SELECT COUNT(*) INTO tbl_count FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'composition';
    PERFORM hc_test.assert_equal(tbl_count, 1, 'composition_table_exists', 'SCHEMA');
    
    -- Check composition_child table exists (junction table)
    SELECT COUNT(*) INTO tbl_count FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'composition_child';
    PERFORM hc_test.assert_equal(tbl_count, 1, 'composition_child_table_exists', 'SCHEMA');
    
    -- Check relation table exists (semantic edges)
    SELECT COUNT(*) INTO tbl_count FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'relation';
    PERFORM hc_test.assert_equal(tbl_count, 1, 'relation_table_exists', 'SCHEMA');
    
    -- Check model table exists
    SELECT COUNT(*) INTO tbl_count FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'model';
    PERFORM hc_test.assert_equal(tbl_count, 1, 'model_table_exists', 'SCHEMA');
    
    -- Check atom table columns (id, codepoint, value, geom, hilbert_lo, hilbert_hi)
    SELECT COUNT(*) INTO col_count FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'atom' 
    AND column_name IN ('id', 'codepoint', 'value', 'geom', 'hilbert_lo', 'hilbert_hi');
    PERFORM hc_test.assert_equal(col_count, 6, 'atom_has_required_columns', 'SCHEMA');
    
    -- Check composition table columns
    SELECT COUNT(*) INTO col_count FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'composition' 
    AND column_name IN ('id', 'depth', 'child_count', 'atom_count', 'centroid');
    PERFORM hc_test.assert_equal(col_count, 5, 'composition_has_required_columns', 'SCHEMA');
    
    -- Check composition_child table columns
    SELECT COUNT(*) INTO col_count FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'composition_child' 
    AND column_name IN ('composition_id', 'ordinal', 'child_type', 'child_id');
    PERFORM hc_test.assert_equal(col_count, 4, 'composition_child_has_required_columns', 'SCHEMA');
    
    -- Check atom has GiST index on geom
    SELECT COUNT(*) INTO idx_count FROM pg_indexes
    WHERE tablename = 'atom' AND indexdef LIKE '%gist%';
    PERFORM hc_test.assert_greater_than(idx_count, 0, 'atom_has_gist_index', 'SCHEMA');
    
    -- Check atom has hilbert index
    SELECT COUNT(*) INTO idx_count FROM pg_indexes
    WHERE tablename = 'atom' AND indexname LIKE '%hilbert%';
    PERFORM hc_test.assert_greater_than(idx_count, 0, 'atom_has_hilbert_index', 'SCHEMA');
    
    -- Check composition has centroid index
    SELECT COUNT(*) INTO idx_count FROM pg_indexes
    WHERE tablename = 'composition' AND indexname LIKE '%centroid%';
    PERFORM hc_test.assert_greater_than(idx_count, 0, 'composition_has_centroid_index', 'SCHEMA');
END $$;

-- =============================================================================
-- SUITE 3: DATA INTEGRITY (Three-Table Schema)
-- =============================================================================
-- atom: ALL are Unicode leaves (no depth column, all are codepoints)
-- composition: aggregations with depth, centroid, etc.
-- composition_child: references atoms ('A') or compositions ('C')
-- =============================================================================

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- SUITE 3: Data Integrity ---'; END $$;

DO $$
DECLARE
    v_atom_count BIGINT;
    v_comp_count BIGINT;
    v_min_cp INT;
    v_max_cp INT;
    v_orphan_count BIGINT;
    v_invalid_coords BIGINT;
    v_null_geom BIGINT;
BEGIN
    -- Check atom count (should be 1,114,112 = full Unicode)
    SELECT COUNT(*) INTO v_atom_count FROM atom;
    PERFORM hc_test.assert_greater_than(v_atom_count, 0, 'atoms_exist', 'DATA_INTEGRITY');
    RAISE NOTICE '  Total atoms: %', v_atom_count;
    
    -- Check codepoint coverage (atoms ARE codepoints, no depth column)
    SELECT MIN(codepoint), MAX(codepoint) INTO v_min_cp, v_max_cp FROM atom;
    PERFORM hc_test.assert_equal(v_min_cp, 0, 'min_codepoint_is_zero', 'DATA_INTEGRITY');
    PERFORM hc_test.assert_equal(v_max_cp, 1114111, 'max_codepoint_is_10FFFF', 'DATA_INTEGRITY');
    
    -- Full Unicode coverage is 1,114,112 codepoints, minus 2,048 surrogates (D800-DFFF)
    -- which are invalid Unicode scalar values = 1,112,064 valid codepoints
    PERFORM hc_test.assert_equal(v_atom_count::INT, 1112064, 'full_unicode_coverage', 'DATA_INTEGRITY');
    
    -- Check all atoms have geometry (geom, not coords)
    SELECT COUNT(*) INTO v_null_geom FROM atom WHERE geom IS NULL;
    PERFORM hc_test.assert_equal(v_null_geom::INT, 0, 'all_atoms_have_geometry', 'DATA_INTEGRITY');
    
    -- Check all atoms have valid 4D coordinates (x,y,z,m all defined)
    -- Note: Coordinates are Hilbert-mapped integers, NOT normalized [0,1]
    SELECT COUNT(*) INTO v_invalid_coords FROM atom
    WHERE geom IS NOT NULL AND (
        ST_X(geom) IS NULL OR
        ST_Y(geom) IS NULL OR
        ST_Z(geom) IS NULL OR
        ST_M(geom) IS NULL
    );
    PERFORM hc_test.assert_equal(v_invalid_coords::INT, 0, 'all_atom_coords_valid', 'DATA_INTEGRITY');
    
    -- Check codepoints are unique
    PERFORM hc_test.assert_true(
        (SELECT COUNT(*) = COUNT(DISTINCT codepoint) FROM atom),
        'codepoints_are_unique',
        'DATA_INTEGRITY'
    );
    
    -- Check composition count (may be 0 if no text ingested yet)
    SELECT COUNT(*) INTO v_comp_count FROM composition;
    RAISE NOTICE '  Total compositions: %', v_comp_count;
    
    -- Check no orphan children in composition_child
    SELECT COUNT(*) INTO v_orphan_count
    FROM composition_child cc
    WHERE cc.child_type = 'A' AND NOT EXISTS (SELECT 1 FROM atom WHERE id = cc.child_id)
       OR cc.child_type = 'C' AND NOT EXISTS (SELECT 1 FROM composition WHERE id = cc.child_id);
    PERFORM hc_test.assert_equal(v_orphan_count::INT, 0, 'no_orphan_composition_children', 'DATA_INTEGRITY');
    
    -- Check composition child_count matches actual children
    SELECT COUNT(*) INTO v_orphan_count
    FROM composition c
    WHERE c.child_count != (SELECT COUNT(*) FROM composition_child WHERE composition_id = c.id);
    PERFORM hc_test.assert_equal(v_orphan_count::INT, 0, 'composition_child_count_accurate', 'DATA_INTEGRITY');
END $$;

-- =============================================================================
-- SUITE 4: CORE FUNCTIONS
-- =============================================================================
-- Functions from 002_core_functions.sql and extensions
-- =============================================================================

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- SUITE 4: Core Functions ---'; END $$;

DO $$
DECLARE
    fn_count INT;
    v_id BYTEA;
    v_text TEXT;
    v_result TEXT;
BEGIN
    -- Check core atom functions exist
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'atom_is_leaf';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'atom_is_leaf_exists', 'CORE_FUNCTIONS');
    
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'atom_centroid';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'atom_centroid_exists', 'CORE_FUNCTIONS');
    
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'atom_children';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'atom_children_exists', 'CORE_FUNCTIONS');
    
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'atom_reconstruct_text';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'atom_reconstruct_text_exists', 'CORE_FUNCTIONS');
    
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'atom_knn';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'atom_knn_exists', 'CORE_FUNCTIONS');
    
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'atom_by_codepoint';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'atom_by_codepoint_exists', 'CORE_FUNCTIONS');
    
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'atom_stats';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'atom_stats_exists', 'CORE_FUNCTIONS');
    
    -- Check semantic functions exist
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'semantic_neighbors';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'semantic_neighbors_exists', 'CORE_FUNCTIONS');
    
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'analogy';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'analogy_exists', 'CORE_FUNCTIONS');
    
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'attention';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'attention_exists', 'CORE_FUNCTIONS');
    
    -- Check distance functions exist
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'centroid_distance';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'centroid_distance_exists', 'CORE_FUNCTIONS');
    
    SELECT COUNT(*) INTO fn_count FROM pg_proc WHERE proname = 'centroid_similarity';
    PERFORM hc_test.assert_greater_than(fn_count, 0, 'centroid_similarity_exists', 'CORE_FUNCTIONS');
    
    -- Test atom_by_codepoint returns correct atom
    v_id := atom_by_codepoint(65); -- 'A'
    PERFORM hc_test.assert_true(v_id IS NOT NULL, 'atom_by_codepoint_works', 'CORE_FUNCTIONS');
    
    -- Test atom_text returns correct character
    v_text := atom_text(v_id);
    PERFORM hc_test.assert_equal(v_text, 'A', 'atom_text_returns_A', 'CORE_FUNCTIONS');
    
    -- Test atom_is_leaf returns true for atom
    PERFORM hc_test.assert_true(atom_is_leaf(v_id), 'atom_is_leaf_for_atom', 'CORE_FUNCTIONS');
END $$;

-- =============================================================================
-- SUITE 5: QUERY API
-- =============================================================================

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- SUITE 5: Query API ---'; END $$;

DO $$
DECLARE
    v_knn_count INT;
    v_upper_A_id BYTEA;
    v_lower_a_id BYTEA;
    v_upper_A_geom GEOMETRY;
    v_lower_a_geom GEOMETRY;
    v_Z_geom GEOMETRY;
    v_dist DOUBLE PRECISION;
BEGIN
    -- Get atom IDs and geometries for test
    SELECT id, geom INTO v_upper_A_id, v_upper_A_geom FROM atom WHERE codepoint = 65; -- 'A'
    SELECT id, geom INTO v_lower_a_id, v_lower_a_geom FROM atom WHERE codepoint = 97; -- 'a'
    SELECT geom INTO v_Z_geom FROM atom WHERE codepoint = 90; -- 'Z'
    
    PERFORM hc_test.assert_true(v_upper_A_geom IS NOT NULL, 'A_has_geometry', 'QUERY_API');
    PERFORM hc_test.assert_true(v_lower_a_geom IS NOT NULL, 'a_has_geometry', 'QUERY_API');
    
    -- Test atom_knn returns correct count
    SELECT COUNT(*) INTO v_knn_count FROM atom_knn(v_upper_A_id, 10);
    PERFORM hc_test.assert_equal(v_knn_count, 10, 'atom_knn_returns_10', 'QUERY_API');
    
    SELECT COUNT(*) INTO v_knn_count FROM atom_knn(v_upper_A_id, 100);
    PERFORM hc_test.assert_equal(v_knn_count, 100, 'atom_knn_returns_100', 'QUERY_API');
    
    -- Test attention function
    SELECT COUNT(*) INTO v_knn_count FROM attention(v_upper_A_id, 10);
    PERFORM hc_test.assert_equal(v_knn_count, 10, 'attention_returns_10', 'QUERY_API');
    
    -- Test centroid_distance function
    v_dist := centroid_distance(v_upper_A_geom, v_lower_a_geom);
    PERFORM hc_test.assert_true(v_dist >= 0, 'distance_is_non_negative', 'QUERY_API');
    PERFORM hc_test.assert_true(v_dist < 4294967296, 'distance_is_bounded', 'QUERY_API'); -- 2^32 max
    
    -- Test centroid_similarity function
    v_dist := centroid_similarity(v_upper_A_geom, v_lower_a_geom);
    PERFORM hc_test.assert_true(v_dist > 0, 'similarity_is_positive', 'QUERY_API');
    PERFORM hc_test.assert_true(v_dist <= 1, 'similarity_is_bounded', 'QUERY_API');
END $$;

-- =============================================================================
-- SUITE 6: COMPOSITION & TEXT HANDLING
-- =============================================================================

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- SUITE 6: Composition & Text Handling ---'; END $$;

DO $$
DECLARE
    v_upper_id BYTEA;
    v_lower_id BYTEA;
    v_text TEXT;
    v_comp_count BIGINT;
    v_fn_count INT;
BEGIN
    -- Test atom_text for single characters
    v_upper_id := atom_by_codepoint(65); -- 'A'
    v_text := atom_text(v_upper_id);
    PERFORM hc_test.assert_equal(v_text, 'A', 'atom_text_for_A', 'COMPOSITION');
    
    v_lower_id := atom_by_codepoint(97); -- 'a'
    v_text := atom_text(v_lower_id);
    PERFORM hc_test.assert_equal(v_text, 'a', 'atom_text_for_a', 'COMPOSITION');
    
    -- Test atom_reconstruct_text for atoms
    v_text := atom_reconstruct_text(v_upper_id);
    PERFORM hc_test.assert_equal(v_text, 'A', 'atom_reconstruct_text_for_atom', 'COMPOSITION');
    
    -- Check if composition functions exist
    SELECT COUNT(*) INTO v_fn_count FROM pg_proc WHERE proname = 'composition_text';
    PERFORM hc_test.assert_greater_than(v_fn_count, 0, 'composition_text_exists', 'COMPOSITION');
    
    SELECT COUNT(*) INTO v_fn_count FROM pg_proc WHERE proname = 'find_composition';
    PERFORM hc_test.assert_greater_than(v_fn_count, 0, 'find_composition_exists', 'COMPOSITION');
    
    -- Check composition table state
    SELECT COUNT(*) INTO v_comp_count FROM composition;
    RAISE NOTICE '  Existing compositions: %', v_comp_count;
    
    -- If compositions exist, test reconstruction
    IF v_comp_count > 0 THEN
        SELECT atom_reconstruct_text(id) INTO v_text
        FROM composition WHERE label IS NOT NULL LIMIT 1;
        PERFORM hc_test.assert_true(v_text IS NOT NULL OR v_text = '', 'composition_reconstruct_works', 'COMPOSITION');
    ELSE
        RAISE NOTICE '  (No compositions to test reconstruction)';
        INSERT INTO hc_test.test_results(suite, test_name, passed, details)
        VALUES ('COMPOSITION', 'composition_reconstruct_works', true, 'SKIPPED: no compositions');
    END IF;
END $$;

-- =============================================================================
-- SUITE 7: UNICODE HANDLING
-- =============================================================================

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- SUITE 7: Unicode Handling ---'; END $$;

DO $$
DECLARE
    v_id BYTEA;
    v_codepoint INT;
    v_count BIGINT;
    v_geom GEOMETRY;
BEGIN
    -- Test ASCII letters exist
    v_id := atom_by_codepoint(65); -- 'A'
    PERFORM hc_test.assert_true(v_id IS NOT NULL, 'ascii_A_exists', 'UNICODE');
    
    -- Test Japanese Hiragana exists (U+3042 codepoint 12354)
    v_id := atom_by_codepoint(12354);
    PERFORM hc_test.assert_true(v_id IS NOT NULL, 'japanese_hiragana_exists', 'UNICODE');
    SELECT codepoint INTO v_codepoint FROM atom WHERE id = v_id;
    PERFORM hc_test.assert_equal(v_codepoint::TEXT, '12354', 'japanese_hiragana_codepoint', 'UNICODE');
    
    -- Test Chinese character exists (U+4E16 codepoint 19990)
    v_id := atom_by_codepoint(19990);
    PERFORM hc_test.assert_true(v_id IS NOT NULL, 'chinese_char_exists', 'UNICODE');
    SELECT codepoint INTO v_codepoint FROM atom WHERE id = v_id;
    PERFORM hc_test.assert_equal(v_codepoint::TEXT, '19990', 'chinese_char_codepoint', 'UNICODE');
    
    -- Test Arabic exists (U+0645 codepoint 1605)
    v_id := atom_by_codepoint(1605);
    PERFORM hc_test.assert_true(v_id IS NOT NULL, 'arabic_char_exists', 'UNICODE');
    
    -- Test emoji exists (U+1F389 codepoint 127881)
    v_id := atom_by_codepoint(127881);
    PERFORM hc_test.assert_true(v_id IS NOT NULL, 'emoji_exists', 'UNICODE');
    SELECT codepoint INTO v_codepoint FROM atom WHERE id = v_id;
    PERFORM hc_test.assert_equal(v_codepoint::TEXT, '127881', 'emoji_codepoint', 'UNICODE');
    
    -- Test high codepoints (supplementary planes) exist
    SELECT COUNT(*) INTO v_count FROM atom WHERE codepoint >= 65536;
    PERFORM hc_test.assert_greater_than(v_count, 0, 'supplementary_planes_populated', 'UNICODE');
    RAISE NOTICE '  Supplementary plane atoms: %', v_count;
    
    -- Test emoji has valid geometry (Hilbert coordinates, not [0,1])
    SELECT geom INTO v_geom FROM atom WHERE codepoint = 127881;
    PERFORM hc_test.assert_true(v_geom IS NOT NULL, 'emoji_has_geometry', 'UNICODE');
    -- Hilbert coordinates are integers in the ~4 billion range
    PERFORM hc_test.assert_true(
        ST_X(v_geom) >= 0 AND ST_Y(v_geom) >= 0,
        'emoji_geometry_valid',
        'UNICODE'
    );
END $$;

-- =============================================================================
-- SUITE 8: PERFORMANCE VALIDATION
-- =============================================================================

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- SUITE 8: Performance Validation ---'; END $$;

DO $$
DECLARE
    v_start TIMESTAMP;
    v_duration_ms NUMERIC;
    v_count INT;
    v_A_id BYTEA;
    v_A_geom GEOMETRY;
BEGIN
    -- Get test atom
    SELECT id, geom INTO v_A_id, v_A_geom FROM atom WHERE codepoint = 65;
    
    -- atom_knn should be fast (< 500ms for 100 results - conservative for GiST)
    v_start := clock_timestamp();
    SELECT COUNT(*) INTO v_count FROM atom_knn(v_A_id, 100);
    v_duration_ms := EXTRACT(MILLISECONDS FROM clock_timestamp() - v_start);
    RAISE NOTICE '  atom_knn(100) took %.2fms', v_duration_ms;
    PERFORM hc_test.assert_true(v_duration_ms < 500, 'atom_knn_100_under_500ms', 'PERFORMANCE');
    
    -- Codepoint lookup should be very fast (indexed)
    v_start := clock_timestamp();
    SELECT COUNT(*) INTO v_count FROM atom WHERE codepoint BETWEEN 65 AND 90;
    v_duration_ms := EXTRACT(MILLISECONDS FROM clock_timestamp() - v_start);
    RAISE NOTICE '  Codepoint range A-Z took %.2fms', v_duration_ms;
    PERFORM hc_test.assert_true(v_duration_ms < 50, 'codepoint_range_under_50ms', 'PERFORMANCE');
    
    -- atom_by_codepoint should be instant
    v_start := clock_timestamp();
    PERFORM atom_by_codepoint(12345);
    v_duration_ms := EXTRACT(MILLISECONDS FROM clock_timestamp() - v_start);
    RAISE NOTICE '  atom_by_codepoint took %.2fms', v_duration_ms;
    PERFORM hc_test.assert_true(v_duration_ms < 10, 'atom_by_codepoint_under_10ms', 'PERFORMANCE');
    
    -- Hilbert range query should use index
    v_start := clock_timestamp();
    SELECT COUNT(*) INTO v_count FROM atom 
    WHERE hilbert_hi = 0 AND hilbert_lo BETWEEN 0 AND 1000000;
    v_duration_ms := EXTRACT(MILLISECONDS FROM clock_timestamp() - v_start);
    RAISE NOTICE '  Hilbert range query took %.2fms', v_duration_ms;
    PERFORM hc_test.assert_true(v_duration_ms < 100, 'hilbert_range_under_100ms', 'PERFORMANCE');
    
    -- centroid_distance calculation should be fast
    v_start := clock_timestamp();
    SELECT COUNT(*) INTO v_count FROM (
        SELECT centroid_distance(v_A_geom, geom) FROM atom LIMIT 1000
    ) t;
    v_duration_ms := EXTRACT(MILLISECONDS FROM clock_timestamp() - v_start);
    RAISE NOTICE '  1000 distance calculations took %.2fms', v_duration_ms;
    PERFORM hc_test.assert_true(v_duration_ms < 100, 'distance_calc_under_100ms', 'PERFORMANCE');
END $$;

-- =============================================================================
-- TEST SUMMARY
-- =============================================================================

DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '=============================================================================='; END $$;
DO $$ BEGIN RAISE NOTICE '  TEST SUMMARY'; END $$;
DO $$ BEGIN RAISE NOTICE '=============================================================================='; END $$;

SELECT 
    suite,
    COUNT(*) FILTER (WHERE passed) as passed,
    COUNT(*) FILTER (WHERE NOT passed) as failed,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE passed) / COUNT(*), 1) as pass_rate
FROM hc_test.test_results
GROUP BY suite
ORDER BY suite;

SELECT 
    'TOTAL' as status,
    COUNT(*) FILTER (WHERE passed) as passed,
    COUNT(*) FILTER (WHERE NOT passed) as failed,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE passed) / COUNT(*), 1) as pass_rate
FROM hc_test.test_results;

-- Cleanup
DROP SCHEMA hc_test CASCADE;

COMMIT;

\echo ''
\echo 'Enterprise Test Suite Complete'
