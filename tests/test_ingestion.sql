-- Extended Ingestion Tests
-- Tests for text ingestion, model vocab, and semantic trajectories
-- Run: psql -d hypercube -f test_ingestion.sql

\set ON_ERROR_STOP on
\timing on

BEGIN;

-- Helper functions
CREATE OR REPLACE FUNCTION assert_true(condition boolean, test_name text)
RETURNS void AS $$
BEGIN
    IF NOT condition THEN
        RAISE EXCEPTION 'FAILED: %', test_name;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION assert_equal(val1 anyelement, val2 anyelement, test_name text)
RETURNS void AS $$
BEGIN
    IF val1 IS DISTINCT FROM val2 THEN
        RAISE EXCEPTION 'FAILED: % (got % expected %)', test_name, val1, val2;
    END IF;
    RAISE NOTICE 'PASSED: % (got %)', test_name, val1;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN RAISE NOTICE '============================================'; END $$;
DO $$ BEGIN RAISE NOTICE '  Extended Ingestion Test Suite'; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;
DO $$ BEGIN RAISE NOTICE ''; END $$;

-- ===========================================================================
-- TEST 1: Hello World ingestion
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE '--- Test 1: Hello World ---'; END $$;

DO $$
DECLARE
    v_hello_id bytea;
    v_world_id bytea;
    v_hello_text text;
    v_world_text text;
BEGIN
    -- Ingest "Hello" and "World"
    v_hello_id := hypercube_ingest_text('Hello');
    v_world_id := hypercube_ingest_text('World');
    
    PERFORM assert_true(v_hello_id IS NOT NULL, 'Hello ingested successfully');
    PERFORM assert_true(v_world_id IS NOT NULL, 'World ingested successfully');
    
    -- Retrieve and verify
    v_hello_text := hypercube_retrieve_text(v_hello_id);
    v_world_text := hypercube_retrieve_text(v_world_id);
    
    PERFORM assert_equal(v_hello_text, 'Hello', 'Hello retrieves correctly');
    PERFORM assert_equal(v_world_text, 'World', 'World retrieves correctly');
    
    -- Verify different content = different hashes
    PERFORM assert_true(v_hello_id != v_world_id, 'Hello and World have different hashes');
    
    -- Verify same content = same hash (idempotent)
    PERFORM assert_equal(hypercube_ingest_text('Hello'), v_hello_id, 'Re-ingesting Hello returns same ID');
END $$;

-- ===========================================================================
-- TEST 2: Lorem Ipsum - longer text
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 2: Lorem Ipsum ---'; END $$;

DO $$
DECLARE
    v_lorem text := 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.';
    v_lorem_id bytea;
    v_lorem_retrieved text;
    v_atom_count bigint;
BEGIN
    v_lorem_id := hypercube_ingest_text(v_lorem);
    
    PERFORM assert_true(v_lorem_id IS NOT NULL, 'Lorem ipsum ingested');
    
    -- Check atom count in the composition
    SELECT atom_count INTO v_atom_count FROM relation WHERE id = v_lorem_id;
    PERFORM assert_true(v_atom_count = char_length(v_lorem), 
        format('Lorem ipsum has correct atom count (expected %s, got %s)', 
               char_length(v_lorem), v_atom_count));
    
    -- Retrieve and verify
    v_lorem_retrieved := hypercube_retrieve_text(v_lorem_id);
    PERFORM assert_equal(v_lorem_retrieved, v_lorem, 'Lorem ipsum retrieves correctly');
END $$;

-- ===========================================================================
-- TEST 3: Mississippi - repeated characters (RLE-like pattern)
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 3: Mississippi (repeated chars) ---'; END $$;

DO $$
DECLARE
    v_miss_id bytea;
    v_miss_text text;
    v_relation_count bigint;
BEGIN
    v_miss_id := hypercube_ingest_text('Mississippi');
    
    PERFORM assert_true(v_miss_id IS NOT NULL, 'Mississippi ingested');
    
    v_miss_text := hypercube_retrieve_text(v_miss_id);
    PERFORM assert_equal(v_miss_text, 'Mississippi', 'Mississippi retrieves correctly');
    
    -- Count distinct relations created (should reuse repeated n-grams)
    SELECT COUNT(DISTINCT re.child_id) INTO v_relation_count
    FROM relation_edge re
    WHERE re.parent_id = v_miss_id;
    
    RAISE NOTICE 'Mississippi created % distinct child references', v_relation_count;
END $$;

-- ===========================================================================
-- TEST 4: Banana - simple repetition
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 4: Banana ---'; END $$;

DO $$
DECLARE
    v_banana_id bytea;
    v_banana_text text;
BEGIN
    v_banana_id := hypercube_ingest_text('Banana');
    v_banana_text := hypercube_retrieve_text(v_banana_id);
    
    PERFORM assert_equal(v_banana_text, 'Banana', 'Banana retrieves correctly');
END $$;

-- ===========================================================================
-- TEST 5: Tennessee - another repetition pattern
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 5: Tennessee ---'; END $$;

DO $$
DECLARE
    v_tenn_id bytea;
    v_tenn_text text;
BEGIN
    v_tenn_id := hypercube_ingest_text('Tennessee');
    v_tenn_text := hypercube_retrieve_text(v_tenn_id);
    
    PERFORM assert_equal(v_tenn_text, 'Tennessee', 'Tennessee retrieves correctly');
END $$;

-- ===========================================================================
-- TEST 6: Pangram - all letters
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 6: Pangram ---'; END $$;

DO $$
DECLARE
    v_pangram text := 'The quick brown fox jumps over the lazy dog.';
    v_pangram_id bytea;
    v_pangram_text text;
    v_trajectory geometry;
    v_num_points integer;
BEGIN
    v_pangram_id := hypercube_ingest_text(v_pangram);
    v_pangram_text := hypercube_retrieve_text(v_pangram_id);
    
    PERFORM assert_equal(v_pangram_text, v_pangram, 'Pangram retrieves correctly');
    
    -- Get trajectory
    v_trajectory := hypercube_text_to_linestring(v_pangram);
    v_num_points := ST_NumPoints(v_trajectory);
    
    PERFORM assert_equal(v_num_points, char_length(v_pangram), 
        'Pangram trajectory has correct number of points');
END $$;

-- ===========================================================================
-- TEST 7: Similar word similarity
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 7: Word Similarity ---'; END $$;

DO $$
DECLARE
    v_cat_id bytea;
    v_car_id bytea;
    v_dog_id bytea;
    v_cat_car_sim double precision;
    v_cat_dog_sim double precision;
BEGIN
    v_cat_id := hypercube_ingest_text('cat');
    v_car_id := hypercube_ingest_text('car');
    v_dog_id := hypercube_ingest_text('dog');
    
    v_cat_car_sim := hypercube_similarity(v_cat_id, v_car_id);
    v_cat_dog_sim := hypercube_similarity(v_cat_id, v_dog_id);
    
    RAISE NOTICE 'Similarity cat-car: %, cat-dog: %', v_cat_car_sim, v_cat_dog_sim;
    
    -- cat and car share "ca" - should be more similar than cat and dog
    PERFORM assert_true(v_cat_car_sim > v_cat_dog_sim, 
        'cat-car more similar than cat-dog (shared prefix)');
END $$;

-- ===========================================================================
-- TEST 8: Unicode text
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 8: Unicode Text ---'; END $$;

DO $$
DECLARE
    v_chinese text := '中文测试';  -- Chinese test
    v_chinese_id bytea;
    v_chinese_text text;
BEGIN
    v_chinese_id := hypercube_ingest_text(v_chinese);
    
    PERFORM assert_true(v_chinese_id IS NOT NULL, 'Chinese text ingested');
    
    v_chinese_text := hypercube_retrieve_text(v_chinese_id);
    PERFORM assert_equal(v_chinese_text, v_chinese, 'Chinese text retrieves correctly');
END $$;

-- ===========================================================================
-- TEST 9: Empty and single-char edge cases
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 9: Edge Cases ---'; END $$;

DO $$
DECLARE
    v_single_id bytea;
    v_single_text text;
BEGIN
    -- Single character
    v_single_id := hypercube_ingest_text('A');
    v_single_text := hypercube_retrieve_text(v_single_id);
    
    PERFORM assert_equal(v_single_text, 'A', 'Single char retrieves correctly');
END $$;

-- ===========================================================================
-- TEST 10: Deduplication across ingestions
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 10: Deduplication ---'; END $$;

DO $$
DECLARE
    v_count_before bigint;
    v_count_after bigint;
    v_test_id bytea;
BEGIN
    SELECT COUNT(*) INTO v_count_before FROM relation;
    
    -- Ingest the same content multiple times
    v_test_id := hypercube_ingest_text('deduplication test');
    PERFORM hypercube_ingest_text('deduplication test');
    PERFORM hypercube_ingest_text('deduplication test');
    
    SELECT COUNT(*) INTO v_count_after FROM relation;
    
    -- Count should only increase by the compositions for one ingestion
    RAISE NOTICE 'Relations before: %, after: %', v_count_before, v_count_after;
    
    PERFORM assert_true(v_count_after = v_count_before + 
        (SELECT COUNT(*) FROM relation_edge WHERE parent_id = v_test_id) + 1,
        'Duplicate ingestions do not create duplicate relations');
END $$;

-- ===========================================================================
-- TEST 11: Relation depth hierarchy
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 11: Depth Hierarchy ---'; END $$;

DO $$
DECLARE
    v_long_text text;
    v_long_id bytea;
    v_max_depth integer;
BEGIN
    -- Create a longer text that should create multiple hierarchy levels
    v_long_text := 'This is a longer piece of text that should create multiple levels in the Merkle DAG hierarchy. It contains enough characters to require chunking.';
    
    v_long_id := hypercube_ingest_text(v_long_text);
    
    -- Get maximum depth in the DAG
    WITH RECURSIVE dag AS (
        SELECT v_long_id as id, 0 as depth
        UNION ALL
        SELECT e.child_id, d.depth + 1
        FROM dag d
        JOIN relation_edge e ON e.parent_id = d.id
        WHERE d.depth < 20
    )
    SELECT MAX(depth) INTO v_max_depth FROM dag;
    
    RAISE NOTICE 'Long text DAG max depth: %', v_max_depth;
    
    PERFORM assert_true(v_max_depth >= 1, 'Long text creates multi-level DAG');
END $$;

-- ===========================================================================
-- Summary
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;
DO $$ BEGIN RAISE NOTICE '  All ingestion tests completed'; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;

-- Stats
DO $$
DECLARE
    v_atom_count bigint;
    v_relation_count bigint;
    v_edge_count bigint;
BEGIN
    SELECT COUNT(*) INTO v_atom_count FROM atom;
    SELECT COUNT(*) INTO v_relation_count FROM relation;
    SELECT COUNT(*) INTO v_edge_count FROM relation_edge;
    
    RAISE NOTICE 'Atoms: %, Relations: %, Edges: %', 
        v_atom_count, v_relation_count, v_edge_count;
END $$;

-- Cleanup
DROP FUNCTION IF EXISTS assert_true(boolean, text);
DROP FUNCTION IF EXISTS assert_equal(anyelement, anyelement, text);

ROLLBACK;  -- Don't commit test artifacts
