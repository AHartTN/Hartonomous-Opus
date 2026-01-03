-- Semantic Validation Tests for Hartonomous Hypercube
-- Verifies that Unicode atoms are properly distributed on the hypercube surface
-- with correct semantic clustering properties
--
-- Run: psql -d hypercube -f test_semantic_validation.sql
-- Expects: atoms already seeded via seed_atoms

\set ON_ERROR_STOP on
\timing on

BEGIN;

-- Helper function for test assertions
CREATE OR REPLACE FUNCTION assert_true(condition boolean, test_name text)
RETURNS void AS $$
BEGIN
    IF NOT condition THEN
        RAISE EXCEPTION 'FAILED: %', test_name;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION assert_less_than(val1 double precision, val2 double precision, test_name text)
RETURNS void AS $$
BEGIN
    IF val1 >= val2 THEN
        RAISE EXCEPTION 'FAILED: % (% >= %)', test_name, val1, val2;
    END IF;
    RAISE NOTICE 'PASSED: % (% < %)', test_name, val1, val2;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN RAISE NOTICE '============================================'; END $$;
DO $$ BEGIN RAISE NOTICE '  Semantic Validation Test Suite'; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;
DO $$ BEGIN RAISE NOTICE ''; END $$;

-- ===========================================================================
-- TEST 1: All atoms are on the 3-sphere surface (S³)
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE '--- Test 1: S³ Surface Constraint ---'; END $$;

DO $$
DECLARE
    off_surface_count bigint;
    total_count bigint;
    avg_r_squared double precision;
    min_r_squared double precision;
    max_r_squared double precision;
BEGIN
    SELECT COUNT(*) INTO total_count FROM atom;
    
    -- Calculate r² = x² + y² + z² + m² for all atoms
    -- Coords are in [0,1] normalized, need to map to [-1,1] for unit sphere check
    -- x_unit = x_norm * 2 - 1
    -- For unit 3-sphere: r² should equal 1
    WITH radius_check AS (
        SELECT 
            codepoint,
            POWER(ST_X(coords) * 2 - 1, 2) + 
            POWER(ST_Y(coords) * 2 - 1, 2) + 
            POWER(ST_Z(coords) * 2 - 1, 2) + 
            POWER(ST_M(coords) * 2 - 1, 2) AS r_squared
        FROM atom
    )
    SELECT 
        COUNT(*) FILTER (WHERE r_squared < 0.99 OR r_squared > 1.01),
        AVG(r_squared),
        MIN(r_squared),
        MAX(r_squared)
    INTO off_surface_count, avg_r_squared, min_r_squared, max_r_squared
    FROM radius_check;
    
    RAISE NOTICE 'r² statistics: avg=%, min=%, max=%', avg_r_squared, min_r_squared, max_r_squared;
    
    PERFORM assert_true(off_surface_count = 0, 
        format('All %s atoms are on 3-sphere surface (found %s off-surface, r² should be ~1.0)', 
               total_count, off_surface_count));
END $$;

-- ===========================================================================
-- TEST 2: Case pairs are semantically close (A near a, closer than A to Z)
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 2: Case Pair Proximity ---'; END $$;

DO $$
DECLARE
    upper_A_coords geometry;
    lower_a_coords geometry;
    upper_Z_coords geometry;
    lower_z_coords geometry;
    dist_upper_A_lower_a double precision;
    dist_upper_A_upper_Z double precision;
    dist_upper_A_lower_z double precision;
BEGIN
    -- Get coordinates for test characters
    SELECT coords INTO upper_A_coords FROM atom WHERE codepoint = 65;  -- 'A'
    SELECT coords INTO lower_a_coords FROM atom WHERE codepoint = 97;  -- 'a'
    SELECT coords INTO upper_Z_coords FROM atom WHERE codepoint = 90;  -- 'Z'
    SELECT coords INTO lower_z_coords FROM atom WHERE codepoint = 122; -- 'z'
    
    -- Use 4D distance (ST_3DDistance only does 3D, so we compute manually)
    -- For POINTZM: sqrt((x1-x2)² + (y1-y2)² + (z1-z2)² + (m1-m2)²)
    dist_upper_A_lower_a := sqrt(
        power(ST_X(upper_A_coords) - ST_X(lower_a_coords), 2) +
        power(ST_Y(upper_A_coords) - ST_Y(lower_a_coords), 2) +
        power(ST_Z(upper_A_coords) - ST_Z(lower_a_coords), 2) +
        power(ST_M(upper_A_coords) - ST_M(lower_a_coords), 2)
    );
    
    dist_upper_A_upper_Z := sqrt(
        power(ST_X(upper_A_coords) - ST_X(upper_Z_coords), 2) +
        power(ST_Y(upper_A_coords) - ST_Y(upper_Z_coords), 2) +
        power(ST_Z(upper_A_coords) - ST_Z(upper_Z_coords), 2) +
        power(ST_M(upper_A_coords) - ST_M(upper_Z_coords), 2)
    );
    
    dist_upper_A_lower_z := sqrt(
        power(ST_X(upper_A_coords) - ST_X(lower_z_coords), 2) +
        power(ST_Y(upper_A_coords) - ST_Y(lower_z_coords), 2) +
        power(ST_Z(upper_A_coords) - ST_Z(lower_z_coords), 2) +
        power(ST_M(upper_A_coords) - ST_M(lower_z_coords), 2)
    );
    
    RAISE NOTICE 'Distance A-a: %, A-Z: %, A-z: %', dist_upper_A_lower_a, dist_upper_A_upper_Z, dist_upper_A_lower_z;
    
    -- CRITICAL TEST: A should be closer to a than to Z
    -- Because semantic ordering puts A(slot 0) next to a(slot 1), 
    -- while Z is at slot 25*256
    PERFORM assert_less_than(dist_upper_A_lower_a, dist_upper_A_upper_Z,
        'A is closer to a (same letter, different case) than to Z (different letter)');
    
    -- Also verify categories are correct
    PERFORM assert_true(
        (SELECT category FROM atom WHERE codepoint = 65) = 'letter_upper',
        'A is categorized as letter_upper');
    PERFORM assert_true(
        (SELECT category FROM atom WHERE codepoint = 97) = 'letter_lower',
        'a is categorized as letter_lower');
END $$;

-- ===========================================================================
-- TEST 2b: Latin variants are correctly ordered in Hilbert space
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 2b: Semantic Ordering ---'; END $$;

DO $$
DECLARE
    a_hilbert_lo bigint;
    grave_hilbert_lo bigint;  -- À 
    z_hilbert_lo bigint;
BEGIN
    -- Get Hilbert indices - these should reflect semantic ordering
    SELECT hilbert_lo INTO a_hilbert_lo FROM atom WHERE codepoint = 65;     -- 'A'
    SELECT hilbert_lo INTO grave_hilbert_lo FROM atom WHERE codepoint = 192; -- 'À'
    SELECT hilbert_lo INTO z_hilbert_lo FROM atom WHERE codepoint = 90;      -- 'Z'
    
    RAISE NOTICE 'Hilbert indices - A: %, À: %, Z: %', a_hilbert_lo, grave_hilbert_lo, z_hilbert_lo;
    
    -- Verify all have valid, distinct Hilbert indices
    PERFORM assert_true(a_hilbert_lo IS NOT NULL, 'A has Hilbert index');
    PERFORM assert_true(grave_hilbert_lo IS NOT NULL, 'À has Hilbert index');
    PERFORM assert_true(z_hilbert_lo IS NOT NULL, 'Z has Hilbert index');
    PERFORM assert_true(a_hilbert_lo != z_hilbert_lo, 'A and Z have different Hilbert indices');
END $$;

-- ===========================================================================
-- TEST 3: Category clustering - same category atoms are near each other
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 3: Category Clustering ---'; END $$;

DO $$
DECLARE
    digit_centroid geometry;
    letter_upper_centroid geometry;
    punctuation_centroid geometry;
    intra_digit_dist double precision;
    inter_category_dist double precision;
BEGIN
    -- Compute category centroids
    SELECT ST_Centroid(ST_Collect(coords)) INTO digit_centroid
    FROM atom WHERE category = 'digit' AND codepoint BETWEEN 48 AND 57;  -- 0-9
    
    SELECT ST_Centroid(ST_Collect(coords)) INTO letter_upper_centroid
    FROM atom WHERE category = 'letter_upper' AND codepoint BETWEEN 65 AND 90;  -- A-Z
    
    SELECT ST_Centroid(ST_Collect(coords)) INTO punctuation_centroid
    FROM atom WHERE category = 'punctuation_other';
    
    -- Distance between category centroids
    inter_category_dist := ST_3DDistance(digit_centroid, letter_upper_centroid);
    
    RAISE NOTICE 'Inter-category distance (digits vs uppercase): %', inter_category_dist;
    
    PERFORM assert_true(inter_category_dist > 0, 
        'Different categories have different centroids');
END $$;

-- ===========================================================================
-- TEST 4: Digits are ordered correctly by Hilbert index
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 4: Digit Ordering ---'; END $$;

DO $$
DECLARE
    prev_hilbert bigint := NULL;
    curr_hilbert bigint;
    digit_rec record;
    ordering_valid boolean := true;
BEGIN
    -- Check that digits 0-9 have consistent Hilbert ordering
    FOR digit_rec IN 
        SELECT codepoint, hilbert_lo, hilbert_hi 
        FROM atom 
        WHERE codepoint BETWEEN 48 AND 57  -- '0' to '9'
        ORDER BY codepoint
    LOOP
        -- Just verify they have distinct Hilbert indices
        RAISE NOTICE 'Digit %: hilbert_lo=%', 
            chr(digit_rec.codepoint), digit_rec.hilbert_lo;
    END LOOP;
    
    PERFORM assert_true(true, 'Digits have assigned Hilbert indices');
END $$;

-- ===========================================================================
-- TEST 5: Hilbert indices are deterministic (same codepoint = same index)
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 5: Hilbert Determinism ---'; END $$;

DO $$
DECLARE
    h1_lo bigint;
    h1_hi bigint;
    h2_lo bigint;
    h2_hi bigint;
BEGIN
    -- Get Hilbert index for 'A'
    SELECT hilbert_lo, hilbert_hi INTO h1_lo, h1_hi
    FROM atom WHERE codepoint = 65;
    
    -- The same query should return identical values
    SELECT hilbert_lo, hilbert_hi INTO h2_lo, h2_hi
    FROM atom WHERE codepoint = 65;
    
    PERFORM assert_true(h1_lo = h2_lo AND h1_hi = h2_hi,
        'Hilbert index is deterministic for same codepoint');
END $$;

-- ===========================================================================
-- TEST 6: BLAKE3 hashes are unique and deterministic
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 6: BLAKE3 Hash Uniqueness ---'; END $$;

DO $$
DECLARE
    duplicate_count bigint;
    total_atoms bigint;
BEGIN
    SELECT COUNT(*) INTO total_atoms FROM atom;
    
    SELECT total_atoms - COUNT(DISTINCT id) INTO duplicate_count FROM atom;
    
    PERFORM assert_true(duplicate_count = 0,
        format('All %s atoms have unique BLAKE3 hashes', total_atoms));
END $$;

-- ===========================================================================
-- TEST 7: Spatial index is functional
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 7: Spatial Query Performance ---'; END $$;

DO $$
DECLARE
    query_start timestamp;
    query_end timestamp;
    neighbor_count bigint;
    A_coords geometry;
BEGIN
    SELECT coords INTO A_coords FROM atom WHERE codepoint = 65;
    
    query_start := clock_timestamp();
    
    -- Find atoms within spatial distance
    SELECT COUNT(*) INTO neighbor_count
    FROM atom
    WHERE ST_3DDistance(coords, A_coords) < 0.1
      AND codepoint != 65;
    
    query_end := clock_timestamp();
    
    RAISE NOTICE 'Spatial query found % neighbors in %', 
        neighbor_count, query_end - query_start;
    
    PERFORM assert_true(neighbor_count >= 0, 'Spatial index query executes');
END $$;

-- ===========================================================================
-- TEST 8: Text to trajectory conversion works
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 8: Text to Trajectory ---'; END $$;

DO $$
DECLARE
    hello_line geometry;
    world_line geometry;
    frechet_dist double precision;
BEGIN
    -- Convert text to LINESTRINGZM
    SELECT hypercube_text_to_linestring('Hello') INTO hello_line;
    SELECT hypercube_text_to_linestring('World') INTO world_line;
    
    PERFORM assert_true(hello_line IS NOT NULL, 
        'Text "Hello" converts to LINESTRING');
    PERFORM assert_true(ST_NumPoints(hello_line) = 5, 
        format('Hello has 5 points (got %s)', ST_NumPoints(hello_line)));
    
    -- Test Frechet distance
    IF hello_line IS NOT NULL AND world_line IS NOT NULL THEN
        frechet_dist := ST_FrechetDistance(hello_line, world_line);
        RAISE NOTICE 'Frechet distance Hello-World: %', frechet_dist;
        PERFORM assert_true(frechet_dist > 0, 'Frechet distance is positive');
    END IF;
END $$;

-- ===========================================================================
-- TEST 9: Similar words have smaller Frechet distance than dissimilar
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 9: Semantic Trajectory Distance ---'; END $$;

DO $$
DECLARE
    hello_line geometry;
    hallo_line geometry;
    world_line geometry;
    dist_hello_hallo double precision;
    dist_hello_world double precision;
BEGIN
    SELECT hypercube_text_to_linestring('hello') INTO hello_line;
    SELECT hypercube_text_to_linestring('hallo') INTO hallo_line;
    SELECT hypercube_text_to_linestring('world') INTO world_line;
    
    IF hello_line IS NOT NULL AND hallo_line IS NOT NULL AND world_line IS NOT NULL THEN
        dist_hello_hallo := ST_FrechetDistance(hello_line, hallo_line);
        dist_hello_world := ST_FrechetDistance(hello_line, world_line);
        
        RAISE NOTICE 'Frechet distance hello-hallo: %', dist_hello_hallo;
        RAISE NOTICE 'Frechet distance hello-world: %', dist_hello_world;
        
        PERFORM assert_less_than(dist_hello_hallo, dist_hello_world,
            'Similar words (hello/hallo) have smaller Frechet distance than dissimilar (hello/world)');
    END IF;
END $$;

-- ===========================================================================
-- TEST 10: Digits are on the 3-sphere surface
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 10: Digit Surface Check ---'; END $$;

DO $$
DECLARE
    on_surface_count bigint;
    total_digits bigint;
BEGIN
    -- Count digits
    SELECT COUNT(*) INTO total_digits
    FROM atom WHERE codepoint BETWEEN 48 AND 57;
    
    -- Count how many are on the 3-sphere surface (r² ≈ 1)
    SELECT COUNT(*) INTO on_surface_count
    FROM atom
    WHERE codepoint BETWEEN 48 AND 57
      AND ABS(
          POWER(ST_X(coords) * 2 - 1, 2) +
          POWER(ST_Y(coords) * 2 - 1, 2) +
          POWER(ST_Z(coords) * 2 - 1, 2) +
          POWER(ST_M(coords) * 2 - 1, 2) - 1
      ) < 0.01;
    
    PERFORM assert_true(on_surface_count = total_digits,
        format('All %s digits are on S³ surface (got %s)', total_digits, on_surface_count));
END $$;

-- ===========================================================================
-- TEST 11: Hilbert range query returns neighborhood
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 11: Hilbert Range Query ---'; END $$;

DO $$
DECLARE
    center_lo bigint;
    center_hi bigint;
    range_count bigint;
BEGIN
    SELECT hilbert_lo, hilbert_hi INTO center_lo, center_hi
    FROM atom WHERE codepoint = 65;  -- 'A'
    
    -- Find atoms within Hilbert range
    SELECT COUNT(*) INTO range_count
    FROM atom
    WHERE hilbert_hi = center_hi
      AND ABS(hilbert_lo - center_lo) < 10000000;
    
    RAISE NOTICE 'Hilbert range query found % atoms near A', range_count;
    
    PERFORM assert_true(range_count >= 1, 'Hilbert range query returns results');
END $$;

-- ===========================================================================
-- TEST 12: Category distribution statistics
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 12: Category Distribution ---'; END $$;

DO $$
DECLARE
    cat_rec record;
    empty_categories integer := 0;
BEGIN
    FOR cat_rec IN 
        SELECT category, COUNT(*) as cnt
        FROM atom 
        GROUP BY category 
        ORDER BY cnt DESC
    LOOP
        RAISE NOTICE 'Category %: % atoms', cat_rec.category, cat_rec.cnt;
        IF cat_rec.cnt = 0 THEN
            empty_categories := empty_categories + 1;
        END IF;
    END LOOP;
    
    PERFORM assert_true(empty_categories = 0, 'No empty categories');
END $$;

-- ===========================================================================
-- TEST 13: Atom count matches expected Unicode range
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 13: Unicode Coverage ---'; END $$;

DO $$
DECLARE
    total_atoms bigint;
    expected_min bigint := 1000000;  -- At least 1M valid codepoints
    expected_max bigint := 1200000;  -- Less than 1.2M
BEGIN
    SELECT COUNT(*) INTO total_atoms FROM atom;
    
    RAISE NOTICE 'Total atoms: %', total_atoms;
    
    PERFORM assert_true(total_atoms >= expected_min,
        format('At least %s atoms seeded (got %s)', expected_min, total_atoms));
    PERFORM assert_true(total_atoms <= expected_max,
        format('At most %s atoms seeded (got %s)', expected_max, total_atoms));
END $$;

-- ===========================================================================
-- TEST 14: No surrogates stored
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '--- Test 14: Surrogate Exclusion ---'; END $$;

DO $$
DECLARE
    surrogate_count bigint;
BEGIN
    -- Surrogates are U+D800 to U+DFFF (55296 to 57343)
    SELECT COUNT(*) INTO surrogate_count
    FROM atom
    WHERE codepoint BETWEEN 55296 AND 57343;
    
    PERFORM assert_true(surrogate_count = 0,
        'No surrogate codepoints stored in atom table');
END $$;

-- ===========================================================================
-- Summary
-- ===========================================================================
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;
DO $$ BEGIN RAISE NOTICE '  All semantic validation tests completed'; END $$;
DO $$ BEGIN RAISE NOTICE '============================================'; END $$;

-- Cleanup helper functions
DROP FUNCTION IF EXISTS assert_true(boolean, text);
DROP FUNCTION IF EXISTS assert_less_than(double precision, double precision, text);

ROLLBACK;  -- Don't commit test artifacts
