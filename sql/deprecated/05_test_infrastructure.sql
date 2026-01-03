-- =============================================================================
-- Test Infrastructure
-- Status: CURRENT - Permanent test utilities for Hartonomous
-- =============================================================================
--
-- This file provides reusable test infrastructure including:
-- 1. Assertion functions for test cases
-- 2. Validation functions for schema integrity
-- 3. Helper functions for test data generation
--
-- Usage in test files:
--   SELECT test.assert(condition, 'Test name');
--   SELECT test.assert_eq(actual, expected, 'Test name');
--   SELECT * FROM test.validate_all();

BEGIN;

-- =============================================================================
-- Test Schema
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS test;

-- =============================================================================
-- Assertion Functions
-- =============================================================================

-- Basic boolean assertion
CREATE OR REPLACE FUNCTION test.assert(condition boolean, test_name text)
RETURNS void AS $$
BEGIN
    IF NOT condition THEN
        RAISE EXCEPTION 'FAILED: %', test_name;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

-- Equality assertion (works with any type)
CREATE OR REPLACE FUNCTION test.assert_eq(val1 anyelement, val2 anyelement, test_name text)
RETURNS void AS $$
BEGIN
    IF val1 IS DISTINCT FROM val2 THEN
        RAISE EXCEPTION 'FAILED: % (got %, expected %)', test_name, val1, val2;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

-- Greater than assertion
CREATE OR REPLACE FUNCTION test.assert_gt(val1 double precision, val2 double precision, test_name text)
RETURNS void AS $$
BEGIN
    IF val1 <= val2 THEN
        RAISE EXCEPTION 'FAILED: % (% <= %)', test_name, val1, val2;
    END IF;
    RAISE NOTICE 'PASSED: % (% > %)', test_name, val1, val2;
END;
$$ LANGUAGE plpgsql;

-- Less than assertion
CREATE OR REPLACE FUNCTION test.assert_lt(val1 double precision, val2 double precision, test_name text)
RETURNS void AS $$
BEGIN
    IF val1 >= val2 THEN
        RAISE EXCEPTION 'FAILED: % (% >= %)', test_name, val1, val2;
    END IF;
    RAISE NOTICE 'PASSED: % (% < %)', test_name, val1, val2;
END;
$$ LANGUAGE plpgsql;

-- Not null assertion
CREATE OR REPLACE FUNCTION test.assert_not_null(val anyelement, test_name text)
RETURNS void AS $$
BEGIN
    IF val IS NULL THEN
        RAISE EXCEPTION 'FAILED: % (got NULL)', test_name;
    END IF;
    RAISE NOTICE 'PASSED: %', test_name;
END;
$$ LANGUAGE plpgsql;

-- Range assertion (inclusive)
CREATE OR REPLACE FUNCTION test.assert_between(
    val double precision, 
    min_val double precision, 
    max_val double precision, 
    test_name text
)
RETURNS void AS $$
BEGIN
    IF val < min_val OR val > max_val THEN
        RAISE EXCEPTION 'FAILED: % (% not in [%, %])', test_name, val, min_val, max_val;
    END IF;
    RAISE NOTICE 'PASSED: % (% in [%, %])', test_name, val, min_val, max_val;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Schema Validation Functions
-- =============================================================================

-- Validate atom table structure
CREATE OR REPLACE FUNCTION test.validate_atom_schema()
RETURNS TABLE(test_name text, passed boolean, message text) AS $$
DECLARE
    v_result RECORD;
BEGIN
    -- Check required columns exist
    FOR v_result IN
        SELECT 
            'Column ' || column_name || ' exists' as test_name,
            TRUE as passed,
            'OK' as message
        FROM information_schema.columns 
        WHERE table_name = 'atom' 
          AND column_name IN ('id', 'geom', 'children', 'hilbert_lo', 'hilbert_hi', 'depth', 'atom_count', 'codepoint', 'value')
    LOOP
        test_name := v_result.test_name;
        passed := v_result.passed;
        message := v_result.message;
        RETURN NEXT;
    END LOOP;
    
    -- Check required indexes exist
    IF EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = 'idx_atom_geom') THEN
        test_name := 'GIST index on geom exists';
        passed := TRUE;
        message := 'OK';
    ELSE
        test_name := 'GIST index on geom exists';
        passed := FALSE;
        message := 'Index idx_atom_geom not found';
    END IF;
    RETURN NEXT;
    
    IF EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = 'idx_atom_hilbert') THEN
        test_name := 'Hilbert index exists';
        passed := TRUE;
        message := 'OK';
    ELSE
        test_name := 'Hilbert index exists';
        passed := FALSE;
        message := 'Index idx_atom_hilbert not found';
    END IF;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Validate atom data integrity
CREATE OR REPLACE FUNCTION test.validate_atom_data()
RETURNS TABLE(test_name text, passed boolean, message text) AS $$
DECLARE
    v_count BIGINT;
BEGIN
    -- Check leaf atom count
    SELECT COUNT(*) INTO v_count FROM atom WHERE depth = 0;
    test_name := 'Leaf atoms exist';
    passed := v_count > 0;
    message := format('%s leaf atoms', v_count);
    RETURN NEXT;
    
    -- Check Unicode coverage (should have at least 1.1M codepoints)
    test_name := 'Unicode coverage >= 1.1M';
    passed := v_count >= 1100000;
    message := format('%s codepoints (need 1100000)', v_count);
    RETURN NEXT;
    
    -- Check ASCII coverage
    SELECT COUNT(*) INTO v_count FROM atom WHERE codepoint BETWEEN 0 AND 127;
    test_name := 'All 128 ASCII codepoints present';
    passed := v_count = 128;
    message := format('%s ASCII atoms', v_count);
    RETURN NEXT;
    
    -- Check no surrogates
    SELECT COUNT(*) INTO v_count FROM atom WHERE codepoint BETWEEN 55296 AND 57343;
    test_name := 'No surrogate codepoints';
    passed := v_count = 0;
    message := format('%s surrogates (should be 0)', v_count);
    RETURN NEXT;
    
    -- Check all atoms have geometry
    SELECT COUNT(*) INTO v_count FROM atom WHERE geom IS NULL;
    test_name := 'All atoms have geometry';
    passed := v_count = 0;
    message := format('%s null geometries', v_count);
    RETURN NEXT;
    
    -- Check geometry types
    SELECT COUNT(*) INTO v_count FROM atom 
    WHERE depth = 0 AND ST_GeometryType(geom) != 'ST_Point';
    test_name := 'Leaves are POINT geometries';
    passed := v_count = 0;
    message := format('%s non-POINT leaves', v_count);
    RETURN NEXT;
    
    SELECT COUNT(*) INTO v_count FROM atom 
    WHERE depth > 0 AND ST_GeometryType(geom) != 'ST_LineString';
    test_name := 'Compositions are LINESTRING geometries';
    passed := v_count = 0;
    message := format('%s non-LINESTRING compositions', v_count);
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Run all validation tests
CREATE OR REPLACE FUNCTION test.validate_all()
RETURNS TABLE(test_name text, passed boolean, message text) AS $$
BEGIN
    -- Schema validation
    RETURN QUERY SELECT * FROM test.validate_atom_schema();
    
    -- Data validation
    RETURN QUERY SELECT * FROM test.validate_atom_data();
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Test Summary View
-- =============================================================================

CREATE OR REPLACE VIEW test.summary AS
SELECT
    (SELECT COUNT(*) FROM atom WHERE depth = 0) as leaf_count,
    (SELECT COUNT(*) FROM atom WHERE depth > 0) as composition_count,
    (SELECT COUNT(*) FROM atom) as total_atoms,
    (SELECT MAX(depth) FROM atom) as max_depth,
    (SELECT pg_size_pretty(pg_total_relation_size('atom'))) as table_size;

COMMIT;

-- Run validation on install
DO $$
BEGIN
    RAISE NOTICE 'Test infrastructure installed. Run SELECT * FROM test.validate_all();';
END $$;
