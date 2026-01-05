-- =============================================================================
-- THREE-TABLE SCHEMA TESTS (020_four_tables.sql compatible)
-- =============================================================================
-- Tests for: atom (codepoints), composition (aggregations), relation (edges)
-- =============================================================================

DO $$
DECLARE
    v_count BIGINT;
    v_test_id BYTEA;
    v_result TEXT;
    v_passed INTEGER := 0;
    v_failed INTEGER := 0;
BEGIN
    RAISE NOTICE '=== THREE-TABLE SCHEMA TESTS ===';
    RAISE NOTICE '';

    -- =========================================================================
    -- TEST 1: Atom table has Unicode codepoints
    -- =========================================================================
    SELECT COUNT(*) INTO v_count FROM atom;
    IF v_count > 100000 THEN
        RAISE NOTICE 'PASS: Atom table has % codepoints', v_count;
        v_passed := v_passed + 1;
    ELSE
        RAISE NOTICE 'FAIL: Atom table has only % rows (expected 100k+ Unicode)', v_count;
        v_failed := v_failed + 1;
    END IF;

    -- =========================================================================
    -- TEST 2: Atoms have 4D geometry (geom column)
    -- =========================================================================
    SELECT COUNT(*) INTO v_count FROM atom WHERE geom IS NOT NULL;
    IF v_count > 0 THEN
        RAISE NOTICE 'PASS: % atoms have 4D geometry', v_count;
        v_passed := v_passed + 1;
    ELSE
        RAISE NOTICE 'FAIL: No atoms have geometry';
        v_failed := v_failed + 1;
    END IF;

    -- =========================================================================
    -- TEST 3: Composition table exists with compositions
    -- =========================================================================
    SELECT COUNT(*) INTO v_count FROM composition;
    IF v_count > 0 THEN
        RAISE NOTICE 'PASS: Composition table has % entries', v_count;
        v_passed := v_passed + 1;
    ELSE
        RAISE NOTICE 'WARN: Composition table is empty (run model ingestion)';
        v_failed := v_failed + 1;
    END IF;

    -- =========================================================================
    -- TEST 4: Compositions have centroids
    -- =========================================================================
    SELECT COUNT(*) INTO v_count FROM composition WHERE centroid IS NOT NULL;
    IF v_count > 0 THEN
        RAISE NOTICE 'PASS: % compositions have 4D centroids', v_count;
        v_passed := v_passed + 1;
    ELSE
        RAISE NOTICE 'WARN: No compositions have centroids (run recompute_composition_centroids)';
        v_failed := v_failed + 1;
    END IF;

    -- =========================================================================
    -- TEST 5: Compositions have labels
    -- =========================================================================
    SELECT COUNT(*) INTO v_count FROM composition WHERE label IS NOT NULL;
    IF v_count > 0 THEN
        RAISE NOTICE 'PASS: % compositions have labels', v_count;
        v_passed := v_passed + 1;
    ELSE
        RAISE NOTICE 'FAIL: No compositions have labels';
        v_failed := v_failed + 1;
    END IF;

    -- =========================================================================
    -- TEST 6: Composition_child table has entries
    -- =========================================================================
    SELECT COUNT(*) INTO v_count FROM composition_child;
    IF v_count > 0 THEN
        RAISE NOTICE 'PASS: composition_child has % entries', v_count;
        v_passed := v_passed + 1;
    ELSE
        RAISE NOTICE 'WARN: composition_child is empty';
        v_failed := v_failed + 1;
    END IF;

    -- =========================================================================
    -- TEST 7: Relation table has semantic edges
    -- =========================================================================
    SELECT COUNT(*) INTO v_count FROM relation;
    IF v_count > 0 THEN
        RAISE NOTICE 'PASS: Relation table has % edges', v_count;
        v_passed := v_passed + 1;
    ELSE
        RAISE NOTICE 'WARN: Relation table is empty (run generate_knn_edges)';
        v_failed := v_failed + 1;
    END IF;

    -- =========================================================================
    -- TEST 8: Relation edges use correct column names (source_id, target_id)
    -- =========================================================================
    BEGIN
        EXECUTE 'SELECT source_id, target_id FROM relation LIMIT 1';
        RAISE NOTICE 'PASS: Relation uses source_id/target_id columns';
        v_passed := v_passed + 1;
    EXCEPTION WHEN undefined_column THEN
        RAISE NOTICE 'FAIL: Relation missing source_id/target_id columns';
        v_failed := v_failed + 1;
    END;

    -- =========================================================================
    -- TEST 9: Hilbert indices computed for atoms
    -- =========================================================================
    SELECT COUNT(*) INTO v_count FROM atom WHERE hilbert_lo IS NOT NULL;
    IF v_count > 0 THEN
        RAISE NOTICE 'PASS: % atoms have Hilbert indices', v_count;
        v_passed := v_passed + 1;
    ELSE
        RAISE NOTICE 'FAIL: No atoms have Hilbert indices';
        v_failed := v_failed + 1;
    END IF;

    -- =========================================================================
    -- TEST 10: Sample atom lookup by codepoint
    -- =========================================================================
    SELECT id INTO v_test_id FROM atom WHERE codepoint = 65; -- 'A'
    IF v_test_id IS NOT NULL THEN
        RAISE NOTICE 'PASS: Found atom for codepoint 65 (A)';
        v_passed := v_passed + 1;
    ELSE
        RAISE NOTICE 'FAIL: No atom for codepoint 65';
        v_failed := v_failed + 1;
    END IF;

    -- =========================================================================
    -- SUMMARY
    -- =========================================================================
    RAISE NOTICE '';
    RAISE NOTICE '=== RESULTS: % passed, % failed ===', v_passed, v_failed;
    
    IF v_failed > 0 THEN
        RAISE EXCEPTION 'TESTS FAILED: % failures', v_failed;
    END IF;
END $$;
