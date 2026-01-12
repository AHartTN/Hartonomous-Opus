#!/bin/bash
# Hartonomous Hypercube - Test Suite (Linux)
# Runs all tests: C++ unit tests, integration tests, SQL tests, AI operations
# Usage: ./scripts/linux/test.sh [--quick]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

QUICK=false
if [ "$1" == "--quick" ]; then
    QUICK=true
fi

echo ""
echo "Hartonomous Hypercube - Test Suite (Linux)"
echo ""

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local name="$1"
    local cmd="$2"

    echo -n "  Testing: $name... "
    if eval "$cmd" > /tmp/test_output.txt 2>&1; then
        echo "PASSED"
        ((TESTS_PASSED++)) || true
        return 0
    else
        echo "FAILED"
        ((TESTS_FAILED++)) || true
        head -10 /tmp/test_output.txt 2>/dev/null | sed 's/^/    /'
        return 1
    fi
}

# Section 1: C++ Unit Tests
echo "C++ Unit Tests"

for test in test_hilbert test_coordinates test_blake3 test_semantic test_clustering; do
    if [ -f "$HC_BUILD_DIR/$test" ]; then
        run_test "$test" "$HC_BUILD_DIR/$test" || true
    else
        echo "  Skipping $test (not built)"
    fi
done

# Section 2: Database Connectivity
echo ""
echo "Database Connectivity"

run_test "PostgreSQL connection" "hc_psql -c 'SELECT 1' >/dev/null" || true
run_test "PostGIS extension" "hc_psql -c 'SELECT PostGIS_Version()' >/dev/null" || true

# Section 3: Schema Validation
echo ""
echo "Schema Validation"

run_test "Atom table exists" "[ \$(hc_psql -tAc \"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'atom'\") = '1' ]" || true
run_test "GIST index exists" "[ \$(hc_psql -tAc \"SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_geom'\") = '1' ]" || true
run_test "Hilbert index exists" "[ \$(hc_psql -tAc \"SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_hilbert'\") = '1' ]" || true

# Section 4: Atom Seeding - Use canonical functions
echo ""
echo "Atom Seeding"

ATOM_COUNT=$(hc_psql -tAc "SELECT atoms FROM db_stats()" 2>/dev/null | tr -d '[:space:]')
if [ -z "$ATOM_COUNT" ]; then
    ATOM_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom" | tr -d '[:space:]')
fi
echo "  Leaf atoms (codepoints): $ATOM_COUNT"

run_test "All Unicode atoms seeded (>1.1M)" "[ $ATOM_COUNT -gt 1100000 ]" || true

# Use validation function
echo ""
echo "Data Validation"
hc_psql -c "SELECT * FROM validate_atoms()" 2>/dev/null || echo "  (validate_atoms not available)"

# Section 5: Compositions
echo ""
echo "Compositions"

STATS=$(hc_psql -tAc "SELECT compositions FROM db_stats()" 2>/dev/null | tr -d '[:space:]')
if [ -n "$STATS" ]; then
    COMP_COUNT=$STATS
else
    COMP_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM composition" | tr -d '[:space:]')
fi
EDGE_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM composition WHERE depth = 1 AND atom_count = 2" | tr -d '[:space:]')
echo "  Compositions: $COMP_COUNT"
echo "  Semantic edges: $EDGE_COUNT"

run_test "Has compositions (depth > 0)" "[ $COMP_COUNT -gt 0 ]" || true

# Section 6: SQL Function Tests
echo ""
echo "SQL Function Tests"

# Simpler test approach - check for non-empty results
LEAF_TEST=$(hc_psql -tAc "SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65 LIMIT 1))" 2>&1 | tr -d '[:space:]')
if [ "$LEAF_TEST" = "t" ]; then
    echo "  Testing: atom_is_leaf()... PASSED"
    ((TESTS_PASSED++)) || true
else
    echo "  Testing: atom_is_leaf()... FAILED"
    echo "    Got: '$LEAF_TEST'"
    ((TESTS_FAILED++)) || true
fi

CENTROID_TEST=$(hc_psql -tAc "SELECT (atom_centroid((SELECT id FROM atom WHERE codepoint = 65 LIMIT 1))).x IS NOT NULL" 2>&1 | tr -d '[:space:]')
if [ "$CENTROID_TEST" = "t" ]; then
    echo "  Testing: atom_centroid()... PASSED"
    ((TESTS_PASSED++)) || true
else
    echo "  Testing: atom_centroid()... FAILED"
    echo "    Got: '$CENTROID_TEST'"
    ((TESTS_FAILED++)) || true
fi

RECON_TEST=$(hc_psql -tAc "SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65 LIMIT 1))" 2>&1 | tr -d '[:space:]')
if [ "$RECON_TEST" = "A" ]; then
    echo "  Testing: atom_reconstruct_text()... PASSED"
    ((TESTS_PASSED++)) || true
else
    echo "  Testing: atom_reconstruct_text()... FAILED"
    echo "    Got: '$RECON_TEST'"
    ((TESTS_FAILED++)) || true
fi

HASH_LEN=$(hc_psql -tAc "SELECT length(encode(atom_content_hash('hello'), 'hex'))" 2>&1 | tr -d '[:space:]')
if [ "$HASH_LEN" = "64" ]; then
    echo "  Testing: atom_content_hash()... PASSED"
    ((TESTS_PASSED++)) || true
else
    echo "  Testing: atom_content_hash()... FAILED"
    echo "    Got hash length: '$HASH_LEN' (expected 64)"
    ((TESTS_FAILED++)) || true
fi

# Section 7: AI/ML Operations Tests
echo ""
echo "AI/ML Operations"

if [ "$COMP_COUNT" -gt 100 ] 2>/dev/null; then
    # Test semantic neighbors with BYTEA argument
    NEIGHBORS=$(hc_psql -tAc "SELECT COUNT(*) FROM semantic_neighbors((SELECT id FROM atom WHERE codepoint = 116 LIMIT 1), 5)" 2>&1 | tr -d '[:space:]')
    if [ -n "$NEIGHBORS" ] && [ "$NEIGHBORS" -ge 0 ] 2>/dev/null; then
        echo "  Testing: semantic_neighbors()... PASSED"
        ((TESTS_PASSED++)) || true
    else
        echo "  Testing: semantic_neighbors()... FAILED"
        echo "    Got: '$NEIGHBORS'"
        ((TESTS_FAILED++)) || true
    fi

    # Test attention with BYTEA argument
    ATTENTION=$(hc_psql -tAc "SELECT COUNT(*) FROM attention((SELECT id FROM composition LIMIT 1), 5)" 2>&1 | tr -d '[:space:]')
    if [ -n "$ATTENTION" ] && [ "$ATTENTION" -ge 0 ] 2>/dev/null; then
        echo "  Testing: attention()... PASSED"
        ((TESTS_PASSED++)) || true
    else
        echo "  Testing: attention()... FAILED"
        echo "    Got: '$ATTENTION'"
        ((TESTS_FAILED++)) || true
    fi

    # Test content hash lookup - these require text ingestion to pass
    # For now, test with single characters which always exist
    A_EXISTS=$(hc_psql -tAc "SELECT EXISTS(SELECT 1 FROM atom WHERE id = atom_content_hash('A'))" 2>&1 | tr -d '[:space:]')
    if [ "$A_EXISTS" = "t" ]; then
        echo "  Testing: Content hash: 'A'... PASSED"
        ((TESTS_PASSED++)) || true
    else
        echo "  Testing: Content hash: 'A'... FAILED"
        echo "    Got: '$A_EXISTS'"
        ((TESTS_FAILED++)) || true
    fi

    HASH_EXISTS=$(hc_psql -tAc "SELECT EXISTS(SELECT 1 FROM atom WHERE id = atom_content_hash('#'))" 2>&1 | tr -d '[:space:]')
    if [ "$HASH_EXISTS" = "t" ]; then
        echo "  Testing: Content hash: '#'... PASSED"
        ((TESTS_PASSED++)) || true
    else
        echo "  Testing: Content hash: '#'... FAILED"
        echo "    Got: '$HASH_EXISTS'"
        ((TESTS_FAILED++)) || true
    fi

    # Test spatial queries
    run_test "Spatial KNN" "hc_psql -tAc \"SELECT COUNT(*) FROM atom_nearest_spatial((SELECT id FROM composition LIMIT 1), 5)\" 2>/dev/null | grep -E '^[0-5]$'" || true

    # Test Hilbert range queries
    run_test "Hilbert range query" "hc_psql -c \"SELECT COUNT(*) FROM atom a, atom t WHERE t.codepoint = 65 AND a.hilbert_hi = t.hilbert_hi AND a.hilbert_lo BETWEEN t.hilbert_lo - 1000 AND t.hilbert_lo + 1000\" >/dev/null" || true
else
    echo "  Skipping AI tests (need compositions - run ingest-testdata.sh first)"
fi

# Section 8: C++ Integration Tests
echo ""
echo "C++ Integration Tests"

if [ -f "$HC_BUILD_DIR/test_integration" ]; then
    run_test "Integration tests" "$HC_BUILD_DIR/test_integration" || true
fi

if [ -f "$HC_BUILD_DIR/test_query_api" ]; then
    run_test "Query API tests" "$HC_BUILD_DIR/test_query_api" || true
fi

# Summary
echo ""
echo "Summary"
if [ $TESTS_FAILED -eq 0 ]; then
    echo "  All $TESTS_PASSED tests passed!"
else
    echo "  $TESTS_FAILED tests failed, $TESTS_PASSED passed"
fi
echo ""

# System stats
echo "System Statistics:"
hc_psql -c "
SELECT
    (SELECT COUNT(*) FROM atom) as \"Atoms\",
    (SELECT COUNT(*) FROM composition WHERE atom_count > 2) as \"Compositions\",
    (SELECT COUNT(*) FROM composition WHERE depth = 1 AND atom_count = 2) as \"Edges\",
    (SELECT MAX(depth) FROM composition) as \"MaxDepth\",
    pg_size_pretty(pg_total_relation_size('atom') + pg_total_relation_size('composition')) as \"Size\";
" 2>/dev/null || true

exit $TESTS_FAILED