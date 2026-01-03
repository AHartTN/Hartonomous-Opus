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

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Hartonomous Hypercube - Test Suite (Linux)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local name="$1"
    local cmd="$2"
    
    echo -n "  Testing: $name... "
    if eval "$cmd" > /tmp/test_output.txt 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((TESTS_PASSED++)) || true
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        ((TESTS_FAILED++)) || true
        head -10 /tmp/test_output.txt 2>/dev/null | sed 's/^/    /'
        return 1
    fi
}

# Section 1: C++ Unit Tests
echo -e "${BLUE}─── C++ Unit Tests ────────────────────────────────────────${NC}"

for test in test_hilbert test_coordinates test_blake3 test_semantic test_clustering; do
    if [ -f "$HC_BUILD_DIR/$test" ]; then
        run_test "$test" "$HC_BUILD_DIR/$test" || true
    else
        echo -e "  Skipping $test ${YELLOW}(not built)${NC}"
    fi
done

# Section 2: Database Connectivity
echo ""
echo -e "${BLUE}─── Database Connectivity ─────────────────────────────────${NC}"

run_test "PostgreSQL connection" "hc_psql -c 'SELECT 1' >/dev/null" || true
run_test "PostGIS extension" "hc_psql -c 'SELECT PostGIS_Version()' >/dev/null" || true

# Section 3: Schema Validation
echo ""
echo -e "${BLUE}─── Schema Validation ─────────────────────────────────────${NC}"

run_test "Atom table exists" "[ \$(hc_psql -tAc \"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'atom'\") = '1' ]" || true
run_test "GIST index exists" "[ \$(hc_psql -tAc \"SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_geom'\") = '1' ]" || true
run_test "Hilbert index exists" "[ \$(hc_psql -tAc \"SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_hilbert'\") = '1' ]" || true

# Section 4: Atom Seeding - Use canonical functions
echo ""
echo -e "${BLUE}─── Atom Seeding ──────────────────────────────────────────${NC}"

ATOM_COUNT=$(hc_psql -tAc "SELECT leaf_atoms FROM db_stats()" 2>/dev/null | tr -d '[:space:]')
if [ -z "$ATOM_COUNT" ]; then
    ATOM_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom WHERE depth = 0" | tr -d '[:space:]')
fi
echo "  Leaf atoms (codepoints): $ATOM_COUNT"

run_test "All Unicode atoms seeded (>1.1M)" "[ $ATOM_COUNT -gt 1100000 ]" || true

# Use validation function
echo ""
echo -e "${BLUE}─── Data Validation ───────────────────────────────────────${NC}"
hc_psql -c "SELECT * FROM validate_atoms()" 2>/dev/null || echo "  (validate_atoms not available)"

# Section 5: Compositions
echo ""
echo -e "${BLUE}─── Compositions ──────────────────────────────────────────${NC}"

STATS=$(hc_psql -tAc "SELECT compositions FROM db_stats()" 2>/dev/null | tr -d '[:space:]')
if [ -n "$STATS" ]; then
    COMP_COUNT=$STATS
else
    COMP_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0" | tr -d '[:space:]')
fi
EDGE_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom WHERE depth = 1 AND atom_count = 2" | tr -d '[:space:]')
echo "  Compositions: $COMP_COUNT"
echo "  Semantic edges: $EDGE_COUNT"

run_test "Has compositions (depth > 0)" "[ $COMP_COUNT -gt 0 ]" || true

# Section 6: SQL Function Tests
echo ""
echo -e "${BLUE}─── SQL Function Tests ────────────────────────────────────${NC}"

run_test "atom_is_leaf()" "[ \$(hc_psql -tAc \"SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))\") = 't' ]" || true
run_test "atom_centroid()" "[ \$(hc_psql -tAc \"SELECT (atom_centroid((SELECT id FROM atom WHERE codepoint = 65))).x IS NOT NULL\") = 't' ]" || true
run_test "atom_reconstruct_text()" "[ \"\$(hc_psql -tAc \"SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))\")\" = 'A' ]" || true
run_test "atom_content_hash()" "[ \$(hc_psql -tAc \"SELECT length(encode(atom_content_hash('hello'), 'hex'))\") = '64' ]" || true

# Section 7: AI/ML Operations Tests
echo ""
echo -e "${BLUE}─── AI/ML Operations ──────────────────────────────────────${NC}"

if [ $COMP_COUNT -gt 100 ]; then
    # Test semantic neighbors
    run_test "semantic_neighbors()" "hc_psql -tAc \"SELECT COUNT(*) FROM semantic_neighbors((SELECT id FROM atom WHERE codepoint = 116), 5)\" | grep -E '^[0-5]$'" || true
    
    # Test attention
    run_test "attention()" "hc_psql -tAc \"SELECT COUNT(*) FROM attention((SELECT id FROM atom WHERE depth > 0 LIMIT 1), 5)\" | grep -E '^[0-5]$'" || true
    
    # Test content hash lookup
    run_test "Content hash: 'the'" "[ \$(hc_psql -tAc \"SELECT EXISTS(SELECT 1 FROM atom WHERE id = atom_content_hash('the'))\") = 't' ]" || true
    run_test "Content hash: 'man'" "[ \$(hc_psql -tAc \"SELECT EXISTS(SELECT 1 FROM atom WHERE id = atom_content_hash('man'))\") = 't' ]" || true
    
    # Test spatial queries
    run_test "Spatial KNN" "hc_psql -tAc \"SELECT COUNT(*) FROM atom_nearest_spatial((SELECT id FROM atom WHERE depth > 0 LIMIT 1), 5)\" 2>/dev/null | grep -E '^[0-5]$'" || true
    
    # Test Hilbert range queries
    run_test "Hilbert range query" "hc_psql -c \"SELECT COUNT(*) FROM atom a, atom t WHERE t.codepoint = 65 AND a.hilbert_hi = t.hilbert_hi AND a.hilbert_lo BETWEEN t.hilbert_lo - 1000 AND t.hilbert_lo + 1000\" >/dev/null" || true
else
    echo -e "  ${YELLOW}Skipping AI tests (need compositions - run ingest-testdata.sh first)${NC}"
fi

# Section 8: C++ Integration Tests
echo ""
echo -e "${BLUE}─── C++ Integration Tests ─────────────────────────────────${NC}"

if [ -f "$HC_BUILD_DIR/test_integration" ]; then
    run_test "Integration tests" "$HC_BUILD_DIR/test_integration" || true
fi

if [ -f "$HC_BUILD_DIR/test_query_api" ]; then
    run_test "Query API tests" "$HC_BUILD_DIR/test_query_api" || true
fi

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "  ${GREEN}All $TESTS_PASSED tests passed!${NC}"
else
    echo -e "  ${RED}$TESTS_FAILED tests failed${NC}, $TESTS_PASSED passed"
fi
echo "════════════════════════════════════════════════════════════════"
echo ""

# System stats
echo "System Statistics:"
hc_psql -c "
SELECT 
    COUNT(*) FILTER (WHERE depth = 0) as \"Atoms\",
    COUNT(*) FILTER (WHERE depth > 0 AND atom_count > 2) as \"Compositions\",
    COUNT(*) FILTER (WHERE depth = 1 AND atom_count = 2) as \"Edges\",
    MAX(depth) as \"MaxDepth\",
    pg_size_pretty(pg_total_relation_size('atom')) as \"Size\"
FROM atom;
" 2>/dev/null || true

exit $TESTS_FAILED
