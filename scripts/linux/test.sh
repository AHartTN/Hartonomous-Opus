#!/bin/bash
# Hartonomous Hypercube - Test Suite (Linux)
# Runs all tests: C++ unit tests, integration tests, SQL tests
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

for test in test_hilbert test_coordinates test_blake3 test_semantic; do
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

# Section 4: Atom Seeding
echo ""
echo -e "${BLUE}─── Atom Seeding ──────────────────────────────────────────${NC}"

ATOM_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom WHERE depth = 0" | tr -d '[:space:]')
echo "  Leaf atoms (codepoints): $ATOM_COUNT"

run_test "All Unicode atoms seeded (>1.1M)" "[ $ATOM_COUNT -gt 1100000 ]" || true
run_test "SRID = 0 for all atoms" "[ \$(hc_psql -tAc 'SELECT COUNT(*) FROM atom WHERE ST_SRID(geom) != 0') = '0' ]" || true

# Section 5: SQL Function Tests
echo ""
echo -e "${BLUE}─── SQL Function Tests ────────────────────────────────────${NC}"

run_test "atom_is_leaf()" "[ \$(hc_psql -tAc \"SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))\") = 't' ]" || true
run_test "atom_centroid()" "[ \$(hc_psql -tAc \"SELECT (atom_centroid((SELECT id FROM atom WHERE codepoint = 65))).x IS NOT NULL\") = 't' ]" || true
run_test "atom_reconstruct_text()" "[ \"\$(hc_psql -tAc \"SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))\")\" = 'A' ]" || true

# Section 6: C++ Integration Tests
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

exit $TESTS_FAILED
