#!/bin/bash
#
# Hartonomous Hypercube - Full System Validation
#
# Runs all tests and validations for the semantic web infrastructure:
# 1. C++ unit tests (Hilbert, coordinates, BLAKE3, semantic)
# 2. C++ integration tests (database connectivity, query API)
# 3. SQL schema validation tests
# 4. Performance benchmarks
#
# Usage: ./validate.sh [--quick]
#   --quick   Skip performance tests

set -e

cd "$(dirname "$0")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

QUICK_MODE=false
if [ "$1" == "--quick" ]; then
    QUICK_MODE=true
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Hartonomous Hypercube - Full System Validation           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local name="$1"
    local cmd="$2"
    local timeout_sec="${3:-30}"  # Default 30s, optional third arg
    
    echo -n "  Testing: $name... "
    
    # Use env to pass all PG* variables to child process
    if timeout "$timeout_sec" env PGHOST="$PGHOST" PGPORT="$PGPORT" PGUSER="$PGUSER" PGPASSWORD="$PGPASSWORD" PGDATABASE="$PGDATABASE" bash -c "$cmd" > /tmp/test_output.txt 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((TESTS_PASSED++)) || true
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((TESTS_FAILED++)) || true
        cat /tmp/test_output.txt | head -20
        return 1
    fi
}

# =============================================================================
# Section 1: C++ Unit Tests
# =============================================================================
echo -e "${BLUE}─── C++ Unit Tests ────────────────────────────────────────${NC}"

if [ -f cpp/build/test_hilbert ]; then
    run_test "Hilbert curve" "./cpp/build/test_hilbert"
    run_test "Coordinates" "./cpp/build/test_coordinates"
    run_test "BLAKE3" "./cpp/build/test_blake3"
    run_test "Semantic mapping" "./cpp/build/test_semantic"
else
    echo -e "${YELLOW}  ⚠ C++ tests not built. Run: cd cpp/build && cmake .. && make${NC}"
fi

# =============================================================================
# Section 2: Database Connectivity
# =============================================================================
echo ""
echo -e "${BLUE}─── Database Connectivity ─────────────────────────────────${NC}"

run_test "PostgreSQL connection" "psql -c 'SELECT 1' > /dev/null"
run_test "PostGIS extension" "psql -c 'SELECT PostGIS_Version()' > /dev/null"

# =============================================================================
# Section 3: Schema Validation
# =============================================================================
echo ""
echo -e "${BLUE}─── Schema Validation ─────────────────────────────────────${NC}"

run_test "Atom table exists" "psql -tAc \"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'atom'\" | grep -q '^1$'"
run_test "GIST index exists" "psql -tAc \"SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_geom'\" | grep -q '^1$'"
run_test "Hilbert index exists" "psql -tAc \"SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_hilbert'\" | grep -q '^1$'"

# =============================================================================
# Section 4: Atom Seeding
# =============================================================================
echo ""
echo -e "${BLUE}─── Atom Seeding ──────────────────────────────────────────${NC}"

ATOM_COUNT=$(psql -tAc "SELECT COUNT(*) FROM atom WHERE depth = 0")
echo "  Leaf atoms (codepoints): $ATOM_COUNT"

run_test "All Unicode atoms seeded (>1.1M)" "[ $ATOM_COUNT -gt 1100000 ]"

ASCII_COUNT=$(psql -tAc "SELECT COUNT(*) FROM atom WHERE codepoint BETWEEN 0 AND 127")
run_test "ASCII complete (128)" "[ $ASCII_COUNT -eq 128 ]"

BAD_ID_COUNT=$(psql -tAc "SELECT COUNT(*) FROM atom WHERE length(id) != 32")
run_test "All have 32-byte IDs" "[ $BAD_ID_COUNT -eq 0 ]"

BAD_SRID_COUNT=$(psql -tAc "SELECT COUNT(*) FROM atom WHERE ST_SRID(geom) != 0")
run_test "SRID = 0 for all" "[ $BAD_SRID_COUNT -eq 0 ]"

# =============================================================================
# Section 5: Compositions
# =============================================================================
echo ""
echo -e "${BLUE}─── Compositions ──────────────────────────────────────────${NC}"

COMP_COUNT=$(psql -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0")
echo "  Compositions: $COMP_COUNT"

if [ "$COMP_COUNT" -gt 0 ]; then
    NULL_CHILDREN=$(psql -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0 AND children IS NULL")
    run_test "Compositions have children" "[ $NULL_CHILDREN -eq 0 ]"
    
    BAD_POINT=$(psql -tAc "SELECT COUNT(*) FROM atom WHERE depth = 0 AND ST_GeometryType(geom) != 'ST_Point'")
    run_test "Leaves are POINT" "[ $BAD_POINT -eq 0 ]"
else
    echo "  (No compositions yet - run ./cpp/build/cpe_ingest on some files)"
fi

# =============================================================================
# Section 6: C++ Integration Tests
# =============================================================================
echo ""
echo -e "${BLUE}─── C++ Integration Tests ─────────────────────────────────${NC}"

if [ -f tests/test_integration ]; then
    run_test "Integration test suite" "./tests/test_integration -h localhost"
elif [ -f cpp/build/test_integration ]; then
    run_test "Integration test" "./cpp/build/test_integration"
fi

if [ -f cpp/build/test_query_api ]; then
    run_test "Query API test" "./cpp/build/test_query_api" 60
fi

# =============================================================================
# Section 7: SQL Function Tests
# =============================================================================
echo ""
echo -e "${BLUE}─── SQL Function Tests ────────────────────────────────────${NC}"

run_test "atom_is_leaf()" "psql -tAc \"SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))\" | grep -q 't'"
run_test "atom_centroid()" "psql -tAc \"SELECT (atom_centroid((SELECT id FROM atom WHERE codepoint = 65))).x IS NOT NULL\" | grep -q 't'"
run_test "atom_distance()" "psql -tAc \"SELECT atom_distance((SELECT id FROM atom WHERE codepoint = 65), (SELECT id FROM atom WHERE codepoint = 97)) > 0\" | grep -q 't'"
run_test "atom_reconstruct_text()" "psql -tAc \"SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))\" | grep -q 'A'"

# =============================================================================
# Section 8: Performance Tests
# =============================================================================
if [ "$QUICK_MODE" = false ]; then
    echo ""
    echo -e "${BLUE}─── Performance Tests ─────────────────────────────────────${NC}"
    
    # Time a Hilbert range query
    echo -n "  Hilbert range query: "
    TIME_START=$(date +%s.%N)
    psql -c "SELECT COUNT(*) FROM atom a, atom t 
             WHERE t.codepoint = 65 
             AND a.hilbert_hi = t.hilbert_hi 
             AND a.hilbert_lo BETWEEN t.hilbert_lo - 1000000 AND t.hilbert_lo + 1000000" > /dev/null
    TIME_END=$(date +%s.%N)
    ELAPSED=$(echo "$TIME_END - $TIME_START" | bc)
    echo "${ELAPSED}s"
    
    # Time a spatial KNN query  
    echo -n "  Spatial KNN (5): "
    TIME_START=$(date +%s.%N)
    psql -c "SELECT COUNT(*) FROM atom_nearest_spatial((SELECT id FROM atom WHERE codepoint = 65), 5)" > /dev/null
    TIME_END=$(date +%s.%N)
    ELAPSED=$(echo "$TIME_END - $TIME_START" | bc)
    echo "${ELAPSED}s"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════"
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "  ${GREEN}All $TESTS_PASSED tests passed!${NC}"
else
    echo -e "  ${RED}$TESTS_FAILED tests failed${NC}, $TESTS_PASSED passed"
fi
echo "════════════════════════════════════════════════════════════════"
echo ""

# Stats summary
echo "System Statistics:"
echo "  ├─ Atoms (depth=0):      $(psql -tAc "SELECT COUNT(*) FROM atom WHERE depth = 0")"
echo "  ├─ Compositions (depth>0): $(psql -tAc "SELECT COUNT(*) FROM atom WHERE depth > 0")"
echo "  ├─ Max depth:            $(psql -tAc "SELECT MAX(depth) FROM atom")"
echo "  └─ Table size:           $(psql -tAc "SELECT pg_size_pretty(pg_total_relation_size('atom'))")"
echo ""

exit $TESTS_FAILED
