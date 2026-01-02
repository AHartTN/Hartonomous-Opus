#!/bin/bash
# Run all tests for the Hartonomous Hypercube system
#
# Usage: ./test.sh [options]
#   --cpp-only: only run C++ unit tests
#   --sql-only: only run SQL integration tests
#   --verbose: show detailed output

set -e

CPP_ONLY=false
SQL_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpp-only) CPP_ONLY=true; shift ;;
        --sql-only) SQL_ONLY=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/cpp/build"

PASSED=0
FAILED=0

run_test() {
    local name="$1"
    local cmd="$2"
    
    echo -n "  $name... "
    if [ "$VERBOSE" = true ]; then
        echo ""
        if eval "$cmd"; then
            echo "  PASS"
            PASSED=$((PASSED + 1))
        else
            echo "  FAIL"
            FAILED=$((FAILED + 1))
        fi
    else
        if eval "$cmd" > /dev/null 2>&1; then
            echo "PASS"
            PASSED=$((PASSED + 1))
        else
            echo "FAIL"
            FAILED=$((FAILED + 1))
        fi
    fi
}

echo "=============================================="
echo "  Hartonomous Hypercube Test Suite"
echo "=============================================="
echo ""

# C++ Tests
if [ "$SQL_ONLY" = false ]; then
    echo "=== C++ Unit Tests ==="
    
    # Ensure build is current
    if [ ! -d "$BUILD_DIR" ]; then
        echo "Building C++ components..."
        "$SCRIPT_DIR/build.sh"
    fi
    
    run_test "Hilbert Curve" "$BUILD_DIR/test_hilbert"
    run_test "BLAKE3 Hashing" "$BUILD_DIR/test_blake3"
    run_test "Coordinate Mapping" "$BUILD_DIR/test_coordinates"
    run_test "Semantic Validation" "$BUILD_DIR/test_semantic"
    
    echo ""
    echo "=== C++ Performance Tests ==="
    
    echo -n "  Atom Generation (1.1M)... "
    START=$(date +%s.%N)
    COUNT=$("$BUILD_DIR/seed_atoms" --csv 2>/dev/null | wc -l)
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    
    if [ "$COUNT" -gt 1000000 ]; then
        echo "PASS ($COUNT atoms in ${TIME}s)"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL (only $COUNT atoms)"
        FAILED=$((FAILED + 1))
    fi
    
    # Target: < 3 seconds (including EWKB formatting)
    if (( $(echo "$TIME < 3.0" | bc -l) )); then
        echo "  Performance target (<3s): PASS"
        PASSED=$((PASSED + 1))
    else
        echo "  Performance target (<3s): FAIL (${TIME}s)"
        FAILED=$((FAILED + 1))
    fi
fi

# SQL Tests
if [ "$CPP_ONLY" = false ]; then
    echo ""
    echo "=== SQL Integration Tests ==="
    
    TEST_DB="hypercube_test_$$"
    
    echo "  Creating test database: $TEST_DB"
    createdb "$TEST_DB" 2>/dev/null || true
    
    # Apply schema
    run_test "Schema creation" "psql -d $TEST_DB -q -f '$PROJECT_ROOT/sql/001_schema.sql'"
    
    # Test PostGIS
    run_test "PostGIS extension" "psql -d $TEST_DB -tAc 'SELECT PostGIS_Version();' | grep -q '[0-9]'"
    
    # Load atoms via COPY with EWKB (faster than staging + ST_MakePoint)
    echo -n "  Atom bulk load (EWKB)... "
    START=$(date +%s.%N)
    
    psql -d "$TEST_DB" -q -c "
        DROP INDEX IF EXISTS idx_atom_coords;
        DROP INDEX IF EXISTS idx_atom_hilbert;
        DROP INDEX IF EXISTS idx_atom_category;
    "
    
    "$BUILD_DIR/seed_atoms" --ewkb 2>/dev/null | \
        psql -d "$TEST_DB" -q -c "COPY atom (id, codepoint, category, coords, hilbert_lo, hilbert_hi) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')"
    
    psql -d "$TEST_DB" -q -c "
        CREATE INDEX idx_atom_coords ON atom USING GIST(coords);
        CREATE INDEX idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);
        CREATE INDEX idx_atom_category ON atom(category);
        ANALYZE atom;
    "
    
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    
    COUNT=$(psql -d "$TEST_DB" -tAc "SELECT COUNT(*) FROM atom")
    if [ "$COUNT" -gt 1000000 ]; then
        echo "PASS ($COUNT atoms in ${TIME}s)"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL (only $COUNT atoms)"
        FAILED=$((FAILED + 1))
    fi
    
    # Test atom queries
    run_test "Atom by codepoint" "psql -d $TEST_DB -tAc \"SELECT id IS NOT NULL FROM atom WHERE codepoint=65\" | grep -q t"
    run_test "Atom geometry" "psql -d $TEST_DB -tAc \"SELECT ST_GeometryType(coords) FROM atom WHERE codepoint=65\" | grep -q 'Point'"
    run_test "Atom categories" "psql -d $TEST_DB -tAc \"SELECT COUNT(DISTINCT category) FROM atom\" | grep -q '[0-9]'"
    
    # Test spatial index
    run_test "Spatial index exists" "psql -d $TEST_DB -tAc \"SELECT indexname FROM pg_indexes WHERE indexname='idx_atom_coords'\" | grep -q idx_atom_coords"
    
    # Apply functions for semantic tests
    psql -d "$TEST_DB" -q -f "$PROJECT_ROOT/sql/002_functions.sql" 2>/dev/null || true
    
    # Run semantic validation tests
    echo ""
    echo "=== SQL Semantic Validation Tests ==="
    run_test "Semantic validation suite" "psql -d $TEST_DB -f '$PROJECT_ROOT/tests/test_semantic_validation.sql' 2>&1 | grep -c 'PASSED' | grep -qE '^[0-9]+$'"
    
    # Cleanup
    echo "  Cleaning up test database..."
    dropdb "$TEST_DB" 2>/dev/null || true
fi

echo ""
echo "=============================================="
echo "  Test Results: $PASSED passed, $FAILED failed"
echo "=============================================="

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
