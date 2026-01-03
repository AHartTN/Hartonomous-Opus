#!/bin/bash
# Hartonomous Hypercube - Full Setup Pipeline (Linux)
# ============================================================================
# Runs the complete setup from clean slate to working LLM-like system:
#   1. Clean build artifacts
#   2. Build/Compile all C/C++
#   3. Install extensions to PostgreSQL
#   4. Drop database (greenfield)
#   5. Create database + schema + extensions
#   6. Seed Unicode atoms
#   7. Ingest embedding model (MiniLM)
#   8. Ingest test content (Moby Dick + images + audio)
#   9. Run full test suite including AI/ML operations
#
# Usage: ./scripts/linux/setup-all.sh [--skip-clean] [--skip-build] [--skip-tests]
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SKIP_CLEAN=false
SKIP_BUILD=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-clean) SKIP_CLEAN=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-tests) SKIP_TESTS=true; shift ;;
        *) shift ;;
    esac
done

START_TIME=$(date +%s)

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       HARTONOMOUS HYPERCUBE - FULL SETUP PIPELINE                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo "  Project:  $HC_PROJECT_ROOT"
echo ""

# ============================================================================
# STEP 1: CLEAN
# ============================================================================
if [ "$SKIP_CLEAN" = false ]; then
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│ STEP 1/5: CLEANING BUILD ARTIFACTS                               │"
    echo "└──────────────────────────────────────────────────────────────────┘"
    
    "$SCRIPT_DIR/clean.sh"
    echo ""
else
    echo "── Skipping clean (--skip-clean) ──"
fi

# ============================================================================
# STEP 2: BUILD
# ============================================================================
if [ "$SKIP_BUILD" = false ]; then
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│ STEP 2/5: BUILDING C/C++ COMPONENTS                              │"
    echo "└──────────────────────────────────────────────────────────────────┘"
    
    "$SCRIPT_DIR/build.sh"
    echo ""
else
    echo "── Skipping build (--skip-build) ──"
fi

# ============================================================================
# STEP 3: DATABASE SETUP (DROP + CREATE + SCHEMA + SEED)
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│ STEP 3/5: DATABASE SETUP (GREENFIELD)                            │"
echo "└──────────────────────────────────────────────────────────────────┘"

"$SCRIPT_DIR/setup-db.sh" --reset
echo ""

# ============================================================================
# STEP 4: INGEST ALL TEST DATA
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│ STEP 4/5: INGESTING TEST DATA                                    │"
echo "└──────────────────────────────────────────────────────────────────┘"

"$SCRIPT_DIR/ingest-testdata.sh" || {
    echo "Warning: Ingestion had issues (continuing anyway)"
}
echo ""

# ============================================================================
# STEP 5: RUN FULL TEST SUITE
# ============================================================================
if [ "$SKIP_TESTS" = false ]; then
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│ STEP 5/5: RUNNING TEST SUITE                                     │"
    echo "└──────────────────────────────────────────────────────────────────┘"
    
    "$SCRIPT_DIR/test.sh"
    TEST_EXIT=$?
    echo ""
else
    echo "── Skipping tests (--skip-tests) ──"
    TEST_EXIT=0
fi

# ============================================================================
# SUMMARY
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      PIPELINE COMPLETE                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Duration: ${DURATION} seconds"
echo ""

# Final stats
echo "  Final Database State:"
hc_psql -c "
SELECT 
    COUNT(*) FILTER (WHERE depth = 0) as \"Leaf Atoms\",
    COUNT(*) FILTER (WHERE depth > 0) as \"Compositions\",
    COUNT(*) FILTER (WHERE depth = 1 AND atom_count = 2) as \"Semantic Edges\",
    MAX(depth) as \"Max Depth\",
    pg_size_pretty(pg_total_relation_size('atom')) as \"Total Size\"
FROM atom;
"

echo ""
if [ $TEST_EXIT -eq 0 ]; then
    echo "  ✓ All systems operational!"
else
    echo "  ⚠ Some tests failed (exit code: $TEST_EXIT)"
fi
echo ""

exit $TEST_EXIT
