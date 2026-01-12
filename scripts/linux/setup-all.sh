#!/bin/bash
# Hartonomous Hypercube - Full Setup Pipeline (Linux)
# ============================================================================
# SAFE BY DEFAULT: Only creates/applies schema if missing. Does NOT destroy data.
# Use --reset flag ONLY for true greenfield setup.
#
# Pipeline:
#   1. Clean build artifacts (optional)
#   2. Build/Compile all C/C++
#   3. Install extensions to PostgreSQL
#   4. Create database + apply schema (idempotent - safe to re-run)
#   5. Seed Unicode atoms (if not already seeded)
#   6. Ingest test content (Moby Dick)
#   7. Run test suite
#
# Usage:
#   ./setup-all.sh                 # Safe: preserves existing data
#   ./setup-all.sh --reset         # DESTRUCTIVE: drops database first
#   ./setup-all.sh --skip-clean    # Keep build artifacts
#   ./setup-all.sh --skip-build    # Skip C++ compilation
#   ./setup-all.sh --skip-ingest   # Skip data ingestion
#   ./setup-all.sh --skip-tests    # Skip test suite
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

RESET=false
SKIP_CLEAN=false
SKIP_BUILD=false
SKIP_INGEST=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --reset) RESET=true; shift ;;
        --skip-clean) SKIP_CLEAN=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-ingest) SKIP_INGEST=true; shift ;;
        --skip-tests) SKIP_TESTS=true; shift ;;
        *) shift ;;
    esac
done

START_TIME=$(date +%s)

echo ""
echo "Hartonomous Hypercube - Full Setup Pipeline"
echo ""
echo "  Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo "  Project:  $HC_PROJECT_ROOT"
echo ""

# ============================================================================
# STEP 1: CLEAN
# ============================================================================
if [ "$SKIP_CLEAN" = false ]; then
    echo "Step 1/5: Cleaning Build Artifacts"

    "$SCRIPT_DIR/clean.sh"
    echo ""
else
    echo "Skipping clean (--skip-clean)"
fi

# ============================================================================
# STEP 2: BUILD
# ============================================================================
if [ "$SKIP_BUILD" = false ]; then
    echo "Step 2/5: Building C/C++ Components"

    "$SCRIPT_DIR/build.sh"
    echo ""
else
    echo "Skipping build (--skip-build)"
fi

# ============================================================================
# STEP 3: DATABASE SETUP (idempotent unless --reset specified)
# ============================================================================
if [ "$RESET" = true ]; then
    echo "Step 3/5: Database Setup (Greenfield - Destructive)"
    echo ""
    echo "!!! --reset flag specified. Database will be dropped and recreated !!!"
    echo ""
    "$SCRIPT_DIR/setup-db.sh" --reset
else
    echo "Step 3/5: Database Setup (Safe - Preserving Data)"
    "$SCRIPT_DIR/setup-db.sh"
fi
echo ""

# ============================================================================
# STEP 4: INGEST TEST DATA
# ============================================================================
if [ "$SKIP_INGEST" = false ]; then
    echo "Step 4/5: Ingesting Test Data"

    "$SCRIPT_DIR/ingest-testdata.sh" || {
        echo "Warning: Ingestion had issues (continuing anyway)"
    }
    echo ""
else
    echo "Skipping ingest (--skip-ingest)"
fi

# ============================================================================
# STEP 5: RUN FULL TEST SUITE
# ============================================================================
if [ "$SKIP_TESTS" = false ]; then
    echo "Step 5/5: Running Test Suite"

    "$SCRIPT_DIR/test.sh"
    TEST_EXIT=$?
    echo ""
else
    echo "Skipping tests (--skip-tests)"
    TEST_EXIT=0
fi

# ============================================================================
# SUMMARY
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "Pipeline Complete"
echo ""
echo "  Duration: ${DURATION} seconds"
echo ""

# Final stats
echo "  Final Database State:"
FINAL_ATOMS=$(hc_psql -tAc "SELECT atoms FROM db_stats()" 2>/dev/null | tr -d '[:space:]' || echo "0")
FINAL_COMPS=$(hc_psql -tAc "SELECT compositions FROM db_stats()" 2>/dev/null | tr -d '[:space:]' || echo "0")
FINAL_RELS=$(hc_psql -tAc "SELECT relations FROM db_stats()" 2>/dev/null | tr -d '[:space:]' || echo "0")

echo "    Atoms:        $FINAL_ATOMS"
echo "    Compositions: $FINAL_COMPS"
echo "    Relations:    $FINAL_RELS"

echo ""
if [ $TEST_EXIT -eq 0 ]; then
    echo "  All systems operational!"
else
    echo "  Some tests failed (exit code: $TEST_EXIT)"
fi
echo ""

exit $TEST_EXIT
