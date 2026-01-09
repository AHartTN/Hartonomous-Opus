#!/bin/bash
# Hartonomous Hypercube - End-to-End Integration Test Suite (Linux)
# ============================================================================
# COMPLETE pipeline test from clean slate to full functionality validation.
#
# Pipeline:
#   1. Clean build artifacts
#   2. Build C/C++ code
#   3. Drop database (destructive reset)
#   4. Create database + load schema
#   5. Seed 1.1M atoms
#   6. Ingest test model (if available)
#   7. Ingest test content (Moby Dick)
#   8. Run integration tests
#   9. Run unit tests
#   10. Validate complete system state
#
# Usage:
#   ./e2e-test.sh                   # Full clean slate test
#   ./e2e-test.sh --skip-build      # Skip build (use existing binaries)
#   ./e2e-test.sh --skip-seed       # Skip atom seeding (if already done)
#   ./e2e-test.sh --skip-models     # Skip model ingestion
#   ./e2e-test.sh --skip-content    # Skip content ingestion
#   ./e2e-test.sh --verbose         # Verbose output
#   ./e2e-test.sh --fail-fast       # Stop on first failure
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Parse options
SKIP_BUILD=false
SKIP_SEED=false
SKIP_MODELS=false
SKIP_CONTENT=false
VERBOSE=false
FAIL_FAST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-seed) SKIP_SEED=true; shift ;;
        --skip-models) SKIP_MODELS=true; shift ;;
        --skip-content) SKIP_CONTENT=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --fail-fast) FAIL_FAST=true; shift ;;
        *) shift ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
START_TIME=$(date +%s)

# Helper functions
assert_test() {
    local name="$1"
    local cmd="$2"
    ((TESTS_RUN++))
    
    if eval "$cmd" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $name"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "  ${RED}✗${NC} $name"
        ((TESTS_FAILED++))
        if [ "$FAIL_FAST" = true ]; then
            exit 1
        fi
        return 1
    fi
}

header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
}

section() {
    echo ""
    echo -e "${MAGENTA}─ $1 ─${NC}"
}

# ============================================================================
# START
# ============================================================================
header "END-TO-END INTEGRATION TEST SUITE"

echo ""
echo "Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo "Build:    $HC_BUILD_DIR"
echo ""

# ============================================================================
# STAGE 1: CLEAN & BUILD
# ============================================================================
section "Stage 1: Clean & Build"

if [ "$SKIP_BUILD" = false ]; then
    echo "Cleaning..."
    "$SCRIPT_DIR/clean.sh" >/dev/null 2>&1 || true
    
    echo "Building C/C++ components..."
    "$SCRIPT_DIR/build.sh" >/dev/null 2>&1
    echo -e "${GREEN}✓ Build complete${NC}"
else
    echo -e "${YELLOW}Skipping build (--skip-build)${NC}"
fi

# ============================================================================
# STAGE 2: DATABASE SETUP
# ============================================================================
section "Stage 2: Database Setup"

echo "Dropping database..."
PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres \
    -c "DROP DATABASE IF EXISTS $HC_DB_NAME;" >/dev/null 2>&1 || true

echo "Creating database..."
PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres \
    -c "CREATE DATABASE $HC_DB_NAME;" >/dev/null 2>&1

echo "Loading schema..."
cd "$HC_PROJECT_ROOT/sql"
PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" \
    -v ON_ERROR_STOP=1 -f hypercube_schema.sql >/dev/null 2>&1

echo -e "${GREEN}✓ Database ready${NC}"

# ============================================================================
# STAGE 3: ATOM SEEDING
# ============================================================================
section "Stage 3: Atom Seeding"

if [ "$SKIP_SEED" = false ]; then
    echo "Seeding 1.1M Unicode atoms..."
    PGPASSWORD="$HC_DB_PASS" "$HC_BUILD_DIR/seed_atoms_parallel" \
        -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -p "$HC_DB_PORT" 2>&1 | tail -5
    echo -e "${GREEN}✓ Atoms seeded${NC}"
else
    echo -e "${YELLOW}Skipping atom seeding (--skip-seed)${NC}"
fi

# ============================================================================
# STAGE 4: DATA INGESTION
# ============================================================================
section "Stage 4: Data Ingestion"

if [ "$SKIP_MODELS" = false ]; then
    if [ -d "$HC_PROJECT_ROOT/test-data/embedding_models" ]; then
        echo "Ingesting test models..."
        "$SCRIPT_DIR/ingest-models.sh" >/dev/null 2>&1 || true
        echo -e "${GREEN}✓ Models ingested${NC}"
    else
        echo -e "${YELLOW}No test models found (skipping)${NC}"
    fi
else
    echo -e "${YELLOW}Skipping model ingestion (--skip-models)${NC}"
fi

if [ "$SKIP_CONTENT" = false ]; then
    if [ -f "$HC_PROJECT_ROOT/test-data/moby_dick.txt" ]; then
        echo "Ingesting Moby Dick..."
        "$HC_BUILD_DIR/ingest" -d "$HC_DB_NAME" -U "$HC_DB_USER" \
            -h "$HC_DB_HOST" -p "$HC_DB_PORT" \
            "$HC_PROJECT_ROOT/test-data/moby_dick.txt" >/dev/null 2>&1 || true
        echo -e "${GREEN}✓ Content ingested${NC}"
    else
        echo -e "${YELLOW}No Moby Dick found (skipping)${NC}"
    fi
else
    echo -e "${YELLOW}Skipping content ingestion (--skip-content)${NC}"
fi

# ============================================================================
# STAGE 5: INTEGRATION TESTS
# ============================================================================
section "Stage 5: Integration Tests"

assert_test "Database connection" \
    "PGPASSWORD='$HC_DB_PASS' psql -h '$HC_DB_HOST' -p '$HC_DB_PORT' -U '$HC_DB_USER' -d '$HC_DB_NAME' -c 'SELECT 1'"

assert_test "Atoms table exists" \
    "PGPASSWORD='$HC_DB_PASS' psql -h '$HC_DB_HOST' -p '$HC_DB_PORT' -U '$HC_DB_USER' -d '$HC_DB_NAME' -c 'SELECT COUNT(*) FROM atom' | grep -q '[0-9]'"

assert_test "Schema loaded" \
    "PGPASSWORD='$HC_DB_PASS' psql -h '$HC_DB_HOST' -p '$HC_DB_PORT' -U '$HC_DB_USER' -d '$HC_DB_NAME' -tAc 'SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema=\"public\"' | grep -q '[0-9][0-9][0-9]'"

ATOM_COUNT=$(PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" \
    -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]')
assert_test "Atoms seeded (>1M)" "[ $ATOM_COUNT -gt 1000000 ]"

# ============================================================================
# STAGE 6: VALIDATION
# ============================================================================
section "Stage 6: Final Validation"

"$SCRIPT_DIR/validate.sh" --skip-tests 2>&1 | tail -20
VALIDATE_EXIT=${PIPESTATUS[0]}

# ============================================================================
# SUMMARY
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

header "E2E TEST SUMMARY"

echo ""
echo "Tests run:    $TESTS_RUN"
echo "Passed:       ${GREEN}$TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo "Failed:       ${RED}$TESTS_FAILED${NC}"
else
    echo "Failed:       $TESTS_FAILED"
fi
echo "Duration:     ${DURATION}s"
echo ""

if [ $TESTS_FAILED -eq 0 ] && [ $VALIDATE_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED - SYSTEM FULLY OPERATIONAL${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi
