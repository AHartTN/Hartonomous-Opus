#!/bin/bash
# Hartonomous Hypercube - Validation Script (Linux)
# ============================================================================
# Validates database state and runs test suite.
# SAFE: Does NOT modify data. Only reads and reports.
#
# Usage:
#   ./validate.sh             # Quick validation + test suite
#   ./validate.sh --full      # Full validation with benchmarks
#   ./validate.sh --quick     # Just show database state
#   ./validate.sh --skip-tests # Validate state but skip test suite
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

FULL=false
QUICK=false
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --full) FULL=true; shift ;;
        --quick) QUICK=true; shift ;;
        --skip-tests) SKIP_TESTS=true; shift ;;
        *) shift ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DGRAY='\033[1;30m'
NC='\033[0m'

VALIDATION_PASSED=true

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Hartonomous Hypercube - Validation                       ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# STEP 1: CONNECTION TEST
# ============================================================================
echo -n "[1/4] Testing PostgreSQL connection..."
if ! hc_psql_admin -c "SELECT 1" &>/dev/null; then
    echo -e " ${RED}FAILED${NC}"
    echo "  Cannot connect to PostgreSQL"
    exit 1
fi
echo -e " ${GREEN}OK${NC}"

# ============================================================================
# STEP 2: DATABASE EXISTS
# ============================================================================
echo -n "[2/4] Checking database '$HC_DB_NAME'..."
if ! hc_psql_admin -tAc "SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'" 2>/dev/null | grep -q 1; then
    echo -e " ${RED}NOT FOUND${NC}"
    echo ""
    echo "Database does not exist. Run: ./scripts/linux/setup-db.sh"
    exit 1
fi
echo -e " ${GREEN}EXISTS${NC}"

# ============================================================================
# STEP 3: DATA STATE
# ============================================================================
echo -e "[3/4] Reading database state..."
echo ""

ATOM_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]' || echo "0")
COMP_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM composition" 2>/dev/null | tr -d '[:space:]' || echo "0")
REL_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM relation" 2>/dev/null | tr -d '[:space:]' || echo "0")
CHILD_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM composition_child" 2>/dev/null | tr -d '[:space:]' || echo "0")
CENTROID_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM composition WHERE centroid IS NOT NULL" 2>/dev/null | tr -d '[:space:]' || echo "0")

printf "  ┌─────────────────────────────────────────┐\n"
printf "  │  TABLE               │  COUNT           │\n"
printf "  ├─────────────────────────────────────────┤\n"

if [ "$ATOM_COUNT" -ge 1100000 ]; then
    printf "  │  Atoms               │%12s  │\n" "$ATOM_COUNT"
else
    printf "  │  Atoms               │${YELLOW}%12s${NC}  │\n" "$ATOM_COUNT"
fi

if [ "$COMP_COUNT" -gt 0 ]; then
    printf "  │  Compositions        │${GREEN}%12s${NC}  │\n" "$COMP_COUNT"
else
    printf "  │  Compositions        │${YELLOW}%12s${NC}  │\n" "$COMP_COUNT"
fi

if [ "$REL_COUNT" -gt 0 ]; then
    printf "  │  Relations           │${GREEN}%12s${NC}  │\n" "$REL_COUNT"
else
    printf "  │  Relations           │${YELLOW}%12s${NC}  │\n" "$REL_COUNT"
fi

printf "  │  Composition Children│${CYAN}%12s${NC}  │\n" "$CHILD_COUNT"
printf "  │  With Centroids      │${CYAN}%12s${NC}  │\n" "$CENTROID_COUNT"
printf "  └─────────────────────────────────────────┘\n"
echo ""

# ============================================================================
# STEP 4: TEST SUITE
# ============================================================================
if [ "$QUICK" = true ]; then
    echo -e "[4/4] Tests skipped (--quick)"
elif [ "$SKIP_TESTS" = true ]; then
    echo -e "[4/4] Tests skipped (--skip-tests)"
else
    echo -e "[4/4] Running test suite..."
    echo ""
    
    if [ "$FULL" = true ]; then
        "$SCRIPT_DIR/test.sh" || VALIDATION_PASSED=false
    else
        "$SCRIPT_DIR/test.sh" --quick || VALIDATION_PASSED=false
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  VALIDATION PASSED${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    exit 0
else
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  VALIDATION FAILED${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    exit 1
fi
