#!/bin/bash
# Hartonomous Hypercube - Full Setup Pipeline (Linux)
# ============================================================================
# ORCHESTRATES THE COMPLETE SETUP:
#
# 1. Clean build artifacts
# 2. Build C/C++ components with Intel MKL
# 3. Drop existing database (destructive reset)
# 4. Create fresh database
# 5. Load schema, functions, procedures, extensions
# 6. Seed 1.1M+ Unicode atoms
# 7. Build indexes
# 8. Validate database integrity
#
# This is the "clean slate" setup - everything from scratch.
#
# Usage:
#   ./full-setup.sh                  # Complete setup from scratch
#   ./full-setup.sh --skip-build     # Skip C++ build (use existing)
#   ./full-setup.sh --skip-seed      # Skip atom seeding (already done)
#   ./full-setup.sh --verbose        # Show detailed output
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SKIP_BUILD=false
SKIP_SEED=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-seed) SKIP_SEED=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        *) shift ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           HARTONOMOUS HYPERCUBE - FULL SETUP PIPELINE            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo "User: $HC_DB_USER"
echo "Project: $HC_PROJECT_ROOT"
echo ""

# ============================================================================
# STEP 1: CLEAN
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ STEP 1/6: CLEANING BUILD ARTIFACTS                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
"$SCRIPT_DIR/clean.sh"
echo ""

# ============================================================================
# STEP 2: BUILD C/C++
# ============================================================================
if [ "$SKIP_BUILD" = true ]; then
    echo -e "${YELLOW}[STEP 2/6] Skipping build (--skip-build)${NC}"
else
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ STEP 2/6: BUILDING C/C++ COMPONENTS                             ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
    "$SCRIPT_DIR/build.sh"
fi
echo ""

# ============================================================================
# STEP 3: DROP DATABASE (DESTRUCTIVE!)
# ============================================================================
echo -e "${YELLOW}⚠️  WARNING: DESTRUCTIVE OPERATION ⚠️${NC}"
echo -e "${RED}This will DROP the database '$HC_DB_NAME' and ALL data!${NC}"
echo ""
echo -n "Type YES to continue: "
read -r CONFIRM
if [ "$CONFIRM" != "YES" ]; then
    echo "Cancelled by user"
    exit 1
fi
echo ""

echo -e "${CYAN}Dropping database $HC_DB_NAME...${NC}"
PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres \
    -c "DROP DATABASE IF EXISTS $HC_DB_NAME;" 2>/dev/null || true
echo -e "${GREEN}✓ Database dropped${NC}"
echo ""

# ============================================================================
# STEP 4: CREATE DATABASE & LOAD SCHEMA
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ STEP 3/6: CREATING DATABASE & LOADING SCHEMA                    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"

echo "Creating database..."
PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres \
    -c "CREATE DATABASE $HC_DB_NAME;" >/dev/null
echo -e "${GREEN}✓ Database created${NC}"

echo "Applying schema (35+ SQL files)..."
cd "$HC_PROJECT_ROOT/sql"
if PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" \
    -v ON_ERROR_STOP=1 -f hypercube_schema.sql >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Schema loaded (tables, indexes, 813 functions)${NC}"
else
    echo -e "${RED}✗ Schema loading failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# STEP 5: SEED ATOMS
# ============================================================================
if [ "$SKIP_SEED" = true ]; then
    echo -e "${YELLOW}[STEP 4/6] Skipping atom seeding (--skip-seed)${NC}"
else
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ STEP 4/6: SEEDING 1.1M UNICODE ATOMS                            ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
    
    SEEDER="$HC_BUILD_DIR/seed_atoms_parallel"
    if [ ! -x "$SEEDER" ]; then
        echo -e "${RED}✗ seed_atoms_parallel not found at $SEEDER${NC}"
        exit 1
    fi
    
    PGPASSWORD="$HC_DB_PASS" "$SEEDER" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -p "$HC_DB_PORT"
fi
echo ""

# ============================================================================
# STEP 6: VALIDATION
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ STEP 5/6: VALIDATING DATABASE                                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"

ATOM_COUNT=$(PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" \
    -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]')
FUNC_COUNT=$(PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" \
    -tAc "SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema='public'" 2>/dev/null | tr -d '[:space:]')

echo "Atoms:     $ATOM_COUNT (expect ~1,114,112)"
echo "Functions: $FUNC_COUNT (expect ~813)"

if [ "$ATOM_COUNT" -lt 1000000 ]; then
    echo -e "${YELLOW}⚠ Warning: Low atom count${NC}"
else
    echo -e "${GREEN}✓ Atoms properly seeded${NC}"
fi

if [ "$FUNC_COUNT" -lt 800 ]; then
    echo -e "${RED}✗ Functions not loaded${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Functions loaded${NC}"
fi
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ STEP 6/6: SETUP COMPLETE                                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ Full setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  ./ingest-testdata.sh   # Ingest test data (model + Moby Dick)"
echo "  ./validate.sh          # Run full validation suite"
echo "  ./test.sh              # Run all tests"
echo ""
