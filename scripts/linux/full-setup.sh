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

echo ""
echo "Hartonomous Hypercube - Full Setup Pipeline"
echo ""
echo "Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo "User: $HC_DB_USER"
echo "Project: $HC_PROJECT_ROOT"
echo ""

# ============================================================================
# STEP 1: CLEAN
# ============================================================================
echo "Step 1/6: Cleaning Build Artifacts"
"$SCRIPT_DIR/clean.sh"
echo ""

# ============================================================================
# STEP 2: BUILD C/C++
# ============================================================================
if [ "$SKIP_BUILD" = true ]; then
    echo "[STEP 2/6] Skipping build (--skip-build)"
else
    echo "Step 2/6: Building C/C++ Components"
    "$SCRIPT_DIR/build.sh"
fi
echo ""

# ============================================================================
# STEP 3: DROP DATABASE (DESTRUCTIVE!)
# ============================================================================
echo "WARNING: DESTRUCTIVE OPERATION"
echo "This will DROP the database '$HC_DB_NAME' and ALL data!"
echo ""
echo -n "Type YES to continue: "
read -r CONFIRM
if [ "$CONFIRM" != "YES" ]; then
    echo "Cancelled by user"
    exit 1
fi
echo ""

echo "Dropping database $HC_DB_NAME..."
PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres \
    -c "DROP DATABASE IF EXISTS $HC_DB_NAME;" 2>/dev/null || true
echo "Database dropped"
echo ""

# ============================================================================
# STEP 4: CREATE DATABASE & LOAD SCHEMA
# ============================================================================
echo "Step 3/6: Creating Database & Loading Schema"

echo "Creating database..."
PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres \
    -c "CREATE DATABASE $HC_DB_NAME;" >/dev/null
echo "Database created"

echo "Applying schema (35+ SQL files)..."
cd "$HC_PROJECT_ROOT/sql"
if PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" \
    -v ON_ERROR_STOP=1 -f hypercube_schema.sql >/dev/null 2>&1; then
    echo "Schema loaded (tables, indexes, 813 functions)"
else
    echo "Schema loading failed"
    exit 1
fi
echo ""

# ============================================================================
# STEP 5: SEED ATOMS
# ============================================================================
if [ "$SKIP_SEED" = true ]; then
    echo "[STEP 4/6] Skipping atom seeding (--skip-seed)"
else
    echo "Step 4/6: Seeding 1.1M Unicode Atoms"

    SEEDER="$HC_BUILD_DIR/seed_atoms_parallel"
    if [ ! -x "$SEEDER" ]; then
        echo "seed_atoms_parallel not found at $SEEDER"
        exit 1
    fi

    PGPASSWORD="$HC_DB_PASS" "$SEEDER" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -p "$HC_DB_PORT"
fi
echo ""

# ============================================================================
# STEP 6: VALIDATION
# ============================================================================
echo "Step 5/6: Validating Database"

ATOM_COUNT=$(PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" \
    -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]')
FUNC_COUNT=$(PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" \
    -tAc "SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema='public'" 2>/dev/null | tr -d '[:space:]')

echo "Atoms:     $ATOM_COUNT (expect ~1,114,112)"
echo "Functions: $FUNC_COUNT (expect ~813)"

if [ "$ATOM_COUNT" -lt 1000000 ]; then
    echo "Warning: Low atom count"
else
    echo "Atoms properly seeded"
fi

if [ "$FUNC_COUNT" -lt 800 ]; then
    echo "Functions not loaded"
    exit 1
else
    echo "Functions loaded"
fi
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "Step 6/6: Setup Complete"
echo ""
echo "Full setup complete!"
echo ""
echo "Next steps:"
echo "  ./ingest-testdata.sh   # Ingest test data (model + Moby Dick)"
echo "  ./validate.sh          # Run full validation suite"
echo "  ./test.sh              # Run all tests"
echo ""
