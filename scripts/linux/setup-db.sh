#!/bin/bash
# Hartonomous Hypercube - Database Setup (Linux)
# ============================================================================
# SAFE BY DEFAULT: Creates database and schema if they don't exist.
# Does NOT drop, truncate, or modify existing data unless explicitly requested.
#
# Usage:
#   ./setup-db.sh            # Safe: creates if missing, skips if exists
#   ./setup-db.sh --reset    # DESTRUCTIVE: drops and recreates database
#   ./setup-db.sh --seed     # Only seed atoms (skip schema)
#   ./setup-db.sh --force    # Force re-seed atoms even if populated
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

RESET=false
SEED_ONLY=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --reset) RESET=true; shift ;;
        --seed) SEED_ONLY=true; shift ;;
        --force) FORCE=true; shift ;;
        *) shift ;;
    esac
done

echo ""
echo "=== Hypercube Database Setup ==="
echo "  Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo "  User: $HC_DB_USER"
echo ""

# ============================================================================
# CONNECTION TEST
# ============================================================================
echo -n "[1/5] Testing PostgreSQL connection..."
if ! hc_psql_admin -c "SELECT 1" &>/dev/null; then
    echo " FAILED"
    echo ""
    echo "Cannot connect to PostgreSQL."
    echo "Check:"
    echo "  1. PostgreSQL is running"
    echo "  2. Credentials in scripts/config.env are correct"
    echo "  3. User '$HC_DB_USER' exists and has permissions"
    exit 1
fi
echo " OK"

# ============================================================================
# RESET (DESTRUCTIVE - only if explicitly requested)
# ============================================================================
if [ "$RESET" = true ]; then
    echo ""
    echo "!!! DESTRUCTIVE OPERATION !!!"
    echo ""
    echo "The --reset flag will DROP the database and ALL data!"
    echo ""
    
    # Check for existing data
    if hc_psql_admin -tAc "SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'" 2>/dev/null | grep -q 1; then
        COMP_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM composition" 2>/dev/null | tr -d '[:space:]' || echo "0")
        REL_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM relation" 2>/dev/null | tr -d '[:space:]' || echo "0")
        if [ "$COMP_COUNT" -gt 0 ] || [ "$REL_COUNT" -gt 0 ]; then
            echo "Current database contains:"
            echo "  - $COMP_COUNT compositions"
            echo "  - $REL_COUNT relations"
            echo ""
            echo "This data CANNOT be recovered after reset."
            echo ""
        fi
    fi
    
    echo "[RESET] Dropping database $HC_DB_NAME..."
    hc_psql_admin -c "DROP DATABASE IF EXISTS $HC_DB_NAME" 2>/dev/null || true
    echo "[RESET] Database dropped"
fi

# ============================================================================
# DATABASE CREATION (idempotent)
# ============================================================================
if [ "$SEED_ONLY" = false ]; then
    echo -n "[2/5] Checking database..."
    DB_EXISTS=$(hc_psql_admin -tAc "SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'" | tr -d '[:space:]')
    
    if [ "$DB_EXISTS" != "1" ]; then
        echo -n " creating..."
        hc_psql_admin -c "CREATE DATABASE $HC_DB_NAME" >/dev/null
        echo " CREATED"
    else
        echo " exists"
    fi

    # ========================================================================
    # SCHEMA APPLICATION (idempotent - uses CREATE IF NOT EXISTS)
    # ========================================================================
    echo "[3/5] Applying schema..."
    for sqlfile in "$HC_PROJECT_ROOT"/sql/*.sql; do
        [ -f "$sqlfile" ] || continue
        [[ "$sqlfile" == *"archive"* ]] && continue
        filename=$(basename "$sqlfile")
        echo -n "      $filename..."
        if hc_psql -v ON_ERROR_STOP=1 -f "$sqlfile" >/dev/null 2>&1; then
            echo " OK"
        else
            echo " FAILED"
            exit 1
        fi
    done

    # ========================================================================
    # C++ EXTENSIONS (idempotent)
    # ========================================================================
    echo "[4/5] Loading extensions..."
    for ext in hypercube hypercube_ops semantic_ops embedding_ops generative; do
        echo -n "      $ext..."
        if hc_psql -c "CREATE EXTENSION IF NOT EXISTS $ext;" >/dev/null 2>&1; then
            echo " OK"
        else
            echo " not available"
        fi
    done
else
    echo "[2/5] Skipping database check (--seed)" 
    echo "[3/5] Skipping schema (--seed)"
    echo "[4/5] Skipping extensions (--seed)"
fi

# ============================================================================
# ATOM SEEDING (idempotent - checks count first)
# ============================================================================
echo -n "[5/5] Checking atoms..."
ATOM_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]' || echo "0")

if [ "$ATOM_COUNT" -ge 1100000 ] && [ "$FORCE" = false ]; then
    echo " $ATOM_COUNT atoms (already seeded)"
else
    if [ "$ATOM_COUNT" -ge 1100000 ] && [ "$FORCE" = true ]; then
        echo " re-seeding (forced)..."
    else
        echo " seeding Unicode atoms..."
    fi

    echo ""
    SEEDER="$HC_BUILD_DIR/seed_atoms_parallel"
    if [ -x "$SEEDER" ]; then
        echo "Using optimized parallel atom seeder: $SEEDER"
        "$SEEDER" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -p "$HC_DB_PORT"
        if [ $? -ne 0 ]; then
            echo "Standalone seeder failed"
            echo "Database setup failed!"
            exit 1
        fi
    else
        echo "Standalone seeder not found, using PostgreSQL extension function..."
        echo "Running SQL: SELECT seed_atoms();"
        if ! hc_psql -tAc "SELECT seed_atoms();" >/dev/null 2>&1; then
            echo "Extension seeder failed"
            echo "Database setup failed!"
            exit 1
        fi
        echo "Extension seeder completed successfully"
    fi

    NEW_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom" | tr -d '[:space:]')
    echo "      Seeded $NEW_COUNT atoms"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=== Database Ready ==="
echo ""

FINAL_ATOMS=$(hc_psql -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]' || echo "0")
FINAL_COMPS=$(hc_psql -tAc "SELECT COUNT(*) FROM composition" 2>/dev/null | tr -d '[:space:]' || echo "0")
FINAL_RELS=$(hc_psql -tAc "SELECT COUNT(*) FROM relation" 2>/dev/null | tr -d '[:space:]' || echo "0")

echo "  Atoms:        $FINAL_ATOMS"
echo "  Compositions: $FINAL_COMPS"
echo "  Relations:    $FINAL_RELS"
echo ""
