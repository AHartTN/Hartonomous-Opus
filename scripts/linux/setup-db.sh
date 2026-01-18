#!/bin/bash
# ============================================================================
# Hartonomous Hypercube - Database Setup (Linux)
# ============================================================================
# Idempotent: Creates/fixes whatever is missing. Run as many times as needed.
#
# Usage:
#   ./scripts/linux/setup-db.sh          # Set up everything that's missing
#   ./scripts/linux/setup-db.sh --reset  # DESTRUCTIVE: drops and recreates
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

RESET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --reset|-r) RESET=true; shift ;;
        *) shift ;;
    esac
done

echo ""
echo "=== Hypercube Database Setup ==="
echo "  Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo "  User: $HC_DB_USER"
echo ""

export PGPASSWORD="$HC_DB_PASS"

# ============================================================================
# CONNECTION TEST
# ============================================================================
echo -n "[1/5] Testing PostgreSQL connection..."
if ! psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -tAc "SELECT 1" > /dev/null 2>&1; then
    echo " FAILED"
    echo "  Cannot connect to PostgreSQL"
    echo "  Check: server running, credentials in scripts/config.env, pg_hba.conf"
    exit 1
fi
echo " OK"

# ============================================================================
# RESET (DESTRUCTIVE)
# ============================================================================
if [ "$RESET" = true ]; then
    echo ""
    echo "!!! DESTRUCTIVE: Dropping database $HC_DB_NAME !!!"
    psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $HC_DB_NAME" > /dev/null 2>&1
    echo "  Database dropped"
    echo ""
fi

# ============================================================================
# DATABASE CREATION
# ============================================================================
echo -n "[2/5] Database..."
DB_EXISTS=$(psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'" 2>/dev/null | tr -d '[:space:]')

if [ "$DB_EXISTS" != "1" ]; then
    psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -c "CREATE DATABASE $HC_DB_NAME" > /dev/null 2>&1
    echo " created"
else
    echo " exists"
fi

# ============================================================================
# SCHEMA
# ============================================================================
echo -n "[3/5] Schema..."
cd "$HC_PROJECT_ROOT/sql"
psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -v ON_ERROR_STOP=1 -f "hypercube_schema.sql" > /dev/null 2>&1
echo " OK"

# ============================================================================
# EXTENSIONS
# ============================================================================
echo -n "[4/5] Extensions..."
EXTENSIONS=(hypercube hypercube_ops embedding_ops semantic_ops generative)
LOADED=0
FAILED=()

for ext in "${EXTENSIONS[@]}"; do
    if psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS $ext;" > /dev/null 2>&1; then
        ((LOADED++))
    else
        FAILED+=("$ext")
    fi
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo " $LOADED loaded"
else
    echo " $LOADED loaded, missing: ${FAILED[*]}"
    echo "  Run build.sh to compile and install extensions"
fi

# ============================================================================
# ATOM SEEDING
# ============================================================================
echo -n "[5/5] Atoms..."
ATOM_COUNT=$(psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]')
ATOM_COUNT=${ATOM_COUNT:-0}

if [ "$ATOM_COUNT" -ge 1100000 ]; then
    echo " $ATOM_COUNT (complete)"
else
    echo " $ATOM_COUNT (need ~1.1M, seeding...)"

    SEEDER="$HC_BIN_DIR/seed_atoms_parallel"
    SEEDER_OK=1

    if [ -x "$SEEDER" ]; then
        echo "  Using: $SEEDER"
        if "$SEEDER" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -p "$HC_DB_PORT"; then
            SEEDER_OK=0
        else
            echo "  Standalone seeder failed, trying extension..."
        fi
    fi

    if [ $SEEDER_OK -ne 0 ]; then
        echo "  Using: SELECT seed_atoms();"
        RESULT=$(psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -tAc "SELECT seed_atoms();" 2>&1)
        if [ $? -ne 0 ]; then
            echo "  Extension seeder failed: $RESULT"
            echo "  Ensure extensions are installed (run build.sh)"
            exit 1
        fi
    fi

    NEW_COUNT=$(psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]')
    if [ "$NEW_COUNT" -ge 1100000 ]; then
        echo "  Seeded $NEW_COUNT atoms"
    else
        echo "  WARNING: Only $NEW_COUNT atoms seeded (expected ~1.1M)"
        exit 1
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=== Database Ready ==="

STATS=$(psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -tAc "SELECT * FROM db_stats()" 2>/dev/null)
if [ -n "$STATS" ]; then
    IFS='|' read -ra S <<< "$STATS"
    echo "  Atoms: ${S[0]}  Compositions: ${S[1]}  Relations: ${S[3]}"
fi
echo ""
