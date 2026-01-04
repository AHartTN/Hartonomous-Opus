#!/bin/bash
# Hartonomous Hypercube - Database Setup (Linux)
# Creates database, applies schema, loads extensions, seeds Unicode atoms
# Fully idempotent - safe to run multiple times
# Usage: ./scripts/linux/setup-db.sh [--reset]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

RESET=false
if [ "$1" == "--reset" ]; then
    RESET=true
fi

echo "=== Hypercube Database Setup ==="
echo "Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"

# Test connection
echo -e "\nTesting connection..."
if ! hc_psql_admin -c "SELECT 1" &>/dev/null; then
    echo "Cannot connect to PostgreSQL"
    echo "Ensure PostgreSQL is running and credentials are correct in scripts/config.env"
    exit 1
fi
echo "Connected"

# Reset if requested
if [ "$RESET" = true ]; then
    echo -e "\nDropping existing database..."
    hc_psql_admin -c "DROP DATABASE IF EXISTS $HC_DB_NAME"
fi

# Check if database exists
DB_EXISTS=$(hc_psql_admin -tAc "SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'" | tr -d '[:space:]')

if [ "$DB_EXISTS" != "1" ]; then
    echo -e "\nCreating database $HC_DB_NAME..."
    hc_psql_admin -c "CREATE DATABASE $HC_DB_NAME"
fi
echo "Database exists"

# Load C++ extensions FIRST (some SQL depends on them)
echo -e "\nLoading C++ extensions..."
if output=$(hc_psql -c "CREATE EXTENSION IF NOT EXISTS hypercube CASCADE; CREATE EXTENSION IF NOT EXISTS semantic_ops CASCADE; CREATE EXTENSION IF NOT EXISTS hypercube_ops CASCADE;" 2>&1); then
    echo "C++ extensions loaded"
else
    echo "Warning: C++ extensions not available (run build.sh first)"
    echo "$output" | grep -E "ERROR|FATAL" | head -3
    echo "  Continuing without extensions - some functions may be slower"
fi

# Apply ALL SQL schema files in order
echo -e "\nApplying schema files..."
for sqlfile in "$HC_PROJECT_ROOT"/sql/*.sql; do
    [ -f "$sqlfile" ] || continue
    filename=$(basename "$sqlfile")
    echo -n "  $filename..."
    if output=$(hc_psql -f "$sqlfile" 2>&1); then
        echo " OK"
    else
        echo " FAILED"
        echo "$output" | grep -E "ERROR|FATAL" | head -3
        exit 1
    fi
done

# Check atom count using canonical function (4-table schema)
ATOM_COUNT=$(hc_psql -tAc "SELECT atoms FROM db_stats()" 2>/dev/null | tr -d '[:space:]')
# Fallback if function doesn't exist yet - atom table = leaf atoms in 4-table schema
if [ -z "$ATOM_COUNT" ]; then
    ATOM_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom" | tr -d '[:space:]')
fi
echo -e "\nAtom count: $ATOM_COUNT"

if [ "$ATOM_COUNT" -lt 1100000 ]; then
    echo -e "\nSeeding Unicode atoms (this takes ~30 seconds)..."
    
    SEEDER="$HC_BUILD_DIR/seed_atoms_parallel"
    if [ ! -x "$SEEDER" ]; then
        echo "Seeder not found. Run build.sh first."
        exit 1
    fi
    
    "$SEEDER" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -p "$HC_DB_PORT"
    echo "Atoms seeded"
else
    echo "Atoms already seeded"
fi

# Run manifold projection if embeddings exist but centroids are missing
echo -e "\nChecking for manifold projection..."
NEEDS_PROJECTION=$(hc_psql -tAc "SELECT COUNT(*) FROM shape s JOIN composition c ON c.id = s.entity_id WHERE c.centroid IS NULL AND s.dim_count = 384" | tr -d '[:space:]')

if [ "$NEEDS_PROJECTION" -gt 0 ]; then
    echo "  $NEEDS_PROJECTION compositions need projection..."
    
    PROJECTOR="$HC_BUILD_DIR/manifold_project"
    if [ -x "$PROJECTOR" ]; then
        "$PROJECTOR" --batch 5000
        echo "Manifold projection complete"
    else
        echo "  Projector not found, run: SELECT project_all_embeddings() in psql"
    fi
else
    echo "  All embeddings already projected"
fi

echo -e "\n=== Setup Complete ==="

# Show stats using canonical function
echo -e "\nDatabase Statistics:"
hc_psql -c "SELECT * FROM db_stats()" 2>/dev/null || \
hc_psql -c "SELECT * FROM atom_stats" 2>/dev/null || \
echo "(stats function not available)"
