#!/bin/bash
# Hartonomous Hypercube - Database Setup (Linux)
# Creates database, applies schema, seeds Unicode atoms
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

# Apply schema
echo -e "\nApplying unified schema..."
hc_psql -f "$HC_PROJECT_ROOT/sql/011_unified_atom.sql"
echo "Schema applied"

# Check atom count
ATOM_COUNT=$(hc_psql -tAc "SELECT COUNT(*) FROM atom WHERE depth = 0" | tr -d '[:space:]')
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

echo -e "\n=== Setup Complete ==="

# Show stats
echo -e "\nDatabase Statistics:"
hc_psql -c "SELECT * FROM atom_stats"
