#!/bin/bash
set -e

# Resolve absolute path to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DB_NAME="hypercube"
DB_USER="postgres"
DB_HOST="localhost"

echo "========================================================"
echo "          HYPERCUBE DATABASE RESET (NUKE & PAVE)        "
echo "========================================================"

# Terminate existing connections
echo "[1/4] Terminating existing connections..."
psql -h $DB_HOST -U $DB_USER -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();" > /dev/null 2>&1 || true

# Drop database
echo "[2/4] Dropping database '$DB_NAME'..."
dropdb -h $DB_HOST -U $DB_USER --if-exists $DB_NAME

# Create database
echo "[3/4] Creating database '$DB_NAME'..."
createdb -h $DB_HOST -U $DB_USER $DB_NAME

# Apply schema
echo "[4/4] Applying schema..."
cd "$PROJECT_ROOT/sql"
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f hypercube_schema.sql

echo "========================================================"
echo "          SEEDING ATOMS (REQUIRED)                      "
echo "========================================================"
cd "$PROJECT_ROOT"
if [ -f "cpp/build/bin/Release/seed_atoms_parallel" ]; then
    echo "Running atom seeder..."
    ./cpp/build/bin/Release/seed_atoms_parallel --dbname $DB_NAME --host $DB_HOST --user $DB_USER
else
    echo "ERROR: Seeder binary not found at cpp/build/bin/Release/seed_atoms_parallel"
    echo "Please build the project first."
    exit 1
fi

echo "========================================================"
echo "          DATABASE RESET COMPLETE                       "
echo "========================================================"
