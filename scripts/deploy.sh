#!/bin/bash
# Full deployment script for Hartonomous Hypercube
# Creates database, builds C++ components, seeds atoms, applies all schema
#
# Usage: ./deploy.sh [database_name] [options]
#   database_name: defaults to 'hypercube'
#   --force: drop and recreate everything
#   --extension: also install PostgreSQL C++ extension (requires sudo)

set -e

DB_NAME="${1:-hypercube}"
shift || true

FORCE=false
INSTALL_EXTENSION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE=true; shift ;;
        --extension) INSTALL_EXTENSION=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  Hartonomous Hypercube Deployment"
echo "=============================================="
echo "Database: $DB_NAME"
echo "Force: $FORCE"
echo "Install Extension: $INSTALL_EXTENSION"
echo ""

# Step 1: Build C++ components
echo "=== Step 1: Building C++ Components ==="
"$SCRIPT_DIR/build.sh"

# Step 2: Create and seed database
echo ""
echo "=== Step 2: Creating and Seeding Database ==="
if [ "$FORCE" = true ]; then
    "$SCRIPT_DIR/seed_atoms.sh" "$DB_NAME" --force --skip-build
else
    "$SCRIPT_DIR/seed_atoms.sh" "$DB_NAME" --skip-build
fi

# Step 3: Apply all SQL functions
echo ""
echo "=== Step 3: Applying SQL Functions ==="
psql -d "$DB_NAME" -q -f "$PROJECT_ROOT/sql/002_functions.sql" 2>/dev/null || \
    echo "  Note: Some functions require C++ extension (hypercube_blake3, hypercube_coords_to_hilbert)"
psql -d "$DB_NAME" -q -f "$PROJECT_ROOT/sql/003_ingestion.sql" 2>/dev/null || \
    echo "  Note: Some ingestion functions require C++ extension"

# Step 4: Install C++ extension if requested
if [ "$INSTALL_EXTENSION" = true ]; then
    echo ""
    echo "=== Step 4: Installing PostgreSQL Extension ==="
    cd "$PROJECT_ROOT/cpp/build"
    sudo make install
    psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS hypercube;"
fi

# Step 5: Run verification
echo ""
echo "=== Step 5: Verification ==="
echo "Checking atom table..."
ATOM_COUNT=$(psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM atom")
echo "  Atoms: $ATOM_COUNT"

echo ""
echo "Sample atoms (A, a, 0):"
psql -d "$DB_NAME" -c "
SELECT 
    codepoint,
    chr(codepoint) as char,
    category,
    ROUND(ST_X(coords)::numeric, 4) as x,
    ROUND(ST_Y(coords)::numeric, 4) as y,
    ROUND(ST_Z(coords)::numeric, 4) as z,
    ROUND(ST_M(coords)::numeric, 4) as m
FROM atom 
WHERE codepoint IN (65, 97, 48, 20013)
ORDER BY codepoint;
"

echo ""
echo "=============================================="
echo "  Deployment Complete!"
echo "=============================================="
echo ""
echo "Usage examples:"
echo "  # Query an atom"
echo "  psql -d $DB_NAME -c \"SELECT * FROM atom WHERE codepoint = 65;\""
echo ""
echo "  # Find neighbors of 'A'"
echo "  psql -d $DB_NAME -c \"SELECT * FROM hypercube_find_neighbors(65);\""
echo ""
echo "  # Category statistics"
echo "  psql -d $DB_NAME -c \"SELECT * FROM atom_stats;\""
