#!/bin/bash
# Seed the atom table with all Unicode codepoints
# Uses EWKB format for direct geometry insertion (no ST_MakePoint overhead)
#
# Usage: ./seed_atoms.sh [database_name] [options]
#   database_name: defaults to 'hypercube'
#   --force: drop and recreate the database
#   --skip-build: skip rebuilding the seed tool

set -e

DB_NAME="${1:-hypercube}"
shift || true

FORCE=false
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SEED_TOOL="$PROJECT_ROOT/cpp/build/seed_atoms"

echo "=== Seeding Hypercube Atoms ==="
echo "Database: $DB_NAME"
echo "Force: $FORCE"

# Build seed tool if needed
if [ ! -x "$SEED_TOOL" ] && [ "$SKIP_BUILD" = false ]; then
    echo "Building seed_atoms tool..."
    "$SCRIPT_DIR/build.sh"
fi

if [ ! -x "$SEED_TOOL" ]; then
    echo "ERROR: seed_atoms tool not found at $SEED_TOOL"
    echo "Run: ./scripts/build.sh"
    exit 1
fi

# Handle database creation
if [ "$FORCE" = true ]; then
    echo "Dropping existing database..."
    dropdb --if-exists "$DB_NAME" 2>/dev/null || true
fi

if ! psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo "Creating database $DB_NAME..."
    createdb "$DB_NAME"
fi

# Apply schema
echo "Applying schema..."
psql -d "$DB_NAME" -q -f "$PROJECT_ROOT/sql/001_schema.sql"

# Generate and load atoms using parallel seeder (fastest)
echo "Generating and loading atoms..."
START_TIME=$(date +%s.%N)

PARALLEL_SEED_TOOL="$PROJECT_ROOT/cpp/build/seed_atoms_parallel"

if [ -x "$PARALLEL_SEED_TOOL" ]; then
    # Use parallel seeder (12 connections, drops/rebuilds indexes internally)
    "$PARALLEL_SEED_TOOL" -d "$DB_NAME" 2>&1 | grep -E "(Generated|Partitioned|COPY|Index|Complete|Rate)"
else
    # Fallback to single-connection COPY with EWKB format
    echo "Parallel seeder not found, using single-connection COPY..."
    
    # Drop indexes for faster insert
    psql -d "$DB_NAME" -q <<'EOF'
DROP INDEX IF EXISTS idx_atom_coords;
DROP INDEX IF EXISTS idx_atom_hilbert;
DROP INDEX IF EXISTS idx_atom_category;
DROP INDEX IF EXISTS idx_atom_letters;
DROP INDEX IF EXISTS idx_atom_digits;
TRUNCATE atom CASCADE;
EOF

    # COPY with EWKB geometry format (no ST_MakePoint overhead)
    "$SEED_TOOL" --ewkb 2>/dev/null | \
        psql -d "$DB_NAME" -q -c "COPY atom (id, codepoint, category, coords, hilbert_lo, hilbert_hi) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')"

    # Rebuild indexes
    echo "Rebuilding indexes..."
    psql -d "$DB_NAME" -q <<'EOF'
CREATE INDEX idx_atom_coords ON atom USING GIST(coords);
CREATE INDEX idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);
CREATE INDEX idx_atom_category ON atom(category);
CREATE INDEX idx_atom_letters ON atom(codepoint) 
    WHERE category IN ('letter_upper', 'letter_lower', 'letter_titlecase', 'letter_other');
CREATE INDEX idx_atom_digits ON atom(codepoint)
    WHERE category = 'digit';
ANALYZE atom;
EOF
fi

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)
LOAD_ONLY="$TOTAL_TIME"

# Apply additional functions
echo "Applying functions..."
psql -d "$DB_NAME" -q -f "$PROJECT_ROOT/sql/002_functions.sql" 2>/dev/null || echo "  (some functions require C++ extension)"

# Report
COUNT=$(psql -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM atom")
CATEGORIES=$(psql -d "$DB_NAME" -tAc "SELECT COUNT(DISTINCT category) FROM atom")

echo ""
echo "=== Seeding Complete ==="
echo "Atoms loaded: $COUNT"
echo "Categories: $CATEGORIES"
echo "COPY time: ${LOAD_ONLY}s"
echo "Total time: ${TOTAL_TIME}s"
echo ""
echo "Verify:"
echo "  psql -d $DB_NAME -c \"SELECT codepoint, category, ST_AsText(coords) FROM atom WHERE codepoint IN (65, 97, 48);\""
echo ""
echo "Statistics:"
psql -d "$DB_NAME" -c "SELECT category, COUNT(*) as count FROM atom GROUP BY category ORDER BY count DESC LIMIT 10;"
