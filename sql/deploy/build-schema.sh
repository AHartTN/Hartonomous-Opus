#!/bin/bash
# Build consolidated schema file from source SQL files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_FILE="$SCRIPT_DIR/full_schema.sql"

echo "Building consolidated schema..."
echo "  Source: $SQL_ROOT"
echo "  Output: $OUTPUT_FILE"

# Order matters - dependencies first
FILES=(
    # 1. Tables and structure
    "schema/01_tables.sql"
    "schema/02_indexes.sql"
    "schema/03_constraints.sql"

    # 2. Geometry functions (foundational)
    "functions/geometry/distance.sql"

    # 3. Atom functions
    "functions/atoms/atom_is_leaf.sql"
    "functions/atoms/atom_centroid.sql"
    "functions/atoms/atom_exists.sql"
    "functions/atoms/atom_text.sql"
    "functions/atoms/atom_reconstruct_text.sql"
    "functions/atoms/atom_knn.sql"
    "functions/atoms/lookup.sql"
    "functions/atoms/atom_hilbert_range.sql"
    "functions/atoms/atom_by_codepoint.sql"
    "functions/atoms/get_atoms_by_codepoints.sql"
    "functions/atoms/atom_distance.sql"

    # 4. Composition functions
    "functions/compositions/atom_children.sql"
    "functions/compositions/atom_child_count.sql"
    "functions/compositions/find_composition.sql"
    "functions/compositions/compute_composition_centroid.sql"
    "functions/compositions/recompute_composition_centroids.sql"
    "functions/compositions/maintain_child_count_integrity.sql"
    "functions/compositions/validate_child_references.sql"

    # 5. Relation functions
    "functions/relations/semantic_neighbors.sql"
    "functions/relations/attention.sql"
    "functions/relations/analogy.sql"
    "functions/relations/upsert_relation.sql"
    "functions/relations/edges.sql"

    # 6. Query functions
    "functions/queries/search_text.sql"
    "functions/queries/ask.sql"
    "functions/queries/encode_prompt.sql"
    "functions/queries/score_candidates.sql"
    "functions/queries/generate_tokens.sql"
    "functions/queries/complete.sql"
    "functions/queries/vector_analogy.sql"

    # 7. Stats
    "functions/stats/db_stats.sql"

    # 8. Procedures
    "procedures/ingestion/seed_atoms.sql"
    "procedures/maintenance/prune_all.sql"
    "procedures/maintenance/prune_projections_deduplication.sql"
    "procedures/maintenance/prune_projections_quality.sql"
    "procedures/maintenance/prune_relations_weight.sql"
)

# Build consolidated file
cat > "$OUTPUT_FILE" << 'HEADER'
-- =============================================================================
-- HYPERCUBE DATABASE SCHEMA - CONSOLIDATED DEPLOYMENT FILE
-- =============================================================================
-- This file contains the complete schema and can be run directly with psql.
--
-- Usage:
--   psql -h HOST -U USER -d DATABASE -f full_schema.sql
--
-- Prerequisites:
--   - PostgreSQL 14+ with PostGIS extension
--   - Database must exist (CREATE DATABASE hypercube;)
-- =============================================================================

-- Suppress NOTICE messages during deployment
SET client_min_messages = WARNING;

-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;

HEADER

# Add timestamp
echo "-- Generated: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

INCLUDED=0
MISSING=()

for file in "${FILES[@]}"; do
    filepath="$SQL_ROOT/$file"
    if [[ -f "$filepath" ]]; then
        echo "" >> "$OUTPUT_FILE"
        echo "-- =============================================================================" >> "$OUTPUT_FILE"
        echo "-- FILE: $file" >> "$OUTPUT_FILE"
        echo "-- =============================================================================" >> "$OUTPUT_FILE"
        cat "$filepath" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        ((INCLUDED++))
    else
        MISSING+=("$file")
    fi
done

cat >> "$OUTPUT_FILE" << 'FOOTER'

-- =============================================================================
-- DEPLOYMENT COMPLETE
-- =============================================================================
-- To verify: SELECT * FROM db_stats();
-- =============================================================================

-- Restore normal message level
SET client_min_messages = NOTICE;
FOOTER

echo ""
echo "Consolidated $INCLUDED files into full_schema.sql"

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "Missing files:"
    for m in "${MISSING[@]}"; do
        echo "  - $m"
    done
fi

echo ""
