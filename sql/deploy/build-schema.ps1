# Build consolidated schema file from source SQL files
# Works on Windows and Linux (PowerShell Core)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sqlRoot = Split-Path -Parent $scriptDir
$outputFile = Join-Path $scriptDir "full_schema.sql"

Write-Host "Building consolidated schema..." -ForegroundColor Cyan
Write-Host "  Source: $sqlRoot"
Write-Host "  Output: $outputFile"

# Order matters - dependencies first
$files = @(
    # 1. Extensions
    "schema/01_tables.sql",
    "schema/02_indexes.sql",
    "schema/03_constraints.sql",

    # 2. Geometry functions (foundational)
    "functions/geometry/distance.sql",

    # 3. Atom functions
    "functions/atoms/atom_is_leaf.sql",
    "functions/atoms/atom_centroid.sql",
    "functions/atoms/atom_exists.sql",
    "functions/atoms/atom_text.sql",
    "functions/atoms/atom_reconstruct_text.sql",
    "functions/atoms/atom_knn.sql",
    "functions/atoms/lookup.sql",
    "functions/atoms/atom_hilbert_range.sql",
    "functions/atoms/atom_by_codepoint.sql",
    "functions/atoms/get_atoms_by_codepoints.sql",
    "functions/atoms/atom_distance.sql",

    # 4. Composition functions
    "functions/compositions/atom_children.sql",
    "functions/compositions/atom_child_count.sql",
    "functions/compositions/find_composition.sql",
    "functions/compositions/compute_composition_centroid.sql",
    "functions/compositions/recompute_composition_centroids.sql",
    "functions/compositions/maintain_child_count_integrity.sql",
    "functions/compositions/validate_child_references.sql",

    # 5. Relation functions
    "functions/relations/semantic_neighbors.sql",
    "functions/relations/attention.sql",
    "functions/relations/analogy.sql",
    "functions/relations/upsert_relation.sql",
    "functions/relations/edges.sql",

    # 6. Query functions
    "functions/queries/search_text.sql",
    "functions/queries/ask.sql",
    "functions/queries/encode_prompt.sql",
    "functions/queries/score_candidates.sql",
    "functions/queries/generate_tokens.sql",
    "functions/queries/complete.sql",
    "functions/queries/vector_analogy.sql",

    # 7. Stats
    "functions/stats/db_stats.sql",

    # 8. Procedures
    "procedures/ingestion/seed_atoms.sql",
    "procedures/maintenance/prune_all.sql",
    "procedures/maintenance/prune_projections_deduplication.sql",
    "procedures/maintenance/prune_projections_quality.sql",
    "procedures/maintenance/prune_relations_weight.sql"
)

# Build consolidated file
$header = @"
-- =============================================================================
-- HYPERCUBE DATABASE SCHEMA - CONSOLIDATED DEPLOYMENT FILE
-- =============================================================================
-- Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
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

"@

$content = $header
$included = 0
$missing = @()

foreach ($file in $files) {
    $filePath = Join-Path $sqlRoot $file
    if (Test-Path $filePath) {
        $fileContent = Get-Content $filePath -Raw -Encoding UTF8
        $content += @"

-- =============================================================================
-- FILE: $file
-- =============================================================================
$fileContent

"@
        $included++
    } else {
        $missing += $file
    }
}

$footer = @"

-- =============================================================================
-- DEPLOYMENT COMPLETE
-- =============================================================================
-- To verify: SELECT * FROM db_stats();
-- =============================================================================

-- Restore normal message level
SET client_min_messages = NOTICE;
"@

$content += $footer

# Write with UTF-8 encoding (no BOM for Linux compatibility)
[System.IO.File]::WriteAllText($outputFile, $content, [System.Text.UTF8Encoding]::new($false))

Write-Host ""
Write-Host "Consolidated $included files into full_schema.sql" -ForegroundColor Green

if ($missing.Count -gt 0) {
    Write-Host "Missing files:" -ForegroundColor Yellow
    foreach ($m in $missing) {
        Write-Host "  - $m" -ForegroundColor Yellow
    }
}

Write-Host ""
