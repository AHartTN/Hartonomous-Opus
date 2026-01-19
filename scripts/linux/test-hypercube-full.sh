#!/bin/bash
# ============================================================================
# Hartonomous Hypercube - Full Integrity Test Suite (Linux)
# ============================================================================
# This script ports the comprehensive integrity test suite from the
# Windows PowerShell script Test-Hypercube-Full.ps1 to Linux Bash.
#
# Run this script from the project root:
#   ./scripts/linux/test-hypercube-full.sh > logs/Test-Hypercube-Full-log.txt 2>&1
# ============================================================================

# Source environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

run_sql() {
    local query="$1"
    local label="$2"

    echo ""
    echo -e "\e[36m====================================================\e[0m"
    echo -e "\e[36m $label\e[0m"
    echo -e "\e[36m====================================================\e[0m"

    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -c "$query" 2>&1
}

# ============================================================
# 1. ATOM TABLE TESTS
# ============================================================

run_sql "
SELECT COUNT(*) FROM atom;
" "Atom Count"

run_sql "
SELECT cp
FROM generate_series(0, 0x10FFFF) cp
LEFT JOIN atom a ON a.codepoint = cp
WHERE a.codepoint IS NULL;
" "Missing Unicode Codepoints"

run_sql "
SELECT hilbert_hi, hilbert_lo, COUNT(*)
FROM atom
GROUP BY hilbert_hi, hilbert_lo
HAVING COUNT(*) > 1;
" "Duplicate Hilbert Indices"

run_sql "
SELECT COUNT(*) FROM atom WHERE geom IS NULL;
" "Atoms Missing Geometry"

run_sql "
SELECT codepoint FROM atom WHERE NOT ST_IsValid(geom);
" "Invalid Atom Geometries"

run_sql "
SELECT COUNT(*) FROM atom WHERE ST_NDims(geom) <> 4;
" "Atoms With Wrong Dimensionality"

run_sql "
SELECT COUNT(*) FROM atom WHERE value = '' OR value IS NULL;
" "Atoms With Missing UTF-8 Bytes"

# ============================================================
# 2. COMPOSITION TESTS
# ============================================================

run_sql "
SELECT COUNT(*) FROM composition;
" "Composition Count"

run_sql "
SELECT id, label FROM composition WHERE centroid IS NULL;
" "Compositions Missing Centroid"

run_sql "
SELECT id, label FROM composition WHERE geom IS NULL;
" "Compositions Missing Geometry Path"

run_sql "
SELECT c.id, c.child_count, COUNT(cc.*) AS actual
FROM composition c
LEFT JOIN composition_child cc ON cc.composition_id = c.id
GROUP BY c.id
HAVING COUNT(cc.*) <> c.child_count
limit 10;
" "Compositions With Mismatched Child Counts"

run_sql "
SELECT id, label, depth
FROM composition
WHERE depth < 1 OR depth > 100;
" "Compositions With Invalid Depth"

run_sql "
SELECT COUNT(0)
FROM composition
WHERE atom_count < child_count;
" "Composition Count With Impossible Atom Counts"

run_sql "
SELECT id, label
FROM composition
WHERE atom_count < child_count
limit 10;
" "Composition Samples With Impossible Atom Counts"

run_sql "
SELECT id, label
FROM composition
WHERE child_count = 1 AND atom_count <> 1;
" "Single-Child Compositions With Wrong Atom Count"

# ============================================================
# 3. COMPOSITION_CHILD TESTS
# ============================================================

run_sql "
SELECT COUNT(*) FROM composition_child;
" "Composition Child Count"

run_sql "
SELECT *
FROM composition_child cc
LEFT JOIN atom a ON (cc.child_type='A' AND cc.child_id=a.id)
LEFT JOIN composition c ON (cc.child_type='C' AND cc.child_id=c.id)
WHERE (cc.child_type='A' AND a.id IS NULL)
   OR (cc.child_type='C' AND c.id IS NULL);
" "Invalid Composition Child References"

run_sql "
SELECT composition_id
FROM composition_child
GROUP BY composition_id
HAVING COUNT(*) <> (MAX(ordinal) + 1);
" "Non-Contiguous Ordinals"

run_sql "
SELECT composition_id, ordinal
FROM composition_child
WHERE ordinal < 0;
" "Negative Ordinals"

# ============================================================
# 4. HIERARCHY / DAG TESTS
# ============================================================

run_sql "
WITH RECURSIVE cycle_check(id, path) AS (
    SELECT id, ARRAY[id]
    FROM composition
    UNION ALL
    SELECT cc.child_id, path || cc.child_id
    FROM cycle_check c
    JOIN composition_child cc ON cc.composition_id = c.id
    WHERE cc.child_type='C' AND NOT cc.child_id = ANY(path)
),
RESULTS as (SELECT DISTINCT id FROM cycle_check
WHERE id = ANY(path[2:]))
SELECT COUNT(*) FROM RESULTS;
" "Composition DAG Cycle Detection Count"

run_sql "
WITH RECURSIVE cycle_check(id, path) AS (
    SELECT id, ARRAY[id]
    FROM composition
    UNION ALL
    SELECT cc.child_id, path || cc.child_id
    FROM cycle_check c
    JOIN composition_child cc ON cc.composition_id = c.id
    WHERE cc.child_type='C' AND NOT cc.child_id = ANY(path)
)
SELECT DISTINCT id FROM cycle_check
WHERE id = ANY(path[2:])
limit 10;
" "Composition DAG Cycle Detection Samples"

# ============================================================
# 5. RELATION TESTS
# ============================================================

run_sql "
SELECT COUNT(*) FROM relation;
" "Relation Count"

run_sql "
SELECT source_id, target_id, relation_type, COUNT(*)
FROM relation
GROUP BY source_id, target_id, relation_type
HAVING COUNT(*) > 1;
" "Duplicate Relations"

run_sql "
SELECT *
FROM relation r
LEFT JOIN atom a1 ON (r.source_type='A' AND r.source_id=a1.id)
LEFT JOIN composition c1 ON (r.source_type='C' AND r.source_id=c1.id)
LEFT JOIN atom a2 ON (r.target_type='A' AND r.target_id=a2.id)
LEFT JOIN composition c2 ON (r.target_type='C' AND r.target_id=c2.id)
WHERE (r.source_type='A' AND a1.id IS NULL)
   OR (r.source_type='C' AND c1.id IS NULL)
   OR (r.target_type='A' AND a2.id IS NULL)
   OR (r.target_type='C' AND c2.id IS NULL);
" "Orphan Relations"

# ============================================================
# 6. MODEL REGISTRY TESTS
# ============================================================

run_sql "
SELECT COUNT(*) FROM model;
" "Model Count"

run_sql "
SELECT id, name, source, version, embedding_dim
FROM model;
" "Model Registry"

# ============================================================
# 7. PROJECTION METADATA TESTS
# ============================================================

run_sql "
SELECT model_id, tensor_name, converged, variance_explained, quality_score
FROM projection_metadata
WHERE converged = FALSE OR quality_score < 2.0;
" "Projection Failures or Low Quality"

run_sql "
SELECT COUNT(*) FROM projection_metadata;
" "Projection Metadata Count"

# ============================================================
# 8. BIGRAM / UNIGRAM TESTS
# ============================================================

run_sql "
SELECT COUNT(*) FROM unigram_stats;
" "Unigram Count"

run_sql "
SELECT COUNT(*) FROM bigram_stats;
" "Bigram Count"

run_sql "
SELECT *
FROM bigram_stats
ORDER BY pmi DESC
LIMIT 50;
" "Top PMI Bigrams"

# ============================================================
# 9. RELATION EVIDENCE TESTS
# ============================================================

run_sql "
SELECT COUNT(*) FROM relation_evidence;
" "Relation Evidence Count"

run_sql "
SELECT source_id, target_id, relation_type,
       STDDEV(rating) AS disagreement,
       COUNT(*) AS num_models
FROM relation_evidence
GROUP BY source_id, target_id, relation_type
HAVING STDDEV(rating) > 200;
" "High-Disagreement Relations"

# ============================================================
# 10. CONSENSUS VIEW TESTS
# ============================================================

run_sql "
REFRESH MATERIALIZED VIEW relation_consensus;
" "Refresh Consensus View"

run_sql "
SELECT COUNT(0)
FROM relation_consensus;
" "Top Consensus Relation Count"

run_sql "
SELECT *
FROM relation_consensus
ORDER BY consensus_weight DESC
LIMIT 10;
" "Top Consensus Relation Samples"

# ============================================================
# 11. FULL INTEGRITY SWEEP
# ============================================================

run_sql "
SELECT COUNT(*) AS missing_atoms
FROM generate_series(0, 0x10FFFF) cp
LEFT JOIN atom a ON a.codepoint = cp
WHERE a.codepoint IS NULL;
" "Integrity: Missing Atoms"

run_sql "
SELECT COUNT(*) AS invalid_atom_geom
FROM atom
WHERE NOT ST_IsValid(geom);
" "Integrity: Invalid Atom Geometry"

run_sql "
SELECT COUNT(*) AS bad_compositions
FROM (
    SELECT c.id
    FROM composition c
    LEFT JOIN composition_child cc ON cc.composition_id = c.id
    GROUP BY c.id, c.child_count
    HAVING COUNT(cc.*) <> c.child_count
) q;
" "Integrity: Bad Compositions"

run_sql "
SELECT COUNT(*) AS orphan_children
FROM composition_child cc
LEFT JOIN composition c ON cc.composition_id = c.id
WHERE c.id IS NULL;
" "Integrity: Orphan Children"

run_sql "
SELECT COUNT(*) AS orphan_relations
FROM relation r
LEFT JOIN atom a1 ON (r.source_type='A' AND r.source_id=a1.id)
LEFT JOIN composition c1 ON (r.source_type='C' AND r.source_id=c1.id)
LEFT JOIN atom a2 ON (r.target_type='A' AND r.target_id=a2.id)
LEFT JOIN composition c2 ON (r.target_type='C' AND r.target_id=c2.id)
WHERE (r.source_type='A' AND a1.id IS NULL)
   OR (r.source_type='C' AND c1.id IS NULL)
   OR (r.target_type='A' AND a2.id IS NULL)
   OR (r.target_type='C' AND c2.id IS NULL);
" "Integrity: Orphan Relations"