param(
    [string]$Hostname = "hart-server",
    [int]$Port = 5432,
    [string]$User = "postgres",
    [string]$Database = "hypercube"
)

function Run-SQL {
    param([string]$Query, [string]$Label)

    Write-Host ""
    Write-Host "====================================================" -ForegroundColor Cyan
    Write-Host " $Label" -ForegroundColor Cyan
    Write-Host "====================================================" -ForegroundColor Cyan

    $escaped = $Query.Replace("`n"," ")
    psql -h $Hostname -p $Port -U $User -d $Database -c "$escaped" 2>&1
}

# ============================================================
# 1. ATOM TABLE TESTS
# ============================================================

Run-SQL @"
SELECT COUNT(*) FROM atom;
"@ "Atom Count"

Run-SQL @"
SELECT cp
FROM generate_series(0, 0x10FFFF) cp
LEFT JOIN atom a ON a.codepoint = cp
WHERE a.codepoint IS NULL;
"@ "Missing Unicode Codepoints"

Run-SQL @"
SELECT hilbert_hi, hilbert_lo, COUNT(*)
FROM atom
GROUP BY hilbert_hi, hilbert_lo
HAVING COUNT(*) > 1;
"@ "Duplicate Hilbert Indices"

Run-SQL @"
SELECT COUNT(*) FROM atom WHERE geom IS NULL;
"@ "Atoms Missing Geometry"

Run-SQL @"
SELECT codepoint FROM atom WHERE NOT ST_IsValid(geom);
"@ "Invalid Atom Geometries"

Run-SQL @"
SELECT COUNT(*) FROM atom WHERE ST_NDims(geom) <> 4;
"@ "Atoms With Wrong Dimensionality"

Run-SQL @"
SELECT COUNT(*) FROM atom WHERE value = '' OR value IS NULL;
"@ "Atoms With Missing UTF-8 Bytes"

# ============================================================
# 2. COMPOSITION TESTS
# ============================================================

Run-SQL @"
SELECT COUNT(*) FROM composition;
"@ "Composition Count"

Run-SQL @"
SELECT id, label FROM composition WHERE centroid IS NULL;
"@ "Compositions Missing Centroid"

Run-SQL @"
SELECT id, label FROM composition WHERE geom IS NULL;
"@ "Compositions Missing Geometry Path"

Run-SQL @"
SELECT c.id, c.child_count, COUNT(cc.*) AS actual
FROM composition c
LEFT JOIN composition_child cc ON cc.composition_id = c.id
GROUP BY c.id
HAVING COUNT(cc.*) <> c.child_count
limit 10;
"@ "Compositions With Mismatched Child Counts"

Run-SQL @"
SELECT id, label, depth
FROM composition
WHERE depth < 1 OR depth > 100;
"@ "Compositions With Invalid Depth"

Run-SQL @"
SELECT COUNT(0)
FROM composition
WHERE atom_count < child_count;
"@ "Composition Count With Impossible Atom Counts"

Run-SQL @"
SELECT id, label
FROM composition
WHERE atom_count < child_count
limit 10;
"@ "Composition Samples With Impossible Atom Counts"

Run-SQL @"
SELECT id, label
FROM composition
WHERE child_count = 1 AND atom_count <> 1;
"@ "Single-Child Compositions With Wrong Atom Count"

# ============================================================
# 3. COMPOSITION_CHILD TESTS
# ============================================================

Run-SQL @"
SELECT COUNT(*) FROM composition_child;
"@ "Composition Child Count"

Run-SQL @"
SELECT *
FROM composition_child cc
LEFT JOIN atom a ON (cc.child_type='A' AND cc.child_id=a.id)
LEFT JOIN composition c ON (cc.child_type='C' AND cc.child_id=c.id)
WHERE (cc.child_type='A' AND a.id IS NULL)
   OR (cc.child_type='C' AND c.id IS NULL);
"@ "Invalid Composition Child References"

Run-SQL @"
SELECT composition_id
FROM composition_child
GROUP BY composition_id
HAVING COUNT(*) <> (MAX(ordinal) + 1);
"@ "Non-Contiguous Ordinals"

Run-SQL @"
SELECT composition_id, ordinal
FROM composition_child
WHERE ordinal < 0;
"@ "Negative Ordinals"

# ============================================================
# 4. HIERARCHY / DAG TESTS
# ============================================================

Run-SQL @"
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
"@ "Composition DAG Cycle Detection Count"

Run-SQL @"
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
"@ "Composition DAG Cycle Detection Samples"

# ============================================================
# 5. RELATION TESTS
# ============================================================

Run-SQL @"
SELECT COUNT(*) FROM relation;
"@ "Relation Count"

Run-SQL @"
SELECT source_id, target_id, relation_type, COUNT(*)
FROM relation
GROUP BY source_id, target_id, relation_type
HAVING COUNT(*) > 1;
"@ "Duplicate Relations"

Run-SQL @"
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
"@ "Orphan Relations"

# ============================================================
# 6. MODEL REGISTRY TESTS
# ============================================================

Run-SQL @"
SELECT COUNT(*) FROM model;
"@ "Model Count"

Run-SQL @"
SELECT id, name, source, version, embedding_dim
FROM model;
"@ "Model Registry"

# ============================================================
# 7. PROJECTION METADATA TESTS
# ============================================================

Run-SQL @"
SELECT model_id, tensor_name, converged, variance_explained, quality_score
FROM projection_metadata
WHERE converged = FALSE OR quality_score < 2.0;
"@ "Projection Failures or Low Quality"

Run-SQL @"
SELECT COUNT(*) FROM projection_metadata;
"@ "Projection Metadata Count"

# ============================================================
# 8. BIGRAM / UNIGRAM TESTS
# ============================================================

Run-SQL @"
SELECT COUNT(*) FROM unigram_stats;
"@ "Unigram Count"

Run-SQL @"
SELECT COUNT(*) FROM bigram_stats;
"@ "Bigram Count"

Run-SQL @"
SELECT *
FROM bigram_stats
ORDER BY pmi DESC
LIMIT 50;
"@ "Top PMI Bigrams"

# ============================================================
# 9. RELATION EVIDENCE TESTS
# ============================================================

Run-SQL @"
SELECT COUNT(*) FROM relation_evidence;
"@ "Relation Evidence Count"

Run-SQL @"
SELECT source_id, target_id, relation_type,
       STDDEV(rating) AS disagreement,
       COUNT(*) AS num_models
FROM relation_evidence
GROUP BY source_id, target_id, relation_type
HAVING STDDEV(rating) > 200;
"@ "High-Disagreement Relations"

# ============================================================
# 10. CONSENSUS VIEW TESTS
# ============================================================

Run-SQL @"
REFRESH MATERIALIZED VIEW relation_consensus;
"@ "Refresh Consensus View"

Run-SQL @"
SELECT COUNT(0)
FROM relation_consensus;
"@ "Top Consensus Relation Count"

Run-SQL @"
SELECT *
FROM relation_consensus
ORDER BY consensus_weight DESC
LIMIT 10;
"@ "Top Consensus Relation Samples"

# ============================================================
# 11. FULL INTEGRITY SWEEP
# ============================================================

Run-SQL @"
SELECT COUNT(*) AS missing_atoms
FROM generate_series(0, 0x10FFFF) cp
LEFT JOIN atom a ON a.codepoint = cp
WHERE a.codepoint IS NULL;
"@ "Integrity: Missing Atoms"

Run-SQL @"
SELECT COUNT(*) AS invalid_atom_geom
FROM atom
WHERE NOT ST_IsValid(geom);
"@ "Integrity: Invalid Atom Geometry"

Run-SQL @"
SELECT COUNT(*) AS bad_compositions
FROM (
    SELECT c.id
    FROM composition c
    LEFT JOIN composition_child cc ON cc.composition_id = c.id
    GROUP BY c.id, c.child_count
    HAVING COUNT(cc.*) <> c.child_count
) q;
"@ "Integrity: Bad Compositions"

Run-SQL @"
SELECT COUNT(*) AS orphan_children
FROM composition_child cc
LEFT JOIN composition c ON cc.composition_id = c.id
WHERE c.id IS NULL;
"@ "Integrity: Orphan Children"

Run-SQL @"
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
"@ "Integrity: Orphan Relations"
