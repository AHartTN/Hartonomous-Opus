# SQL Consolidation Plan

**Generated:** January 5, 2026  
**Purpose:** Final analysis and execution order for 4-table schema

---

## 1. Canonical Schema: 020_four_tables.sql

The **4-table schema** is the canonical schema:

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `atom` | Unicode codepoints ONLY (leaf nodes) | `id`, `codepoint`, `value`, `geom`, `hilbert_lo`, `hilbert_hi` |
| `composition` | Aggregations of atoms/compositions | `id`, `label`, `depth`, `child_count`, `atom_count`, `geom`, `centroid`, `hilbert_lo`, `hilbert_hi` |
| `composition_child` | Ordered parent-child relationships | `composition_id`, `ordinal`, `child_type`, `child_id` |
| `relation` | Semantic edges (knowledge graph) | `id`, `source_type`, `source_id`, `target_type`, `target_id`, `relation_type`, `weight`, etc. |

**Critical:** In this schema, `atom` has NO `depth`, `children`, `centroid`, or `label` fields!

---

## 2. Correct Execution Order

### ✅ EXECUTE THESE (in order):

| Order | File | Purpose |
|-------|------|---------|
| 1 | `020_four_tables.sql` | Schema: atom, composition, composition_child, relation + upsert_relation() |
| 2 | `025_four_table_functions.sql` | Core functions for 4-table schema |
| 3 | `022_generative_engine.sql` | 4D similarity, scoring, generation (centroid_distance, etc.) |
| 4 | `023_bigram_stats.sql` | Bigram tables and PMI functions |
| 5 | `021_query_api.sql` | High-level query API (composition_text, similar, related, generative_walk) |
| 6 | `030_qa_search.sql` | Q&A search functions (search_text, ask, ask_exact, find_entities) |

### ⛔ DO NOT EXECUTE (wrong schema):

| File | Reason |
|------|--------|
| `012_semantic_udf.sql` | References `atom.depth`, `atom.centroid` which don't exist |
| `014_ai_operations.sql` | References `atom.depth`, `relation.parent_id` (wrong relation structure) |
| `015_centroid_optimization.sql` | References `atom.centroid` which doesn't exist |
| `016_semantic_queries.sql` | References `atom.depth`, `atom.centroid`, `atom.children` |
| `017_ingestion_ops.sql` | References `atom.children`, `atom.depth`, `atom.atom_count` |
| `018_core_queries.sql` | References `atom.depth`, `atom.node_role`, `atom.value` in wrong context |
| `019_function_aliases.sql` | References wrong function signatures from above files |

---

## 3. Duplicate Function Analysis

### Functions Defined Multiple Times (⚠️ conflicts):

| Function | Defined In | Winner | Notes |
|----------|------------|--------|-------|
| `atom_text(BYTEA)` | 012, 018, 025 | **025** | Others reference wrong schema |
| `atom_reconstruct_text(BYTEA)` | 012, 018, 025 | **025** | PL/pgSQL recursive, correct schema |
| `atom_find(TEXT)` | 012, 018, 019 | **025** (via composition_find) | Others broken |
| `semantic_neighbors(...)` | 014, 016, 025 | **025** | Uses relation table correctly |
| `attention(BYTEA, INT)` | 014, 025 | **025** | 014 uses wrong schema |
| `analogy(...)` | 014 (BYTEA), 016 (TEXT stub), 021 (TEXT), 025 (BYTEA) | **025** (BYTEA), **022/021** (TEXT) | Keep both signatures |
| `db_stats()` | 018, 021 | **021** | 018 has wrong signature |
| `content_exists(TEXT)` | 016, 017 | **Neither** | Both use wrong schema |
| `composition_text(BYTEA)` | 012, 021 | **021** | Uses composition_child correctly |
| `similar(...)` | 016 (TEXT, frechet), 021 (TEXT, centroid) | **021** | 016 broken |

### Safe Duplicates (same functionality):

- `atom_by_codepoint(INTEGER)` - All implementations equivalent
- `atom_exists(BYTEA)` - All check existence

---

## 4. Files by Schema Compatibility

### ✅ 4-Table Schema Compatible:

| File | Status | Notes |
|------|--------|-------|
| `020_four_tables.sql` | ✅ Canonical | Defines the schema |
| `021_query_api.sql` | ✅ Compatible | Uses composition, composition_child, relation |
| `022_generative_engine.sql` | ✅ Compatible | Uses composition.centroid, relation |
| `023_bigram_stats.sql` | ✅ Compatible | Creates new tables, uses composition |
| `025_four_table_functions.sql` | ✅ Compatible | Explicitly 4-table |
| `030_qa_search.sql` | ✅ Compatible | Uses composition, atom_reconstruct_text |

### ⛔ OLD Schema (INCOMPATIBLE):

| File | Problem Fields Referenced |
|------|---------------------------|
| `012_semantic_udf.sql` | `atom.depth`, `atom.centroid`, `atom.children`, `atom.atom_count` |
| `014_ai_operations.sql` | `atom.depth`, `atom.centroid`, `atom.atom_count`, `relation.parent_id` |
| `015_centroid_optimization.sql` | `atom.centroid` (atoms don't have centroids in 4-table) |
| `016_semantic_queries.sql` | `atom.depth`, `atom.centroid`, `atom.atom_count`, `atom.children` |
| `017_ingestion_ops.sql` | `atom.children`, `atom.depth`, `atom.atom_count`, `atom.geom` (wrong usage) |
| `018_core_queries.sql` | `atom.depth`, `atom.value`, `atom.node_role` |
| `019_function_aliases.sql` | Wraps functions from above broken files |

---

## 5. Recommended Actions

### Immediate: Create Archive Directory

```powershell
# Create archive directory
New-Item -ItemType Directory -Path "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\archive" -Force

# Move incompatible files
Move-Item -Path "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\012_semantic_udf.sql" -Destination "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\archive\012_semantic_udf.sql.DEPRECATED"
Move-Item -Path "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\014_ai_operations.sql" -Destination "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\archive\014_ai_operations.sql.DEPRECATED"
Move-Item -Path "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\015_centroid_optimization.sql" -Destination "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\archive\015_centroid_optimization.sql.DEPRECATED"
Move-Item -Path "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\016_semantic_queries.sql" -Destination "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\archive\016_semantic_queries.sql.DEPRECATED"
Move-Item -Path "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\017_ingestion_ops.sql" -Destination "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\archive\017_ingestion_ops.sql.DEPRECATED"
Move-Item -Path "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\018_core_queries.sql" -Destination "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\archive\018_core_queries.sql.DEPRECATED"
Move-Item -Path "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\019_function_aliases.sql" -Destination "D:\Repositories\Github\AHartTN\Hartonomous-Opus\sql\archive\019_function_aliases.sql.DEPRECATED"
```

### After Archive: Remaining Production Files

```
sql/
├── 020_four_tables.sql           # Schema definition
├── 021_query_api.sql             # Query API  
├── 022_generative_engine.sql     # Generative functions
├── 023_bigram_stats.sql          # Bigram/PMI tables
├── 025_four_table_functions.sql  # Core functions
├── 030_qa_search.sql             # Q&A search
└── archive/                      # Deprecated files
    ├── 011_unified_atom.sql.DEPRECATED
    ├── 012_semantic_udf.sql.DEPRECATED
    ├── 014_ai_operations.sql.DEPRECATED
    ├── 015_centroid_optimization.sql.DEPRECATED
    ├── 016_semantic_queries.sql.DEPRECATED
    ├── 017_ingestion_ops.sql.DEPRECATED
    ├── 018_core_queries.sql.DEPRECATED
    └── 019_function_aliases.sql.DEPRECATED
```

---

## 6. Execution Script

```bash
#!/bin/bash
# Execute SQL files in correct order for 4-table schema

DB_NAME="hypercube"

echo "=== Installing 4-Table Schema ==="

echo "Step 1: Schema definition..."
psql -d $DB_NAME -f sql/020_four_tables.sql

echo "Step 2: Core functions..."
psql -d $DB_NAME -f sql/025_four_table_functions.sql

echo "Step 3: Generative engine..."
psql -d $DB_NAME -f sql/022_generative_engine.sql

echo "Step 4: Bigram stats..."
psql -d $DB_NAME -f sql/023_bigram_stats.sql

echo "Step 5: Query API..."
psql -d $DB_NAME -f sql/021_query_api.sql

echo "Step 6: Q&A search..."
psql -d $DB_NAME -f sql/030_qa_search.sql

echo "=== Complete ==="
```

---

## 7. Function Inventory (Final State)

### From 020_four_tables.sql:
- `upsert_relation(...)` - Upsert with weight averaging

### From 025_four_table_functions.sql:
- `atom_is_leaf(BYTEA)` - Check if atom
- `atom_centroid(BYTEA)` - Get centroid (atom.geom or composition.centroid)
- `atom_children(BYTEA)` - Get children via composition_child
- `atom_child_count(BYTEA)` - Child count
- `atom_by_codepoint(INTEGER)` - Lookup by codepoint
- `atom_exists(BYTEA)` - Existence check
- `atom_text(BYTEA)` - Single character from atom
- `atom_reconstruct_text(BYTEA)` - Recursive text reconstruction
- `atom_knn(BYTEA, INTEGER)` - K-nearest neighbors
- `atom_hilbert_range(...)` - Hilbert range query
- `semantic_neighbors(BYTEA, INTEGER)` - Via relation table
- `attention(BYTEA, INTEGER)` - 4D spatial scoring
- `analogy(BYTEA, BYTEA, BYTEA, INTEGER)` - Vector arithmetic
- `atom_stats()` - Database statistics view
- `atom_distance(BYTEA, BYTEA)` - 3D distance

### From 022_generative_engine.sql:
- `centroid_distance(GEOMETRY, GEOMETRY)` - 4D Euclidean
- `centroid_similarity(GEOMETRY, GEOMETRY)` - Inverse distance
- `hilbert_distance(...)` - 128-bit Hilbert distance
- `similar_tokens(TEXT, INTEGER)` - By label
- `similar_tokens_fast(TEXT, INTEGER, DOUBLE)` - With Hilbert pre-filter
- `encode_prompt(TEXT)` - Tokenization
- `score_candidates(BYTEA[], INTEGER)` - Candidate scoring
- `generate_sql(TEXT, INTEGER, REAL, INTEGER)` - Text generation
- `complete(TEXT, INTEGER)` - Wrapper for gen_complete
- `vector_analogy(TEXT, TEXT, TEXT, INTEGER)` - Label-based analogy
- `semantic_midpoint(TEXT, TEXT, INTEGER)` - Midpoint search
- `gen_db_stats()` - Generation statistics

### From 023_bigram_stats.sql:
- Tables: `bigram_stats`, `unigram_stats`, `token_corpus_stats`
- `increment_bigram(BYTEA, BYTEA, BIGINT)` - Update bigram count
- `increment_unigram(BYTEA, BIGINT)` - Update unigram count
- `compute_pmi_scores()` - Compute PMI from counts
- `get_bigram_pmi(BYTEA, BYTEA)` - Get PMI score
- `get_bigram_count(BYTEA, BYTEA)` - Get raw count
- `top_continuations_pmi(BYTEA, INTEGER)` - Top by PMI
- `extract_bigrams_from_compositions()` - Extract from existing data

### From 021_query_api.sql:
- `composition_text(BYTEA)` - Reconstruct text via CTE
- `text(BYTEA)` - Alias
- `find_composition(TEXT)` - Find by label
- `get_centroid(TEXT)` - Get centroid by label
- `similar_by_centroid(TEXT, INTEGER)` - Similar by centroid
- `similar(TEXT, INTEGER)` - Alias
- `related_by_attention(TEXT, INTEGER, REAL)` - Via attention edges
- `related(TEXT, INTEGER)` - Alias
- `generative_walk(TEXT, INTEGER, REAL)` - Random walk
- `spatial_walk(TEXT, INTEGER)` - 4D spatial walk
- `analogy(TEXT, TEXT, TEXT, INTEGER)` - Label-based analogy
- `db_stats()` - Returns atoms, compositions, relations, models

### From 030_qa_search.sql:
- `search_text(TEXT, INTEGER)` - Keyword search
- `ask(TEXT)` - Natural language Q&A
- `ask_exact(TEXT, INTEGER)` - Exact phrase search
- `find_entities(TEXT)` - Named entity extraction

---

## 8. Summary

| Metric | Count |
|--------|-------|
| **Files to KEEP** | 6 |
| **Files to ARCHIVE** | 7 |
| **Tables in schema** | 4 + 3 bigram tables |
| **Core functions** | ~50 |
| **Duplicate functions resolved** | 12+ |
