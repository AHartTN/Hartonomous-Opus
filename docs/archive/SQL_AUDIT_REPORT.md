# SQL Files Comprehensive Audit Report

**Generated:** January 5, 2026  
**Scope:** `sql/` and `tests/sql/` directories

---

## Executive Summary

| Category | Count |
|----------|-------|
| Production Files (sql/) | 17 files |
| Test Files (tests/sql/) | 5 files (2 deprecated) |
| Files with FIXME/TODO/Stub | 1 file |
| Duplicate Function Definitions | 12+ functions |
| File Numbering Conflicts | 2 pairs |
| Already Deprecated | 3 files |

### Critical Issues Found
1. **Duplicate file numbering:** 026 and 027 have two files each
2. **Significant function duplication:** Many core functions redefined in multiple files
3. **Schema evolution confusion:** Files reference different schema versions (3-table vs 4-table)
4. **Missing file in sequence:** 013 is missing; 024, 028, 029 are gaps

---

## Detailed File Analysis

### sql/ Directory

#### 011_unified_atom.sql.DEPRECATED
| Attribute | Value |
|-----------|-------|
| **Status** | DEPRECATED (already marked) |
| **Action** | ✅ KEEP AS-IS (correct archive status) |
| **Notes** | Already correctly deprecated - old unified schema |

---

#### 012_semantic_udf.sql
| Attribute | Value |
|-----------|-------|
| **Status** | MIXED - Contains production code but outdated schema references |
| **Lines** | 454 |
| **Action** | 🔀 ARCHIVE or MERGE |
| **Schema** | THREE-TABLE (atom/composition/relation) - but header says "020 Schema Compatible" |

**Defines:**
- `atom_is_leaf()` - DUPLICATED in 025, 027
- `atom_coords()` 
- `atom_centroid()` - DUPLICATED in 025, 027
- `composition_centroid()`
- `composition_children()` - uses composition_child table
- `atom_by_codepoint()` - DUPLICATED in 025, 027
- `entity_exists()`, `atom_exists()` - DUPLICATED
- `composition_reconstruct()` - recursive CTE
- `composition_text()`, `atom_text()`, `atom_reconstruct_text()` - DUPLICATED in 025
- `composition_distance()`, `atom_distance()` - DUPLICATED
- `composition_knn()`, `atom_knn()` - DUPLICATED
- `atom_hilbert_range()`, `composition_hilbert_range()` - DUPLICATED in 025
- `composition_parents()`, `atom_parents()`
- `composition_search()`, `atom_search()`
- `composition_find()`, `atom_find()` - DUPLICATED
- Views: `v_atom_stats`, `v_depth_stats`, `atom_stats`, `atom_type_stats`
- `atom_content_hash()` - DUPLICATED in 018

**Issues:**
- Massive duplication with 025_four_table_functions.sql
- Uses `composition_child` table (from 020) correctly
- Functions reference THREE-TABLE schema but numbered before 020

**Recommendation:** ARCHIVE - functionality is better organized in 025_four_table_functions.sql

---

#### 014_ai_operations.sql
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - but uses older `relation` schema |
| **Lines** | 180 |
| **Action** | 🔀 REVIEW & UPDATE |
| **Schema** | Uses `relation` table with `parent_id`/`child_id` (OLD) |

**Defines:**
- `semantic_neighbors(BYTEA, INTEGER)` - CONFLICTING signatures with 016, 025
- `semantic_degree()`
- `semantic_top_edges()`
- `attention()` - DUPLICATED in 025
- `analogy(BYTEA, BYTEA, BYTEA, INTEGER)` - DUPLICATED in 025
- `random_walk()`
- `semantic_path()`
- `related_by_children()`
- `semantic_jaccard()`

**Issues:**
- Uses OLD relation schema (parent_id/child_id) not (source_id/target_id)
- `semantic_neighbors()` defined differently here vs 016 vs 025
- Queries `relation` table with `relation_type = 'C'` (Child)

**Recommendation:** UPDATE - change to source_id/target_id schema, then KEEP

---

#### 015_centroid_optimization.sql
| Attribute | Value |
|-----------|-------|
| **Status** | LEGACY MAINTENANCE |
| **Lines** | 57 |
| **Action** | 🔀 ARCHIVE |
| **Schema** | References old unified `atom` table with `centroid` column |

**Defines:**
- `maintenance_centroids()` - updates `atom.centroid` from `atom.geom`
- View: `v_centroid_status`

**Issues:**
- Assumes `atom` table has both `geom` AND `centroid` columns
- In 020 schema: `atom` only has `geom`, `composition` has `centroid`
- Functionality superseded by `recompute_composition_centroids()` in 027_centroid_4d.sql

**Recommendation:** ARCHIVE - obsolete for 4-table schema

---

#### 016_semantic_queries.sql
| Attribute | Value |
|-----------|-------|
| **Status** | INCOMPLETE - contains stub functions |
| **Lines** | 293 |
| **Action** | ⚠️ COMPLETE OR ARCHIVE |
| **Schema** | Mixed - references both `atom` and `composition` tables |

**Defines:**
- `content_exists()` - DUPLICATED in 017
- `content_get()`
- `text_frechet_similar()` - Fréchet distance queries
- `similar(TEXT, INTEGER)` - alias
- `semantic_neighbors(TEXT, INTEGER)` - DIFFERENT signature than 014/025!
- `neighbors()` - alias
- `semantic_follow()`
- `follows()` - alias
- `semantic_walk()` - calls C++ extension `hypercube_semantic_walk()`
- **`analogy(TEXT, TEXT, TEXT, INTEGER)` - STUB IMPLEMENTATION** ⚠️
- `compound_similar()` - references `composition` table
- `composition_info()` - references `atom` table with `children` column (OLD)
- `edge_count()`

**Critical Issues:**
- Line 183-196: `analogy()` function is a STUB:
  ```sql
  -- Stub - requires C++ extension hypercube_ops for efficient analogy computation
  SELECT NULL::TEXT, NULL::DOUBLE PRECISION WHERE FALSE;  -- Stub until implemented
  ```
- `composition_info()` references `a.children` array (old unified schema)
- `semantic_neighbors()` here takes TEXT input, 014/025 take BYTEA

**Recommendation:** UPDATE to fix stubs and schema, or ARCHIVE if 021/022 covers this

---

#### 017_ingestion_ops.sql
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION |
| **Lines** | 54 |
| **Action** | ✅ KEEP (with minor update) |
| **Schema** | Uses `atom` table |

**Defines:**
- `upsert_composition()` - inserts into `atom` table (misnamed!)
- `content_hash()` - BLAKE3 hash
- `find_content()` - lookup by hash
- `content_exists()` - DUPLICATED in 016

**Issues:**
- `upsert_composition()` inserts into `atom` table, not `composition`
- Function name is misleading for 4-table schema
- `content_exists()` duplicated in 016

**Recommendation:** RENAME function or update for clarity; remove duplicate

---

#### 018_core_queries.sql
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - Core query functions |
| **Lines** | 351 |
| **Action** | ✅ KEEP |
| **Schema** | Mixed - designed for 3-table schema |

**Defines:**
- `atom_text()`, `atom_reconstruct_text()` - delegates to C `semantic_reconstruct()`
- `atom_content_hash()` - DUPLICATED in 012
- `atom_find()` - DUPLICATED
- `db_stats()` - DUPLICATED in 021, 027
- `db_depth_distribution()`
- `db_index_status()`
- `get_atom()`, `get_atom_by_codepoint()`, `get_atoms_by_codepoints()`
- `hash_exists()`, `text_exists()`
- `hilbert_range()`, `hilbert_nearest()` - proper 128-bit handling
- `spatial_nearest()`
- `batch_insert_prepare()`, `batch_insert_finalize()`
- `validate_atoms()`

**Notes:**
- Well-documented with proper 128-bit Hilbert handling
- Core lookup functions
- `db_stats()` returns different columns than 021/027 versions

**Recommendation:** KEEP as canonical source; resolve db_stats() conflicts

---

#### 019_function_aliases.sql
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - Backward compatibility layer |
| **Lines** | 65 |
| **Action** | ✅ KEEP |
| **Schema** | N/A - pure aliases |

**Defines:**
- `atom_by_codepoint()` → `get_atom_by_codepoint()`
- `atoms_by_codepoints()` → `get_atoms_by_codepoints()`
- `atom_exists()` → `hash_exists()` - CONFLICTS with 025, 027
- `atom_find()` → `find_content()`
- `atom_find_exact()` → `find_content()`
- `ingestion_stats()` → `db_stats()` - REFERENCES wrong columns!
- `depth_distribution()` → `db_depth_distribution()`

**Issues:**
- Line 56: `ingestion_stats()` references `leaf_atoms, compositions, max_depth, db_size` 
  but 018's `db_stats()` returns different columns
- Creates aliases that conflict with function definitions elsewhere

**Recommendation:** KEEP but FIX `ingestion_stats()` column references

---

#### 020_four_tables.sql
| Attribute | Value |
|-----------|-------|
| **Status** | **PRODUCTION - SCHEMA DEFINITION** |
| **Lines** | 127 |
| **Action** | ✅ KEEP (CRITICAL) |
| **Schema** | DEFINES the 4-table schema |

**Defines Tables:**
- `atom` - Unicode codepoints (geom, NOT centroid)
- `composition` - Aggregations (centroid)
- `composition_child` - Ordered children
- `relation` - Semantic edges (source_id/target_id)

**Defines Functions:**
- `upsert_relation()` - with weight averaging

**Notes:**
- Despite the name "FOUR TABLES", creates 4 tables: atom, composition, composition_child, relation
- This is THE canonical schema definition
- Drops and recreates `relation` and `atom` tables!

**Recommendation:** KEEP - This is the canonical schema. Consider renaming for clarity.

---

#### 021_query_api.sql
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - Query API for 3-table schema |
| **Lines** | 301 |
| **Action** | ✅ KEEP |
| **Schema** | 3-table (atom/composition/relation with source_id/target_id) |

**Defines:**
- `composition_text()` - recursive tree traversal
- `text()` - alias
- `find_composition()`
- `get_centroid()`
- `similar_by_centroid()`
- `similar()` - alias (CONFLICTS with 016!)
- `related_by_attention()`
- `related()` - alias
- `generative_walk()` - PL/pgSQL
- `spatial_walk()` - PL/pgSQL
- `analogy(TEXT, TEXT, TEXT, INTEGER)` - delegates to `vector_analogy`
- `db_stats()` - DUPLICATED in 018, 027

**Notes:**
- Well-structured query API
- Uses `centroid_distance()` from 022
- `analogy()` properly delegates to `vector_analogy()` from 022

**Recommendation:** KEEP - Canonical query API layer

---

#### 022_generative_engine.sql
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - Generative/inference engine |
| **Lines** | 447 |
| **Action** | ✅ KEEP |
| **Schema** | 3-table with 4D centroids |

**Defines:**
- `centroid_distance()` - 4D Euclidean
- `centroid_similarity()` - inverse distance
- `hilbert_distance()` - 128-bit approximation
- `similar_tokens()`, `similar_tokens_fast()`
- `encode_prompt()` - tokenization
- `score_candidates()` - LLM-style scoring
- `generate_sql()` - generative walk (renamed to avoid C extension conflict)
- `vector_analogy()` - 4D vector arithmetic

**Notes:**
- Comprehensive generative engine
- `generate_sql()` renamed from `generate()` to avoid conflict with C extension
- Well-documented with clear 4D coordinate usage

**Recommendation:** KEEP - Core generative functionality

---

#### 023_bigram_stats.sql
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION |
| **Lines** | 223 |
| **Action** | ✅ KEEP |
| **Schema** | Supplementary tables for PMI scoring |

**Defines Tables:**
- `bigram_stats` - left_id → right_id counts
- `unigram_stats` - token counts
- `token_corpus_stats` - totals

**Defines Functions:**
- `increment_bigram()`, `increment_unigram()`
- `compute_pmi_scores()`
- `get_bigram_pmi()`, `get_bigram_count()`
- `top_continuations_pmi()`
- `extract_bigrams_from_compositions()` - INCOMPLETE (function body cut off)

**Notes:**
- Supports PMI scoring for generative engine
- `extract_bigrams_from_compositions()` appears incomplete in file

**Recommendation:** KEEP - Verify function completeness

---

#### 025_four_table_functions.sql
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - Core functions for 4-table schema |
| **Lines** | 343 |
| **Action** | ✅ KEEP (CANONICAL) |
| **Schema** | 4-table (atom/composition/composition_child/relation) |

**Defines:**
- Drops and recreates many core functions
- `atom_is_leaf()`, `atom_centroid()`, `atom_children()`
- `atom_child_count()`, `atom_by_codepoint()`, `atom_exists()`
- `atom_text()`, `atom_reconstruct_text()` - PL/pgSQL recursive
- `atom_knn()`, `atom_hilbert_range()`
- `semantic_neighbors(BYTEA, INTEGER)` - uses relation table
- `attention()`, `analogy(BYTEA, BYTEA, BYTEA, INTEGER)` - 4D geometry
- `atom_stats()` - function and view
- `atom_type_stats` view

**Notes:**
- Explicitly drops old functions before recreating
- Designed for the 4-table schema from 020
- **This should be the canonical source for these functions**

**Recommendation:** KEEP - Make this the canonical function definitions

---

#### 026_laplacian_projection.sql ⚠️ DUPLICATE NUMBER
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - Laplacian 4D projection helpers |
| **Lines** | 338 |
| **Action** | 🔢 RENUMBER to 028 |
| **Schema** | 4-table |

**Defines:**
- `upsert_atom_4d()`, `upsert_composition_4d()`
- `merge_atom_projections()`, `merge_composition_projections()`
- `atoms_near_hilbert()`, `compositions_near_hilbert()` - proper 128-bit
- `atom_coord_stats()`, `composition_coord_stats()`
- `validate_hilbert_indices()`

**Issues:**
- **Same file number as 026_model_registry.sql!**

**Recommendation:** RENUMBER to 028_laplacian_projection.sql

---

#### 026_model_registry.sql ⚠️ DUPLICATE NUMBER
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - Model tracking |
| **Lines** | 89 |
| **Action** | 🔢 RENUMBER to 029 |
| **Schema** | Supplementary table |

**Defines Table:**
- `model` - tracks ingested models

**Defines Functions:**
- `upsert_model()`

**Defines Views:**
- `model_stats`

**Issues:**
- **Same file number as 026_laplacian_projection.sql!**

**Recommendation:** RENUMBER to 029_model_registry.sql

---

#### 027_batch_operations.sql ⚠️ DUPLICATE NUMBER
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - Optimized batch operations |
| **Lines** | 499 |
| **Action** | ✅ KEEP (resolve number conflict) |
| **Schema** | Mixed - references both old and new |

**Defines:**
- PARALLEL SAFE versions of core functions
- `atom_is_leaf()`, `atom_centroid()`, etc. - REDEFINED with PARALLEL SAFE
- `get_atoms_by_codepoints()`, `get_atoms_by_ids()`
- `atoms_exist()`
- `extract_bigrams_batch()` - set-based extraction
- `db_stats()` - DUPLICATED

**Issues:**
- **Same file number as 027_centroid_4d.sql!**
- Redefines many functions from 025 with PARALLEL SAFE
- Uses OLD unified schema in some places (references `atom.depth`)

**Recommendation:** KEEP, but coordinate with 025 - perhaps merge PARALLEL SAFE declarations

---

#### 027_centroid_4d.sql ⚠️ DUPLICATE NUMBER
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - 4D centroid computation |
| **Lines** | ~130 |
| **Action** | 🔢 RENUMBER to 031 |
| **Schema** | 4-table (uses composition_child) |

**Defines:**
- `st_centroid_4d()` - true 4D centroid for PostGIS
- `compute_composition_centroid()` - from atom children
- `recompute_composition_centroids()` - batch update
- `generate_knn_edges()` - k-NN edge generation

**Notes:**
- Critical for 4D geometry
- Properly uses composition_child table

**Issues:**
- **Same file number as 027_batch_operations.sql!**

**Recommendation:** RENUMBER to 031_centroid_4d.sql

---

#### 030_qa_search.sql
| Attribute | Value |
|-----------|-------|
| **Status** | PRODUCTION - Q&A search functions |
| **Lines** | ~160 |
| **Action** | ✅ KEEP |
| **Schema** | Uses composition table |

**Defines:**
- `search_text()` - regex pattern search
- `ask()` - Q&A with evidence
- `ask_exact()` - exact phrase matching
- `find_entities()` - named entity extraction

**Notes:**
- High-level Q&A API
- Uses `atom_reconstruct_text()` for text retrieval

**Recommendation:** KEEP - User-facing search API

---

### tests/sql/ Directory

#### test_ingestion.sql
| Status | PRODUCTION TEST |
| Action | ✅ KEEP |

Tests text ingestion with `hypercube_ingest_text()`, `hypercube_retrieve_text()`, `hypercube_text_to_linestring()`. References C extensions.

---

#### test_semantic_validation.sql
| Status | PRODUCTION TEST |
| Action | ✅ KEEP |

Validates Unicode atom distribution on 3-sphere, case pair proximity, category clustering. References OLD schema columns (`coords` instead of `geom`, `category` column).

**Issues:** Uses `atom.coords` and `atom.category` - may need update for current schema.

---

#### test_three_table_schema.sql
| Status | PRODUCTION TEST |
| Action | ✅ KEEP |

Tests 4-table schema from 020. Despite name saying "three", it tests: atom, composition, composition_child, relation.

---

#### test_unified_schema.sql.DEPRECATED
| Status | DEPRECATED |
| Action | ✅ KEEP AS-IS |

Already correctly deprecated.

---

#### test_unified_schema_v2.sql.DEPRECATED
| Status | DEPRECATED |
| Action | ✅ KEEP AS-IS |

Already correctly deprecated.

---

## Function Duplication Analysis

### Most Duplicated Functions

| Function | Files Defining It |
|----------|------------------|
| `atom_is_leaf()` | 012, 025, 027 |
| `atom_centroid()` | 012, 025, 027 |
| `atom_exists()` | 012, 019, 025, 027 |
| `atom_by_codepoint()` | 012, 025, 027 |
| `atom_text()` | 012, 018, 025 |
| `atom_reconstruct_text()` | 012, 018, 025 |
| `atom_knn()` | 012, 025, 027 |
| `atom_content_hash()` | 012, 018 |
| `db_stats()` | 018, 021, 027 |
| `semantic_neighbors()` | 014, 016, 025 (different signatures!) |
| `analogy()` | 014, 016, 021, 025 (different signatures!) |
| `content_exists()` | 016, 017 |

### Signature Conflicts

| Function | File | Signature |
|----------|------|-----------|
| `semantic_neighbors` | 014 | `(BYTEA, INTEGER)` returns `neighbor_id, weight` |
| `semantic_neighbors` | 016 | `(TEXT, INTEGER)` returns `id, content, distance, depth` |
| `semantic_neighbors` | 025 | `(BYTEA, INTEGER)` returns `neighbor_id, weight, relation_type` |
| `analogy` | 014 | `(BYTEA, BYTEA, BYTEA, INTEGER)` returns `result_id, similarity` |
| `analogy` | 016 | `(TEXT, TEXT, TEXT, INTEGER)` returns STUB (NULL) |
| `analogy` | 021 | `(TEXT, TEXT, TEXT, INTEGER)` delegates to vector_analogy |
| `analogy` | 025 | `(BYTEA, BYTEA, BYTEA, INTEGER)` returns `result_id, similarity` |

---

## Recommended Actions Summary

### Files to ARCHIVE
| File | Reason |
|------|--------|
| 012_semantic_udf.sql | Superseded by 025_four_table_functions.sql |
| 015_centroid_optimization.sql | Obsolete for 4-table schema |
| 016_semantic_queries.sql | Contains stubs; functionality in 021/022 |

### Files to RENUMBER
| Current | New | Reason |
|---------|-----|--------|
| 026_laplacian_projection.sql | 028_laplacian_projection.sql | Duplicate 026 |
| 026_model_registry.sql | 029_model_registry.sql | Duplicate 026 |
| 027_centroid_4d.sql | 031_centroid_4d.sql | Duplicate 027 |

### Files to UPDATE/FIX
| File | Issue |
|------|-------|
| 014_ai_operations.sql | Uses old relation schema (parent_id/child_id) |
| 017_ingestion_ops.sql | `upsert_composition()` inserts into wrong table |
| 019_function_aliases.sql | `ingestion_stats()` references wrong columns |
| test_semantic_validation.sql | References `atom.coords` and `atom.category` |

### Files to KEEP (No Changes)
- 011_unified_atom.sql.DEPRECATED
- 018_core_queries.sql
- 020_four_tables.sql
- 021_query_api.sql
- 022_generative_engine.sql
- 023_bigram_stats.sql
- 025_four_table_functions.sql
- 027_batch_operations.sql
- 030_qa_search.sql
- test_ingestion.sql
- test_three_table_schema.sql

---

## Suggested Final File Structure

```
sql/
├── 011_unified_atom.sql.DEPRECATED
├── 012_semantic_udf.sql.DEPRECATED        # ARCHIVE
├── 014_ai_operations.sql                  # UPDATE schema refs
├── 015_centroid_optimization.sql.DEPRECATED  # ARCHIVE
├── 016_semantic_queries.sql.DEPRECATED    # ARCHIVE (stubs)
├── 017_ingestion_ops.sql                  # FIX function name
├── 018_core_queries.sql                   # KEEP - canonical lookups
├── 019_function_aliases.sql               # FIX column refs
├── 020_four_tables.sql                    # KEEP - SCHEMA DEFINITION
├── 021_query_api.sql                      # KEEP - query layer
├── 022_generative_engine.sql              # KEEP - inference
├── 023_bigram_stats.sql                   # KEEP - PMI support
├── 025_four_table_functions.sql           # KEEP - canonical functions
├── 028_laplacian_projection.sql           # RENAME from 026
├── 029_model_registry.sql                 # RENAME from 026
├── 027_batch_operations.sql               # KEEP - parallel ops
├── 030_qa_search.sql                      # KEEP - Q&A API
└── 031_centroid_4d.sql                    # RENAME from 027

tests/sql/
├── test_ingestion.sql
├── test_semantic_validation.sql           # UPDATE schema refs
├── test_three_table_schema.sql
├── test_unified_schema.sql.DEPRECATED
└── test_unified_schema_v2.sql.DEPRECATED
```

---

## Execution Order

For a fresh database, files should be executed in this order:
1. 020_four_tables.sql (schema)
2. 018_core_queries.sql (core lookups)
3. 021_query_api.sql (query layer)
4. 022_generative_engine.sql (inference)
5. 023_bigram_stats.sql (PMI tables)
6. 025_four_table_functions.sql (canonical functions)
7. 027_batch_operations.sql (parallel versions)
8. 028_laplacian_projection.sql (4D helpers)
9. 029_model_registry.sql (model tracking)
10. 030_qa_search.sql (Q&A API)
11. 031_centroid_4d.sql (centroid computation)
12. 019_function_aliases.sql (backward compat - LAST)

Files 014, 016, 017 are legacy and should not be executed with the 4-table schema.
