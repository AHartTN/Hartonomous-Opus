# Hartonomous-Opus Comprehensive Audit & Action Plan
**Date**: 2026-01-02 (Updated with Repository Cleanup Plan)
**Auditor**: Claude Opus 4.5

---

## 🔴 CRITICAL: Files to Archive/Remove

### C++ Source Files - DEPRECATED (Not Compiled)

| File | Lines | Issue | Action |
|------|-------|-------|--------|
| `cpp/src/hypercube.cpp` | 405 | Old C++ PG extension, replaced by `pg/hypercube_pg.c` | **ARCHIVE** |
| `cpp/src/semantic_ops.cpp` | 581 | Old C++ PG extension, replaced by `pg/semantic_ops_pg.c` | **ARCHIVE** |
| `cpp/src/extract_embeddings.cpp` | 682 | Superseded by `ingest_safetensor.cpp` | **ARCHIVE** |
| `cpp/src/deprecated/seed_atoms.cpp` | ~300 | Old single-threaded seeder | **DELETE** |
| `cpp/src/deprecated/seed_atoms_direct.cpp` | ~200 | Old direct seeder | **DELETE** |
| `cpp/src/deprecated/ingest_model.cpp` | ~400 | Old model ingester | **DELETE** |

### SQL Files - DEPRECATED (Old 3-Table Schema)

All 10 files in `sql/deprecated/` should be deleted - they reference the old 3-table schema (atom/relation/relation_edge) that was replaced by the unified atom table.

### Scripts - DEPRECATED

All 6 files in `scripts/deprecated/` should be deleted - superseded by `setup.sh` and platform-specific scripts in `linux/` and `windows/`.

---

## 🟠 STRUCTURAL ISSUES

### Monolithic Files Needing Decomposition

| File | Lines | Contains | Recommended Split |
|------|-------|----------|-------------------|
| `cpp/src/cpe_ingest.cpp` | 792 | Atom cache, file reader, CPE algorithm, DB batch | `cpe/atom_cache.cpp`, `cpe_encoder.cpp`, `db_batch.cpp` |
| `cpp/src/sequitur_ingest.cpp` | 1020 | Sequitur grammar, digram index, DB ingestion | `sequitur.cpp`, `sequitur_db.cpp` |
| `cpp/src/coordinates.cpp` | 651 | Unicode categorization, Hopf fibration, surface mapping | `unicode_category.cpp`, `hopf_fibration.cpp` |
| `sql/014_ai_operations.sql` | 588 | Attention, transform, infer, generate | `ai_attention.sql`, `ai_transform.sql`, etc. |

### Multiple Concerns Per File

| File | Issue |
|------|-------|
| `cpp/include/hypercube/types.hpp` | Contains 4 types (`Point4D`, `HilbertIndex`, `Blake3Hash`, `AtomCategory`) - should split |

### Embedded Structs (Should Be In Headers)

| File | Structs |
|------|---------|
| `cpp/src/cpe_ingest.cpp` | `AtomInfo` |
| `cpp/src/extract_embeddings.cpp` | `TensorInfo`, `SemanticEdge`, `TokenAtom` |
| `cpp/src/semantic_ops.cpp` | `TraverseNode`, `TraverseState`, `HashKey` |

---

## 🟡 DATABASE ISSUES

### Test Helpers Not Permanent

Test files create temporary functions that should be permanent:

```sql
-- These exist ONLY in test files, not in production schema:
test_assert(bool, text)
test_assert_equal(anyelement, anyelement, text)
assert_less_than(double, double, text)
```

**Recommendation**: Create `sql/099_test_infrastructure.sql` with:
```sql
CREATE SCHEMA IF NOT EXISTS test;
CREATE OR REPLACE FUNCTION test.assert(bool, text) ...
CREATE OR REPLACE FUNCTION test.assert_eq(anyelement, anyelement, text) ...
CREATE OR REPLACE FUNCTION test.validate_all() RETURNS TABLE(...) ...
```

### Schema Mismatch

`tests/test_unified_schema_v2.sql` references `centroid` column, but current schema uses `ST_Centroid(geom)`.

---

## Executive Summary

Hartonomous-Opus is a **universal semantic substrate** - a content-addressed 4D hypercube where ALL digital content (text, code, images, audio, ML models) is atomized into a single queryable space. The foundation is complete and working. Critical gaps exist in vocabulary-aware tokenization and the Sequitur grammar inference engine.

### The Vision (Confirmed Understanding)

This is NOT another RAG system. This is a **reinvention of AI** that replaces GPU matrix multiplication with spatial B-tree/R-tree queries:

```
All Digital Content
    └─→ Unicode codepoints (1.1M atoms = fixed landmarks on S³)
        └─→ Cascading Pair Encoding compositions
            └─→ Merkle DAG (content-addressed)
                └─→ 4D Hilbert-indexed space
                    └─→ Spatial proximity = semantic similarity
```

**Key Properties**:
- Language agnostic: Python = JavaScript = English = Chinese
- Modality agnostic: text = code = audio = video = ML weights
- Model agnostic: Query FLUX + Llama with ONE SQL statement
- Universal graph: Every piece of content ever created is a subgraph

---

## Current State Assessment

### ✅ WORKING (Production Ready)

| Component | Status | Performance |
|-----------|--------|-------------|
| Unicode atom seeding | ✅ | 1.1M in ~30s (parallel) |
| CPE ingestion | ✅ | ~1 MB/s, 14K comps/sec |
| Hilbert indexing | ✅ | 128-bit, sub-ms queries |
| BLAKE3 hashing | ✅ | Content-addressed |
| PostgreSQL/PostGIS | ✅ | Unified atom table |
| Spatial queries | ✅ | GIST index on centroids |
| Text reconstruction | ✅ | Bit-perfect DFS traversal |
| Test suite | ✅ | 22 tests passing |

### 🔴 BROKEN (Needs Fix)

| Component | Issue | Root Cause | Priority |
|-----------|-------|------------|----------|
| Sequitur ingester | Segfaults on "abab" | Use-after-free at line 331-335 | HIGH |
| Windows CMake | GCC flags on MSVC | `-O3`, `-march=native`, `pthread` | HIGH |

### ⚠️ INCOMPLETE (Needs Implementation)

| Component | Status | Impact |
|-----------|--------|--------|
| Safetensor weight extraction | Code exists, line 306 says "not yet implemented" | Can't extract attention matrices |
| Vocabulary-aware tokenizer | Described in SEMANTIC_WEB_DESIGN.md, not coded | Creates wasteful compositions |
| Case-insensitive queries | Frechet logic exists, not wired | Low |

---

## Critical Bug Analysis

### Sequitur Segfault (sequitur_ingest.cpp:331-335)

```cpp
// BUG: Use-after-free
container->remove(first->next);  // Modifies linked list
container->remove(first);         // first->next is now stale!
delete first->next;               // SEGFAULT: accessing freed memory
delete first;
```

**Fix**: Save pointers before removal:
```cpp
Symbol* second = first->next;     // Save before disconnect
container->remove(second);
container->remove(first);
delete second;
delete first;
```

### Windows CMake Issues (CMakeLists.txt:16-17)

```cmake
# BUG: GCC-only flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra -fsanitize=address,undefined")
```

**Fix**: Use generator expressions:
```cmake
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /W4")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /DDEBUG /W4 /Zi")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
endif()
```

Also: Remove `pthread` from Windows builds (use `Threads::Threads` via `find_package(Threads)`).

---

## D:\Models Ingestion Strategy

### Available Models

```
D:\Models\
├── embedding_models\
│   ├── models--sentence-transformers--all-MiniLM-L6-v2\
│   │   └── snapshots\*\
│   │       ├── vocab.txt           ← INGEST: token vocabulary
│   │       ├── tokenizer.json      ← INGEST: BPE merges → semantic edges
│   │       └── model.safetensors   ← FUTURE: attention weights
│   └── models--meta-llama--Llama-4-Maverick-17B-128E\
│
├── detection_models\
│   ├── Florence-2-base\
│   ├── Grounding-DINO-Base\
│   ├── DETR-ResNet-101\
│   └── yolo11x\
│
├── generation_models\
│   └── models--black-forest-labs--FLUX.2-dev\
│
└── temp-llama-models\
    └── llama_models\          ← INGEST: All Python code as CPE compositions
        ├── *.py               ← Code becomes semantic patterns
        ├── *.md               ← Documentation
        └── *.json             ← Configs
```

### Ingestion Plan

#### Phase 1: Vocabulary Bootstrap (vocab.txt + tokenizer.json)
```powershell
# Ingest MiniLM vocabulary and BPE merges
.\cpp\build\Release\ingest_safetensor.exe `
    -d hypercube -U hartonomous -h localhost `
    "D:\Models\embedding_models\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\*"
```

This creates:
- Token compositions (each vocab entry = composition)
- BPE merge edges (semantic relationships)

#### Phase 2: Code/Documentation Ingestion (CPE)
```powershell
# Ingest all llama_models Python code and docs
.\cpp\build\Release\cpe_ingest.exe `
    -d hypercube -U hartonomous -h localhost `
    "D:\Models\temp-llama-models\llama_models"
```

This creates:
- CPE compositions for all .py, .md, .json files
- Same code patterns across files → same hash → automatic deduplication

#### Phase 3: Cross-Model Semantic Queries
```sql
-- Find patterns shared between MiniLM vocab and Llama code
SELECT m.id as minilm_pattern, l.id as llama_pattern,
       ST_Distance(ST_Centroid(m.geom), ST_Centroid(l.geom)) as semantic_dist
FROM atom m
JOIN atom l ON ST_DWithin(ST_Centroid(m.geom), ST_Centroid(l.geom), 1000000)
WHERE m.depth > 0 AND l.depth > 0;

-- Diff Model A vs Model B (find unique patterns)
WITH model_a AS (
    SELECT id, atom_reconstruct_text(id) as text FROM atom WHERE ...
), model_b AS (
    SELECT id, atom_reconstruct_text(id) as text FROM atom WHERE ...
)
SELECT * FROM model_a WHERE id NOT IN (SELECT id FROM model_b);
```

---

## Action Plan (Priority Order)

### Phase 1: Critical Fixes (Week 1)

1. **Fix Sequitur segfault** (sequitur_ingest.cpp:331-335)
   - Save `first->next` before removal
   - Add unit tests for "abab", "aaaa", etc.

2. **Fix Windows CMake** (CMakeLists.txt)
   - MSVC-compatible flags
   - Replace pthread with Threads::Threads
   - Test build on Windows

3. **Create scripts/config.env** for Windows
   - Copy config.env.example
   - Set Windows PostgreSQL paths

### Phase 2: Model Ingestion (Week 2)

4. **Test ingest_safetensor** with real models
   - MiniLM vocab.txt + tokenizer.json
   - Verify BPE edges in database

5. **Bulk CPE ingestion** of D:\Models
   - llama_models Python code
   - Documentation files

6. **Implement safetensor weight reading**
   - Parse safetensor header JSON
   - Extract attention matrix structure (not values)
   - Create semantic edges from attention patterns

### Phase 3: Vocabulary-Aware Tokenizer (Week 3-4)

7. **Implement VocabTrie** (cpp/src/vocab_trie.hpp)
   - Load existing compositions into trie
   - O(m) longest-match lookup

8. **Create vocab_ingest.cpp**
   - Greedy tokenization against vocab trie
   - Only create NEW compositions for unknown patterns
   - Massive reduction in composition count

### Phase 4: Polish (Ongoing)

9. **Archive deprecated files**
   - sql/deprecated/
   - cpp/src/deprecated/
   - scripts/deprecated/

10. **Expand test suite**
    - Sequitur edge cases
    - Cross-model queries
    - Performance benchmarks

---

## Immediate Next Steps

To get D:\Models ingested TODAY:

```powershell
# 1. Source environment
. .\scripts\windows\env.ps1

# 2. Build (may fail on Windows - CMake needs fixing)
.\scripts\windows\build.ps1

# 3. If build fails, fix CMakeLists.txt first

# 4. Once built, ingest MiniLM vocabulary
.\cpp\build\Release\ingest_safetensor.exe -d hypercube -U hartonomous "D:\Models\embedding_models\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

# 5. Ingest llama_models code via CPE
.\cpp\build\Release\cpe_ingest.exe -d hypercube -U hartonomous "D:\Models\temp-llama-models\llama_models"
```

---

## Appendix: File Status Summary

### Current Files (Keep)
```
cpp/src/
  cpe_ingest.cpp           # Working CPE ingester
  seed_atoms_parallel.cpp  # Fast Unicode seeder
  ingest_safetensor.cpp    # Vocab/BPE ingester (weight reading incomplete)
  sequitur_ingest.cpp      # NEEDS FIX: segfault
  semantic_ops.cpp         # PostgreSQL UDFs
  hypercube.cpp            # PG extension core
  blake3_pg.cpp            # BLAKE3 for PG
  hilbert.cpp              # Hilbert curves
  coordinates.cpp          # Coordinate mapping

sql/
  011_unified_atom.sql     # Main schema
  012_semantic_udf.sql     # Semantic functions
  013_model_infrastructure.sql  # AI model tables
  014_ai_operations.sql    # AI/ML operations
  015_centroid_optimization.sql # Centroid queries
```

### Deprecated Files (Archive)
```
sql/001-010_*.sql          # Old schema versions
cpp/src/seed_atoms.cpp     # Old single-threaded
cpp/src/extract_embeddings.cpp  # Superseded
scripts/deprecated/        # Old scripts
```

---

*This audit confirms Hartonomous-Opus is architecturally sound. The critical path to ingesting D:\Models is: Fix Windows CMake → Build → Test safetensor → Bulk CPE ingest.*

---

## 📋 REPOSITORY CLEANUP CHECKLIST

### Phase 1: Immediate Cleanup (Safe Deletes)

```powershell
# Delete deprecated C++ source files
Remove-Item -Recurse -Force cpp\src\deprecated

# Archive old C++ PG extensions (keep for reference)
New-Item -ItemType Directory -Path cpp\src\archive -Force
Move-Item cpp\src\hypercube.cpp cpp\src\archive\
Move-Item cpp\src\semantic_ops.cpp cpp\src\archive\
Move-Item cpp\src\extract_embeddings.cpp cpp\src\archive\

# Delete deprecated SQL (old schema)
Remove-Item -Recurse -Force sql\deprecated

# Delete deprecated scripts
Remove-Item -Recurse -Force scripts\deprecated
```

- [ ] Delete `cpp/src/deprecated/*` (3 files)
- [ ] Archive `cpp/src/hypercube.cpp`, `semantic_ops.cpp`, `extract_embeddings.cpp`
- [ ] Delete `sql/deprecated/*` (10 files)  
- [ ] Delete `scripts/deprecated/*` (6 files)

### Phase 2: Create Permanent Test Infrastructure

- [ ] Create `sql/099_test_infrastructure.sql` with test schema utilities
- [ ] Migrate test helpers from inline definitions to permanent schema
- [ ] Fix `test_unified_schema_v2.sql` centroid column reference

### Phase 3: Refactor Monolithic Files (Future)

- [ ] Split `types.hpp` into separate headers per type
- [ ] Split `coordinates.cpp` into focused modules
- [ ] Split `cpe_ingest.cpp` into subdirectory modules

---

## Summary Metrics

| Category | Current | After Cleanup |
|----------|---------|---------------|
| C++ source files (cpp/src) | 14 | 8 + 3 archived |
| SQL schema files | 15 | 6 |
| Total deprecated files | 19 | 0 |
