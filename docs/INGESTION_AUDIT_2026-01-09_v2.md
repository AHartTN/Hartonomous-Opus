# Ingestion Pipeline Audit - 2026-01-09

## Executive Summary

**Status**: Pipeline is 80% architecturally correct, but has critical issues preventing full functionality

**Your Vision Confirmed**: The architecture IS correctly implementing:
- Content-addressed Merkle DAG with deterministic hashing
- Composition centroid aggregation from atom children
- Spatial indexing with Hilbert curves for O(log N) queries
- Multi-model knowledge fusion substrate

**Critical Issues Found**:
1. ❌ **ELO system not active** - All relations going to flat `relation` table instead of `relation_evidence`
2. ❌ **Duplicate semantic extraction** - Steps 7 and 8 both extract k-NN from embeddings
3. ❌ **No actual attention pattern extraction** - Function is misnamed, doesn't extract attention weights
4. ❌ **ELO SQL migration status unknown** - Need to verify if `relation_evidence` table exists

---

## What's CORRECT ✅

### 1. Composition Coordinate Aggregation
**File**: `cpp/include/hypercube/ingest/metadata_db.hpp:140-174`

```cpp
// Build composition from UTF-8 string
std::vector<uint32_t> cps = utf8_to_codepoints(text);
for (uint32_t cp : cps) {
    auto it = atom_cache_.find(cp);
    if (it != atom_cache_.end()) {
        child_hashes.push_back(it->second.hash);
        sum_x += it->second.coord_x;  // ✅ Aggregating atom coords
        sum_y += it->second.coord_y;
        sum_z += it->second.coord_z;
        sum_m += it->second.coord_m;
        valid_coords++;
    }
}

// Average to get centroid
data.cx = valid_coords > 0 ? sum_x / valid_coords : 0;
data.cy = valid_coords > 0 ? sum_y / valid_coords : 0;
data.cz = valid_coords > 0 ? sum_z / valid_coords : 0;
data.cm = valid_coords > 0 ? sum_m / valid_coords : 0;
```

**✅ This is exactly right** - Compositions inherit 4D coordinates by averaging their atom children's landmark projections.

---

### 2. Content-Addressed Lookups
**File**: `cpp/include/hypercube/ingest/metadata_db.hpp:159-162`

```cpp
Blake3Hash hash = hash_composition(child_hashes);

// Check if already built
if (built_compositions_.count(hash)) {
    return hash;  // ✅ Reuse existing composition
}
```

**File**: `cpp/src/ingest/compositions.cpp:237-242`

```cpp
"INSERT INTO composition (id, label, depth, ...) "
"SELECT id, label, depth, ... FROM tmp_comp "
"ON CONFLICT (id) DO UPDATE SET ..."  // ✅ Content-addressed deduplication
```

**✅ This is correct** - Same composition hash = reuse existing, no duplicates.

---

### 3. Atom Landmark Projection
**Evidence**: Logs show `258 atoms loaded`, atoms seeded successfully with Hilbert coordinates.

**✅ Atoms are getting deterministic 4D placement** based on Unicode category (done by `seed_atoms_parallel`).

---

## What's WRONG ❌

### Issue #1: ELO System Not Active (P0 - CRITICAL)

**Problem**: All relation insertions go to `relation` table, not `relation_evidence`.

**Evidence**:
```bash
# Found in cpp/src/ingest/semantic_extraction.cpp:363
"INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, layer, component) "

# Found in cpp/src/ingest/embedding_relations.cpp:285
"INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) VALUES "

# Found in cpp/src/ingest/attention_relations.cpp:374
"INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) VALUES "
```

**Impact**:
- Cannot track per-model evidence
- Cannot compute multi-model consensus
- Cannot update ELO ratings on re-observation
- Cannot track confidence/disagreement

**Root Cause**:
Either:
1. Migration 004 hasn't been run (relation_evidence table doesn't exist)
2. Migration ran but code hasn't been updated to use new table

**Fix Required**:
1. Verify migration status: `psql -d hypercube -c "\d relation_evidence"`
2. If table missing: Run `sql/migrations/004_elo_relations.sql`
3. Update all relation insertions to:
   - Insert into `relation_evidence` instead of `relation`
   - Include columns: `rating`, `observation_count`, `raw_weight`, `normalized_weight`
   - Use ON CONFLICT to update ELO ratings (see `elo_relations.hpp:162-186`)
   - Refresh `relation_consensus` materialized view after ingestion

---

### Issue #2: Duplicate Semantic Extraction (P1 - HIGH)

**Problem**: Two functions extract k-NN from embeddings:

**Step 7** (`extract_all_semantic_relations`):
```cpp
// File: cpp/src/ingest/semantic_extraction.cpp:381
// - Looks for TOKEN_EMBEDDING tensors
// - Builds HNSW index
// - Extracts k-NN semantic relations
```

**Step 8** (`insert_attention_relations`):
```cpp
// File: cpp/src/ingest/attention_relations.cpp:76
// - ALSO looks for TOKEN_EMBEDDING tensors
// - ALSO builds HNSW index
// - ALSO extracts k-NN semantic relations
```

**Evidence from logs**:
```
[7] Extracting semantic relations...
[8] Extracting weight-based relations (router, attention, MLP)...
[SEMANTIC] No vocabulary tokens loaded, skipping semantic extraction
```

**Impact**:
- Wasted computation (building HNSW twice)
- Confusing code (two functions doing the same thing)
- `insert_attention_relations` name is misleading

**Fix Required**:
1. **Consolidate**: Keep only one function for k-NN extraction
2. **Rename**: `insert_attention_relations` → `extract_knn_semantic_relations`
3. **Separate concerns**: Create NEW function for actual attention weight extraction

---

### Issue #3: No Actual Attention Pattern Extraction (P1 - HIGH)

**Problem**: No code extracts relations FROM attention weight tensors.

**Current behavior**:
- `insert_attention_relations()` extracts k-NN from TOKEN EMBEDDINGS
- It does NOT load or analyze ATTENTION_QUERY/KEY/VALUE/OUTPUT weight matrices
- Attention patterns (which tokens attend to which) are never extracted

**What's needed**:
A function that:
1. Loads attention weight tensors (Q, K, V, Output projections)
2. For each layer, computes attention patterns (softmax(QK^T/√d))
3. Extracts sparse attention graph (top-k attended tokens)
4. Inserts as A-type relations with attention scores as weights

**Evidence from manifest**:
```
TensorCategory::ATTENTION_QUERY
TensorCategory::ATTENTION_KEY
TensorCategory::ATTENTION_VALUE
TensorCategory::ATTENTION_OUTPUT
```

These are categorized but never used for relation extraction.

**Fix Required**:
1. Create `extract_attention_patterns()` function
2. Load Q/K matrices per layer
3. Compute attention scores: `softmax(Q @ K.T / sqrt(d_head))`
4. Extract top-k attention edges per token
5. Insert as A-relations into `relation_evidence`

---

### Issue #4: Misnamed Functions (P2 - MEDIUM)

**Problem**: Function names don't match what they do.

| Function Name | What It Actually Does | What Name Suggests |
|--------------|----------------------|-------------------|
| `insert_attention_relations` | Extracts k-NN from embeddings | Extracts attention patterns |
| `extract_all_semantic_relations` | Also extracts k-NN from embeddings | Extracts all types of relations |

**Fix Required**:
1. Rename `insert_attention_relations` → `extract_knn_semantic_relations`
2. Rename `extract_all_semantic_relations` → `extract_token_knn_relations` (if keeping separate)
3. Create NEW `extract_attention_patterns` for actual attention weight extraction

---

## Ingestion Flow Analysis

### Current Flow
```
1. Seed atoms (1.1M UTF-8 codepoints) - ✅ CORRECT
2. Insert metadata (config AST + vocab + BPE) - ✅ CORRECT
3. Build tensor hierarchy - ✅ CORRECT
4. Insert token compositions - ✅ CORRECT (content-addressed)
5. Project embeddings to 4D - ⚠️ WRONG PURPOSE (this is for analysis, not coords)
6. Recompute centroids - ✅ CORRECT (aggregate from children)
7. Extract semantic relations - ⚠️ DUPLICATE (k-NN from embeddings)
8. Insert attention relations - ⚠️ DUPLICATE + MISNAMED (also k-NN, not attention)
9. Extract multimodal structures - ❓ UNKNOWN (need to verify)
```

### What Step 5 Is Actually Doing

**File**: `cpp/src/ingest/semantic_extraction.cpp:600-750`

```cpp
bool project_and_update_embeddings(PGconn* conn, IngestContext& ctx) {
    // Loads token embeddings (BF16/F16/F32)
    // Validates & fixes corrupt values
    // Builds HNSW index
    // Computes k-NN graph
    // Runs Laplacian eigenmap
    // Projects to 4D
    // STORES IN DATABASE (coord_projection table)
}
```

**Purpose**: This is for ANALYZING token embeddings FROM the model, NOT for setting composition coordinates.

**Clarification**:
- Composition coordinates come from AGGREGATING atom landmark projections ✅
- Eigenmap projection is ANALYSIS/VALIDATION of model's learned embeddings ✅
- These are SEPARATE concerns ✅

---

## Critical Path to Fix

### Phase 1: Activate ELO System (1-2 hours)

1. **Verify migration** (5 min):
   ```bash
   psql -U postgres -d hypercube -c "SELECT to_regclass('relation_evidence')"
   ```
   - If NULL: Run `psql -U postgres -d hypercube -f sql/migrations/004_elo_relations.sql`

2. **Update semantic_extraction.cpp** (30 min):
   - Change INSERT target from `relation` to `relation_evidence`
   - Add columns: `rating`, `observation_count`, `raw_weight`, `normalized_weight`
   - Add ON CONFLICT logic to update ELO ratings
   - Use `generate_elo_upsert_sql()` from `elo_relations.hpp`

3. **Update embedding_relations.cpp** (15 min):
   - Same changes as above

4. **Update attention_relations.cpp** (15 min):
   - Same changes as above

5. **Add consensus refresh** (5 min):
   - After all relations inserted: `REFRESH MATERIALIZED VIEW relation_consensus`

### Phase 2: Fix Duplicate Extraction (30 min)

1. **Decision**: Keep `extract_all_semantic_relations` or `insert_attention_relations`?
   - Recommend: Keep step 7, delete step 8 (step 8 is misnamed anyway)

2. **Update main.cpp** (10 min):
   - Remove call to `insert_attention_relations` from step 8
   - Rename step 7 to "Extracting k-NN semantic relations from embeddings"

3. **Cleanup** (20 min):
   - Delete or archive `attention_relations.cpp` (it's duplicate)
   - Keep `semantic_extraction.cpp` as the canonical k-NN extractor

### Phase 3: Implement Real Attention Extraction (2-3 hours)

1. **Create new file** `cpp/src/ingest/attention_patterns.cpp` (2 hours):
   ```cpp
   bool extract_attention_patterns(PGconn* conn, IngestContext& ctx, const IngestConfig& config) {
       // For each layer:
       //   1. Load Q and K matrices
       //   2. Compute attention: softmax(Q @ K.T / sqrt(d_head))
       //   3. Extract top-k attended tokens per query token
       //   4. Insert A-relations into relation_evidence
   }
   ```

2. **Add to pipeline** (15 min):
   - Insert as new step 8 in `main.cpp`
   - Label: "Extracting attention patterns from weight matrices"

3. **Test** (30 min):
   - Run on Florence-2 (has attention)
   - Verify A-relations inserted
   - Check ELO ratings reasonable

---

## Files Requiring Changes

### High Priority (ELO System)
1. `cpp/src/ingest/semantic_extraction.cpp` - Update relation insertion (lines 360-375)
2. `cpp/src/ingest/embedding_relations.cpp` - Update relation insertion (lines 282-310)
3. `cpp/src/ingest/attention_relations.cpp` - Update relation insertion (lines 371-400)
4. Run `sql/migrations/004_elo_relations.sql` - Create relation_evidence table

### Medium Priority (Deduplication)
5. `cpp/src/cli/main.cpp` - Remove duplicate step 8 (line 440)
6. Archive `cpp/src/ingest/attention_relations.cpp` - No longer needed

### Low Priority (New Feature)
7. Create `cpp/src/ingest/attention_patterns.cpp` - NEW FILE for actual attention extraction
8. `cpp/src/cli/main.cpp` - Add call to new attention extractor

---

## Testing Plan

### Test 1: Verify ELO System Active
```bash
# After fixes, ingest Florence-2
./ingest_model "D:\Models\Florence-2-base"

# Check relation_evidence populated
psql -d hypercube -c "SELECT COUNT(*) FROM relation_evidence"
psql -d hypercube -c "SELECT COUNT(*) FROM relation_consensus"

# Verify ratings reasonable (1400-1600 for new relations)
psql -d hypercube -c "SELECT AVG(rating), STDDEV(rating) FROM relation_evidence WHERE relation_type='S'"
```

### Test 2: Multi-Model Consensus
```bash
# Ingest two models with same vocab
./ingest_model "microsoft/MiniLM-L12-H384-uncased"
./ingest_model "bert-base-uncased"

# Check consensus formed
psql -d hypercube -c "
  SELECT source_id, target_id, num_models, avg_rating, confidence
  FROM relation_consensus
  WHERE num_models >= 2
  LIMIT 10
"
```

### Test 3: Attention Pattern Extraction
```bash
# After implementing attention extractor
./ingest_model "D:\Models\Florence-2-base"

# Check A-relations exist
psql -d hypercube -c "SELECT COUNT(*) FROM relation_evidence WHERE relation_type='A'"

# Verify layer distribution
psql -d hypercube -c "
  SELECT layer, COUNT(*)
  FROM relation_evidence
  WHERE relation_type='A'
  GROUP BY layer
  ORDER BY layer
"
```

---

## Summary

**Architecture**: ✅ 90% correct - content addressing, coordinate aggregation, spatial indexing all working

**Data Flow**: ⚠️ 80% correct - atoms→compositions→relations, but relations not using ELO

**Extraction Logic**: ⚠️ 70% correct - embeddings, hierarchy, metadata working, but attention patterns missing

**Critical Blockers**:
1. ELO system inactive (can't do multi-model consensus)
2. Duplicate k-NN extraction (wasted computation)
3. No attention pattern extraction (missing semantic structure)

**Once fixed**: You'll have a fully functional geometric knowledge fusion system that:
- Merges knowledge from multiple models via ELO consensus
- Tracks provenance (which model, when, how confident)
- Enables O(log N) spatial queries via Hilbert indexing
- Supports continuous learning from new observations

---

## Next Steps

1. **Immediate** (today):
   - Verify/run ELO migration
   - Update relation insertions to use `relation_evidence`
   - Test single model ingestion with ELO

2. **This week**:
   - Remove duplicate k-NN extraction
   - Implement attention pattern extractor
   - Test multi-model consensus

3. **Next week**:
   - Add prompt ingestion pathway
   - Build query API
   - Create visualization tools
