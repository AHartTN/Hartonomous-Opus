# Comprehensive Ingester Audit - January 5, 2026

## Summary

The `ingest_safetensor.cpp` ingester is fundamentally broken due to:
1. **Syntax errors** - Dead code left after "DISABLED" comments prevents compilation
2. **Architecture violations** - Creating fake atoms instead of using existing token atoms/compositions
3. **Missing functionality** - Laplacian projection not integrated properly
4. **Performance disasters** - O(n²) code paths that explode on large vocabs

---

## Part 1: Compile Errors (BUILD BLOCKERS)

### Issue 1.1: Dead Code After DISABLED Comment
**File:** `ingest_safetensor.cpp` lines 1480-1657
**Problem:** The "DISABLED" comment just prints a message, then 180+ lines of unreachable code follow using undefined variables (`vocab_size`, `num_threads`, `embed_dim`)

```cpp
// Line 1485-1486:
std::cerr << "[SIMILARITY] Skipped - use SQL generate_knn_edges() on atom centroids instead\n";
    
    std::cerr << "[SIMILARITY] Computing pairwise similarity for " << vocab_size  // UNDEFINED!
```

**Impact:** 100+ compile errors cascade from this

### Issue 1.2: Broken Weight Tensor Loop
**File:** `ingest_safetensor.cpp` line 1657+
**Problem:** PART 2 loop is outside any function scope after the compile errors break parsing

---

## Part 2: Architectural Violations

### Issue 2.1: Fake Atom Creation
**File:** `ingest_safetensor.cpp` lines 1720-1725
**Problem:** Creating fake atoms for "tensor row N" and "tensor column M" - these aren't knowledge, they're noise

```cpp
std::string from_key = config.model_name + ":" + tensor->name + ":r" + std::to_string(r);
std::string to_key = config.model_name + ":" + tensor->name + ":c" + std::to_string(c);

auto from_hash = AtomCalculator::compute_vocab_token(from_key).hash;  // FAKE ATOM!
```

**Correct approach:** Tokens from vocab ARE the atoms. Weight matrices define relationships between them, not new entities.

### Issue 2.2: Embeddings Stored as Shapes, Not Used for Projection
**File:** `ingest_safetensor.cpp` - `insert_shapes()` function
**Problem:** Currently stores embedding vectors as PostGIS LineStrings on compositions, but doesn't use them to compute 4D coordinates via Laplacian projection

**What should happen:**
1. Read `embed_tokens.weight` - each row is a token's N-dimensional embedding
2. Build sparse similarity graph from embeddings (only edges > threshold)
3. Compute unnormalized Laplacian L = D - W
4. Find 4 smallest non-zero eigenvectors (Lanczos)
5. Gram-Schmidt orthonormalize
6. Each token's 4D coords = its values in those 4 eigenvectors
7. Store 4D coords on existing atoms/compositions (NOT as new entities)

### Issue 2.3: Step [5] Calls Wrong Function
**File:** `ingest_safetensor.cpp` lines 1175-1185
**Problem:** Step [5] calls `recompute_composition_centroids()` which computes centroids from constituent atom coordinates - but the atoms haven't had their coordinates computed yet!

**Order should be:**
1. Parse tokenizer → create atoms/compositions (structure only)
2. Read embeddings → build Laplacian → eigenvectors → 4D coords for ALL tokens
3. Store 4D coords on atoms/compositions
4. THEN compute composition centroids from their constituent atoms

---

## Part 3: Missing/Broken Functionality

### Issue 3.1: Laplacian Projector Not Called
**File:** `ingest_safetensor.cpp`
**Problem:** `project_and_update_embeddings()` is defined but never called in `main()`

Step [5] in main just calls SQL `recompute_composition_centroids()` instead of the proper Laplacian projection.

### Issue 3.2: Router Weights Not Extracted
**File:** `ingest_safetensor.cpp`
**Problem:** MoE models have `router.weight` tensors that define which tokens activate which experts. These are sparse relationships (top-k routing) that should become edges.

Currently the code tries to extract ALL weight matrices cell-by-cell, which is:
1. Wrong conceptually
2. Impossible performance-wise (128 experts × 5120 × 16384 = 10B cells)

### Issue 3.3: 3D Tensor Handling Added But Broken
**File:** `ingest_safetensor.cpp` lines 1660-1663
**Problem:** Recent edit added 3D tensor handling but it's syntactically broken and conceptually wrong

---

## Part 4: Performance Issues

### Issue 4.1: O(n²) Similarity Code Still Present
**File:** `ingest_safetensor.cpp` lines 1487-1650
**Problem:** Even though "DISABLED", the code is still there and would compute pairwise similarity for all vocab pairs

For LLaMA 4 with 202,048 tokens: 202,048² = 40+ BILLION comparisons

### Issue 4.2: Weight Matrix Cell-by-Cell Extraction
**File:** `ingest_safetensor.cpp` PART 2
**Problem:** Tries to create an edge for every cell in every weight matrix above threshold

For a single Q projection [5120, 5120]: up to 26M edges PER TENSOR

---

## Part 5: What Should Happen (Correct Architecture)

### Step 1: Parse Tokenizer
- `tokenizer.json` → BPE merges → Composition 'C' relations
- Each token becomes an atom (single codepoint) or composition (merged tokens)
- Store the parent-child relationships in `children` array

### Step 2: Project Embeddings to 4D
- Read `embed_tokens.weight` tensor
- Build k-NN similarity graph (k=15) using HNSWLIB - O(n log n)
- Compute sparse Laplacian L = D - W
- Lanczos eigensolver → 4 smallest non-trivial eigenvectors
- Gram-Schmidt orthonormalize the columns
- Each token's 4D coordinate = its row in the eigenvector matrix
- Update atoms/compositions with their 4D coordinates

### Step 3: Compute Composition Centroids
- NOW call `recompute_composition_centroids()`
- Each composition's centroid = average of its constituent atoms' 4D coords

### Step 4: Extract Sparse Relations (router.weight only)
- For MoE models: `router.weight` [num_experts, hidden_dim]
- Each row is an expert's routing vector
- Create EXPERT atoms (one per expert)
- Sparse edges: token → expert where routing weight > threshold
- This is O(vocab × num_experts) with sparsity, not O(n²)

### Step 5: Store Embeddings as Geometry (Optional)
- If PostGIS LineString storage is desired, convert embedding to geometry
- This is the raw N-dimensional fingerprint, separate from 4D coords

---

## Files Affected

| File | Status | Issues |
|------|--------|--------|
| `ingest_safetensor.cpp` | BROKEN | Dead code, fake atoms, missing projection |
| `laplacian_4d.cpp` | FUNCTIONAL | Needs integration |
| `lanczos.cpp` | FUNCTIONAL | Working eigensolver |
| `insert_compositions()` | OK | Correctly creates atoms/compositions |
| `project_and_update_embeddings()` | NOT CALLED | Dead code |
| `insert_attention_relations()` | BROKEN | Wrong approach entirely |

---

## Fix Priority

1. **Delete dead code** - Remove all code after "DISABLED" comment through end of PART 1
2. **Delete PART 2** - Remove fake atom weight extraction
3. **Call Laplacian projection** - Wire `project_and_update_embeddings()` into main flow
4. **Reorder steps** - Projection THEN centroid computation
5. **Add router extraction** - Create expert atoms and sparse routing edges
6. **Test with MiniLM** - Small model first
7. **Test with LLaMA 4** - Full 55-shard model

---

## Test Data Available

- `test-data/moby_dick.txt` - Text ingestion
- `test-data/simple.txt` - Minimal test
- `D:\Models\embedding_models\models--sentence-transformers--all-MiniLM-L6-v2` - Small model (80MB)
- `D:\Models\embedding_models\models--meta-llama--Llama-4-Maverick-17B-128E` - Large MoE model (800GB)
