# Hartonomous Hypercube - Architecture Document

**Last Updated**: 2026-01-05  
**Status**: Canonical - 3-Table Schema with Laplacian Eigenmaps (v6)

---

## Executive Summary

The Hypercube is a **deterministic, lossless, content-addressable geometric semantic substrate**. All digital content is decomposed into a Merkle DAG stored in PostgreSQL/PostGIS with 4D spatial indexing:

1. **Atoms** (~1.1M Unicode codepoints) are fixed landmarks at 4D coordinates (POINTZM)
2. **Compositions** aggregate atoms/compositions with 4D Laplacian-projected centroids
3. **Relations** store semantic edges (attention, PMI, sequence) between nodes
4. **Deduplication** is global - identical patterns share the same hash
5. **Reconstruction** is bit-perfect via ordered traversal of composition_child

### The Three Table Model

| Table | Purpose | Row Count |
|-------|---------|-----------|
| `atom` | Unicode codepoints (leaves only) | ~1.1M fixed |
| `composition` | Aggregations (BPE tokens, words, phrases) | Grows with content |
| `relation` | Semantic edges (attention, PMI, sequence) | Grows with models |

Supporting junction table:

| Table | Purpose |
|-------|---------|
| `composition_child` | Ordered children of compositions |

**Key Innovation**: N-dimensional embeddings from AI models are projected to 4D during ingestion via **Laplacian Eigenmaps + Gram-Schmidt orthonormalization**. Raw embeddings NEVER touch the database - only 4D coordinates.

---

## Core Principles

### 1. Determinism
- Same bytes → same composition ID, always
- No randomness, no floating-point approximations
- Hash = BLAKE3 of canonical ordered child hashes concatenated

### 2. Losslessness
- Bit-perfect reconstruction from composition
- DFS traversal of composition_child → original byte sequence
- All coordinates stored with full precision in POINTZM geometry

### 3. Global Deduplication
- "the" from Moby Dick = "the" from Python docs = same composition ID
- First ingest creates the pattern; subsequent ingests reference it
- The more you ingest, the more deduplication occurs

### 4. 4D Laplacian Projection
- Model embeddings (384D, 768D, 4096D) are projected to 4D
- Uses Laplacian Eigenmaps for structure-preserving projection
- Gram-Schmidt orthonormalization ensures orthogonal axes
- Spatial proximity in 4D = semantic similarity

### 5. Hypersphere Geometry
- Atoms: On the surface of the 4D hypersphere (YOUR coordinate system)
- Compositions: Centroids move inward as depth increases
- Origin = most abstract/complex, Perimeter = most atomic

---

## Data Model

### 1. Atom Table (Leaves Only)
```sql
CREATE TABLE atom (
    id              BYTEA PRIMARY KEY,              -- BLAKE3(codepoint bytes)
    codepoint       INTEGER NOT NULL UNIQUE,        -- Unicode codepoint (0-0x10FFFF)
    value           BYTEA NOT NULL,                 -- UTF-8 bytes of the character
    geom            GEOMETRY(POINTZM, 0) NOT NULL,  -- YOUR 4D coordinate mapping
    hilbert_lo      BIGINT NOT NULL,                -- Hilbert index (low 64 bits)
    hilbert_hi      BIGINT NOT NULL,                -- Hilbert index (high 64 bits)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```
**~1.1M rows seeded once via Hopf fibration coordinate mapping.**

### 2. Composition Table (Aggregations)
```sql
CREATE TABLE composition (
    id              BYTEA PRIMARY KEY,              -- BLAKE3(child_ids concatenated)
    label           TEXT,                           -- Human-readable (e.g., "whale", "##ing")
    depth           INTEGER NOT NULL DEFAULT 1,     -- 1 = direct atom children, 2+ = nested
    child_count     INTEGER NOT NULL,               -- Number of direct children
    atom_count      BIGINT NOT NULL,                -- Total leaf atoms in subtree
    geom            GEOMETRY(LINESTRINGZM, 0),      -- Path through child centroids
    centroid        GEOMETRY(POINTZM, 0),           -- 4D centroid (Laplacian projected)
    hilbert_lo      BIGINT,                         -- Hilbert index (low 64 bits)
    hilbert_hi      BIGINT,                         -- Hilbert index (high 64 bits)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 3. Composition Children (Junction Table)
```sql
CREATE TABLE composition_child (
    composition_id  BYTEA NOT NULL REFERENCES composition(id),
    ordinal         SMALLINT NOT NULL,              -- Position in sequence (0-based)
    child_type      CHAR(1) NOT NULL,               -- 'A' = atom, 'C' = composition
    child_id        BYTEA NOT NULL,                 -- References atom.id or composition.id
    PRIMARY KEY (composition_id, ordinal)
);
```

### 4. Relation Table (Semantic Edges)
```sql
CREATE TABLE relation (
    id              BIGSERIAL PRIMARY KEY,
    source_type     CHAR(1) NOT NULL,               -- 'A' = atom, 'C' = composition
    source_id       BYTEA NOT NULL,                 -- References atom.id or composition.id
    target_type     CHAR(1) NOT NULL,               -- 'A' = atom, 'C' = composition
    target_id       BYTEA NOT NULL,                 -- References atom.id or composition.id
    relation_type   CHAR(1) NOT NULL,               -- S=sequence, A=attention, P=proximity
    weight          REAL NOT NULL DEFAULT 1.0,      -- Edge weight/strength
    source_model    TEXT NOT NULL DEFAULT '',       -- Which model contributed this edge
    source_count    INTEGER NOT NULL DEFAULT 1,     -- Occurrence count (for averaging)
    layer           INTEGER NOT NULL DEFAULT -1,    -- Model layer (-1 = N/A)
    component       TEXT NOT NULL DEFAULT '',       -- Model component (attention, mlp)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    UNIQUE (source_id, target_id, relation_type, source_model, layer, component)
);
```

**Relation Types:**
- `S` = Sequence (document order)
- `A` = Attention (from model attention matrices)
- `P` = Proximity (spatial neighborhood)
- `T` = Temporal (time-based relationships)

---

## Ingestion Pipeline

### Text/Content Ingestion
```
Input → UTF-8 → Codepoints → Greedy Vocabulary Match → Compositions
```

1. Content decoded to UTF-8 codepoints
2. Codepoints mapped to existing atoms
3. PMI/CPE algorithm creates compositions
4. Compositions stored with geometry computed from children

### Model Ingestion (Safetensor)
```
Safetensor → Token Embeddings → Laplacian Eigenmaps → 4D Centroids → Compositions
```

1. Parse safetensor metadata to find embedding matrices
2. Extract token embeddings (N-dimensional vectors)
3. Build k-NN graph for Laplacian matrix
4. Compute top 4 eigenvectors via Lanczos iteration
5. Apply Gram-Schmidt orthonormalization
6. Store as 4D centroids on compositions

---

## File Structure

```
Hartonomous-Opus/
├── cpp/
│   ├── CMakeLists.txt          # Build configuration
│   ├── include/hypercube/      # Header files
│   │   ├── types.hpp           # Core type definitions
│   │   ├── coordinates.hpp     # Hopf fibration mapping
│   │   ├── hilbert.hpp         # 128-bit Hilbert curve
│   │   ├── blake3.hpp          # BLAKE3 hashing
│   │   ├── lanczos.hpp         # Eigensolver
│   │   ├── laplacian_4d.hpp    # 4D projection
│   │   └── db/                 # Database abstractions
│   │       ├── atom_cache.hpp
│   │       ├── geometry.hpp
│   │       └── insert.hpp
│   ├── src/
│   │   ├── pg/                 # PostgreSQL extensions (pure C)
│   │   │   ├── hypercube_pg.c
│   │   │   ├── semantic_ops_pg.c
│   │   │   ├── hypercube_ops_pg.c
│   │   │   ├── embedding_ops_pg.c
│   │   │   └── generative_pg.c
│   │   ├── ingest/             # Ingestion algorithms
│   │   │   ├── cpe.cpp         # Cascading Pair Encoding
│   │   │   ├── sequitur.cpp    # Grammar compression
│   │   │   └── pmi_contraction.cpp
│   │   └── db/                 # Database layer
│   └── tests/                  # C++ unit tests
├── sql/
│   ├── 001_schema.sql          # 3-table schema definition
│   ├── 002_core_functions.sql  # Core SQL functions
│   ├── 003_query_api.sql       # Query layer
│   ├── 004_generative_engine.sql
│   ├── 005_bigram_stats.sql    # PMI/bigram tables
│   ├── 006_qa_search.sql       # Q&A search
│   └── 007_model_registry.sql  # Model tracking
├── scripts/
│   ├── linux/                  # Linux shell scripts
│   └── windows/                # PowerShell scripts
├── tests/sql/                  # SQL test files
└── test-data/                  # Test fixtures
```

---

## Build Architecture

### Layer 1: hypercube_core (C++ Static Library)
Pure C++20 library with no PostgreSQL dependencies:
- Hilbert curve, coordinates, BLAKE3, SIMD operations
- Lanczos eigensolver, Laplacian projection

### Layer 2: C API Bridges (Shared Libraries)
Expose C++ functionality via `extern "C"`:
- `hypercube_c` - Core operations
- `embedding_c` - Embedding operations
- `generative_c` - Generative engine

### Layer 3: PostgreSQL Extensions (Pure C)
Link to C bridges, include PostgreSQL headers:
- `hypercube` - Core extension
- `semantic_ops` - Semantic queries
- `hypercube_ops` - Batch operations
- `embedding_ops` - SIMD embeddings
- `generative` - Generative walks

### Layer 4: CLI Tools
Executables requiring libpq:
- `seed_atoms_parallel` - Parallel Unicode seeding
- `ingest_safetensor_4d` - Model ingestion with Laplacian projection
- `ingest` - Universal content ingester
- `model_discovery` - HuggingFace model scanner

---

## Key Algorithms

### Hopf Fibration (Coordinate Mapping)
Maps Unicode codepoints to S³ hypersphere surface via Hopf fibration:
- Category → spherical angle θ
- Subcategory → spherical angle φ
- Codepoint → fiber angle ψ
- All coordinates deterministic and reproducible

### Laplacian Eigenmaps
Structure-preserving dimensionality reduction:
1. Build k-NN graph from high-D embeddings
2. Construct graph Laplacian: L = D - W
3. Solve generalized eigenvalue problem: L·v = λ·D·v
4. Take bottom 4 non-trivial eigenvectors as 4D coordinates

### Gram-Schmidt Orthonormalization
Ensures orthogonal 4D axes after Laplacian projection:
- Iteratively orthogonalize columns
- Normalize to unit vectors
- Preserves relative distances

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Atom lookup by codepoint | O(1) | Indexed |
| Composition lookup by hash | O(1) | Primary key |
| Spatial KNN query | O(log n) | GIST index |
| Hilbert range query | O(log n) | B-tree index |
| Text reconstruction | O(k) | k = leaf count |
| Laplacian projection | O(n²) | n = vocab size, done once per model |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v6 | 2026-01-05 | 3-table schema, Laplacian projection, major cleanup |
| v5 | 2025-01-16 | Binary PMI Merkle DAG |
| v4 | 2025-01-10 | Unified atom table |
| v3 | 2025-01-05 | Relation-edge separation |
| v2 | 2024-12-20 | Initial PostGIS integration |
| v1 | 2024-12-01 | Proof of concept |
