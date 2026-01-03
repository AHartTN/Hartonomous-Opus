# Hartonomous Hypercube - Architecture Document

**Last Updated**: 2026-01-02
**Status**: Canonical - Unified Schema (v3)

---

## Executive Summary

The Hypercube is a **deterministic, lossless, content-addressable geometric semantic substrate**. All digital content is decomposed into a Merkle DAG where:

1. **Atoms** (Unicode codepoints) form the perimeter landmarks at fixed 4D Hilbert coordinates
2. **Compositions** are binary pairs merged upward via Cascading Pair Encoding (CPE)
3. **Children** are stored as BYTEA arrays, encoding sequence order
4. **Deduplication** is global - identical patterns share the same hash regardless of source
5. **Reconstruction** is bit-perfect via DFS traversal of children arrays

### The Single Table Model (IMPLEMENTED)

The system uses a **single `atom` table** where:
- **POINTZM** = Unicode codepoints (the seeded perimeter - depth 0)
- **LINESTRINGZM** = Compositions (trajectory through child centroids - depth > 0)

The `geom GEOMETRY(GEOMETRYZM, 0)` column stores both geometry types.
The `children BYTEA[]` column stores child hash references for compositions.

---

## Core Principles

### 1. Determinism
- Same bytes → same composition ID, always
- No randomness, no floating-point conversion, no approximations
- Hash = BLAKE3 of canonical child sequence

### 2. Losslessness
- Bit-perfect reconstruction from root composition
- DFS traversal of edges → original byte sequence
- All coordinates stored as 32-bit signed integers (bit pattern same as uint32)

### 3. Global Deduplication
- "the" from Moby Dick = "the" from a children's book = same composition ID
- First ingest creates the pattern; subsequent ingests only add edge references
- The more you ingest, the more deduplication occurs

### 4. Cascading Pair Encoding (CPE)

**Level 0 → Level 1: Overlapping Sliding Window**
- Every adjacent atom pair becomes a composition
- "Captain" → Ca, ap, pt, ta, ai, in (6 bigrams from 7 atoms)
- Captures **complete sequential structure** at atomic level

**Level 1+: Binary Cascade**  
- Non-overlapping pairs for efficiency
- (Ca,ap), (pt,ta), (ai,in) → higher compositions
- O(log N) levels after the first

**Total compositions ≈ 2N** (N-1 from overlapping + ~N from cascade)

### 5. Emergent Topology as Semantics

**The structure IS the meaning.** This is fundamentally different from:
- Vector embeddings (opaque dimensions)
- Probability distributions (training artifacts)
- Learned projections (black box)

Semantic signal emerges from:
- **Connectivity**: How many edges does this node have?
- **Trajectory shape**: What path through 4D space? (Fréchet distance)
- **Neighborhood density**: How clustered are connections?
- **Path multiplicity**: How many ways to reach X from Y?

Edge weight is just a hint ("how often"). The graph topology itself speaks:
- "ap" connects Captain, happy, trap, apple,eraptor...
- The intersection of neighborhoods = semantic relationship
- No training, no loss function, no gradients

---

## Data Model

### Unified Atom Table
```sql
atom (
    -- Content-addressed identifier
    id              BYTEA PRIMARY KEY,      -- BLAKE3 hash (32 bytes)

    -- Geometry: POINTZM for leaves, LINESTRINGZM for compositions
    geom            GEOMETRY(GEOMETRYZM, 0) NOT NULL,

    -- Child references for compositions (NULL for leaves)
    children        BYTEA[],                -- Array of child hashes in order

    -- Canonical value for leaves only (UTF-8 bytes)
    value           BYTEA,                  -- NULL for compositions

    -- Unicode codepoint for leaf atoms (NULL for compositions)
    codepoint       INTEGER UNIQUE,

    -- 128-bit Hilbert index (from ST_Centroid -> 4D coords)
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,

    -- Depth in DAG (0 = leaf, 1 = pair of atoms, etc.)
    depth           INTEGER NOT NULL DEFAULT 0,

    -- Total leaf atoms in subtree (1 for leaves)
    atom_count      BIGINT NOT NULL DEFAULT 1
)
```

**Key Points**:
- Single table for ALL content - atoms AND compositions
- `depth = 0` means leaf (POINTZM geometry, has value and codepoint)
- `depth > 0` means composition (LINESTRINGZM geometry, has children array)
- LINESTRINGZM vertices are child centroids in sequence order
- ST_Centroid(geom) gives the 4D centroid for Hilbert encoding
- DFS traversal of children arrays = original byte sequence

---

## Ingestion Pipeline

### Current Implementation (C++ CPE Ingester)

```
File → UTF-8 decode → Codepoints → Atom lookup → CPE cascade → Compositions
```

**CPE Cascade Algorithm**:
```
1. Convert codepoints to atom IDs (from cache)
2. While len(nodes) > 1:
   a. Pair adjacent nodes: [a,b,c,d,e] → [(a,b), (c,d), e]
   b. Each pair becomes a composition with:
      - hash = BLAKE3(0||left_hash || 1||right_hash)
      - centroid = average of child coords (integer arithmetic)
   c. nodes = merged results
3. Return single root
```

**Performance**:
- N characters → ~2N compositions (binary tree)
- O(N) time, O(N) space
- Batch insert via PostgreSQL COPY

### What C++ Does (Heavy Lifting)
1. Load atom cache once (codepoint → hash, coords)
2. UTF-8 decode
3. CPE cascade (hash computation, centroid calculation, Hilbert indexing)
4. Batch COPY to PostgreSQL

### What SQL Does (Orchestration)
1. Store compositions and edges (INSERT...ON CONFLICT DO NOTHING)
2. Spatial queries (PostGIS for atom table only)
3. Hilbert range queries for relations
4. DFS traversal for reconstruction

---

## Type System - CRITICAL

### Coordinate Storage

| Field | PostgreSQL Type | Interpretation | Range |
|-------|----------------|----------------|-------|
| coord_x/y/z/m | INTEGER | Signed 32-bit | -2,147,483,648 to 2,147,483,647 |
| (As uint32) | N/A | Unsigned 32-bit | 0 to 4,294,967,295 |

**Conversion** (lossless bit reinterpretation):
```c++
// C++
uint32_t as_unsigned = static_cast<uint32_t>(signed_val);
int32_t as_signed = static_cast<int32_t>(unsigned_val);

// SQL
int32_to_uint32(INTEGER) → BIGINT  -- Add 2^32 if negative
uint32_to_int32(BIGINT) → INTEGER  -- Subtract 2^32 if >= 2^31
```

### Hash Storage
- `BYTEA(32)` - 256-bit BLAKE3 hash
- Domain: `blake3_hash` with length check

### Hilbert Index
- 128-bit split into two `BIGINT` columns
- `hilbert_hi` (upper 64 bits), `hilbert_lo` (lower 64 bits)
- Stored as signed but interpreted as unsigned for ordering

---

## FORBIDDEN Patterns

### ❌ Never Do This:

1. **Lossy double conversion**:
   ```sql
   -- BAD: Loses precision!
   ST_X(coords) * 4294967295  -- Double precision loses bits
   ```

2. **PostGIS for relation centroids**:
   ```sql
   -- BAD: PostGIS normalizes to [-1,1] with float64
   ST_SetSRID(ST_MakePoint(...), 0)
   ```

3. **Sliding n-grams**:
   ```sql
   -- BAD: O(n²) explosion
   FOR i IN 1..(len - ngram_size + 1)
       v_ngram_ids := array_append(v_ngram_ids, substring(...))
   ```

4. **Line-by-line ingestion**:
   ```bash
   # BAD: Creates disconnected compositions per line
   while read line; do psql -c "SELECT ingest('$line')"; done
   ```

5. **Temp tables for computation**:
   ```sql
   -- BAD: C++ should compute, SQL should only store
   CREATE TEMP TABLE tmp_compute...
   ```

### ✅ Always Do This:

1. **Integer arithmetic for centroids**:
   ```c++
   uint64_t sum = uint32_a + uint32_b;
   uint32_t avg = sum / 2;  // Lossless
   ```

2. **CPE binary cascade**:
   ```c++
   while (nodes.size() > 1) {
       merge_pairs(nodes);  // Halves count each pass
   }
   ```

3. **Whole-file ingestion**:
   ```c++
   std::string content = read_file(path);
   auto codepoints = decode_utf8(content);
   auto root = cpe_cascade(codepoints);
   ```

4. **COPY for batch insert**:
   ```c++
   PQexec(conn, "COPY relation FROM STDIN...");
   // Send all compositions
   PQexec(conn, "INSERT...ON CONFLICT DO NOTHING");
   ```

---

## Reconstruction

To reconstruct original content from a composition ID:

```sql
-- Using the built-in function
SELECT atom_reconstruct_text('\x...'::BYTEA);

-- Or manually with recursive CTE
WITH RECURSIVE tree AS (
    SELECT id, children, value, ARRAY[]::INTEGER[] as path
    FROM atom WHERE id = $root_id

    UNION ALL

    SELECT a.id, a.children, a.value, t.path || c.ordinal
    FROM tree t
    CROSS JOIN LATERAL unnest(t.children) WITH ORDINALITY AS c(child_id, ordinal)
    JOIN atom a ON a.id = c.child_id
    WHERE t.children IS NOT NULL
)
SELECT string_agg(value, ''::BYTEA ORDER BY path)
FROM tree WHERE value IS NOT NULL;
```

Key insight: The `children` array stores child hashes in sequence order.
DFS traversal concatenates leaf `value` fields = original bytes.

---

## Similarity/Querying

### Centroid Distance (4D Euclidean)
```sql
sqrt(
    (r.coord_x - q.coord_x)^2 +
    (r.coord_y - q.coord_y)^2 +
    (r.coord_z - q.coord_z)^2 +
    (r.coord_m - q.coord_m)^2
)
```

### Hilbert Range Query (for relations)
```sql
WHERE hilbert_hi = target_hi
  AND abs(hilbert_lo - target_lo) < range
```

### PostGIS Spatial (for atoms only)
```sql
WHERE ST_DWithin(coords, target_geom, radius)
```

---

## File Layout

```
Hartonomous-Opus/
├── cpp/
│   ├── src/
│   │   ├── cpe_ingest.cpp           # Main CPE ingester (C++)
│   │   ├── seed_atoms_parallel.cpp  # Unicode seeder (unified schema)
│   │   ├── hypercube.cpp            # PostgreSQL extension
│   │   ├── blake3_pg.cpp            # BLAKE3 for PostgreSQL
│   │   ├── hilbert.cpp              # Hilbert curve implementation
│   │   └── coordinates.cpp          # Coordinate utilities
│   ├── include/hypercube/
│   │   ├── types.hpp                # Core types (Blake3Hash, Point4D, etc.)
│   │   ├── hilbert.hpp              # Hilbert curve interface
│   │   ├── coordinates.hpp          # Coordinate utilities
│   │   └── blake3.hpp               # BLAKE3 wrapper
│   └── sql/
│       └── hypercube--1.0.sql       # Extension SQL
├── sql/
│   ├── 001_schema.sql               # (DEPRECATED - use 011)
│   ├── 011_unified_atom.sql         # Unified atom table schema
│   └── ...
├── setup.sh                         # Single entry point
└── ARCHITECTURE.md                  # This document
```

---

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| Init (seed atoms) | ~30s | 1.1M Unicode codepoints |
| Ingest text | ~1 MB/s | CPE in C++ |
| Ingest large file (100MB) | ~2 min | Chunked processing |
| Query similar | <100ms | Hilbert index |
| Reconstruct | O(n) | DFS traversal |

---

## Scaling Considerations

1. **Deduplication improves with scale** - More content = more shared substructure
2. **Hilbert index enables range sharding** - Split on hilbert_hi
3. **CPE is embarrassingly parallel** - Chunk files, cascade results
4. **Edge storage dominates** - ~2 edges per composition

---

## Change Log

### 2026-01-02 - Unified Schema (v3)
- **CONSOLIDATED**: Single `atom` table replaces atom + relation + relation_edge
- **ADDED**: `geom GEOMETRY(GEOMETRYZM, 0)` stores both POINTZM and LINESTRINGZM
- **ADDED**: `children BYTEA[]` stores child hash references (replaces relation_edge)
- **ADDED**: `value BYTEA` stores canonical bytes for leaves
- **ADDED**: `codepoint INTEGER` for fast leaf atom lookup
- **UPDATED**: C++ ingester outputs LINESTRINGZM for compositions
- **UPDATED**: Reconstruction uses children array traversal

### 2026-01-02 - Course Correction (v2)
- **REMOVED**: PostGIS coords column from relation table
- **ADDED**: Lossless integer coordinates as source of truth
- **FIXED**: CPE cascade (was O(n²) sliding windows, now O(n) binary merge)
- **FIXED**: Whole-file ingestion (was per-line)
- **DOCUMENTED**: Type system and forbidden patterns
