# Spatial Index Architecture: How Hartonomous Achieves O(log N)

## The Core Insight

**4D coordinates are PRIMARY. Hilbert index is DERIVED.**

```
Atom/Composition
    ↓
4D Laplacian Coordinates (x, y, z, w)
    ↓ [compute]
Hilbert Curve Index (128-bit: hi + lo)
    ↓ [index]
B-tree on (hilbert_hi, hilbert_lo) ← Range queries O(log N)
R-tree on GEOMETRY(POINTZM) ← Spatial queries O(log N)
```

## Why This Works

### Hilbert Curve: Space-Filling for Locality
The Hilbert curve maps 4D space to 1D while preserving spatial locality:

```
Points close in 4D → Similar Hilbert indices
Similar Hilbert indices → Efficient B-tree range query
```

**Example**:
```
Point A: (0.1, 0.1, 0.1, 0.1) → Hilbert: 0x0000...0123
Point B: (0.11, 0.12, 0.09, 0.11) → Hilbert: 0x0000...0156
Point C: (0.9, 0.9, 0.9, 0.9) → Hilbert: 0xFFFF...FE89

Query: Points within 0.05 of A
→ B-tree range: [0x0000...0000, 0x0000...0200]
→ Returns A, B (not C) in O(log N) time
```

### R-tree: Direct Spatial Queries
PostGIS GIST index builds an R-tree on 4D geometry:

```
Root: Bounding box of all points
├─ Node 1: Bounding box [0.0-0.5] in all dimensions
│  ├─ Leaf 1: Points (0.1, 0.1, 0.1, 0.1), ...
│  └─ Leaf 2: Points (0.2, 0.2, 0.2, 0.2), ...
└─ Node 2: Bounding box [0.5-1.0] in all dimensions
   ├─ Leaf 3: Points (0.6, 0.6, 0.6, 0.6), ...
   └─ Leaf 4: Points (0.9, 0.9, 0.9, 0.9), ...

Query: ST_DWithin(centroid, target, 0.05)
→ Traverse: Root → Node 1 → Leaf 1
→ Prunes 75% of data at each level
→ O(log N) traversal
```

## Actual Schema (from full_schema.sql)

```sql
-- ATOM TABLE: Base Unicode codepoints
CREATE TABLE atom (
    id              BYTEA PRIMARY KEY,           -- BLAKE3(codepoint)
    codepoint       INTEGER NOT NULL UNIQUE,
    geom            GEOMETRY(POINTZM, 0),        -- 4D coordinates (PRIMARY)
    hilbert_lo      NUMERIC(20,0),               -- Hilbert index low 64 bits (DERIVED)
    hilbert_hi      NUMERIC(20,0),               -- Hilbert index high 64 bits (DERIVED)
    ...
);

-- B-tree on Hilbert for range queries
CREATE INDEX idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);

-- R-tree (GIST) on 4D geometry for spatial queries
CREATE INDEX idx_atom_geom ON atom USING GIST(geom);

-- COMPOSITION TABLE: Aggregated atoms with centroids
CREATE TABLE composition (
    id              BYTEA PRIMARY KEY,
    centroid        GEOMETRY(POINTZM, 0),        -- 4D centroid (PRIMARY)
    hilbert_lo      NUMERIC(20,0),               -- Derived from centroid
    hilbert_hi      NUMERIC(20,0),               -- Derived from centroid
    ...
);

-- B-tree on Hilbert for composition range queries
CREATE INDEX idx_comp_hilbert ON composition(hilbert_hi, hilbert_lo);

-- R-tree (GIST) on 4D centroids for spatial queries
CREATE INDEX idx_comp_centroid ON composition USING GIST(centroid);
```

## Query Patterns

### 1. Nearest Neighbors (R-tree)
```sql
-- Find 10 nearest compositions to a query point
SELECT
    id,
    ST_Distance(centroid, ST_MakePoint($1, $2, $3, $4)) AS distance
FROM composition
ORDER BY centroid <-> ST_MakePoint($1, $2, $3, $4)  -- KNN operator uses R-tree!
LIMIT 10;

-- Complexity: O(log N) R-tree traversal
```

### 2. Range Query (B-tree on Hilbert)
```sql
-- Find compositions in a Hilbert range (locality-preserving)
SELECT id, centroid
FROM composition
WHERE (hilbert_hi, hilbert_lo) BETWEEN ($1, $2) AND ($3, $4)
ORDER BY hilbert_hi, hilbert_lo
LIMIT 100;

-- Complexity: O(log N) B-tree seek + O(K) scan where K = results
```

### 3. Spatial Intersection (R-tree)
```sql
-- Find compositions within a 4D bounding box
SELECT id, centroid
FROM composition
WHERE ST_3DIntersects(
    centroid,
    ST_MakeBox3D(
        ST_MakePoint($1, $2, $3),
        ST_MakePoint($4, $5, $6)
    )
);

-- Complexity: O(log N) R-tree traversal
```

### 4. Relation Lookup (B-tree on indexed columns)
```sql
-- Find high-quality relations from a source
SELECT target_id, weight
FROM relation
WHERE source_id = $1
  AND weight >= $2
ORDER BY weight DESC
LIMIT 20;

-- Uses: idx_relation_source (B-tree on source_id)
-- Complexity: O(log N) + O(K) where K = matching relations
```

### 5. Multi-hop Graph Traversal
```sql
WITH RECURSIVE paths AS (
    -- Seed: start from query composition
    SELECT source_id, target_id, weight, 1 as hops
    FROM relation
    WHERE source_id = $1 AND weight >= $2

    UNION

    -- Expand: follow edges (uses B-tree repeatedly)
    SELECT r.source_id, r.target_id, r.weight, p.hops + 1
    FROM relation r
    JOIN paths p ON r.source_id = p.target_id
    WHERE p.hops < $3 AND r.weight >= $2
)
SELECT DISTINCT target_id, MAX(weight) as best_weight
FROM paths
GROUP BY target_id
ORDER BY best_weight DESC;

-- Complexity: O(K × hops × log N) where K = avg relations per node
```

## Performance Analysis

### Seeding 1.1M Atoms (from logs/06_setup-db-log.txt)

```
[4/5] Parallel COPY to atom table (8 connections)...
      Parallel COPY in 2532 ms

[5/5] Building indexes...
      Index build in 5276 ms
      - B-tree on hilbert_hi, hilbert_lo
      - R-tree (GIST) on geom

Total atoms: 1,114,112
Total time: 11.384 s
Rate: 97,866 atoms/sec
```

**Index build**: 5.3 seconds for 1.1M 4D points = 210,000 points/sec

### Query Performance Estimates

**Nearest neighbor** (R-tree):
- Tree height for 1M points: log₄(1,000,000) ≈ 10 levels
- Distance calculations: ~40 (4 per level × 10 levels)
- **Latency**: 1-10ms depending on SSD speed

**Range query** (B-tree on Hilbert):
- Tree height for 1M points: log₁₀₀(1,000,000) = 2 levels
- Hilbert range mapping: ~100 candidates
- **Latency**: 1-5ms

**Relation lookup** (B-tree on source_id):
- Hash-based B-tree: O(1) bucket + O(log K) within bucket
- **Latency**: <1ms for cached, 1-5ms for disk

**Multi-hop traversal** (K=100, hops=3):
- Lookups: 100 × 3 = 300 relation queries
- Each query: O(log N) ≈ 20 comparisons
- Total: 6,000 comparisons
- **Latency**: 10-50ms depending on cache hit rate

## Scaling to Billions

### 1 Billion Atoms

**Index sizes**:
```
B-tree on Hilbert (16 bytes per entry):
    1B × 16 = 16 GB

R-tree on 4D geom (64 bytes per entry + overhead):
    1B × 100 = 100 GB

Total indexes: ~120 GB (fits on single SSD)
```

**Query performance**:
```
R-tree height: log₄(1,000,000,000) ≈ 15 levels
B-tree height: log₁₀₀(1,000,000,000) = 3 levels

Nearest neighbor: 15 × 4 = 60 distance calculations
Range query: 3 levels + K results

Still O(log N)! Only 50% slower than 1M atoms.
```

### Horizontal Sharding

```
Shard 1: Atoms 0x00-0x3F (Hilbert range)
Shard 2: Atoms 0x40-0x7F
Shard 3: Atoms 0x80-0xBF
Shard 4: Atoms 0xC0-0xFF

Query distribution:
- Point query: 1 shard (Hilbert hash)
- Range query: 1-2 shards (Hilbert range)
- Global query: All shards (parallel)
```

**Benefits**:
- Each shard: 250M atoms, fits in memory
- Parallel queries across shards
- Standard PostgreSQL replication

## Why This Beats Traditional AI

### Traditional AI: Flat Vector Search

```python
# Brute force: O(N × D)
for embedding in all_embeddings:  # 1M vectors
    sim = cosine(query, embedding)  # 2048 dimensions
    # 1M × 2048 = 2B operations

# With HNSW: O(log N × D)
# Still needs vector quantization, pruning heuristics
```

**Problem**: Scales linearly with dimension (D) and number of vectors (N).

### Hartonomous: Spatial Index Search

```sql
-- O(log N) with R-tree or B-tree
SELECT * FROM composition
WHERE centroid <-> ST_MakePoint(...) < 0.5
LIMIT 10;

-- 4D space: fixed dimensionality
-- Index height: log(N) regardless of dimension
```

**Advantage**: Dimension is fixed (4D), scaling is purely O(log N).

## Implementation in C++

### Computing Hilbert Index (from coordinates.cpp)

```cpp
// Map 4D coordinates to Hilbert index
HilbertIndex coords_to_index(const Point4D& point) {
    // Quantize 4D doubles to 32-bit integers per dimension
    uint32_t x = static_cast<uint32_t>((point.x + 1.0) / 2.0 * UINT32_MAX);
    uint32_t y = static_cast<uint32_t>((point.y + 1.0) / 2.0 * UINT32_MAX);
    uint32_t z = static_cast<uint32_t>((point.z + 1.0) / 2.0 * UINT32_MAX);
    uint32_t m = static_cast<uint32_t>((point.m + 1.0) / 2.0 * UINT32_MAX);

    // Encode via Hilbert curve (32 bits per dimension = 128-bit index)
    return encode_hilbert_4d(x, y, z, m);
}
```

**Locality preservation**:
- Points close in 4D → Similar Hilbert indices
- Hilbert range query → Spatial neighborhood
- No hash collisions (bijective mapping)

### Using in Database Queries

```cpp
// C++ API for Hartonomous
hc_hilbert_t query_hilbert = hc_coords_to_hilbert(query_coords);

// Compute range for spatial search (± epsilon in Hilbert space)
hc_hilbert_t range_min = subtract(query_hilbert, epsilon);
hc_hilbert_t range_max = add(query_hilbert, epsilon);

// SQL query (B-tree range scan)
cur.execute("""
    SELECT id, centroid
    FROM composition
    WHERE (hilbert_hi, hilbert_lo) BETWEEN (%s, %s) AND (%s, %s)
    LIMIT 100
""", (range_min.hi, range_min.lo, range_max.hi, range_max.lo));
```

## Conclusion

**The Hartonomous spatial indexing strategy**:

1. **4D coordinates are primary** (x, y, z, w from Laplacian projection)
2. **Hilbert index is derived** (for B-tree locality-preserving range queries)
3. **R-tree on coordinates** (for direct spatial queries)
4. **B-tree on relations** (for graph traversal)

**Result**:
- O(log N) nearest neighbor
- O(log N) range query
- O(K × hops × log N) multi-hop graph traversal

**No brute force. No O(N²). No GPUs needed.**

This is why Hartonomous can scale to billions of compositions while remaining CPU-bound and database-native.
