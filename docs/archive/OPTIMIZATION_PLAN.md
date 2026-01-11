# Hypercube Optimization Plan

## Current Issues

### 1. SQL Functions with RBAR (Row-By-Agonizing-Row) Patterns

| Function | File | Problem | Priority |
|----------|------|---------|----------|
| `atom_reconstruct` | 01_unified_atom.sql | WITH RECURSIVE CTE, 1 query per node | HIGH |
| `atom_find_exact` | 01_unified_atom.sql | Full table scan with text comparison | HIGH |
| `atom_search_text` | 02_semantic_udf.sql | LIKE queries, returns text per row | MEDIUM |
| `semantic_walk` | 03_ai_operations.sql | WHILE loop, 1 query per step | HIGH |
| `semantic_path` | 03_ai_operations.sql | BFS with nested loops | HIGH |
| `generate_random_walk` | 03_ai_operations.sql | WHILE loop, RANDOM(), 1 query per step | HIGH |
| `generate_directed` | 03_ai_operations.sql | WHILE loop with ORDER BY per step | HIGH |
| `infer_path` | 03_ai_operations.sql | WHILE loop with spatial queries | MEDIUM |
| `maintenance_compute_centroids` | 04_centroid_optimization.sql | Row-by-row centroid update | MEDIUM |

### 2. Existing C++ Functions (in semantic_ops.cpp)

These already exist but have sub-optimal patterns:

| Function | Issue |
|----------|-------|
| `semantic_traverse` | SPI_execute in loop (1 query per node) |
| `semantic_reconstruct` | SPI_execute in loop (1 query per node) |

## Optimization Strategy

### Phase 1: Batch Loading (Eliminate N+1 Queries)

Instead of querying one node at a time, pre-load related data:

```cpp
// BEFORE: N queries
for (each node in traversal) {
    SPI_execute("SELECT children FROM atom WHERE id = ?", node.id);
}

// AFTER: 1 query + local lookup
SPI_execute("SELECT id, children FROM atom WHERE depth <= ? ORDER BY depth", max_depth);
// Build hash map in memory, then traverse locally
```

### Phase 2: Prepared Statements with SPI_prepare

```cpp
// BEFORE: Parse query each time
SPI_execute("SELECT children FROM atom WHERE id = $1", ...);

// AFTER: Parse once, execute many
SPIPlanPtr plan = SPI_prepare("SELECT children FROM atom WHERE id = $1", 1, argtypes);
SPI_execute_plan(plan, values, nulls, true, 0);
```

### Phase 3: Bulk Operations with COPY Protocol

For ingestion:
```cpp
// Use COPY FROM for bulk insert
PQputCopyData(conn, binary_data, len);
```

### Phase 4: Parallel Processing

Mark functions as `PARALLEL SAFE` and use:
- PostgreSQL parallel workers
- Internal thread pools for compute-heavy operations
- Partitioned queries that can run concurrently

### Phase 5: Index-Aware Algorithms

- Use Hilbert index for range scans
- Use GIN index on children[] for containment queries  
- Use BRIN index for depth-ordered access

## New C++ Functions to Implement

### Batch Lookups

```sql
-- Lookup multiple atoms in one call
CREATE FUNCTION atom_batch_lookup(ids BYTEA[])
RETURNS TABLE(id BYTEA, value BYTEA, children BYTEA[], depth INT, centroid GEOMETRY)
AS 'hypercube_ops' LANGUAGE C PARALLEL SAFE;

-- Reconstruct multiple texts in one call
CREATE FUNCTION atom_batch_reconstruct(ids BYTEA[])
RETURNS TABLE(id BYTEA, content TEXT)
AS 'hypercube_ops' LANGUAGE C PARALLEL SAFE;
```

### Graph Algorithms (In-Memory)

```sql
-- Load subgraph into memory, then walk
CREATE FUNCTION semantic_walk_batch(
    seed_ids BYTEA[],
    max_steps INT,
    max_nodes INT DEFAULT 100000
) RETURNS TABLE(seed_id BYTEA, step INT, node_id BYTEA, edge_weight FLOAT8)
AS 'hypercube_ops' LANGUAGE C PARALLEL SAFE;

-- In-memory BFS for shortest path
CREATE FUNCTION semantic_path_fast(
    from_id BYTEA,
    to_id BYTEA,
    max_depth INT DEFAULT 6
) RETURNS TABLE(step INT, node_id BYTEA)
AS 'hypercube_ops' LANGUAGE C PARALLEL SAFE;
```

### Bulk Ingestion

```sql
-- Bulk insert with pre-computed hashes
CREATE FUNCTION atom_bulk_insert(
    ids BYTEA[],
    values BYTEA[],
    children_arrays BYTEA[][],
    geometries GEOMETRY[],
    hilbert_lo BIGINT[],
    hilbert_hi BIGINT[]
) RETURNS INT  -- Number inserted
AS 'hypercube_ops' LANGUAGE C;
```

## Memory Model

### Per-Query Memory Context

```cpp
// Use PostgreSQL memory contexts properly
MemoryContext batch_ctx = AllocSetContextCreate(
    CurrentMemoryContext,
    "HypercubeBatch",
    ALLOCSET_DEFAULT_SIZES
);

MemoryContext old = MemoryContextSwitchTo(batch_ctx);
// ... batch operations ...
MemoryContextSwitchTo(old);
MemoryContextDelete(batch_ctx);  // Free all at once
```

### Shared Memory for Large Graphs

For very large graph operations, consider:
- DSM (Dynamic Shared Memory) segments
- Hash tables in shared memory
- Background workers for async processing

## Build Configuration

### CMakeLists.txt Updates

```cmake
# Enable parallel compilation
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

# Link-time optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# New optimized extension
add_library(hypercube_ops SHARED
    src/hypercube_ops.cpp
    src/batch_ops.cpp
    src/graph_ops.cpp
    src/hilbert.cpp
    src/blake3_impl.cpp
)
```

## Implementation Order

1. **Week 1**: Batch lookup functions
   - `atom_batch_lookup`
   - `atom_batch_reconstruct`
   - Update existing SPI loops to use batch loads

2. **Week 2**: Graph algorithms
   - `semantic_walk_batch` with in-memory graph
   - `semantic_path_fast` with BFS
   - `semantic_neighbors_batch`

3. **Week 3**: Bulk ingestion
   - `atom_bulk_insert` with COPY protocol
   - Update C++ ingest tools to use bulk functions
   - Parallel ingestion with thread pool

4. **Week 4**: Advanced optimizations
   - Prepared statement caching
   - Shared memory for hot data
   - Background workers for async operations

## Metrics to Track

- Queries per operation (target: 1-3 vs current N)
- Time per 10K node traversal
- Memory usage during batch operations
- CPU utilization (should see multi-core usage)
