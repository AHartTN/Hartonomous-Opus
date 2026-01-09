# Architectural Changes Plan

## Overview

## Architectural Vision: Towards Production-Ready Semantic Infrastructure

### Core Architectural Principles

**1. Algorithmic Completeness**
- All semantic operations must be computationally tractable
- Mathematical foundations must be sound and well-implemented
- Performance must scale sub-linearly or linearly with data size

**2. Data Integrity Guarantees**
- Atomic operations ensure consistency
- Referential integrity maintained across all relationships
- Automatic maintenance of derived data (centroids, indices)

**3. Computational Efficiency**
- SIMD utilization for mathematical operations
- Parallel processing for independent computations
- Memory-efficient algorithms avoiding quadratic complexity

**4. Semantic Fidelity**
- All operations preserve semantic relationships
- Coordinate transformations maintain relative distances
- Query results reflect true semantic similarity

### Long-Term Architectural Goals

**Phase 1 (Current): Algorithmic Foundation**
- Complete core semantic algorithms (Laplacian projection, graph operations)
- Establish mathematical correctness and performance baselines
- Implement missing functionality gaps

**Phase 2 (Next): Scalability Infrastructure**
- Distributed processing capabilities
- Advanced indexing strategies (HNSW, IVF, etc.)
- Memory-efficient data structures

**Phase 3 (Future): Intelligence Layer**
- Machine learning optimization of semantic operations
- Adaptive algorithm selection based on data characteristics
- Self-tuning performance parameters

## Priority: HIGH - Critical for Production Scalability

## 1. Batch Graph Operations Architecture

### Problem Analysis: RBAR Anti-Pattern Crisis

**Current State**: Row-By-Agony-Row (RBAR) queries dominate graph operations, creating O(N) database round-trips for what should be O(1) in-memory operations.

**Performance Impact**:
- Semantic queries taking seconds instead of milliseconds
- Database connection pool exhaustion
- Memory pressure from result set accumulation

**Root Cause**: SQL recursive CTEs and loops used for graph traversal instead of in-memory algorithms.

### Comprehensive Graph Processing Architecture

**Phase 1: Graph Data Structure Design**
Implement efficient in-memory graph representations:

```cpp
// cpp/include/hypercube/graph/batch_graph.hpp
class BatchGraphProcessor {
public:
    // Compressed Sparse Row (CSR) representation for efficiency
    struct CSRGraph {
        std::vector<size_t> row_ptr;     // Row start indices
        std::vector<size_t> col_idx;     // Column indices (neighbors)
        std::vector<float> weights;      // Edge weights
        std::vector<uint32_t> node_ids;  // Map to database IDs

        // Efficient neighbor access
        std::span<const size_t> get_neighbors(size_t node_idx) const {
            size_t start = row_ptr[node_idx];
            size_t end = row_ptr[node_idx + 1];
            return std::span(col_idx.data() + start, end - start);
        }
    };

private:
    // Multi-resolution graph storage
    CSRGraph semantic_graph_;      // Full semantic relationships
    CSRGraph similarity_graph_;    // k-NN similarity structure
    std::unordered_map<uint32_t, size_t> id_to_index_;  // Database ID → graph index
};
```

**Phase 2: Batch Algorithm Implementations**
Implement vectorized graph algorithms:

```cpp
// cpp/src/graph/batch_algorithms.cpp
class BatchGraphAlgorithms {
public:
    // Vectorized BFS for multiple sources
    std::vector<std::vector<size_t>> multi_source_bfs(
        const CSRGraph& graph,
        const std::vector<size_t>& sources,
        size_t max_depth
    ) {
        const size_t num_sources = sources.size();
        std::vector<std::vector<size_t>> results(num_sources);

        // Parallel BFS for each source
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < num_sources; ++i) {
            results[i] = single_source_bfs(graph, sources[i], max_depth);
        }

        return results;
    }

    // SIMD-accelerated similarity computation for k-NN
    std::vector<std::pair<size_t, float>> batch_similarity_search(
        const CSRGraph& graph,
        const std::vector<float>& query_vector,
        size_t k,
        const std::vector<size_t>& candidates
    ) {
        const size_t num_candidates = candidates.size();
        std::vector<std::pair<size_t, float>> results;

        // Batch cosine similarity computation
        #pragma omp parallel
        {
            std::vector<std::pair<size_t, float>> local_results;

            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_candidates; ++i) {
                size_t node_idx = candidates[i];
                // SIMD cosine similarity computation
                float similarity = compute_similarity_simd(query_vector, get_node_vector(node_idx));
                local_results.emplace_back(node_idx, similarity);
            }

            // Merge results with reduction
            #pragma omp critical
            {
                results.insert(results.end(), local_results.begin(), local_results.end());
            }
        }

        // Sort and truncate to top-k
        std::partial_sort(results.begin(), results.begin() + k, results.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        results.resize(k);

        return results;
    }

private:
    // SIMD cosine similarity using AVX2/AVX512
    float compute_similarity_simd(const std::vector<float>& a, const std::vector<float>& b) {
        // Implementation uses AVX2 for 8-float vectors
        // Fallback to scalar for unsupported platforms
    }
};
```

**Phase 3: Database Integration Layer**
Efficient bulk data transfer between database and graph structures:

```cpp
// cpp/src/graph/database_bridge.cpp
class GraphDatabaseBridge {
public:
    // Bulk load graph from database with zero-copy where possible
    CSRGraph load_semantic_graph(PGconn* conn, const std::string& model_filter = "") {
        CSRGraph graph;

        // Single query loads all relationships
        std::string query = R"(
            SELECT source_id, target_id, weight, relation_type
            FROM relation
            WHERE ($1 = '' OR source_model = $1)
            ORDER BY source_id, target_id
        )";

        PGresult* res = PQexecParams(conn, query.c_str(), 1, nullptr,
                                   (const char*[]){model_filter.c_str()},
                                   nullptr, nullptr, 0);

        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            throw std::runtime_error("Failed to load graph: " + std::string(PQerrorMessage(conn)));
        }

        // Build CSR structure in one pass
        build_csr_from_result(graph, res);
        PQclear(res);

        return graph;
    }

    // Batch update database with computed results
    void store_batch_results(PGconn* conn,
                           const std::vector<SemanticQueryResult>& results) {
        // Use PostgreSQL COPY protocol for efficient bulk insert
        std::string copy_query = "COPY temp_semantic_results (query_id, result_id, score) FROM STDIN";
        PGresult* res = PQexec(conn, copy_query.c_str());

        if (PQresultStatus(res) != PGRES_COPY_IN) {
            throw std::runtime_error("Failed to start COPY: " + std::string(PQerrorMessage(conn)));
        }

        // Stream results to database
        for (const auto& result : results) {
            std::string row = std::to_string(result.query_id) + "\t" +
                            std::to_string(result.result_id) + "\t" +
                            std::to_string(result.score) + "\n";
            PQputCopyData(conn, row.c_str(), row.length());
        }

        PQputCopyEnd(conn, nullptr);
        PQclear(res);
    }

private:
    void build_csr_from_result(CSRGraph& graph, PGresult* res) {
        const int num_rows = PQntuples(res);

        // Pre-allocate with estimated sizes
        graph.row_ptr.reserve(num_rows / 10);  // Estimate based on graph density
        graph.col_idx.reserve(num_rows);
        graph.weights.reserve(num_rows);

        uint32_t current_source = 0;
        size_t current_row_start = 0;

        for (int i = 0; i < num_rows; ++i) {
            uint32_t source_id = std::stoul(PQgetvalue(res, i, 0));
            uint32_t target_id = std::stoul(PQgetvalue(res, i, 1));
            float weight = std::stof(PQgetvalue(res, i, 2));

            // New source node - finalize previous row
            if (source_id != current_source && !graph.row_ptr.empty()) {
                graph.row_ptr.push_back(current_row_start);
                current_source = source_id;
                current_row_start = graph.col_idx.size();
            }

            graph.col_idx.push_back(target_id);
            graph.weights.push_back(weight);
        }

        // Finalize last row
        graph.row_ptr.push_back(graph.col_idx.size());
    }
};
```

**Phase 4: SQL Function Integration**
Replace RBAR SQL with calls to batch C++ functions:

```sql
-- sql/003_query_api.sql - Updated semantic_neighbors
CREATE OR REPLACE FUNCTION semantic_neighbors(query_text text, k integer DEFAULT 10)
RETURNS TABLE(content text, distance real, composition_id bytea)
LANGUAGE plpgsql
AS $
DECLARE
    query_id bytea;
    result_rows semantic_result[];
BEGIN
    -- Get query composition ID
    SELECT id INTO query_id FROM content_get(query_text) LIMIT 1;

    IF query_id IS NULL THEN
        RETURN;
    END IF;

    -- Call C++ batch processing function
    SELECT batch_semantic_search(query_id, k) INTO result_rows;

    -- Return formatted results
    RETURN QUERY
    SELECT r.content, r.distance, r.composition_id
    FROM unnest(result_rows) AS r;
END;
$;
```

**Performance Projections**:
- **Current**: O(N) database round-trips, ~5-10 seconds for complex queries
- **Target**: O(1) bulk operations, ~50-200ms for equivalent queries
- **Scalability**: Linear scaling with graph size vs quadratic degradation

### 2. Implement Batch Token Ingestion
**Problem**: Missing CPE batch ingestion for tokens not in embeddings
**Impact**: Incomplete semantic coverage during embedding extraction
**Files**: `cpp/src/tools/extract_embeddings.cpp` (line 244)
**Effort**: Medium (3-4 days) - implement batch CPE ingestion
**Validation**: All missing tokens ingested via CPE
**Dependencies**: None

**Tasks**:
- Locate token ingestion gap in embedding extraction
- Implement batch CPE ingestion for missing tokens
- Ensure proper semantic relationships
- Test complete token coverage in embeddings
- Validate semantic query improvements

### 3. Extract Router Weights Properly for MoE Models
**Problem**: Ignores router.weight tensors, tries cell-by-cell extraction
**Impact**: Missing sparse routing relationships, O(n²) performance disaster
**Files**: `cpp/src/tools/ingest_safetensor.cpp`
**Effort**: High (5-7 days) - implement sparse expert routing extraction
**Validation**: Router weights properly extracted for MoE models
**Dependencies**: After compile fixes (critical-fixes.md #3)

**Tasks**:
- Analyze MoE router weight tensor structure
- Implement sparse routing relationship extraction
- Avoid O(n²) cell-by-cell processing
- Store routing relationships in database
- Test MoE model ingestion performance

### 4. Add SIMD Hilbert Parallelization
**Problem**: Serial processing in coordinate transformations
**Impact**: Missed 4-8x speedup opportunity
**Files**: `cpp/src/core/hilbert.cpp`, coordinate transformation code
**Effort**: High (1 week) - vectorize Hilbert operations with AVX2/AVX512
**Validation**: Hilbert operations show 4-8x performance improvement
**Dependencies**: None

**Tasks**:
- Identify serial bottlenecks in Hilbert processing
- Implement SIMD vectorization for coordinate transforms
- Use AVX2/AVX512 intrinsics where available
- Add runtime CPU feature detection
- Benchmark and validate performance gains

### 5. Standardize Thread Pool Usage
**Problem**: Inconsistent threading patterns, std::async overhead
**Impact**: Suboptimal parallelism, resource management issues
**Files**: All files using threading (replace std::async with unified pool)
**Effort**: Medium (4-5 days) - replace std::async with thread pool
**Validation**: Consistent thread pool usage across codebase
**Dependencies**: None

**Tasks**:
- Audit current threading usage (std::async, threads)
- Replace with unified work-stealing thread pool
- Update all parallel operations to use thread pool
- Test thread safety and performance
- Document threading patterns and best practices

### 6. Add Data Integrity Triggers
**Problem**: Centroids become stale after coordinate updates
**Impact**: Data inconsistency, manual maintenance burden
**Files**: SQL schema files, trigger definitions
**Effort**: Medium (3-4 days) - implement automatic centroid recalculation
**Validation**: Centroids automatically maintained on updates
**Dependencies**: After batch operations (#1 above)

**Tasks**:
- Design triggers for centroid maintenance
- Implement automatic recalculation on coordinate changes
- Ensure trigger performance doesn't impact operations
- Test data consistency under various update scenarios
- Validate semantic query accuracy with triggers