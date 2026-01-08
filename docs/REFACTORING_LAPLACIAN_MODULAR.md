# Laplacian Projection System Refactoring: Modular Architecture

**Date**: 2026-01-08
**Status**: 📋 Design Phase

---

## Executive Summary

Refactor the monolithic 1928-line `laplacian_4d.cpp` into a modular, extensible architecture that supports:

1. **Compositional semantic space** - Atoms, n-grams, and token centroids
2. **Multiple projection strategies** - Pluggable eigensolvers and graph builders
3. **Trajectory computation** - Paths through compositional hierarchyMaintain backward compatibility while enabling future extensions.

---

## Current Problems

### 1. Monolithic Implementation
- **laplacian_4d.cpp**: 1928 lines, 350-line function
- **Mixed concerns**: Matrix ops + graph building + eigensolving + projection
- **No separation**: Can't test or swap components independently

### 2. Configuration Chaos
- **Hardcoded values in 3 places**:
  - `lanczos.hpp:39` → `convergence_tol = 1e-8`
  - `laplacian_4d.hpp:51` → `convergence_tol = 1e-6`
  - `laplacian_4d.cpp:1308` → `convergence_tol = 1e-8` (was just fixed to use config)
- **No single source of truth**
- **Runtime overrides silently ignore user config**

### 3. Missing Compositional Support
- **Current**: Projects token embeddings → 4D coordinates
- **Needed**:
  - Project Unicode atoms → Sphere surface
  - Extract n-grams from tokens
  - Compute composition centroids from constituents
  - Build trajectories through hierarchical structure

### 4. Not Extensible
- **Can't swap eigensolver**: Hardcoded to Lanczos
- **Can't swap graph builder**: Hardcoded to HNSW
- **Can't add new projection types**: All logic in one function

---

## Target Architecture

### Semantic Model

```
Unicode Atoms (Surface)
    ↓
N-grams (Paths)
    ↓
Token Compositions (Interior Centroids)
```

**Example**: Token `"tokenization"` [BPE ID: 42537]

1. **Atoms** (on sphere surface):
   - `'t'`, `'o'`, `'k'`, `'e'`, `'n'`, `'i'`, `'z'`, `'a'`

2. **N-grams** (projected or computed from atoms):
   - 1-grams: `["t", "o", "k", ...]`
   - 2-grams: `["to", "ok", "ke", ...]`
   - 3-grams: `["tok", "oke", "ken", ...]`
   - 4-grams: `["toke", "oken", ...]`
   - ...

3. **Token centroid** (interior):
   - Weighted average of all n-gram coordinates
   - Depth increases → Falls toward center
   - Stored in `composition.centroid`

4. **Trajectory** (path):
   - LINESTRINGZM through child centroids
   - Stored in `composition.geom`

---

## Module Breakdown

### Directory Structure

```
cpp/
├── include/hypercube/
│   ├── core/
│   │   ├── config.hpp                  # Centralized configuration
│   │   ├── sparse_matrix.hpp           # Pure sparse matrix operations
│   │   └── types.hpp                   # Shared types (Coord4D, etc.)
│   │
│   ├── services/
│   │   ├── eigensolver_service.hpp     # Abstract eigensolver interface
│   │   ├── graph_builder_service.hpp   # Abstract graph builder interface
│   │   ├── projection_service.hpp      # Coordinate projection orchestrator
│   │   ├── composition_service.hpp     # Compositional hierarchy manager
│   │   └── relation_service.hpp        # Spatial relationship computation (edges)
│   │
│   ├── strategies/
│   │   ├── lanczos_eigensolver.hpp     # Lanczos implementation
│   │   ├── power_iteration_eigensolver.hpp
│   │   ├── hnsw_graph_builder.hpp      # HNSW-based k-NN
│   │   └── exact_graph_builder.hpp     # Exact k-NN (for small datasets)
│   │
│   ├── helpers/
│   │   ├── orthogonalization.hpp       # Gram-Schmidt helpers
│   │   ├── normalization.hpp           # Hypercube/sphere normalization
│   │   ├── hilbert_curve.hpp           # Hilbert curve computation
│   │   ├── ngram_extractor.hpp         # N-gram extraction
│   │   └── centroid_calculator.hpp     # Weighted centroid computation
│   │
│   └── laplacian_4d.hpp                # High-level orchestrator (backward compat)
│
└── src/
    ├── core/
    │   ├── config.cpp
    │   └── sparse_matrix.cpp
    │
    ├── services/
    │   ├── eigensolver_service.cpp
    │   ├── graph_builder_service.cpp
    │   ├── projection_service.cpp
    │   └── composition_service.cpp
    │
    ├── strategies/
    │   ├── lanczos_eigensolver.cpp
    │   ├── power_iteration_eigensolver.cpp
    │   ├── hnsw_graph_builder.cpp
    │   └── exact_graph_builder.cpp
    │
    ├── helpers/
    │   ├── orthogonalization.cpp
    │   ├── normalization.cpp
    │   ├── hilbert_curve.cpp
    │   ├── ngram_extractor.cpp
    │   └── centroid_calculator.cpp
    │
    └── laplacian_4d.cpp                # Orchestrator implementation
```

---

## Interface Contracts

### 1. Configuration Manager

**Purpose**: Single source of truth for all configuration

```cpp
// config.hpp
namespace hypercube {

struct GraphConfig {
    int k_neighbors = 15;
    float similarity_threshold = 0.0f;
    enum GraphType { HNSW, EXACT, BALLTREE };
    GraphType type = HNSW;
};

struct SolverConfig {
    double convergence_tol = 1e-4;      // Unified tolerance
    int max_iterations = 500;
    int num_threads = 0;                 // 0 = auto
    enum SolverType { LANCZOS, POWER_ITERATION, ARPACK };
    SolverType type = LANCZOS;
};

struct ProjectionConfig {
    bool project_to_sphere = true;
    double sphere_radius = 1.0;
    bool compute_hilbert = true;
};

struct CompositionConfig {
    int min_ngram_size = 1;
    int max_ngram_size = 5;
    bool compute_trajectories = true;
    enum CentroidMethod { WEIGHTED_AVERAGE, GEOMETRIC_MEDIAN };
    CentroidMethod method = WEIGHTED_AVERAGE;
};

class ConfigManager {
public:
    explicit ConfigManager(const LaplacianConfig& user_config);

    const GraphConfig& graph_config() const;
    const SolverConfig& solver_config() const;
    const ProjectionConfig& projection_config() const;
    const CompositionConfig& composition_config() const;

    void validate();  // Throws if config is invalid
};

}
```

---

### 2. Eigensolver Service (Strategy Pattern)

**Purpose**: Abstract eigensolver with pluggable implementations

```cpp
// eigensolver_service.hpp
namespace hypercube {

struct EigenResult {
    std::vector<std::vector<double>> eigenvectors;  // Column vectors
    std::vector<double> eigenvalues;
    int iterations;
    double final_residual;
};

class EigensolverService {
public:
    virtual ~EigensolverService() = default;

    // Find k smallest non-zero eigenvectors of symmetric matrix
    virtual EigenResult solve(
        const SparseSymmetricMatrix& matrix,
        int k,
        const SolverConfig& config
    ) = 0;

    // Set progress callback
    virtual void set_progress_callback(
        std::function<void(const std::string&, int, int)> callback
    ) = 0;
};

// Factory
class EigensolverFactory {
public:
    static std::unique_ptr<EigensolverService> create(
        SolverConfig::SolverType type
    );
};

}
```

**Implementations**:

```cpp
// lanczos_eigensolver.hpp
class LanczosEigensolver : public EigensolverService {
public:
    EigenResult solve(
        const SparseSymmetricMatrix& matrix,
        int k,
        const SolverConfig& config
    ) override;

    void set_progress_callback(...) override;
};

// power_iteration_eigensolver.hpp
class PowerIterationEigensolver : public EigensolverService {
    // Alternative solver for comparison
};
```

---

### 3. Graph Builder Service (Strategy Pattern)

**Purpose**: Abstract k-NN graph construction

```cpp
// graph_builder_service.hpp
namespace hypercube {

struct GraphStats {
    size_t num_nodes;
    size_t num_edges;
    double avg_degree;
    double min_similarity;
    double max_similarity;
};

class GraphBuilderService {
public:
    virtual ~GraphBuilderService() = default;

    // Build k-NN similarity graph from embeddings
    virtual SparseSymmetricMatrix build_similarity_graph(
        const std::vector<std::vector<float>>& embeddings,
        const GraphConfig& config
    ) = 0;

    // Get statistics
    virtual GraphStats get_stats() const = 0;
};

class GraphBuilderFactory {
public:
    static std::unique_ptr<GraphBuilderService> create(
        GraphConfig::GraphType type
    );
};

}
```

---

### 4. Projection Service (Orchestrator)

**Purpose**: Coordinate entire projection pipeline

```cpp
// projection_service.hpp
namespace hypercube {

struct ProjectionResult {
    std::vector<std::array<uint32_t, 4>> coords;
    std::vector<int64_t> hilbert_lo;
    std::vector<int64_t> hilbert_hi;
    std::array<double, 4> eigenvalues;
    double total_variance_explained;
    size_t edge_count;
};

class ProjectionService {
public:
    explicit ProjectionService(const ConfigManager& config);

    // Main projection pipeline
    ProjectionResult project_embeddings(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<std::string>& labels
    );

    // Individual steps (for granular control)
    SparseSymmetricMatrix build_graph(
        const std::vector<std::vector<float>>& embeddings
    );

    SparseSymmetricMatrix compute_laplacian(
        const SparseSymmetricMatrix& similarity_graph
    );

    EigenResult solve_eigenvectors(
        const SparseSymmetricMatrix& laplacian,
        int k = 4
    );

    std::vector<std::array<uint32_t, 4>> normalize_to_hypercube(
        const std::vector<std::vector<double>>& eigenvectors
    );

    void project_to_sphere(
        std::vector<std::array<uint32_t, 4>>& coords
    );

private:
    ConfigManager config_;
    std::unique_ptr<GraphBuilderService> graph_builder_;
    std::unique_ptr<EigensolverService> eigensolver_;
};

}
```

---

### 5. Composition Service (NEW)

**Purpose**: Manage compositional hierarchy (atoms → n-grams → tokens)

```cpp
// composition_service.hpp
namespace hypercube {

struct AtomCoord {
    uint32_t unicode_codepoint;
    std::array<uint32_t, 4> coord;  // On sphere surface
};

struct NgramCoord {
    std::string ngram;
    std::array<uint32_t, 4> coord;
    int depth;  // 1 = atoms only, 2+ = nested
};

struct CompositionResult {
    std::string label;
    std::array<uint32_t, 4> centroid;           // Interior point
    std::vector<NgramCoord> constituent_ngrams;  // Trajectory components
    std::vector<std::array<uint32_t, 4>> trajectory;  // Path through space
    int depth;
    size_t atom_count;
};

class CompositionService {
public:
    explicit CompositionService(
        const ConfigManager& config,
        ProjectionService* projection_service
    );

    // Extract n-grams from token
    std::vector<std::string> extract_ngrams(
        const std::string& token
    );

    // Project n-gram (or retrieve if it's a known atom/composition)
    std::array<uint32_t, 4> project_ngram(
        const std::string& ngram
    );

    // Compute token centroid from constituent n-grams
    CompositionResult compute_composition(
        const std::string& token,
        const std::vector<float>& token_embedding
    );

    // Build trajectory through compositional hierarchy
    std::vector<std::array<uint32_t, 4>> compute_trajectory(
        const std::vector<std::string>& ngrams
    );

private:
    ConfigManager config_;
    ProjectionService* projection_service_;

    // Cache for already-projected atoms/ngrams
    std::unordered_map<std::string, std::array<uint32_t, 4>> ngram_cache_;
    std::unordered_map<uint32_t, std::array<uint32_t, 4>> atom_cache_;
};

}
```

---

### 6. Relation Service (Spatial Edges)

**Purpose**: Compute and store spatial relationships (edges) between atoms and compositions

```cpp
// relation_service.hpp
namespace hypercube {

enum class RelationType : char {
    SEQUENCE = 'S',      // Sequential in text/token order
    ATTENTION = 'A',     // Learned attention weights from model
    PROXIMITY = 'P'      // Geometric proximity in 4D space
};

enum class EntityType : char {
    ATOM = 'A',
    COMPOSITION = 'C'
};

struct Entity {
    EntityType type;
    std::vector<uint8_t> id;  // BYTEA (BLAKE3 hash)
    std::array<uint32_t, 4> coord;
    std::vector<std::array<uint32_t, 4>> trajectory;  // For compositions
};

struct Relation {
    Entity source;
    Entity target;
    RelationType type;
    float weight;
    std::string source_model;
    int layer;
    std::string component;
};

struct SpatialMetrics {
    double euclidean_distance;      // ST_Distance
    double frechet_distance;        // ST_FrechetDistance (trajectory similarity)
    bool intersects;                // ST_Intersects (do trajectories cross?)
    double hausdorff_distance;      // ST_HausdorffDistance
    double line_locate_point;       // ST_LineLocatePoint (position along trajectory)
};

class RelationService {
public:
    explicit RelationService(
        const ConfigManager& config,
        DatabaseConnection* db
    );

    // Compute spatial metrics between two entities
    SpatialMetrics compute_spatial_metrics(
        const Entity& source,
        const Entity& target
    );

    // Find proximity-based relations (nearby in 4D space)
    std::vector<Relation> find_proximity_relations(
        const Entity& entity,
        double max_distance,
        size_t max_results = 100
    );

    // Find trajectory intersections (compositions whose paths cross)
    std::vector<Relation> find_trajectory_intersections(
        const Entity& composition,
        double intersection_tolerance = 0.01
    );

    // Find similar trajectories using Fréchet distance
    std::vector<Relation> find_similar_trajectories(
        const Entity& composition,
        double max_frechet_distance,
        size_t max_results = 100
    );

    // Store or update relation in database
    void upsert_relation(const Relation& relation);

    // Batch insert relations (optimized for bulk operations)
    void batch_upsert_relations(const std::vector<Relation>& relations);

    // Get all relations for an entity
    std::vector<Relation> get_relations(
        const Entity& entity,
        RelationType type = RelationType::PROXIMITY
    );

private:
    ConfigManager config_;
    DatabaseConnection* db_;

    // PostGIS query helpers
    std::string build_spatial_query(
        const std::string& spatial_function,
        const Entity& entity,
        const std::map<std::string, std::string>& params
    );

    // Convert 4D coordinates to PostGIS POINTZM
    std::string coords_to_pointzm(const std::array<uint32_t, 4>& coord);

    // Convert trajectory to PostGIS LINESTRINGZM
    std::string trajectory_to_linestringzm(
        const std::vector<std::array<uint32_t, 4>>& trajectory
    );
};

}
```

**Key spatial operations**:

1. **ST_Distance**: Find atoms/compositions within distance threshold
   ```sql
   SELECT * FROM atom a1, atom a2
   WHERE ST_Distance(a1.geom, a2.geom) < 1000.0
   ```

2. **ST_Intersects**: Find compositions whose trajectories cross
   ```sql
   SELECT * FROM composition c1, composition c2
   WHERE ST_Intersects(c1.geom, c2.geom)
   ```

3. **ST_FrechetDistance**: Measure trajectory similarity
   ```sql
   SELECT c1.label, c2.label,
          ST_FrechetDistance(c1.geom, c2.geom) AS similarity
   FROM composition c1, composition c2
   WHERE ST_FrechetDistance(c1.geom, c2.geom) < 500.0
   ```

4. **ST_HausdorffDistance**: Alternative trajectory similarity metric

5. **ST_LineLocatePoint**: Find position along trajectory
   ```sql
   SELECT ST_LineLocatePoint(c.geom, a.geom) AS position
   FROM composition c, atom a
   ```

**Relation types**:

- **SEQUENCE ('S')**: Token order (e.g., "the" → "quick" → "brown")
- **ATTENTION ('A')**: Model attention weights between tokens
- **PROXIMITY ('P')**: Geometric closeness in 4D space

**Weight interpretation**:

- **Sequence**: Position delta (1.0 = adjacent tokens)
- **Attention**: Raw attention weight from model
- **Proximity**: Inverse distance (closer = higher weight)

---

### 7. Helper Modules

```cpp
// orthogonalization.hpp
namespace hypercube::helpers {
    void gram_schmidt(std::vector<std::vector<double>>& vectors);
    double dot_product(const std::vector<double>& a, const std::vector<double>& b);
    void normalize_vector(std::vector<double>& v);
}

// normalization.hpp
namespace hypercube::helpers {
    std::vector<std::array<uint32_t, 4>> normalize_to_hypercube(
        const std::vector<std::vector<double>>& eigenvectors
    );

    void project_to_sphere(
        std::vector<std::array<uint32_t, 4>>& coords,
        double radius
    );
}

// hilbert_curve.hpp
namespace hypercube::helpers {
    struct HilbertIndex {
        int64_t lo;
        int64_t hi;
    };

    HilbertIndex coords_to_hilbert(const std::array<uint32_t, 4>& coord);
}

// ngram_extractor.hpp
namespace hypercube::helpers {
    std::vector<std::string> extract_ngrams(
        const std::string& text,
        int min_n,
        int max_n
    );
}

// centroid_calculator.hpp
namespace hypercube::helpers {
    std::array<uint32_t, 4> weighted_centroid(
        const std::vector<std::array<uint32_t, 4>>& points,
        const std::vector<double>& weights = {}
    );

    std::array<uint32_t, 4> geometric_median(
        const std::vector<std::array<uint32_t, 4>>& points
    );
}
```

---

## Migration Strategy

### Phase 1: Extract Helpers (LOW RISK)
**Goal**: Move pure functions to helpers without changing logic

1. Extract `gram_schmidt_columns()` → `helpers/orthogonalization.cpp`
2. Extract `normalize_to_hypercube()` → `helpers/normalization.cpp`
3. Extract `project_to_sphere()` → `helpers/normalization.cpp`
4. Extract Hilbert computation → `helpers/hilbert_curve.cpp`
5. **Test**: Verify all tests still pass

### Phase 2: Create Service Interfaces (MEDIUM RISK)
**Goal**: Define contracts without breaking existing code

1. Create `services/eigensolver_service.hpp` interface
2. Create `services/graph_builder_service.hpp` interface
3. Create `services/projection_service.hpp` interface
4. **No implementations yet** - just define APIs

### Phase 3: Implement Strategy Wrappers (MEDIUM RISK)
**Goal**: Wrap existing code in new interfaces

1. Wrap Lanczos solver → `strategies/lanczos_eigensolver.cpp`
2. Wrap HNSW graph builder → `strategies/hnsw_graph_builder.cpp`
3. **Test**: Verify wrappers behave identically to original

### Phase 4: Create Projection Service (HIGH RISK)
**Goal**: Migrate orchestration logic to service

1. Implement `ProjectionService` using strategies
2. Keep old `LaplacianProjector::project()` as thin wrapper
3. **Test**: Run side-by-side comparison (old vs new)
4. **Verify**: Eigenvalues, coordinates, and Hilbert indices match

### Phase 5: Add Composition Support (NEW FEATURE)
**Goal**: Implement compositional hierarchy

1. Implement `CompositionService`
2. Add n-gram extraction
3. Add centroid computation
4. Add trajectory building
5. **Test**: Verify tokens decompose correctly

### Phase 6: Centralize Configuration (LOW RISK)
**Goal**: Single source of truth for all config

1. Create `ConfigManager`
2. Migrate all hardcoded values to config
3. **Remove**: All `= 1e-8` or `= 1e-6` defaults
4. **Test**: Verify user config is respected

### Phase 7: Cleanup & Deprecate (LOW RISK)
**Goal**: Remove old monolithic code

1. Mark old `LaplacianProjector` methods as `[[deprecated]]`
2. Update all callers to use `ProjectionService`
3. **Optional**: Remove deprecated code after validation period

---

## Testing Strategy

### Unit Tests (Per Module)
```
tests/
├── test_orthogonalization.cpp
├── test_normalization.cpp
├── test_hilbert_curve.cpp
├── test_ngram_extractor.cpp
├── test_centroid_calculator.cpp
├── test_lanczos_eigensolver.cpp
├── test_hnsw_graph_builder.cpp
└── test_composition_service.cpp
```

### Integration Tests
```
tests/
├── test_projection_service.cpp      # End-to-end projection
├── test_composition_pipeline.cpp    # Atoms → n-grams → tokens
└── test_config_propagation.cpp      # Config flows correctly
```

### Regression Tests
```
tests/
└── test_backward_compatibility.cpp  # Old API still works
```

### Validation Test
```
tests/
└── test_ingestion_equivalence.cpp   # Old vs new produce identical results
```

---

## Performance Considerations

### Current Bottlenecks
1. **HNSW index build**: 35.5 seconds for 50K tokens
2. **Lanczos iterations**: ~1-5 seconds (depends on convergence)
3. **Gram-Schmidt**: ~100ms
4. **Normalization**: ~50ms

### Optimizations Enabled by Refactoring
1. **Parallel graph building**: Build HNSW for multiple models simultaneously
2. **Cached n-grams**: Reuse projections across tokens
3. **Batch eigensolver**: Solve for multiple tensors at once
4. **Alternative solvers**: Power iteration may be faster for small matrices

### Memory Usage
- **Current**: ~2GB for 50K × 1024D embeddings
- **With n-grams**: ~4-5GB (store all intermediate projections)
- **Mitigation**: Stream n-grams instead of caching all

---

## Backward Compatibility

### Old API (Preserved)
```cpp
LaplacianConfig config;
config.k_neighbors = 15;
config.convergence_tol = 1e-4;

LaplacianProjector projector(config);
auto result = projector.project(embeddings, labels);
```

### New API (Recommended)
```cpp
ConfigManager config_mgr(user_config);
ProjectionService proj_service(config_mgr);

auto result = proj_service.project_embeddings(embeddings, labels);
```

### Transition Period
- **Old API**: Internally delegates to new services
- **Deprecation warnings**: After 1 release cycle
- **Removal**: After 2 release cycles or when all callers migrated

---

## Success Criteria

1. ✅ **Modularity**: Each component has single responsibility
2. ✅ **Extensibility**: Can add new solvers/builders without touching core
3. ✅ **Testability**: Each module has unit tests with >80% coverage
4. ✅ **Performance**: No regression (±5%) in projection time
5. ✅ **Correctness**: Identical results to old implementation (exact eigenvalues/coords)
6. ✅ **Compositional**: Full n-gram → token → centroid pipeline working
7. ✅ **Configuration**: Single source of truth, no hardcoded values

---

## Timeline

### Estimated Effort
- **Phase 1**: 2-3 hours
- **Phase 2**: 1-2 hours
- **Phase 3**: 3-4 hours
- **Phase 4**: 4-5 hours
- **Phase 5**: 5-6 hours (NEW feature)
- **Phase 6**: 1-2 hours
- **Phase 7**: 1-2 hours

**Total**: ~17-24 hours

### Incremental Delivery
- After Phase 1: Helpers available for other uses
- After Phase 3: Can swap eigensolvers
- After Phase 4: Full projection service functional
- After Phase 5: Compositional pipeline complete

---

## Open Questions

1. **N-gram projection**:
   - Use model's embedding for each n-gram substring?
   - Or compute from constituent character embeddings?
   - **Recommendation**: Hybrid - use model embeddings if available, compute otherwise

2. **Atom seeding**:
   - Are all Unicode atoms pre-seeded at startup?
   - Or discovered during ingestion?
   - **Recommendation**: Pre-seed common Unicode planes, discover rare ones

3. **Trajectory granularity**:
   - Store all n-grams (1-5 grams = ~50 points per token)?
   - Or just endpoints (atoms + centroid)?
   - **Recommendation**: Store 1-grams, 2-grams, full token (reduces to ~10-15 points)

4. **Centroid weighting**:
   - Uniform weights for all n-grams?
   - Weight by frequency/importance?
   - **Recommendation**: Weight by n-gram size (longer = more weight)

5. **Database updates**:
   - Update `composition` table with new centroids?
   - Or keep existing structure and add new `composition_ngram` table?
   - **Recommendation**: Use existing schema, populate `geom` and `centroid` fields

---

## Next Steps

**Immediate**:
1. Review this design document
2. Answer open questions
3. Proceed with Phase 1 (extract helpers)

**Once helpers extracted**:
4. Create service interfaces
5. Wrap existing code in strategies
6. Build projection service
7. Add composition support
8. Validate with real ingestion

