# Code Improvements Plan

## Overview

Systematic code quality enhancement to eliminate architectural violations, consolidate duplicated functionality, remove dead code, and establish clean, maintainable codebase foundations. This addresses accumulated technical debt from multiple architectural iterations.

## Priority: HIGH - Critical for Long-term Maintainability

---

## 1. Safetensor Ingestion Architectural Violation Remediation

### Current Implementation Analysis

**File**: `cpp/src/tools/ingest_safetensor_modular.cpp` (current active implementation)

**Architectural Violation Details**:
The ingestion system creates fake atoms for tensor dimensions instead of using existing semantic tokens from the vocabulary. This violates the core principle that **coordinates are model-independent, relationships are model-specific**.

**Current Problematic Flow**:
```cpp
// Current: Creates fake dimension atoms (WRONG)
for (size_t dim = 0; dim < embedding_dim; ++dim) {
    std::string fake_token = "dim_" + std::to_string(dim);
    AtomCalculator::compute_vocab_token(fake_token);  // Fake atom creation
    // Links embeddings to meaningless dimension tokens
}

// Should be: Use existing semantic tokens from vocab
for (const auto& [token_text, token_info] : ctx.vocab_tokens) {
    // Link embeddings to REAL semantic tokens that exist in content
    link_embedding_to_token(embedding_data, token_info.comp);
}
```

**Impact on Semantic Model**:
- Database polluted with meaningless "dim_0", "dim_1", etc. atoms
- Embeddings linked to artificial constructs rather than real semantic content
- Semantic queries return nonsensical results linking dimensions to content
- Violates separation of coordinates (model-independent) vs relations (model-specific)

### Implementation Approach

**Phase 1: Vocabulary Token Analysis**
Audit current vocabulary token extraction in `ingest_safetensor_modular.cpp`:

```cpp
// Current vocab transfer (lines 228-241)
if (!model_meta.vocab_tokens.empty() && ctx.vocab_tokens.empty()) {
    ctx.vocab_tokens.resize(model_meta.vocab_tokens.size());
    for (size_t i = 0; i < model_meta.vocab_tokens.size(); ++i) {
        const auto& vt = model_meta.vocab_tokens[i];
        ingest::TokenInfo info;
        info.text = vt.text;
        info.comp = AtomCalculator::compute_vocab_token(vt.text);  // Creates real atoms
        ctx.vocab_tokens[i] = std::move(info);
        ctx.token_to_idx[vt.text] = i;
    }
}
```

**Phase 2: Embedding-to-Token Mapping Correction**
Replace fake dimension atoms with proper semantic token relationships:

```cpp
// NEW: Proper embedding-to-token linking
void link_embeddings_to_vocab_tokens(
    IngestContext& ctx,
    const std::vector<std::vector<float>>& embeddings
) {
    if (ctx.vocab_tokens.size() != embeddings.size()) {
        std::cerr << "[ERROR] Vocabulary size mismatch: "
                  << ctx.vocab_tokens.size() << " tokens vs "
                  << embeddings.size() << " embeddings\n";
        return;
    }

    // Create semantic relations between embeddings and tokens
    for (size_t i = 0; i < ctx.vocab_tokens.size(); ++i) {
        const auto& token_info = ctx.vocab_tokens[i];
        const auto& embedding = embeddings[i];

        // Store embedding data linked to real token composition
        store_embedding_for_composition(
            ctx.conn,
            token_info.comp.id,  // Real composition ID, not fake dimension
            embedding,
            ctx.model_prefix
        );
    }
}
```

**Phase 3: Database Schema Updates**
Modify embedding storage to use proper semantic relationships:

```sql
-- Current: Potentially wrong embedding storage
-- NEW: Ensure embeddings are linked to real composition IDs

CREATE TABLE IF NOT EXISTS embedding_data (
    composition_id BYTEA NOT NULL REFERENCES composition(id),
    model_name TEXT NOT NULL,
    embedding_vector REAL[],  -- Or use specialized vector extension
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (composition_id, model_name)
);
```

**Validation Criteria**:
- No atoms with names like "dim_0", "dim_1" created during ingestion
- All embeddings linked to existing vocabulary token compositions
- Embedding queries return semantically meaningful results
- Database size doesn't grow artificially with fake atoms

---

## 2. Laplacian Projector Pipeline Integration

### Current Implementation Gap

**Problem**: Laplacian projection exists but is never actually called in the ingestion pipeline.

**Code Analysis**: `cpp/src/tools/ingest_safetensor_modular.cpp` line 305-307

```cpp
// Step 5.5: Project embeddings to 4D using Laplacian eigenmaps
std::cerr << "\n[5.5] Projecting token embeddings to 4D semantic coordinates...\n";
if (!ctx.vocab_tokens.empty()) {
    ingest::db::project_and_update_embeddings(conn, ctx, config);
} else {
    std::cerr << "[PROJECTION] No vocab tokens loaded, skipping Laplacian projection\n";
}
```

The function `project_and_update_embeddings` is called, but analysis shows it may not be properly integrated.

### Laplacian Projection Architecture

**Core Algorithm**: Laplacian Eigenmaps for structure-preserving dimensionality reduction

**Mathematical Foundation**:
1. **Similarity Graph Construction**: k-NN graph from high-D embeddings using cosine similarity
2. **Graph Laplacian**: L = D - W (degree matrix minus adjacency matrix)
3. **Eigensolution**: Find smallest non-zero eigenvectors of L
4. **Gram-Schmidt**: Orthonormalize eigenvector columns
5. **Coordinate Mapping**: Map to 4D hypercube with hypersphere projection

**Current Laplacian Implementation**: `cpp/src/core/laplacian_4d.hpp`

```cpp
class LaplacianProjector {
public:
    ProjectionResult project(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<std::string>& labels = {}
    );

private:
    SparseSymmetricMatrix build_similarity_graph(
        const std::vector<std::vector<float>>& embeddings
    );

    SparseSymmetricMatrix build_laplacian(const SparseSymmetricMatrix& W);

    std::vector<std::vector<double>> find_smallest_eigenvectors(
        SparseSymmetricMatrix& L, int k, std::array<double, 4>& eigenvalues_out
    );

    void gram_schmidt_columns(std::vector<std::vector<double>>& Y);
};
```

### Implementation Issues Identified

**Pipeline Integration Problems**:
1. **Conditional Execution**: Only runs if vocab tokens exist, but embeddings may exist without vocab
2. **Data Flow**: Projection called before centroid computation, but centroids should use projected coordinates
3. **Error Handling**: No validation that projection actually succeeded
4. **Storage**: Projected coordinates may not be stored in correct tables

### Corrected Integration Approach

**Phase 1: Pipeline Reordering**
Fix the ingestion pipeline order in `ingest_safetensor_modular.cpp`:

```cpp
// CORRECTED INGESTION ORDER:

// Step 1: Parse and load vocab tokens
// Step 2: Extract embeddings from tensors
// Step 3: Store raw embeddings temporarily
// Step 4: Insert vocab token compositions
// Step 5: PROJECT EMBEDDINGS TO 4D (moved up)
// Step 6: Compute centroids FROM PROJECTED COORDINATES
// Step 7: Store final semantic relations
```

**Phase 2: Projection Validation**
Add comprehensive validation of projection results:

```cpp
struct ProjectionResult {
    std::vector<std::array<uint32_t, 4>> coords;  // 4D coordinates
    std::vector<int64_t> hilbert_lo, hilbert_hi;  // Hilbert indices
    std::array<double, 4> eigenvalues;            // Eigenvalue spectrum
    size_t edge_count;                            // Graph connectivity
};

void validate_projection_result(const ProjectionResult& result) {
    // Check coordinate bounds (should be in [0, 2^32-1])
    // Check Hilbert index validity
    // Check eigenvalue spectrum (first should be near 0)
    // Check graph connectivity (should be connected)
    // Verify 3-sphere constraint satisfaction
}
```

**Phase 3: Coordinate Storage Integration**
Ensure projected coordinates are stored in the correct database tables:

```cpp
void store_projected_coordinates(
    PGconn* conn,
    const std::vector<std::string>& token_texts,
    const ProjectionResult& projection
) {
    // Update composition table with projected centroids
    // Update atom table if needed for coordinate validation
    // Store Hilbert indices for spatial indexing
    // Validate coordinate uniqueness and distribution
}
```

**Validation Criteria**:
- Laplacian projection runs successfully for all embedding ingestion
- Projected coordinates stored in composition.centroid field
- Hilbert indices computed and stored for spatial queries
- Eigenvalue spectrum indicates proper spectral truncation
- Semantic queries work with projected 4D coordinates

---

## 3. SQL Function Consolidation and Deduplication

### Duplicate Function Inventory

**Audit Findings**: 12+ duplicate SQL functions across multiple files with conflicting signatures.

**Pattern Analysis**:
- Same semantic operations implemented in different files
- Inconsistent parameter orders and naming conventions
- Different return types for equivalent operations
- No clear ownership or deprecation strategy

**Examples of Duplication**:

```sql
-- File: sql/003_query_api.sql
CREATE FUNCTION semantic_neighbors(text, integer) RETURNS TABLE(...)

-- File: sql/004_generative_engine.sql
CREATE FUNCTION semantic_neighbors(bytea, integer, text) RETURNS TABLE(...)

-- File: sql/legacy_functions.sql
CREATE FUNCTION find_semantic_neighbors(text, integer, real) RETURNS TABLE(...)
```

### Function Categorization and Consolidation Strategy

**Phase 1: Complete Function Inventory**
Catalog all SQL functions across all files:

```sql
-- Content Identity Functions
content_exists(text) → boolean
content_get(text) → table
get_composition_id(text) → bytea  -- Duplicate?

-- Semantic Similarity Functions
similar(text, integer) → table
text_frechet_similar(text, bigint, integer) → table
semantic_neighbors(text, integer) → table
neighbors(text, integer) → table  -- Duplicate?

-- Edge Walking Functions
follows(text, integer) → table
semantic_walk(text, integer) → table
walk(text, integer) → table  -- Duplicate?

-- Vector Arithmetic Functions
analogy(text, text, text, integer) → table

-- Analysis Functions
composition_info(bytea) → table
edge_count(bytea) → bigint
stats() → table
```

**Phase 2: Signature Standardization**
Establish canonical signatures for each operation:

```sql
-- STANDARDIZED SIGNATURES:

-- Content identity (exact matching)
CREATE FUNCTION content_exists(input_text text) RETURNS boolean
CREATE FUNCTION content_get(input_text text)
RETURNS TABLE(id bytea, depth integer, atom_count bigint, centroid geometry)

-- Semantic similarity (approximate matching)
CREATE FUNCTION similar(query_text text, k integer DEFAULT 10)
RETURNS TABLE(content text, distance real, composition_id bytea)

-- Spatial neighbors (4D proximity)
CREATE FUNCTION semantic_neighbors(query_text text, k integer DEFAULT 10)
RETURNS TABLE(content text, distance real, composition_id bytea)

-- Edge traversal (graph walking)
CREATE FUNCTION semantic_walk(start_text text, max_steps integer DEFAULT 5)
RETURNS TABLE(step integer, content text, weight real, composition_id bytea)

-- Vector arithmetic
CREATE FUNCTION analogy(a text, b text, c text, k integer DEFAULT 5)
RETURNS TABLE(answer text, distance real, composition_id bytea)
```

**Phase 3: Implementation Consolidation**
Merge duplicate implementations with conflict resolution:

```sql
-- CONSOLIDATED IMPLEMENTATION EXAMPLE:

CREATE OR REPLACE FUNCTION semantic_neighbors(query_text text, k integer DEFAULT 10)
RETURNS TABLE(content text, distance real, composition_id bytea)
LANGUAGE sql
STABLE
AS $
  -- Single implementation that replaces all variants
  WITH query_comp AS (
    SELECT id, centroid FROM content_get(query_text) LIMIT 1
  )
  SELECT
    c.label as content,
    ST_3DMAXDistance(q.centroid, c.centroid) as distance,
    c.id as composition_id
  FROM composition c
  CROSS JOIN query_comp q
  WHERE c.centroid IS NOT NULL
  ORDER BY c.centroid <-> q.centroid  -- KNN with distance
  LIMIT k;
$;
```

**Phase 4: Deprecation and Migration**
Implement gradual deprecation strategy:

```sql
-- Mark old functions as deprecated but keep working
CREATE OR REPLACE FUNCTION neighbors(text, integer)  -- Old signature
RETURNS TABLE(content text, distance real, composition_id bytea)
LANGUAGE sql
AS $
  -- Forward to new canonical implementation
  SELECT * FROM semantic_neighbors($1, $2);
$;

-- Add deprecation notices in comments
COMMENT ON FUNCTION neighbors(text, integer) IS
'DEPRECATED: Use semantic_neighbors(text, integer) instead. Will be removed in v8.';
```

**Validation Criteria**:
- Single canonical implementation for each semantic operation
- All duplicate functions forward to canonical versions
- Consistent parameter naming and return types across all functions
- Clear deprecation path for old functions
- No function signature conflicts in the database

---

## 4. Deprecated Codebase Cleanup

### Archive Directory Analysis

**Location**: `sql/archive/` directory with 15+ deprecated files

**File Inventory**:
- `011_unified_atom.sql.DEPRECATED` - Obsolete unified atom/composition table
- `012_semantic_udf.sql.SUPERSEDED_BY_025` - Replaced by modular design
- Multiple files marked `.DEPRECATED`, `.OLD_SCHEMA`, `.SUPERSEDED_BY_*`

### Cleanup Strategy

**Phase 1: Deprecation Impact Assessment**
Analyze which archived files are still referenced:

```bash
# Search for references to archived files
find docs/ -name "*.md" -exec grep -l "011_unified_atom\|012_semantic_udf" {} \;
grep -r "011_unified_atom" sql/ scripts/ cpp/
```

**Phase 2: Safe Archival Process**
Create permanent archival with clear documentation:

```bash
# Create archival directory structure
mkdir -p sql/archival/v5_unified_schema
mkdir -p sql/archival/v5_legacy_functions

# Move with clear naming and documentation
mv sql/archive/011_unified_atom.sql.DEPRECATED \
   sql/archival/v5_unified_schema/011_unified_atom_schema.sql
cp docs/ARCHITECTURAL_CHANGES.md \
   sql/archival/v5_unified_schema/README.md
```

**Phase 3: Build System Cleanup**
Update build scripts to explicitly exclude archived files:

```bash
# scripts/linux/setup-db.sh - Current filtering (line 103)
[[ "$sqlfile" == *"archive"* ]] && continue

# Update to be more explicit:
[[ "$sqlfile" == *"archival"* ]] && continue
[[ "$sqlfile" == *"deprecated"* ]] && continue
```

**Phase 4: Reference Removal**
Eliminate all references to deprecated components:

```sql
-- Remove from SQL comments
-- REMOVE: -- Based on 011_unified_atom.sql schema

-- Update documentation links
-- CHANGE: See sql/011_unified_atom.sql
-- TO: See docs/ARCHITECTURAL_CHANGES.md for schema evolution
```

**Validation Criteria**:
- No references to archived files in active codebase
- Build scripts skip all archived content
- Clear documentation of archival purpose and contents
- Archived files preserved for historical reference only

---

## 5. CLI Integration Gap Completion

### Current CLI Limitations

**File**: `cpp/src/cli/main.cpp`

**Missing Features** (lines 241, 249, 273):
- Query functionality not integrated
- Statistics reporting not implemented
- Test runner not connected

**Current CLI Structure**:
```cpp
int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];

    if (command == "ingest") {
        return perform_ingest(...);  // IMPLEMENTED
    } else if (command == "query") {
        // TODO: Query not yet integrated (line 241)
        std::cerr << "Query functionality not implemented\n";
        return 1;
    } else if (command == "stats") {
        // TODO: Stats not yet integrated (line 249)
        std::cerr << "Stats functionality not implemented\n";
        return 1;
    } else if (command == "test") {
        // TODO: Test runner not yet integrated (line 273)
        std::cerr << "Test runner not implemented\n";
        return 1;
    }
    // ... other commands
}
```

### Complete CLI Implementation

**Phase 1: Query Command Implementation**
Add semantic search functionality:

```cpp
int perform_query(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: hartonomous query <text> [options]\n";
        return 1;
    }

    std::string query_text = args[1];
    int k = 10;  // Default result count

    // Parse options (-k, --similar, --neighbors, etc.)
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "-k" && i + 1 < args.size()) {
            k = std::stoi(args[++i]);
        }
        // ... other options
    }

    // Connect to database
    PGconn* conn = connect_to_database();
    if (!conn) return 1;

    // Execute semantic query
    std::string sql = "SELECT * FROM similar($1, $2)";
    PGresult* res = PQexecParams(conn, sql.c_str(), 2, nullptr,
                                (const char*[]) {query_text.c_str(), std::to_string(k).c_str()},
                                nullptr, nullptr, 0);

    // Display results
    display_query_results(res);

    PQclear(res);
    PQfinish(conn);
    return 0;
}
```

**Phase 2: Statistics Command Implementation**
Add database statistics reporting:

```cpp
int perform_stats(const std::vector<std::string>& args) {
    PGconn* conn = connect_to_database();
    if (!conn) return 1;

    // Execute stats query
    PGresult* res = PQexec(conn, "SELECT * FROM stats()");

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Stats query failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQfinish(conn);
        return 1;
    }

    // Display formatted statistics
    display_stats_table(res);

    PQclear(res);
    PQfinish(conn);
    return 0;
}
```

**Phase 3: Test Runner Integration**
Connect to existing C++ test framework:

```cpp
int perform_test(const std::vector<std::string>& args) {
    // Parse test options (--unit, --integration, --all, --verbose)
    bool run_unit = false, run_integration = false, verbose = false;

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--unit") run_unit = true;
        else if (args[i] == "--integration") run_integration = true;
        else if (args[i] == "--all") { run_unit = run_integration = true; }
        else if (args[i] == "--verbose") verbose = true;
    }

    if (!run_unit && !run_integration) run_unit = true;  // Default

    // Run appropriate test suites
    int result = 0;
    if (run_unit) {
        result |= run_unit_tests(verbose);
    }
    if (run_integration) {
        result |= run_integration_tests(verbose);
    }

    return result;
}
```

**Validation Criteria**:
- CLI supports `query <text>` for semantic search with configurable result count
- CLI supports `stats` command showing comprehensive database statistics
- CLI supports `test` command running unit and integration test suites
- All commands include proper help text and error handling
- Command-line argument parsing handles options correctly

---

## 6. Ingestion Pipeline Data Flow Correction

### Current Ordering Problem

**Issue**: Laplacian projection called before centroid computation, but centroids should be computed from projected coordinates.

**Current Flow in `ingest_safetensor_modular.cpp`**:
```
// WRONG ORDER (current):
Step 4: Insert vocab token compositions (with unprojected centroids)
Step 5.5: PROJECT EMBEDDINGS TO 4D (too late!)
Step 6: Compute centroids FROM ATOM CHILDREN (ignores projection)
```

**Correct Flow**:
```
// CORRECT ORDER:
Step 4: Insert vocab token compositions (temporary centroids)
Step 5: EXTRACT embeddings from model tensors
Step 5.5: PROJECT EMBEDDINGS TO 4D (while we have embedding data)
Step 6: UPDATE centroids using projected coordinates
Step 7: Build semantic relations using correct centroids
```

### Implementation Correction

**Phase 1: Pipeline Reordering**
Fix the sequence in `ingest_safetensor_modular.cpp`:

```cpp
// CORRECTED INGESTION SEQUENCE:

// Step 4: Insert vocab token compositions (temporary, will be updated)
std::cerr << "\n[4] Inserting token compositions...\n";
ingest::db::insert_compositions(conn, ctx);

// Step 5: Extract embeddings from ALL model tensors
std::cerr << "\n[5] Extracting embeddings from model tensors...\n";
// Extract Q/K/V/O, MLP, attention weights, etc.

// Step 5.5: PROJECT ALL EMBEDDINGS TO 4D (moved up!)
std::cerr << "\n[5.5] Projecting embeddings to 4D semantic coordinates...\n";
if (!ctx.vocab_tokens.empty()) {
    ingest::db::project_and_update_embeddings(conn, ctx, config);
} else {
    std::cerr << "[PROJECTION] Warning: No vocab tokens for projection\n";
}

// Step 6: Compute centroids FROM PROJECTED COORDINATES
std::cerr << "\n[6] Computing centroids from projected coordinates...\n";
update_centroids_from_projections(conn, ctx);
```

**Phase 2: Centroid Update Logic**
Implement centroid computation that uses projected coordinates:

```sql
-- Update composition centroids after projection
CREATE OR REPLACE FUNCTION update_projected_centroids()
RETURNS integer
LANGUAGE sql
AS $
  WITH projected_coords AS (
    -- Get coordinates from projection results
    SELECT composition_id, projected_centroid_4d
    FROM embedding_projections
    WHERE projection_completed = true
  )
  UPDATE composition
  SET centroid = projected_coords.projected_centroid_4d,
      hilbert_lo = coords_to_hilbert(projected_centroid_4d)[1],
      hilbert_hi = coords_to_hilbert(projected_centroid_4d)[2]
  FROM projected_coords
  WHERE composition.id = projected_coords.composition_id
  RETURNING composition.id;
$;
```

**Phase 3: Dependency Chain Validation**
Ensure all downstream operations use correct centroids:

```cpp
void validate_centroid_dependencies(PGconn* conn) {
    // Verify all compositions have valid centroids
    // Check that Hilbert indices are computed correctly
    // Validate that semantic relations use projected coordinates
    // Ensure spatial queries work with 4D data
}
```

**Validation Criteria**:
- Ingestion pipeline executes projection before centroid computation
- Composition centroids computed from 4D projected coordinates
- Hilbert indices reflect projected coordinate space
- Semantic queries operate in correct 4D coordinate system
- No dependency violations in data flow