# Comprehensive SQL File Catalog for Hartonomous-Opus Repository

This document catalogs all SQL files in the repository, including their purpose, structure, and key functions. The current SQL architecture spans multiple files with significant overlap and architectural issues.

## Current SQL File Structure

### Core Schema Files

#### `sql/001_schema.sql`
**Purpose**: Core database schema definition for the 3-table architecture
**Tables Created**:
- `atom`: Unicode codepoints with 4D geometry coordinates
- `composition`: Aggregations of atoms/compositions with centroids
- `composition_child`: Ordered children junction table
- `relation`: Semantic edges between entities

**Functions Created**:
- `upsert_relation()`: Upsert relations with weight averaging

**Issues**: Schema definition mixed with functions; should be separated.

#### `sql/002_core_functions.sql`
**Purpose**: Core functions for the 3-table schema
**Key Functions**:
- **Geometry/Distance**: `centroid_distance()`, `centroid_similarity()`, `st_centroid_4d()`
- **Atom Functions**: `atom_is_leaf()`, `atom_centroid()`, `atom_children()`, `atom_text()`, `atom_reconstruct_text()`
- **Spatial Queries**: `atom_knn()`, `atom_hilbert_range()`
- **Semantic Queries**: `semantic_neighbors()`, `attention()`, `analogy()`
- **Statistics**: `atom_stats()`, `db_stats()`
- **Centroid Computation**: `compute_composition_centroid()`, `recompute_composition_centroids()`, `generate_knn_edges()`

**Issues**: Massive file (547 lines) with too many responsibilities; functions should be organized by domain.

#### `sql/003_query_api.sql`
**Purpose**: Query API for text reconstruction and similarity search
**Key Functions**:
- `composition_text()`: Reconstruct text from compositions
- `find_composition()`: Find by label
- `similar_by_centroid()`: Similarity search
- `related_by_attention()`: Find related entities
- `generative_walk()`: Deterministic walk through relations
- `spatial_walk()`: Walk through 4D space
- `db_stats()`: Database statistics

**Issues**: Mixed concerns (text reconstruction vs. generative walking); duplicate `db_stats()` function.

#### `sql/004_generative_engine.sql`
**Purpose**: Generative AI inference using 4D hypercube geometry
**Key Functions**:
- `similar_tokens()`: Find similar tokens by 4D proximity
- `encode_prompt()`: Tokenize text to composition IDs
- `score_candidates()`: Score next token candidates
- `generate_sql()`: Main inference loop
- `vector_analogy()`: A:B::C:? using vector arithmetic
- `gen_db_stats()`: Statistics function

**Issues**: Function name conflicts (`generate_sql()` vs. extension functions); mixed with statistics.

#### `sql/005_bigram_stats.sql`
**Purpose**: Bigram statistics and PMI scoring for semantic relationships
**Tables Created**:
- `bigram_stats`: Left→right token co-occurrence counts and PMI scores
- `unigram_stats`: Individual token counts
- `token_corpus_stats`: Corpus-wide statistics

**Key Functions**:
- `increment_bigram()`: Update bigram counts
- `compute_pmi_scores()`: Calculate PMI values
- `get_bigram_pmi()`: Retrieve PMI scores
- `top_continuations_pmi()`: Get top continuations by PMI
- `extract_bigrams_from_compositions()`: Extract stats from existing data

**Issues**: Good separation of concerns; could be organized better.

#### `sql/006_qa_search.sql`
**Purpose**: Question-answering and text search capabilities
**Key Functions**:
- `search_text()`: Core text search across compositions
- `ask()`: Main Q&A function
- `ask_exact()`: Exact phrase search
- `find_entities()`: Extract named entities

**Issues**: Good functional separation; could integrate better with other search functions.

#### `sql/007_model_registry.sql`
**Purpose**: Model registry for tracking AI models and their embeddings
**Tables Created**:
- `model`: Model metadata and configurations

**Key Functions**:
- Model registration and retrieval functions

**Note**: File appears minimal or placeholder in current codebase.

### Deprecated/Archive Files

#### `sql/archive/011_unified_atom.sql.DEPRECATED`
**Status**: Superseded by 3-table schema

#### `sql/archive/012_semantic_udf.sql.SUPERSEDED_BY_025`
**Status**: Superseded

#### `sql/archive/014_ai_operations.sql.OLD_SCHEMA`
**Status**: Old schema functions

#### `sql/archive/015_centroid_optimization.sql.OLD_SCHEMA`
**Status**: Old optimization functions

#### `sql/archive/016_semantic_queries.sql.OLD_SCHEMA`
**Status**: Old query functions

#### `sql/archive/017_ingestion_ops.sql.OLD_SCHEMA`
**Status**: Old ingestion functions

#### `sql/archive/018_core_queries.sql.OLD_SCHEMA`
**Status**: Old core functions

#### `sql/archive/019_function_aliases.sql.OLD_SCHEMA`
**Status**: Old function aliases

#### `sql/archive/026_laplacian_projection.sql.NEEDS_RENUMBER`
**Status**: Laplacian projection functions (needs integration)

#### `sql/archive/027_batch_operations.sql.DEPRECATED_OLD_SCHEMA`
**Status**: Old batch operations

### PostgreSQL Extensions

#### `cpp/sql/hypercube--1.0.sql`
**Purpose**: Core PostgreSQL extension with C functions
**Provides**: Low-level geometric operations, Hilbert encoding/decoding

#### `cpp/sql/hypercube_ops--1.0.sql`
**Purpose**: Operations extension
**Provides**: Advanced spatial operations

#### `cpp/sql/hypercube_generative--1.0.sql` (likely)
**Purpose**: Generative AI operations

#### `cpp/sql/semantic_ops--1.0.sql`
**Purpose**: Semantic operations

#### `cpp/sql/embedding_ops--1.0.sql`
**Purpose**: Embedding operations with SIMD acceleration

## Architectural Issues Identified

### 1. **Massive Files with Mixed Concerns**
- `002_core_functions.sql`: 547 lines, handles geometry, atoms, queries, stats, centroids
- `004_generative_engine.sql`: 426 lines, mixes similarity, encoding, generation, stats

### 2. **Function Name Conflicts**
- Multiple `db_stats()` functions with different signatures
- `generate_sql()` conflicts with C extension functions

### 3. **Repeated Code Patterns**
- Multiple centroid distance calculations
- Similar KNN implementations across files
- Duplicate statistics functions

### 4. **Poor Separation of Concerns**
- Schema mixed with functions
- Query logic mixed with business logic
- Statistics scattered across files

### 5. **Inconsistent Naming**
- Some functions prefixed (`atom_*`), others not
- Mixed naming conventions

### 6. **Missing Abstractions**
- No views for common queries
- Complex logic embedded in functions instead of stored procedures
- No proper transaction management abstractions

## Proposed Refactored Architecture

### Directory Structure
```
sql/
├── schema/
│   ├── 01_tables.sql           # Core table definitions
│   ├── 02_indexes.sql          # All indexes
│   ├── 03_constraints.sql      # Constraints and triggers
│   └── 04_types.sql            # Custom types
├── functions/
│   ├── geometry/
│   │   ├── distance.sql        # Distance/similarity functions
│   │   ├── centroids.sql       # Centroid computation
│   │   └── spatial.sql         # Spatial operations
│   ├── atoms/
│   │   ├── core.sql            # Basic atom operations
│   │   ├── lookup.sql          # Atom lookup functions
│   │   └── reconstruction.sql  # Text reconstruction
│   ├── compositions/
│   │   ├── core.sql            # Composition operations
│   │   ├── hierarchy.sql       # Parent/child relationships
│   │   └── centroids.sql       # Composition centroids
│   ├── relations/
│   │   ├── core.sql            # Basic relation operations
│   │   ├── semantic.sql        # Semantic queries
│   │   └── edges.sql           # Edge generation
│   ├── queries/
│   │   ├── search.sql          # Text search
│   │   ├── similarity.sql      # Similarity search
│   │   └── generative.sql      # Generative queries
│   └── stats/
│       ├── core.sql            # Statistics functions
│       ├── views.sql           # Statistical views
│       └── reports.sql         # Report generation
├── procedures/
│   ├── ingestion/
│   │   ├── batch_insert.sql    # Bulk data operations
│   │   ├── embedding_import.sql # Embedding ingestion
│   │   └── cleanup.sql         # Data cleanup procedures
│   └── maintenance/
│       ├── centroid_recompute.sql
│       ├── index_rebuild.sql
│       └── stats_update.sql
├── views/
│   ├── public/
│   │   ├── atom_stats_view.sql
│   │   ├── composition_tree_view.sql
│   │   └── relation_graph_view.sql
│   └── admin/
│       ├── system_stats_view.sql
│       └── performance_metrics_view.sql
└── extensions/
    ├── hypercube.control       # Extension metadata
    └── hypercube--1.0.sql      # C functions
```

### Key Principles for Refactoring

1. **One Object Per File**: Each table, view, function, procedure in its own file
2. **Clear Naming**: Consistent prefixes and descriptive names
3. **Layered Architecture**: Views → Functions → Procedures → Extensions
4. **Transactional Boundaries**: Procedures handle complex multi-step operations
5. **Performance Optimization**: Proper indexing and query optimization
6. **Documentation**: Comprehensive comments and usage examples

This refactored structure will provide clean separation between the SQL orchestrator layer and the application code, making the system more maintainable and enterprise-ready.