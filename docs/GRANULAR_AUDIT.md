# Granular File-by-File Audit: Erased Features and Architectural Changes

## Schema Evolution: Unified Atom → Three-Table Model

### What Was Erased: Unified Schema (`sql/archive/011_unified_atom.sql.DEPRECATED`)
**Original Design**: Single `atom` table storing both Unicode atoms AND compositions in one place
- `geom`: GEOMETRY(GEOMETRYZM, 0) - could be POINTZM (atoms) or LINESTRINGZM (compositions)
- `node_role`: SMALLINT enum (0=composition, 1=unicode_atom, 2=token, etc.)
- `relation` table with `parent_id`/`child_id` for all relationships

**Why Erased**: Performance and normalization issues
- Querying atoms vs compositions required expensive `node_role` filtering
- Complex geometry types in single column made spatial indexes inefficient
- Foreign key relationships were implicit through role enums, not enforced
- No clear separation of concerns between different entity types

**What Replaced It**: Three-table schema (`sql/001_schema.sql`)
- `atom`: Unicode codepoints ONLY (POINTZM geom, no children)
- `composition`: Aggregations ONLY (centroid POINTZM, LINESTRINGZM geom for trajectory)
- `composition_child`: Explicit junction table with ordinal positions
- `relation`: Semantic edges between any entities

**Impact**: Complete rewrite of all SQL functions and C++ database code

---

## Algorithm Evolution: Manifold Projection → Laplacian Eigenmaps

### What Was Erased: Random Projection (`cpp/src/archive/manifold_project.cpp_DEPRECATED`)
**Original Algorithm**: Johnson-Lindenstrauss random projection
- Used deterministic `sin()`-based PRNG for projection matrix
- Mapped 384D embeddings to 4D using fixed random matrix
- Fast O(n×d) but semantically lossy

**Why Erased**: Poor semantic preservation
- JL preserves pairwise distances but not semantic structure
- Random projection doesn't understand linguistic relationships
- No consideration of graph structure in embedding space

**What Replaced It**: Laplacian Eigenmaps (`cpp/include/hypercube/laplacian_4d.hpp`)
- Builds k-NN similarity graph from embeddings
- Solves generalized eigenvalue problem on graph Laplacian L = D - W
- Preserves semantic clusters and relationships through spectral analysis
- Gram-Schmidt orthonormalization ensures orthogonal axes

**Impact**: Complete rewrite of projection pipeline, much slower but semantically superior

---

## Ingestion Evolution: Sequitur Grammar → Cascading Pair Encoding

### What Was Erased: Sequitur Grammar-Based Compression (`cpp/src/archive/sequitur_ingest.cpp_ARCHIVE`)
**Original Algorithm**: Grammatical inference for natural language boundaries
- Variable-length rule creation based on digram frequency
- Discovered word/phrase boundaries automatically
- Complex digram tracking and rule utility pruning
- Created grammar hierarchies with arbitrary child counts

**Why Erased**: Complexity and unpredictability
- Algorithm behavior was non-deterministic across different input orders
- Complex state machine with digram indices and rule merging
- Memory-intensive for large corpora
- Didn't guarantee content-addressable properties consistently

**What Replaced It**: Cascading Pair Encoding (CPE) - fixed binary pairing
- Deterministic: same input always produces same DAG structure
- Simple: always pairs adjacent elements, no complex heuristics
- Memory-efficient: streaming processing
- Content-addressable: hash = BLAKE3(ordinal || child_hash || ordinal || child_hash)

**Impact**: Simplified ingestion pipeline, guaranteed determinism

---

## PostgreSQL Extension Evolution: In-Memory Caching → External Tools

### What Was Erased: Complex C++ PostgreSQL Extension (`cpp/src/archive/hypercube_ops.cpp_ORPHANED`)
**Original Design**: Heavy PostgreSQL extension with SPI-based caching
- SPI_execute() for complex queries with in-memory result caching
- AtomCache loaded via depth ranges and ID lists
- In-memory graph algorithms (BFS, random walk, shortest path)
- PARALLEL SAFE functions with shared memory

**Why Erased**: Maintenance complexity and stability issues
- PostgreSQL extensions can crash the entire database server
- SPI memory management was error-prone
- Version compatibility issues with PostgreSQL upgrades
- Complex C++↔SQL interop made debugging difficult

**What Replaced It**: External C++ tools with clean SQL interfaces
- Separate executables (`seed_atoms_parallel`, `ingest_safetensor_4d`)
- Database stores data, tools do computation
- SQL functions are pure SQL, no C++ extensions
- Easier testing and deployment

**Impact**: Removed ~800 lines of complex SPI code, moved functionality to standalone tools

---

## Vocabulary Processing Evolution: Pre-created Compositions → On-Demand CPE

### What Was Erased: Vocabulary Token Pre-Ingester (`cpp/src/archive/vocab_ingest.cpp_ARCHIVE`)
**Original Design**: Pre-create word-level compositions for all transformer vocab tokens
- Read vocab.txt from models like MiniLM
- Create compositions for every possible vocabulary entry
- Batch insert thousands of pre-computed tokens
- Enable "man" ↔ "woman" analogies without content ingestion

**Why Erased**: Database bloat and unused data
- Most vocabulary tokens never appear in actual content
- Pre-creating compositions for unused words wasted space
- Model vocabularies change frequently, requiring re-ingestion
- Didn't align with content-addressable philosophy (only ingest what exists)

**What Replaced It**: On-demand composition creation via CPE
- Compositions created only when processing actual content
- Vocabulary used for embedding ingestion, not composition creation
- Content drives what gets stored, not model vocabularies
- True content-addressability: identical patterns = identical IDs regardless of source

**Impact**: Removed dedicated vocabulary ingester, integrated vocab processing into embedding workflow

---

## AI Operations Evolution: SQL Fallbacks → Pure C++ Implementations

### What Was Erased: SQL Semantic Operations (`sql/archive/014_ai_operations.sql.OLD_SCHEMA`)
**Original Design**: AI operations implemented as SQL functions
- Semantic neighbors stored as depth=1, atom_count=2 compositions
- Attention scores via SQL geometric distance queries
- Analogy via vector arithmetic in SQL
- Random walks using SQL recursive CTEs

**Why Erased**: Performance and complexity in SQL
- Geometric operations slow in pure SQL vs optimized C++
- Recursive CTEs for graph traversal inefficient
- Complex M-coordinate encoding of weights
- No access to SIMD optimizations or advanced algorithms

**What Replaced It**: Dedicated C++ tools with database storage
- `ingest_safetensor_4d`: Laplacian projection for embeddings
- `extract_embeddings.cpp`: Semantic relation extraction
- Relations stored in dedicated `relation` table with metadata
- C++ algorithms use AVX/MKL for performance

**Impact**: Moved AI operations from SQL to C++, relations properly normalized

---

## Code Architecture Changes

### Disabled Features (Still Present but Inactive)

**Projection k-NN Disabled** (`cpp/src/ingest/semantic_extraction.cpp:462`)
```cpp
if (!ENABLE_PROJECTION_KNN) {
    std::cerr << "[SEMANTIC] Projection k-NN DISABLED (use attention_relations for token semantics)\n";
}
```
**Reason**: Replaced by attention-based relations which capture model-internal relationships better than geometric k-NN.

**FEAST Solver Disabled** (`cpp/src/core/laplacian_4d.cpp:988`)
```cpp
// FEAST DISABLED: for Laplacians of this size (n≈30k, k=4)
// classic sparse Lanczos is faster and simpler.
```
**Reason**: Lanczos algorithm proved more numerically stable and faster for the specific problem size.

### Deprecated Functions (Marked for Removal)

**CompositionRecordBinaryDeprecated** (`cpp/src/ingest/cpe.cpp:545`)
- Old binary composition record format
- Replaced by current CompositionRecord with proper 4D coordinates

**compute_centroids_sql removed** (`cpp/src/ingest/projection_db.cpp:131`)
- Old SQL-based centroid computation
- Replaced by deterministic C++ CoordinateMapper::centroid()

### Configuration Changes

**Database User Defaults Changed** (AUDIT_REPORT.md)
- Old: Expected `hartonomous` user
- New: Environment variables required for database connection
- Reason: Made system work with standard PostgreSQL installations

---

## File Organization Rationale

### Archive Directory Structure
- **`.DEPRECATED`**: Files replaced by newer implementations
- **`.OLD_SCHEMA`**: Files incompatible with current schema
- **`.SUPERSEDED_BY_xxx`**: Files replaced by specific newer versions
- **`.NEEDS_RENUMBER`**: Files requiring version number updates
- **`_ARCHIVE`**: Alternative implementations explored but not chosen

### Build System Changes
**Scripts exclude archived files automatically**:
```bash
[[ "$sqlfile" == *"archive"* ]] && continue
```
**Reason**: Prevent accidental application of deprecated schemas

---

## Summary of Major Erasures

| Component | What Was Erased | Why Erased | Replacement |
|-----------|----------------|------------|-------------|
| Schema | Unified atom table | Performance/query issues | 3-table normalized schema |
| Projection | JL random projection | Poor semantic preservation | Laplacian eigenmaps |
| Ingestion | Sequitur grammar | Non-deterministic, complex | Deterministic CPE |
| Extensions | Complex SPI caching | Stability/maintenance | External tools |
| Vocabulary | Pre-created compositions | Database bloat | On-demand CPE creation |
| AI Ops | SQL implementations | Performance limits | Optimized C++ tools |

**Overall Architectural Shift**: From monolithic PostgreSQL-centric design with complex extensions to modular design with clean separation: database stores data, external tools do computation, C++ handles algorithms.