# Comprehensive Functionality Catalog for Hartonomous-Opus

## Overview
Hartonomous-Opus is a **deterministic, lossless, content-addressable geometric semantic substrate** that maps all digital content into a 4D hypercube coordinate system with Hilbert curve indexing for efficient spatial queries. It uses PostgreSQL as the database backend with custom extensions.

## Core Concepts

### Atoms
- Unicode codepoints as fundamental constants (perimeter landmarks)
- Each codepoint → 4D coordinate (32 bits per dimension)
- BLAKE3 hash as content-addressed ID
- Hilbert curve index (128-bit) for spatial ordering
- All atoms distributed on the 3-sphere surface (S³ in 4D)
- Lossless: PostGIS GEOMETRY stores full double precision (2^53 mantissa)

### Compositions
- Binary Merkle DAG via PMI Contraction
- PMI (Pointwise Mutual Information) identifies significant co-occurrences
- Highest-PMI pairs contracted into new compositions recursively
- Result: Logarithmic dictionary growth, linear content growth
- Content-addressed: "the" from any document = same ID
- Geometry = LINESTRINGZM trajectory through 2 child centroids

### Two Table Model
- `atom` table stores nodes (leaves and compositions)
- `relation` table stores edges (parent→child with ordinal)
- `ordinal = 1` for left child, `ordinal = 2` for right child
- `relation_type = 'C'` for composition edges

### Global Deduplication
- First ingest creates patterns; subsequent ingests reuse existing compositions
- The more content ingested, the more deduplication occurs
- Binary tree structure = exactly 2 children per composition

## Core Components

### Types (`cpp/include/hypercube/types.hpp`)
- `Point4D`: 4D point with uint32 coordinates, comparison operators, surface check
- `HilbertIndex`: 128-bit index (lo/hi uint64), arithmetic operators
- `Blake3Hash`: 32-byte hash, hex conversion, hashing functor
- `Point4F`: Floating-point 4D point for optimization, conversions, distances
- `UnicodeAtom`: Codepoint with metadata
- `Composition`: Merkle DAG node
- `SemanticEdge`: Weighted relationship between nodes
- `AtomCategory`: Unicode category enum with string conversion

### Coordinates (`cpp/include/hypercube/coordinates.hpp`, `cpp/src/core/coordinates.cpp`)
- **CoordinateMapper** class: Hopf fibration mapping from semantic order to 4D coordinates
- `map_codepoint()`: Codepoint to Point4D
- `map_codepoint_full()`: Codepoint to Point4D + HilbertIndex
- `map_codepoint_float()`: Codepoint to Point4F (optimization)
- `categorize()`: Get AtomCategory for codepoint
- `centroid()`, `weighted_centroid()`: Compute composition coordinates
- `euclidean_distance()`: 4D distance calculation
- **Optimization Pipeline**:
  - `compute_diagnostics()`: Distribution statistics
  - `apply_deterministic_jitter()`: Collision resolution
  - `bucketed_tangent_lloyd()`: Local optimization
  - `global_knn_repulsion()`: Global repulsive forces
  - `optimize_distribution()`: Complete pipeline

### Hilbert Curve (`cpp/include/hypercube/hilbert.hpp`, `cpp/src/core/hilbert.cpp`)
- **HilbertCurve** class: 4D Hilbert curve implementation (Skilling's algorithm)
- `coords_to_index()`: Point4D → HilbertIndex
- `index_to_coords()`: HilbertIndex → Point4D
- `index_to_raw_coords()`: HilbertIndex → raw corner-origin Point4D
- `distance()`, `in_range()`: Index operations
- `transpose_to_axes()`, `axes_to_transpose()`: Internal transforms

### Operations (`cpp/include/hypercube/ops.hpp`)
- **SIMD-Optimized Distance Calculations**:
  - `batch_distances_avx2()`, `batch_distances_portable()`
  - `find_knn()`: k-nearest neighbors from distances
- **ThreadPool**: Parallel processing with work stealing
- **AtomCache**: In-memory graph with loading, lookups, graph operations
  - `load_by_depth_range()`, `load_by_ids()`, `load_all_leaves()`
  - `reconstruct_text()`, `get_neighbors()`, `shortest_path()`, `random_walk()`
  - `knn_by_centroid()`, `knn_by_hilbert()`
- **HilbertPartitioning**: Partition atoms for parallel processing
- **Content Hashing**: `compute_content_hash()`, `batch_content_hash()`
- **Fréchet Distance**: `frechet_distance()`, `batch_frechet_knn()`
- **Jaccard Similarity**: `semantic_jaccard()`, `batch_jaccard()`
- **Analogy Operations**: `analogy_knn()` (vector arithmetic)

### Generative Engine (`cpp/include/hypercube/generative.hpp`)
- **VocabularyCache**: Stores vocab entries with centroids
- **BigramCache**: PMI scores for pairs
- **AttentionCache**: Attention weights between tokens
- **GenerativeEngine**: LLM-like generation using hypercube
  - Scoring: centroid (4D proximity), PMI, attention, global frequency
  - Candidate filtering: Hilbert proximity pre-filter
  - Selection: greedy or stochastic (temperature sampling)
  - `generate()`: Generate text sequences
  - `find_similar()`: Find similar tokens

### Database Operations (`cpp/include/hypercube/db/operations.hpp`)
- **Transaction**: RAII transaction wrapper
- **CopyStream**: COPY protocol streaming with chunking
- **CopyWriter**: High-level COPY helper with escaping
- **ParallelBatchBuilder**: Thread-local buffers for parallel inserts
- **PreparedStatementCache**: Cache prepared statements
- **Common Patterns**: Temp table schemas for composition/relation tables

### Ingest Components
- **CPE (Cascading Pair Encoding)**: Sliding window pairing at all tiers
- **Universal Ingester**: General content ingestion
- **Sequitur Ingester**: Grammar-based compression
- **PMI Contraction**: Mutual information-based merging
- **Parallel CPE**: Multi-threaded ingestion
- **SafeTensor Loader**: AI model tensor loading with memory mapping
- **Projection DB**: Store embeddings in database
- **Metadata Ingest**: Model metadata and configuration ingestion

### Laplacian Eigenmaps (`cpp/include/hypercube/laplacian_4d.hpp`, `cpp/src/core/laplacian_4d.cpp`)
- **SparseSymmetricMatrix**: CSR matrix for Laplacian computation (MKL optimized)
- **LaplacianProjector**: Project embeddings to 4D hypercube
  - Build k-NN similarity graph
  - Compute unnormalized Laplacian L = D - W
  - Solve for smallest non-zero eigenvectors (Lanczos)
  - Gram-Schmidt orthonormalization
  - Normalize to hypercube coordinates
  - Optional hypersphere projection

### Lanczos Solver (`cpp/include/hypercube/lanczos.hpp`)
- **TridiagonalEigensolver**: QR algorithm for tridiagonal matrices
- **LanczosSolver**: Iterative eigenvalue solver with deflation
- **ConjugateGradient**: Linear system solver
- SIMD-optimized vector operations

### KNN Engine (`cpp/include/hypercube/knn/hnsw_engine.hpp`)
- **HNSWEngine**: Hierarchical Navigable Small World for approximate KNN
- Configurable dimensions, neighbors, layers
- Add vectors, search with ef parameter

### Embedding Operations (`cpp/include/hypercube/embedding_ops.hpp`)
- **EmbeddingBulkUpdater**: Batch update embeddings in database
- Similarity search functions

### PostgreSQL Extensions
- **hypercube_pg.c**: Core extension functions
- **generative_pg.c**: Generative model functions
- **embedding_ops_pg.c**: Embedding operations
- **semantic_ops_pg.c**: Semantic query functions

### Utility Libraries
- **Threading**: Cancellation tokens, progress trackers, exception propagation
- **UTF8**: Decode/encode UTF-8, extract unique codepoints
- **Tensor Loader**: Load SafeTensor files with memory mapping
- **Vector Ops**: SIMD-optimized vector operations (L2/L1 norm, normalization)
- **CPU Features**: Runtime AVX2 detection
- **BLAKE3**: Cryptographic hashing for content addressing

## SQL Schema (Three Table Model)
- `atom` table: Unicode codepoints only (~1.1M rows)
  - id (BYTEA PRIMARY KEY): BLAKE3 hash of codepoint bytes
  - codepoint (INTEGER UNIQUE): Unicode codepoint (0-0x10FFFF)
  - value (BYTEA): UTF-8 bytes of the character
  - geom (GEOMETRY(POINTZM, 0)): 4D coordinate mapping
  - hilbert_lo/hi (NUMERIC(20,0)): Hilbert index (128-bit split)
  - created_at (TIMESTAMPTZ)
- `composition` table: Aggregations of atoms/compositions
  - id (BYTEA PRIMARY KEY): BLAKE3 hash of concatenated child hashes
  - label (TEXT): Human-readable label (e.g., "whale", "##ing")
  - depth (INTEGER): 1 = direct atom children, 2+ = nested
  - child_count (INTEGER): Number of direct children
  - atom_count (BIGINT): Total leaf atoms in subtree
  - geom (GEOMETRY(LINESTRINGZM, 0)): Path through child centroids
  - centroid (GEOMETRY(POINTZM, 0)): 4D centroid for similarity
  - hilbert_lo/hi (NUMERIC(20,0)): Hilbert index
  - created_at (TIMESTAMPTZ)
- `composition_child` table: Ordered children junction table
  - composition_id (BYTEA FK): References composition.id
  - ordinal (SMALLINT): Position in sequence (0-based)
  - child_type (CHAR(1)): 'A' = atom, 'C' = composition
  - child_id (BYTEA): References atom.id or composition.id
- `relation` table: Semantic edges with weights (knowledge graph)
  - id (BIGSERIAL PRIMARY KEY)
  - source_type/target_type (CHAR(1)): 'A' = atom, 'C' = composition
  - source_id/target_id (BYTEA): References atom/composition ids
  - relation_type (CHAR(1)): S=sequence, A=attention, P=proximity, T=temporal, etc.
  - weight (REAL): Intensity/strength of relation
  - source_model (TEXT): Which model contributed this edge
  - source_count (INTEGER): How many times seen (for averaging)
  - layer (INTEGER): Model layer (-1 = N/A)
  - component (TEXT): Model component (attention, mlp, etc.)
  - created_at (TIMESTAMPTZ)

## SQL Functions
### Core Functions
- `centroid_distance(p_a, p_b)`: Euclidean distance between 4D points
- `centroid_similarity(p_a, p_b)`: Inverse distance similarity (0-1)

### Atom Functions
- `atom_is_leaf(p_id)`: Check if entity is a leaf atom
- `atom_centroid(p_id)`: Get 4D centroid (geom for atoms, centroid for compositions)
- `atom_children(p_id)`: Get ordered children of composition
- `atom_child_count(p_id)`: Get child count (0 for atoms)
- `atom_by_codepoint(p_cp)`: Lookup atom by Unicode codepoint
- `atom_exists(p_id)`: Check if hash exists in atom or composition
- `atom_text(p_id)`: Get character text from atom
- `get_atoms_by_codepoints(p_codepoints[])`: Batch lookup atoms by codepoints

### Text Reconstruction
- `atom_reconstruct_text(p_id)`: Recursively reconstruct text from composition DAG

### Spatial Queries
- `atom_knn(p_id, p_k)`: K-nearest neighbors using 4D geometry
- `atom_hilbert_range(p_hi_lo, p_hi_hi, p_lo_lo, p_lo_hi)`: Hilbert range query

### Semantic Queries
- `semantic_neighbors(p_id, p_limit)`: Get semantic neighbors from relation table
- `attention(p_id, p_k)`: Attention scores (4D proximity)
- `analogy(p_a, p_b, p_c, p_k)`: A:B :: C:D analogy using vector arithmetic

### Statistics
- `atom_stats()`: Return counts of atoms, compositions, relations, max depth
- `atom_type_stats` view: Compatibility view for C++ tests

### Compatibility Functions
- `atom_distance(p_id1, p_id2)`: 3D distance between atoms
- `atom_nearest_spatial(p_id, p_limit)`: Nearest spatial neighbors

### 4D Centroid Functions
- `st_centroid_4d(geom)`: Compute 4D centroid of geometry
- `compute_composition_centroid(comp_id)`: Centroid from atom children
- `recompute_composition_centroids(batch_size)`: Recompute all centroids hierarchically

### Edge Generation
- `generate_knn_edges(p_k, p_model_name)`: Generate k-NN semantic edges
- `upsert_relation(...)`: Upsert relation with weight averaging

### Query API Functions
#### Text Reconstruction
- `composition_text(p_id)`: Reconstruct text from composition DAG
- `text(p_id)`: Short alias for composition_text
- `find_composition(p_label)`: Find composition by label

#### Centroid Queries
- `get_centroid(p_label)`: Get 4D centroid for labeled composition
- `similar_by_centroid(p_label, p_k)`: Find similar by 4D centroid distance

#### Relation Queries
- `related_by_attention(p_label, p_k, p_min_weight)`: Find related via attention edges
- `related(p_label, p_k)`: Short alias for related_by_attention

#### Generative Walk
- `generative_walk(p_start_label, p_steps, p_temperature, p_seed)`: Deterministic walk through relation graph

#### Spatial Walk
- `spatial_walk(p_start_label, p_steps)`: Walk through 4D centroid space

#### Database Statistics
- `db_stats()`: Return counts of atoms, compositions, relations, models

### Generative Engine Functions
#### 4D Similarity Operations
- `hilbert_distance(p_lo_a, p_hi_a, p_lo_b, p_hi_b)`: Hilbert distance for pre-filtering

#### Semantic Similarity Search
- `similar_tokens(p_label, p_k)`: Find similar tokens by 4D centroid proximity
- `similar_tokens_fast(p_label, p_k, p_hilbert_range)`: Similar tokens with Hilbert pre-filtering

#### Prompt Encoding
- `encode_prompt(p_text)`: Tokenize text into composition IDs

#### Candidate Scoring
- `score_candidates(p_context_ids, p_k)`: Score candidate next tokens using centroid/PMI/attention

#### Generative Walk
- `generate_sql(p_prompt, p_max_tokens, p_temperature, p_top_k)`: Generate text completion
- `complete(p_prompt, p_max_tokens)`: Generate text completion as single string

#### Semantic Operations
- `vector_analogy(p_a, p_b, p_c, p_k)`: Semantic analogy using vector arithmetic
- `semantic_midpoint(p_a, p_b, p_k)`: Find tokens between A and B in 4D space

#### Generative Statistics
- `gen_db_stats()`: Statistics on atoms, compositions, relations, edges

### Bigram Statistics Functions
#### Bigram Tables
- `bigram_stats`: left_id, right_id, count, pmi (bigram counts with PMI scores)
- `unigram_stats`: token_id, count (unigram counts)
- `token_corpus_stats`: total_tokens, total_bigrams (corpus statistics)

#### Update Functions
- `increment_bigram(p_left_id, p_right_id, p_count)`: Increment bigram count
- `increment_unigram(p_token_id, p_count)`: Increment unigram count
- `compute_pmi_scores()`: Compute PMI scores for all bigrams

#### Query Functions
- `get_bigram_pmi(p_left_id, p_right_id)`: Get PMI score for bigram
- `get_bigram_count(p_left_id, p_right_id)`: Get raw count for bigram
- `top_continuations_pmi(p_left_id, k)`: Top continuations by PMI

#### Extraction Functions
- `extract_bigrams_from_compositions()`: Extract bigrams from composition_child relationships

### Q&A Search Functions
#### Text Search
- `search_text(query_text, result_limit)`: Core text search across compositions by keyword pattern

#### Q&A Functions
- `ask(question)`: Main Q&A function - takes a question, returns answer with evidence
- `ask_exact(phrase, result_limit)`: Search for exact phrase matches

#### Entity Extraction
- `find_entities(search_pattern)`: Extract named entities (capitalized phrases)

### Model Registry Functions
#### Model Table
- `model`: id, name, path, model_type, tokenizer_type, vocab_size, hidden_size, num_layers, num_experts, is_multimodal, vocab_ingested, edges_extracted, last_scanned

#### Model Functions
- `upsert_model(...)`: Register or update a model in the registry
- `model_stats` view: Statistics view for registered models

## Scripts
- **Setup Scripts**: Build, database setup, data ingestion
- **Platform-specific**: Windows (PowerShell), Linux/macOS (Bash)
- **Validation**: System-wide validation and testing
- **Ingest Tools**: Model ingestion, data seeding

## Testing
- **Unit Tests**: GTest for core components (coordinates, hilbert, ops)
- **Integration Tests**: Full system validation
- **SQL Tests**: Database function testing

## Documentation
- **ARCHITECTURE.md**: Technical specification
- **SEMANTIC_WEB_DESIGN.md**: Semantic design principles
- **OPTIMIZATION_PLAN.md**: Performance optimization strategies
- **AUDIT_REPORT.md**: Code audit findings and fixes

This catalog covers the complete functionality of the Hartonomous-Opus repository, from low-level coordinate mapping and hashing to high-level generative AI capabilities, all built around the 4D hypercube semantic substrate.

## Code Objects Catalog

This section provides an extensive audit and catalog of all classes, structs, enumerators, interfaces, and other code objects in the repository. Each object is documented with its namespace, file location, members, methods, inheritance hierarchy, and detailed description.

### Namespace hypercube

#### Types Module (cpp/include/hypercube/types.hpp)

**struct Point4D**  
- **Members**: 
  - `Coord32 x, y, z, m;` (uint32 coordinates)
- **Methods**:
  - `constexpr Point4D() noexcept` (default constructor)
  - `constexpr Point4D(Coord32 x_, Coord32 y_, Coord32 z_, Coord32 m_) noexcept` (constructor)
  - `constexpr double x_normalized() const noexcept` (deprecated)
  - `constexpr double y_normalized() const noexcept` (deprecated)
  - `constexpr double z_normalized() const noexcept` (deprecated)
  - `constexpr double m_normalized() const noexcept` (deprecated)
  - `constexpr double x_raw() const noexcept` (raw double for PostGIS)
  - `constexpr double y_raw() const noexcept`
  - `constexpr double z_raw() const noexcept`
  - `constexpr double m_raw() const noexcept`
  - `constexpr bool is_on_surface() const noexcept` (check if point is on 3-sphere surface)
- **Description**: Represents a 4D point with 32-bit coordinates per dimension. Coordinates are uint32 with center at 2^31 for semantic directions. Includes operators for comparison and methods for surface checking and PostGIS compatibility.

**struct HilbertIndex**  
- **Members**:
  - `uint64_t lo;` (lower 64 bits)
  - `uint64_t hi;` (upper 64 bits)
- **Methods**:
  - `constexpr HilbertIndex() noexcept` (default constructor)
  - `constexpr HilbertIndex(uint64_t lo_, uint64_t hi_) noexcept` (constructor)
  - Operators: `==`, `!=`, `<`, `<=`, `>`, `>=`
  - `constexpr HilbertIndex& operator++() noexcept` (increment)
  - `constexpr HilbertIndex operator-(const HilbertIndex& other) const noexcept` (subtraction)
  - `constexpr HilbertIndex operator+(const HilbertIndex& other) const noexcept` (addition)
  - `static HilbertIndex abs_distance(const HilbertIndex& a, const HilbertIndex& b) noexcept`
- **Description**: 128-bit Hilbert curve index split into two 64-bit parts for big-endian comparison and arithmetic operations.

**struct Blake3Hash**  
- **Members**:
  - `std::array<uint8_t, 32> bytes;`
- **Methods**:
  - `constexpr Blake3Hash() noexcept` (default constructor)
  - `explicit Blake3Hash(const uint8_t* data) noexcept` (from raw bytes)
  - `explicit Blake3Hash(const uint8_t* data, size_t size) noexcept`
  - Operators: `==`, `!=`, `<`, `<=`, `>`, `>=`
  - `std::string to_hex() const` (hex string representation)
  - `static Blake3Hash from_hex(std::string_view hex)` (parse hex string)
  - `constexpr const uint8_t* data() const noexcept` (raw bytes access)
  - `constexpr uint8_t* data() noexcept`
  - `static constexpr size_t size() noexcept` (returns 32)
  - `constexpr bool is_zero() const noexcept` (check if uninitialized)
- **Description**: 32-byte BLAKE3 hash with hex conversion and comparison operators.

**struct Blake3HashHasher**  
- **Methods**:
  - `size_t operator()(const Blake3Hash& h) const noexcept` (hash function using first 8 bytes)
- **Description**: Hash functor for using Blake3Hash in unordered_map.

**enum class AtomCategory : uint8_t**  
- **Values**:
  - Control, Format, PrivateUse, Surrogate, Noncharacter, Space, PunctuationOpen, PunctuationClose, PunctuationOther, Digit, NumberLetter, MathSymbol, Currency, Modifier, LetterUpper, LetterLower, LetterTitlecase, LetterModifier, LetterOther, MarkNonspacing, MarkSpacing, MarkEnclosing, SymbolOther, Separator, COUNT
- **Description**: Unicode codepoint categories for semantic clustering.

**constexpr const char* category_to_string(AtomCategory cat) noexcept**  
- **Description**: Converts AtomCategory enum to SQL string representation.

**struct UnicodeAtom**  
- **Members**:
  - `uint32_t codepoint;`
  - `AtomCategory category;`
  - `Point4D coords;`
  - `HilbertIndex hilbert;`
  - `Blake3Hash hash;`
- **Description**: Full metadata for a Unicode codepoint including coordinates and hash.

**struct Composition**  
- **Members**:
  - `Blake3Hash hash;`
  - `Point4D centroid;`
  - `HilbertIndex hilbert;`
  - `uint32_t depth;`
  - `uint32_t child_count;`
  - `uint64_t atom_count;`
- **Description**: Merkle DAG node representing a composition of atoms.

**struct SemanticEdge**  
- **Members**:
  - `Blake3Hash source;`
  - `Blake3Hash target;`
  - `double weight;`
- **Methods**:
  - `SemanticEdge() = default`
  - `SemanticEdge(const Blake3Hash& src, const Blake3Hash& tgt, double w)`
- **Description**: Weighted relationship between nodes for embeddings, co-occurrence, etc.

**namespace constants**  
- **Constants**:
  - `DIMENSIONS = 4`, `BITS_PER_DIM = 32`, `TOTAL_BITS = 128`
  - `COORD_ORIGIN = UINT32_MAX / 2` (center of coordinate space)
  - `COORD_RADIUS = COORD_ORIGIN - 1`
  - Unicode ranges: `MAX_CODEPOINT = 0x10FFFF`, `BMP_END = 0xFFFF`, etc.
  - Surface coordinates: `SURFACE_MIN/MAX = 0/UINT32_MAX`
  - `VALID_CODEPOINTS = MAX_CODEPOINT + 1 - (SURROGATE_END - SURROGATE_START + 1)`
  - Semantic bounds: `SEMANTIC_MIN/MAX/CENTER = -1.0/1.0/0.0`

**struct Point4F**  
- **Members**:
  - `double x, y, z, m;`
- **Methods**:
  - `constexpr Point4F() noexcept` (default constructor)
  - `constexpr Point4F(double x_, double y_, double z_, double m_) noexcept` (constructor)
  - `explicit Point4F(const Point4D& p) noexcept` (convert from quantized)
  - `Point4D to_quantized() const noexcept` (convert to quantized)
  - `Point4F normalized() const noexcept` (normalize to unit sphere)
  - `double dot(const Point4F& other) const noexcept` (dot product)
  - `double distance(const Point4F& other) const noexcept` (Euclidean distance)
  - `double geodesic_distance(const Point4F& other) const noexcept` (arccos of dot product)
  - `Point4F operator+(const Point4F& other) const noexcept`
  - `Point4F operator*(double s) const noexcept` (scalar multiplication)
- **Description**: Floating-point 4D point for optimization and calculations on the unit sphere.

#### Ops Module (cpp/include/hypercube/ops.hpp)

**struct AtomData**  
- **Members**:
  - `Blake3Hash id;`
  - `std::vector<Blake3Hash> children;`
  - `std::vector<uint8_t> value;` (UTF-8 bytes for leaves)
  - `int32_t depth = 0;`
  - `int64_t atom_count = 1;`
  - `int32_t codepoint = -1;` (Unicode codepoint for leaves)
  - `double centroid[4] = {0};` (X, Y, Z, M)
  - `HilbertIndex hilbert;`
  - `bool is_leaf = false;`
- **Description**: In-memory representation of atoms with cached metadata.

**struct SemanticEdge**  
- **Members**:
  - `Blake3Hash from;`
  - `Blake3Hash to;`
  - `double weight;`
- **Description**: Weighted semantic relationship between atoms.

**struct DistanceResult**  
- **Members**:
  - `size_t index;`
  - `double distance;`
- **Description**: Result of distance calculation with index and distance value.

**class ThreadPool**  
- **Methods**:
  - `explicit ThreadPool(size_t num_threads = 0)` (0 = hardware concurrency)
  - `~ThreadPool()`
  - `template<typename F, typename... Args> auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>` (submit work)
  - `template<typename Iter, typename Func> void parallel_for(Iter begin, Iter end, Func&& func)` (parallel for loop)
  - `void parallel_for_index(size_t start, size_t end, std::function<void(size_t)> func)` (parallel for with index)
  - `size_t size() const`
- **Description**: Thread pool for parallel processing with work stealing.

**ThreadPool& get_thread_pool()**  
- **Description**: Global thread pool singleton.

**class AtomCache**  
- **Methods**:
  - `void load_by_depth_range(int min_depth, int max_depth)` (load atoms by depth)
  - `void load_by_ids(std::span<const Blake3Hash> ids)` (load specific atoms)
  - `void load_all_leaves()` (load all leaf atoms)
  - `void load_semantic_edges(size_t limit = 100000)` (load semantic edges)
  - `const AtomData* get(const Blake3Hash& id) const` (lookup atom)
  - `bool contains(const Blake3Hash& id) const`
  - `size_t size() const`
  - `std::string reconstruct_text(const Blake3Hash& root) const` (reconstruct text from DAG)
  - `std::vector<Blake3Hash> get_descendants(const Blake3Hash& root, int max_depth = 100) const`
  - `std::vector<std::pair<Blake3Hash, double>> get_neighbors(const Blake3Hash& id) const` (semantic neighbors)
  - `std::vector<Blake3Hash> shortest_path(const Blake3Hash& from, const Blake3Hash& to, int max_depth = 6) const`
  - `std::vector<std::pair<Blake3Hash, double>> random_walk(const Blake3Hash& seed, int steps = 10) const`
  - `std::vector<std::string> batch_reconstruct(std::span<const Blake3Hash> ids) const`
  - `std::vector<DistanceResult> knn_by_centroid(const Blake3Hash& target, size_t k) const`
  - `std::vector<DistanceResult> knn_by_hilbert(const Blake3Hash& target, size_t k) const`
  - `void clear()`
  - Iterators: `iterator`, `const_iterator`, `begin()`, `end()`
  - `std::vector<std::pair<HilbertIndex, Blake3Hash>> get_sorted_hilbert_indices() const`
- **Description**: In-memory graph cache of atoms with graph algorithms and batch operations.

**struct HilbertPartition**  
- **Members**:
  - `HilbertIndex lo;`
  - `HilbertIndex hi;`
  - `size_t count;`
- **Description**: Partition of atoms for parallel processing using Hilbert ranges.

**std::vector<HilbertPartition> partition_by_hilbert(const AtomCache& cache, size_t num_partitions)**  
- **Description**: Partition atoms into Hilbert ranges for locality-preserving parallel processing.

**template<typename Func> void process_partitions(const std::vector<HilbertPartition>& partitions, Func&& func)**  
- **Description**: Process partitions in parallel with work stealing.

**Blake3Hash compute_content_hash(std::string_view text, const AtomCache& cache)**  
- **Description**: Compute content hash for text using CPE cascade.

**std::vector<Blake3Hash> batch_content_hash(std::span<const std::string_view> texts, const AtomCache& cache)**  
- **Description**: Batch compute content hashes with threading.

**double frechet_distance(std::span<const double> traj1, std::span<const double> traj2)**  
- **Description**: Discrete Fréchet distance between trajectories using SIMD.

**struct FrechetResult**  
- **Members**:
  - `Blake3Hash id;`
  - `double distance;`
- **Description**: Result of Fréchet distance calculation.

**std::vector<FrechetResult> batch_frechet_knn(const Blake3Hash& query, size_t k, const AtomCache& cache)**  
- **Description**: Find k most similar trajectories by Fréchet distance.

**double semantic_jaccard(const Blake3Hash& a, const Blake3Hash& b, const AtomCache& cache)**  
- **Description**: Jaccard similarity based on shared descendants.

**std::vector<std::pair<Blake3Hash, double>> batch_jaccard(const Blake3Hash& target, std::span<const Blake3Hash> candidates, const AtomCache& cache)**  
- **Description**: Batch Jaccard similarity calculations.

**std::vector<DistanceResult> analogy_knn(const Blake3Hash& a, const Blake3Hash& b, const Blake3Hash& c, size_t k, const AtomCache& cache)**  
- **Description**: Vector arithmetic analogy (A:B :: C:D) using centroid arithmetic.

#### Generative Module (cpp/include/hypercube/generative.hpp)

**using Blake3Hash = std::array<uint8_t, 32>**  
- **Description**: Type alias for 32-byte BLAKE3 hash.

**struct Blake3Hasher**  
- **Methods**:
  - `size_t operator()(const Blake3Hash& h) const noexcept` (hash using first 8 bytes)
- **Description**: Hash functor for unordered_map with Blake3Hash keys.

**struct Blake3Equal**  
- **Methods**:
  - `bool operator()(const Blake3Hash& a, const Blake3Hash& b) const noexcept`
- **Description**: Equality functor for Blake3Hash.

**struct Centroid4D**  
- **Members**:
  - `double x = 0.0, y = 0.0, z = 0.0, m = 0.0;`
- **Methods**:
  - `bool valid() const` (check if coordinates are set)
  - `bool has_coordinates() const` (alias for valid)
  - `double x_norm() const` (normalized coordinate 0-1)
  - `double y_norm() const`
  - `double z_norm() const`
  - `double m_norm() const`
  - `double distance(const Centroid4D& other) const` (Euclidean distance in normalized space)
  - `double similarity(const Centroid4D& other) const` (inverse distance similarity)
- **Description**: 4D centroid coordinates from Laplacian eigenmap projection, stored as doubles mapping to uint32 range.

**struct TokenCandidate**  
- **Members**:
  - `Blake3Hash id;`
  - `std::string label;`
  - `Centroid4D centroid;`
  - `double hilbert_index;`
  - `double frequency;`
- **Description**: Candidate token with metadata for generation.

**struct TokenState**  
- **Members**:
  - `Blake3Hash id;`
  - `std::string label;`
  - `Centroid4D centroid;`
  - `double hilbert_index;`
- **Description**: Current token state during generation.

**struct ScoredCandidate**  
- **Members**:
  - `size_t index;`
  - `double score_centroid;`
  - `double score_pmi;`
  - `double score_attn;`
  - `double score_global;`
  - `double score_total;`
- **Description**: Scored candidate with individual component scores.

**struct GenerationConfig**  
- **Members**:
  - `double w_centroid = 0.4;` (centroid weight)
  - `double w_pmi = 0.3;`
  - `double w_attn = 0.2;`
  - `double w_global = 0.1;`
  - `bool greedy = true;`
  - `double temperature = 1.0;`
  - `size_t max_candidates = 500;`
  - `double hilbert_range = 0.1;`
  - `size_t max_tokens = 50;`
  - `std::vector<std::string> stop_tokens = {".", "!", "?", "\n"};`
- **Description**: Configuration for generative engine scoring and selection.

**struct VocabEntry**  
- **Members**:
  - `Blake3Hash id;`
  - `std::string label;`
  - `int depth;`
  - `double frequency;`
  - `double hilbert_index;`
  - `Centroid4D centroid;`
- **Description**: Vocabulary entry with all metadata.

**class VocabularyCache**  
- **Members**:
  - `std::vector<VocabEntry> entries;`
  - `std::unordered_map<std::string, size_t> label_to_index;`
  - `std::unordered_map<Blake3Hash, size_t, Blake3Hasher, Blake3Equal> id_to_index;`
- **Methods**:
  - `void clear()`
  - `void add_entry(const VocabEntry& entry)`
  - `void set_centroid(size_t idx, double x, double y, double z, double m)`
  - `int64_t find_label(const std::string& label) const`
  - `const VocabEntry* get_entry(size_t idx) const`
  - `size_t count_with_centroid() const`
- **Description**: Cache for vocabulary entries with centroid coordinates.

**struct BigramKey**  
- **Members**:
  - `Blake3Hash left;`
  - `Blake3Hash right;`
- **Methods**:
  - `bool operator==(const BigramKey& other) const`
- **Description**: Key for bigram PMI cache.

**struct BigramKeyHash**  
- **Methods**:
  - `size_t operator()(const BigramKey& k) const` (FNV-1a hash)
- **Description**: Hash functor for BigramKey.

**class BigramCache**  
- **Members**:
  - `std::unordered_map<BigramKey, double, BigramKeyHash> pmi_scores;`
  - `double max_pmi = 1.0;`
- **Methods**:
  - `void clear()`
  - `void add(const Blake3Hash& left, const Blake3Hash& right, double score)`
  - `double get(const Blake3Hash& left, const Blake3Hash& right) const`
- **Description**: Cache for PMI scores between token pairs.

**using Blake3DoubleMap = std::unordered_map<Blake3Hash, double, Blake3Hasher, Blake3Equal>**  
- **Description**: Type alias for attention edge map.

**class AttentionCache**  
- **Members**:
  - `std::unordered_map<Blake3Hash, Blake3DoubleMap, Blake3Hasher, Blake3Equal> edges;`
- **Methods**:
  - `void clear()`
  - `void add(const Blake3Hash& source, const Blake3Hash& target, double weight)`
  - `double get(const Blake3Hash& source, const Blake3Hash& target) const`
- **Description**: Cache for attention weights between tokens.

**class GenerativeEngine**  
- **Members**:
  - `VocabularyCache vocab;`
  - `BigramCache bigrams;`
  - `AttentionCache attention;`
  - `GenerationConfig config;`
  - `std::mt19937 rng{42};`
- **Methods**:
  - `void seed(uint32_t s)` (set RNG seed)
  - `void seed_from_text(const std::string& text)` (deterministic seed from text)
  - `void clear_all()` (clear all caches)
  - Scoring methods: `score_centroid()`, `score_pmi()`, `score_attn()`, `score_global()`, `score_candidate()`
  - `std::vector<size_t> get_candidates_by_hilbert(const TokenState& current) const` (Hilbert pre-filtering)
  - `std::vector<size_t> get_all_vocab_candidates() const`
  - `size_t select_next_token(const std::vector<ScoredCandidate>& scored)` (greedy or stochastic)
  - `TokenState make_token_state(size_t idx) const`
  - `std::vector<std::string> generate(const std::string& start_label, size_t max_tokens)`
  - `std::vector<ScoredCandidate> find_similar(const std::string& label, size_t k)` (find similar tokens)
- **Description**: LLM-like generative engine using 4D centroid proximity, PMI, and attention scoring.

**GenerativeEngine& get_engine()**
- **Description**: Global generative engine singleton.

#### Hilbert Module (cpp/include/hypercube/hilbert.hpp)

**class HilbertCurve**
- **Constants**:
  - `static constexpr uint32_t DIMS = 4;`
  - `static constexpr uint32_t BITS = 32;`
- **Methods**:
  - `static HilbertIndex coords_to_index(const Point4D& point) noexcept` (4D coords to 128-bit Hilbert index)
  - `static Point4D index_to_coords(const HilbertIndex& index) noexcept` (128-bit Hilbert index to 4D coords)
  - `static HilbertIndex distance(const HilbertIndex& a, const HilbertIndex& b) noexcept` (compute Hilbert distance)
  - `static bool in_range(const HilbertIndex& center, const HilbertIndex& point, const HilbertIndex& range) noexcept` (check if point in range)
  - `static Point4D index_to_raw_coords(const HilbertIndex& index) noexcept` (raw coords without CENTER adjustment)
  - `static void transpose_to_axes(uint32_t* x, uint32_t n, uint32_t bits) noexcept` (transpose for Hilbert curve)
  - `static void axes_to_transpose(uint32_t* x, uint32_t n, uint32_t bits) noexcept` (inverse transpose)
- **Description**: 4D Hilbert curve implementation for spatial indexing, mapping 4D coordinates to 128-bit Hilbert indices with perfect locality preservation.

#### Coordinates Module (cpp/include/hypercube/coordinates.hpp)

**struct CodepointMapping**
- **Members**:
  - `Point4D coords;`
  - `HilbertIndex hilbert;`
- **Description**: Combined result of coordinate mapping with both 4D coordinates and Hilbert index.

**class CoordinateMapper**
- **Methods**:
  - `static Point4D map_codepoint(uint32_t codepoint) noexcept` (map codepoint to 4D surface coords)
  - `static CodepointMapping map_codepoint_full(uint32_t codepoint) noexcept` (map to coords + Hilbert)
  - `static Point4F map_codepoint_float(uint32_t codepoint) noexcept` (map to floating-point coords)
  - `static AtomCategory categorize(uint32_t codepoint) noexcept` (get Unicode category)
  - `static Point4D centroid(const std::vector<Point4D>& points) noexcept` (compute centroid)
  - `static Point4D weighted_centroid(const std::vector<Point4D>& points, const std::vector<double>& weights) noexcept` (weighted centroid)
  - `static double euclidean_distance(const Point4D& a, const Point4D& b) noexcept` (4D Euclidean distance)
  - `static uint32_t get_category_count(AtomCategory cat) noexcept` (count codepoints in category)
- **Struct Diagnostics**:
  - Members: `chordal_nn_mean/median/std/cv/5th/95th`, `geodesic_nn_mean/median/std/cv/5th/95th`, `local_density_mean/std/cv`, `collision_counts`, `bucket_cv`
- **Optimization Methods**:
  - `static Diagnostics compute_diagnostics(const std::map<uint32_t, Point4F>& points)` (comprehensive metrics)
  - `static void apply_deterministic_jitter(std::map<uint32_t, Point4F>& points, double epsilon = 1e-7)` (break collisions)
  - `static void bucketed_tangent_lloyd(std::map<uint32_t, Point4F>& points, size_t k = 32, double alpha = 0.25, int iterations = 4)` (local optimization)
  - `static void global_knn_repulsion(std::map<uint32_t, Point4F>& points, size_t k = 64, double s = 1.0, double eta = 0.001, int iterations = 10)` (global repulsion)
  - `static bool optimize_distribution(std::map<uint32_t, Point4F>& points)` (complete pipeline)
- **Description**: Hopf fibration mapper using golden angle spiral for equidistant 3-sphere distribution of Unicode codepoints, with comprehensive optimization pipeline.

#### Blake3 Module (cpp/include/hypercube/blake3.hpp)

**class Blake3Hasher**
- **Methods**:
  - `static Blake3Hash hash(std::span<const uint8_t> data) noexcept` (hash arbitrary data)
  - `static Blake3Hash hash(std::string_view str) noexcept` (hash string)
  - `static Blake3Hash hash_codepoint(uint32_t codepoint) noexcept` (hash UTF-8 encoding)
  - `static Blake3Hash hash_children(std::span<const Blake3Hash> children) noexcept` (hash ordered children)
  - `static Blake3Hash hash_children_ordered(std::span<const Blake3Hash> children) noexcept` (position-sensitive children hash)
  - `static Blake3Hash keyed_hash(std::span<const uint8_t> key, std::span<const uint8_t> data) noexcept` (keyed hashing)
  - `static Blake3Hash derive_key(std::string_view context, std::span<const uint8_t> key_material) noexcept` (key derivation)
  - `static std::vector<uint8_t> encode_utf8(uint32_t codepoint) noexcept` (UTF-8 encoding)
- **Nested class Incremental**:
  - Methods: `Incremental()`, `~Incremental()`, `update(std::span<const uint8_t>)`, `update(std::string_view)`, `Blake3Hash finalize()`, `reset()`
- **Description**: BLAKE3 cryptographic hashing for content addressing atoms and Merkle DAG compositions, with incremental and keyed variants.

#### Backend Module (cpp/include/hypercube/backend.hpp)

**enum class SIMDLevel : int**
- **Values**: Scalar = 0, SSE2 = 1, SSE4_2 = 2, AVX = 3, AVX2 = 4, AVX512 = 5
- **Functions**: `const char* simd_level_name(SIMDLevel)`, `int simd_width(SIMDLevel)`

**enum class EigensolverBackend : int**
- **Values**: Jacobi = 0, Eigen = 1, MKL = 2
- **Functions**: `const char* eigensolver_name(EigensolverBackend)`

**enum class KNNBackend : int**
- **Values**: BruteForce = 0, HNSWLIB = 1, FAISS = 2
- **Functions**: `const char* knn_backend_name(KNNBackend)`

**struct CPUInfo**
- **Members**: `vendor`, `brand`, `family/model/stepping`, `cores_physical/logical`, `has_sse2/sse4_2/avx/avx2/fma/avx512f/dq/bw/vl`

**struct BackendInfo**
- **Members**: `cpu` (CPUInfo), `simd_level`, `has_mkl/eigen/hnswlib/faiss`, `eigensolver/knn`, `compiler`, `build_type`
- **Methods**: `std::string summary() const` (detailed configuration summary)

**class Backend**
- **Methods**:
  - `static BackendInfo detect()` (runtime backend detection)
  - `static const BackendInfo& info()` (cached singleton)
  - Convenience accessors: `simd_level()`, `eigensolver()`, `knn()`, `has_avx512()`, `has_avx2()`, `has_mkl()`, `has_eigen()`
- **Description**: Modular backend detection for SIMD, eigensolvers, and k-NN libraries with automatic selection of optimal implementations.

#### Atom Calculator Module (cpp/include/hypercube/atom_calculator.hpp)

**struct AtomRecord**
- **Members**: `codepoint`, `hash`, `coords`, `hilbert`, `depth`, `atom_count`

**struct CompositionRecord**
- **Members**: `hash`, `centroid`, `hilbert`, `children`, `child_coords`, `child_depths`, `depth`, `atom_count`

**struct ChildInfo** (nested in AtomCalculator)
- **Members**: `hash`, `coords`, `depth`, `atom_count`

**class AtomCalculator**
- **Methods**:
  - `static AtomRecord compute_atom(uint32_t codepoint) noexcept` (compute all atom properties)
  - `static CompositionRecord compute_composition(const std::vector<uint32_t>& codepoints) noexcept` (from codepoint sequence)
  - `static CompositionRecord compute_composition(const std::vector<ChildInfo>& children) noexcept` (from child records)
  - `static std::vector<uint32_t> decode_utf8(const std::string& text) noexcept` (UTF-8 to codepoints)
  - `static CompositionRecord compute_vocab_token(const std::string& token_text) noexcept` (convenience method)
- **Description**: Zero-roundtrip deterministic computation of all atom and composition properties from input data without database access.

#### DB Connection Module (cpp/include/hypercube/db/connection.hpp)

**namespace hypercube::db**

**struct ConnectionConfig**
- **Members**: `dbname`, `host`, `port`, `user`, `password`
- **Methods**:
  - `ConnectionConfig()` (reads from HC_DB_* env vars)
  - `std::string to_conninfo() const` (build libpq connection string)
  - `bool parse_arg(int argc, char** argv, int& i)` (parse command line args)

**class Connection**
- **Methods**:
  - `Connection()` (default)
  - `explicit Connection(const std::string& conninfo)` (from connection string)
  - `explicit Connection(const ConnectionConfig& config)` (from config)
  - `~Connection()` (RAII cleanup)
  - Move constructors/assignment, no copy
  - `PGconn* get() const` (underlying connection)
  - `bool ok() const` (check status)
  - `const char* error() const` (error message)

**class ConnectionPool**
- **Members**: `conninfo_`, `max_size_`, `active_connections_`
- **Methods**:
  - `ConnectionPool(const ConnectionConfig& config, size_t max_size = 10)`
  - `std::unique_ptr<Connection> acquire()` (get connection from pool)
  - `void release(std::unique_ptr<Connection> conn)` (return to pool)
  - `std::tuple<size_t, size_t> stats() const` (pool statistics)
  - `void drain()` (shutdown cleanup)

**class PooledConnection**
- **Members**: `pool_`, `conn_`
- **Methods**:
  - `PooledConnection(ConnectionPool& pool, std::unique_ptr<Connection> conn)`
  - `~PooledConnection()` (auto-release to pool)
  - No copy/move - managed by pool
  - Accessors: `Connection* operator->()`, `Connection& operator*()`, `PGconn* get()`

**bool exec_ok(PGconn* conn, const char* query, PGresult** out = nullptr)**
- **Description**: Execute query and check success status.

### SQL Objects Catalog

This section catalogs SQL-defined types, functions, tables, and views found in the repository's SQL files.

#### Types (CREATE TYPE)
- **hilbert_index** (cpp/sql/hypercube--1.0.sql): Composite type with hi (bigint), lo (bigint)
- **gen_similar_result** (cpp/sql/generative--1.0.sql): Composite with label (TEXT), score (DOUBLE PRECISION)
- **gen_candidate_result** (cpp/sql/generative--1.0.sql): Composite with label (TEXT), score_centroid/score_pmi/score_attn/score_global/score_total (DOUBLE PRECISION)
- **gen_stats_result** (cpp/sql/generative--1.0.sql): Composite with key (TEXT), value (TEXT)
- **similar_result** (cpp/sql/embedding_ops--1.0.sql): Composite with label (TEXT), score (DOUBLE PRECISION)

#### Tables (CREATE TABLE)
- **atom** (sql/001_schema.sql): Primary table for Unicode atoms with id (BYTEA PK), codepoint (INTEGER), value (BYTEA), geom (GEOMETRY), hilbert_lo/hi (NUMERIC), created_at (TIMESTAMPTZ)
- **composition** (sql/001_schema.sql): Compositions with id (BYTEA PK), label (TEXT), depth/depth (INTEGER), child_count (INTEGER), atom_count (BIGINT), geom/centroid (GEOMETRY), hilbert_lo/hi (NUMERIC), created_at (TIMESTAMPTZ)
- **composition_child** (sql/001_schema.sql): Junction table with composition_id/child_id (BYTEA), ordinal (SMALLINT), child_type (CHAR)
- **relation** (sql/001_schema.sql): Semantic edges with id (BIGSERIAL PK), source_type/target_type (CHAR), source_id/target_id (BYTEA), relation_type (CHAR), weight (REAL), source_model (TEXT), source_count/layer (INTEGER), component (TEXT), created_at (TIMESTAMPTZ)
- **bigram_stats** (sql/005_bigram_stats.sql): Bigram counts with left_id/right_id (BYTEA), count (BIGINT), pmi (DOUBLE PRECISION)
- **unigram_stats** (sql/005_bigram_stats.sql): Unigram counts with token_id (BYTEA PK), count (BIGINT)
- **token_corpus_stats** (sql/005_bigram_stats.sql): Corpus stats with id (INTEGER PK), total_tokens/total_bigrams (BIGINT)
- **model** (sql/007_model_registry.sql): Model registry with id (BIGSERIAL PK), name/path (TEXT), model_type/tokenizer_type (TEXT), vocab_size/hidden_size/num_layers/num_experts (BIGINT), is_multimodal (BOOLEAN), vocab_ingested/edges_extracted (BOOLEAN), last_scanned (TIMESTAMPTZ)
- **test_results** (tests/sql/test_enterprise_suite.sql): Test results table

#### Views (CREATE VIEW)
- **atom_stats_view** (sql/002_core_functions.sql): View wrapping atom_stats() function
- **atom_type_stats** (sql/002_core_functions.sql): Statistics view
- **model_stats** (sql/007_model_registry.sql): Model statistics view

#### Functions (CREATE FUNCTION)
The repository contains over 50 SQL functions across multiple files. Key categories include:
- **Core Atom Functions**: atom_is_leaf, atom_centroid, atom_children, atom_text, get_atoms_by_codepoints
- **Spatial Functions**: atom_knn, atom_hilbert_range, centroid_distance, centroid_similarity
- **Semantic Functions**: semantic_neighbors, attention, analogy
- **Text Reconstruction**: atom_reconstruct_text, composition_text, text
- **Generative Functions**: gen_load_vocab, gen_similar, gen_walk, gen_complete, embedding_similar, embedding_analogy
- **Statistics Functions**: atom_stats, db_stats, gen_stats
- **Bigram Functions**: increment_bigram, get_bigram_pmi, top_continuations_pmi
- **Model Registry**: upsert_model
- **Q&A Functions**: ask, find_entities
- **Embedding Operations**: embedding_cosine_sim, embedding_l2_dist, embedding_vector_add, embedding_analogy_vec

### Python Objects Catalog

No class definitions found in Python files. The Python script (scripts/inspect_safetensor.py) contains utility functions for SafeTensor inspection but no object-oriented structures.

### Summary

This audit has cataloged:
- **C++ Objects**: 126+ structs, classes, and enums across header files in the hypercube namespace
- **SQL Objects**: 40+ CREATE statements defining types, tables, views, and functions
- **Python Objects**: No classes, only functions

The catalog provides extensive detail including members, methods, inheritance, and descriptions for all major code objects in the repository, organized by namespace and module.