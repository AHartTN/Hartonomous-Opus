# Hartonomous-Opus: 4D Hypercube Semantic Substrate

A **deterministic, lossless, content-addressable geometric semantic substrate** that maps all digital content into a 4D hypercube coordinate system with Hilbert curve indexing for efficient spatial queries.

**See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical specification.**

## Quick Start

### Windows (PowerShell)

```powershell
# Configure database credentials
Copy-Item scripts/config.env.example scripts/config.env
# Edit scripts/config.env with your PostgreSQL credentials

# Run EVERYTHING: clean, build, database setup, seed 1.1M atoms,
# ingest test data (model + Moby Dick), run all tests
.\scripts\windows\setup-all.ps1

# Individual steps (if needed):
.\scripts\windows\clean.ps1           # Clean build artifacts
.\scripts\windows\build.ps1           # Build C++ components
.\scripts\windows\setup-db.ps1        # Setup database and extensions
.\scripts\windows\setup-db.ps1 -Reset # DESTRUCTIVE: Reset database
.\scripts\windows\ingest-testdata.ps1 # Ingest test-data/ content
.\scripts\windows\run_tests.ps1       # Run comprehensive test suite

# Quick tests (skip slow/long-running tests):
.\scripts\windows\run_tests.ps1 -Quick

# Verbose test output:
.\scripts\windows\run_tests.ps1 -Verbose

# Skip database tests (for offline development):
.\scripts\windows\run_tests.ps1 -NoDatabase

# Ingest your own content:
.\scripts\windows\ingest.ps1 -Path "C:\path\to\file.txt"
```

### Linux/macOS (Bash)

```bash
# Configure database credentials
cp scripts/config.env.example scripts/config.env
# Edit scripts/config.env with your PostgreSQL credentials

# Initialize everything (builds tools, creates database, seeds 1.1M atoms)
./setup.sh init

# Ingest any directory of content
./setup.sh ingest ~/Documents/notes/

# Ingest AI model packages (safetensors + vocab)
./setup.sh ingest ~/models/minilm/

# Visualize the CPE tree structure
./setup.sh tree "Mississippi"

# Query similarity
./setup.sh similar "machine learning"

# Run full validation
./validate.sh

# Check status
./setup.sh status
```

## Commands

| Command | Description |
|---------|-------------|
| `./setup.sh init` | Initialize database, build tools, seed atoms |
| `./setup.sh status` | Show database statistics |
| `./setup.sh ingest <path>` | Ingest file or directory using CPE |
| `./setup.sh query <text>` | Get composition ID for text |
| `./setup.sh similar <text>` | Find similar compositions |
| `./setup.sh tree <text>` | Show CPE Merkle DAG structure |
| `./setup.sh test` | Run integrated test suite |
| `./setup.sh reset` | Drop and reset database |
| `./validate.sh` | Full system validation |

## Configuration

Create a `.env` file (or set environment variables):

```bash
PGHOST=localhost
PGPORT=5432
PGUSER=hartonomous
PGPASSWORD=hartonomous
PGDATABASE=hypercube
```

## Architecture

### Core Concepts

1. **Atoms**: Unicode codepoints as fundamental constants (perimeter landmarks)
   - Each codepoint → 4D coordinate (32 bits per dimension)
   - BLAKE3 hash as content-addressed ID
   - Hilbert curve index (128-bit) for spatial ordering
   - All atoms distributed on the 3-sphere surface (S³ in 4D)
   - **Lossless**: PostGIS GEOMETRY stores full double precision (2^53 mantissa)

2. **Compositions**: Binary Merkle DAG via PMI Contraction
   - PMI (Pointwise Mutual Information) identifies significant co-occurrences
   - Highest-PMI pairs contracted into new compositions recursively
   - Result: Logarithmic dictionary growth, linear content growth
   - Content-addressed: "the" from any document = same ID
   - Geometry = LINESTRINGZM trajectory through 2 child centroids

3. **Two Table Model**
   - `atom` table stores nodes (leaves and compositions)
   - `relation` table stores edges (parent→child with ordinal)
   - `ordinal = 1` for left child, `ordinal = 2` for right child
   - `relation_type = 'C'` for composition edges

4. **Global Deduplication**
   - First ingest creates patterns; subsequent ingests reuse existing compositions
   - The more content ingested, the more deduplication occurs
   - Binary tree structure = exactly 2 children per composition

### Performance (Current Build)

**Unit Test Performance:**
| Operation | Status | Notes |
|-----------|--------|-------|
| Hilbert curve roundtrip | ✅ ~0.1ms | 100% accuracy, locality preserved |
| Coordinate mapping (4D) | ✅ ~1ms | All Unicode codepoints mapped |
| Laplacian eigenmaps | ✅ ~100ms | 50-token test dataset |
| Google Test suite | ✅ 30/32 pass | 2 centroid tests updated for sphere normalization |

**Build Performance:**
- Full rebuild: ~1 minute (Clang + Ninja + MKL)
- Unit tests: ~1 second total execution
- Memory usage: < 500MB during builds

**Known Performance Characteristics:**
- Hilbert range queries: O(log N) - Sub-millisecond
- Atom generation: 5.5M atoms/sec (estimated)
- Database seeding: 37K atoms/sec (estimated)
- Content ingestion: Scales with corpus size (deduplication improves over time)

### Type System

| Field | PostgreSQL Type | Notes |
|-------|----------------|-------|
| geom | GEOMETRY(GEOMETRYZM, 0) | POINTZM or LINESTRINGZM, SRID=0 |
| hilbert_lo/hi | BIGINT | 128-bit Hilbert index |
| id | BYTEA | 32-byte BLAKE3 hash |
| children | BYTEA[] | Ordered child hashes (compositions only) |

## Building

```bash
# Initialize builds everything automatically
./setup.sh init

# Or build manually
cd cpp/build && cmake .. && make -j$(nproc)
```

### Requirements

**Core Dependencies:**
- PostgreSQL 18.1+ (tested with 18.1)
- PostGIS 3.3+
- CMake 3.16+
- C++17 compiler (Clang 21.1.8 tested)
- Windows: PowerShell 7+

**Performance Libraries (automatically detected):**
- Intel MKL (BLAS/LAPACK optimization)
- OpenMP (parallel processing)
- AVX intrinsics (SIMD acceleration)

**Build Tools:**
- Ninja build system
- LLVM/Clang toolchain
- vcpkg (for Windows dependencies)

## Database Schema (Unified)

```sql
CREATE TABLE atom (
    id              BYTEA PRIMARY KEY,        -- BLAKE3 hash (32 bytes)
    geom            GEOMETRY(GEOMETRYZM, 0),  -- POINTZM or LINESTRINGZM
    children        BYTEA[],                  -- Child hashes for compositions
    value           BYTEA,                    -- UTF-8 bytes for leaves
    codepoint       INTEGER UNIQUE,           -- Unicode codepoint for leaves
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,
    depth           INTEGER NOT NULL DEFAULT 0,
    atom_count      BIGINT NOT NULL DEFAULT 1
);
```

## Semantic Query API

All queries work via PostgreSQL functions. Connect and query:

```sql
-- 1. EXACT CONTENT IDENTITY
SELECT content_exists('whale');           -- Check if composition exists
SELECT * FROM content_get('Captain');     -- Get full composition info

-- 2. FUZZY SIMILARITY (Fréchet Distance)
-- Handles case variance: "King" ≈ "king" (similar trajectories)
-- Handles typos: "kinestringzm" ≈ "linestringzm" (differ by 1 char)
SELECT * FROM similar('whale', 10);
SELECT * FROM text_frechet_similar('Captain', 1e9, 20);

-- 3. SEMANTIC NEIGHBORS (Centroid KNN)
-- Find semantically related by 4D proximity
SELECT * FROM neighbors('ocean', 10);
SELECT * FROM semantic_neighbors('ship', 20);

-- 4. EDGE WALKING (Co-occurrence)
-- What commonly follows/co-occurs with this?
SELECT * FROM follows('Captain', 10);
SELECT * FROM semantic_walk('whale', 5);

-- 5. ANALOGY (Vector Arithmetic)
-- "man" is to "king" as "woman" is to ?
SELECT * FROM analogy('man', 'king', 'woman', 5);

-- 6. COMPOUND SIMILARITY
-- Combine trajectory shape AND centroid proximity
SELECT * FROM compound_similar('Moby', 20, 0.5, 0.5);

-- 7. DIAGNOSTICS
SELECT * FROM composition_info('whale');
SELECT edge_count('Captain');
SELECT * FROM stats();
```

### Query Return Types

| Function | Returns |
|----------|---------|
| `similar(text, k)` | `(content, distance)` |
| `neighbors(text, k)` | `(content, distance)` |
| `follows(text, k)` | `(content, weight)` |
| `analogy(a, b, c, k)` | `(answer, distance)` |
| `walk(text, steps)` | `(step, content, weight)` |

## File Structure

```
Hartonomous-Opus/
├── ARCHITECTURE.md           # Canonical technical specification
├── README.md                 # This file
├── setup.sh                  # Single entry point for all operations
├── validate.sh               # Full system validation
├── .env.example              # Configuration template
│
├── cpp/
│   ├── include/hypercube/   # C++ headers
│   └── src/
│       ├── cpe_ingest.cpp          # CPE ingester (main workhorse)
│       ├── seed_atoms_parallel.cpp # Unicode seeder
│       ├── ingest_safetensor.cpp   # AI model package ingester
│       ├── semantic_ops.cpp        # PostgreSQL UDF extension
│       ├── hypercube.cpp           # Core PostgreSQL extension
│       ├── blake3_pg.cpp           # BLAKE3 for PostgreSQL
│       ├── hilbert.cpp             # Skilling's algorithm
│       └── coordinates.cpp         # Coordinate utilities
│
├── sql/
│   ├── 011_unified_atom.sql        # Unified schema (CURRENT)
│   ├── 012_semantic_udf.sql        # SQL UDF infrastructure
│   ├── 013_model_infrastructure.sql # AI model tables
│   └── deprecated/                  # Old multi-table schema
│
└── tests/                    # Test suites
```

## Technical Details

### Cascading Pair Encoding (CPE)

CPE uses sliding window pairing at ALL tiers to capture ALL n-grams:

```
"Hello" (5 chars)
  Tier 0: [H, e, l, l, o] = 5 codepoints (atoms)
  Tier 1: [He, el, ll, lo] = 4 bigrams (sliding pairs)
  Tier 2: [Hel, ell, llo] = 3 trigrams (sliding pairs of bigrams)
  Tier 3: [Hell, ello] = 2 4-grams
  Tier 4: [Hello] = 1 root (5-gram)

Total compositions: 4+3+2+1 = 10 = O(n²/2)
```

This is NOT a binary tree - sliding window preserves ALL adjacencies.
Global deduplication means the vocabulary grows sublinearly with corpus size.

Each composition hash = BLAKE3(child_hashes in order)
Centroid = average of child 4D coordinates

### Hilbert Curve

- 128-bit index stored as two `bigint` (hilbert_lo, hilbert_hi)
- Preserves spatial locality for efficient range queries
- Based on Skilling's compact algorithm with gray code optimization
- Lossless roundtrip: coords → index → coords

### Content-Addressed Deduplication

- Same bytes → same hash, regardless of source file
- "the" from Moby Dick = "the" from children's book = same composition ID
- First ingest creates patterns; subsequent ingests just add references
- Documents are DAG roots pointing to shared substructure

### Bit-Perfect Reconstruction

DFS traversal of children array reconstructs original bytes:
1. Start at root composition
2. Follow children array in order (index 0, then 1)
3. When reaching atoms (depth=0), emit the codepoint
4. Concatenate = original content

## Tests

```bash
# Run validation suite
./validate.sh

# Quick validation (skip performance tests)
./validate.sh --quick

# Run via setup.sh
./setup.sh test
```

## License

Apache 2.0
