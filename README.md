# Hartonomous-Opus: 4D Hypercube Semantic Substrate

A **deterministic, lossless, content-addressable geometric semantic substrate** that maps all digital content into a 4D hypercube coordinate system with Hilbert curve indexing for efficient spatial queries.

**See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical specification.**

## Quick Start

```bash
# Configure database credentials
cp .env.example .env
# Edit .env with your PostgreSQL credentials

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

2. **Compositions**: Merkle DAG via Cascading Pair Encoding (CPE)
   - Binary tree structure - each composition has exactly 2 children
   - Characters cascade: `N chars → N/2 pairs → N/4 → ... → 1 root`
   - Total compositions ≈ 2N (geometric series, NOT O(n²))
   - Content-addressed deduplication: "the" from any document = same ID
   - Geometry = LINESTRINGZM trajectory through child centroids

3. **Unified Single Table Model**
   - `atom` table stores BOTH leaves (POINTZM) and compositions (LINESTRINGZM)
   - `children BYTEA[]` stores ordered child references for reconstruction
   - `depth = 0` for leaves, `depth > 0` for compositions
   - No separate `relation` or `relation_edge` tables

4. **Global Deduplication**
   - First ingest creates patterns; subsequent ingests reuse existing compositions
   - The more content ingested, the more deduplication occurs
   - ~83% deduplication on typical text corpora

### Performance

| Operation | Time | Rate |
|-----------|------|------|
| Atom generation (1.1M) | ~200ms | 5.5M atoms/sec |
| Full database seeding | ~30s | 37K atoms/sec |
| CPE file processing | - | ~1 MB/s |
| CPE DB insert | - | ~14K comps/sec |
| Hilbert range queries | O(log N) | Sub-millisecond |

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

- PostgreSQL 15+
- PostGIS 3.3+
- CMake 3.16+
- C++17 compiler
- libpq-dev

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

CPE is NOT sliding n-grams (which explodes to O(n²)). It's a binary tree merge:

```
"Hello" (5 chars)
  Pass 1: [H,e,l,l,o] → [(H,e), (l,l), o] = 3 nodes
  Pass 2: [He, ll, o] → [(He,ll), o] = 2 nodes  
  Pass 3: [Hell, o] → [(Hell,o)] = 1 node (root)

Total compositions: 5-1 = 4 (always N-1 for binary tree)
```

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
