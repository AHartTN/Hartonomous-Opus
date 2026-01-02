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

# Ingest any directory of text files
./setup.sh ingest ~/Documents/notes/

# Visualize the CPE tree structure
./setup.sh tree "Mississippi"

# Query similarity
./setup.sh similar "machine learning"

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
| `./setup.sh reset` | Drop and reset database |

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
   - Each codepoint → 4D coordinate (32 bits per dimension, stored as INTEGER)
   - BLAKE3 hash as content-addressed ID
   - Hilbert curve index (128-bit) for spatial ordering
   - All atoms distributed on the 3-sphere surface (S³ in 4D)
   - **Lossless**: Coordinates stored as raw 32-bit integers, not PostGIS floats

2. **Edges**: Transitions between atoms/compositions
   - Record the sequence order (ordinal)
   - Enable bit-perfect reconstruction via DFS traversal
   - Same edge (a→b) is never duplicated

3. **Compositions**: Merkle DAG via Cascading Pair Encoding (CPE)
   - Binary tree structure - each composition has exactly 2 children
   - Characters cascade: `N chars → N/2 pairs → N/4 → ... → 1 root`
   - Total compositions ≈ 2N (geometric series, NOT O(n²))
   - Content-addressed deduplication: "the" from any document = same ID
   - Centroid = integer average of child coordinates (lossless)

4. **Global Deduplication**
   - First ingest creates patterns; subsequent ingests only add edge references
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

### Type System (Critical for Losslessness)

| Field | PostgreSQL Type | Interpretation |
|-------|----------------|----------------|
| coord_x/y/z/m | INTEGER | Signed 32-bit (bit pattern = uint32) |
| hilbert_lo/hi | BIGINT | Unsigned 64-bit (stored as signed) |
| id | BYTEA(32) | 256-bit BLAKE3 hash |

**No PostGIS for relation centroids** - PostGIS normalizes to floats, losing precision.
Spatial queries on atoms use PostGIS; relations use integer coordinates directly.

## Building

```bash
# Build C++ components and run tests
./scripts/build.sh

# Full deployment (creates database, seeds atoms, applies functions)
./scripts/deploy.sh hypercube --force
```

### Requirements

- PostgreSQL 15+
- PostGIS 3.3+
- CMake 3.16+
- C++17 compiler
- libpq-dev

## Database Schema

### atom table (perimeter landmarks)
```sql
CREATE TABLE atom (
    id              BYTEA(32) PRIMARY KEY,    -- BLAKE3 hash
    codepoint       INTEGER NOT NULL UNIQUE,  -- Unicode codepoint
    category        atom_category NOT NULL,   -- Semantic category
    
    -- Lossless 4D coordinates (source of truth)
    coord_x         INTEGER NOT NULL,         -- Signed int32 (bit pattern = uint32)
    coord_y         INTEGER NOT NULL,
    coord_z         INTEGER NOT NULL,
    coord_m         INTEGER NOT NULL,
    
    -- 128-bit Hilbert index
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,
    
    -- PostGIS for spatial queries (derived, atoms only)
    coords          GEOMETRY(POINTZM, 0)
);
```

### relation table (compositions)
```sql
CREATE TABLE relation (
    id              BYTEA(32) PRIMARY KEY,    -- BLAKE3(ordinal||child_hash||...)
    
    -- Lossless centroid coordinates (NO PostGIS)
    coord_x         INTEGER NOT NULL,
    coord_y         INTEGER NOT NULL,
    coord_z         INTEGER NOT NULL,
    coord_m         INTEGER NOT NULL,
    
    -- 128-bit Hilbert index of centroid
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,
    
    depth           INTEGER NOT NULL,         -- DAG depth from atoms
    child_count     INTEGER NOT NULL,         -- Always 2 for CPE
    atom_count      BIGINT NOT NULL           -- Total atoms in subtree
);
```

### relation_edge table (Merkle DAG edges)
```sql
CREATE TABLE relation_edge (
    parent_id       BYTEA(32) REFERENCES relation(id),
    child_id        BYTEA(32),                -- References atom OR relation
    ordinal         INTEGER,                  -- Position (0=left, 1=right)
    is_atom         BOOLEAN,
    PRIMARY KEY (parent_id, ordinal)
);
```

## Usage Examples

### Ingest and visualize CPE tree
```bash
./setup.sh tree "Mississippi"
# Shows binary tree structure:
#   Mississippi
#   ├── Missi
#   │   ├── Mi
#   │   └── ss + i merged
#   └── ssippi
#       └── ...
```

### Query atoms by codepoint
```sql
SELECT codepoint, chr(codepoint), category,
       coord_x, coord_y, coord_z, coord_m
FROM atom WHERE codepoint IN (65, 97, 192);
```

### Find similar compositions (lossless integer distance)
```sql
SELECT encode(r.id, 'hex'),
       sqrt(
           power((r.coord_x - q.coord_x)::numeric, 2) +
           power((r.coord_y - q.coord_y)::numeric, 2) +
           power((r.coord_z - q.coord_z)::numeric, 2) +
           power((r.coord_m - q.coord_m)::numeric, 2)
       ) as distance
FROM relation r, (SELECT * FROM relation WHERE id = $query_id) q
ORDER BY distance LIMIT 10;
```

### Reconstruct original content from composition
```sql
WITH RECURSIVE dag AS (
    SELECT id, false as is_atom, ARRAY[]::int[] as path
    FROM relation WHERE id = $root_id
    UNION ALL
    SELECT e.child_id, e.is_atom, d.path || e.ordinal
    FROM dag d JOIN relation_edge e ON e.parent_id = d.id
    WHERE NOT d.is_atom
)
SELECT string_agg(chr(a.codepoint), '' ORDER BY d.path)
FROM dag d JOIN atom a ON a.id = d.id WHERE d.is_atom;
```

## File Structure

```
Hartonomous-Opus/
├── ARCHITECTURE.md           # Canonical technical specification
├── README.md                 # This file
├── setup.sh                  # Single entry point for all operations
├── .env.example              # Configuration template
│
├── cpp/
│   ├── include/hypercube/
│   │   ├── types.hpp         # Core types (Point4D, HilbertIndex, Blake3Hash)
│   │   ├── hilbert.hpp       # 4D Hilbert curve (128-bit index)
│   │   ├── coordinates.hpp   # Coordinate utilities
│   │   └── blake3.hpp        # BLAKE3 hashing
│   ├── src/
│   │   ├── cpe_ingest.cpp    # CPE ingester (main workhorse)
│   │   ├── seed_atoms_parallel.cpp  # Unicode seeder
│   │   ├── hypercube.cpp     # PostgreSQL extension
│   │   ├── blake3_pg.cpp     # BLAKE3 for PostgreSQL
│   │   ├── hilbert.cpp       # Skilling's algorithm
│   │   └── coordinates.cpp   # Coordinate utilities
│   └── sql/
│       └── hypercube--1.0.sql
│
├── sql/
│   ├── 001_schema.sql        # Core tables (atom, relation, relation_edge)
│   ├── 002_functions.sql     # Utility functions
│   ├── 009_cascading_pair_encoding.sql  # SQL CPE (reference impl)
│   └── 010_lossless_schema.sql  # Lossless coordinate migration
│
├── scripts/                  # Build and deployment scripts
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

Each composition hash = BLAKE3(ordinal||left_hash || ordinal||right_hash)
Centroid = integer average of child coordinates (lossless)

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

DFS traversal of edges in ordinal order reconstructs original bytes:
1. Start at root composition
2. Follow edges to children (ordinal 0 first, then ordinal 1)
3. When reaching atoms, emit the codepoint
4. Concatenate = original content

## Tests

```bash
# Run all tests
./scripts/test.sh

# C++ unit tests only
./scripts/test.sh --cpp-only

# SQL integration tests only
./scripts/test.sh --sql-only

# Verbose output
./scripts/test.sh --verbose
```

## License

Apache 2.0
