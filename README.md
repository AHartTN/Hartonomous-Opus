# Hartonomous-Opus: 4D Hypercube Semantic Substrate

A geometry-first semantic substrate that maps all digital content (text, data, AI models) into a 4D hypercube coordinate system with Hilbert curve indexing for efficient spatial queries.

## Quick Start

```bash
# Build everything
./scripts/build.sh

# Create database and seed all Unicode atoms (~10 seconds)
./scripts/seed_atoms.sh hypercube --force

# Run all tests
./scripts/test.sh

# Verify semantic clustering
psql -d hypercube -c "
SELECT 
    'A-a' as pair,
    sqrt(power(ST_X(a.coords) - ST_X(b.coords), 2) +
         power(ST_Y(a.coords) - ST_Y(b.coords), 2) +
         power(ST_Z(a.coords) - ST_Z(b.coords), 2) +
         power(ST_M(a.coords) - ST_M(b.coords), 2)) as distance_4d
FROM atom a, atom b
WHERE a.codepoint = 65 AND b.codepoint = 97;
"
-- A-a: ~0.00009 (very close - same letter, different case)
-- A-Z: ~0.076 (much farther - different letters)
```

## Architecture

### Core Concepts

1. **Atoms**: Unicode codepoints as fundamental constants
   - Each codepoint → 4D coordinate (32 bits per dimension)
   - BLAKE3 hash as content-addressed ID
   - Hilbert curve index (128-bit) for spatial ordering
   - All atoms distributed on the 3-sphere surface (S³ in 4D)
   - Semantically related characters (A/a/À) are geometrically adjacent

2. **Semantic Clustering**
   - Case pairs (A/a) are extremely close in space
   - Accented variants (A/À/Á) are near their base letter
   - Different categories (letters/digits/punctuation) are separated
   - Hilbert curve ordering preserves spatial locality

3. **Compositions**: Merkle DAG of atoms
   - N-grams, words, sentences, documents
   - LINESTRINGZM trajectories through 4D space
   - Centroids move toward the interior as complexity increases
   - Content-addressed deduplication via BLAKE3

4. **Spatial Operators**: PostGIS for semantic queries
   - `ST_Distance` - semantic distance
   - `ST_FrechetDistance` - trajectory similarity
   - `ST_Intersects` - conceptual overlap
   - `ST_ConvexHull` - semantic envelope

### Performance

| Operation | Time | Rate |
|-----------|------|------|
| Atom generation (1.1M) | ~200ms | 5.5M atoms/sec |
| Full database seeding | ~10s | 110K atoms/sec |
| Spatial index queries | O(log N) | Sub-millisecond |

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

### atom table
```sql
CREATE TABLE atom (
    id              blake3_hash PRIMARY KEY,  -- 32-byte BLAKE3 hash
    codepoint       INTEGER NOT NULL UNIQUE,  -- Unicode codepoint
    category        atom_category NOT NULL,   -- Semantic category
    coords          GEOMETRY(POINTZM, 0),     -- 4D coordinates
    hilbert_lo      BIGINT NOT NULL,          -- Hilbert index (lower 64 bits)
    hilbert_hi      BIGINT NOT NULL           -- Hilbert index (upper 64 bits)
);
```

### relation table
```sql
CREATE TABLE relation (
    id              blake3_hash PRIMARY KEY,  -- Merkle root hash
    coords          GEOMETRY(POINTZM, 0),     -- Centroid in 4D
    hilbert_lo      BIGINT NOT NULL,
    hilbert_hi      BIGINT NOT NULL,
    depth           INTEGER NOT NULL,         -- DAG depth
    child_count     INTEGER NOT NULL,
    atom_count      BIGINT NOT NULL           -- Total atoms in subtree
);
```

## Usage Examples

### Query atoms by codepoint
```sql
SELECT codepoint, chr(codepoint), category, ST_AsText(coords)
FROM atom WHERE codepoint IN (65, 97, 192);
```

### Find semantically similar atoms
```sql
SELECT b.codepoint, chr(b.codepoint), ST_3DDistance(a.coords, b.coords) as dist
FROM atom a, atom b
WHERE a.codepoint = 65  -- 'A'
  AND b.codepoint != 65
ORDER BY ST_3DDistance(a.coords, b.coords)
LIMIT 10;
```

### Convert text to trajectory
```sql
SELECT hypercube_text_to_linestring('Hello');
-- Returns LINESTRINGZM with 5 points
```

### Semantic trajectory distance
```sql
SELECT ST_FrechetDistance(
    hypercube_text_to_linestring('hello'),
    hypercube_text_to_linestring('hallo')
);
-- Returns small distance (similar spelling)
```

## File Structure

```
cpp/
├── include/hypercube/
│   ├── types.hpp       # Core types (Point4D, HilbertIndex, Blake3Hash)
│   ├── hilbert.hpp     # 4D Hilbert curve (128-bit index)
│   ├── coordinates.hpp # Hopf fibration + semantic ordering
│   └── blake3.hpp      # BLAKE3 hashing
├── src/
│   ├── hilbert.cpp     # Skilling's algorithm, optimized
│   ├── coordinates.cpp # Category-based surface projection
│   ├── blake3_pg.cpp   # BLAKE3 implementation
│   ├── hypercube.cpp   # PostgreSQL extension
│   ├── seed_atoms.cpp  # Parallel atom generator (EWKB output)
│   └── seed_atoms_direct.cpp  # Direct database seeder
├── tests/
│   ├── test_hilbert.cpp
│   ├── test_coordinates.cpp
│   ├── test_blake3.cpp
│   └── test_semantic.cpp  # Full semantic validation
└── sql/
    └── hypercube--1.0.sql

sql/
├── 001_schema.sql      # Core tables (atom, relation, relation_edge)
├── 002_functions.sql   # Spatial query functions
└── 003_ingestion.sql   # Text ingestion and composition

scripts/
├── build.sh            # Build C++ components
├── deploy.sh           # Full deployment
├── seed_atoms.sh       # Seed database with all Unicode atoms
└── test.sh             # Run all tests

tests/
└── test_semantic_validation.sql  # SQL integration tests
```

## Technical Details

### 4D Coordinate System (Hopf Fibration)

All 1.1M Unicode codepoints are distributed on the surface of a 3-sphere (S³) in 4D using the Hopf fibration:

- **Uniform distribution**: Every atom is on the surface (r² ≈ 1.0)
- **Semantic ordering**: Adjacent sequence indices → adjacent positions
- **Hierarchical decomposition**: Base-33 digits map to hyperspherical angles

### Semantic Ordering

Characters are ordered by semantic category before mapping to 3-sphere positions:
- Latin letters grouped by base (A/a/À/Á... then B/b/...)
- Digits grouped together
- Punctuation, symbols, control characters in separate regions
- CJK, Hangul, and other scripts in dedicated ranges

### Hilbert Curve

- 128-bit index stored as two `bigint` (hilbert_lo, hilbert_hi)
- Preserves spatial locality for efficient range queries
- Based on Skilling's compact algorithm with gray code optimization
- Lossless roundtrip: coords → index → coords

### BLAKE3 Hashing

- 256-bit content-addressed IDs
- Deterministic: same content = same hash
- Merkle tree composition for hierarchical structures

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
