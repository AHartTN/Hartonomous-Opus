# Hartonomous-Opus Scripts Documentation

Complete reference for all setup, build, ingest, and testing scripts for Windows and Linux.

## Quick Start

### Windows (PowerShell)
```powershell
# Full setup from scratch
.\scripts\windows\full-setup.ps1

# Or individual steps
.\scripts\windows\clean.ps1
.\scripts\windows\build.ps1
.\scripts\windows\setup-db.ps1
.\scripts\windows\ingest-testdata.ps1
.\scripts\windows\run_tests.ps1
```

### Linux/macOS (Bash)
```bash
# Full setup from scratch
./scripts/linux/full-setup.sh

# Or individual steps
./scripts/linux/clean.sh
./scripts/linux/build.sh
./scripts/linux/setup-db.sh
./scripts/linux/ingest-testdata.sh
./scripts/linux/run-tests.sh
```

---

## Script Categories

### 1. Environment & Configuration

#### `env.ps1` / `env.sh`
**Purpose**: Load configuration and set up environment variables

**What it does**:
- Loads database credentials from `scripts/config.env`
- Sets up PostgreSQL connection helpers (`hc_psql`, `hc_psql_admin`)
- Configures build paths and parallel job count
- Sets up Intel MKL environment (if available)
- Exports all HC_* variables for other scripts

**Usage**:
```bash
source scripts/linux/env.sh  # Automatically sourced by other scripts
```

**Configuration file**: `scripts/config.env`
```
HC_DB_HOST=localhost
HC_DB_PORT=5432
HC_DB_USER=hartonomous
HC_DB_PASS=hartonomous
HC_DB_NAME=hypercube
HC_BUILD_TYPE=Release
HC_PARALLEL_JOBS=8
HC_INGEST_THRESHOLD=0.7
```

---

### 2. Build & Compilation

#### `build.sh` / `build.ps1`
**Purpose**: Compile all C/C++ components with CMake

**What it does**:
- Creates `cpp/build` directory
- Runs CMake with Release configuration
- Detects and links Intel MKL (if installed)
- Compiles core libraries, PostgreSQL extensions, and tools
- Produces: `seed_atoms_parallel`, `ingest`, `extract_embeddings`, `*.so` extensions

**Usage**:
```bash
./scripts/linux/build.sh              # Standard build
./scripts/linux/build.sh --clean      # Clean rebuild
./scripts/linux/build.sh --install    # Install PostgreSQL extensions
```

**Options**:
- `--clean` / `-c`: Remove build directory first (clean rebuild)
- `--install` / `-i`: Install .so files to PostgreSQL plugin directory

**Time**: ~2-5 minutes (first build), ~30 seconds (incremental)

**Output**: Binary tools in `cpp/build/`:
- `seed_atoms_parallel` - Parallel Unicode atom seeder
- `ingest` - Content ingestion tool (text, vocab, etc)
- `extract_embeddings` - Model weight extractor
- `ingest_safetensor` - SafeTensor model loader
- `hypercube.so`, `semantic_ops.so`, etc - PostgreSQL extensions

---

#### `clean.sh` / `clean.ps1`
**Purpose**: Remove all build artifacts

**What it does**:
- Removes `cpp/build` directory
- Removes CMake cache files
- Removes compiled `.o`, `.so`, `.a` files
- Removes intermediate build artifacts

**Usage**:
```bash
./scripts/linux/clean.sh
```

**Note**: Does NOT touch database or configuration

---

### 3. Database Setup

#### `setup-db.sh` / `setup-db.ps1`
**Purpose**: Create database and load schema (idempotent)

**What it does**:
1. Tests PostgreSQL connection
2. Creates database if not exists
3. Loads schema from `sql/hypercube_schema.sql`:
   - 3 core tables (atom, composition, relation)
   - 813 SQL functions organized in subdirectories
   - Indexes and constraints
4. Loads PostgreSQL extensions (if built)
5. Seeds 1.1M Unicode atoms (if `seed_atoms_parallel` exists)

**Usage**:
```bash
./scripts/linux/setup-db.sh            # Safe: creates if missing
./scripts/linux/setup-db.sh --reset    # DESTRUCTIVE: drops and recreates
./scripts/linux/setup-db.sh --seed     # Only seed atoms (skip schema)
./scripts/linux/setup-db.sh --force    # Force re-seed atoms
```

**Options**:
- Default: Creates database + loads schema. Safe to run multiple times.
- `--reset`: Drops database (DESTRUCTIVE). Requires confirmation.
- `--seed`: Only run atom seeding, skip schema (useful if schema already loaded)
- `--force`: Force re-seed atoms even if population > 1.1M

**Time**: ~15 seconds (schema load), ~8 seconds (atom seeding)

**Output**: 
- Database `hypercube` with 11 tables
- 813 SQL functions
- 1,114,112 Unicode atoms seeded

---

### 4. Full Setup Pipelines

#### `full-setup.sh` / `full-setup.ps1`
**Purpose**: Complete setup from clean slate

**Pipeline**:
1. Clean build artifacts
2. Build C/C++ components
3. **DROP DATABASE** (destructive!)
4. Create fresh database
5. Load schema (35+ SQL files)
6. Seed 1.1M atoms
7. Build indexes
8. Validate database state

**Usage**:
```bash
./scripts/linux/full-setup.sh                # Complete setup
./scripts/linux/full-setup.sh --skip-build   # Use existing binaries
./scripts/linux/full-setup.sh --skip-seed    # Skip atom seeding
./scripts/linux/full-setup.sh --verbose      # Detailed output
```

**⚠️ WARNING**: This script will **DROP your database**. It asks for confirmation before proceeding.

**Time**: ~5 minutes total (30s build, 15s schema, 8s atoms)

**Best for**: 
- Fresh greenfield setup
- Testing complete pipeline
- CI/CD environments
- Recovery from corrupted database

---

#### `setup-all.sh` / `setup-all.ps1`
**Purpose**: Smart setup that preserves existing data

**Pipeline**:
1. Clean build artifacts (optional)
2. Build C/C++ components (optional)
3. Create database + schema (idempotent, safe)
4. Seed atoms if not already seeded
5. Ingest test data (optional)
6. Run tests (optional)

**Usage**:
```bash
./scripts/linux/setup-all.sh                 # Safe setup
./scripts/linux/setup-all.sh --reset         # Destructive reset
./scripts/linux/setup-all.sh --skip-clean    # Keep build artifacts
./scripts/linux/setup-all.sh --skip-build    # Skip compilation
./scripts/linux/setup-all.sh --skip-ingest   # Skip test data
./scripts/linux/setup-all.sh --skip-tests    # Skip test suite
```

**Options**:
- `--reset`: Include destructive database reset
- `--skip-*`: Skip individual pipeline stages

**Time**: ~10 minutes total (depends on what's skipped)

**Best for**:
- Iterative development
- Preserving existing data
- Partial rebuilds
- Production deployments

---

### 5. Data Ingestion

#### `ingest.sh` / `ingest.ps1`
**Purpose**: Ingest a single file or directory

**What it does**:
- Accepts text files (`.txt`), model files (`.safetensors`), or vocab files
- Uses C++ ingest tool for optimal performance
- Greedy pattern matching against existing vocabulary
- Creates compositions for novel n-grams
- Updates centroid coordinates and Hilbert indexes

**Usage**:
```bash
./scripts/linux/ingest.sh <path>                    # File or directory
./scripts/linux/ingest.sh <path> -t 0.3             # Custom threshold
./scripts/linux/ingest.sh <path> -n "mymodel"       # Custom name
```

**Examples**:
```bash
./scripts/linux/ingest.sh ~/Documents/mytext.txt
./scripts/linux/ingest.sh ~/models/bert-base/snapshots/abc123/
./scripts/linux/ingest.sh ~/data/ -t 0.5
```

**Supports**:
- Text files (`.txt`)
- Model directories with `vocab.txt`, `tokenizer.json`, `*.safetensors`
- Recursive directory ingestion

---

#### `ingest-models.sh` / `ingest-models.ps1`
**Purpose**: Ingest AI models from HuggingFace cache or test-data

**What it does**:
1. Auto-detects model structure (vocab, tokenizer, weights)
2. Loads vocabulary → creates token compositions
3. Extracts embeddings from model weights
4. Computes semantic similarity edges (threshold 0.7)
5. Updates relation table with model-contributed edges

**Usage**:
```bash
./scripts/linux/ingest-models.sh
```

**Looks in**: `test-data/embedding_models/` by default

**Supports**:
- HuggingFace `.safetensors` format
- JSON tokenizer configs
- Vocabulary files (one token per line)

---

#### `ingest-testdata.sh` / `ingest-testdata.ps1`
**Purpose**: Ingest complete test dataset

**Pipeline**:
1. Ingest model vocabulary (vocab.txt)
2. Extract semantic edges from model weights
3. Ingest text content (Moby Dick, etc)
4. Display final database statistics

**Usage**:
```bash
./scripts/linux/ingest-testdata.sh
```

**Requires**:
- Database already seeded with atoms
- Test data in `test-data/` directory

**What it ingests**:
- Model: `test-data/embedding_models/*/snapshots/*/`
- Text: `test-data/*.txt` (includes Moby Dick)

---

### 6. Testing & Validation

#### `test.sh` / `test.ps1`
**Purpose**: Comprehensive test suite

**Tests**:
1. **C++ Unit Tests**: Hilbert, coordinates, Blake3, etc
2. **Database Connectivity**: PostgreSQL, PostGIS
3. **Schema Validation**: Tables, indexes exist
4. **Atom Seeding**: >1.1M Unicode codepoints
5. **SQL Functions**: atom_is_leaf, atom_centroid, etc
6. **AI/ML Operations**: semantic_neighbors, attention, analogy
7. **Spatial Queries**: KNN, Hilbert range queries
8. **Data Integrity**: Full consistency checks

**Usage**:
```bash
./scripts/linux/test.sh              # Full test suite
./scripts/linux/test.sh --quick      # Skip slow tests
```

**Output**:
- Section headers (C++ Tests, Database, Schema, etc)
- Individual test results with ✓/✗ status
- Final statistics and system state

**Time**: ~30 seconds (quick), ~2 minutes (full)

---

#### `validate.sh` / `validate.ps1`
**Purpose**: Validate database state without modifying data

**What it checks**:
1. PostgreSQL connection
2. Database exists
3. Table row counts (atoms, compositions, relations)
4. Statistics (children, centroids)
5. Runs test suite (optional)

**Usage**:
```bash
./scripts/linux/validate.sh             # Full validation + tests
./scripts/linux/validate.sh --quick     # Just show state
./scripts/linux/validate.sh --skip-tests # State only
./scripts/linux/validate.sh --full      # Full validation + benchmarks
```

**Output**:
```
Atoms:           1114112
Compositions:    42187
Relations:       123456
Composition Children: 85674
With Centroids:  41293
```

**Safe**: Does NOT modify any data

---

#### `run-tests.sh` / `run_tests.ps1`
**Purpose**: Test runner with flexible options

**What it does**:
- Delegates to `test.sh` with options
- Provides consistent interface across platforms
- Supports filtering and verbose output

**Usage**:
```bash
./scripts/linux/run-tests.sh                # All tests
./scripts/linux/run-tests.sh --quick        # Skip slow tests
./scripts/linux/run-tests.sh --verbose      # Detailed output
./scripts/linux/run-tests.sh --filter pattern  # Pattern matching
```

---

#### `e2e-test.sh` / `e2e-test.ps1`
**Purpose**: End-to-end integration test from clean slate

**Full Pipeline**:
1. Clean build
2. Build C/C++
3. Drop database (destructive)
4. Create database + load schema
5. Seed atoms (1.1M)
6. Ingest models
7. Ingest content
8. Run integration tests (10+ assertions)
9. Run unit tests
10. Validate complete system

**Usage**:
```bash
./scripts/linux/e2e-test.sh                 # Full E2E test
./scripts/linux/e2e-test.sh --skip-build    # Skip build
./scripts/linux/e2e-test.sh --skip-seed     # Skip atom seeding
./scripts/linux/e2e-test.sh --skip-models   # Skip model ingestion
./scripts/linux/e2e-test.sh --skip-content  # Skip text ingestion
./scripts/linux/e2e-test.sh --verbose       # Verbose output
./scripts/linux/e2e-test.sh --fail-fast     # Stop on first failure
```

**Test Assertions**:
- Database connection works
- Schema fully loaded
- Atoms properly seeded (>1M)
- All core functions present
- Data ingestion successful
- Integration tests pass

**Time**: ~10 minutes complete

**Best for**:
- CI/CD pipelines
- Release validation
- System health checks
- Regression testing

---

## Script Execution Flow

### Typical Development Workflow
```bash
# One-time setup
./scripts/linux/setup-all.sh

# Development iteration
./scripts/linux/build.sh
./scripts/linux/test.sh
./scripts/linux/ingest.sh ~/my_data/

# Validation before commit
./scripts/linux/validate.sh
```

### CI/CD Pipeline
```bash
# In GitHub Actions, GitLab CI, etc
./scripts/linux/e2e-test.sh --fail-fast
# Exit code: 0 = success, non-zero = failure
```

### Production Deployment
```bash
# Fresh system
./scripts/linux/full-setup.sh

# Update with new data
./scripts/linux/setup-all.sh --skip-build

# Validate health
./scripts/linux/validate.sh --full
```

---

## Environment Variables

All scripts respect these environment variables (or load from `scripts/config.env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `HC_DB_HOST` | localhost | PostgreSQL server |
| `HC_DB_PORT` | 5432 | PostgreSQL port |
| `HC_DB_USER` | hartonomous | Database user |
| `HC_DB_PASS` | hartonomous | Database password |
| `HC_DB_NAME` | hypercube | Database name |
| `HC_PROJECT_ROOT` | (auto-detected) | Project root directory |
| `HC_BUILD_DIR` | cpp/build | Build output directory |
| `HC_BUILD_TYPE` | Release | CMake build type |
| `HC_PARALLEL_JOBS` | $(nproc) | Parallel build jobs |
| `HC_INGEST_THRESHOLD` | 0.7 | Semantic similarity threshold |
| `MKLROOT` | (auto-detected) | Intel MKL root path |

---

## Troubleshooting

### Build failures
```bash
./scripts/linux/clean.sh              # Clean build artifacts
./scripts/linux/build.sh               # Rebuild from scratch
```

### Database connection issues
```bash
# Check connection
PGPASSWORD=hartonomous psql -h localhost -U hartonomous -d postgres -c "SELECT 1"

# Check config
cat scripts/config.env
```

### Low atom count after seeding
```bash
./scripts/linux/setup-db.sh --force     # Force re-seed
```

### Schema errors
```bash
./scripts/linux/setup-db.sh --reset     # Full reset
# Or check: sql/README.md for schema documentation
```

### Test failures
```bash
./scripts/linux/test.sh --verbose       # See detailed output
./scripts/linux/validate.sh --full      # Full validation
```

---

## Linux vs Windows Feature Parity

| Feature | Linux | Windows |
|---------|-------|---------|
| Build & compilation | ✓ | ✓ |
| Database setup | ✓ | ✓ |
| Full setup pipeline | ✓ | ✓ |
| Data ingestion | ✓ | ✓ |
| Testing suite | ✓ | ✓ |
| E2E validation | ✓ | ✓ |
| Comprehensive docs | ✓ | ✓ |

Both platforms have identical functionality and feature coverage.

---

## See Also

- [README.md](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture
- [sql/README.md](sql/README.md) - SQL schema documentation
- [cpp/CMakeLists.txt](cpp/CMakeLists.txt) - Build configuration
