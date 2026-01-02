# Hartonomous-Opus Project Audit
**Date**: 2026-01-02

## Current State Summary

The system implements a **Semantic Web** using:
- 4D Hypercube with Hilbert curve indexing
- BLAKE3 content-addressed Merkle DAG
- Cascading Pair Encoding (CPE) for universal content atomization
- PostGIS GEOMETRYZM for spatial storage (POINTZM for atoms, LINESTRINGZM for compositions)

## File Status

### SQL Files - CURRENT
| File | Status | Purpose |
|------|--------|---------|
| 011_unified_atom.sql | **CURRENT** | Single atom table schema |
| 012_semantic_udf.sql | **CURRENT** | Semantic query functions |
| 013_model_infrastructure.sql | **CURRENT** | AI model ingestion infrastructure |

### SQL Files - DEPRECATED (pre-unified schema)
| File | Status | Purpose |
|------|--------|---------|
| 001_schema.sql | DEPRECATED | Old 3-table schema (atom/relation/relation_edge) |
| 002_functions.sql | DEPRECATED | Old utility functions |
| 003_ingestion.sql | DEPRECATED | Old n-gram ingestion (O(n²)) |
| 004_bpe_tokenizer.sql | DEPRECATED | Separate BPE tokenizer |
| 005_model_ingestion.sql | DEPRECATED | Old model ingestion |
| 006_spatial_queries.sql | DEPRECATED | Old spatial queries |
| 007_semantic_edges.sql | DEPRECATED | Old semantic edges |
| 008_metadata.sql | DEPRECATED | Old metadata tables |
| 009_cascading_pair_encoding.sql | DEPRECATED | SQL-based CPE (too slow) |
| 010_lossless_schema.sql | DEPRECATED | Migration to lossless coords |

### C++ Files - CURRENT
| File | Status | Purpose |
|------|--------|---------|
| cpe_ingest.cpp | **CURRENT** | Main CPE ingester (unified schema) |
| seed_atoms_parallel.cpp | **CURRENT** | Fast parallel Unicode seeder |
| semantic_ops.cpp | **CURRENT** | C++ UDFs for semantic operations |
| ingest_safetensor.cpp | **CURRENT** | Safetensor package ingester |
| hypercube.cpp | **CURRENT** | PostgreSQL extension core |
| blake3_pg.cpp | **CURRENT** | BLAKE3 for PostgreSQL |
| hilbert.cpp | **CURRENT** | Hilbert curve implementation |
| coordinates.cpp | **CURRENT** | Coordinate utilities |

### C++ Files - DEPRECATED
| File | Status | Purpose |
|------|--------|---------|
| seed_atoms.cpp | DEPRECATED | Old single-threaded seeder |
| seed_atoms_direct.cpp | DEPRECATED | Old direct seeder |
| extract_embeddings.cpp | DEPRECATED | Superseded by ingest_safetensor.cpp |
| ingest_model.cpp | DEPRECATED | Old model ingester |

### Shell Scripts - CURRENT
| File | Status | Purpose |
|------|--------|---------|
| setup.sh | **CURRENT** | Single entry point for all operations |
| validate.sh | **CURRENT** | Test suite runner |

### Shell Scripts - DEPRECATED
| File | Status | Purpose |
|------|--------|---------|
| scripts/deploy.sh | DEPRECATED | Superseded by setup.sh |
| scripts/build.sh | DEPRECATED | Superseded by setup.sh |
| scripts/seed_atoms.sh | DEPRECATED | Superseded by setup.sh |
| scripts/ingest_content.sh | DEPRECATED | Superseded by setup.sh |
| scripts/ingest_model.sh | DEPRECATED | Superseded by setup.sh |
| scripts/test.sh | DEPRECATED | Superseded by validate.sh |

## Recommended Actions

### Phase 1: Cleanup (Archive deprecated files)
1. Move deprecated SQL to `sql/deprecated/`
2. Move deprecated C++ to `cpp/src/deprecated/`
3. Move deprecated scripts to `scripts/deprecated/`

### Phase 2: Documentation
1. Add status headers to all current files
2. Update ARCHITECTURE.md with file layout
3. Update README.md with current usage

### Phase 3: Build Infrastructure
1. Update CMakeLists.txt to only build current C++ files
2. Add proper test targets
3. Add install targets for extensions

### Phase 4: Test Suite
1. Create comprehensive C++ test suite
2. Create SQL function tests
3. Create integration tests

## Known Issues (as of 2024-01-02)

### Sequitur Ingester Segfault
- **Status**: BROKEN
- **Issue**: Segfaults when input contains repeated digrams (e.g., "abab")
- **Root cause**: Pointer management bug in `substitute_digram()` during rule creation
- **Workaround**: Use `cpe_ingest` instead (binary pair encoding, works correctly)
- **Priority**: Medium - CPE is functional, Sequitur would provide better grammar learning

### CPE vs Sequitur Comparison
| Feature | CPE | Sequitur |
|---------|-----|----------|
| Status | Working | Broken |
| Composition style | Binary pairs | Variable-length grammar rules |
| Natural boundaries | No (fixed binary tree) | Yes (discovers words/phrases) |
| Deduplication | By content hash | By grammar rule reuse |

