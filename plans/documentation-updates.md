# Documentation Updates Plan

## Overview

Comprehensive documentation modernization to align all materials with current 3-table architecture, remove deprecated references, and establish clear developer guidance. This addresses critical documentation drift that has accumulated through multiple architectural pivots.

## Priority: HIGH - Critical for Developer Onboarding

---

## 1. README.md Architecture Description Overhaul

### Current State Analysis
The README.md contains significant architectural inconsistencies that confuse developers and provide incorrect setup information.

**Specific Issues Identified:**

1. **Section 3. "Two Table Model" (lines 102-108)**: Describes obsolete unified schema
   ```markdown
   3. **Two Table Model**
       - `atom` table stores nodes (leaves and compositions)
       - `relation` table stores edges (parent→child with ordinal)
   ```
   This conflicts with current 4-table implementation (`atom`, `composition`, `composition_child`, `relation`)

2. **Section "Database Schema (Unified)" (line 151)**: Shows single `atom` table with `children BYTEA[]`
   ```sql
   CREATE TABLE atom (
       id BYTEA PRIMARY KEY,
       geom GEOMETRY(GEOMETRYZM, 0),
       children BYTEA[],  -- This column doesn't exist in current schema!
       -- ... other fields
   );
   ```
   Current schema uses separate `composition_child` junction table

3. **File Structure References (line 240)**: Points to deprecated files
   ```
   sql/011_unified_atom.sql # Unified schema (CURRENT)  -- WRONG!
   ```
   Should reference `sql/001_schema.sql` as current

### Implementation Approach

**Phase 1: Schema Model Correction**
Replace "Two Table Model" section with accurate 4-table description:
```markdown
## Architecture

### Core Concepts

1. **Atoms**: Unicode codepoints as fundamental constants (perimeter landmarks)
   - Each codepoint → 4D coordinate (32 bits per dimension)
   - BLAKE3 hash as content-addressed ID
   - Hilbert curve index (128-bit) for spatial ordering

2. **Compositions**: Binary Merkle DAG via PMI Contraction
   - PMI identifies significant co-occurrences
   - Highest-PMI pairs contracted into new compositions recursively
   - Content-addressed: "the" from any document = same ID
   - Geometry = LINESTRINGZM trajectory through child centroids

3. **The Four Table Model**
   - `atom` table: Unicode codepoints only (leaves)
   - `composition` table: Aggregations with centroids
   - `composition_child` table: Ordered parent-child relationships
   - `relation` table: Semantic edges between nodes

4. **Global Deduplication**: Same bytes → same composition ID, regardless of source
```

**Phase 2: Database Schema Update**
Replace unified schema with current 4-table definitions:
```sql
-- Current schema: 4-table model
CREATE TABLE atom (
    id BYTEA PRIMARY KEY,
    codepoint INTEGER NOT NULL UNIQUE,
    value BYTEA NOT NULL,
    geom GEOMETRY(POINTZM, 0) NOT NULL,
    hilbert_lo BIGINT NOT NULL,
    hilbert_hi BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE composition (
    id BYTEA PRIMARY KEY,
    label TEXT,
    depth INTEGER NOT NULL DEFAULT 1,
    child_count INTEGER NOT NULL,
    atom_count BIGINT NOT NULL,
    geom GEOMETRY(LINESTRINGZM, 0),
    centroid GEOMETRY(POINTZM, 0),
    hilbert_lo BIGINT,
    hilbert_hi BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE composition_child (
    composition_id BYTEA NOT NULL REFERENCES composition(id),
    ordinal SMALLINT NOT NULL,
    child_type CHAR(1) NOT NULL,
    child_id BYTEA NOT NULL,
    PRIMARY KEY (composition_id, ordinal)
);

CREATE TABLE relation (
    id BIGSERIAL PRIMARY KEY,
    source_type CHAR(1) NOT NULL,
    source_id BYTEA NOT NULL,
    target_type CHAR(1) NOT NULL,
    target_id BYTEA NOT NULL,
    relation_type CHAR(1) NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    source_model TEXT NOT NULL DEFAULT '',
    source_count INTEGER NOT NULL DEFAULT 1,
    layer INTEGER NOT NULL DEFAULT -1,
    component TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Phase 3: File Structure Cleanup**
Update all file references to current active files:
- Remove references to `sql/011_unified_atom.sql.DEPRECATED`
- Point to `sql/001_schema.sql` as current schema
- Update file listings to reflect current structure
- Remove references to archived/deprecated components

**Validation Criteria**:
- README.md accurately describes current 4-table architecture
- Database schema section shows correct table definitions
- All file references point to active, current files
- Setup instructions work with current architecture

---

## 2. Cross-Document Architecture Standardization

### Documentation Inconsistency Analysis

**ARCHITECTURE.md vs README.md Conflict**:
- **README.md**: Describes "Two Table Model" with unified atom/composition storage
- **ARCHITECTURE.md**: Correctly describes "Three Table Model" (actually 4 tables including junction)
- **Missing FUNCTIONALITY_CATALOG.md**: Referenced in audit but file doesn't exist

**Functional vs Architectural Documentation**:
- Architecture docs focus on system design and data models
- Functional docs should describe capabilities and APIs
- Need clear separation of concerns

### Implementation Approach

**Phase 1: Establish Canonical Architecture Source**
Designate ARCHITECTURE.md as the single source of truth for architectural decisions:
- Update README.md to reference ARCHITECTURE.md for detailed technical specs
- Create summary architecture section in README.md that defers to canonical doc
- Remove conflicting architectural descriptions from README.md

**Phase 2: Create Missing Documentation**
Since FUNCTIONALITY_CATALOG.md is referenced but missing, create it with functional specifications:
```markdown
# Functionality Catalog

## Core Capabilities

### Content Ingestion
- Text decomposition via Cascading Pair Encoding (CPE)
- AI model ingestion via Safetensor parsing
- Global deduplication across all content types

### Semantic Query Operations
- Exact content identity matching
- Fuzzy similarity via Fréchet distance
- Semantic neighbors via 4D centroid KNN
- Analogy computation (vector arithmetic)
- Edge walking via co-occurrence graphs

### Spatial Operations
- Hilbert curve range queries
- 4D geometric operations
- Hypersphere surface constraints
```

**Phase 3: Documentation Hierarchy**
Establish clear documentation levels:
- **README.md**: Quick start, high-level overview, basic usage
- **ARCHITECTURE.md**: Canonical technical specification, data models, algorithms
- **FUNCTIONALITY_CATALOG.md**: Feature inventory, API capabilities
- **docs/API_REFERENCE.md**: Detailed API documentation
- **docs/MIGRATION_GUIDE.md**: Breaking changes and migration paths

**Validation Criteria**:
- All documents describe consistent architecture
- Clear separation between architectural and functional documentation
- No conflicting schema descriptions across files

---

## 3. Schema File Reference Cleanup

### Deprecated File Inventory

**Current Active Schema Files**:
- `sql/001_schema.sql` - Core 4-table schema definition
- `sql/002_core_functions.sql` - Basic database functions
- `sql/003_query_api.sql` - Semantic query operations
- `sql/004_generative_engine.sql` - Generative AI capabilities

**Deprecated/Archived Files** (DO NOT REFERENCE):
- `sql/archive/011_unified_atom.sql.DEPRECATED`
- `sql/archive/012_semantic_udf.sql.SUPERSEDED_BY_025`
- Multiple files in `sql/archive/` marked as `.DEPRECATED`, `.OLD_SCHEMA`

### Implementation Approach

**Phase 1: Comprehensive Reference Audit**
Search all documentation files for deprecated references:
- README.md file structure section
- Any setup/installation guides
- Code comments referencing old files
- Build scripts and configuration files

**Phase 2: Reference Updates**
Replace all deprecated file references:
```markdown
# BEFORE (WRONG):
├── sql/
│   ├── 011_unified_atom.sql        # Unified schema (CURRENT)
│   ├── 012_semantic_udf.sql        # SQL UDF infrastructure

# AFTER (CORRECT):
├── sql/
│   ├── 001_schema.sql              # 4-table schema definition
│   ├── 002_core_functions.sql      # Core SQL functions
│   ├── 003_query_api.sql           # Semantic query API
```

**Phase 3: Archive Documentation**
Create documentation for the archive directory structure:
```markdown
## Archived Components

The `sql/archive/` directory contains deprecated schema versions and functions.
These are maintained for historical reference but should not be used:

- `011_unified_atom.sql.DEPRECATED` - Obsolete unified atom/composition table
- `012_semantic_udf.sql.SUPERSEDED_BY_025` - Replaced by modular function design
- Files marked `.OLD_SCHEMA` - Pre-4-table architecture attempts
```

**Validation Criteria**:
- Zero references to deprecated files in active documentation
- All file listings show current active files
- Clear documentation of archive purpose and contents

---

## 4. Architectural Evolution Documentation

### Missing Historical Context

**Major Architectural Pivots** (Currently Undocumented):
1. **Unified → 4-Table Schema**: Migration from single `atom` table to normalized design
2. **JL → Laplacian Eigenmaps**: Shift from Johnson-Lindenstrauss to spectral projection
3. **Sequitur → CPE**: Algorithm change for content decomposition
4. **2D → 4D Projection**: Dimensionality increase for better semantic representation

**Institutional Knowledge Gaps**:
- Rationale for each architectural decision
- Performance implications of changes
- Lessons learned from failed approaches
- Future considerations based on experience

### Implementation Approach

**Phase 1: Create Architectural History Document**
New file: `docs/ARCHITECTURAL_CHANGES.md`

```markdown
# Architectural Evolution

## Version 6 (2026-01-05): 4-Table Schema with Laplacian Eigenmaps

### Changes
- Migrated from unified atom table to normalized 4-table schema
- Implemented Laplacian Eigenmaps for structure-preserving 4D projection
- Separated semantic relations into dedicated relation table

### Rationale
- Normalized schema eliminates data redundancy
- Laplacian projection preserves local structure better than JL transform
- Separate relation table enables richer semantic edge types

### Impact
- Improved query performance through proper normalization
- Better semantic accuracy with spectral projection
- Enhanced scalability with dedicated relation storage

## Version 5 (2025-01-16): Binary PMI Merkle DAG

### Changes
- Implemented Cascading Pair Encoding (CPE) for content decomposition
- Replaced Sequitur algorithm with PMI-based contraction

### Rationale
- PMI provides better semantic clustering than pure frequency
- Sliding window approach captures all n-grams efficiently

### Lessons Learned
- Pure frequency-based algorithms miss semantic relationships
- PMI contraction provides logarithmic vocabulary growth
```

**Phase 2: Decision Documentation**
For each major change, document:
- Problem being solved
- Alternative approaches considered
- Performance benchmarks before/after
- Compatibility impact and migration strategy

**Phase 3: Future Considerations**
Document architectural debt and planned improvements:
- Areas needing further optimization
- Scalability limitations identified
- Research directions that could enhance the system

**Validation Criteria**:
- Complete historical record of all major architectural changes
- Clear rationale documented for each decision
- Lessons learned captured to prevent future mistakes
- Future roadmap informed by historical experience

---

## 5. API Reference and Migration Guide Creation

### Current Documentation Gaps

**Missing Developer Resources**:
- Complete SQL function reference with signatures
- C++ API documentation
- Migration guides for architectural changes
- Troubleshooting guides for common issues
- Performance tuning recommendations

**Developer Experience Issues**:
- No comprehensive API reference
- Missing code examples for common operations
- No guidance for handling breaking changes
- Limited troubleshooting information

### Implementation Approach

**Phase 1: SQL API Reference**
Create `docs/API_REFERENCE.md` with complete function documentation:

```markdown
# SQL API Reference

## Content Identity Functions

### content_exists(text) → boolean
Check if composition exists for given text.

**Parameters:**
- `text`: Input text to check

**Returns:** `true` if composition exists, `false` otherwise

**Example:**
```sql
SELECT content_exists('whale');  -- Returns: true
```

### content_get(text) → table
Get full composition information for text.

**Returns:**
- `id` BYTEA: Composition hash
- `depth` INTEGER: Composition depth
- `atom_count` BIGINT: Total atoms in subtree
- `centroid` GEOMETRY: 4D centroid coordinate

**Example:**
```sql
SELECT * FROM content_get('whale');
```

## Semantic Query Functions

### similar(text, k) → table(content, distance)
Find k most similar compositions via Fréchet distance.

**Parameters:**
- `text`: Query text
- `k` INTEGER: Number of results

**Returns:**
- `content` TEXT: Similar text
- `distance` REAL: Similarity distance (lower = more similar)

**Example:**
```sql
SELECT * FROM similar('whale', 10);
```
```

**Phase 2: C++ API Documentation**
Document key C++ classes and interfaces:

```markdown
# C++ API Reference

## Core Classes

### CoordinateMapper
Maps Unicode codepoints to 4D hypersphere coordinates.

**Key Methods:**
- `map_codepoint(uint32_t cp) → Point4D`: Get 4D coordinates for codepoint
- `categorize(uint32_t cp) → AtomCategory`: Get Unicode category
- `centroid(const std::vector<Point4D>&) → Point4D`: Compute geometric centroid

### LaplacianProjector
Projects high-dimensional embeddings to 4D using spectral methods.

**Key Methods:**
- `project(const std::vector<std::vector<float>>& embeddings) → ProjectionResult`
- Builds k-NN similarity graph
- Computes Laplacian eigenmaps
- Applies Gram-Schmidt orthonormalization
```

**Phase 3: Migration Guide**
Create `docs/MIGRATION_GUIDE.md` for architectural changes:

```markdown
# Migration Guide

## From Version 5 to Version 6 (4-Table Schema)

### Database Schema Changes

**Old Schema (Version 5):**
```sql
-- Single unified atom table
CREATE TABLE atom (
    id BYTEA PRIMARY KEY,
    children BYTEA[],  -- Nested composition storage
    -- ... other fields
);
```

**New Schema (Version 6):**
```sql
-- Normalized 4-table schema
CREATE TABLE atom (...);           -- Leaves only
CREATE TABLE composition (...);    -- Aggregations only
CREATE TABLE composition_child (
    composition_id BYTEA REFERENCES composition(id),
    ordinal SMALLINT,
    child_type CHAR(1),
    child_id BYTEA,
    PRIMARY KEY (composition_id, ordinal)
);
CREATE TABLE relation (...);       -- Semantic edges
```

### Migration Steps

1. **Backup existing database**
2. **Run migration script** (provided in `sql/migration_v5_to_v6.sql`)
3. **Update application code** to use new table structure
4. **Rebuild indexes** on new schema
5. **Validate data integrity**

### Code Changes Required

**Query Changes:**
```sql
-- OLD: Single table queries
SELECT * FROM atom WHERE id = $1;

-- NEW: Type-aware queries
SELECT * FROM composition WHERE id = $1
UNION ALL
SELECT * FROM atom WHERE id = $1;
```

**Insertion Changes:**
```sql
-- OLD: Single table insertion
INSERT INTO atom (id, children, ...) VALUES ($1, $2, ...);

-- NEW: Separate table insertions
INSERT INTO composition (id, ...) VALUES ($1, ...);
INSERT INTO composition_child (composition_id, ordinal, child_type, child_id)
VALUES ($1, 0, 'A', $child1), ($1, 1, 'C', $child2);
```
```

**Validation Criteria**:
- Complete API reference covering all public interfaces
- Migration guide for all breaking architectural changes
- Code examples for common usage patterns
- Troubleshooting section with solutions to known issues