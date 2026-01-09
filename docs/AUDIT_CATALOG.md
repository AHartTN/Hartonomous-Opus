# Catalog of Inconsistencies, Conflicts, and Decision Changes

## Documentation Inconsistencies

### 1. Schema Model Description Mismatch
**Location**: README.md vs ARCHITECTURE.md
**Issue**: README describes "Two Table Model", ARCHITECTURE describes "Three Table Model"
**Evidence**:
- README.md line 102-108: "Two Table Model - `atom` table stores nodes (leaves and compositions) - `relation` table stores edges"
- ARCHITECTURE.md line 17-31: "The Three Table Model" with separate atom, composition, relation tables
- FUNCTIONALITY_CATALOG.md follows the three-table model

**Decision Change**: Migration from unified schema (v5) to normalized three-table schema (v6)

### 2. Schema File References
**Location**: README.md vs actual SQL files
**Issue**: README references `sql/011_unified_atom.sql` as current, but it's deprecated
**Evidence**:
- README.md line 240: "sql/011_unified_atom.sql # Unified schema (CURRENT)"
- Actual current schema: `sql/001_schema.sql` (three tables)
- Deprecated file: `sql/archive/011_unified_atom.sql.DEPRECATED`

## Code vs Documentation Conflicts

### 3. Database Schema Implementation
**Conflict**: README shows single `atom` table with `children BYTEA[]`, but code uses separate tables
**Evidence**:
- README.md database schema section: single `atom` table with `children BYTEA[]`
- Actual schema (001_schema.sql): separate `atom`, `composition`, `composition_child` tables
- Code consistently uses three-table model in SQL functions and C++ database operations

### 4. Architecture Version Status
**Conflict**: ARCHITECTURE.md marked as "Canonical - 3-Table Schema with Laplacian Eigenmaps (v6)" but README implies older version
**Evidence**:
- ARCHITECTURE.md line 4: "Status: Canonical - 3-Table Schema with Laplacian Eigenmaps (v6)"
- README.md implies unified schema is current

## Decision Changes Catalog

### 5. Table Schema Restructuring
**Old Decision (v5)**: Unified `atom` table containing both leaves and compositions with nested `children BYTEA[]`
**New Decision (v6)**: Normalized three-table schema for better performance and clarity:
- `atom`: Unicode codepoints only
- `composition`: Aggregations with centroids
- `composition_child`: Junction table for ordered relationships
- `relation`: Semantic edges

**Migration Impact**: All SQL functions rewritten, C++ database code updated, old functions dropped with CASCADE

### 6. Eigensolver Implementation Change
**Old Issue**: Conjugate Gradient solver failing at iteration 0 (AUDIT_REPORT.md)
**New Decision**: Use Lanczos eigensolver with deflation for Laplacian eigenmap projection
**Evidence**: Current `laplacian_4d.hpp` uses Lanczos, no CG implementation found

### 7. Unicode Categorization Fix
**Old Bug**: `coordinates.cpp` unicode_blocks lumped punctuation/symbols into single categories
**New Implementation**: Proper splitting of categories (e.g., '(' → PunctuationOpen, '+' → MathSymbol)
**Evidence**: Current `unicode_blocks[]` array has individual entries for proper categorization

## Configuration and Environment Issues

### 8. Database Defaults for Testing
**Issue**: Tests expect user `hartonomous` but PostgreSQL interprets as database name
**Resolution**: Environment variables required: `$env:PGUSER = "postgres"` etc.
**Location**: AUDIT_REPORT.md recommendations (may still be relevant)

## File Organization Changes

### 9. Archive Directory Structure
**Decision**: Deprecated schemas and functions moved to `sql/archive/` with descriptive names:
- `011_unified_atom.sql.DEPRECATED`
- `012_semantic_udf.sql.SUPERSEDED_BY_025`
- Multiple `OLD_SCHEMA` and `DEPRECATED` files

**Pattern**: Files marked with `.DEPRECATED`, `.OLD_SCHEMA`, or `.NEEDS_RENUMBER`

### 10. Build Script Filtering
**Decision**: Scripts automatically exclude archived files: `[[ "$sqlfile" == *"archive"* ]] && continue`
**Location**: `scripts/linux/setup-db.sh` line 103

## Unimplemented Features Catalog

### 11. CLI Integration Gaps
**Missing**: Full CLI integration for query, stats, and testing
**Evidence**: `cpp/src/cli/main.cpp` has TODO comments:
- Line 241: "Query not yet integrated"
- Line 249: "Stats not yet integrated"
- Line 273: "Test runner not yet integrated"

### 12. Batch Token Ingestion
**Missing**: CPE batch ingestion for missing tokens during embedding extraction
**Evidence**: `cpp/src/tools/extract_embeddings.cpp` line 244: "TODO: Batch ingest missing tokens via CPE"

## Summary of Changes

| Type | Count | Status |
|------|-------|--------|
| Documentation Inconsistencies | 4 | Requires update |
| Code Conflicts | 2 | Resolved in code |
| Decision Changes | 4 | Implemented |
| Unimplemented Features | 2 | Minor impact |

**Primary Action Required**: Update README.md to reflect current three-table architecture and remove references to deprecated unified schema.