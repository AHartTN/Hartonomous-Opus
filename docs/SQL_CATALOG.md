# In-Line SQL Catalog for Hartonomous-Opus Repository

This document catalogs all in-line SQL statements found embedded within source code files across the repository. In-line SQL refers to SQL code embedded in programming languages such as C#, C++, PowerShell, etc., as opposed to dedicated SQL files.

## C# Files

### csharp/HypercubeGenerativeApi/Services/PostgresService.cs

#### Line 53: Connection Check
```sql
SELECT 1
```

#### Lines 78-90: Database Statistics Query
```sql
SELECT 'atoms' as stat_name, COUNT(*) as stat_value FROM atom
UNION ALL
SELECT 'compositions', COUNT(*) FROM composition
UNION ALL
SELECT 'compositions_with_centroid', COUNT(*) FROM composition WHERE centroid IS NOT NULL
UNION ALL
SELECT 'relations', COUNT(*) FROM relation
UNION ALL
SELECT 'attention_edges', COUNT(*) FROM relation WHERE relation_type = 'A'
UNION ALL
SELECT 'sequence_edges', COUNT(*) FROM relation WHERE relation_type = 'S'
```

#### Lines 121-127: Token Existence Check
```sql
SELECT 1
FROM composition
WHERE label = @token
  AND centroid IS NOT NULL
LIMIT 1
```

## PowerShell Files

PowerShell scripts extensively use in-line SQL for database operations via psql commands. Below is a summary of SQL usage by file:

### scripts/windows/validate.ps1
Contains multiple SELECT queries for validation checks:
- Connection test: `SELECT 1`
- Database existence: `SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'`
- Atom count: `SELECT COUNT(*) FROM atom`
- Composition count: `SELECT COUNT(*) FROM composition`
- Relation count: `SELECT COUNT(*) FROM relation`
- Composition child count: `SELECT COUNT(*) FROM composition_child`
- Centroid count: `SELECT COUNT(*) FROM composition WHERE centroid IS NOT NULL`
- GIST index check: `SELECT COUNT(*) FROM pg_indexes WHERE indexname='idx_atom_geom'`
- Hilbert index check: `SELECT COUNT(*) FROM pg_indexes WHERE indexname='idx_atom_hilbert'`
- Sample atom query: `SELECT 'cp=' || codepoint || ', h=(' || hilbert_hi || ',' || hilbert_lo || '), g=(' || ROUND(ST_X(geom)::numeric,2) || ',' || ROUND(ST_Y(geom)::numeric,2) || ',' || ROUND(ST_Z(geom)::numeric,2) || ',' || ROUND(ST_M(geom)::numeric,2) || ')' FROM atom WHERE codepoint = 65`
- Leaf check: `SELECT atom_is_leaf((SELECT id FROM atom WHERE codepoint = 65))`
- Centroid query: `SELECT CONCAT('X=', ROUND(ST_X(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))::numeric,2), ' Y=', ROUND(ST_Y(atom_centroid((SELECT id FROM atom WHERE codepoint = 65)))::numeric,2))`
- Reconstruct text: `SELECT atom_reconstruct_text((SELECT id FROM atom WHERE codepoint = 65))`
- KNN count: `SELECT COUNT(*) FROM atom_knn((SELECT id FROM atom WHERE codepoint = 65), 5)`
- Composition depth: `SELECT COALESCE(MAX(depth), 0) FROM composition`
- Sample compositions: `SELECT '    │ depth=' || depth || ', children=' || child_count || ', id=' || LEFT(encode(id, 'hex'), 16) || '...' FROM composition ORDER BY depth DESC, child_count LIMIT 5`
- Top edges: `SELECT '    │ w=' || ROUND(r.weight::numeric, 4) || ' | ' || LEFT(encode(r.source_id, 'hex'), 12) || '.. ↔ ' || LEFT(encode(r.target_id, 'hex'), 12) || '..' FROM relation r ORDER BY r.weight DESC LIMIT 5`
- Neighbors count: `SELECT COUNT(*) FROM semantic_neighbors((SELECT source_id FROM relation ORDER BY weight DESC LIMIT 1), 5)`

### scripts/windows/setup-db.ps1
- Connection test: `SELECT 1`
- Database existence: `SELECT 1 FROM pg_database WHERE datname='$env:HC_DB_NAME'`
- Existing data checks: `SELECT COUNT(*) FROM composition`, `SELECT COUNT(*) FROM relation`
- Seed atoms: `SELECT seed_atoms();`
- Final counts: `SELECT COUNT(*) FROM atom`, `SELECT COUNT(*) FROM composition`, `SELECT COUNT(*) FROM relation`

### scripts/windows/test.ps1
- Version query: `SELECT version()`
- PostGIS version: `SELECT PostGIS_Lib_Version()`
- Extension checks: `SELECT COUNT(*) FROM pg_extension WHERE extname IN ('hypercube','hypercube_ops','semantic_ops','generative','postgis')`
- Table existence: `SELECT COUNT(*) FROM information_schema.tables WHERE table_name='atom'`
- Index checks: `SELECT COUNT(*) FROM pg_indexes WHERE indexname LIKE 'idx_%'`
- Atom stats: `SELECT COUNT(*) FROM atom`
- Sample queries: `SELECT hilbert_lo, hilbert_hi FROM atom WHERE codepoint = 65`
- Function calls: `SELECT atom_is_leaf(...)`, `SELECT atom_reconstruct_text(...)`, `SELECT atom_knn(...)`
- Composition queries: `SELECT COUNT(*) FROM composition`
- Relation queries: `SELECT COUNT(*) FROM relation`
- Model queries: `SELECT * FROM model LIMIT 5`

### scripts/windows/run_tests.ps1
- Connection test: `SELECT 1`
- Table checks: `SELECT COUNT(*) FROM information_schema.tables WHERE table_name='...'`
- Function tests: `SELECT atom_is_leaf(...)`, `SELECT atom_reconstruct_text(...)`
- Atom count: `SELECT COUNT(*) FROM atom`

### scripts/windows/ingest-testdata.ps1
- Count queries: `SELECT COUNT(*) FROM composition`, `SELECT COUNT(*) FROM atom`, `SELECT COUNT(*) FROM relation`, `SELECT COUNT(*) FROM composition WHERE centroid IS NOT NULL`

### scripts/windows/ingest-models.ps1
- Relation count: `SELECT COUNT(*) FROM relation WHERE source_model = '$($m.Name)'`
- Atom count: `SELECT COUNT(*) FROM atom`
- Composition count: `SELECT COUNT(*) FROM composition`
- Relation count: `SELECT COUNT(*) FROM relation`

### scripts/windows/e2e-test.ps1
- Connection test: `SELECT 1`
- Atom count: `SELECT COUNT(*) FROM atom`
- Extensions: `SELECT COUNT(*) FROM pg_extension WHERE extname IN ('hypercube','hypercube_ops','semantic_ops','generative','postgis')`
- Functions: `SELECT COUNT(*) FROM pg_proc WHERE proname LIKE 'atom_%'`
- Function calls: `SELECT atom_by_codepoint(65)`, `SELECT atom_text(atom_by_codepoint(65))`, `SELECT atom_knn(atom_by_codepoint(65), 10)`, `SELECT attention(atom_by_codepoint(65), 10)`, `SELECT centroid_distance(...)`, `SELECT COUNT(*) FROM atom WHERE hilbert_lo IS NOT NULL`, `SELECT COUNT(*) FROM atom WHERE ST_X(geom) >= 0 AND ST_Y(geom) >= 0`, `SELECT COUNT(DISTINCT codepoint) FROM atom`, `SELECT COUNT(*) FROM atom WHERE codepoint >= 65536`, `SELECT COUNT(*) FROM composition_child cc WHERE NOT EXISTS (SELECT 1 FROM composition c WHERE c.id = cc.composition_id)`, `SELECT * FROM atom_knn(atom_by_codepoint(65), 100)`, `SELECT COUNT(*) FROM atom WHERE codepoint BETWEEN 65 AND 90`

### scripts/windows/setup-all.ps1
- Statistics query with subqueries for counts

### scripts/windows/archive/ingest-safetensor.ps1
- Edge count: `SELECT COUNT(*) FROM relation WHERE source_model = 'centroid_knn'`
- KNN generation: `SELECT generate_knn_edges(10, 'centroid_knn')`

## C++ Files

C++ code uses PQexec and related functions to execute in-line SQL. Below is a summary by file:

### cpp/src/seed_atoms_parallel.cpp
- `SET synchronous_commit = off`
- `SET maintenance_work_mem = '2GB'`
- `SET work_mem = '256MB'`
- `DROP INDEX IF EXISTS idx_atom_geom`
- `DROP INDEX IF EXISTS idx_atom_hilbert`
- `DROP INDEX IF EXISTS idx_atom_codepoint`
- `TRUNCATE atom CASCADE`
- `CREATE INDEX IF NOT EXISTS idx_atom_codepoint ON atom(codepoint)`
- `CREATE INDEX IF NOT EXISTS idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo)`
- `CREATE INDEX IF NOT EXISTS idx_atom_geom ON atom USING GIST(geom)`
- `ANALYZE atom`

### cpp/src/tools/extract_embeddings.cpp
- Transaction commands: `BEGIN`, `COMMIT`
- COPY commands: `COPY relation FROM STDIN WITH (FORMAT text)`
- Statistics queries: `SELECT count(*) as total_edges FROM relation`
- Metadata table creation: `CREATE TABLE IF NOT EXISTS metadata (...)`

### cpp/src/tools/ingest_safetensor_modular.cpp
- `SELECT recompute_composition_centroids()`
- Recursive composition update query

### cpp/src/ingest/compositions.cpp
- `DROP INDEX IF EXISTS idx_comp_label`
- INSERT into composition and composition_child tables
- `CREATE INDEX idx_comp_label ON composition(label)`

### cpp/src/ingest/attention_relations.cpp
- Various CREATE TEMP TABLE and INSERT statements for batch processing

### cpp/src/ingest/tensor_hierarchy.cpp
- Batch SQL for inserting hierarchy data
- UPDATE for child counts

### cpp/src/ingest/projection_db.cpp
- PREPARE statements and EXECUTE for bulk inserts

### cpp/tests/gtest/test_sql_schema.cpp
- Extension checks: `SELECT PostGIS_Version()`
- Table checks: `SELECT column_name FROM information_schema.columns`
- Index checks: `SELECT COUNT(*) FROM pg_indexes`

### cpp/tests/gtest/test_sql_query_api.cpp
- Various SELECT queries for testing database functions

### cpp/src/db/insert.cpp
- Transaction management: `BEGIN`, `COMMIT`, `ROLLBACK`
- COPY commands for bulk insert
- UPDATE for composition centroids

### cpp/src/cli/main.cpp
- `SELECT recompute_composition_centroids()`

### cpp/src/archive/ingest_safetensor_monolith.cpp
- Extensive SQL for bulk operations: CREATE TEMP TABLE, COPY, INSERT, UPDATE, COMMIT

### cpp/src/archive/semantic_ops.cpp
- SPI_execute for internal PostgreSQL function calls

### cpp/src/archive/test_integration_standalone.cpp
- Various validation and test queries

### cpp/src/archive/manifold_project.cpp
- Batch update queries with transactions

### cpp/src/archive/manifold_4d.cpp
- Bulk projection updates

### cpp/src/archive/ingest_safetensor_universal.cpp
- Bulk insert operations with transactions

## Shell Scripts

Shell scripts (.sh) extensively use in-line SQL for database operations via psql commands. Below is a summary of SQL usage by file:

### scripts/validate.sh
Contains multiple SELECT queries for validation:
- Connection test: `SELECT 1`
- PostGIS version: `SELECT PostGIS_Version()`
- Table existence: `SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'atom'`
- Index checks: `SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_geom'`, `SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_hilbert'`
- Atom counts: `SELECT COUNT(*) FROM atom WHERE depth = 0`, `SELECT COUNT(*) FROM atom WHERE codepoint BETWEEN 0 AND 127`
- ID validation: `SELECT COUNT(*) FROM atom WHERE length(id) != 32`
- SRID checks: `SELECT COUNT(*) FROM atom WHERE ST_SRID(geom) != 0`
- Composition counts: `SELECT COUNT(*) FROM atom WHERE depth > 0`
- Children validation: `SELECT COUNT(*) FROM atom WHERE depth > 0 AND children IS NULL`
- Geometry type checks: `SELECT COUNT(*) FROM atom WHERE depth = 0 AND ST_GeometryType(geom) != 'ST_Point'`
- Function tests: `SELECT atom_is_leaf(...)`, `SELECT atom_centroid(...)`, `SELECT atom_distance(...)`, `SELECT atom_reconstruct_text(...)`
- Performance tests: `SELECT COUNT(*) FROM atom a, atom t WHERE t.codepoint = 65 AND ...`, `SELECT COUNT(*) FROM atom_nearest_spatial(...)`
- Statistics: `SELECT COUNT(*) FROM atom WHERE depth = 0`, `SELECT COUNT(*) FROM atom WHERE depth > 0`, `SELECT MAX(depth) FROM atom`, `SELECT pg_size_pretty(pg_total_relation_size('atom'))`

### scripts/setup.sh
- Connection test: `SELECT 1`
- Database existence: `SELECT 1 FROM pg_database WHERE datname='$PGDATABASE'`
- Schema checks: `SELECT 1 FROM information_schema.tables WHERE table_name='composition'`
- Migration checks: `SELECT 1 FROM pg_tables WHERE tablename='relation_edge'`
- Atom counts: `SELECT COUNT(*) FROM atom WHERE depth = 0`
- Function existence: `SELECT 1 FROM pg_proc WHERE proname='get_atoms_by_codepoints'`, `SELECT 1 FROM pg_proc WHERE proname='recompute_composition_centroids'`
- Extension checks: `SELECT 1 FROM pg_proc WHERE proname='hypercube_blake3'`
- Statistics queries with subqueries
- Depth distribution: `SELECT * FROM atom_stats LIMIT 10;`
- Recursive tree queries for visualization
- Ingest function: `SELECT encode(cpe_ingest_document(...), 'hex');`
- KNN queries with complex geometry operations

### scripts/linux/validate.sh
- Connection test: `SELECT 1`
- Database existence: `SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'`
- Counts: `SELECT COUNT(*) FROM atom`, `SELECT COUNT(*) FROM composition`, `SELECT COUNT(*) FROM relation`, `SELECT COUNT(*) FROM composition_child`, `SELECT COUNT(*) FROM composition WHERE centroid IS NOT NULL`

### scripts/linux/test.sh
- Connection tests: `SELECT 1`, `SELECT PostGIS_Version()`
- Table checks: `SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'atom'`
- Index checks: `SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_geom'`, `SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_atom_hilbert'`
- Stats: `SELECT leaf_atoms FROM db_stats()`, `SELECT COUNT(*) FROM atom WHERE depth = 0`, `SELECT * FROM validate_atoms()`
- Composition counts: `SELECT compositions FROM db_stats()`, `SELECT COUNT(*) FROM atom WHERE depth > 0`
- Edge counts: `SELECT COUNT(*) FROM atom WHERE depth = 1 AND atom_count = 2`
- Function tests: `SELECT atom_is_leaf(...)`, `SELECT atom_centroid(...)`, `SELECT atom_reconstruct_text(...)`, `SELECT length(encode(atom_content_hash('hello'), 'hex'))`
- Semantic tests: `SELECT COUNT(*) FROM semantic_neighbors(...)`, `SELECT COUNT(*) FROM attention(...)`
- Existence checks: `SELECT EXISTS(SELECT 1 FROM atom WHERE id = atom_content_hash('A'))`
- Spatial queries: `SELECT COUNT(*) FROM atom_nearest_spatial(...)`
- Hilbert queries: `SELECT COUNT(*) FROM atom a, atom t WHERE t.codepoint = 65 AND a.hilbert_hi = t.hilbert_hi AND a.hilbert_lo BETWEEN ...`
- Summary stats: `SELECT COUNT(*) FILTER (WHERE depth = 0) as "Atoms"`

### scripts/linux/setup-db.sh
- Connection test: `SELECT 1`
- Database checks: `SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'`
- Existing data: `SELECT COUNT(*) FROM composition`, `SELECT COUNT(*) FROM relation`
- Atom seeding: `SELECT seed_atoms();`
- Final counts: `SELECT COUNT(*) FROM atom`

### scripts/linux/setup-all.sh
- Final counts: `SELECT COUNT(*) FROM atom`, `SELECT COUNT(*) FROM composition`, `SELECT COUNT(*) FROM relation`

### scripts/linux/ingest-testdata.sh
- Composition count: `SELECT COUNT(*) FROM atom WHERE depth > 0`
- Edge count: `SELECT COUNT(*) FROM atom WHERE depth = 1 AND atom_count = 2`
- Summary stats: `SELECT COUNT(*) FILTER (WHERE depth = 0) as "Leaf Atoms"`

### scripts/linux/ingest-models.sh
- Counts: `SELECT COUNT(*) FROM atom`, `SELECT COUNT(*) FROM composition`, `SELECT COUNT(*) FROM relation`

## Python Files

No in-line SQL found in Python files.

## Summary

- **C#**: 3 SQL statements in PostgresService.cs for database connectivity and statistics
- **PowerShell**: Extensive use in setup and test scripts, primarily for validation and monitoring
- **C++**: Heavy use in database interaction code, including bulk operations, indexing, and transactions
- **Shell Scripts**: Comprehensive SQL usage in Linux scripts for validation, setup, and testing
- **Total files with in-line SQL**: Approximately 30+ files across C#, PowerShell, C++, and Shell script codebases

This catalog focuses on SQL embedded within application code rather than dedicated SQL migration/schema files.