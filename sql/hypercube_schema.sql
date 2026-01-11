-- =============================================================================
-- HYPERCUBE SCHEMA - ENTERPRISE REFACTORED
-- =============================================================================
-- This is the master schema file that includes all components in proper order.
-- Each component is now in its own focused file for maintainability.
-- ZERO in-line SQL - all operations use proper stored procedures/functions.
-- =============================================================================

-- =============================================================================
-- 1. FOUNDATIONS: Extensions and basic setup
-- =============================================================================
-- Required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS hypercube; -- # Base: BLAKE3, Hilbert, coordinates
CREATE EXTENSION IF NOT EXISTS hypercube_ops; -- # Depends on: hypercube
CREATE EXTENSION IF NOT EXISTS embedding_ops; -- # Depends on: hypercube
CREATE EXTENSION IF NOT EXISTS semantic_ops; --# Depends on: hypercube, embedding_ops
CREATE EXTENSION IF NOT EXISTS generative; --# Depends on: semantic_ops, embedding_ops

-- =============================================================================
-- 2. SCHEMA: Tables, indexes, constraints
-- =============================================================================
\i sql/schema/01_tables.sql
\i sql/schema/02_indexes.sql
\i sql/schema/03_constraints.sql

-- =============================================================================
-- 3. FUNCTIONS: Core business logic (ONE FUNCTION PER FILE)
-- =============================================================================

-- Geometry and math functions (foundational)
\i sql/functions/geometry/distance.sql

-- Atom operations (Unicode leaves) - ONE FUNCTION PER FILE
\i sql/functions/atoms/atom_is_leaf.sql
\i sql/functions/atoms/atom_centroid.sql
\i sql/functions/atoms/atom_exists.sql
\i sql/functions/atoms/atom_text.sql
\i sql/functions/atoms/atom_reconstruct_text.sql
\i sql/functions/atoms/atom_knn.sql
\i sql/functions/atoms/lookup.sql
\i sql/functions/atoms/atom_hilbert_range.sql

-- Additional atom functions
\i sql/functions/atoms/atom_by_codepoint.sql
\i sql/functions/atoms/get_atoms_by_codepoints.sql
\i sql/functions/atoms/atom_distance.sql

-- Composition operations (token aggregations)
\i sql/functions/compositions/atom_children.sql
\i sql/functions/compositions/atom_child_count.sql
\i sql/functions/compositions/find_composition.sql
\i sql/functions/compositions/compute_composition_centroid.sql
\i sql/functions/compositions/recompute_composition_centroids.sql

-- Relation operations (semantic graph)
\i sql/functions/relations/semantic_neighbors.sql
\i sql/functions/relations/attention.sql
\i sql/functions/relations/analogy.sql
\i sql/functions/relations/upsert_relation.sql
\i sql/functions/relations/edges.sql

-- Query operations (user-facing)
\i sql/functions/queries/search_text.sql
\i sql/functions/queries/ask.sql
\i sql/functions/queries/encode_prompt.sql
\i sql/functions/queries/score_candidates.sql
\i sql/functions/queries/generate_tokens.sql
\i sql/functions/queries/complete.sql
\i sql/functions/queries/vector_analogy.sql

-- Statistics and reporting
\i sql/functions/stats/db_stats.sql

-- =============================================================================
-- 4. PROCEDURES: Complex multi-step operations (NO in-line SQL)
-- =============================================================================
\i sql/procedures/ingestion/seed_atoms.sql
\i sql/procedures/maintenance/prune_all.sql
\i sql/procedures/maintenance/prune_projections_deduplication.sql
\i sql/procedures/maintenance/prune_projections_quality.sql
\i sql/procedures/maintenance/prune_relations_weight.sql

-- =============================================================================
-- 5. EXTENSIONS: PostgreSQL C extensions (low-level operations)
-- =============================================================================
-- Extensions are created by CREATE EXTENSION commands in setup-db.ps1
-- The SQL files are loaded automatically by CREATE EXTENSION
-- No \i needed here

-- =============================================================================
-- SETUP COMPLETE - ENTERPRISE GRADE
-- =============================================================================
-- The hypercube database is now properly structured with:
-- • ZERO in-line SQL in application code
-- • One object per file principle (FINALLY!)
-- • Clean separation of concerns
-- • Enterprise-grade maintainability
-- • Proper layered architecture
-- • SRID 0 throughout (no WGS84 pollution)
-- =============================================================================