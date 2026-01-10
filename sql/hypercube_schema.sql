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

-- =============================================================================
-- 2. SCHEMA: Tables, indexes, constraints
-- =============================================================================
\i schema/01_tables.sql
\i schema/02_indexes.sql
\i schema/03_constraints.sql

-- =============================================================================
-- 3. FUNCTIONS: Core business logic (ONE FUNCTION PER FILE)
-- =============================================================================

-- Geometry and math functions (foundational)
\i functions/geometry/distance.sql

-- Atom operations (Unicode leaves) - ONE FUNCTION PER FILE
\i functions/atoms/atom_is_leaf.sql
\i functions/atoms/atom_centroid.sql
\i functions/atoms/atom_exists.sql
\i functions/atoms/atom_text.sql
\i functions/atoms/atom_reconstruct_text.sql
\i functions/atoms/atom_knn.sql
\i functions/atoms/lookup.sql
\i functions/atoms/atom_hilbert_range.sql

-- Additional atom functions
\i functions/atoms/atom_by_codepoint.sql
\i functions/atoms/get_atoms_by_codepoints.sql
\i functions/atoms/atom_distance.sql

-- Composition operations (token aggregations)
\i functions/compositions/atom_children.sql
\i functions/compositions/atom_child_count.sql
\i functions/compositions/find_composition.sql
\i functions/compositions/compute_composition_centroid.sql
\i functions/compositions/recompute_composition_centroids.sql

-- Relation operations (semantic graph)
\i functions/relations/semantic_neighbors.sql
\i functions/relations/attention.sql
\i functions/relations/analogy.sql
\i functions/relations/upsert_relation.sql
\i functions/relations/edges.sql

-- Query operations (user-facing)
\i functions/queries/search_text.sql
\i functions/queries/ask.sql
\i functions/queries/encode_prompt.sql
\i functions/queries/score_candidates.sql
\i functions/queries/generate_tokens.sql
\i functions/queries/complete.sql
\i functions/queries/vector_analogy.sql

-- Statistics and reporting
\i functions/stats/db_stats.sql

-- =============================================================================
-- 4. PROCEDURES: Complex multi-step operations (NO in-line SQL)
-- =============================================================================
\i procedures/ingestion/seed_atoms.sql
\i procedures/maintenance/prune_all.sql
\i procedures/maintenance/prune_projections_deduplication.sql
\i procedures/maintenance/prune_projections_quality.sql
\i procedures/maintenance/prune_relations_weight.sql

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