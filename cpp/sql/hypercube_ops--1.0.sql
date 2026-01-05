-- Hypercube Optimized Operations SQL definitions
-- High-performance batch and in-memory graph operations
-- These replace the RBAR SQL functions with efficient C++ implementations

-- =============================================================================
-- Batch Text Reconstruction
-- =============================================================================

CREATE OR REPLACE FUNCTION hypercube_batch_reconstruct(ids BYTEA[])
RETURNS TABLE(
    id BYTEA,
    content TEXT
)
AS 'hypercube_ops', 'hypercube_batch_reconstruct'
LANGUAGE C STABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION hypercube_batch_reconstruct IS
'Reconstruct text for multiple atoms in a single call.
Loads all needed atoms with ONE query, then traverses in-memory.
10-100x faster than calling atom_reconstruct_text in a loop.';

-- =============================================================================
-- Fast Semantic Walk (In-Memory Graph)
-- =============================================================================

CREATE OR REPLACE FUNCTION hypercube_semantic_walk(
    seed_id BYTEA,
    steps INTEGER DEFAULT 10
)
RETURNS TABLE(
    step INTEGER,
    atom_id BYTEA,
    edge_weight DOUBLE PRECISION
)
AS 'hypercube_ops', 'hypercube_semantic_walk'
LANGUAGE C STABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION hypercube_semantic_walk IS
'Random walk through semantic co-occurrence graph.
Loads ALL semantic edges with ONE query, then walks in-memory.
Much faster than the WHILE-loop SQL version.';

-- =============================================================================
-- Fast Semantic Path (In-Memory BFS)
-- =============================================================================

CREATE OR REPLACE FUNCTION hypercube_semantic_path(
    from_id BYTEA,
    to_id BYTEA,
    max_depth INTEGER DEFAULT 6
)
RETURNS TABLE(
    step INTEGER,
    atom_id BYTEA
)
AS 'hypercube_ops', 'hypercube_semantic_path'
LANGUAGE C STABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION hypercube_semantic_path IS
'Find shortest path between atoms via co-occurrence edges.
Uses in-memory BFS after loading edges with ONE query.
Replaces the nested-loop SQL version.';

-- =============================================================================
-- Batch Atom Lookup
-- =============================================================================

CREATE OR REPLACE FUNCTION hypercube_batch_lookup(ids BYTEA[])
RETURNS TABLE(
    id BYTEA,
    depth INTEGER,
    is_leaf BOOLEAN,
    child_count INTEGER,
    centroid_x DOUBLE PRECISION
)
AS 'hypercube_ops', 'hypercube_batch_lookup'
LANGUAGE C STABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION hypercube_batch_lookup IS
'Look up multiple atoms in a single call.
Uses array IN clause for efficient batch fetch.';

-- =============================================================================
-- Convenience Wrappers
-- =============================================================================

-- Wrapper to reconstruct multiple texts from a query
CREATE OR REPLACE FUNCTION reconstruct_all(p_query TEXT)
RETURNS TABLE(id BYTEA, content TEXT) AS $$
BEGIN
    RETURN QUERY
    EXECUTE format(
        'WITH ids AS (%s) 
         SELECT * FROM hypercube_batch_reconstruct(ARRAY(SELECT id FROM ids))',
        p_query
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- Fast version of semantic_walk using C++ implementation
CREATE OR REPLACE FUNCTION fast_walk(p_seed_id BYTEA, p_steps INTEGER DEFAULT 10)
RETURNS TABLE(step INTEGER, atom_id BYTEA, edge_weight DOUBLE PRECISION) AS $$
    SELECT * FROM hypercube_semantic_walk(p_seed_id, p_steps);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- Fast version of semantic_path using C++ implementation
CREATE OR REPLACE FUNCTION fast_path(p_from BYTEA, p_to BYTEA, p_max_depth INTEGER DEFAULT 6)
RETURNS TABLE(step INTEGER, atom_id BYTEA) AS $$
    SELECT * FROM hypercube_semantic_path(p_from, p_to, p_max_depth);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- =============================================================================
-- Fast K-Nearest Neighbors (In-Memory)
-- =============================================================================

CREATE OR REPLACE FUNCTION hypercube_knn_batch(
    target_id BYTEA,
    k INTEGER DEFAULT 10,
    depth_filter INTEGER DEFAULT -1
)
RETURNS TABLE(
    neighbor_id BYTEA,
    distance DOUBLE PRECISION
)
AS 'hypercube_ops', 'hypercube_knn_batch'
LANGUAGE C STABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION hypercube_knn_batch IS
'K-nearest neighbors using 4D Euclidean distance.
Loads all centroids once, computes distances in-memory.
Much faster than ORDER BY with distance function.';

-- =============================================================================
-- Fast Attention Scoring (In-Memory)
-- =============================================================================

CREATE OR REPLACE FUNCTION hypercube_attention(
    target_id BYTEA,
    k INTEGER DEFAULT 10
)
RETURNS TABLE(
    composition_id BYTEA,
    attention_score DOUBLE PRECISION
)
AS 'hypercube_ops', 'hypercube_attention'
LANGUAGE C STABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION hypercube_attention IS
'Compute attention scores for all compositions.
Score = 1/(1+distance) for inverse distance weighting.
Returns top k by attention score.';

-- =============================================================================
-- ATOM SEEDING - High-Performance Unicode Atom Generation
-- =============================================================================

CREATE OR REPLACE FUNCTION seed_atoms()
RETURNS BIGINT
AS 'hypercube_ops', 'seed_atoms'
LANGUAGE C VOLATILE STRICT;

COMMENT ON FUNCTION seed_atoms() IS
'Seed all ~1.1M Unicode codepoint atoms.
Computes BLAKE3 hash, 4D coordinates, and Hilbert index for each.
Idempotent: returns existing count if already seeded.
Usage: SELECT seed_atoms();';

-- Convenience wrappers
CREATE OR REPLACE FUNCTION fast_knn(p_id BYTEA, p_k INTEGER DEFAULT 10)
RETURNS TABLE(id BYTEA, dist DOUBLE PRECISION) AS $$
    SELECT neighbor_id, distance FROM hypercube_knn_batch(p_id, p_k);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

CREATE OR REPLACE FUNCTION fast_attention(p_id BYTEA, p_k INTEGER DEFAULT 10)
RETURNS TABLE(id BYTEA, score DOUBLE PRECISION) AS $$
    SELECT composition_id, attention_score FROM hypercube_attention(p_id, p_k);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

