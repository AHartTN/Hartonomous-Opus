-- ============================================================================
-- Embedding Operations PostgreSQL Extension
-- SIMD-accelerated vector operations using AVX2 C++
-- ============================================================================

-- =============================================================================
-- Drop conflicting functions from base SQL files
-- These are replaced by superior cached/SIMD versions below
-- =============================================================================

DROP FUNCTION IF EXISTS similar(TEXT, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS analogy(TEXT, TEXT, TEXT, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS analogy(BYTEA, BYTEA, BYTEA, INTEGER) CASCADE;

-- =============================================================================
-- Type for similarity/analogy results
-- =============================================================================

DROP TYPE IF EXISTS similar_result CASCADE;
CREATE TYPE similar_result AS (
    label TEXT,
    similarity DOUBLE PRECISION
);

-- =============================================================================
-- Core Vector Operations (float4[] input)
-- =============================================================================

-- Cosine similarity between two float arrays (AVX2 accelerated)
CREATE FUNCTION embedding_cosine_sim(float4[], float4[])
RETURNS DOUBLE PRECISION
AS 'MODULE_PATHNAME', 'embedding_cosine_sim'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION embedding_cosine_sim(float4[], float4[]) IS
'Compute cosine similarity between two vectors using SIMD AVX2.
Returns value in range [-1, 1] where 1 = identical, -1 = opposite.';

-- L2 (Euclidean) distance between two float arrays (AVX2 accelerated)
CREATE FUNCTION embedding_l2_dist(float4[], float4[])
RETURNS DOUBLE PRECISION
AS 'MODULE_PATHNAME', 'embedding_l2_dist'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION embedding_l2_dist(float4[], float4[]) IS
'Compute L2 (Euclidean) distance between two vectors using SIMD AVX2.';

-- Vector addition: a + b
CREATE FUNCTION embedding_vector_add(float4[], float4[])
RETURNS float4[]
AS 'MODULE_PATHNAME', 'embedding_vector_add'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION embedding_vector_add(float4[], float4[]) IS
'Element-wise vector addition using SIMD.';

-- Vector subtraction: a - b  
CREATE FUNCTION embedding_vector_sub(float4[], float4[])
RETURNS float4[]
AS 'MODULE_PATHNAME', 'embedding_vector_sub'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION embedding_vector_sub(float4[], float4[]) IS
'Element-wise vector subtraction using SIMD.';

-- Analogy vector: c + (a - b)
CREATE FUNCTION embedding_analogy_vec(float4[], float4[], float4[])
RETURNS float4[]
AS 'MODULE_PATHNAME', 'embedding_analogy_vec'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION embedding_analogy_vec(float4[], float4[], float4[]) IS
'Compute analogy vector: c + (a - b). Used for word analogies.';

-- =============================================================================
-- Cache Management Functions
-- =============================================================================

-- Initialize the embedding cache
CREATE FUNCTION embedding_cache_init()
RETURNS VOID
AS 'MODULE_PATHNAME', 'embedding_cache_init'
LANGUAGE C VOLATILE;

-- Clear the embedding cache
CREATE FUNCTION embedding_cache_clear()
RETURNS VOID
AS 'MODULE_PATHNAME', 'embedding_cache_clear'
LANGUAGE C VOLATILE;

-- Get number of embeddings in cache
CREATE FUNCTION embedding_cache_count()
RETURNS BIGINT
AS 'MODULE_PATHNAME', 'embedding_cache_count'
LANGUAGE C STABLE;

-- Get dimension of cached embeddings
CREATE FUNCTION embedding_cache_dim()
RETURNS BIGINT
AS 'MODULE_PATHNAME', 'embedding_cache_dim'
LANGUAGE C STABLE;

-- Load 4D centroids from composition table into cache
CREATE FUNCTION embedding_cache_load(model_name TEXT DEFAULT NULL)
RETURNS BIGINT
AS 'MODULE_PATHNAME', 'embedding_cache_load'
LANGUAGE C VOLATILE;

COMMENT ON FUNCTION embedding_cache_load(TEXT) IS
'Load all 4D centroids from composition table into C++ memory cache for fast search.
The model_name parameter is ignored (centroids are model-agnostic).
Returns the number of centroids loaded.
Example: SELECT embedding_cache_load()';

-- =============================================================================
-- Cached Similarity Search
-- =============================================================================

-- Find K most similar tokens (uses cache)
CREATE FUNCTION embedding_similar(label TEXT, k INTEGER DEFAULT 10)
RETURNS SETOF similar_result
AS 'MODULE_PATHNAME', 'embedding_similar'
LANGUAGE C STABLE STRICT;

COMMENT ON FUNCTION embedding_similar(TEXT, INTEGER) IS
'Find K most similar tokens using cached 4D centroids (SIMD-accelerated).
Requires embedding_cache_load() to be called first.
Example: SELECT * FROM embedding_similar(''whale'', 10)';

-- =============================================================================
-- Cached Vector Analogy
-- =============================================================================

-- Vector analogy: positive is to negative as query is to ?
CREATE FUNCTION embedding_analogy(
    positive TEXT,
    negative TEXT,
    query TEXT,
    k INTEGER DEFAULT 5
)
RETURNS SETOF similar_result
AS 'MODULE_PATHNAME', 'embedding_analogy'
LANGUAGE C STABLE STRICT;

COMMENT ON FUNCTION embedding_analogy(TEXT, TEXT, TEXT, INTEGER) IS
'Vector analogy using cached 4D centroids (SIMD-accelerated).
Computes: query + (positive - negative) and finds nearest neighbors in 4D space.
Requires embedding_cache_load() to be called first.
Example: SELECT * FROM embedding_analogy(''king'', ''man'', ''woman'', 5)';

-- =============================================================================
-- Convenience Wrappers
-- =============================================================================

-- Extract 4D centroid from composition table as float4[]
CREATE OR REPLACE FUNCTION centroid_to_array(
    entity_id BYTEA
)
RETURNS float4[]
LANGUAGE SQL STABLE AS $$
    SELECT ARRAY[
        ST_X(centroid)::float4,
        ST_Y(centroid)::float4,
        ST_Z(centroid)::float4,
        ST_M(centroid)::float4
    ]
    FROM composition
    WHERE id = $1 AND centroid IS NOT NULL;
$$;

COMMENT ON FUNCTION centroid_to_array(BYTEA) IS
'Extract 4D centroid from composition table as float4[] for SIMD operations.';

-- Quick similar search with auto-cache
CREATE OR REPLACE FUNCTION similar(query_label TEXT, k INTEGER DEFAULT 10)
RETURNS TABLE(label TEXT, similarity DOUBLE PRECISION)
LANGUAGE PLPGSQL STABLE AS $$
BEGIN
    -- Try cached version first
    IF embedding_cache_count() > 0 THEN
        RETURN QUERY SELECT * FROM embedding_similar(query_label, k);
    ELSE
        -- Load cache and retry (model-agnostic 4D centroids)
        PERFORM embedding_cache_load();
        RETURN QUERY SELECT * FROM embedding_similar(query_label, k);
    END IF;
END;
$$;

COMMENT ON FUNCTION similar(TEXT, INTEGER) IS
'Find similar tokens using 4D centroids with auto-cache loading.
Example: SELECT * FROM similar(''whale'', 10)';

-- Quick analogy with auto-cache
CREATE OR REPLACE FUNCTION analogy(
    positive TEXT,
    negative TEXT,
    query TEXT,
    k INTEGER DEFAULT 5
)
RETURNS TABLE(label TEXT, similarity DOUBLE PRECISION)
LANGUAGE PLPGSQL STABLE AS $$
BEGIN
    -- Try cached version first
    IF embedding_cache_count() > 0 THEN
        RETURN QUERY SELECT * FROM embedding_analogy(positive, negative, query, k);
    ELSE
        -- Load cache and retry
        PERFORM embedding_cache_load('minilm');
        RETURN QUERY SELECT * FROM embedding_analogy(positive, negative, query, k);
    END IF;
END;
$$;

COMMENT ON FUNCTION analogy(TEXT, TEXT, TEXT, INTEGER) IS
'Vector analogy with auto-cache loading.
Example: SELECT * FROM analogy(''king'', ''man'', ''woman'', 5)';
