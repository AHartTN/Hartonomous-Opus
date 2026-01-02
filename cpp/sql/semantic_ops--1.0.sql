-- Semantic Operations Extension SQL definitions
-- High-performance C++ implementations for AI/ML query operations
-- SRID = 0 for all geometries (raw 4D coordinate space)

-- =============================================================================
-- DFS Traversal: Returns all descendants of a composition
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_traverse(
    root_id BYTEA,
    max_depth INTEGER DEFAULT 100
) RETURNS TABLE(
    id BYTEA,
    depth INTEGER,
    ordinal INTEGER,
    path_len INTEGER
)
AS 'semantic_ops', 'semantic_traverse'
LANGUAGE C STABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION semantic_traverse IS 
'DFS traversal of composition DAG. Returns all descendants with depth info.
Uses C++ iterative stack instead of recursive SQL CTE for performance.';

-- =============================================================================
-- Fast Text Reconstruction
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_reconstruct(root_id BYTEA)
RETURNS TEXT
AS 'semantic_ops', 'semantic_reconstruct'
LANGUAGE C STABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION semantic_reconstruct IS
'Reconstruct original text from composition ID.
Uses C++ iterative DFS instead of recursive SQL for 10x+ speedup on deep trees.';

-- =============================================================================
-- 128-bit Hilbert Distance
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_hilbert_distance_128(
    lo1 BIGINT, hi1 BIGINT,
    lo2 BIGINT, hi2 BIGINT
) RETURNS TABLE(dist_lo BIGINT, dist_hi BIGINT)
AS 'semantic_ops', 'semantic_hilbert_distance_128'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION semantic_hilbert_distance_128 IS
'Compute absolute 128-bit Hilbert distance between two indices.
Returns (lo, hi) representing the full 128-bit distance value.';

-- =============================================================================
-- True 4D Euclidean Distance
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_4d_distance(
    x1 DOUBLE PRECISION, y1 DOUBLE PRECISION, z1 DOUBLE PRECISION, m1 DOUBLE PRECISION,
    x2 DOUBLE PRECISION, y2 DOUBLE PRECISION, z2 DOUBLE PRECISION, m2 DOUBLE PRECISION
) RETURNS DOUBLE PRECISION
AS 'semantic_ops', 'semantic_4d_distance'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION semantic_4d_distance IS
'Compute Euclidean distance in 4D space (X, Y, Z, M dimensions).
Uses raw uint32 coordinates stored as doubles (no normalization).';

-- =============================================================================
-- 4D Centroid Computation
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_centroid_4d(
    x DOUBLE PRECISION[],
    y DOUBLE PRECISION[],
    z DOUBLE PRECISION[],
    m DOUBLE PRECISION[]
) RETURNS TABLE(cx DOUBLE PRECISION, cy DOUBLE PRECISION, cz DOUBLE PRECISION, cm DOUBLE PRECISION)
AS 'semantic_ops', 'semantic_centroid_4d'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION semantic_centroid_4d IS
'Compute centroid of multiple 4D points. Handles large uint32 coordinates safely.';

-- =============================================================================
-- Hilbert <-> Coords Conversion
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_coords_from_hilbert(
    lo BIGINT, hi BIGINT
) RETURNS TABLE(x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION)
AS 'semantic_ops', 'semantic_coords_from_hilbert'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION semantic_coords_from_hilbert IS
'Convert 128-bit Hilbert index to 4D coordinates. Inverse of coords_to_hilbert.';

CREATE OR REPLACE FUNCTION semantic_hilbert_from_coords(
    x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION
) RETURNS TABLE(lo BIGINT, hi BIGINT)
AS 'semantic_ops', 'semantic_hilbert_from_coords'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION semantic_hilbert_from_coords IS
'Convert 4D coordinates to 128-bit Hilbert index. Uses raw uint32 values.';

-- =============================================================================
-- Convenience Wrappers
-- =============================================================================

-- Find K nearest neighbors by Hilbert distance (fast)
CREATE OR REPLACE FUNCTION semantic_knn_hilbert(
    target_id BYTEA,
    k INTEGER DEFAULT 10
) RETURNS TABLE(
    neighbor_id BYTEA,
    hilbert_dist_lo BIGINT,
    hilbert_dist_hi BIGINT,
    content TEXT
) AS $$
    WITH target AS (
        SELECT hilbert_lo, hilbert_hi FROM atom WHERE id = target_id
    )
    SELECT 
        a.id,
        d.dist_lo,
        d.dist_hi,
        semantic_reconstruct(a.id)
    FROM atom a, target t,
    LATERAL semantic_hilbert_distance_128(a.hilbert_lo, a.hilbert_hi, t.hilbert_lo, t.hilbert_hi) d
    WHERE a.id != target_id
    ORDER BY d.dist_hi, d.dist_lo
    LIMIT k;
$$ LANGUAGE SQL STABLE;

-- Find K nearest neighbors by 4D Euclidean distance (accurate)
CREATE OR REPLACE FUNCTION semantic_knn_spatial(
    target_id BYTEA,
    k INTEGER DEFAULT 10
) RETURNS TABLE(
    neighbor_id BYTEA,
    distance DOUBLE PRECISION,
    content TEXT
) AS $$
    WITH target AS (
        SELECT ST_X(geom) as x, ST_Y(geom) as y, ST_Z(geom) as z, ST_M(geom) as m
        FROM atom WHERE id = target_id
    )
    SELECT 
        a.id,
        semantic_4d_distance(
            ST_X(a.geom), ST_Y(a.geom), ST_Z(a.geom), ST_M(a.geom),
            t.x, t.y, t.z, t.m
        ),
        semantic_reconstruct(a.id)
    FROM atom a, target t
    WHERE a.id != target_id AND a.depth > 0
    ORDER BY semantic_4d_distance(
        ST_X(a.geom), ST_Y(a.geom), ST_Z(a.geom), ST_M(a.geom),
        t.x, t.y, t.z, t.m
    )
    LIMIT k;
$$ LANGUAGE SQL STABLE;

-- Hilbert range query: find atoms within a Hilbert distance
CREATE OR REPLACE FUNCTION semantic_hilbert_range(
    center_id BYTEA,
    range_lo BIGINT,
    range_hi BIGINT DEFAULT 0,
    max_results INTEGER DEFAULT 100
) RETURNS TABLE(
    id BYTEA,
    hilbert_lo BIGINT,
    hilbert_hi BIGINT,
    depth INTEGER
) AS $$
    WITH center AS (
        SELECT hilbert_lo, hilbert_hi FROM atom WHERE id = center_id
    )
    SELECT a.id, a.hilbert_lo, a.hilbert_hi, a.depth
    FROM atom a, center c
    WHERE 
        -- Same hi region
        a.hilbert_hi = c.hilbert_hi
        -- Within range
        AND a.hilbert_lo BETWEEN c.hilbert_lo - range_lo AND c.hilbert_lo + range_lo
        AND a.id != center_id
    ORDER BY ABS(a.hilbert_lo - c.hilbert_lo)
    LIMIT max_results;
$$ LANGUAGE SQL STABLE;

-- Get composition statistics
CREATE OR REPLACE FUNCTION semantic_composition_stats(p_id BYTEA)
RETURNS TABLE(
    total_nodes BIGINT,
    max_depth INTEGER,
    leaf_count BIGINT,
    composition_count BIGINT,
    reconstructed_text TEXT
) AS $$
    WITH traversal AS (
        SELECT * FROM semantic_traverse(p_id, 50)
    )
    SELECT
        COUNT(*),
        MAX(depth),
        COUNT(*) FILTER (WHERE depth = (SELECT MAX(depth) FROM traversal)),
        COUNT(*) FILTER (WHERE depth > 0),
        semantic_reconstruct(p_id);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- AI/ML Inference Support Functions
-- =============================================================================

-- Attention-like: Find compositions that share sub-compositions
CREATE OR REPLACE FUNCTION semantic_shared_descendants(
    id1 BYTEA,
    id2 BYTEA,
    max_depth INTEGER DEFAULT 5
) RETURNS TABLE(
    shared_id BYTEA,
    depth1 INTEGER,
    depth2 INTEGER
) AS $$
    WITH 
    tree1 AS (SELECT id, depth FROM semantic_traverse(id1, max_depth)),
    tree2 AS (SELECT id, depth FROM semantic_traverse(id2, max_depth))
    SELECT t1.id, t1.depth, t2.depth
    FROM tree1 t1
    JOIN tree2 t2 ON t1.id = t2.id;
$$ LANGUAGE SQL STABLE;

-- Vocabulary frequency: Most common compositions at a given depth
CREATE OR REPLACE FUNCTION semantic_vocabulary(
    p_depth INTEGER DEFAULT 1,
    p_limit INTEGER DEFAULT 50
) RETURNS TABLE(
    id BYTEA,
    content TEXT,
    usage_count BIGINT
) AS $$
    SELECT 
        a.id,
        semantic_reconstruct(a.id),
        (SELECT COUNT(*) FROM atom p WHERE a.id = ANY(p.children))
    FROM atom a
    WHERE a.depth = p_depth
    ORDER BY (SELECT COUNT(*) FROM atom p WHERE a.id = ANY(p.children)) DESC
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Find unknown/novel sequences (compositions used only once)
CREATE OR REPLACE FUNCTION semantic_novel_compositions(
    p_min_atoms INTEGER DEFAULT 3,
    p_max_atoms INTEGER DEFAULT 20,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE(
    id BYTEA,
    content TEXT,
    atom_count BIGINT
) AS $$
    SELECT a.id, semantic_reconstruct(a.id), a.atom_count
    FROM atom a
    WHERE a.depth > 0 
      AND a.atom_count BETWEEN p_min_atoms AND p_max_atoms
      AND NOT EXISTS (SELECT 1 FROM atom p WHERE a.id = ANY(p.children))
    ORDER BY a.atom_count DESC
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Semantic embedding: Get 4D coordinates for an atom/composition
CREATE OR REPLACE FUNCTION semantic_embedding(p_id BYTEA)
RETURNS TABLE(x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION) AS $$
    SELECT ST_X(geom), ST_Y(geom), ST_Z(geom), ST_M(geom)
    FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- Batch embedding lookup
CREATE OR REPLACE FUNCTION semantic_embeddings_batch(p_ids BYTEA[])
RETURNS TABLE(id BYTEA, x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION) AS $$
    SELECT a.id, ST_X(a.geom), ST_Y(a.geom), ST_Z(a.geom), ST_M(a.geom)
    FROM atom a
    WHERE a.id = ANY(p_ids);
$$ LANGUAGE SQL STABLE;
