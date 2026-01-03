-- =============================================================================
-- Hartonomous Hypercube - Advanced Semantic Queries
-- =============================================================================
-- Fréchet distance for trajectory similarity (case-insensitive matching)
-- Edge walking for semantic graph traversal
-- Co-occurrence queries
-- =============================================================================

BEGIN;

-- =============================================================================
-- 1. EXACT CONTENT IDENTITY
-- =============================================================================
-- Direct lookup by content hash - strict equality, case-sensitive

-- Already exists: atom_content_hash(text) -> bytea
-- Already exists: atom_find(text) -> bytea

-- Convenience: check if exact content exists
CREATE OR REPLACE FUNCTION content_exists(p_text TEXT)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = atom_content_hash(p_text));
$$ LANGUAGE SQL STABLE;

-- Get composition by exact text with full info
CREATE OR REPLACE FUNCTION content_get(p_text TEXT)
RETURNS TABLE(
    id BYTEA,
    content TEXT,
    depth INTEGER,
    atom_count BIGINT,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    z DOUBLE PRECISION,
    m DOUBLE PRECISION
) AS $$
    SELECT 
        a.id,
        atom_text(a.id),
        a.depth,
        a.atom_count,
        ST_X(a.centroid),
        ST_Y(a.centroid),
        ST_Z(a.centroid),
        ST_M(a.centroid)
    FROM atom a
    WHERE a.id = atom_content_hash(p_text);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- 2. FUZZY TEXT SIMILARITY (Fréchet Distance)
-- =============================================================================
-- Find compositions with similar trajectories
-- Handles: "King" vs "king", "kinestringzm" vs "linestringzm"

-- Find similar by trajectory shape (Fréchet distance)
-- Fréchet similarity - optimized to avoid repeated ST_FrechetDistance calls
CREATE OR REPLACE FUNCTION text_frechet_similar(
    p_text TEXT,
    p_max_distance DOUBLE PRECISION DEFAULT 1e9,
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE(
    id BYTEA,
    content TEXT,
    frechet_distance DOUBLE PRECISION,
    depth INTEGER,
    atom_count BIGINT
) AS $$
    WITH query AS (
        SELECT a.id, a.geom, a.atom_count 
        FROM atom a 
        WHERE a.id = atom_content_hash(p_text)
    ),
    candidates AS (
        SELECT a.id, a.geom, a.depth, a.atom_count
        FROM atom a, query q
        WHERE a.id != q.id
          AND a.depth > 0
          AND a.atom_count BETWEEN GREATEST(1, (q.atom_count * 7) / 10) 
                               AND (q.atom_count * 13) / 10
    ),
    with_distance AS (
        SELECT c.id, c.depth, c.atom_count,
               ST_FrechetDistance(c.geom, q.geom) AS dist
        FROM candidates c, query q
        WHERE ST_FrechetDistance(c.geom, q.geom) < p_max_distance
    )
    SELECT w.id, semantic_reconstruct(w.id), w.dist, w.depth, w.atom_count
    FROM with_distance w
    ORDER BY w.dist
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Short alias
CREATE OR REPLACE FUNCTION similar(p_text TEXT, p_k INTEGER DEFAULT 10)
RETURNS TABLE(content TEXT, distance DOUBLE PRECISION) AS $$
    SELECT content, frechet_distance 
    FROM text_frechet_similar(p_text, 1e10, p_k);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- 3. SEMANTIC NEIGHBORS (Centroid KNN)
-- =============================================================================
-- Find semantically related compositions by centroid proximity
-- Uses C++ extension via fast_knn when available

CREATE OR REPLACE FUNCTION semantic_neighbors(
    p_text TEXT,
    p_k INTEGER DEFAULT 10
)
RETURNS TABLE(
    id BYTEA,
    content TEXT,
    distance DOUBLE PRECISION,
    depth INTEGER
) AS $$
    WITH query AS (
        SELECT a.id, a.centroid FROM atom a WHERE a.id = atom_content_hash(p_text)
    )
    SELECT a.id, semantic_reconstruct(a.id), a.centroid <-> q.centroid, a.depth
    FROM atom a, query q
    WHERE a.id != q.id AND a.depth > 0
    ORDER BY a.centroid <-> q.centroid
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Short alias
CREATE OR REPLACE FUNCTION neighbors(p_text TEXT, p_k INTEGER DEFAULT 10)
RETURNS TABLE(content TEXT, distance DOUBLE PRECISION) AS $$
    SELECT content, distance FROM semantic_neighbors(p_text, p_k);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- 4. EDGE-WALK QUERIES
-- =============================================================================
-- Follow semantic edges to find related content

-- What follows this text? (co-occurrence edges)
CREATE OR REPLACE FUNCTION semantic_follow(
    p_text TEXT,
    p_k INTEGER DEFAULT 20
)
RETURNS TABLE(
    content TEXT,
    weight DOUBLE PRECISION
) AS $$
    WITH qid AS (SELECT atom_content_hash(p_text) AS id)
    SELECT ae.other_text, ae.weight
    FROM qid, atom_edges(qid.id, p_k) ae
    WHERE qid.id IS NOT NULL
    ORDER BY ae.weight DESC;
$$ LANGUAGE SQL STABLE;

-- Short alias
CREATE OR REPLACE FUNCTION follows(p_text TEXT, p_k INTEGER DEFAULT 10)
RETURNS TABLE(content TEXT, weight DOUBLE PRECISION) AS $$
    SELECT * FROM semantic_follow(p_text, p_k);
$$ LANGUAGE SQL STABLE;

-- Recursive walk through the graph - USE C++ EXTENSION
CREATE OR REPLACE FUNCTION semantic_walk(
    p_start TEXT,
    p_steps INTEGER DEFAULT 5
)
RETURNS TABLE(
    step INTEGER,
    content TEXT,
    weight DOUBLE PRECISION
) AS $$
    WITH qid AS (SELECT atom_content_hash(p_start) AS id)
    SELECT sw.step, semantic_reconstruct(sw.atom_id), sw.edge_weight
    FROM qid, hypercube_semantic_walk(qid.id, p_steps) sw
    WHERE qid.id IS NOT NULL;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- 5. ANALOGY QUERIES
-- =============================================================================
-- Vector arithmetic over composition centroids

-- Analogy: A is to B as C is to ?
-- Already exists: atom_analogy(a, b, c, limit)

-- Convenience wrapper with better output
CREATE OR REPLACE FUNCTION analogy(
    p_a TEXT,  -- e.g., "man"
    p_b TEXT,  -- e.g., "king"
    p_c TEXT,  -- e.g., "woman"
    p_k INTEGER DEFAULT 5
)
RETURNS TABLE(
    answer TEXT,
    distance DOUBLE PRECISION
) AS $$
    SELECT content, distance FROM atom_analogy(p_a, p_b, p_c, p_k);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- 6. COMPOUND SIMILARITY (Fréchet + Centroid)
-- =============================================================================
-- Combines trajectory shape AND semantic proximity
-- Optimized to use CTEs instead of variables

CREATE OR REPLACE FUNCTION compound_similar(
    p_text TEXT,
    p_k INTEGER DEFAULT 20,
    p_frechet_weight DOUBLE PRECISION DEFAULT 0.5,
    p_centroid_weight DOUBLE PRECISION DEFAULT 0.5
)
RETURNS TABLE(
    id BYTEA,
    content TEXT,
    combined_score DOUBLE PRECISION,
    frechet_score DOUBLE PRECISION,
    centroid_score DOUBLE PRECISION
) AS $$
    WITH query AS (
        SELECT a.id, a.geom, a.centroid 
        FROM atom a 
        WHERE a.id = atom_content_hash(p_text)
    ),
    -- Sample for normalization (first 1000 compositions)
    sample_stats AS (
        SELECT 
            GREATEST(MAX(ST_FrechetDistance(a.geom, q.geom)), 1.0) AS max_frechet,
            GREATEST(MAX(a.centroid <-> q.centroid), 1.0) AS max_centroid
        FROM (SELECT * FROM atom WHERE depth > 0 LIMIT 1000) a, query q
        WHERE a.id != q.id
    ),
    scored AS (
        SELECT 
            a.id,
            ST_FrechetDistance(a.geom, q.geom) / s.max_frechet AS frechet_norm,
            (a.centroid <-> q.centroid) / s.max_centroid AS centroid_norm
        FROM atom a, query q, sample_stats s
        WHERE a.id != q.id AND a.depth > 0
    )
    SELECT 
        sc.id,
        semantic_reconstruct(sc.id),
        p_frechet_weight * (1.0 - sc.frechet_norm) + p_centroid_weight * (1.0 - sc.centroid_norm),
        1.0 - sc.frechet_norm,
        1.0 - sc.centroid_norm
    FROM scored sc
    ORDER BY p_frechet_weight * (1.0 - sc.frechet_norm) + p_centroid_weight * (1.0 - sc.centroid_norm) DESC
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- 7. DIAGNOSTIC QUERIES
-- =============================================================================

-- Show composition structure
CREATE OR REPLACE FUNCTION composition_info(p_text TEXT)
RETURNS TABLE(
    id TEXT,
    content TEXT,
    depth INTEGER,
    atom_count BIGINT,
    child_count INTEGER,
    centroid_x DOUBLE PRECISION,
    centroid_y DOUBLE PRECISION,
    centroid_z DOUBLE PRECISION,
    centroid_m DOUBLE PRECISION,
    hilbert_lo BIGINT,
    hilbert_hi BIGINT
) AS $$
    SELECT 
        encode(a.id, 'hex'),
        semantic_reconstruct(a.id),
        a.depth,
        a.atom_count,
        COALESCE(array_length(a.children, 1), 0),
        ST_X(a.centroid),
        ST_Y(a.centroid),
        ST_Z(a.centroid),
        ST_M(a.centroid),
        a.hilbert_lo,
        a.hilbert_hi
    FROM atom a
    WHERE a.id = atom_content_hash(p_text);
$$ LANGUAGE SQL STABLE;

-- Count edges for a composition
CREATE OR REPLACE FUNCTION edge_count(p_text TEXT)
RETURNS BIGINT AS $$
    SELECT COUNT(*)
    FROM atom_edges(atom_content_hash(p_text), 10000);
$$ LANGUAGE SQL STABLE;

COMMIT;
