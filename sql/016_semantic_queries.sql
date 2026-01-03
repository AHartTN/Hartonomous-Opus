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
DECLARE
    v_query_id BYTEA;
    v_query_geom GEOMETRY;
    v_query_atoms BIGINT;
BEGIN
    -- Get query composition
    v_query_id := atom_content_hash(p_text);
    IF v_query_id IS NULL THEN
        RETURN;
    END IF;
    
    SELECT geom, atom_count INTO v_query_geom, v_query_atoms
    FROM atom WHERE id = v_query_id;
    
    IF v_query_geom IS NULL THEN
        -- Query not in database, can't compute Fréchet
        RETURN;
    END IF;
    
    -- Find compositions with similar trajectories
    -- Restrict to similar length (±30%) for efficiency
    RETURN QUERY
    SELECT 
        a.id,
        atom_text(a.id),
        ST_FrechetDistance(a.geom, v_query_geom),
        a.depth,
        a.atom_count
    FROM atom a
    WHERE a.id != v_query_id
      AND a.depth > 0
      AND a.atom_count BETWEEN 
          GREATEST(1, (v_query_atoms * 7) / 10) AND 
          (v_query_atoms * 13) / 10
      AND ST_FrechetDistance(a.geom, v_query_geom) < p_max_distance
    ORDER BY ST_FrechetDistance(a.geom, v_query_geom)
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

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
DECLARE
    v_query_id BYTEA;
BEGIN
    v_query_id := atom_content_hash(p_text);
    IF v_query_id IS NULL THEN
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT 
        a.id,
        atom_text(a.id),
        a.centroid <-> (SELECT centroid FROM atom WHERE id = v_query_id),
        a.depth
    FROM atom a
    WHERE a.id != v_query_id
      AND a.depth > 0
    ORDER BY a.centroid <-> (SELECT centroid FROM atom WHERE id = v_query_id)
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

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
DECLARE
    v_id BYTEA;
BEGIN
    v_id := atom_content_hash(p_text);
    IF v_id IS NULL THEN
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT ae.other_text, ae.weight
    FROM atom_edges(v_id, p_k) ae
    ORDER BY ae.weight DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Short alias
CREATE OR REPLACE FUNCTION follows(p_text TEXT, p_k INTEGER DEFAULT 10)
RETURNS TABLE(content TEXT, weight DOUBLE PRECISION) AS $$
    SELECT * FROM semantic_follow(p_text, p_k);
$$ LANGUAGE SQL STABLE;

-- Recursive walk through the graph
CREATE OR REPLACE FUNCTION semantic_walk(
    p_start TEXT,
    p_steps INTEGER DEFAULT 5
)
RETURNS TABLE(
    step INTEGER,
    content TEXT,
    weight DOUBLE PRECISION
) AS $$
DECLARE
    v_start_id BYTEA;
BEGIN
    v_start_id := atom_content_hash(p_start);
    IF v_start_id IS NULL THEN
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT aw.step, aw.node_text, aw.edge_weight
    FROM atom_walk(v_start_id, p_steps) aw;
END;
$$ LANGUAGE plpgsql STABLE;

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
DECLARE
    v_query_id BYTEA;
    v_query_geom GEOMETRY;
    v_query_centroid GEOMETRY;
    v_max_frechet DOUBLE PRECISION;
    v_max_centroid DOUBLE PRECISION;
BEGIN
    v_query_id := atom_content_hash(p_text);
    IF v_query_id IS NULL THEN
        RETURN;
    END IF;
    
    SELECT geom, centroid INTO v_query_geom, v_query_centroid
    FROM atom WHERE id = v_query_id;
    
    IF v_query_geom IS NULL THEN
        RETURN;
    END IF;
    
    -- Get max values for normalization
    SELECT MAX(ST_FrechetDistance(a.geom, v_query_geom)),
           MAX(a.centroid <-> v_query_centroid)
    INTO v_max_frechet, v_max_centroid
    FROM atom a
    WHERE a.id != v_query_id AND a.depth > 0
    LIMIT 1000;
    
    IF v_max_frechet IS NULL OR v_max_frechet = 0 THEN v_max_frechet := 1; END IF;
    IF v_max_centroid IS NULL OR v_max_centroid = 0 THEN v_max_centroid := 1; END IF;
    
    RETURN QUERY
    SELECT 
        a.id,
        atom_text(a.id),
        p_frechet_weight * (1.0 - ST_FrechetDistance(a.geom, v_query_geom) / v_max_frechet) +
        p_centroid_weight * (1.0 - (a.centroid <-> v_query_centroid) / v_max_centroid),
        1.0 - ST_FrechetDistance(a.geom, v_query_geom) / v_max_frechet,
        1.0 - (a.centroid <-> v_query_centroid) / v_max_centroid
    FROM atom a
    WHERE a.id != v_query_id
      AND a.depth > 0
    ORDER BY 
        p_frechet_weight * (1.0 - ST_FrechetDistance(a.geom, v_query_geom) / v_max_frechet) +
        p_centroid_weight * (1.0 - (a.centroid <-> v_query_centroid) / v_max_centroid) DESC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

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
        atom_text(a.id),
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
