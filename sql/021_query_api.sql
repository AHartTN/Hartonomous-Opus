-- =============================================================================
-- QUERY API FOR 4-TABLE SCHEMA
-- =============================================================================
-- Generative walks, semantic queries, text reconstruction - all for the new
-- Atom/Composition/Relation/Shape architecture.
-- =============================================================================

BEGIN;

-- =============================================================================
-- TEXT RECONSTRUCTION: Decode compositions back to readable text
-- =============================================================================

-- Reconstruct text from a composition by walking its children
CREATE OR REPLACE FUNCTION composition_text(p_id BYTEA)
RETURNS TEXT AS $$
WITH RECURSIVE tree AS (
    -- Start with the composition's children
    SELECT cc.child_id, cc.child_type, cc.ordinal, ARRAY[cc.ordinal] as path
    FROM composition_child cc
    WHERE cc.composition_id = p_id
    
    UNION ALL
    
    -- Recurse into child compositions
    SELECT cc.child_id, cc.child_type, cc.ordinal, t.path || cc.ordinal
    FROM tree t
    JOIN composition_child cc ON cc.composition_id = t.child_id
    WHERE t.child_type = 'C'
      AND array_length(t.path, 1) < 50  -- Safety limit
)
SELECT string_agg(convert_from(a.value, 'UTF8'), '' ORDER BY t.path)
FROM tree t
JOIN atom a ON a.id = t.child_id
WHERE t.child_type = 'A';
$$ LANGUAGE SQL STABLE;

-- Short alias
CREATE OR REPLACE FUNCTION text(p_id BYTEA)
RETURNS TEXT AS $$
    SELECT composition_text(p_id);
$$ LANGUAGE SQL STABLE;

-- Find composition by label (e.g., 'whale' from MiniLM vocab)
CREATE OR REPLACE FUNCTION find_composition(p_label TEXT)
RETURNS BYTEA AS $$
    SELECT id FROM composition WHERE label = p_label LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- SHAPE QUERIES: Use embeddings for similarity
-- =============================================================================

-- Get embedding for a labeled composition
CREATE OR REPLACE FUNCTION get_embedding(p_label TEXT, p_model TEXT DEFAULT 'minilm')
RETURNS GEOMETRY AS $$
    SELECT s.embedding
    FROM shape s
    JOIN composition c ON c.id = s.entity_id
    WHERE c.label = p_label AND s.model_name = p_model;
$$ LANGUAGE SQL STABLE;

-- Find similar by embedding distance (using shapes)
CREATE OR REPLACE FUNCTION similar_by_embedding(
    p_label TEXT,
    p_model TEXT DEFAULT 'minilm',
    p_k INTEGER DEFAULT 10
)
RETURNS TABLE(
    label TEXT,
    distance DOUBLE PRECISION
) AS $$
    WITH query_shape AS (
        SELECT s.embedding
        FROM shape s
        JOIN composition c ON c.id = s.entity_id
        WHERE c.label = p_label AND s.model_name = p_model
        LIMIT 1
    )
    SELECT c.label, s.embedding <-> q.embedding as dist
    FROM shape s
    JOIN composition c ON c.id = s.entity_id
    CROSS JOIN query_shape q
    WHERE c.label IS NOT NULL
      AND c.label != p_label
      AND s.model_name = p_model
    ORDER BY s.embedding <-> q.embedding
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Short alias
CREATE OR REPLACE FUNCTION similar(p_label TEXT, p_k INTEGER DEFAULT 10)
RETURNS TABLE(label TEXT, distance DOUBLE PRECISION) AS $$
    SELECT * FROM similar_by_embedding(p_label, 'minilm', p_k);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- RELATION QUERIES: Navigate the knowledge graph
-- =============================================================================

-- Find related entities via attention edges
CREATE OR REPLACE FUNCTION related_by_attention(
    p_label TEXT,
    p_k INTEGER DEFAULT 20,
    p_min_weight REAL DEFAULT 0.1
)
RETURNS TABLE(
    related_label TEXT,
    weight REAL,
    relation_type CHAR(1)
) AS $$
    WITH source AS (
        SELECT id FROM composition WHERE label = p_label LIMIT 1
    )
    SELECT c2.label, r.weight, r.relation_type
    FROM relation r
    JOIN source s ON r.source_id = s.id
    JOIN composition c2 ON r.target_id = c2.id
    WHERE r.weight >= p_min_weight
      AND c2.label IS NOT NULL
    ORDER BY r.weight DESC
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Short alias
CREATE OR REPLACE FUNCTION related(p_label TEXT, p_k INTEGER DEFAULT 10)
RETURNS TABLE(label TEXT, weight REAL) AS $$
    SELECT related_label, weight FROM related_by_attention(p_label, p_k, 0.0);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- GENERATIVE WALK: Random walk through relation graph
-- =============================================================================

CREATE OR REPLACE FUNCTION generative_walk(
    p_start_label TEXT,
    p_steps INTEGER DEFAULT 10,
    p_temperature REAL DEFAULT 1.0
)
RETURNS TABLE(
    step INTEGER,
    label TEXT,
    transition_weight REAL
) AS $$
DECLARE
    v_current_id BYTEA;
    v_next_id BYTEA;
    v_weight REAL;
    v_step INTEGER := 0;
    v_label TEXT;
BEGIN
    -- Find starting composition
    SELECT id, p_start_label INTO v_current_id, v_label
    FROM composition WHERE label = p_start_label LIMIT 1;
    
    IF v_current_id IS NULL THEN
        RAISE NOTICE 'Label not found: %', p_start_label;
        RETURN;
    END IF;
    
    -- Return starting point
    step := 0; label := v_label; transition_weight := 1.0;
    RETURN NEXT;
    
    WHILE v_step < p_steps LOOP
        v_step := v_step + 1;
        
        -- Pick next based on weighted random from relations
        SELECT r.target_id, c.label, r.weight
        INTO v_next_id, v_label, v_weight
        FROM relation r
        JOIN composition c ON r.target_id = c.id
        WHERE r.source_id = v_current_id
          AND c.label IS NOT NULL
          AND c.id != v_current_id
        ORDER BY random() * p_temperature + r.weight DESC
        LIMIT 1;
        
        IF v_next_id IS NULL THEN
            -- No outgoing edges, try reverse direction
            SELECT r.source_id, c.label, r.weight
            INTO v_next_id, v_label, v_weight
            FROM relation r
            JOIN composition c ON r.source_id = c.id
            WHERE r.target_id = v_current_id
              AND c.label IS NOT NULL
              AND c.id != v_current_id
            ORDER BY random() * p_temperature + r.weight DESC
            LIMIT 1;
        END IF;
        
        IF v_next_id IS NULL THEN
            EXIT;  -- Dead end
        END IF;
        
        v_current_id := v_next_id;
        step := v_step; label := v_label; transition_weight := v_weight;
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- SPATIAL WALK: Walk through embedding space
-- =============================================================================

CREATE OR REPLACE FUNCTION spatial_walk(
    p_start_label TEXT,
    p_steps INTEGER DEFAULT 10,
    p_model TEXT DEFAULT 'minilm'
)
RETURNS TABLE(
    step INTEGER,
    label TEXT,
    distance DOUBLE PRECISION
) AS $$
DECLARE
    v_current_id BYTEA;
    v_current_emb GEOMETRY;
    v_next_id BYTEA;
    v_next_emb GEOMETRY;
    v_label TEXT;
    v_step INTEGER := 0;
    v_dist DOUBLE PRECISION;
BEGIN
    -- Find starting composition
    SELECT c.id, s.embedding, c.label
    INTO v_current_id, v_current_emb, v_label
    FROM composition c
    JOIN shape s ON s.entity_id = c.id
    WHERE c.label = p_start_label AND s.model_name = p_model
    LIMIT 1;
    
    IF v_current_id IS NULL THEN
        RAISE NOTICE 'Label not found: %', p_start_label;
        RETURN;
    END IF;
    
    -- Return starting point
    step := 0; label := v_label; distance := 0.0;
    RETURN NEXT;
    
    WHILE v_step < p_steps LOOP
        v_step := v_step + 1;
        
        -- Find nearest neighbor we haven't visited (simplified: just nearest)
        SELECT c.id, s.embedding, c.label, s.embedding <-> v_current_emb
        INTO v_next_id, v_next_emb, v_label, v_dist
        FROM composition c
        JOIN shape s ON s.entity_id = c.id
        WHERE c.id != v_current_id
          AND c.label IS NOT NULL
          AND s.model_name = p_model
        ORDER BY s.embedding <-> v_current_emb
        OFFSET v_step - 1  -- Skip already-visited neighbors
        LIMIT 1;
        
        IF v_next_id IS NULL THEN
            EXIT;
        END IF;
        
        v_current_id := v_next_id;
        v_current_emb := v_next_emb;
        step := v_step; label := v_label; distance := v_dist;
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- ANALOGY: Vector arithmetic on embeddings
-- =============================================================================

-- Analogy: A is to B as C is to ?
-- Result = embedding(C) + (embedding(B) - embedding(A))
CREATE OR REPLACE FUNCTION analogy(
    p_a TEXT,  -- e.g., "king"
    p_b TEXT,  -- e.g., "man"
    p_c TEXT,  -- e.g., "queen"
    p_k INTEGER DEFAULT 5,
    p_model TEXT DEFAULT 'minilm'
)
RETURNS TABLE(
    answer TEXT,
    distance DOUBLE PRECISION
) AS $$
    -- Get the three embeddings as arrays of points
    WITH emb_a AS (
        SELECT s.embedding FROM shape s
        JOIN composition c ON c.id = s.entity_id
        WHERE c.label = p_a AND s.model_name = p_model
    ),
    emb_b AS (
        SELECT s.embedding FROM shape s
        JOIN composition c ON c.id = s.entity_id
        WHERE c.label = p_b AND s.model_name = p_model
    ),
    emb_c AS (
        SELECT s.embedding FROM shape s
        JOIN composition c ON c.id = s.entity_id
        WHERE c.label = p_c AND s.model_name = p_model
    ),
    -- Compute target: C + (B - A) ≈ midpoint between C and B, shifted away from A
    -- Simplified: find words closest to C that are also close to B but far from A
    scores AS (
        SELECT 
            c.label,
            s.embedding <-> (SELECT embedding FROM emb_c) as dist_c,
            s.embedding <-> (SELECT embedding FROM emb_b) as dist_b,
            s.embedding <-> (SELECT embedding FROM emb_a) as dist_a
        FROM shape s
        JOIN composition c ON c.id = s.entity_id
        WHERE c.label IS NOT NULL
          AND c.label NOT IN (p_a, p_b, p_c)
          AND s.model_name = p_model
    )
    SELECT label, (dist_c + dist_b - dist_a * 0.5) as score
    FROM scores
    ORDER BY (dist_c + dist_b - dist_a * 0.5)
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- DATABASE STATISTICS
-- =============================================================================

CREATE OR REPLACE FUNCTION db_stats()
RETURNS TABLE(
    atoms BIGINT,
    compositions BIGINT,
    relations BIGINT,
    shapes BIGINT,
    models TEXT[]
) AS $$
    SELECT 
        (SELECT count(*) FROM atom),
        (SELECT count(*) FROM composition),
        (SELECT count(*) FROM relation),
        (SELECT count(*) FROM shape),
        (SELECT array_agg(DISTINCT model_name) FROM shape);
$$ LANGUAGE SQL STABLE;

COMMIT;
