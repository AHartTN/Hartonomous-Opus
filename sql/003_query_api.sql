-- =============================================================================
-- QUERY API FOR 3-TABLE SCHEMA
-- =============================================================================
-- Generative walks, semantic queries, text reconstruction using
-- Atom/Composition/Relation with 4D Laplacian-projected centroids.
-- =============================================================================

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
-- CENTROID QUERIES: Use 4D Laplacian projections for similarity
-- =============================================================================

-- Get 4D centroid for a labeled composition
CREATE OR REPLACE FUNCTION get_centroid(p_label TEXT)
RETURNS GEOMETRY AS $$
    SELECT centroid FROM composition WHERE label = p_label LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Find similar by 4D centroid distance
CREATE OR REPLACE FUNCTION similar_by_centroid(
    p_label TEXT,
    p_k INTEGER DEFAULT 10
)
RETURNS TABLE(
    label TEXT,
    distance DOUBLE PRECISION
) AS $$
    WITH query_centroid AS (
        SELECT centroid FROM composition 
        WHERE label = p_label AND centroid IS NOT NULL
        LIMIT 1
    )
    SELECT c.label, centroid_distance(c.centroid, q.centroid) as dist
    FROM composition c
    CROSS JOIN query_centroid q
    WHERE c.label IS NOT NULL
      AND c.label != p_label
      AND c.centroid IS NOT NULL
    ORDER BY centroid_distance(c.centroid, q.centroid)
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Short alias
CREATE OR REPLACE FUNCTION similar(p_label TEXT, p_k INTEGER DEFAULT 10)
RETURNS TABLE(label TEXT, distance DOUBLE PRECISION) AS $$
    SELECT * FROM similar_by_centroid(p_label, p_k);
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
-- GENERATIVE WALK: Deterministic walk through relation graph
-- Uses seeded random for reproducibility. Same seed = same walk path.
-- =============================================================================

CREATE OR REPLACE FUNCTION generative_walk(
    p_start_label TEXT,
    p_steps INTEGER DEFAULT 10,
    p_temperature REAL DEFAULT 1.0,
    p_seed DOUBLE PRECISION DEFAULT 0.42  -- Seed for deterministic walks
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
    -- Set seed for reproducible random sequence
    PERFORM setseed(p_seed);
    
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
        
        -- Pick next based on seeded random weighted selection
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
-- SPATIAL WALK: Walk through 4D centroid space
-- =============================================================================

CREATE OR REPLACE FUNCTION spatial_walk(
    p_start_label TEXT,
    p_steps INTEGER DEFAULT 10
)
RETURNS TABLE(
    step INTEGER,
    label TEXT,
    distance DOUBLE PRECISION
) AS $$
DECLARE
    v_current_id BYTEA;
    v_current_centroid GEOMETRY;
    v_next_id BYTEA;
    v_next_centroid GEOMETRY;
    v_label TEXT;
    v_step INTEGER := 0;
    v_dist DOUBLE PRECISION;
BEGIN
    -- Find starting composition
    SELECT c.id, c.centroid, c.label
    INTO v_current_id, v_current_centroid, v_label
    FROM composition c
    WHERE c.label = p_start_label AND c.centroid IS NOT NULL
    LIMIT 1;
    
    IF v_current_id IS NULL THEN
        RAISE NOTICE 'Label not found with centroid: %', p_start_label;
        RETURN;
    END IF;
    
    -- Return starting point
    step := 0; label := v_label; distance := 0.0;
    RETURN NEXT;
    
    WHILE v_step < p_steps LOOP
        v_step := v_step + 1;
        
        -- Find nearest neighbor by 4D centroid distance
        SELECT c.id, c.centroid, c.label, centroid_distance(c.centroid, v_current_centroid)
        INTO v_next_id, v_next_centroid, v_label, v_dist
        FROM composition c
        WHERE c.id != v_current_id
          AND c.label IS NOT NULL
          AND c.centroid IS NOT NULL
        ORDER BY centroid_distance(c.centroid, v_current_centroid)
        OFFSET v_step - 1  -- Skip already-visited neighbors
        LIMIT 1;
        
        IF v_next_id IS NULL THEN
            EXIT;
        END IF;
        
        v_current_id := v_next_id;
        v_current_centroid := v_next_centroid;
        step := v_step; label := v_label; distance := v_dist;
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- ANALOGY: Vector arithmetic on 4D centroids
-- =============================================================================

-- Analogy: A is to B as C is to ?
-- Result = centroid(C) + (centroid(B) - centroid(A))
CREATE OR REPLACE FUNCTION analogy(
    p_a TEXT,  -- e.g., "king"
    p_b TEXT,  -- e.g., "man"
    p_c TEXT,  -- e.g., "queen"
    p_k INTEGER DEFAULT 5
)
RETURNS TABLE(
    answer TEXT,
    distance DOUBLE PRECISION
) AS $$
DECLARE
    v_a GEOMETRY;
    v_b GEOMETRY;
    v_c GEOMETRY;
    v_target GEOMETRY;
BEGIN
    -- Get centroids
    SELECT centroid INTO v_a FROM composition WHERE label = p_a AND centroid IS NOT NULL LIMIT 1;
    SELECT centroid INTO v_b FROM composition WHERE label = p_b AND centroid IS NOT NULL LIMIT 1;
    SELECT centroid INTO v_c FROM composition WHERE label = p_c AND centroid IS NOT NULL LIMIT 1;
    
    IF v_a IS NULL OR v_b IS NULL OR v_c IS NULL THEN
        RAISE WARNING 'One or more tokens not found with centroid';
        RETURN;
    END IF;
    
    -- Compute target: C + (B - A)
    v_target := ST_SetSRID(ST_MakePoint(
        ST_X(v_c) + (ST_X(v_b) - ST_X(v_a)),
        ST_Y(v_c) + (ST_Y(v_b) - ST_Y(v_a)),
        ST_Z(v_c) + (ST_Z(v_b) - ST_Z(v_a)),
        ST_M(v_c) + (ST_M(v_b) - ST_M(v_a))
    ), 0);
    
    -- Find nearest to target
    RETURN QUERY
    SELECT 
        c.label,
        centroid_distance(c.centroid, v_target)
    FROM composition c
    WHERE c.centroid IS NOT NULL
      AND c.label IS NOT NULL
      AND c.label NOT IN (p_a, p_b, p_c)
      AND c.label NOT LIKE '[%'
    ORDER BY centroid_distance(c.centroid, v_target) ASC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- DATABASE STATISTICS
-- =============================================================================

CREATE OR REPLACE FUNCTION db_stats()
RETURNS TABLE(
    atoms BIGINT,
    compositions BIGINT,
    compositions_with_centroid BIGINT,
    relations BIGINT,
    models TEXT[]
) AS $$
    SELECT 
        (SELECT count(*) FROM atom),
        (SELECT count(*) FROM composition),
        (SELECT count(*) FROM composition WHERE centroid IS NOT NULL),
        (SELECT count(*) FROM relation),
        (SELECT array_agg(DISTINCT source_model) FROM relation WHERE source_model IS NOT NULL AND source_model != '');
$$ LANGUAGE SQL STABLE;
