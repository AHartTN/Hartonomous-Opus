-- =============================================================================
-- AI/ML Operations Infrastructure
-- Status: CURRENT - Core AI operations using spatial substrate
-- =============================================================================
--
-- This file implements AI/ML operations on the Hartonomous hypercube:
-- 1. ATTENTION: Find relevant compositions based on query context
-- 2. TRANSFORMATION: Map between semantic spaces
-- 3. INFERENCE: Derive new knowledge from existing relationships
-- 4. GENERATION: Create new content by walking the semantic space
--
-- All operations leverage:
-- - Hilbert curve proximity for fast neighbor lookup
-- - 4D spatial distance for semantic similarity
-- - Composition structure for context understanding
-- - Content-addressed deduplication for efficiency
--
-- SRID = 0 throughout (raw 4D space)
-- Note: Leaf atoms (depth=0) are POINT geometries, compositions are LINESTRING
-- Note: For compositions (LINESTRING), ST_Centroid gives the true 4D center

BEGIN;

-- =============================================================================
-- Helper: Get representative 4D point from any geometry type
-- =============================================================================

CREATE OR REPLACE FUNCTION get_atom_point(p_geom GEOMETRY)
RETURNS GEOMETRY AS $$
    SELECT ST_Centroid(p_geom);
$$ LANGUAGE SQL IMMUTABLE;

-- Helper: Get 4D coordinates from any atom geometry
CREATE OR REPLACE FUNCTION get_atom_coords(p_geom GEOMETRY)
RETURNS TABLE(x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION) AS $$
    SELECT 
        ST_X(get_atom_point(p_geom)),
        ST_Y(get_atom_point(p_geom)),
        ST_Z(get_atom_point(p_geom)),
        ST_M(get_atom_point(p_geom));
$$ LANGUAGE SQL IMMUTABLE;

-- =============================================================================
-- ATTENTION: Context-aware relevance scoring
-- =============================================================================

-- Self-attention: find compositions most relevant to a query
-- This is analogous to transformer self-attention but uses spatial proximity
CREATE OR REPLACE FUNCTION attention_self(
    p_query_id BYTEA,
    p_context_depth INTEGER DEFAULT 3,
    p_k INTEGER DEFAULT 10
) RETURNS TABLE(
    composition_id BYTEA,
    content TEXT,
    attention_score DOUBLE PRECISION,
    comp_depth INTEGER
) AS $$
DECLARE
    v_query_x DOUBLE PRECISION;
    v_query_y DOUBLE PRECISION;
    v_query_z DOUBLE PRECISION;
    v_query_m DOUBLE PRECISION;
    v_query_depth INTEGER;
BEGIN
    -- Get query position (using representative point)
    SELECT c.x, c.y, c.z, c.m, a.depth
    INTO v_query_x, v_query_y, v_query_z, v_query_m, v_query_depth
    FROM atom a, get_atom_coords(a.geom) c
    WHERE a.id = p_query_id;
    
    IF v_query_x IS NULL THEN
        RAISE EXCEPTION 'Query composition not found';
    END IF;
    
    -- Return compositions with attention scores (inverse distance)
    RETURN QUERY
    SELECT 
        a.id,
        atom_reconstruct_text(a.id),
        -- Attention score: inverse of 4D distance, scaled by depth similarity
        1.0 / (1.0 + sqrt(
            power(c.x - v_query_x, 2) +
            power(c.y - v_query_y, 2) +
            power(c.z - v_query_z, 2) +
            power(c.m - v_query_m, 2)
        )) * (1.0 / (1.0 + ABS(a.depth - v_query_depth))),
        a.depth
    FROM atom a, get_atom_coords(a.geom) c
    WHERE a.id != p_query_id
      AND a.depth BETWEEN v_query_depth - p_context_depth AND v_query_depth + p_context_depth
      AND a.depth > 0  -- Only compositions, not leaf atoms
    ORDER BY 3 DESC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- Cross-attention: find compositions relevant to multiple query terms
CREATE OR REPLACE FUNCTION attention_cross(
    p_query_ids BYTEA[],
    p_k INTEGER DEFAULT 10
) RETURNS TABLE(
    composition_id BYTEA,
    content TEXT,
    total_attention DOUBLE PRECISION,
    contributing_queries INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH query_positions AS (
        SELECT a.id, c.x, c.y, c.z, c.m
        FROM atom a, get_atom_coords(a.geom) c
        WHERE a.id = ANY(p_query_ids)
    ),
    attention_scores AS (
        SELECT 
            a.id as comp_id,
            q.id as query_id,
            1.0 / (1.0 + sqrt(
                power(ac.x - q.x, 2) +
                power(ac.y - q.y, 2) +
                power(ac.z - q.z, 2) +
                power(ac.m - q.m, 2)
            )) as score
        FROM atom a, get_atom_coords(a.geom) ac
        CROSS JOIN query_positions q
        WHERE a.depth > 0
          AND a.id != ALL(p_query_ids)
    )
    SELECT 
        comp_id,
        atom_reconstruct_text(comp_id),
        SUM(score),
        COUNT(DISTINCT query_id)::INTEGER
    FROM attention_scores
    GROUP BY comp_id
    ORDER BY SUM(score) DESC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- TRANSFORMATION: Semantic space mapping
-- =============================================================================

-- Transform: find the analogous composition in a different region
-- "King is to Queen as Man is to ?" -> finds Woman
CREATE OR REPLACE FUNCTION transform_analogy(
    p_source_id BYTEA,      -- e.g., "King"
    p_target_id BYTEA,      -- e.g., "Queen"  
    p_query_id BYTEA,       -- e.g., "Man"
    p_k INTEGER DEFAULT 5
) RETURNS TABLE(
    result_id BYTEA,
    content TEXT,
    similarity DOUBLE PRECISION
) AS $$
DECLARE
    v_delta_x DOUBLE PRECISION;
    v_delta_y DOUBLE PRECISION;
    v_delta_z DOUBLE PRECISION;
    v_delta_m DOUBLE PRECISION;
    v_target_x DOUBLE PRECISION;
    v_target_y DOUBLE PRECISION;
    v_target_z DOUBLE PRECISION;
    v_target_m DOUBLE PRECISION;
BEGIN
    -- Compute transformation vector: target - source (using representative points)
    SELECT 
        tc.x - sc.x,
        tc.y - sc.y,
        tc.z - sc.z,
        tc.m - sc.m
    INTO v_delta_x, v_delta_y, v_delta_z, v_delta_m
    FROM atom s, get_atom_coords(s.geom) sc,
         atom t, get_atom_coords(t.geom) tc
    WHERE s.id = p_source_id AND t.id = p_target_id;
    
    -- Apply transformation to query
    SELECT 
        qc.x + v_delta_x,
        qc.y + v_delta_y,
        qc.z + v_delta_z,
        qc.m + v_delta_m
    INTO v_target_x, v_target_y, v_target_z, v_target_m
    FROM atom q, get_atom_coords(q.geom) qc
    WHERE q.id = p_query_id;
    
    -- Find nearest compositions to the transformed point
    RETURN QUERY
    SELECT 
        a.id,
        atom_reconstruct_text(a.id),
        1.0 / (1.0 + sqrt(
            power(ac.x - v_target_x, 2) +
            power(ac.y - v_target_y, 2) +
            power(ac.z - v_target_z, 2) +
            power(ac.m - v_target_m, 2)
        ))
    FROM atom a, get_atom_coords(a.geom) ac
    WHERE a.id != p_query_id
      AND a.depth > 0
    ORDER BY sqrt(
        power(ac.x - v_target_x, 2) +
        power(ac.y - v_target_y, 2) +
        power(ac.z - v_target_z, 2) +
        power(ac.m - v_target_m, 2)
    )
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- INFERENCE: Derive knowledge from structure
-- =============================================================================

-- Infer relationships between compositions based on shared children
CREATE OR REPLACE FUNCTION infer_related(
    p_id BYTEA,
    p_min_shared INTEGER DEFAULT 1,
    p_limit INTEGER DEFAULT 20
) RETURNS TABLE(
    related_id BYTEA,
    content TEXT,
    shared_children INTEGER,
    relationship_strength DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    WITH target_children AS (
        SELECT unnest(children) as child_id
        FROM atom WHERE id = p_id
    ),
    related AS (
        SELECT 
            a.id,
            COUNT(DISTINCT tc.child_id)::INTEGER as shared_count,
            array_length(a.children, 1) as total_children
        FROM atom a
        CROSS JOIN target_children tc
        WHERE tc.child_id = ANY(a.children)
          AND a.id != p_id
        GROUP BY a.id, a.children
        HAVING COUNT(DISTINCT tc.child_id) >= p_min_shared
    )
    SELECT 
        r.id,
        atom_reconstruct_text(r.id),
        r.shared_count,
        r.shared_count::DOUBLE PRECISION / GREATEST(r.total_children, 1)
    FROM related r
    ORDER BY r.shared_count DESC, 4 DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Infer path between two compositions (conceptual distance)
CREATE OR REPLACE FUNCTION infer_path(
    p_start_id BYTEA,
    p_end_id BYTEA,
    p_max_steps INTEGER DEFAULT 5
) RETURNS TABLE(
    step INTEGER,
    composition_id BYTEA,
    content TEXT,
    distance_to_goal DOUBLE PRECISION
) AS $$
DECLARE
    v_end_x DOUBLE PRECISION;
    v_end_y DOUBLE PRECISION;
    v_end_z DOUBLE PRECISION;
    v_end_m DOUBLE PRECISION;
    v_current_id BYTEA;
    v_step INTEGER := 0;
BEGIN
    -- Get end position using representative point
    SELECT c.x, c.y, c.z, c.m
    INTO v_end_x, v_end_y, v_end_z, v_end_m
    FROM atom a, get_atom_coords(a.geom) c
    WHERE a.id = p_end_id;
    
    v_current_id := p_start_id;
    
    WHILE v_step < p_max_steps AND v_current_id != p_end_id LOOP
        -- Return current step
        RETURN QUERY
        SELECT 
            v_step,
            a.id,
            atom_reconstruct_text(a.id),
            sqrt(
                power(ac.x - v_end_x, 2) +
                power(ac.y - v_end_y, 2) +
                power(ac.z - v_end_z, 2) +
                power(ac.m - v_end_m, 2)
            )
        FROM atom a, get_atom_coords(a.geom) ac
        WHERE a.id = v_current_id;
        
        -- Move to nearest neighbor closer to goal
        SELECT a.id INTO v_current_id
        FROM atom a, get_atom_coords(a.geom) ac
        WHERE a.id != v_current_id
          AND a.depth > 0
          AND sqrt(
              power(ac.x - v_end_x, 2) +
              power(ac.y - v_end_y, 2) +
              power(ac.z - v_end_z, 2) +
              power(ac.m - v_end_m, 2)
          ) < (
              SELECT sqrt(
                  power(cc.x - v_end_x, 2) +
                  power(cc.y - v_end_y, 2) +
                  power(cc.z - v_end_z, 2) +
                  power(cc.m - v_end_m, 2)
              ) FROM atom c, get_atom_coords(c.geom) cc WHERE c.id = v_current_id
          )
        ORDER BY sqrt(
            power(ac.x - v_end_x, 2) +
            power(ac.y - v_end_y, 2) +
            power(ac.z - v_end_z, 2) +
            power(ac.m - v_end_m, 2)
        )
        LIMIT 1;
        
        v_step := v_step + 1;
        
        IF v_current_id IS NULL THEN
            EXIT;
        END IF;
    END LOOP;
    
    -- Add final destination
    IF v_current_id = p_end_id OR v_step = p_max_steps THEN
        RETURN QUERY
        SELECT 
            v_step,
            p_end_id,
            atom_reconstruct_text(p_end_id),
            0.0::DOUBLE PRECISION;
    END IF;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- GENERATION: Create new content
-- =============================================================================

-- Generate by random walk from seed
CREATE OR REPLACE FUNCTION generate_random_walk(
    p_seed_id BYTEA,
    p_steps INTEGER DEFAULT 10,
    p_temperature DOUBLE PRECISION DEFAULT 1.0
) RETURNS TABLE(
    step INTEGER,
    composition_id BYTEA,
    content TEXT
) AS $$
DECLARE
    v_current_id BYTEA;
    v_current_x DOUBLE PRECISION;
    v_current_y DOUBLE PRECISION;
    v_current_z DOUBLE PRECISION;
    v_current_m DOUBLE PRECISION;
    v_step INTEGER := 0;
BEGIN
    v_current_id := p_seed_id;
    
    WHILE v_step < p_steps LOOP
        -- Get current position
        SELECT c.x, c.y, c.z, c.m
        INTO v_current_x, v_current_y, v_current_z, v_current_m
        FROM atom a, get_atom_coords(a.geom) c
        WHERE a.id = v_current_id;
        
        -- Return current step
        RETURN QUERY
        SELECT 
            v_step,
            v_current_id,
            atom_reconstruct_text(v_current_id);
        
        -- Pick random neighbor weighted by proximity
        SELECT a.id INTO v_current_id
        FROM atom a, get_atom_coords(a.geom) ac
        WHERE a.id != v_current_id
          AND a.depth > 0
        ORDER BY 
            -- Temperature controls randomness: higher = more random
            random() * p_temperature + 
            1.0 / (1.0 + sqrt(
                power(ac.x - v_current_x, 2) +
                power(ac.y - v_current_y, 2) +
                power(ac.z - v_current_z, 2) +
                power(ac.m - v_current_m, 2)
            ))
        LIMIT 1;
        
        v_step := v_step + 1;
        
        IF v_current_id IS NULL THEN
            EXIT;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- Generate by directed walk toward a target region
CREATE OR REPLACE FUNCTION generate_directed(
    p_seed_id BYTEA,
    p_target_x DOUBLE PRECISION,
    p_target_y DOUBLE PRECISION,
    p_target_z DOUBLE PRECISION,
    p_target_m DOUBLE PRECISION,
    p_steps INTEGER DEFAULT 10
) RETURNS TABLE(
    step INTEGER,
    composition_id BYTEA,
    content TEXT,
    distance_to_target DOUBLE PRECISION
) AS $$
DECLARE
    v_current_id BYTEA;
    v_step INTEGER := 0;
BEGIN
    v_current_id := p_seed_id;
    
    WHILE v_step < p_steps LOOP
        -- Return current step
        RETURN QUERY
        SELECT 
            v_step,
            v_current_id,
            atom_reconstruct_text(v_current_id),
            sqrt(
                power(ac.x - p_target_x, 2) +
                power(ac.y - p_target_y, 2) +
                power(ac.z - p_target_z, 2) +
                power(ac.m - p_target_m, 2)
            )
        FROM atom a, get_atom_coords(a.geom) ac
        WHERE a.id = v_current_id;
        
        -- Move toward target
        SELECT a.id INTO v_current_id
        FROM atom a, get_atom_coords(a.geom) ac
        WHERE a.id != v_current_id
          AND a.depth > 0
        ORDER BY sqrt(
            power(ac.x - p_target_x, 2) +
            power(ac.y - p_target_y, 2) +
            power(ac.z - p_target_z, 2) +
            power(ac.m - p_target_m, 2)
        )
        LIMIT 1;
        
        v_step := v_step + 1;
        
        IF v_current_id IS NULL THEN
            EXIT;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- Generate continuation from existing composition
CREATE OR REPLACE FUNCTION generate_continuation(
    p_prefix_id BYTEA,
    p_length INTEGER DEFAULT 5
) RETURNS TABLE(
    seq_position INTEGER,
    composition_id BYTEA,
    content TEXT
) AS $$
BEGIN
    -- Start with the prefix
    RETURN QUERY
    SELECT 0::INTEGER, p_prefix_id, atom_reconstruct_text(p_prefix_id);
    
    -- Find compositions that commonly follow this one
    RETURN QUERY
    WITH following AS (
        -- Find compositions that share structure with prefix
        SELECT 
            a.id,
            a.atom_count,
            ROW_NUMBER() OVER (ORDER BY a.hilbert_lo) as rn
        FROM atom a
        WHERE a.depth > 0
          AND a.id != p_prefix_id
          AND a.hilbert_lo > (SELECT hilbert_lo FROM atom WHERE id = p_prefix_id)
        ORDER BY a.hilbert_lo
        LIMIT p_length
    )
    SELECT 
        rn::INTEGER,
        f.id,
        atom_reconstruct_text(f.id)
    FROM following f
    ORDER BY rn;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- CONVENIENCE WRAPPERS
-- =============================================================================

-- Quick attention query by text
CREATE OR REPLACE FUNCTION attend(p_text TEXT, p_k INTEGER DEFAULT 10)
RETURNS TABLE(content TEXT, score DOUBLE PRECISION) AS $$
DECLARE
    v_id BYTEA;
BEGIN
    v_id := atom_find_exact(p_text);
    IF v_id IS NULL THEN
        RAISE NOTICE 'Text not found in database, using nearest match';
        -- Find closest match
        SELECT id INTO v_id FROM atom_search_text(p_text, 1, 100, 1);
    END IF;
    
    IF v_id IS NOT NULL THEN
        RETURN QUERY
        SELECT a.content, a.attention_score
        FROM attention_self(v_id, 3, p_k) a;
    END IF;
END;
$$ LANGUAGE plpgsql STABLE;

-- Quick analogy query
CREATE OR REPLACE FUNCTION analogy(
    p_a TEXT,  -- "King"
    p_b TEXT,  -- "Queen"
    p_c TEXT   -- "Man" -> returns "Woman"
) RETURNS TABLE(answer TEXT, similarity DOUBLE PRECISION) AS $$
DECLARE
    v_a_id BYTEA;
    v_b_id BYTEA;
    v_c_id BYTEA;
BEGIN
    v_a_id := atom_find_exact(p_a);
    v_b_id := atom_find_exact(p_b);
    v_c_id := atom_find_exact(p_c);
    
    IF v_a_id IS NULL OR v_b_id IS NULL OR v_c_id IS NULL THEN
        RAISE NOTICE 'One or more terms not found';
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT t.content, t.similarity
    FROM transform_analogy(v_a_id, v_b_id, v_c_id, 5) t;
END;
$$ LANGUAGE plpgsql STABLE;

-- Quick walk from text
CREATE OR REPLACE FUNCTION walk(p_text TEXT, p_steps INTEGER DEFAULT 5)
RETURNS TABLE(step INTEGER, content TEXT) AS $$
DECLARE
    v_id BYTEA;
BEGIN
    v_id := atom_find_exact(p_text);
    IF v_id IS NULL THEN
        SELECT id INTO v_id FROM atom_search_text(p_text, 1, 100, 1);
    END IF;
    
    IF v_id IS NOT NULL THEN
        RETURN QUERY
        SELECT g.step, g.content
        FROM generate_random_walk(v_id, p_steps, 0.5) g;
    END IF;
END;
$$ LANGUAGE plpgsql STABLE;

COMMIT;

-- =============================================================================
-- Post-transaction: Create indexes for AI operations
-- =============================================================================

-- Index for faster child lookups (used by infer_related)
CREATE INDEX IF NOT EXISTS idx_atom_children_gin ON atom USING GIN(children);

-- NOTE: Run ANALYZE atom manually after bulk data loads for optimal query performance
