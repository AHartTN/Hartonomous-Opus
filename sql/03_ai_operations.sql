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
-- Note: We use the pre-computed 'centroid' column for 4D coordinates

BEGIN;

-- =============================================================================
-- Helper: Get 4D coordinates efficiently from centroid column
-- =============================================================================

-- Get 4D coordinates from the pre-computed centroid column (preferred method)
CREATE OR REPLACE FUNCTION get_atom_coords_by_id(p_id BYTEA)
RETURNS TABLE(x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION) AS $$
    SELECT 
        ST_X(centroid),
        ST_Y(centroid),
        ST_Z(centroid),
        ST_M(centroid)
    FROM atom
    WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- For POINT: return the point directly
-- For LINESTRING: compute proper 4D centroid from all vertices
CREATE OR REPLACE FUNCTION get_atom_point(p_geom GEOMETRY)
RETURNS GEOMETRY AS $$
    SELECT CASE 
        WHEN ST_GeometryType(p_geom) = 'ST_Point' THEN p_geom
        ELSE 
            -- For LINESTRING, compute 4D centroid manually
            ST_SetSRID(
                ST_MakePoint(
                    ST_X(ST_Centroid(p_geom)),
                    ST_Y(ST_Centroid(p_geom)),
                    (SELECT AVG(ST_Z(geom)) FROM ST_DumpPoints(p_geom)),
                    (SELECT AVG(ST_M(geom)) FROM ST_DumpPoints(p_geom))
                ),
                0
            )
    END;
$$ LANGUAGE SQL IMMUTABLE;

-- Fallback: Get 4D coordinates from geometry (slower, computes on the fly)
CREATE OR REPLACE FUNCTION get_atom_coords(p_geom GEOMETRY)
RETURNS TABLE(x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION) AS $$
    WITH pt AS (SELECT get_atom_point(p_geom) as p)
    SELECT 
        ST_X(pt.p),
        ST_Y(pt.p),
        ST_Z(pt.p),
        ST_M(pt.p)
    FROM pt;
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
    -- Get query position from centroid column
    SELECT ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid), depth
    INTO v_query_x, v_query_y, v_query_z, v_query_m, v_query_depth
    FROM atom
    WHERE id = p_query_id;
    
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
            power(ST_X(a.centroid) - v_query_x, 2) +
            power(ST_Y(a.centroid) - v_query_y, 2) +
            power(ST_Z(a.centroid) - v_query_z, 2) +
            power(ST_M(a.centroid) - v_query_m, 2)
        )) * (1.0 / (1.0 + ABS(a.depth - v_query_depth))),
        a.depth
    FROM atom a
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
        SELECT a.id, ST_X(a.centroid) as x, ST_Y(a.centroid) as y, 
               ST_Z(a.centroid) as z, ST_M(a.centroid) as m
        FROM atom a
        WHERE a.id = ANY(p_query_ids)
    ),
    attention_scores AS (
        SELECT 
            a.id as comp_id,
            q.id as query_id,
            1.0 / (1.0 + sqrt(
                power(ST_X(a.centroid) - q.x, 2) +
                power(ST_Y(a.centroid) - q.y, 2) +
                power(ST_Z(a.centroid) - q.z, 2) +
                power(ST_M(a.centroid) - q.m, 2)
            )) as score
        FROM atom a
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
    -- Compute transformation vector: target - source (using centroid column)
    SELECT 
        ST_X(t.centroid) - ST_X(s.centroid),
        ST_Y(t.centroid) - ST_Y(s.centroid),
        ST_Z(t.centroid) - ST_Z(s.centroid),
        ST_M(t.centroid) - ST_M(s.centroid)
    INTO v_delta_x, v_delta_y, v_delta_z, v_delta_m
    FROM atom s, atom t
    WHERE s.id = p_source_id AND t.id = p_target_id;
    
    -- Apply transformation to query
    SELECT 
        ST_X(q.centroid) + v_delta_x,
        ST_Y(q.centroid) + v_delta_y,
        ST_Z(q.centroid) + v_delta_z,
        ST_M(q.centroid) + v_delta_m
    INTO v_target_x, v_target_y, v_target_z, v_target_m
    FROM atom q
    WHERE q.id = p_query_id;
    
    -- Find nearest compositions to the transformed point
    RETURN QUERY
    SELECT 
        a.id,
        atom_reconstruct_text(a.id),
        1.0 / (1.0 + sqrt(
            power(ST_X(a.centroid) - v_target_x, 2) +
            power(ST_Y(a.centroid) - v_target_y, 2) +
            power(ST_Z(a.centroid) - v_target_z, 2) +
            power(ST_M(a.centroid) - v_target_m, 2)
        ))
    FROM atom a
    WHERE a.id != p_query_id
      AND a.depth > 0
    ORDER BY sqrt(
        power(ST_X(a.centroid) - v_target_x, 2) +
        power(ST_Y(a.centroid) - v_target_y, 2) +
        power(ST_Z(a.centroid) - v_target_z, 2) +
        power(ST_M(a.centroid) - v_target_m, 2)
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
    -- Get end position from centroid column
    SELECT ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid)
    INTO v_end_x, v_end_y, v_end_z, v_end_m
    FROM atom
    WHERE id = p_end_id;
    
    v_current_id := p_start_id;
    
    WHILE v_step < p_max_steps AND v_current_id != p_end_id LOOP
        -- Return current step
        RETURN QUERY
        SELECT 
            v_step,
            a.id,
            atom_reconstruct_text(a.id),
            sqrt(
                power(ST_X(a.centroid) - v_end_x, 2) +
                power(ST_Y(a.centroid) - v_end_y, 2) +
                power(ST_Z(a.centroid) - v_end_z, 2) +
                power(ST_M(a.centroid) - v_end_m, 2)
            )
        FROM atom a
        WHERE a.id = v_current_id;
        
        -- Move to nearest neighbor closer to goal
        SELECT a.id INTO v_current_id
        FROM atom a
        WHERE a.id != v_current_id
          AND a.depth > 0
          AND sqrt(
              power(ST_X(a.centroid) - v_end_x, 2) +
              power(ST_Y(a.centroid) - v_end_y, 2) +
              power(ST_Z(a.centroid) - v_end_z, 2) +
              power(ST_M(a.centroid) - v_end_m, 2)
          ) < (
              SELECT sqrt(
                  power(ST_X(c.centroid) - v_end_x, 2) +
                  power(ST_Y(c.centroid) - v_end_y, 2) +
                  power(ST_Z(c.centroid) - v_end_z, 2) +
                  power(ST_M(c.centroid) - v_end_m, 2)
              ) FROM atom c WHERE c.id = v_current_id
          )
        ORDER BY sqrt(
            power(ST_X(a.centroid) - v_end_x, 2) +
            power(ST_Y(a.centroid) - v_end_y, 2) +
            power(ST_Z(a.centroid) - v_end_z, 2) +
            power(ST_M(a.centroid) - v_end_m, 2)
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
        -- Get current position from centroid
        SELECT ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid)
        INTO v_current_x, v_current_y, v_current_z, v_current_m
        FROM atom
        WHERE id = v_current_id;
        
        -- Return current step
        RETURN QUERY
        SELECT 
            v_step,
            v_current_id,
            atom_reconstruct_text(v_current_id);
        
        -- Pick random neighbor weighted by proximity
        SELECT a.id INTO v_current_id
        FROM atom a
        WHERE a.id != v_current_id
          AND a.depth > 0
        ORDER BY 
            -- Temperature controls randomness: higher = more random
            random() * p_temperature + 
            1.0 / (1.0 + sqrt(
                power(ST_X(a.centroid) - v_current_x, 2) +
                power(ST_Y(a.centroid) - v_current_y, 2) +
                power(ST_Z(a.centroid) - v_current_z, 2) +
                power(ST_M(a.centroid) - v_current_m, 2)
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
                power(ST_X(a.centroid) - p_target_x, 2) +
                power(ST_Y(a.centroid) - p_target_y, 2) +
                power(ST_Z(a.centroid) - p_target_z, 2) +
                power(ST_M(a.centroid) - p_target_m, 2)
            )
        FROM atom a
        WHERE a.id = v_current_id;
        
        -- Move toward target
        SELECT a.id INTO v_current_id
        FROM atom a
        WHERE a.id != v_current_id
          AND a.depth > 0
        ORDER BY sqrt(
            power(ST_X(a.centroid) - p_target_x, 2) +
            power(ST_Y(a.centroid) - p_target_y, 2) +
            power(ST_Z(a.centroid) - p_target_z, 2) +
            power(ST_M(a.centroid) - p_target_m, 2)
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

-- =============================================================================
-- SEMANTIC EDGE QUERIES: Co-occurrence based relationships
-- =============================================================================

-- Find atoms that co-occur with a given atom (from semantic edges)
-- Semantic edges are stored as depth=1, atom_count=2 compositions
-- with the M coordinate = co-occurrence weight
CREATE OR REPLACE FUNCTION semantic_cooccurrence(
    p_atom_id BYTEA,
    p_k INTEGER DEFAULT 20
) RETURNS TABLE(
    neighbor_id BYTEA,
    neighbor_text TEXT,
    cooccurrence_weight DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CASE WHEN e.children[1] = p_atom_id THEN e.children[2] ELSE e.children[1] END,
        atom_reconstruct_text(
            CASE WHEN e.children[1] = p_atom_id THEN e.children[2] ELSE e.children[1] END
        ),
        ST_M(ST_StartPoint(e.geom))
    FROM atom e
    WHERE e.depth = 1 
      AND e.atom_count = 2
      AND (e.children[1] = p_atom_id OR e.children[2] = p_atom_id)
      AND ST_M(ST_StartPoint(e.geom)) < 100000  -- Semantic edges have low M values
    ORDER BY ST_M(ST_StartPoint(e.geom)) DESC
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- Find the strongest semantic edges in the system
CREATE OR REPLACE FUNCTION semantic_top_edges(
    p_limit INTEGER DEFAULT 50
) RETURNS TABLE(
    atom1_text TEXT,
    atom2_text TEXT,
    weight DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        atom_reconstruct_text(e.children[1]),
        atom_reconstruct_text(e.children[2]),
        ST_M(ST_StartPoint(e.geom))
    FROM atom e
    JOIN atom a1 ON a1.id = e.children[1] AND a1.depth = 0
    JOIN atom a2 ON a2.id = e.children[2] AND a2.depth = 0
    WHERE e.depth = 1 
      AND e.atom_count = 2
      AND ST_M(ST_StartPoint(e.geom)) < 100000  -- Semantic edges
    ORDER BY ST_M(ST_StartPoint(e.geom)) DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Semantic walk: follow co-occurrence edges (with visited tracking)
CREATE OR REPLACE FUNCTION semantic_walk(
    p_seed_id BYTEA,
    p_steps INTEGER DEFAULT 10
) RETURNS TABLE(
    step INTEGER,
    atom_id BYTEA,
    atom_text TEXT,
    edge_weight DOUBLE PRECISION
) AS $$
DECLARE
    v_current_id BYTEA;
    v_next_id BYTEA;
    v_weight DOUBLE PRECISION;
    v_step INTEGER := 0;
    v_visited BYTEA[] := ARRAY[]::BYTEA[];
BEGIN
    v_current_id := p_seed_id;
    
    WHILE v_step < p_steps LOOP
        -- Add current to visited
        v_visited := array_append(v_visited, v_current_id);
        
        -- Return current step
        step := v_step;
        atom_id := v_current_id;
        atom_text := atom_reconstruct_text(v_current_id);
        edge_weight := v_weight;
        RETURN NEXT;
        
        -- Find strongest co-occurrence edge to unvisited node
        SELECT 
            CASE WHEN e.children[1] = v_current_id THEN e.children[2] ELSE e.children[1] END,
            ST_M(ST_StartPoint(e.geom))
        INTO v_next_id, v_weight
        FROM atom e
        WHERE e.depth = 1 
          AND e.atom_count = 2
          AND (e.children[1] = v_current_id OR e.children[2] = v_current_id)
          AND ST_M(ST_StartPoint(e.geom)) < 100000
          AND NOT (CASE WHEN e.children[1] = v_current_id THEN e.children[2] ELSE e.children[1] END = ANY(v_visited))
        ORDER BY ST_M(ST_StartPoint(e.geom)) DESC
        LIMIT 1;
        
        IF v_next_id IS NULL THEN
            EXIT;
        END IF;
        
        v_current_id := v_next_id;
        v_step := v_step + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- Find shortest semantic path between two atoms via co-occurrence edges
CREATE OR REPLACE FUNCTION semantic_path(
    p_from_id BYTEA,
    p_to_id BYTEA,
    p_max_depth INTEGER DEFAULT 6
) RETURNS TABLE(
    step INTEGER,
    atom_id BYTEA,
    atom_text TEXT,
    edge_weight DOUBLE PRECISION
) AS $$
DECLARE
    v_frontier BYTEA[][];
    v_next_frontier BYTEA[][];
    v_visited BYTEA[];
    v_path BYTEA[];
    v_found BOOLEAN := FALSE;
    v_depth INTEGER := 0;
    v_parent BYTEA;
    v_child BYTEA;
    v_weight DOUBLE PRECISION;
    rec RECORD;
BEGIN
    -- BFS initialization
    v_frontier := ARRAY[ARRAY[p_from_id, NULL::BYTEA]];
    v_visited := ARRAY[p_from_id];
    
    WHILE v_depth < p_max_depth AND NOT v_found LOOP
        v_next_frontier := ARRAY[]::BYTEA[][];
        
        FOR i IN 1..array_length(v_frontier, 1) LOOP
            v_parent := v_frontier[i][1];
            
            -- Find all neighbors via co-occurrence edges
            FOR rec IN 
                SELECT 
                    CASE WHEN e.children[1] = v_parent THEN e.children[2] ELSE e.children[1] END as neighbor,
                    ST_M(ST_StartPoint(e.geom)) as w
                FROM atom e
                WHERE e.depth = 1 
                  AND e.atom_count = 2
                  AND (e.children[1] = v_parent OR e.children[2] = v_parent)
                  AND ST_M(ST_StartPoint(e.geom)) < 100000
            LOOP
                IF rec.neighbor = p_to_id THEN
                    -- Found target! Reconstruct path
                    v_found := TRUE;
                    
                    -- Build path from frontier
                    step := 0;
                    atom_id := p_from_id;
                    atom_text := atom_reconstruct_text(p_from_id);
                    edge_weight := NULL;
                    RETURN NEXT;
                    
                    -- Return intermediate steps (simplified - just returns start and end for now)
                    step := v_depth + 1;
                    atom_id := p_to_id;
                    atom_text := atom_reconstruct_text(p_to_id);
                    edge_weight := rec.w;
                    RETURN NEXT;
                    EXIT;
                ELSIF NOT rec.neighbor = ANY(v_visited) THEN
                    v_visited := array_append(v_visited, rec.neighbor);
                    v_next_frontier := v_next_frontier || ARRAY[ARRAY[rec.neighbor, v_parent]];
                END IF;
            END LOOP;
            
            IF v_found THEN EXIT; END IF;
        END LOOP;
        
        v_frontier := v_next_frontier;
        v_depth := v_depth + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- Count semantic edges for an atom
CREATE OR REPLACE FUNCTION semantic_degree(
    p_atom_id BYTEA
) RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*)::INTEGER
        FROM atom e
        WHERE e.depth = 1 
          AND e.atom_count = 2
          AND (e.children[1] = p_atom_id OR e.children[2] = p_atom_id)
          AND ST_M(ST_StartPoint(e.geom)) < 100000
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- Get semantic statistics for an atom
CREATE OR REPLACE FUNCTION semantic_stats(
    p_atom_id BYTEA
) RETURNS TABLE(
    total_edges INTEGER,
    total_weight DOUBLE PRECISION,
    avg_weight DOUBLE PRECISION,
    max_weight DOUBLE PRECISION,
    min_weight DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER,
        SUM(ST_M(ST_StartPoint(e.geom))),
        AVG(ST_M(ST_StartPoint(e.geom))),
        MAX(ST_M(ST_StartPoint(e.geom))),
        MIN(ST_M(ST_StartPoint(e.geom)))
    FROM atom e
    WHERE e.depth = 1 
      AND e.atom_count = 2
      AND (e.children[1] = p_atom_id OR e.children[2] = p_atom_id)
      AND ST_M(ST_StartPoint(e.geom)) < 100000;
END;
$$ LANGUAGE plpgsql STABLE;

-- Find atoms with highest semantic connectivity (most co-occurrence edges)
CREATE OR REPLACE FUNCTION semantic_hubs(
    p_limit INTEGER DEFAULT 20
) RETURNS TABLE(
    atom_id BYTEA,
    atom_text TEXT,
    edge_count BIGINT,
    total_weight DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    WITH edge_counts AS (
        SELECT 
            unnest(e.children) as child_id,
            ST_M(ST_StartPoint(e.geom)) as w
        FROM atom e
        WHERE e.depth = 1 
          AND e.atom_count = 2
          AND ST_M(ST_StartPoint(e.geom)) < 100000
    )
    SELECT 
        ec.child_id,
        atom_reconstruct_text(ec.child_id),
        COUNT(*),
        SUM(ec.w)
    FROM edge_counts ec
    JOIN atom a ON a.id = ec.child_id AND a.depth = 0
    GROUP BY ec.child_id
    ORDER BY COUNT(*) DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Compute semantic similarity between two atoms based on shared neighbors
CREATE OR REPLACE FUNCTION semantic_similarity(
    p_atom1 BYTEA,
    p_atom2 BYTEA
) RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_neighbors1 BYTEA[];
    v_neighbors2 BYTEA[];
    v_intersection INTEGER;
    v_union INTEGER;
BEGIN
    -- Get neighbors for atom1
    SELECT array_agg(CASE WHEN e.children[1] = p_atom1 THEN e.children[2] ELSE e.children[1] END)
    INTO v_neighbors1
    FROM atom e
    WHERE e.depth = 1 
      AND e.atom_count = 2
      AND (e.children[1] = p_atom1 OR e.children[2] = p_atom1)
      AND ST_M(ST_StartPoint(e.geom)) < 100000;
    
    -- Get neighbors for atom2
    SELECT array_agg(CASE WHEN e.children[1] = p_atom2 THEN e.children[2] ELSE e.children[1] END)
    INTO v_neighbors2
    FROM atom e
    WHERE e.depth = 1 
      AND e.atom_count = 2
      AND (e.children[1] = p_atom2 OR e.children[2] = p_atom2)
      AND ST_M(ST_StartPoint(e.geom)) < 100000;
    
    IF v_neighbors1 IS NULL OR v_neighbors2 IS NULL THEN
        RETURN 0.0;
    END IF;
    
    -- Jaccard similarity: |intersection| / |union|
    SELECT COUNT(*) INTO v_intersection
    FROM unnest(v_neighbors1) n1
    WHERE n1 = ANY(v_neighbors2);
    
    v_union := array_length(v_neighbors1, 1) + array_length(v_neighbors2, 1) - v_intersection;
    
    IF v_union = 0 THEN
        RETURN 0.0;
    END IF;
    
    RETURN v_intersection::DOUBLE PRECISION / v_union::DOUBLE PRECISION;
END;
$$ LANGUAGE plpgsql STABLE;

COMMIT;

-- =============================================================================
-- Post-transaction: Create indexes for AI operations
-- =============================================================================

-- Index for faster child lookups (used by infer_related)
CREATE INDEX IF NOT EXISTS idx_atom_children_gin ON atom USING GIN(children);

-- NOTE: Run ANALYZE atom manually after bulk data loads for optimal query performance
