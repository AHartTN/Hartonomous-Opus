-- =============================================================================
-- Hartonomous Hypercube - AI Operations v4
-- =============================================================================
-- High-level AI operations. Complex graph algorithms use C extensions.
-- These SQL versions are fallbacks when C extension is unavailable.
-- =============================================================================

BEGIN;

-- =============================================================================
-- Semantic Edge Queries
-- =============================================================================
-- Semantic edges are stored as depth=1, atom_count=2 compositions
-- M coordinate encodes co-occurrence weight

-- Get co-occurrences for an atom
CREATE OR REPLACE FUNCTION semantic_neighbors(p_id BYTEA, p_k INTEGER DEFAULT 20)
RETURNS TABLE(neighbor_id BYTEA, weight DOUBLE PRECISION) AS $$
    SELECT 
        CASE WHEN children[1] = p_id THEN children[2] ELSE children[1] END,
        ST_M(ST_StartPoint(geom))
    FROM atom
    WHERE depth = 1 AND atom_count = 2
      AND (children[1] = p_id OR children[2] = p_id)
    ORDER BY ST_M(ST_StartPoint(geom)) DESC
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Count semantic edges
CREATE OR REPLACE FUNCTION semantic_degree(p_id BYTEA)
RETURNS INTEGER AS $$
    SELECT COUNT(*)::INTEGER
    FROM atom
    WHERE depth = 1 AND atom_count = 2
      AND (children[1] = p_id OR children[2] = p_id);
$$ LANGUAGE SQL STABLE;

-- Top semantic edges by weight
CREATE OR REPLACE FUNCTION semantic_top_edges(p_limit INTEGER DEFAULT 50)
RETURNS TABLE(atom1 BYTEA, atom2 BYTEA, weight DOUBLE PRECISION) AS $$
    SELECT children[1], children[2], ST_M(ST_StartPoint(geom))
    FROM atom
    WHERE depth = 1 AND atom_count = 2
    ORDER BY ST_M(ST_StartPoint(geom)) DESC
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Attention (Inverse Distance Scoring)
-- =============================================================================

-- Self-attention: find related compositions
CREATE OR REPLACE FUNCTION attention(p_id BYTEA, p_k INTEGER DEFAULT 10)
RETURNS TABLE(comp_id BYTEA, score DOUBLE PRECISION) AS $$
    WITH target AS (SELECT centroid, depth FROM atom WHERE id = p_id)
    SELECT 
        a.id,
        1.0 / (1.0 + (a.centroid <-> t.centroid))
    FROM atom a, target t
    WHERE a.id != p_id AND a.depth > 0
    ORDER BY a.centroid <-> t.centroid
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Analogy (Vector Arithmetic)
-- =============================================================================

-- A:B :: C:? -> find D
CREATE OR REPLACE FUNCTION analogy(p_a BYTEA, p_b BYTEA, p_c BYTEA, p_k INTEGER DEFAULT 5)
RETURNS TABLE(result_id BYTEA, similarity DOUBLE PRECISION) AS $$
    WITH vecs AS (
        SELECT 
            (SELECT centroid FROM atom WHERE id = p_a) AS a,
            (SELECT centroid FROM atom WHERE id = p_b) AS b,
            (SELECT centroid FROM atom WHERE id = p_c) AS c
    ),
    target AS (
        SELECT ST_SetSRID(ST_MakePoint(
            ST_X(v.c) + ST_X(v.b) - ST_X(v.a),
            ST_Y(v.c) + ST_Y(v.b) - ST_Y(v.a),
            ST_Z(v.c) + ST_Z(v.b) - ST_Z(v.a),
            ST_M(v.c) + ST_M(v.b) - ST_M(v.a)
        ), 0) AS pt
        FROM vecs v
    )
    SELECT a.id, 1.0 / (1.0 + (a.centroid <-> t.pt))
    FROM atom a, target t
    WHERE a.id NOT IN (p_a, p_b, p_c) AND a.depth > 0
    ORDER BY a.centroid <-> t.pt
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Random Walk (SQL fallback - C version is much faster)
-- =============================================================================

CREATE OR REPLACE FUNCTION random_walk(p_seed BYTEA, p_steps INTEGER DEFAULT 10)
RETURNS TABLE(step INTEGER, node_id BYTEA) AS $$
DECLARE
    v_current BYTEA := p_seed;
    v_visited BYTEA[] := ARRAY[p_seed];
    v_step INTEGER := 0;
BEGIN
    WHILE v_step < p_steps LOOP
        step := v_step;
        node_id := v_current;
        RETURN NEXT;
        
        -- Pick random neighbor
        SELECT n.neighbor_id INTO v_current
        FROM semantic_neighbors(v_current, 50) n
        WHERE NOT (n.neighbor_id = ANY(v_visited))
        ORDER BY random() * (1.0 + n.weight)
        LIMIT 1;
        
        IF v_current IS NULL THEN EXIT; END IF;
        v_visited := array_append(v_visited, v_current);
        v_step := v_step + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Path Finding (SQL fallback - use C for production)
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_path(p_from BYTEA, p_to BYTEA, p_max_depth INTEGER DEFAULT 6)
RETURNS TABLE(step INTEGER, node_id BYTEA) AS $$
DECLARE
    v_frontier BYTEA[];
    v_parents BYTEA[];
    v_visited BYTEA[];
    v_depth INTEGER := 0;
    v_found BOOLEAN := FALSE;
    v_next BYTEA;
    v_current BYTEA;
BEGIN
    v_frontier := ARRAY[p_from];
    v_parents := ARRAY[p_from];
    v_visited := ARRAY[p_from];
    
    WHILE v_depth < p_max_depth AND NOT v_found LOOP
        FOR i IN 1..array_length(v_frontier, 1) LOOP
            v_current := v_frontier[i];
            
            FOR v_next IN 
                SELECT n.neighbor_id 
                FROM semantic_neighbors(v_current, 100) n
                WHERE NOT (n.neighbor_id = ANY(v_visited))
            LOOP
                IF v_next = p_to THEN
                    v_found := TRUE;
                    EXIT;
                END IF;
                v_visited := array_append(v_visited, v_next);
            END LOOP;
            
            IF v_found THEN EXIT; END IF;
        END LOOP;
        
        v_depth := v_depth + 1;
    END LOOP;
    
    -- Return simple path (start and end)
    step := 0; node_id := p_from; RETURN NEXT;
    IF v_found THEN
        step := v_depth; node_id := p_to; RETURN NEXT;
    END IF;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Shared Children (Relationship Inference)
-- =============================================================================

CREATE OR REPLACE FUNCTION related_by_children(p_id BYTEA, p_min_shared INTEGER DEFAULT 1, p_limit INTEGER DEFAULT 20)
RETURNS TABLE(related_id BYTEA, shared_count INTEGER) AS $$
    WITH my_children AS (SELECT unnest(children) AS child FROM atom WHERE id = p_id)
    SELECT a.id, COUNT(*)::INTEGER
    FROM atom a
    JOIN my_children mc ON mc.child = ANY(a.children)
    WHERE a.id != p_id
    GROUP BY a.id
    HAVING COUNT(*) >= p_min_shared
    ORDER BY COUNT(*) DESC
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Jaccard Similarity
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_jaccard(p_a BYTEA, p_b BYTEA)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    v_n1 BYTEA[];
    v_n2 BYTEA[];
    v_inter INTEGER;
    v_union INTEGER;
BEGIN
    SELECT array_agg(neighbor_id) INTO v_n1 FROM semantic_neighbors(p_a, 100);
    SELECT array_agg(neighbor_id) INTO v_n2 FROM semantic_neighbors(p_b, 100);
    
    IF v_n1 IS NULL OR v_n2 IS NULL THEN RETURN 0.0; END IF;
    
    SELECT COUNT(*) INTO v_inter FROM unnest(v_n1) n WHERE n = ANY(v_n2);
    v_union := array_length(v_n1, 1) + array_length(v_n2, 1) - v_inter;
    
    RETURN CASE WHEN v_union = 0 THEN 0.0 ELSE v_inter::DOUBLE PRECISION / v_union END;
END;
$$ LANGUAGE plpgsql STABLE;

COMMIT;
