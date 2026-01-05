-- =============================================================================
-- Hartonomous Hypercube - AI Operations v5
-- =============================================================================
-- High-level AI operations. Complex graph algorithms use C extensions.
-- These SQL versions are fallbacks when C extension is unavailable.
-- Uses relation table for all edge traversal.
-- =============================================================================

-- No transaction - allow partial success

-- =============================================================================
-- Semantic Edge Queries
-- =============================================================================
-- Semantic edges are stored as depth=1, atom_count=2 compositions
-- M coordinate encodes co-occurrence weight

-- Get co-occurrences for an atom
CREATE OR REPLACE FUNCTION semantic_neighbors(p_id BYTEA, p_k INTEGER DEFAULT 20)
RETURNS TABLE(neighbor_id BYTEA, weight DOUBLE PRECISION) AS $$
    SELECT 
        CASE WHEN r1.child_id = p_id THEN r2.child_id ELSE r1.child_id END,
        ST_M(ST_StartPoint(a.geom))
    FROM atom a
    JOIN relation r1 ON r1.parent_id = a.id AND r1.relation_type = 'C' AND r1.ordinal = 1
    JOIN relation r2 ON r2.parent_id = a.id AND r2.relation_type = 'C' AND r2.ordinal = 2
    WHERE a.depth = 1 AND a.atom_count = 2
      AND (r1.child_id = p_id OR r2.child_id = p_id)
    ORDER BY ST_M(ST_StartPoint(a.geom)) DESC
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Count semantic edges
CREATE OR REPLACE FUNCTION semantic_degree(p_id BYTEA)
RETURNS INTEGER AS $$
    SELECT COUNT(*)::INTEGER
    FROM relation r
    WHERE r.child_id = p_id AND r.relation_type = 'C';
$$ LANGUAGE SQL STABLE;

-- Top semantic edges by weight
CREATE OR REPLACE FUNCTION semantic_top_edges(p_limit INTEGER DEFAULT 50)
RETURNS TABLE(atom1 BYTEA, atom2 BYTEA, weight DOUBLE PRECISION) AS $$
    SELECT r1.child_id, r2.child_id, ST_M(ST_StartPoint(a.geom))
    FROM atom a
    JOIN relation r1 ON r1.parent_id = a.id AND r1.relation_type = 'C' AND r1.ordinal = 1
    JOIN relation r2 ON r2.parent_id = a.id AND r2.relation_type = 'C' AND r2.ordinal = 2
    WHERE a.depth = 1 AND a.atom_count = 2
    ORDER BY ST_M(ST_StartPoint(a.geom)) DESC
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
-- Random Walk - Pure SQL fallback (limited depth)
-- =============================================================================

CREATE OR REPLACE FUNCTION random_walk(p_seed BYTEA, p_steps INTEGER DEFAULT 10)
RETURNS TABLE(step INTEGER, node_id BYTEA) AS $$
    WITH RECURSIVE walk AS (
        SELECT 0 AS step, p_seed AS node
        UNION ALL
        SELECT w.step + 1, 
               (SELECT neighbor_id FROM semantic_neighbors(w.node, 100) ORDER BY random() LIMIT 1)
        FROM walk w
        WHERE w.step < p_steps AND w.node IS NOT NULL
    )
    SELECT step, node FROM walk WHERE node IS NOT NULL;
$$ LANGUAGE SQL VOLATILE;

-- =============================================================================
-- Path Finding - Pure SQL fallback (BFS, limited depth)
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_path(p_from BYTEA, p_to BYTEA, p_max_depth INTEGER DEFAULT 6)
RETURNS TABLE(step INTEGER, node_id BYTEA) AS $$
    -- Simple BFS returning first path found to target
    WITH RECURSIVE bfs AS (
        SELECT 0 AS step, p_from AS node, ARRAY[p_from] AS path, FALSE AS found
        UNION ALL
        SELECT b.step + 1, n.neighbor_id, b.path || n.neighbor_id,
               n.neighbor_id = p_to
        FROM bfs b
        CROSS JOIN LATERAL semantic_neighbors(b.node, 20) n
        WHERE b.step < p_max_depth 
          AND NOT b.found
          AND n.neighbor_id != ALL(b.path)
    )
    SELECT step, unnest(path) AS node_id 
    FROM bfs 
    WHERE found 
    ORDER BY step 
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Shared Children (Relationship Inference)
-- =============================================================================

CREATE OR REPLACE FUNCTION related_by_children(p_id BYTEA, p_min_shared INTEGER DEFAULT 1, p_limit INTEGER DEFAULT 20)
RETURNS TABLE(related_id BYTEA, shared_count INTEGER) AS $$
    WITH my_children AS (
        SELECT child_id AS child 
        FROM relation 
        WHERE parent_id = p_id AND relation_type = 'C'
    )
    SELECT r.parent_id, COUNT(*)::INTEGER
    FROM relation r
    JOIN my_children mc ON mc.child = r.child_id
    WHERE r.parent_id != p_id AND r.relation_type = 'C'
    GROUP BY r.parent_id
    HAVING COUNT(*) >= p_min_shared
    ORDER BY COUNT(*) DESC
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Jaccard Similarity - optimized SQL using set operations
-- =============================================================================

CREATE OR REPLACE FUNCTION semantic_jaccard(p_a BYTEA, p_b BYTEA)
RETURNS DOUBLE PRECISION AS $$
    WITH 
    n1 AS (SELECT neighbor_id FROM semantic_neighbors(p_a, 100)),
    n2 AS (SELECT neighbor_id FROM semantic_neighbors(p_b, 100)),
    inter AS (SELECT COUNT(*) AS cnt FROM n1 WHERE neighbor_id IN (SELECT neighbor_id FROM n2)),
    sizes AS (SELECT (SELECT COUNT(*) FROM n1) AS s1, (SELECT COUNT(*) FROM n2) AS s2)
    SELECT CASE 
        WHEN s1 = 0 OR s2 = 0 THEN 0.0 
        ELSE inter.cnt::DOUBLE PRECISION / (s1 + s2 - inter.cnt)
    END
    FROM inter, sizes;
$$ LANGUAGE SQL STABLE;

COMMIT;
