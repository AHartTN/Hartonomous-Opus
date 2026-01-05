-- Proper ordered tree reconstruction
-- The key is to traverse LEFT child before RIGHT child

WITH RECURSIVE tree(id, codepoint, level, path) AS (
    -- Start with a depth-3 composition (simpler to trace)
    (SELECT id, codepoint, 0, ARRAY[0::bigint]
     FROM atom WHERE depth = 3 LIMIT 1)
    
    UNION ALL
    
    -- Traverse children IN ORDER (first child = left = 1, second child = right = 2)
    SELECT c.id, c.codepoint, t.level + 1, 
           t.path || (child_ord * 1000000 + t.level)::bigint
    FROM tree t
    CROSS JOIN LATERAL (
        SELECT child_id, ordinality as child_ord 
        FROM unnest((SELECT children FROM atom WHERE id = t.id)) 
        WITH ORDINALITY as u(child_id, ordinality)
    ) children
    JOIN atom c ON c.id = children.child_id
    WHERE t.level < 10 AND t.codepoint IS NULL  -- Only expand non-leaves
)
SELECT 
    level,
    codepoint,
    CASE WHEN codepoint IS NOT NULL THEN chr(codepoint) ELSE NULL END as char,
    path
FROM tree
ORDER BY path;
