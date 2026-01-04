-- Recursively trace a composition down to its leaf codepoints
WITH RECURSIVE tree(id, depth, atom_count, children, codepoint, level) AS (
    -- Start with a depth-5 composition
    (SELECT id, depth, atom_count, children, codepoint, 0 
     FROM atom 
     WHERE depth = 5 
     LIMIT 1)
    
    UNION ALL
    
    -- Recurse to children
    SELECT c.id, c.depth, c.atom_count, c.children, c.codepoint, t.level + 1
    FROM tree t, unnest(t.children) child_id, atom c 
    WHERE c.id = child_id AND t.level < 10
)
SELECT 
    level,
    depth,
    atom_count,
    CASE WHEN codepoint IS NOT NULL THEN chr(codepoint) ELSE '(comp)' END as node
FROM tree
ORDER BY level, depth DESC;
