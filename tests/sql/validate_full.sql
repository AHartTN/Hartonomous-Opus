-- COMPREHENSIVE VALIDATION OF HYPERCUBE INGESTION
-- ================================================

-- 1. Overall statistics
SELECT 'STATS' as test, 
    COUNT(*) as total_atoms,
    COUNT(*) FILTER (WHERE depth = 0) as codepoints,
    COUNT(*) FILTER (WHERE depth > 0) as compositions,
    MAX(depth) as max_depth
FROM atom;

-- 2. Verify all compositions have exactly 2 children
SELECT 'BINARY_CHECK' as test,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END as result,
    COUNT(*) as non_binary_count
FROM atom 
WHERE depth > 0 AND array_length(children, 1) != 2;

-- 3. Verify no orphan children (all references valid)
SELECT 'ORPHAN_CHECK' as test,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END as result,
    COUNT(*) as orphan_count
FROM atom a, unnest(a.children) as child_id
WHERE NOT EXISTS (SELECT 1 FROM atom b WHERE b.id = child_id);

-- 4. Verify atom_count consistency (parent = sum of children)
SELECT 'ATOM_COUNT_CHECK' as test,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END as result,
    COUNT(*) as mismatched_count
FROM atom parent
WHERE parent.depth > 0
AND parent.atom_count != (
    SELECT COALESCE(SUM(c.atom_count), 0)
    FROM unnest(parent.children) child_id
    JOIN atom c ON c.id = child_id
);

-- 5. Verify all codepoints (depth 0) have no children
SELECT 'LEAF_CHECK' as test,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END as result,
    COUNT(*) as leaves_with_children
FROM atom
WHERE depth = 0 AND children IS NOT NULL;

-- 6. Verify all compositions (depth > 0) have children
SELECT 'COMP_CHECK' as test,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END as result,
    COUNT(*) as comps_without_children
FROM atom
WHERE depth > 0 AND children IS NULL;

-- 7. Verify geometric data exists
SELECT 'GEOMETRY_CHECK' as test,
    COUNT(*) FILTER (WHERE centroid IS NOT NULL) as with_centroid,
    COUNT(*) FILTER (WHERE geom IS NOT NULL) as with_geom,
    COUNT(*) as total
FROM atom WHERE depth > 0;

-- 8. Sample text reconstruction from a composition
WITH RECURSIVE tree(id, codepoint, level, path) AS (
    (SELECT id, codepoint, 0, ARRAY[id]
     FROM atom WHERE depth = 10 LIMIT 1)
    UNION ALL
    SELECT c.id, c.codepoint, t.level + 1, t.path || c.id
    FROM tree t, unnest((SELECT children FROM atom WHERE id = t.id)) child_id, atom c 
    WHERE c.id = child_id AND t.level < 15
)
SELECT 'TEXT_RECONSTRUCTION' as test,
    string_agg(chr(codepoint), '' ORDER BY path) as reconstructed_text
FROM tree
WHERE codepoint IS NOT NULL;
