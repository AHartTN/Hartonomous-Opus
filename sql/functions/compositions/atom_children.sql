-- =============================================================================
-- ATOM_CHILDREN
-- =============================================================================
-- Get children of a composition
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_children(p_id BYTEA)
RETURNS TABLE(child_id BYTEA, child_type CHAR(1), ordinal SMALLINT) AS $$
    SELECT cc.child_id, cc.child_type, cc.ordinal
    FROM composition_child cc
    WHERE cc.composition_id = p_id
    ORDER BY cc.ordinal;
$$ LANGUAGE SQL STABLE;