-- =============================================================================
-- COMPOSITION CORE FUNCTIONS
-- =============================================================================
-- Basic composition operations for the 3-table hypercube schema
-- =============================================================================

-- Get children of a composition (empty for atoms)
CREATE OR REPLACE FUNCTION atom_children(p_id BYTEA)
RETURNS TABLE(child_id BYTEA, child_type CHAR(1), ordinal SMALLINT) AS $$
    SELECT cc.child_id, cc.child_type, cc.ordinal
    FROM composition_child cc
    WHERE cc.composition_id = p_id
    ORDER BY cc.ordinal;
$$ LANGUAGE SQL STABLE;

-- Get child count (0 for atoms)
CREATE OR REPLACE FUNCTION atom_child_count(p_id BYTEA)
RETURNS INTEGER AS $$
    SELECT COALESCE(
        (SELECT child_count FROM composition WHERE id = p_id),
        0
    );
$$ LANGUAGE SQL STABLE;