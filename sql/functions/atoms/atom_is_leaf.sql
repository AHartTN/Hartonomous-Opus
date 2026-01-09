-- =============================================================================
-- ATOM_IS_LEAF
-- =============================================================================
-- Check if entity is a leaf (atom) vs composition
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_is_leaf(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id);
$$ LANGUAGE SQL STABLE;