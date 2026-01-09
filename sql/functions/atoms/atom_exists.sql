-- =============================================================================
-- ATOM_EXISTS
-- =============================================================================
-- Check if entity exists (atom or composition)
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_exists(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id)
        OR EXISTS(SELECT 1 FROM composition WHERE id = p_id);
$$ LANGUAGE SQL STABLE;