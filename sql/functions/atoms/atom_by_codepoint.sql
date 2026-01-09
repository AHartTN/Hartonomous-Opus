-- =============================================================================
-- ATOM_BY_CODEPOINT
-- =============================================================================
-- Lookup atom by Unicode codepoint
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_by_codepoint(p_cp INTEGER)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE codepoint = p_cp;
$$ LANGUAGE SQL STABLE;