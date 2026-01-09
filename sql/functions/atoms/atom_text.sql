-- =============================================================================
-- ATOM_TEXT
-- =============================================================================
-- Get text representation of an atom
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_text(p_id BYTEA)
RETURNS TEXT AS $$
    SELECT chr(codepoint) FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;