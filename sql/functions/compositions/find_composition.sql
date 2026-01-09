-- =============================================================================
-- FIND_COMPOSITION
-- =============================================================================
-- Find composition by label (vocabulary lookup)
-- =============================================================================

CREATE OR REPLACE FUNCTION find_composition(p_label TEXT)
RETURNS BYTEA AS $$
    SELECT id FROM composition WHERE label = p_label LIMIT 1;
$$ LANGUAGE SQL STABLE;