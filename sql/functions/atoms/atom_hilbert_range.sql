-- =============================================================================
-- ATOM_HILBERT_RANGE
-- =============================================================================
-- Hilbert range query for atoms using hilbert_hi and hilbert_lo bounds
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_hilbert_range(p_hi_lo BIGINT, p_hi_hi BIGINT, p_lo_lo BIGINT, p_lo_hi BIGINT)
RETURNS TABLE(id BYTEA, codepoint INTEGER) AS $$
    SELECT id, codepoint FROM atom
    WHERE hilbert_hi BETWEEN p_hi_lo AND p_hi_hi
      AND hilbert_lo BETWEEN p_lo_lo AND p_lo_hi
    ORDER BY hilbert_hi, hilbert_lo;
$$ LANGUAGE SQL STABLE;