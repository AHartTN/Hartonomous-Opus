-- =============================================================================
-- ATOM LOOKUP FUNCTIONS
-- =============================================================================
-- Functions for finding and retrieving atoms by various criteria
-- =============================================================================

-- Lookup by codepoint
CREATE OR REPLACE FUNCTION atom_by_codepoint(p_cp INTEGER)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE codepoint = p_cp;
$$ LANGUAGE SQL STABLE;

-- Batch lookup atoms by codepoints (for ingestion)
CREATE OR REPLACE FUNCTION get_atoms_by_codepoints(p_codepoints INTEGER[])
RETURNS TABLE(
    codepoint INTEGER,
    id_hex TEXT,
    coord_x BIGINT,
    coord_y BIGINT,
    coord_z BIGINT,
    coord_m BIGINT
) AS $$
    SELECT
        a.codepoint,
        encode(a.id, 'hex'),
        ST_X(a.geom)::BIGINT,
        ST_Y(a.geom)::BIGINT,
        ST_Z(a.geom)::BIGINT,
        ST_M(a.geom)::BIGINT
    FROM atom a
    WHERE a.codepoint = ANY(p_codepoints);
$$ LANGUAGE SQL STABLE;

-- Hilbert range query for atoms
CREATE OR REPLACE FUNCTION atom_hilbert_range(p_hi_lo BIGINT, p_hi_hi BIGINT, p_lo_lo BIGINT, p_lo_hi BIGINT)
RETURNS TABLE(id BYTEA, codepoint INTEGER) AS $$
    SELECT id, codepoint FROM atom
    WHERE hilbert_hi BETWEEN p_hi_lo AND p_hi_hi
      AND hilbert_lo BETWEEN p_lo_lo AND p_lo_hi
    ORDER BY hilbert_hi, hilbert_lo;
$$ LANGUAGE SQL STABLE;