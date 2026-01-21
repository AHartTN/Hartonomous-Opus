-- =============================================================================
-- GET_ATOMS_BY_CODEPOINTS
-- =============================================================================
-- Batch lookup atoms by Unicode codepoint array
-- =============================================================================

CREATE OR REPLACE FUNCTION get_atoms_by_codepoints(p_codepoints INTEGER[])
RETURNS TABLE(
    codepoint INTEGER,
    id_hex TEXT,
    coord_x DOUBLE PRECISION,
    coord_y DOUBLE PRECISION,
    coord_z DOUBLE PRECISION,
    coord_m DOUBLE PRECISION
) AS $$
    SELECT
        a.codepoint,
        encode(a.id, 'hex'),
        ST_X(a.geom)::DOUBLE PRECISION,
        ST_Y(a.geom)::DOUBLE PRECISION,
        ST_Z(a.geom)::DOUBLE PRECISION,
        ST_M(a.geom)::DOUBLE PRECISION
    FROM atom a
    WHERE a.codepoint = ANY(p_codepoints);
$$ LANGUAGE SQL STABLE;