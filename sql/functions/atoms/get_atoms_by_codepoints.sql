-- =============================================================================
-- GET_ATOMS_BY_CODEPOINTS
-- =============================================================================
-- Batch lookup atoms by Unicode codepoint array
-- =============================================================================

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