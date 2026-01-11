-- =============================================================================
-- ATOM_STATS
-- =============================================================================
-- Get comprehensive atom-level statistics
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_stats()
RETURNS TABLE(
    total_atoms BIGINT,
    valid_geometry BIGINT,
    invalid_geometry BIGINT,
    min_codepoint INTEGER,
    max_codepoint INTEGER,
    unique_codepoints BIGINT,
    hilbert_indexed BIGINT,
    sphere_surface BIGINT
) AS $$
    SELECT
        (SELECT count(*) FROM atom) as total_atoms,
        (SELECT count(*) FROM atom WHERE geom IS NOT NULL) as valid_geometry,
        (SELECT count(*) FROM atom WHERE geom IS NULL) as invalid_geometry,
        (SELECT min(codepoint) FROM atom) as min_codepoint,
        (SELECT max(codepoint) FROM atom) as max_codepoint,
        (SELECT count(distinct codepoint) FROM atom) as unique_codepoints,
        (SELECT count(*) FROM atom WHERE hilbert_lo IS NOT NULL AND hilbert_hi IS NOT NULL) as hilbert_indexed,
        (SELECT count(*) FROM atom WHERE geom IS NOT NULL AND
                                      ST_X(geom) IS NOT NULL AND
                                      ST_Y(geom) IS NOT NULL AND
                                      ST_Z(geom) IS NOT NULL AND
                                      ST_M(geom) IS NOT NULL) as sphere_surface
$$ LANGUAGE SQL STABLE;