-- =============================================================================
-- DB_STATS
-- =============================================================================
-- Get comprehensive database statistics
-- =============================================================================

CREATE OR REPLACE FUNCTION db_stats()
RETURNS TABLE(
    atoms BIGINT,
    compositions BIGINT,
    compositions_with_centroid BIGINT,
    relations BIGINT,
    models TEXT[]
) AS $$
    SELECT
        (SELECT count(*) FROM atom),
        (SELECT count(*) FROM composition),
        (SELECT count(*) FROM composition WHERE centroid IS NOT NULL),
        (SELECT count(*) FROM relation),
        (SELECT array_agg(DISTINCT source_model) FROM relation WHERE source_model IS NOT NULL AND source_model != '');
$$ LANGUAGE SQL STABLE;