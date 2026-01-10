-- Procedure to prune low-quality projections
-- Deletes projections where quality_score < 2.0 or converged = false

CREATE OR REPLACE PROCEDURE prune_projections_quality()
LANGUAGE plpgsql AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM projection_metadata
    WHERE quality_score < 2.0 OR converged = false;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Pruned % low-quality projections', deleted_count;
END;
$$;