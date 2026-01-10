-- Procedure to prune duplicate projections, keeping top 3 champion models per role
-- Uses get_champion_models to identify champions and removes non-champion projections

CREATE OR REPLACE PROCEDURE prune_projections_deduplication()
LANGUAGE plpgsql AS $$
DECLARE
    champion RECORD;
    deleted_count INTEGER := 0;
    total_deleted INTEGER := 0;
BEGIN
    -- For each role, delete projections not from champion models
    FOR champion IN
        SELECT DISTINCT role FROM projection_metadata
    LOOP
        -- Delete projections where model_id not in champions for this role
        DELETE FROM projection_metadata pm
        WHERE pm.role = champion.role
          AND pm.model_id NOT IN (
              SELECT model_id
              FROM get_champion_models(champion.role)
          );

        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        total_deleted := total_deleted + deleted_count;

        RAISE NOTICE 'Pruned % duplicate projections for role %', deleted_count, champion.role;
    END LOOP;

    RAISE NOTICE 'Total pruned duplicate projections: %', total_deleted;
END;
$$;