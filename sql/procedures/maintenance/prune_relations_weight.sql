-- Procedure to prune relations with weights below threshold (default 0.1)

CREATE OR REPLACE PROCEDURE prune_relations_weight(threshold REAL DEFAULT 0.1)
LANGUAGE plpgsql AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM relation
    WHERE weight < threshold;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Pruned % relations with weight < %', deleted_count, threshold;
END;
$$;