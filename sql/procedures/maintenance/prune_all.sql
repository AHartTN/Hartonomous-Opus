-- Wrapper procedure to run all pruning operations
-- Callable from ingestion pipeline or API

CREATE OR REPLACE PROCEDURE prune_all(relation_threshold REAL DEFAULT 0.1)
LANGUAGE plpgsql AS $$
BEGIN
    RAISE NOTICE 'Starting pruning operations...';

    -- Prune low-quality projections
    CALL prune_projections_quality();

    -- Prune duplicate projections
    CALL prune_projections_deduplication();

    -- Prune low-weight relations
    CALL prune_relations_weight(relation_threshold);

    RAISE NOTICE 'All pruning operations completed.';
END;
$$;