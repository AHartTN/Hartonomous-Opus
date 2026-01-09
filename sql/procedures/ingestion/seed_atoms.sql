-- =============================================================================
-- SEED_ATOMS PROCEDURE
-- =============================================================================
-- Bulk load all Unicode atoms with proper session tuning and indexing
-- =============================================================================

CREATE OR REPLACE PROCEDURE seed_atoms_setup()
LANGUAGE plpgsql AS $$
BEGIN
    -- Session tuning for bulk load
    SET synchronous_commit = off;
    SET maintenance_work_mem = '2GB';
    SET work_mem = '256MB';

    -- Drop indexes for fast bulk insert
    DROP INDEX IF EXISTS idx_atom_geom;
    DROP INDEX IF EXISTS idx_atom_hilbert;
    DROP INDEX IF EXISTS idx_atom_codepoint;

    -- Truncate atom table
    TRUNCATE atom CASCADE;
END;
$$;

CREATE OR REPLACE PROCEDURE seed_atoms_finalize()
LANGUAGE plpgsql AS $$
BEGIN
    -- Restore session settings
    SET synchronous_commit = on;
    SET maintenance_work_mem = '64MB';
    SET work_mem = '4MB';

    -- Rebuild indexes
    SET maintenance_work_mem = '2GB';
    SET max_parallel_maintenance_workers = 4;

    CREATE INDEX IF NOT EXISTS idx_atom_codepoint ON atom(codepoint);
    CREATE INDEX IF NOT EXISTS idx_atom_hilbert ON atom(hilbert_hi, hilbert_lo);
    CREATE INDEX IF NOT EXISTS idx_atom_geom ON atom USING GIST(geom);

    -- Analyze for query optimization
    ANALYZE atom;

    -- Reset maintenance settings
    SET maintenance_work_mem = '64MB';
END;
$$;