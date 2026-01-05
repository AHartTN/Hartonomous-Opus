-- =============================================================================
-- Hartonomous Hypercube - Ingestion Operations
-- =============================================================================
-- SQL functions specific to content ingestion.
-- General query functions are in 018_core_queries.sql
-- =============================================================================

BEGIN;

-- =============================================================================
-- Composition Insertion (single row - for testing/small batches)
-- =============================================================================

CREATE OR REPLACE FUNCTION upsert_composition(
    p_id BYTEA,
    p_geom GEOMETRY,
    p_children BYTEA[],
    p_hilbert_lo BIGINT,
    p_hilbert_hi BIGINT,
    p_depth INTEGER,
    p_atom_count BIGINT
) RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count)
    VALUES (p_id, p_geom, p_children, p_hilbert_lo, p_hilbert_hi, p_depth, p_atom_count)
    ON CONFLICT (id) DO NOTHING;
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Content Hash Functions (for semantic lookup)
-- =============================================================================

-- Compute raw BLAKE3 hash for text content (just the bytes, not CPE)
CREATE OR REPLACE FUNCTION content_hash(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT hypercube_blake3(convert_to(p_text, 'UTF8'));
$$ LANGUAGE sql IMMUTABLE STRICT;

-- Find composition by text content (uses CPE hash)
CREATE OR REPLACE FUNCTION find_content(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE id = atom_content_hash(p_text);
$$ LANGUAGE sql STABLE;

-- Check if content exists (uses CPE hash)
CREATE OR REPLACE FUNCTION content_exists(p_text TEXT)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = atom_content_hash(p_text));
$$ LANGUAGE sql STABLE;

COMMIT;
