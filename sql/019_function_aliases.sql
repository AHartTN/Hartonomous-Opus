-- Deprecation aliases for old function names
-- =============================================================================
-- Hartonomous Hypercube - Function Aliases & Deprecations
-- =============================================================================
-- Creates aliases for backward compatibility.
-- New code should use the canonical names from 018_core_queries.sql
-- =============================================================================

BEGIN;

-- Canonical: get_atom_by_codepoint -> Alias: atom_by_codepoint
CREATE OR REPLACE FUNCTION atom_by_codepoint(p_cp INTEGER)
RETURNS BYTEA AS $$
    SELECT id FROM get_atom_by_codepoint(p_cp);
$$ LANGUAGE sql STABLE;

-- Canonical: get_atoms_by_codepoints -> Alias: atoms_by_codepoints  
CREATE OR REPLACE FUNCTION atoms_by_codepoints(p_codepoints INTEGER[])
RETURNS TABLE (
    codepoint INTEGER,
    id BYTEA,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    z DOUBLE PRECISION,
    m DOUBLE PRECISION
) AS $$
    SELECT * FROM get_atoms_by_codepoints(p_codepoints);
$$ LANGUAGE sql STABLE;

-- Canonical: hash_exists -> Alias: atom_exists
CREATE OR REPLACE FUNCTION atom_exists(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT hash_exists(p_id);
$$ LANGUAGE sql STABLE;

-- Canonical: find_content -> Alias: atom_find
CREATE OR REPLACE FUNCTION atom_find(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT find_content(p_text);
$$ LANGUAGE sql STABLE;

-- Canonical: find_content -> Alias: atom_find_exact
CREATE OR REPLACE FUNCTION atom_find_exact(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT find_content(p_text);
$$ LANGUAGE sql STABLE;

-- Canonical: db_stats -> Alias: ingestion_stats (for backward compat)
CREATE OR REPLACE FUNCTION ingestion_stats()
RETURNS TABLE (
    leaf_atoms BIGINT,
    compositions BIGINT,
    max_depth INTEGER,
    total_size TEXT
) AS $$
    SELECT leaf_atoms, compositions, max_depth, db_size FROM db_stats();
$$ LANGUAGE sql STABLE;

-- Canonical: db_depth_distribution -> Alias: depth_distribution
CREATE OR REPLACE FUNCTION depth_distribution()
RETURNS TABLE (
    depth INTEGER,
    count BIGINT,
    min_atoms BIGINT,
    max_atoms BIGINT
) AS $$
    SELECT depth, count, min_atoms, max_atoms FROM db_depth_distribution();
$$ LANGUAGE sql STABLE;

COMMIT;
