-- =============================================================================
-- Hartonomous Hypercube - Core Query Functions
-- =============================================================================
-- SQL helper functions for the C++ tools. Reconstruction functions are 
-- provided by the semantic_ops C extension for performance.
-- =============================================================================

-- =============================================================================
-- Reconstruction Aliases (delegates to C extension)
-- =============================================================================
-- Note: semantic_reconstruct is provided by semantic_ops extension

CREATE OR REPLACE FUNCTION atom_text(p_id BYTEA)
RETURNS TEXT AS $$
    SELECT semantic_reconstruct(p_id);
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION atom_reconstruct_text(p_id BYTEA)
RETURNS TEXT AS $$
    SELECT semantic_reconstruct(p_id);
$$ LANGUAGE sql STABLE;

-- =============================================================================
-- Content Hash Functions
-- =============================================================================

CREATE OR REPLACE FUNCTION atom_content_hash(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT hypercube_blake3(convert_to(p_text, 'UTF8'));
$$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION atom_find(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE id = atom_content_hash(p_text);
$$ LANGUAGE sql STABLE;

-- =============================================================================
-- Statistics & Diagnostics (4-TABLE SCHEMA)
-- =============================================================================

-- Complete database statistics for 4-table schema
CREATE OR REPLACE FUNCTION db_stats()
RETURNS TABLE (
    atoms BIGINT,
    compositions BIGINT,
    relations BIGINT,
    shapes BIGINT,
    models TEXT[]
) AS $$
    SELECT 
        (SELECT COUNT(*) FROM atom),
        (SELECT COUNT(*) FROM composition),
        (SELECT COUNT(*) FROM relation),
        (SELECT COUNT(*) FROM shape),
        (SELECT array_agg(DISTINCT model_name) FROM shape WHERE model_name IS NOT NULL)
$$ LANGUAGE sql STABLE;

-- Depth distribution (from composition table)
CREATE OR REPLACE FUNCTION db_depth_distribution()
RETURNS TABLE (
    depth INTEGER,
    count BIGINT,
    min_atoms BIGINT,
    max_atoms BIGINT,
    avg_atoms NUMERIC
) AS $$
    SELECT 
        c.depth,
        COUNT(*),
        MIN(c.atom_count),
        MAX(c.atom_count),
        AVG(c.atom_count)::NUMERIC(20,2)
    FROM composition c
    GROUP BY c.depth
    ORDER BY c.depth;
$$ LANGUAGE sql STABLE;

-- Index health check
CREATE OR REPLACE FUNCTION db_index_status()
RETURNS TABLE (
    index_name TEXT,
    table_name TEXT,
    index_size TEXT
) AS $$
    SELECT 
        indexname::TEXT,
        tablename::TEXT,
        pg_size_pretty(pg_relation_size(indexname::regclass))
    FROM pg_indexes 
    WHERE tablename = 'atom'
    ORDER BY indexname;
$$ LANGUAGE sql STABLE;

-- =============================================================================
-- Atom Lookup Functions
-- =============================================================================

-- Get atom by hash
CREATE OR REPLACE FUNCTION get_atom(p_id BYTEA)
RETURNS TABLE (
    id BYTEA,
    depth INTEGER,
    atom_count BIGINT,
    codepoint INTEGER,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    z DOUBLE PRECISION,
    m DOUBLE PRECISION,
    hilbert_lo BIGINT,
    hilbert_hi BIGINT
) AS $$
    SELECT 
        a.id,
        a.depth,
        a.atom_count,
        a.codepoint,
        ST_X(a.geom),
        ST_Y(a.geom),
        ST_Z(a.geom),
        ST_M(a.geom),
        a.hilbert_lo,
        a.hilbert_hi
    FROM atom a
    WHERE a.id = p_id;
$$ LANGUAGE sql STABLE;

-- Get atom by codepoint (full data)
CREATE OR REPLACE FUNCTION get_atom_by_codepoint(p_cp INTEGER)
RETURNS TABLE (
    id BYTEA,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    z DOUBLE PRECISION,
    m DOUBLE PRECISION,
    hilbert_lo BIGINT,
    hilbert_hi BIGINT
) AS $$
    SELECT 
        a.id,
        ST_X(a.geom),
        ST_Y(a.geom),
        ST_Z(a.geom),
        ST_M(a.geom),
        a.hilbert_lo,
        a.hilbert_hi
    FROM atom a
    WHERE a.depth = 0 AND a.codepoint = p_cp;
$$ LANGUAGE sql STABLE;

-- Batch lookup atoms by codepoint array
CREATE OR REPLACE FUNCTION get_atoms_by_codepoints(p_codepoints INTEGER[])
RETURNS TABLE (
    codepoint INTEGER,
    id BYTEA,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    z DOUBLE PRECISION,
    m DOUBLE PRECISION
) AS $$
    SELECT 
        a.codepoint,
        a.id,
        ST_X(a.geom),
        ST_Y(a.geom),
        ST_Z(a.geom),
        ST_M(a.geom)
    FROM atom a
    WHERE a.depth = 0 
      AND a.codepoint = ANY(p_codepoints);
$$ LANGUAGE sql STABLE;

-- =============================================================================
-- Existence Checks
-- =============================================================================

-- Check if hash exists
CREATE OR REPLACE FUNCTION hash_exists(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id);
$$ LANGUAGE sql STABLE;

-- Check if content exists (by text)
CREATE OR REPLACE FUNCTION text_exists(p_text TEXT)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = atom_content_hash(p_text));
$$ LANGUAGE sql STABLE;

-- =============================================================================
-- Hilbert Range Queries
-- =============================================================================

-- Get atoms in Hilbert range
CREATE OR REPLACE FUNCTION hilbert_range(
    p_center_lo BIGINT,
    p_center_hi BIGINT,
    p_range BIGINT,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    id BYTEA,
    depth INTEGER,
    hilbert_lo BIGINT,
    hilbert_hi BIGINT
) AS $$
    SELECT a.id, a.depth, a.hilbert_lo, a.hilbert_hi
    FROM atom a
    WHERE a.hilbert_hi = p_center_hi
      AND a.hilbert_lo BETWEEN p_center_lo - p_range AND p_center_lo + p_range
    ORDER BY ABS(a.hilbert_lo - p_center_lo)
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;

-- Get nearest by Hilbert index
CREATE OR REPLACE FUNCTION hilbert_nearest(p_id BYTEA, p_limit INTEGER DEFAULT 10)
RETURNS TABLE (
    id BYTEA,
    depth INTEGER,
    distance BIGINT
) AS $$
    WITH target AS (
        SELECT hilbert_lo, hilbert_hi FROM atom WHERE id = p_id
    )
    SELECT 
        a.id,
        a.depth,
        ABS(a.hilbert_lo - t.hilbert_lo)::BIGINT as distance
    FROM atom a, target t
    WHERE a.id != p_id
      AND a.hilbert_hi = t.hilbert_hi
    ORDER BY ABS(a.hilbert_lo - t.hilbert_lo)
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;

-- =============================================================================
-- Spatial Queries
-- =============================================================================

-- Get nearest by spatial distance
CREATE OR REPLACE FUNCTION spatial_nearest(p_id BYTEA, p_limit INTEGER DEFAULT 10)
RETURNS TABLE (
    id BYTEA,
    depth INTEGER,
    distance DOUBLE PRECISION
) AS $$
    WITH target AS (
        SELECT centroid FROM atom WHERE id = p_id
    )
    SELECT 
        a.id,
        a.depth,
        a.centroid <-> t.centroid as distance
    FROM atom a, target t
    WHERE a.id != p_id
    ORDER BY a.centroid <-> t.centroid
    LIMIT p_limit;
$$ LANGUAGE sql STABLE;

-- =============================================================================
-- Batch Insert Support
-- =============================================================================

-- Prepare staging table for batch insert
CREATE OR REPLACE FUNCTION batch_insert_prepare()
RETURNS VOID AS $$
BEGIN
    DROP TABLE IF EXISTS tmp_atom;
    CREATE TEMP TABLE tmp_atom (
        id BYTEA,
        geom TEXT,
        value BYTEA,
        hilbert_lo BIGINT,
        hilbert_hi BIGINT,
        depth INTEGER,
        atom_count BIGINT,
        node_role SMALLINT DEFAULT 0
    ) ON COMMIT DROP;
END;
$$ LANGUAGE plpgsql;

-- Finalize batch insert
CREATE OR REPLACE FUNCTION batch_insert_finalize()
RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    INSERT INTO atom (id, geom, value, hilbert_lo, hilbert_hi, depth, atom_count, node_role)
    SELECT id, geom::geometry, value, hilbert_lo, hilbert_hi, depth, atom_count, COALESCE(node_role, 0)
    FROM tmp_atom
    ON CONFLICT (id) DO NOTHING;
    
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Validation Functions
-- =============================================================================

-- Validate atom integrity
CREATE OR REPLACE FUNCTION validate_atoms()
RETURNS TABLE (
    check_name TEXT,
    passed BOOLEAN,
    details TEXT
) AS $$
BEGIN
    -- Check 1: All leaves have codepoints
    check_name := 'leaves_have_codepoints';
    SELECT COUNT(*) = 0 INTO passed
    FROM atom WHERE depth = 0 AND codepoint IS NULL;
    details := CASE WHEN passed THEN 'OK' ELSE 'Found leaves without codepoints' END;
    RETURN NEXT;
    
    -- Check 2: All compositions have relations
    check_name := 'compositions_have_relations';
    SELECT COUNT(*) = 0 INTO passed
    FROM atom a
    WHERE a.depth > 0 
      AND NOT EXISTS (SELECT 1 FROM relation r WHERE r.parent_id = a.id AND r.relation_type = 'C');
    details := CASE WHEN passed THEN 'OK' ELSE 'Found compositions without relations' END;
    RETURN NEXT;
    
    -- Check 3: All have valid geometry
    check_name := 'valid_geometry';
    SELECT COUNT(*) = 0 INTO passed
    FROM atom WHERE geom IS NULL;
    details := CASE WHEN passed THEN 'OK' ELSE 'Found atoms without geometry' END;
    RETURN NEXT;
    
    -- Check 4: Hilbert indices are set
    check_name := 'hilbert_indices_set';
    SELECT COUNT(*) = 0 INTO passed
    FROM atom WHERE hilbert_lo IS NULL OR hilbert_hi IS NULL;
    details := CASE WHEN passed THEN 'OK' ELSE 'Found atoms without Hilbert indices' END;
    RETURN NEXT;
    
    RETURN;
END;
$$ LANGUAGE plpgsql STABLE;
