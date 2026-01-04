-- =============================================================================
-- Hartonomous Hypercube - Optimized Batch Operations v2
-- =============================================================================
-- All functions are PARALLEL SAFE, set-based, no row-by-row loops.
-- Uses CTEs with MATERIALIZED for predictable execution.
-- =============================================================================

BEGIN;

-- =============================================================================
-- PARALLEL SAFE Core Functions
-- =============================================================================

-- Drop and recreate with PARALLEL SAFE
DROP FUNCTION IF EXISTS atom_is_leaf(BYTEA) CASCADE;
CREATE OR REPLACE FUNCTION atom_is_leaf(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT depth = 0 FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

DROP FUNCTION IF EXISTS atom_centroid(BYTEA) CASCADE;
CREATE OR REPLACE FUNCTION atom_centroid(p_id BYTEA)
RETURNS TABLE(x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION) AS $$
    SELECT ST_X(geom), ST_Y(geom), ST_Z(geom), ST_M(geom)
    FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

DROP FUNCTION IF EXISTS atom_children(BYTEA) CASCADE;
CREATE OR REPLACE FUNCTION atom_children(p_id BYTEA)
RETURNS TABLE(child_id BYTEA, ordinal INTEGER) AS $$
    SELECT r.child_id, r.ordinal
    FROM relation r
    WHERE r.parent_id = p_id AND r.relation_type = 'C'
    ORDER BY r.ordinal;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

DROP FUNCTION IF EXISTS atom_child_count(BYTEA) CASCADE;
CREATE OR REPLACE FUNCTION atom_child_count(p_id BYTEA)
RETURNS INTEGER AS $$
    SELECT COUNT(*)::INTEGER 
    FROM relation 
    WHERE parent_id = p_id AND relation_type = 'C';
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

DROP FUNCTION IF EXISTS atom_by_codepoint(INTEGER) CASCADE;
CREATE OR REPLACE FUNCTION atom_by_codepoint(p_cp INTEGER)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE codepoint = p_cp;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

DROP FUNCTION IF EXISTS atom_exists(BYTEA) CASCADE;
CREATE OR REPLACE FUNCTION atom_exists(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

DROP FUNCTION IF EXISTS atom_distance(BYTEA, BYTEA) CASCADE;
CREATE OR REPLACE FUNCTION atom_distance(p_id1 BYTEA, p_id2 BYTEA)
RETURNS DOUBLE PRECISION AS $$
    SELECT sqrt(
        power(ST_X(a.geom) - ST_X(b.geom), 2) +
        power(ST_Y(a.geom) - ST_Y(b.geom), 2) +
        power(ST_Z(a.geom) - ST_Z(b.geom), 2) +
        power(ST_M(a.geom) - ST_M(b.geom), 2)
    )
    FROM atom a, atom b
    WHERE a.id = p_id1 AND b.id = p_id2;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

DROP FUNCTION IF EXISTS atom_knn(BYTEA, INTEGER) CASCADE;
CREATE OR REPLACE FUNCTION atom_knn(p_id BYTEA, p_k INTEGER DEFAULT 10)
RETURNS TABLE(neighbor_id BYTEA, distance DOUBLE PRECISION, depth INTEGER) AS $$
    WITH target AS MATERIALIZED (SELECT geom FROM atom WHERE id = p_id)
    SELECT a.id, a.geom <-> t.geom, a.depth
    FROM atom a, target t
    WHERE a.id != p_id
    ORDER BY a.geom <-> t.geom
    LIMIT p_k;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- =============================================================================
-- Batch Lookup Functions (set-based, no loops)
-- =============================================================================

-- Batch lookup atoms by codepoints (single query, uses index)
DROP FUNCTION IF EXISTS get_atoms_by_codepoints(INTEGER[]) CASCADE;
CREATE OR REPLACE FUNCTION get_atoms_by_codepoints(p_codepoints INTEGER[])
RETURNS TABLE (
    codepoint INTEGER,
    id BYTEA,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    z DOUBLE PRECISION,
    m DOUBLE PRECISION,
    hilbert_lo BIGINT,
    hilbert_hi BIGINT
) AS $$
    SELECT 
        a.codepoint,
        a.id,
        ST_X(a.geom),
        ST_Y(a.geom),
        ST_Z(a.geom),
        ST_M(a.geom),
        a.hilbert_lo,
        a.hilbert_hi
    FROM atom a
    WHERE a.depth = 0 
      AND a.codepoint = ANY(p_codepoints);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- Batch lookup atoms by IDs (single query)
DROP FUNCTION IF EXISTS get_atoms_by_ids(BYTEA[]) CASCADE;
CREATE OR REPLACE FUNCTION get_atoms_by_ids(p_ids BYTEA[])
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
    WHERE a.id = ANY(p_ids);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- Batch check existence (single query, returns present IDs)
DROP FUNCTION IF EXISTS atoms_exist(BYTEA[]) CASCADE;
CREATE OR REPLACE FUNCTION atoms_exist(p_ids BYTEA[])
RETURNS TABLE(id BYTEA) AS $$
    SELECT a.id FROM atom a WHERE a.id = ANY(p_ids);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- =============================================================================
-- Optimized Bigram Extraction (set-based, no loops)
-- =============================================================================

-- Extract all bigrams in a single set-based query using window functions
DROP FUNCTION IF EXISTS extract_bigrams_batch() CASCADE;
CREATE OR REPLACE FUNCTION extract_bigrams_batch()
RETURNS TABLE(inserted_bigrams BIGINT, inserted_unigrams BIGINT) AS $$
DECLARE
    v_bigrams BIGINT;
    v_unigrams BIGINT;
BEGIN
    -- Clear existing data
    TRUNCATE bigram_stats, unigram_stats;
    
    -- Extract bigrams using window function LAG (no loops!)
    INSERT INTO bigram_stats (left_id, right_id, count)
    SELECT left_id, right_id, COUNT(*) as count
    FROM (
        SELECT 
            LAG(cc.child_id) OVER (PARTITION BY cc.composition_id ORDER BY cc.ordinal) as left_id,
            cc.child_id as right_id
        FROM composition_child cc
        JOIN composition p ON p.id = cc.composition_id AND p.depth = 2
        JOIN composition c ON c.id = cc.child_id AND c.depth = 1 
            AND c.label IS NOT NULL 
            AND c.label NOT LIKE '[%'
        WHERE cc.child_type = 'C'
    ) pairs
    WHERE left_id IS NOT NULL
    GROUP BY left_id, right_id
    ON CONFLICT (left_id, right_id) 
    DO UPDATE SET count = bigram_stats.count + EXCLUDED.count;
    
    GET DIAGNOSTICS v_bigrams = ROW_COUNT;
    
    -- Extract unigrams in single query
    INSERT INTO unigram_stats (token_id, count)
    SELECT cc.child_id, COUNT(*) as count
    FROM composition_child cc
    JOIN composition p ON p.id = cc.composition_id AND p.depth = 2
    JOIN composition c ON c.id = cc.child_id AND c.depth = 1 
        AND c.label IS NOT NULL 
        AND c.label NOT LIKE '[%'
    WHERE cc.child_type = 'C'
    GROUP BY cc.child_id
    ON CONFLICT (token_id) 
    DO UPDATE SET count = unigram_stats.count + EXCLUDED.count;
    
    GET DIAGNOSTICS v_unigrams = ROW_COUNT;
    
    -- Update corpus stats
    UPDATE token_corpus_stats 
    SET total_tokens = (SELECT COALESCE(SUM(count), 0) FROM unigram_stats),
        total_bigrams = (SELECT COALESCE(SUM(count), 0) FROM bigram_stats)
    WHERE id = 1;
    
    -- Compute PMI in single UPDATE (no loops)
    PERFORM compute_pmi_batch();
    
    RETURN QUERY SELECT v_bigrams, v_unigrams;
END;
$$ LANGUAGE PLPGSQL;

-- Batch PMI computation (single UPDATE, no loops)
DROP FUNCTION IF EXISTS compute_pmi_batch() CASCADE;
CREATE OR REPLACE FUNCTION compute_pmi_batch()
RETURNS INTEGER AS $$
DECLARE
    v_total_bigrams DOUBLE PRECISION;
    v_total_tokens DOUBLE PRECISION;
    updated_count INTEGER;
BEGIN
    SELECT SUM(count)::DOUBLE PRECISION INTO v_total_bigrams FROM bigram_stats;
    SELECT SUM(count)::DOUBLE PRECISION INTO v_total_tokens FROM unigram_stats;
    
    IF v_total_bigrams IS NULL OR v_total_bigrams = 0 THEN
        RETURN 0;
    END IF;
    
    -- Single UPDATE with JOIN (no loop)
    WITH counts AS MATERIALIZED (
        SELECT 
            b.left_id,
            b.right_id,
            b.count as bigram_count,
            ul.count as left_count,
            ur.count as right_count
        FROM bigram_stats b
        JOIN unigram_stats ul ON ul.token_id = b.left_id
        JOIN unigram_stats ur ON ur.token_id = b.right_id
    )
    UPDATE bigram_stats b
    SET pmi = ln(
        (c.bigram_count * v_total_tokens * v_total_tokens) /
        (c.left_count::DOUBLE PRECISION * c.right_count * v_total_bigrams)
    )
    FROM counts c
    WHERE b.left_id = c.left_id AND b.right_id = c.right_id;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE PLPGSQL;

-- =============================================================================
-- Batch Atom/Composition Upsert (COPY-friendly, no loops)
-- =============================================================================

-- Bulk upsert atoms from temp table (used with COPY)
DROP FUNCTION IF EXISTS upsert_atoms_bulk() CASCADE;
CREATE OR REPLACE FUNCTION upsert_atoms_bulk()
RETURNS BIGINT AS $$
DECLARE
    affected BIGINT;
BEGIN
    -- Expects temp table: _bulk_atoms(id, depth, atom_count, codepoint, x, y, z, m, hilbert_lo, hilbert_hi)
    INSERT INTO atom (id, depth, atom_count, codepoint, geom, hilbert_lo, hilbert_hi)
    SELECT 
        id, depth, atom_count, codepoint,
        ST_SetSRID(ST_MakePointZM(x, y, z, m), 0),
        hilbert_lo, hilbert_hi
    FROM _bulk_atoms
    ON CONFLICT (id) DO UPDATE SET
        geom = EXCLUDED.geom,
        hilbert_lo = EXCLUDED.hilbert_lo,
        hilbert_hi = EXCLUDED.hilbert_hi;
    
    GET DIAGNOSTICS affected = ROW_COUNT;
    DROP TABLE IF EXISTS _bulk_atoms;
    RETURN affected;
END;
$$ LANGUAGE PLPGSQL;

-- Bulk upsert compositions from temp table
DROP FUNCTION IF EXISTS upsert_compositions_bulk() CASCADE;
CREATE OR REPLACE FUNCTION upsert_compositions_bulk()
RETURNS BIGINT AS $$
DECLARE
    affected BIGINT;
BEGIN
    -- Expects temp table: _bulk_compositions(id, label, depth, atom_count, x, y, z, m, hilbert_lo, hilbert_hi)
    INSERT INTO composition (id, label, depth, atom_count, centroid, hilbert_lo, hilbert_hi)
    SELECT 
        id, label, depth, atom_count,
        ST_SetSRID(ST_MakePointZM(x, y, z, m), 0),
        hilbert_lo, hilbert_hi
    FROM _bulk_compositions
    ON CONFLICT (id) DO UPDATE SET
        label = COALESCE(EXCLUDED.label, composition.label),
        centroid = EXCLUDED.centroid,
        hilbert_lo = EXCLUDED.hilbert_lo,
        hilbert_hi = EXCLUDED.hilbert_hi;
    
    GET DIAGNOSTICS affected = ROW_COUNT;
    DROP TABLE IF EXISTS _bulk_compositions;
    RETURN affected;
END;
$$ LANGUAGE PLPGSQL;

-- =============================================================================
-- Batch Semantic Neighbor Queries
-- =============================================================================

-- Get semantic neighbors for multiple tokens at once
DROP FUNCTION IF EXISTS semantic_neighbors_batch(BYTEA[], INTEGER) CASCADE;
CREATE OR REPLACE FUNCTION semantic_neighbors_batch(
    p_token_ids BYTEA[],
    p_k INTEGER DEFAULT 5
)
RETURNS TABLE (
    source_id BYTEA,
    neighbor_id BYTEA,
    neighbor_label TEXT,
    weight DOUBLE PRECISION,
    rank INTEGER
) AS $$
    WITH ranked AS (
        SELECT 
            r.parent_id as source_id,
            r.child_id as neighbor_id,
            c.label as neighbor_label,
            r.weight,
            ROW_NUMBER() OVER (PARTITION BY r.parent_id ORDER BY r.weight DESC) as rn
        FROM relation r
        JOIN composition c ON c.id = r.child_id
        WHERE r.parent_id = ANY(p_token_ids)
          AND r.relation_type = 'S'
    )
    SELECT source_id, neighbor_id, neighbor_label, weight, rn::INTEGER
    FROM ranked
    WHERE rn <= p_k;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- =============================================================================
-- Parallel Hilbert Range Scan (128-bit composite ordering)
-- =============================================================================
-- CRITICAL: Hilbert index is 128-bit = (hilbert_hi << 64) | hilbert_lo
-- Queries MUST use composite ordering on (hilbert_hi, hilbert_lo)
-- Using only hilbert_lo collapses 4D space to 2D!
-- =============================================================================

-- Compute true 128-bit Hilbert distance as NUMERIC
-- Distance = |((hi1 - hi2) << 64) + (lo1 - lo2)|
DROP FUNCTION IF EXISTS hilbert_dist_128(BIGINT, BIGINT, BIGINT, BIGINT) CASCADE;
CREATE OR REPLACE FUNCTION hilbert_dist_128(
    p_hi1 BIGINT, p_lo1 BIGINT,
    p_hi2 BIGINT, p_lo2 BIGINT
)
RETURNS NUMERIC AS $$
    -- Cast to NUMERIC to avoid 64-bit overflow, then compute 128-bit diff
    SELECT ABS(
        (p_hi1::NUMERIC - p_hi2::NUMERIC) * 18446744073709551616::NUMERIC +
        (p_lo1::NUMERIC - p_lo2::NUMERIC)
    );
$$ LANGUAGE SQL IMMUTABLE PARALLEL SAFE;

-- Alternative: Add hilbert_128 computed column 
-- (for databases that support it, or as materialized column)
-- hilbert_128 = (hilbert_hi::NUMERIC * 18446744073709551616::NUMERIC) + hilbert_lo::NUMERIC

-- Optimized Hilbert range query using composite (hi, lo) ordering
DROP FUNCTION IF EXISTS hilbert_range_scan(BIGINT, BIGINT, BIGINT, INTEGER) CASCADE;
CREATE OR REPLACE FUNCTION hilbert_range_scan(
    p_center_lo BIGINT,
    p_center_hi BIGINT,
    p_range BIGINT,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    id BYTEA,
    depth INTEGER,
    hilbert_lo BIGINT,
    hilbert_hi BIGINT,
    hilbert_dist NUMERIC
) AS $$
    -- Use composite index on (hilbert_hi, hilbert_lo) for efficient range scan
    -- ORDER BY must use composite key, not just hilbert_lo!
    SELECT 
        a.id,
        a.depth,
        a.hilbert_lo,
        a.hilbert_hi,
        hilbert_dist_128(a.hilbert_hi, a.hilbert_lo, p_center_hi, p_center_lo) as hilbert_dist
    FROM atom a
    WHERE a.hilbert_hi BETWEEN p_center_hi - 1 AND p_center_hi + 1
      AND a.hilbert_lo BETWEEN p_center_lo - p_range AND p_center_lo + p_range
    ORDER BY (a.hilbert_hi, a.hilbert_lo)  -- Composite ordering!
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- Hilbert nearest neighbors using proper 128-bit distance
DROP FUNCTION IF EXISTS hilbert_knn(BIGINT, BIGINT, INTEGER, INTEGER) CASCADE;
CREATE OR REPLACE FUNCTION hilbert_knn(
    p_center_lo BIGINT,
    p_center_hi BIGINT,
    p_k INTEGER DEFAULT 10,
    p_max_range BIGINT DEFAULT 9223372036854775807  -- 2^63 - 1
)
RETURNS TABLE (
    id BYTEA,
    depth INTEGER,
    hilbert_lo BIGINT,
    hilbert_hi BIGINT,
    hilbert_dist NUMERIC
) AS $$
    -- Full table scan with proper 128-bit distance
    -- For small datasets, this is fine. For large datasets, use hilbert_range_scan
    SELECT 
        a.id,
        a.depth,
        a.hilbert_lo,
        a.hilbert_hi,
        hilbert_dist_128(a.hilbert_hi, a.hilbert_lo, p_center_hi, p_center_lo) as hilbert_dist
    FROM atom a
    ORDER BY hilbert_dist_128(a.hilbert_hi, a.hilbert_lo, p_center_hi, p_center_lo)
    LIMIT p_k;
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- Hilbert range between two 128-bit indices
DROP FUNCTION IF EXISTS hilbert_between(BIGINT, BIGINT, BIGINT, BIGINT) CASCADE;
CREATE OR REPLACE FUNCTION hilbert_between(
    p_lo_min BIGINT, p_hi_min BIGINT,
    p_lo_max BIGINT, p_hi_max BIGINT
)
RETURNS TABLE (
    id BYTEA,
    depth INTEGER,
    hilbert_lo BIGINT,
    hilbert_hi BIGINT
) AS $$
    -- Range query using composite ordering
    SELECT a.id, a.depth, a.hilbert_lo, a.hilbert_hi
    FROM atom a
    WHERE (a.hilbert_hi, a.hilbert_lo) >= (p_hi_min, p_lo_min)
      AND (a.hilbert_hi, a.hilbert_lo) <= (p_hi_max, p_lo_max)
    ORDER BY (a.hilbert_hi, a.hilbert_lo);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

-- =============================================================================
-- Statistics Functions (PARALLEL SAFE)
-- =============================================================================

DROP FUNCTION IF EXISTS db_stats() CASCADE;
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
        (SELECT array_agg(DISTINCT model_name) FROM shape WHERE model_name IS NOT NULL);
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

DROP FUNCTION IF EXISTS db_depth_distribution() CASCADE;
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
$$ LANGUAGE SQL STABLE PARALLEL SAFE;

COMMIT;

-- =============================================================================
-- ANALYZE tables after bulk operations for query planner
-- =============================================================================
ANALYZE atom;
ANALYZE composition;
ANALYZE relation;
ANALYZE bigram_stats;
ANALYZE unigram_stats;
