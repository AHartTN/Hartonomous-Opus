-- =============================================================================
-- Hartonomous Hypercube - Core Functions v4
-- =============================================================================
-- Minimal SQL functions. Heavy operations use C extensions.
-- =============================================================================

BEGIN;

-- =============================================================================
-- Basic Accessors (fast, indexed)
-- =============================================================================

-- Check if atom is a leaf
CREATE OR REPLACE FUNCTION atom_is_leaf(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT depth = 0 FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- Get 4D centroid (uses pre-computed column)
CREATE OR REPLACE FUNCTION atom_centroid(p_id BYTEA)
RETURNS TABLE(x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION, m DOUBLE PRECISION) AS $$
    SELECT ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid)
    FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- Get children of a composition
CREATE OR REPLACE FUNCTION atom_children(p_id BYTEA)
RETURNS TABLE(child_id BYTEA, ordinal INTEGER) AS $$
    SELECT unnest(children), generate_series(1, array_length(children, 1))
    FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- Get child count
CREATE OR REPLACE FUNCTION atom_child_count(p_id BYTEA)
RETURNS INTEGER AS $$
    SELECT COALESCE(array_length(children, 1), 0) FROM atom WHERE id = p_id;
$$ LANGUAGE SQL STABLE;

-- Lookup by codepoint
CREATE OR REPLACE FUNCTION atom_by_codepoint(p_cp INTEGER)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE codepoint = p_cp;
$$ LANGUAGE SQL STABLE;

-- Check if hash exists
CREATE OR REPLACE FUNCTION atom_exists(p_id BYTEA)
RETURNS BOOLEAN AS $$
    SELECT EXISTS(SELECT 1 FROM atom WHERE id = p_id);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Text Reconstruction
-- =============================================================================

-- Reconstruct bytes from composition (recursive)
-- Reconstruct bytes from composition (recursive, handles RLE)
-- RLE compositions have 1 child but atom_count > 1, meaning child is repeated
CREATE OR REPLACE FUNCTION atom_reconstruct(p_id BYTEA) RETURNS BYTEA AS $$
WITH RECURSIVE tree AS (
    -- Base case: start with the root
    SELECT 
        id, 
        children, 
        value, 
        atom_count,
        ARRAY[]::INTEGER[] AS path,
        1 AS repeat_idx,
        -- RLE: if 1 child but atom_count > 1, need to repeat
        CASE 
            WHEN array_length(children, 1) = 1 AND atom_count > 1 
            THEN atom_count 
            ELSE 1 
        END AS repeat_count
    FROM atom WHERE id = p_id
    
    UNION ALL
    
    -- Recursive: expand children, handling RLE repetition
    SELECT 
        a.id, 
        a.children, 
        a.value,
        a.atom_count,
        t.path || c.ord::INTEGER,
        1 AS repeat_idx,
        CASE 
            WHEN array_length(a.children, 1) = 1 AND a.atom_count > 1 
            THEN a.atom_count 
            ELSE 1 
        END AS repeat_count
    FROM tree t
    CROSS JOIN LATERAL (
        -- For RLE (1 child, atom_count > 1), repeat the child atom_count times
        SELECT t.children[1] AS child_id, g.n AS ord
        FROM generate_series(1, 
            CASE 
                WHEN array_length(t.children, 1) = 1 AND t.atom_count > 1 
                THEN t.atom_count::INTEGER
                ELSE array_length(t.children, 1) 
            END
        ) g(n)
        WHERE array_length(t.children, 1) = 1 AND t.atom_count > 1
        
        UNION ALL
        
        -- Normal: unnest children as usual
        SELECT child_id, ord::INTEGER
        FROM unnest(t.children) WITH ORDINALITY AS u(child_id, ord)
        WHERE array_length(t.children, 1) != 1 OR t.atom_count = 1
    ) c
    JOIN atom a ON a.id = c.child_id
    WHERE t.children IS NOT NULL
)
SELECT string_agg(value, ''::BYTEA ORDER BY path)
FROM tree WHERE value IS NOT NULL;
$$ LANGUAGE SQL STABLE;

-- Reconstruct as UTF-8 text
CREATE OR REPLACE FUNCTION atom_text(p_id BYTEA) RETURNS TEXT AS $$
DECLARE
    v_bytes BYTEA;
BEGIN
    v_bytes := atom_reconstruct(p_id);
    IF v_bytes IS NULL THEN RETURN NULL; END IF;
    
    BEGIN
        RETURN convert_from(v_bytes, 'UTF8');
    EXCEPTION WHEN OTHERS THEN
        RETURN encode(v_bytes, 'hex');
    END;
END;
$$ LANGUAGE plpgsql STABLE;

-- Alias for backwards compatibility
CREATE OR REPLACE FUNCTION atom_reconstruct_text(p_id BYTEA) RETURNS TEXT AS $$
    SELECT atom_text(p_id);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Distance Functions (simple SQL - C versions are faster for batch)
-- =============================================================================

-- 4D Euclidean distance using pre-computed centroids
CREATE OR REPLACE FUNCTION atom_distance(p_id1 BYTEA, p_id2 BYTEA)
RETURNS DOUBLE PRECISION AS $$
    SELECT sqrt(
        power(ST_X(a.centroid) - ST_X(b.centroid), 2) +
        power(ST_Y(a.centroid) - ST_Y(b.centroid), 2) +
        power(ST_Z(a.centroid) - ST_Z(b.centroid), 2) +
        power(ST_M(a.centroid) - ST_M(b.centroid), 2)
    )
    FROM atom a, atom b
    WHERE a.id = p_id1 AND b.id = p_id2;
$$ LANGUAGE SQL STABLE;

-- K-nearest neighbors (uses GIST index on centroid)
CREATE OR REPLACE FUNCTION atom_knn(p_id BYTEA, p_k INTEGER DEFAULT 10)
RETURNS TABLE(neighbor_id BYTEA, distance DOUBLE PRECISION, depth INTEGER) AS $$
    WITH target AS (SELECT centroid FROM atom WHERE id = p_id)
    SELECT a.id, a.centroid <-> t.centroid, a.depth
    FROM atom a, target t
    WHERE a.id != p_id
    ORDER BY a.centroid <-> t.centroid
    LIMIT p_k;
$$ LANGUAGE SQL STABLE;

-- Find atoms within Hilbert range (uses NUMERIC to avoid overflow)
CREATE OR REPLACE FUNCTION atom_hilbert_range(p_id BYTEA, p_range BIGINT, p_limit INTEGER DEFAULT 100)
RETURNS TABLE(neighbor_id BYTEA, hilbert_dist NUMERIC) AS $$
    WITH t AS (SELECT hilbert_lo::NUMERIC AS lo, hilbert_hi::NUMERIC AS hi FROM atom WHERE id = p_id)
    SELECT a.id, ABS(a.hilbert_lo::NUMERIC - t.lo)
    FROM atom a, t
    WHERE a.id != p_id
      AND ABS(a.hilbert_hi::NUMERIC - t.hi) <= 1
      AND ABS(a.hilbert_lo::NUMERIC - t.lo) <= p_range::NUMERIC
    ORDER BY ABS(a.hilbert_lo::NUMERIC - t.lo)
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Composition Queries
-- =============================================================================

-- Find parents containing a child
CREATE OR REPLACE FUNCTION atom_parents(p_child_id BYTEA, p_limit INTEGER DEFAULT 20)
RETURNS TABLE(parent_id BYTEA, depth INTEGER, atom_count BIGINT) AS $$
    SELECT id, depth, atom_count
    FROM atom
    WHERE p_child_id = ANY(children)
    ORDER BY depth
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Find by text pattern (slow - use for small datasets)
CREATE OR REPLACE FUNCTION atom_search(p_pattern TEXT, p_limit INTEGER DEFAULT 20)
RETURNS TABLE(id BYTEA, depth INTEGER, atom_count BIGINT, content TEXT) AS $$
    SELECT a.id, a.depth, a.atom_count, atom_text(a.id)
    FROM atom a
    WHERE a.depth > 0 
      AND a.atom_count <= 100
      AND atom_text(a.id) LIKE '%' || p_pattern || '%'
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Find exact text
CREATE OR REPLACE FUNCTION atom_find(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT id FROM atom
    WHERE depth > 0 AND atom_text(id) = p_text
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Statistics Views
-- =============================================================================

CREATE OR REPLACE VIEW v_atom_stats AS
SELECT
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE depth = 0) AS leaves,
    COUNT(*) FILTER (WHERE depth > 0) AS compositions,
    MAX(depth) AS max_depth,
    pg_size_pretty(pg_total_relation_size('atom')) AS size
FROM atom;

CREATE OR REPLACE VIEW v_depth_stats AS
SELECT depth, COUNT(*) AS count, AVG(atom_count)::NUMERIC(10,2) AS avg_atoms
FROM atom GROUP BY depth ORDER BY depth;

-- Quick stats function
CREATE OR REPLACE FUNCTION stats()
RETURNS TABLE(total BIGINT, leaves BIGINT, compositions BIGINT, max_depth INTEGER, size TEXT) AS $$
    SELECT total, leaves, compositions, max_depth, size FROM v_atom_stats;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Legacy Views (for backwards compatibility with tests)
-- =============================================================================

CREATE OR REPLACE VIEW atom_stats AS
SELECT
    depth,
    COUNT(*) as count,
    AVG(atom_count)::NUMERIC(10,2) as avg_atoms,
    SUM(atom_count) as total_atoms,
    pg_size_pretty(pg_total_relation_size('atom')) as table_size
FROM atom
GROUP BY depth
ORDER BY depth;

CREATE OR REPLACE VIEW atom_type_stats AS
SELECT
    ST_GeometryType(geom) as geom_type,
    depth,
    COUNT(*) as count
FROM atom
GROUP BY ST_GeometryType(geom), depth
ORDER BY depth, geom_type;

-- =============================================================================
-- Legacy Neighbor Functions (for backwards compatibility with tests)
-- =============================================================================

-- Find nearest neighbors by Hilbert index (fast approximate)
CREATE OR REPLACE FUNCTION atom_nearest_hilbert(p_id BYTEA, p_limit INTEGER DEFAULT 10)
RETURNS TABLE(neighbor_id BYTEA, hilbert_distance NUMERIC) AS $$
    WITH target AS (
        SELECT hilbert_lo, hilbert_hi FROM atom WHERE id = p_id
    )
    SELECT
        a.id,
        ABS(a.hilbert_lo::NUMERIC - t.hilbert_lo::NUMERIC) +
        ABS(a.hilbert_hi::NUMERIC - t.hilbert_hi::NUMERIC) * 9223372036854775808::NUMERIC
    FROM atom a, target t
    WHERE a.id != p_id
    ORDER BY ABS(a.hilbert_hi::NUMERIC - t.hilbert_hi::NUMERIC),
             ABS(a.hilbert_lo::NUMERIC - t.hilbert_lo::NUMERIC)
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- Find nearest neighbors by spatial distance (accurate)
CREATE OR REPLACE FUNCTION atom_nearest_spatial(p_id BYTEA, p_limit INTEGER DEFAULT 10)
RETURNS TABLE(neighbor_id BYTEA, distance DOUBLE PRECISION) AS $$
    SELECT a.id, a.centroid <-> t.centroid
    FROM atom a, (SELECT centroid FROM atom WHERE id = p_id) t
    WHERE a.id != p_id
    ORDER BY a.centroid <-> t.centroid
    LIMIT p_limit;
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Convenience Aliases
-- =============================================================================

CREATE OR REPLACE FUNCTION txt(p_id BYTEA) RETURNS TEXT AS $$
    SELECT atom_text(p_id);
$$ LANGUAGE SQL STABLE;

CREATE OR REPLACE FUNCTION knn(p_id BYTEA, p_k INTEGER DEFAULT 10)
RETURNS TABLE(id BYTEA, dist DOUBLE PRECISION) AS $$
    SELECT neighbor_id, distance FROM atom_knn(p_id, p_k);
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- Content Hash Lookup (CPE-compatible)
-- =============================================================================

-- Compute the CPE cascade hash for a text string
-- This matches how CPE builds composition hashes deterministically
-- NOTE: Uses little-endian ordinals to match C++ implementation
CREATE OR REPLACE FUNCTION atom_content_hash(p_text TEXT)
RETURNS BYTEA AS $$
DECLARE
    v_codepoints INTEGER[];
    v_hashes BYTEA[];
    v_hash BYTEA;
    v_left BYTEA;
    v_right BYTEA;
    -- Little-endian ordinal constants (matches C++ native byte order)
    v_ord0 BYTEA := '\x00000000'::BYTEA;  -- 0 as uint32 little-endian
    v_ord1 BYTEA := '\x01000000'::BYTEA;  -- 1 as uint32 little-endian
BEGIN
    -- Convert text to codepoints and get atom hashes
    SELECT array_agg(codepoint ORDER BY ordinality), array_agg(id ORDER BY ordinality)
    INTO v_codepoints, v_hashes
    FROM (
        SELECT ascii(chr) as codepoint, a.id, ordinality
        FROM unnest(string_to_array(p_text, NULL)) WITH ORDINALITY AS chars(chr, ordinality)
        JOIN atom a ON a.codepoint = ascii(chr)
    ) t;

    IF v_hashes IS NULL OR array_length(v_hashes, 1) = 0 THEN
        RETURN NULL;
    END IF;

    IF array_length(v_hashes, 1) = 1 THEN
        RETURN v_hashes[1];
    END IF;

    -- CPE cascade: merge pairs until single root
    WHILE array_length(v_hashes, 1) > 1 LOOP
        DECLARE
            v_merged BYTEA[] := ARRAY[]::BYTEA[];
            v_len INTEGER := array_length(v_hashes, 1);
            v_i INTEGER := 1;
        BEGIN
            WHILE v_i <= v_len LOOP
                IF v_i + 1 <= v_len THEN
                    -- Merge pair: BLAKE3(ordinal0 || left_hash || ordinal1 || right_hash)
                    -- Ordinals use little-endian to match C++ implementation
                    v_left := v_hashes[v_i];
                    v_right := v_hashes[v_i + 1];
                    v_hash := hypercube_blake3(
                        v_ord0 || v_left || v_ord1 || v_right
                    );
                    v_merged := v_merged || v_hash;
                    v_i := v_i + 2;
                ELSE
                    -- Odd element - carry forward
                    v_merged := v_merged || v_hashes[v_i];
                    v_i := v_i + 1;
                END IF;
            END LOOP;
            v_hashes := v_merged;
        END;
    END LOOP;

    RETURN v_hashes[1];
END;
$$ LANGUAGE plpgsql STABLE;

-- Find composition by content (using CPE hash)
CREATE OR REPLACE FUNCTION atom_find(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT id FROM atom WHERE id = atom_content_hash(p_text);
$$ LANGUAGE SQL STABLE;

-- Alias for backwards compatibility
CREATE OR REPLACE FUNCTION atom_find_exact(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT atom_find(p_text);
$$ LANGUAGE SQL STABLE;

-- Get text and verify it exists
CREATE OR REPLACE FUNCTION atom_get(p_text TEXT)
RETURNS TABLE(id BYTEA, content TEXT, depth INTEGER, atom_count BIGINT) AS $$
    SELECT a.id, atom_text(a.id), a.depth, a.atom_count
    FROM atom a
    WHERE a.id = atom_content_hash(p_text);
$$ LANGUAGE SQL STABLE;

COMMIT;
