-- Additional functions for trajectory operations and composition handling
-- These extend the base hypercube extension with semantic query capabilities

-- Create a LINESTRINGZM from a sequence of atoms
CREATE OR REPLACE FUNCTION hypercube_atoms_to_linestring(codepoints integer[])
RETURNS geometry(LINESTRINGZM, 0)
AS $$
DECLARE
    points geometry[];
    cp integer;
    atom_record record;
BEGIN
    points := ARRAY[]::geometry[];
    
    FOREACH cp IN ARRAY codepoints
    LOOP
        SELECT coords INTO atom_record FROM atom WHERE codepoint = cp;
        IF FOUND THEN
            points := array_append(points, atom_record.coords);
        END IF;
    END LOOP;
    
    IF array_length(points, 1) < 2 THEN
        RETURN NULL;
    END IF;
    
    RETURN ST_MakeLine(points);
END;
$$ LANGUAGE plpgsql STABLE;

-- Convert text to LINESTRINGZM trajectory
CREATE OR REPLACE FUNCTION hypercube_text_to_linestring(input_text text)
RETURNS geometry(LINESTRINGZM, 0)
AS $$
DECLARE
    codepoints integer[];
    i integer;
BEGIN
    codepoints := ARRAY[]::integer[];
    
    FOR i IN 1..length(input_text)
    LOOP
        codepoints := array_append(codepoints, ascii(substring(input_text, i, 1)));
    END LOOP;
    
    RETURN hypercube_atoms_to_linestring(codepoints);
END;
$$ LANGUAGE plpgsql STABLE;

-- Create a composition (Merkle node) from child hashes
CREATE OR REPLACE FUNCTION hypercube_create_composition(
    child_ids bytea[],
    child_ordinals integer[]
) RETURNS TABLE(
    id bytea,
    centroid_x bigint,
    centroid_y bigint,
    centroid_z bigint,
    centroid_m bigint,
    hilbert_lo bigint,
    hilbert_hi bigint,
    depth integer,
    child_count integer,
    atom_count bigint
)
AS $$
DECLARE
    concat_data bytea;
    computed_hash bytea;
    i integer;
    v_child_coords geometry[] := ARRAY[]::geometry[];
    v_child_coord geometry;
    v_child_depth integer;
    v_child_atoms bigint;
    centroid_geom geometry;
    max_depth integer := 0;
    total_atoms bigint := 0;
BEGIN
    -- Build concatenated hash data (ordinal + hash pairs)
    concat_data := ''::bytea;
    
    FOR i IN 1..array_length(child_ids, 1)
    LOOP
        -- Add ordinal as 4 bytes little-endian
        concat_data := concat_data || 
            int4send(child_ordinals[i]) || 
            child_ids[i];
        
        -- Get child coordinates (either from atom or relation table)
        SELECT coords, 1, 1::bigint
        INTO v_child_coord, v_child_depth, v_child_atoms
        FROM atom WHERE id = child_ids[i];
        
        IF NOT FOUND THEN
            SELECT coords, r.depth + 1, r.atom_count
            INTO v_child_coord, v_child_depth, v_child_atoms
            FROM relation r WHERE r.id = child_ids[i];
        END IF;
        
        v_child_coords := array_append(v_child_coords, v_child_coord);
        
        IF v_child_depth > max_depth THEN
            max_depth := v_child_depth;
        END IF;
        total_atoms := total_atoms + COALESCE(v_child_atoms, 0);
    END LOOP;
    
    -- Compute BLAKE3 hash
    computed_hash := hypercube_blake3(concat_data);
    
    -- Compute centroid
    centroid_geom := ST_Centroid(ST_Collect(v_child_coords));
    
    -- Get Hilbert index for centroid
    WITH centroid_coords AS (
        SELECT 
            (ST_X(centroid_geom) * 4294967295)::bigint as cx,
            (ST_Y(centroid_geom) * 4294967295)::bigint as cy,
            (ST_Z(centroid_geom) * 4294967295)::bigint as cz,
            (ST_M(centroid_geom) * 4294967295)::bigint as cm
    )
    SELECT 
        computed_hash,
        cx, cy, cz, cm,
        h.hilbert_lo, h.hilbert_hi,
        max_depth,
        array_length(child_ids, 1),
        total_atoms
    FROM centroid_coords, 
         LATERAL hypercube_coords_to_hilbert(cx, cy, cz, cm) h
    INTO id, centroid_x, centroid_y, centroid_z, centroid_m, 
         hilbert_lo, hilbert_hi, depth, child_count, atom_count;
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql STABLE;

-- Find semantically similar atoms using Hilbert range query
CREATE OR REPLACE FUNCTION hypercube_find_neighbors(
    center_codepoint integer,
    hilbert_range bigint DEFAULT 1000000
) RETURNS TABLE(
    codepoint integer,
    category atom_category,
    hilbert_distance bigint
)
AS $$
DECLARE
    center_hi bigint;
    center_lo bigint;
BEGIN
    -- Get center point's Hilbert index
    SELECT a.hilbert_hi, a.hilbert_lo 
    INTO center_hi, center_lo
    FROM atom a 
    WHERE a.codepoint = center_codepoint;
    
    IF NOT FOUND THEN
        RETURN;
    END IF;
    
    -- Find neighbors within Hilbert range
    RETURN QUERY
    SELECT 
        a.codepoint,
        a.category,
        ABS(a.hilbert_lo - center_lo) as hilbert_distance
    FROM atom a
    WHERE a.hilbert_hi = center_hi
      AND ABS(a.hilbert_lo - center_lo) <= hilbert_range
      AND a.codepoint != center_codepoint
    ORDER BY hilbert_distance
    LIMIT 100;
END;
$$ LANGUAGE plpgsql STABLE;

-- Frechet distance between two text sequences (using PostGIS)
CREATE OR REPLACE FUNCTION hypercube_frechet_distance(text1 text, text2 text)
RETURNS double precision
AS $$
DECLARE
    line1 geometry;
    line2 geometry;
BEGIN
    line1 := hypercube_text_to_linestring(text1);
    line2 := hypercube_text_to_linestring(text2);
    
    IF line1 IS NULL OR line2 IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN ST_FrechetDistance(line1, line2);
END;
$$ LANGUAGE plpgsql STABLE;

-- Semantic distance between two compositions/atoms
CREATE OR REPLACE FUNCTION hypercube_semantic_distance(id1 bytea, id2 bytea)
RETURNS double precision
AS $$
DECLARE
    coords1 geometry;
    coords2 geometry;
BEGIN
    -- Try atom table first
    SELECT coords INTO coords1 FROM atom WHERE id = id1;
    IF NOT FOUND THEN
        SELECT coords INTO coords1 FROM relation WHERE id = id1;
    END IF;
    
    SELECT coords INTO coords2 FROM atom WHERE id = id2;
    IF NOT FOUND THEN
        SELECT coords INTO coords2 FROM relation WHERE id = id2;
    END IF;
    
    IF coords1 IS NULL OR coords2 IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN ST_3DDistance(coords1, coords2);
END;
$$ LANGUAGE plpgsql STABLE;

-- Create index on relation_edge for efficient child lookups
CREATE INDEX IF NOT EXISTS idx_relation_edge_parent_ordinal 
ON relation_edge(parent_id, ordinal);

-- Create spatial index on relation coords
CREATE INDEX IF NOT EXISTS idx_relation_coords_gist 
ON relation USING GIST(coords);

-- Materialized view for category statistics (refresh periodically)
CREATE MATERIALIZED VIEW IF NOT EXISTS atom_category_stats AS
SELECT 
    category,
    COUNT(*) as atom_count,
    ST_Extent(coords::geometry) as bounding_box,
    AVG(ST_X(coords)) as avg_x,
    AVG(ST_Y(coords)) as avg_y,
    AVG(ST_Z(coords)) as avg_z,
    AVG(ST_M(coords)) as avg_m,
    MIN(hilbert_lo) as min_hilbert_lo,
    MAX(hilbert_lo) as max_hilbert_lo
FROM atom
GROUP BY category;

CREATE UNIQUE INDEX IF NOT EXISTS idx_atom_category_stats_cat 
ON atom_category_stats(category);

-- Refresh function
CREATE OR REPLACE FUNCTION hypercube_refresh_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY atom_category_stats;
END;
$$ LANGUAGE plpgsql;
