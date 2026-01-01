-- Ingestion and composition management for the hypercube substrate
-- These functions handle document/content ingestion and Merkle DAG construction

-- Insert a composition into the relation table with its edges
CREATE OR REPLACE FUNCTION hypercube_insert_composition(
    p_child_ids bytea[],
    p_child_is_atom boolean[]
) RETURNS bytea
AS $$
DECLARE
    v_id bytea;
    v_centroid geometry;
    v_hilbert_lo bigint;
    v_hilbert_hi bigint;
    v_depth integer;
    v_atom_count bigint;
    i integer;
    v_child_coords geometry[] := ARRAY[]::geometry[];
    v_child_coord geometry;
    v_child_depth integer;
    v_child_atoms bigint;
    v_max_depth integer := 0;
    v_total_atoms bigint := 0;
    v_concat_data bytea := ''::bytea;
BEGIN
    -- Build hash input and collect metadata
    FOR i IN 1..array_length(p_child_ids, 1)
    LOOP
        -- Concatenate ordinal (4 bytes) + hash (32 bytes)
        v_concat_data := v_concat_data || 
            int4send(i - 1) || 
            p_child_ids[i];
        
        -- Get child coordinates and metadata
        IF p_child_is_atom[i] THEN
            SELECT coords, 1, 1::bigint
            INTO v_child_coord, v_child_depth, v_child_atoms
            FROM atom WHERE id = p_child_ids[i];
        ELSE
            SELECT coords, depth + 1, atom_count
            INTO v_child_coord, v_child_depth, v_child_atoms
            FROM relation WHERE id = p_child_ids[i];
        END IF;
        
        -- Append to array
        v_child_coords := array_append(v_child_coords, v_child_coord);
        
        IF v_child_depth > v_max_depth THEN
            v_max_depth := v_child_depth;
        END IF;
        v_total_atoms := v_total_atoms + COALESCE(v_child_atoms, 0);
    END LOOP;
    
    -- Compute hash
    v_id := hypercube_blake3(v_concat_data);
    
    -- Check if already exists (content-addressed deduplication)
    IF EXISTS (SELECT 1 FROM relation WHERE id = v_id) THEN
        RETURN v_id;
    END IF;
    
    -- Compute 4D centroid manually (ST_Centroid loses Z/M dimensions)
    SELECT 
        ST_SetSRID(ST_MakePoint(
            AVG(ST_X(c)),
            AVG(ST_Y(c)),
            AVG(ST_Z(c)),
            AVG(ST_M(c))
        ), 0)
    INTO v_centroid
    FROM unnest(v_child_coords) AS c;
    
    -- Get Hilbert index
    SELECT h.hilbert_lo, h.hilbert_hi
    INTO v_hilbert_lo, v_hilbert_hi
    FROM hypercube_coords_to_hilbert(
        (ST_X(v_centroid) * 4294967295)::bigint,
        (ST_Y(v_centroid) * 4294967295)::bigint,
        (ST_Z(v_centroid) * 4294967295)::bigint,
        (ST_M(v_centroid) * 4294967295)::bigint
    ) h;
    
    -- Insert relation
    INSERT INTO relation (id, coords, hilbert_lo, hilbert_hi, depth, child_count, atom_count)
    VALUES (v_id, v_centroid, v_hilbert_lo, v_hilbert_hi, v_max_depth, 
            array_length(p_child_ids, 1), v_total_atoms);
    
    -- Insert edges
    FOR i IN 1..array_length(p_child_ids, 1)
    LOOP
        INSERT INTO relation_edge (parent_id, child_id, ordinal, is_atom)
        VALUES (v_id, p_child_ids[i], i - 1, p_child_is_atom[i]);
    END LOOP;
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Ingest a text string as a Merkle tree of compositions
-- Uses hierarchical chunking (characters -> n-grams -> sentences -> paragraphs -> document)
CREATE OR REPLACE FUNCTION hypercube_ingest_text(
    p_text text,
    p_ngram_size integer DEFAULT 3
) RETURNS bytea
AS $$
DECLARE
    v_codepoints integer[];
    v_atom_ids bytea[];
    v_ngram_ids bytea[];
    v_chunk_ids bytea[];
    v_result_id bytea;
    i integer;
    j integer;
    v_ngram_atoms bytea[];
    v_ngram_is_atom boolean[];
    v_chunk_size integer := 32;
BEGIN
    -- Step 1: Convert text to codepoints and get atom IDs
    v_codepoints := ARRAY[]::integer[];
    v_atom_ids := ARRAY[]::bytea[];
    
    FOR i IN 1..length(p_text)
    LOOP
        v_codepoints := array_append(v_codepoints, ascii(substring(p_text, i, 1)));
    END LOOP;
    
    -- Get atom hashes for each codepoint
    SELECT array_agg(id ORDER BY ordinality)
    INTO v_atom_ids
    FROM atom, unnest(v_codepoints) WITH ORDINALITY AS cp(val, ordinality)
    WHERE atom.codepoint = cp.val;
    
    IF array_length(v_atom_ids, 1) IS NULL OR array_length(v_atom_ids, 1) < p_ngram_size THEN
        -- Text too short, return single composition of atoms
        v_ngram_is_atom := array_fill(true, ARRAY[array_length(v_atom_ids, 1)]);
        RETURN hypercube_insert_composition(v_atom_ids, v_ngram_is_atom);
    END IF;
    
    -- Step 2: Create n-gram compositions (sliding window)
    v_ngram_ids := ARRAY[]::bytea[];
    FOR i IN 1..(array_length(v_atom_ids, 1) - p_ngram_size + 1)
    LOOP
        v_ngram_atoms := v_atom_ids[i:i+p_ngram_size-1];
        v_ngram_is_atom := array_fill(true, ARRAY[p_ngram_size]);
        v_ngram_ids := array_append(
            v_ngram_ids, 
            hypercube_insert_composition(v_ngram_atoms, v_ngram_is_atom)
        );
    END LOOP;
    
    IF array_length(v_ngram_ids, 1) <= v_chunk_size THEN
        -- Few enough n-grams, create single parent
        v_ngram_is_atom := array_fill(false, ARRAY[array_length(v_ngram_ids, 1)]);
        RETURN hypercube_insert_composition(v_ngram_ids, v_ngram_is_atom);
    END IF;
    
    -- Step 3: Chunk n-grams into groups and create hierarchy
    v_chunk_ids := ARRAY[]::bytea[];
    i := 1;
    WHILE i <= array_length(v_ngram_ids, 1)
    LOOP
        j := LEAST(i + v_chunk_size - 1, array_length(v_ngram_ids, 1));
        v_ngram_is_atom := array_fill(false, ARRAY[j - i + 1]);
        v_chunk_ids := array_append(
            v_chunk_ids,
            hypercube_insert_composition(v_ngram_ids[i:j], v_ngram_is_atom)
        );
        i := j + 1;
    END LOOP;
    
    -- Create root composition from chunks
    v_ngram_is_atom := array_fill(false, ARRAY[array_length(v_chunk_ids, 1)]);
    v_result_id := hypercube_insert_composition(v_chunk_ids, v_ngram_is_atom);
    
    RETURN v_result_id;
END;
$$ LANGUAGE plpgsql;

-- Retrieve the full text from a composition ID (reverse Merkle traversal)
-- Uses depth-first traversal with proper ordering
CREATE OR REPLACE FUNCTION hypercube_retrieve_text(p_id bytea)
RETURNS text
AS $$
DECLARE
    v_result text := '';
BEGIN
    -- Recursive CTE with path tracking for proper ordering
    WITH RECURSIVE dag AS (
        -- Start with the root
        SELECT 
            p_id as id,
            false as is_atom,
            ARRAY[]::integer[] as path,
            0 as depth
        
        UNION ALL
        
        -- Traverse children in ordinal order
        SELECT 
            e.child_id,
            e.is_atom,
            d.path || e.ordinal,
            d.depth + 1
        FROM dag d
        JOIN relation_edge e ON e.parent_id = d.id
        WHERE NOT d.is_atom AND d.depth < 100  -- Safety limit
    )
    SELECT string_agg(chr(a.codepoint), '' ORDER BY d.path)
    INTO v_result
    FROM dag d
    JOIN atom a ON a.id = d.id
    WHERE d.is_atom;
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql STABLE;

-- Find all compositions containing a specific atom or composition
CREATE OR REPLACE FUNCTION hypercube_find_parents(p_child_id bytea)
RETURNS TABLE(
    parent_id bytea,
    ordinal integer,
    depth integer,
    atom_count bigint
)
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        r.id as parent_id,
        e.ordinal,
        r.depth,
        r.atom_count
    FROM relation_edge e
    JOIN relation r ON r.id = e.parent_id
    WHERE e.child_id = p_child_id
    ORDER BY r.depth DESC, r.atom_count DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Find compositions within a semantic region (4D bounding box)
CREATE OR REPLACE FUNCTION hypercube_find_in_region(
    p_x_min float8, p_x_max float8,
    p_y_min float8, p_y_max float8,
    p_z_min float8, p_z_max float8,
    p_m_min float8, p_m_max float8
) RETURNS TABLE(
    id bytea,
    depth integer,
    atom_count bigint
)
AS $$
DECLARE
    v_bbox geometry;
BEGIN
    -- Create 3D bounding box (PostGIS doesn't support 4D bbox natively)
    v_bbox := ST_MakeEnvelope(p_x_min, p_y_min, p_x_max, p_y_max);
    
    RETURN QUERY
    SELECT 
        r.id,
        r.depth,
        r.atom_count
    FROM relation r
    WHERE r.coords && v_bbox
      AND ST_Z(r.coords) BETWEEN p_z_min AND p_z_max
      AND ST_M(r.coords) BETWEEN p_m_min AND p_m_max
    ORDER BY r.depth DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Compute semantic similarity between two compositions using trajectory comparison
CREATE OR REPLACE FUNCTION hypercube_similarity(id1 bytea, id2 bytea)
RETURNS double precision
AS $$
DECLARE
    v_coords1 geometry[];
    v_coords2 geometry[];
    v_line1 geometry;
    v_line2 geometry;
    v_frechet double precision;
    v_centroid_dist double precision;
BEGIN
    -- Get leaf atom coordinates for each composition
    WITH RECURSIVE dag1 AS (
        SELECT id, is_atom, ordinal FROM (
            SELECT id, true as is_atom, 0 as ordinal FROM atom WHERE id = id1
            UNION ALL
            SELECT e.child_id, e.is_atom, e.ordinal 
            FROM relation_edge e WHERE e.parent_id = id1
        ) t
        UNION ALL
        SELECT e.child_id, e.is_atom, e.ordinal
        FROM dag1 d
        JOIN relation_edge e ON e.parent_id = d.id
        WHERE NOT d.is_atom
    )
    SELECT array_agg(a.coords ORDER BY d.ordinal)
    INTO v_coords1
    FROM dag1 d
    JOIN atom a ON a.id = d.id
    WHERE d.is_atom;
    
    WITH RECURSIVE dag2 AS (
        SELECT id, is_atom, ordinal FROM (
            SELECT id, true as is_atom, 0 as ordinal FROM atom WHERE id = id2
            UNION ALL
            SELECT e.child_id, e.is_atom, e.ordinal 
            FROM relation_edge e WHERE e.parent_id = id2
        ) t
        UNION ALL
        SELECT e.child_id, e.is_atom, e.ordinal
        FROM dag2 d
        JOIN relation_edge e ON e.parent_id = d.id
        WHERE NOT d.is_atom
    )
    SELECT array_agg(a.coords ORDER BY d.ordinal)
    INTO v_coords2
    FROM dag2 d
    JOIN atom a ON a.id = d.id
    WHERE d.is_atom;
    
    -- Create linestrings from coordinates
    IF array_length(v_coords1, 1) >= 2 THEN
        v_line1 := ST_MakeLine(v_coords1);
    END IF;
    
    IF array_length(v_coords2, 1) >= 2 THEN
        v_line2 := ST_MakeLine(v_coords2);
    END IF;
    
    -- Compute Frechet distance if both are valid lines
    IF v_line1 IS NOT NULL AND v_line2 IS NOT NULL THEN
        v_frechet := ST_FrechetDistance(v_line1, v_line2);
    ELSE
        -- Fall back to centroid distance
        SELECT ST_3DDistance(r1.coords, r2.coords)
        INTO v_centroid_dist
        FROM relation r1, relation r2
        WHERE r1.id = id1 AND r2.id = id2;
        
        RETURN 1.0 / (1.0 + COALESCE(v_centroid_dist, 0));
    END IF;
    
    -- Convert distance to similarity (inverse with normalization)
    RETURN 1.0 / (1.0 + v_frechet);
END;
$$ LANGUAGE plpgsql STABLE;

-- Batch ingest multiple texts and return their root IDs
CREATE OR REPLACE FUNCTION hypercube_batch_ingest(
    p_texts text[],
    p_ngram_size integer DEFAULT 3
) RETURNS bytea[]
AS $$
DECLARE
    v_results bytea[];
    v_text text;
BEGIN
    v_results := ARRAY[]::bytea[];
    
    FOREACH v_text IN ARRAY p_texts
    LOOP
        v_results := array_append(
            v_results, 
            hypercube_ingest_text(v_text, p_ngram_size)
        );
    END LOOP;
    
    RETURN v_results;
END;
$$ LANGUAGE plpgsql;

-- Statistics view for compositions
CREATE OR REPLACE VIEW relation_stats AS
SELECT 
    depth,
    COUNT(*) as count,
    AVG(child_count) as avg_children,
    AVG(atom_count) as avg_atoms,
    MIN(atom_count) as min_atoms,
    MAX(atom_count) as max_atoms,
    SUM(atom_count) as total_atoms
FROM relation
GROUP BY depth
ORDER BY depth;
