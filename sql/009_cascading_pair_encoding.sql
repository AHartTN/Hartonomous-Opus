-- Cascading Pair Encoding (CPE) for Hypercube Substrate
-- 
-- Like BPE but applied hierarchically at ingest time:
--   char → pair → merged_pairs → words → phrases → sentences → document
--
-- Key insight: Each tier REDUCES count. "Mississippi" has 11 chars but
-- becomes fewer compositions through pair merging:
--   M-i-s-s-i-s-s-i-p-p-i (11 atoms)
--   → Mi-ss-is-si-pp-i (cascading merges)
--   → Mississippi (1 composition referencing its structure)
--
-- The Merkle DAG preserves the hierarchy - you can always traverse down
-- to recover the original sequence.

BEGIN;

-- =============================================================================
-- TIER 1: Pair Composition (chars → pairs)
-- =============================================================================

-- Create or retrieve a pair composition for two adjacent items
-- This is the fundamental building block - always binary
CREATE OR REPLACE FUNCTION cpe_get_pair(
    p_left_id BYTEA,
    p_right_id BYTEA,
    p_left_is_atom BOOLEAN,
    p_right_is_atom BOOLEAN
) RETURNS BYTEA AS $$
DECLARE
    v_id BYTEA;
    v_concat_data BYTEA;
BEGIN
    -- Build deterministic hash input: ordinal(4) + hash(32) for each child
    v_concat_data := int4send(0) || p_left_id || int4send(1) || p_right_id;
    v_id := hypercube_blake3(v_concat_data);
    
    -- Return existing if found
    IF EXISTS (SELECT 1 FROM relation WHERE id = v_id) THEN
        RETURN v_id;
    END IF;
    
    -- Create new pair composition
    RETURN hypercube_insert_composition(
        ARRAY[p_left_id, p_right_id],
        ARRAY[p_left_is_atom, p_right_is_atom]
    );
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TIER 2: Cascading Merge (reduce sequence by merging pairs)
-- =============================================================================

-- Merge adjacent pairs in a sequence, reducing count by ~half each pass
-- Returns the merged sequence (fewer elements)
CREATE OR REPLACE FUNCTION cpe_merge_pass(
    p_ids BYTEA[],
    p_is_atom BOOLEAN[]
) RETURNS TABLE(ids BYTEA[], is_atom BOOLEAN[]) AS $$
DECLARE
    v_len INTEGER;
    v_result_ids BYTEA[] := ARRAY[]::BYTEA[];
    v_result_is_atom BOOLEAN[] := ARRAY[]::BOOLEAN[];
    v_i INTEGER := 1;
    v_pair_id BYTEA;
BEGIN
    v_len := array_length(p_ids, 1);
    
    IF v_len IS NULL OR v_len = 0 THEN
        ids := p_ids;
        is_atom := p_is_atom;
        RETURN NEXT;
        RETURN;
    END IF;
    
    -- Merge pairs: [a,b,c,d,e] → [ab, cd, e]
    WHILE v_i <= v_len LOOP
        IF v_i < v_len THEN
            -- Merge this pair
            v_pair_id := cpe_get_pair(
                p_ids[v_i], p_ids[v_i + 1],
                p_is_atom[v_i], p_is_atom[v_i + 1]
            );
            v_result_ids := array_append(v_result_ids, v_pair_id);
            v_result_is_atom := array_append(v_result_is_atom, false);
            v_i := v_i + 2;
        ELSE
            -- Odd element - keep as-is
            v_result_ids := array_append(v_result_ids, p_ids[v_i]);
            v_result_is_atom := array_append(v_result_is_atom, p_is_atom[v_i]);
            v_i := v_i + 1;
        END IF;
    END LOOP;
    
    ids := v_result_ids;
    is_atom := v_result_is_atom;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Cascade merges until we have a single root or hit a depth limit
CREATE OR REPLACE FUNCTION cpe_cascade_to_root(
    p_ids BYTEA[],
    p_is_atom BOOLEAN[],
    p_max_depth INTEGER DEFAULT 64
) RETURNS BYTEA AS $$
DECLARE
    v_ids BYTEA[] := p_ids;
    v_is_atom BOOLEAN[] := p_is_atom;
    v_new_ids BYTEA[];
    v_new_is_atom BOOLEAN[];
    v_depth INTEGER := 0;
BEGIN
    -- Handle edge cases
    IF v_ids IS NULL OR array_length(v_ids, 1) IS NULL THEN
        RETURN NULL;
    END IF;
    
    IF array_length(v_ids, 1) = 1 THEN
        RETURN v_ids[1];
    END IF;
    
    -- Cascade until single root
    WHILE array_length(v_ids, 1) > 1 AND v_depth < p_max_depth LOOP
        SELECT m.ids, m.is_atom 
        INTO v_new_ids, v_new_is_atom
        FROM cpe_merge_pass(v_ids, v_is_atom) m;
        
        v_ids := v_new_ids;
        v_is_atom := v_new_is_atom;
        v_depth := v_depth + 1;
    END LOOP;
    
    RETURN v_ids[1];
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TIER 3: Text Ingestion with CPE
-- =============================================================================

-- Convert text to atom IDs (codepoints → atom hashes)
CREATE OR REPLACE FUNCTION cpe_text_to_atoms(p_text TEXT)
RETURNS TABLE(id BYTEA, is_atom BOOLEAN) AS $$
BEGIN
    RETURN QUERY
    SELECT a.id::BYTEA, true::BOOLEAN
    FROM (
        SELECT i, ascii(substring(p_text, i, 1)) as cp
        FROM generate_series(1, length(p_text)) as i
    ) chars
    JOIN atom a ON a.codepoint = chars.cp
    ORDER BY chars.i;
END;
$$ LANGUAGE plpgsql STABLE;

-- Main ingestion function: text → cascaded composition
CREATE OR REPLACE FUNCTION cpe_ingest_text(p_text TEXT)
RETURNS BYTEA AS $$
DECLARE
    v_ids BYTEA[];
    v_is_atom BOOLEAN[];
BEGIN
    -- Handle empty/null
    IF p_text IS NULL OR length(p_text) = 0 THEN
        RETURN NULL;
    END IF;
    
    -- Get atoms for all characters
    SELECT array_agg(t.id), array_agg(t.is_atom)
    INTO v_ids, v_is_atom
    FROM cpe_text_to_atoms(p_text) t;
    
    -- Handle missing atoms (characters not in Unicode seed)
    IF v_ids IS NULL OR array_length(v_ids, 1) IS NULL THEN
        RETURN NULL;
    END IF;
    
    -- Single character - return atom directly
    IF array_length(v_ids, 1) = 1 THEN
        RETURN v_ids[1];
    END IF;
    
    -- Cascade to root
    RETURN cpe_cascade_to_root(v_ids, v_is_atom);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TIER 4: Chunked Ingestion for Large Files
-- =============================================================================

-- Ingest large text in chunks, then cascade the chunks
-- This prevents O(n²) behavior on huge files
CREATE OR REPLACE FUNCTION cpe_ingest_large_text(
    p_text TEXT,
    p_chunk_size INTEGER DEFAULT 4096
) RETURNS BYTEA AS $$
DECLARE
    v_len INTEGER;
    v_chunk_ids BYTEA[] := ARRAY[]::BYTEA[];
    v_chunk_is_atom BOOLEAN[] := ARRAY[]::BOOLEAN[];
    v_i INTEGER := 1;
    v_chunk TEXT;
    v_chunk_id BYTEA;
BEGIN
    v_len := length(p_text);
    
    -- Small enough to ingest directly
    IF v_len <= p_chunk_size THEN
        RETURN cpe_ingest_text(p_text);
    END IF;
    
    -- Chunk the text and ingest each chunk
    WHILE v_i <= v_len LOOP
        v_chunk := substring(p_text FROM v_i FOR p_chunk_size);
        v_chunk_id := cpe_ingest_text(v_chunk);
        
        IF v_chunk_id IS NOT NULL THEN
            v_chunk_ids := array_append(v_chunk_ids, v_chunk_id);
            v_chunk_is_atom := array_append(v_chunk_is_atom, false);
        END IF;
        
        v_i := v_i + p_chunk_size;
    END LOOP;
    
    -- Cascade the chunk roots
    IF array_length(v_chunk_ids, 1) = 1 THEN
        RETURN v_chunk_ids[1];
    END IF;
    
    RETURN cpe_cascade_to_root(v_chunk_ids, v_chunk_is_atom);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TIER 5: Document Ingestion (respects semantic boundaries)
-- =============================================================================

-- Ingest a document by splitting on natural boundaries (paragraphs, sentences)
-- then cascading up: sentences → paragraphs → sections → document
CREATE OR REPLACE FUNCTION cpe_ingest_document(p_text TEXT)
RETURNS BYTEA AS $$
DECLARE
    v_paragraphs TEXT[];
    v_para_ids BYTEA[] := ARRAY[]::BYTEA[];
    v_para_is_atom BOOLEAN[] := ARRAY[]::BOOLEAN[];
    v_para TEXT;
    v_para_id BYTEA;
BEGIN
    -- Split on paragraph boundaries (double newline or blank lines)
    v_paragraphs := regexp_split_to_array(p_text, E'\n\\s*\n');
    
    -- Ingest each paragraph
    FOREACH v_para IN ARRAY v_paragraphs LOOP
        v_para := trim(v_para);
        IF length(v_para) > 0 THEN
            v_para_id := cpe_ingest_large_text(v_para);
            IF v_para_id IS NOT NULL THEN
                v_para_ids := array_append(v_para_ids, v_para_id);
                v_para_is_atom := array_append(v_para_is_atom, false);
            END IF;
        END IF;
    END LOOP;
    
    -- Handle empty or single paragraph
    IF array_length(v_para_ids, 1) IS NULL THEN
        RETURN NULL;
    END IF;
    
    IF array_length(v_para_ids, 1) = 1 THEN
        RETURN v_para_ids[1];
    END IF;
    
    -- Cascade paragraphs to document root
    RETURN cpe_cascade_to_root(v_para_ids, v_para_is_atom);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- UTILITY: Reconstruct text from composition
-- =============================================================================

CREATE OR REPLACE FUNCTION cpe_reconstruct_text(p_id BYTEA)
RETURNS TEXT AS $$
BEGIN
    -- Use existing retrieval function
    RETURN hypercube_retrieve_text(p_id);
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- UTILITY: Show composition structure (for debugging)
-- =============================================================================

CREATE OR REPLACE FUNCTION cpe_show_tree(p_id BYTEA, p_max_depth INTEGER DEFAULT 5)
RETURNS TABLE(
    depth INTEGER,
    path TEXT,
    id_hex TEXT,
    content TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE tree AS (
        SELECT 
            0 as lvl,
            ''::TEXT as pth,
            p_id as nid,
            EXISTS(SELECT 1 FROM atom WHERE atom.id::BYTEA = p_id) as is_atm
        
        UNION ALL
        
        SELECT 
            t.lvl + 1,
            t.pth || '/' || e.ordinal::TEXT,
            e.child_id::BYTEA,
            e.is_atom
        FROM tree t
        JOIN relation_edge e ON e.parent_id::BYTEA = t.nid
        WHERE NOT t.is_atm AND t.lvl < p_max_depth
    )
    SELECT 
        t.lvl as depth,
        COALESCE(NULLIF(t.pth, ''), '/') as path,
        encode(t.nid, 'hex')::TEXT as id_hex,
        CASE 
            WHEN t.is_atm THEN chr(a.codepoint)
            ELSE '['||r.child_count::TEXT||' children, '||r.atom_count::TEXT||' atoms]'
        END as content
    FROM tree t
    LEFT JOIN atom a ON a.id::BYTEA = t.nid AND t.is_atm
    LEFT JOIN relation r ON r.id::BYTEA = t.nid AND NOT t.is_atm
    ORDER BY t.pth;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- STATS
-- =============================================================================

CREATE OR REPLACE VIEW cpe_stats AS
SELECT 
    depth,
    COUNT(*) as compositions,
    SUM(child_count) as total_children,
    SUM(atom_count) as total_atoms,
    AVG(child_count)::NUMERIC(10,2) as avg_children
FROM relation
GROUP BY depth
ORDER BY depth;

COMMIT;
