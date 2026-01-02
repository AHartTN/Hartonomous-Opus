-- BPE-style Tokenizer for Hypercube Substrate
-- Proper hierarchical composition: "Hel" = "He" + "l", not "H" + "e" + "l"
--
-- Key principle: We learn which pairs to merge from frequency,
-- then apply those merges hierarchically during ingestion.
-- Each merge creates a composition that references its two children.

BEGIN;

-- Table to store learned merge rules (BPE vocabulary)
-- Priority determines merge order (lower = merge first = more frequent)
CREATE TABLE IF NOT EXISTS bpe_merge (
    id              SERIAL PRIMARY KEY,
    left_id         BYTEA NOT NULL,      -- Hash of left child (atom or composition)
    right_id        BYTEA NOT NULL,      -- Hash of right child
    merged_id       BYTEA NOT NULL,      -- Hash of resulting composition
    frequency       BIGINT NOT NULL DEFAULT 0,
    priority        INTEGER NOT NULL,    -- Lower = higher priority = merge first
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    UNIQUE(left_id, right_id)
);

CREATE INDEX IF NOT EXISTS idx_bpe_merge_priority ON bpe_merge(priority);
CREATE INDEX IF NOT EXISTS idx_bpe_merge_left ON bpe_merge(left_id);
CREATE INDEX IF NOT EXISTS idx_bpe_merge_right ON bpe_merge(right_id);
CREATE INDEX IF NOT EXISTS idx_bpe_merge_merged ON bpe_merge(merged_id);

-- Composition cache for fast lookup by sequence
-- Maps a sequence of atom/composition IDs to its merged composition ID
CREATE TABLE IF NOT EXISTS composition_cache (
    sequence_hash   BYTEA PRIMARY KEY,   -- Hash of concatenated child IDs
    composition_id  BYTEA NOT NULL,      -- Resulting composition in relation table
    depth           INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Count pair frequencies in a text corpus (for learning merges)
CREATE OR REPLACE FUNCTION bpe_count_pairs(p_text TEXT)
RETURNS TABLE(
    left_cp INTEGER,
    right_cp INTEGER,
    pair_count BIGINT
) AS $$
DECLARE
    len INTEGER;
BEGIN
    len := length(p_text);
    
    RETURN QUERY
    WITH chars AS (
        SELECT 
            i,
            ascii(substring(p_text, i, 1)) as cp
        FROM generate_series(1, len) as i
    )
    SELECT 
        c1.cp as left_cp,
        c2.cp as right_cp,
        COUNT(*)::BIGINT as pair_count
    FROM chars c1
    JOIN chars c2 ON c2.i = c1.i + 1
    GROUP BY c1.cp, c2.cp
    ORDER BY COUNT(*) DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Learn BPE merges from a corpus (batch of texts)
-- This is the training phase - run once on representative data
CREATE OR REPLACE FUNCTION bpe_learn_merges(
    p_texts TEXT[],
    p_num_merges INTEGER DEFAULT 10000,
    p_min_frequency INTEGER DEFAULT 2
) RETURNS INTEGER AS $$
DECLARE
    v_text TEXT;
    v_merge_count INTEGER := 0;
    v_left_id BYTEA;
    v_right_id BYTEA;
    v_merged_id BYTEA;
    v_priority INTEGER := 1;
BEGIN
    -- Create temp table to accumulate pair counts
    CREATE TEMP TABLE IF NOT EXISTS pair_counts (
        left_id BYTEA,
        right_id BYTEA,
        frequency BIGINT,
        PRIMARY KEY (left_id, right_id)
    ) ON COMMIT DROP;
    
    -- Count pairs across all texts
    FOREACH v_text IN ARRAY p_texts
    LOOP
        INSERT INTO pair_counts (left_id, right_id, frequency)
        SELECT 
            a1.id, a2.id, pc.pair_count
        FROM bpe_count_pairs(v_text) pc
        JOIN atom a1 ON a1.codepoint = pc.left_cp
        JOIN atom a2 ON a2.codepoint = pc.right_cp
        ON CONFLICT (left_id, right_id) 
        DO UPDATE SET frequency = pair_counts.frequency + EXCLUDED.frequency;
    END LOOP;
    
    -- Learn merges in frequency order
    FOR v_left_id, v_right_id IN
        SELECT left_id, right_id 
        FROM pair_counts 
        WHERE frequency >= p_min_frequency
        ORDER BY frequency DESC
        LIMIT p_num_merges
    LOOP
        -- Create the merged composition
        v_merged_id := hypercube_insert_composition(
            ARRAY[v_left_id, v_right_id],
            ARRAY[true, true]  -- Both are atoms initially
        );
        
        -- Record the merge rule
        INSERT INTO bpe_merge (left_id, right_id, merged_id, frequency, priority)
        SELECT v_left_id, v_right_id, v_merged_id, pc.frequency, v_priority
        FROM pair_counts pc
        WHERE pc.left_id = v_left_id AND pc.right_id = v_right_id
        ON CONFLICT (left_id, right_id) DO NOTHING;
        
        v_merge_count := v_merge_count + 1;
        v_priority := v_priority + 1;
    END LOOP;
    
    DROP TABLE IF EXISTS pair_counts;
    
    RETURN v_merge_count;
END;
$$ LANGUAGE plpgsql;

-- Apply BPE tokenization to text
-- Returns array of composition/atom IDs representing the tokenized text
CREATE OR REPLACE FUNCTION bpe_tokenize(p_text TEXT)
RETURNS BYTEA[] AS $$
DECLARE
    v_tokens BYTEA[];
    v_new_tokens BYTEA[];
    v_len INTEGER;
    v_i INTEGER;
    v_left_id BYTEA;
    v_right_id BYTEA;
    v_merged_id BYTEA;
    v_changed BOOLEAN;
    v_max_iterations INTEGER := 1000;
    v_iteration INTEGER := 0;
BEGIN
    -- Start with atom IDs for each character
    SELECT array_agg(a.id ORDER BY i)
    INTO v_tokens
    FROM (
        SELECT i, ascii(substring(p_text, i, 1)) as cp
        FROM generate_series(1, length(p_text)) as i
    ) chars
    JOIN atom a ON a.codepoint = chars.cp;
    
    IF v_tokens IS NULL OR array_length(v_tokens, 1) < 2 THEN
        RETURN v_tokens;
    END IF;
    
    -- Iteratively apply merges in priority order
    LOOP
        v_changed := false;
        v_new_tokens := ARRAY[]::BYTEA[];
        v_i := 1;
        v_len := array_length(v_tokens, 1);
        
        WHILE v_i <= v_len LOOP
            IF v_i < v_len THEN
                -- Check if this pair has a merge rule
                SELECT merged_id INTO v_merged_id
                FROM bpe_merge
                WHERE left_id = v_tokens[v_i] AND right_id = v_tokens[v_i + 1]
                ORDER BY priority
                LIMIT 1;
                
                IF v_merged_id IS NOT NULL THEN
                    -- Apply merge
                    v_new_tokens := array_append(v_new_tokens, v_merged_id);
                    v_i := v_i + 2;
                    v_changed := true;
                    CONTINUE;
                END IF;
            END IF;
            
            -- No merge, keep token
            v_new_tokens := array_append(v_new_tokens, v_tokens[v_i]);
            v_i := v_i + 1;
        END LOOP;
        
        v_tokens := v_new_tokens;
        v_iteration := v_iteration + 1;
        
        EXIT WHEN NOT v_changed OR v_iteration >= v_max_iterations;
    END LOOP;
    
    RETURN v_tokens;
END;
$$ LANGUAGE plpgsql STABLE;

-- Ingest text using BPE tokenization
-- Creates proper hierarchical Merkle DAG: "Hel" = "He" + "l"
CREATE OR REPLACE FUNCTION bpe_ingest_text(p_text TEXT)
RETURNS BYTEA AS $$
DECLARE
    v_tokens BYTEA[];
    v_result BYTEA;
    v_is_atom BOOLEAN[];
    v_i INTEGER;
BEGIN
    -- Tokenize using learned BPE merges
    v_tokens := bpe_tokenize(p_text);
    
    IF v_tokens IS NULL OR array_length(v_tokens, 1) = 0 THEN
        RETURN NULL;
    END IF;
    
    -- If single token, return it directly
    IF array_length(v_tokens, 1) = 1 THEN
        RETURN v_tokens[1];
    END IF;
    
    -- Build is_atom array (check if each token is an atom or composition)
    v_is_atom := ARRAY[]::BOOLEAN[];
    FOR v_i IN 1..array_length(v_tokens, 1) LOOP
        v_is_atom := array_append(v_is_atom, 
            EXISTS(SELECT 1 FROM atom WHERE id = v_tokens[v_i]));
    END LOOP;
    
    -- Create final composition from tokens
    v_result := hypercube_insert_composition(v_tokens, v_is_atom);
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Reconstruct text from a composition ID
-- Traverses the Merkle DAG to get leaf atoms
CREATE OR REPLACE FUNCTION bpe_reconstruct_text(p_id BYTEA)
RETURNS TEXT AS $$
DECLARE
    v_result TEXT := '';
BEGIN
    WITH RECURSIVE dag AS (
        -- Base case: check if it's an atom
        SELECT 
            p_id as id,
            EXISTS(SELECT 1 FROM atom WHERE id = p_id) as is_atom,
            ARRAY[]::INTEGER[] as path,
            0 as depth
        
        UNION ALL
        
        -- Recursive case: traverse children
        SELECT 
            e.child_id,
            e.is_atom,
            d.path || e.ordinal,
            d.depth + 1
        FROM dag d
        JOIN relation_edge e ON e.parent_id = d.id
        WHERE NOT d.is_atom AND d.depth < 100
    )
    SELECT string_agg(chr(a.codepoint), '' ORDER BY d.path)
    INTO v_result
    FROM dag d
    JOIN atom a ON a.id = d.id
    WHERE d.is_atom;
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql STABLE;

-- Get the composition structure (for debugging/visualization)
CREATE OR REPLACE FUNCTION bpe_show_structure(p_id BYTEA)
RETURNS TABLE(
    node_depth INTEGER,
    node_path INTEGER[],
    node_id BYTEA,
    node_is_atom BOOLEAN,
    content TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE dag AS (
        SELECT 
            0 as lvl,
            ARRAY[]::INTEGER[] as pth,
            p_id as nid,
            EXISTS(SELECT 1 FROM atom WHERE atom.id = p_id) as is_atm
        
        UNION ALL
        
        SELECT 
            d.lvl + 1,
            d.pth || e.ordinal,
            e.child_id,
            e.is_atom
        FROM dag d
        JOIN relation_edge e ON e.parent_id = d.nid
        WHERE NOT d.is_atm AND d.lvl < 20
    )
    SELECT 
        d.lvl,
        d.pth,
        d.nid,
        d.is_atm,
        CASE 
            WHEN d.is_atm THEN chr(a.codepoint)
            ELSE '[composition]'
        END as content
    FROM dag d
    LEFT JOIN atom a ON a.id = d.nid AND d.is_atm
    ORDER BY d.pth;
END;
$$ LANGUAGE plpgsql STABLE;

-- Bootstrap with basic character-pair merges for common English patterns
-- This provides a starting point before learning from actual corpus
CREATE OR REPLACE FUNCTION bpe_bootstrap_english()
RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER := 0;
    v_pairs TEXT[] := ARRAY[
        'th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd',
        'ti', 'es', 'or', 'te', 'of', 'ed', 'is', 'it', 'al', 'ar',
        'st', 'to', 'nt', 'ng', 'se', 'ha', 'as', 'ou', 'io', 'le',
        've', 'co', 'me', 'de', 'hi', 'ri', 'ro', 'ic', 'ne', 'ea',
        'ra', 'ce', 'li', 'ch', 'll', 'be', 'ma', 'si', 'om', 'ur',
        'Th', 'He', 'In', 'An', 'Th', 'Th', 'The', 'the', 'and', 'ing'
    ];
    v_pair TEXT;
    v_left_id BYTEA;
    v_right_id BYTEA;
    v_merged_id BYTEA;
    v_priority INTEGER := 1;
BEGIN
    FOREACH v_pair IN ARRAY v_pairs
    LOOP
        -- Get atom IDs for the pair
        SELECT id INTO v_left_id FROM atom WHERE codepoint = ascii(substring(v_pair, 1, 1));
        SELECT id INTO v_right_id FROM atom WHERE codepoint = ascii(substring(v_pair, 2, 1));
        
        IF v_left_id IS NOT NULL AND v_right_id IS NOT NULL THEN
            -- Skip if already exists
            IF NOT EXISTS (SELECT 1 FROM bpe_merge WHERE left_id = v_left_id AND right_id = v_right_id) THEN
                -- Create composition
                v_merged_id := hypercube_insert_composition(
                    ARRAY[v_left_id, v_right_id],
                    ARRAY[true, true]
                );
                
                -- Record merge rule
                INSERT INTO bpe_merge (left_id, right_id, merged_id, frequency, priority)
                VALUES (v_left_id, v_right_id, v_merged_id, 1000000 - v_priority, v_priority);
                
                v_count := v_count + 1;
            END IF;
        END IF;
        
        v_priority := v_priority + 1;
    END LOOP;
    
    RETURN v_count;
END;
$$ LANGUAGE plpgsql;

COMMIT;
