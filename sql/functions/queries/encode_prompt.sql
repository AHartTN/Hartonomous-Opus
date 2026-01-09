-- =============================================================================
-- GENERATIVE QUERY FUNCTIONS
-- =============================================================================
-- Token generation and prompting capabilities for the hypercube
-- =============================================================================

-- Tokenize text into composition IDs
CREATE OR REPLACE FUNCTION encode_prompt(p_text TEXT)
RETURNS TABLE(
    pos INTEGER,
    token TEXT,
    composition_id BYTEA,
    has_centroid BOOLEAN
) AS $$
DECLARE
    v_words TEXT[];
    v_word TEXT;
    v_pos INTEGER := 0;
    v_comp_id BYTEA;
    v_has_centroid BOOLEAN;
BEGIN
    -- Simple word tokenization
    v_words := regexp_split_to_array(lower(p_text), E'[\\s,.!?;:\'"()\\[\\]{}]+');

    FOREACH v_word IN ARRAY v_words LOOP
        IF v_word = '' THEN
            CONTINUE;
        END IF;

        -- Look up in vocabulary
        SELECT c.id, c.centroid IS NOT NULL
        INTO v_comp_id, v_has_centroid
        FROM composition c
        WHERE c.label = v_word
        LIMIT 1;

        pos := v_pos;
        token := v_word;
        composition_id := v_comp_id;
        has_centroid := COALESCE(v_has_centroid, FALSE);
        RETURN NEXT;

        v_pos := v_pos + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql STABLE;