-- Compute content hash using C extension (fast CPE cascade)
-- This computes the composition hash for a text string
CREATE OR REPLACE FUNCTION composition_content_hash(p_text TEXT)
RETURNS BYTEA AS $$
DECLARE
    v_hashes BYTEA[];
BEGIN
    -- Get atom hashes in order
    SELECT array_agg(a.id ORDER BY ordinality)
    INTO v_hashes
    FROM unnest(string_to_array(p_text, NULL)) WITH ORDINALITY AS chars(chr, ordinality)
    JOIN atom a ON a.codepoint = ascii(chr);

    IF v_hashes IS NULL OR array_length(v_hashes, 1) = 0 THEN
        RETURN NULL;
    END IF;

    IF array_length(v_hashes, 1) = 1 THEN
        RETURN v_hashes[1];
    END IF;

    -- Use C extension for CPE cascade (much faster than plpgsql loops)
    RETURN hypercube_content_hash(v_hashes);
END;
$$ LANGUAGE plpgsql STABLE;