-- =============================================================================
-- QUERY SEARCH FUNCTIONS
-- =============================================================================
-- Text search and Q&A capabilities for the hypercube
-- =============================================================================

-- Search compositions by keyword pattern, returns matches with scores
CREATE OR REPLACE FUNCTION search_text(
    query_text TEXT,
    result_limit INTEGER DEFAULT 50
)
RETURNS TABLE(
    id BYTEA,
    text TEXT,
    depth INTEGER,
    atom_count BIGINT,
    score FLOAT
)
AS $$
DECLARE
    pattern TEXT;
BEGIN
    -- Build regex pattern from keywords (words > 2 chars)
    SELECT array_to_string(array_agg(word), '|') INTO pattern
    FROM (
        SELECT unnest(string_to_array(lower(query_text), ' ')) AS word
    ) w
    WHERE length(word) > 2;

    IF pattern IS NULL OR pattern = '' THEN
        pattern := lower(query_text);
    END IF;

    RETURN QUERY
    SELECT
        c.id,
        atom_reconstruct_text(c.id)::TEXT as text,
        c.depth,
        c.atom_count,
        (c.depth::FLOAT * 0.5 + c.atom_count::FLOAT * 0.1) as score
    FROM composition c
    WHERE atom_reconstruct_text(c.id) ~* pattern
      AND c.depth >= 3
      AND length(atom_reconstruct_text(c.id)) > 3
    ORDER BY c.depth DESC, c.atom_count DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;