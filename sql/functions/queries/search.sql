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

-- Q&A function: takes a natural language question, returns answer with evidence
CREATE OR REPLACE FUNCTION ask(question TEXT)
RETURNS TEXT
AS $$
DECLARE
    answer TEXT := '';
    evidence TEXT[];
    top_results RECORD;
    pattern TEXT;
BEGIN
    -- Extract keywords from question
    SELECT array_to_string(array_agg(word), '|') INTO pattern
    FROM (
        SELECT unnest(string_to_array(lower(question), ' ')) AS word
    ) w
    WHERE length(word) > 2
      AND word NOT IN ('what', 'who', 'where', 'when', 'why', 'how',
                        'the', 'is', 'are', 'was', 'were', 'did', 'does',
                        'can', 'could', 'would', 'should', 'will');

    IF pattern IS NULL OR pattern = '' THEN
        RETURN 'Please provide a more specific question.';
    END IF;

    -- Gather evidence from compositions
    SELECT array_agg(text ORDER BY score DESC)
    INTO evidence
    FROM (
        SELECT DISTINCT ON (text) text, score
        FROM search_text(pattern, 100)
        WHERE length(text) > 5
        ORDER BY text, score DESC
        LIMIT 20
    ) sub;

    IF evidence IS NULL OR array_length(evidence, 1) = 0 THEN
        RETURN 'No information found for: ' || question;
    END IF;

    -- Build answer from evidence
    answer := 'Based on the text, I found:' || E'\n\n';

    FOR i IN 1..LEAST(10, array_length(evidence, 1)) LOOP
        answer := answer || '  - ' || trim(evidence[i]) || E'\n';
    END LOOP;

    RETURN answer;
END;
$$ LANGUAGE plpgsql;

-- Search for exact phrase matches
CREATE OR REPLACE FUNCTION ask_exact(phrase TEXT, result_limit INTEGER DEFAULT 20)
RETURNS TABLE(
    text TEXT,
    depth INTEGER
)
AS $$
BEGIN
    RETURN QUERY
    SELECT
        atom_reconstruct_text(c.id)::TEXT as text,
        c.depth
    FROM composition c
    WHERE atom_reconstruct_text(c.id) ILIKE '%' || phrase || '%'
    ORDER BY c.depth DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;