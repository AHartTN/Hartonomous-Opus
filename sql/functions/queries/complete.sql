-- =============================================================================
-- GENERATIVE QUERY FUNCTIONS
-- =============================================================================
-- Token generation and prompting capabilities for the hypercube
-- =============================================================================

-- Generate token completion as a single string
CREATE OR REPLACE FUNCTION complete(
    p_prompt TEXT,
    p_max_tokens INTEGER DEFAULT 20
)
RETURNS TEXT AS $$
    SELECT string_agg(token, ' ' ORDER BY pos)
    FROM generate_tokens(p_prompt, p_max_tokens, 0.7, 40);
$$ LANGUAGE SQL STABLE;