-- Legacy alias for composition_content_hash
CREATE OR REPLACE FUNCTION atom_content_hash(p_text TEXT)
RETURNS BYTEA AS $$
    SELECT composition_content_hash(p_text);
$$ LANGUAGE SQL STABLE;