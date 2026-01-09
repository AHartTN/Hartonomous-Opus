CREATE OR REPLACE FUNCTION atom_child_count(p_id BYTEA)
RETURNS INTEGER AS $$
    SELECT COALESCE(
        (SELECT child_count FROM composition WHERE id = p_id),
        0
    );
$$ LANGUAGE SQL STABLE;