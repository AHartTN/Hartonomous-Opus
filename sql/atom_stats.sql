CREATE OR REPLACE FUNCTION atom_stats()
RETURNS TABLE(atoms BIGINT, compositions BIGINT, relations BIGINT) AS $$
    SELECT
        (SELECT COUNT(*) FROM atom) as atoms,
        (SELECT COUNT(*) FROM composition) as compositions,
        (SELECT COUNT(*) FROM relation) as relations;
$$ LANGUAGE SQL STABLE;