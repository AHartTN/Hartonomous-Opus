-- Validate atom integrity
CREATE OR REPLACE FUNCTION validate_atoms()
RETURNS TABLE (
    check_name TEXT,
    passed BOOLEAN,
    details TEXT
) AS $$
BEGIN
    -- Check 1: All leaves have codepoints
    check_name := 'leaves_have_codepoints';
    SELECT COUNT(*) = 0 INTO passed
    FROM atom WHERE depth = 0 AND codepoint IS NULL;
    details := CASE WHEN passed THEN 'OK' ELSE 'Found leaves without codepoints' END;
    RETURN NEXT;

    -- Check 2: All compositions have relations
    check_name := 'compositions_have_relations';
    SELECT COUNT(*) = 0 INTO passed
    FROM atom a
    WHERE a.depth > 0
      AND NOT EXISTS (SELECT 1 FROM relation r WHERE r.parent_id = a.id AND r.relation_type = 'C');
    details := CASE WHEN passed THEN 'OK' ELSE 'Found compositions without relations' END;
    RETURN NEXT;

    -- Check 3: All have valid geometry
    check_name := 'valid_geometry';
    SELECT COUNT(*) = 0 INTO passed
    FROM atom WHERE geom IS NULL;
    details := CASE WHEN passed THEN 'OK' ELSE 'Found atoms without geometry' END;
    RETURN NEXT;

    -- Check 4: Hilbert indices are set
    check_name := 'hilbert_indices_set';
    SELECT COUNT(*) = 0 INTO passed
    FROM atom WHERE hilbert_lo IS NULL OR hilbert_hi IS NULL;
    details := CASE WHEN passed THEN 'OK' ELSE 'Found atoms without Hilbert indices' END;
    RETURN NEXT;

    RETURN;
END;
$$ LANGUAGE plpgsql STABLE;