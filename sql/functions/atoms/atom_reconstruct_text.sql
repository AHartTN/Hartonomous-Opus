-- =============================================================================
-- ATOM TEXT RECONSTRUCTION FUNCTIONS
-- =============================================================================
-- Functions for reconstructing readable text from atom/composition hierarchies
-- =============================================================================

-- Reconstruct text from a composition by walking its children (recursive)
CREATE OR REPLACE FUNCTION atom_reconstruct_text(p_id BYTEA)
RETURNS TEXT AS $$
DECLARE
    v_result TEXT := '';
    v_is_atom BOOLEAN;
    v_child RECORD;
BEGIN
    -- Check if it's a leaf atom
    SELECT TRUE INTO v_is_atom FROM atom WHERE id = p_id;

    IF v_is_atom THEN
        -- Direct atom: return character
        SELECT chr(codepoint) INTO v_result FROM atom WHERE id = p_id;
        RETURN v_result;
    END IF;

    -- It's a composition: recurse through children
    FOR v_child IN
        SELECT cc.child_id, cc.child_type
        FROM composition_child cc
        WHERE cc.composition_id = p_id
        ORDER BY cc.ordinal
    LOOP
        IF v_child.child_type = 'A' THEN
            -- Child is an atom: append character
            v_result := v_result || COALESCE((SELECT chr(codepoint) FROM atom WHERE id = v_child.child_id), '');
        ELSE
            -- Child is a composition: recurse
            v_result := v_result || COALESCE(atom_reconstruct_text(v_child.child_id), '');
        END IF;
    END LOOP;

    -- Try composition label as fallback
    IF v_result = '' THEN
        SELECT label INTO v_result FROM composition WHERE id = p_id;
    END IF;

    RETURN v_result;
END;
$$ LANGUAGE plpgsql STABLE;