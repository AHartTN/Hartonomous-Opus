-- =============================================================================
-- CHILD REFERENCE VALIDATION TRIGGER
-- =============================================================================
-- Validates that child_id in composition_child references an existing record
-- in either atom or composition table based on child_type.
--
-- This enforces referential integrity that can't be done with standard FK
-- constraints due to the polymorphic nature of child_id (can reference atom
-- or composition based on child_type).
--
-- CRITICAL: Without this trigger, children can reference non-existent records,
-- causing silent data corruption and broken graph traversals.
-- =============================================================================

-- Drop existing trigger and function if they exist
DROP TRIGGER IF EXISTS trigger_validate_child_references ON composition_child;
DROP FUNCTION IF EXISTS validate_child_references();

-- Validation function
CREATE OR REPLACE FUNCTION validate_child_references()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate based on child_type
    IF NEW.child_type = 'A' THEN
        -- Child must exist in atom table
        IF NOT EXISTS (SELECT 1 FROM atom WHERE id = NEW.child_id) THEN
            RAISE EXCEPTION 'Invalid atom child_id: % (not found in atom table)', encode(NEW.child_id, 'hex')
                USING HINT = 'Ensure atoms are seeded before inserting compositions';
        END IF;
    ELSIF NEW.child_type = 'C' THEN
        -- Child must exist in composition table
        IF NOT EXISTS (SELECT 1 FROM composition WHERE id = NEW.child_id) THEN
            RAISE EXCEPTION 'Invalid composition child_id: % (not found in composition table)', encode(NEW.child_id, 'hex')
                USING HINT = 'Ensure parent compositions are inserted before children';
        END IF;
    ELSE
        RAISE EXCEPTION 'Invalid child_type: % (must be A or C)', NEW.child_type;
    END IF;

    -- Also validate that composition_id exists
    IF NOT EXISTS (SELECT 1 FROM composition WHERE id = NEW.composition_id) THEN
        RAISE EXCEPTION 'Invalid composition_id: % (not found in composition table)', encode(NEW.composition_id, 'hex')
            USING HINT = 'Ensure composition is inserted before its children';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create the BEFORE INSERT trigger (BEFORE so we can reject invalid data)
CREATE TRIGGER trigger_validate_child_references
    BEFORE INSERT OR UPDATE ON composition_child
    FOR EACH ROW EXECUTE FUNCTION validate_child_references();

-- Add helpful comment
COMMENT ON FUNCTION validate_child_references() IS
    'Validates referential integrity of composition_child.child_id against atom/composition tables based on child_type';

-- =============================================================================
-- REPAIR FUNCTION: Find and report orphaned children
-- =============================================================================
-- Use this to diagnose existing data integrity issues

CREATE OR REPLACE FUNCTION find_orphaned_children()
RETURNS TABLE (
    composition_id_hex TEXT,
    ordinal SMALLINT,
    child_type CHAR(1),
    child_id_hex TEXT,
    issue TEXT
) AS $$
BEGIN
    -- Find atom children referencing non-existent atoms
    RETURN QUERY
    SELECT
        encode(cc.composition_id, 'hex'),
        cc.ordinal,
        cc.child_type,
        encode(cc.child_id, 'hex'),
        'Atom child references non-existent atom'::TEXT
    FROM composition_child cc
    WHERE cc.child_type = 'A'
      AND NOT EXISTS (SELECT 1 FROM atom WHERE id = cc.child_id);

    -- Find composition children referencing non-existent compositions
    RETURN QUERY
    SELECT
        encode(cc.composition_id, 'hex'),
        cc.ordinal,
        cc.child_type,
        encode(cc.child_id, 'hex'),
        'Composition child references non-existent composition'::TEXT
    FROM composition_child cc
    WHERE cc.child_type = 'C'
      AND NOT EXISTS (SELECT 1 FROM composition WHERE id = cc.child_id);

    -- Find children with invalid composition_id
    RETURN QUERY
    SELECT
        encode(cc.composition_id, 'hex'),
        cc.ordinal,
        cc.child_type,
        encode(cc.child_id, 'hex'),
        'Child references non-existent parent composition'::TEXT
    FROM composition_child cc
    WHERE NOT EXISTS (SELECT 1 FROM composition WHERE id = cc.composition_id);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION find_orphaned_children() IS
    'Finds composition_child records that reference non-existent atoms or compositions';

-- =============================================================================
-- REPAIR FUNCTION: Delete orphaned children
-- =============================================================================
-- Use with caution - only after reviewing find_orphaned_children() results

CREATE OR REPLACE FUNCTION delete_orphaned_children()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH deleted AS (
        DELETE FROM composition_child cc
        WHERE
            -- Orphaned atom children
            (cc.child_type = 'A' AND NOT EXISTS (SELECT 1 FROM atom WHERE id = cc.child_id))
            -- Orphaned composition children
            OR (cc.child_type = 'C' AND NOT EXISTS (SELECT 1 FROM composition WHERE id = cc.child_id))
            -- Children with missing parent
            OR NOT EXISTS (SELECT 1 FROM composition WHERE id = cc.composition_id)
        RETURNING 1
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION delete_orphaned_children() IS
    'Deletes orphaned composition_child records. Run find_orphaned_children() first to review.';
