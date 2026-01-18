-- =============================================================================
-- CHILD COUNT INTEGRITY MAINTENANCE
-- =============================================================================
-- Maintains composition.child_count to always match the actual number of
-- children in composition_child.
--
-- DESIGN DECISION:
-- - Bulk insert mode: C++ sets child_count directly, trigger is disabled
-- - Incremental mode: Trigger maintains count on DELETE/UPDATE only
--
-- The INSERT trigger is DISABLED because:
-- 1. Bulk inserts from C++ already set child_count correctly
-- 2. Having the trigger increment on INSERT would double the count
-- 3. Compositions are created with accurate child_count, children added after
--
-- DELETE and UPDATE triggers remain active to handle:
-- - Removing children from existing compositions
-- - Moving children between compositions
-- =============================================================================

-- Drop existing trigger if present (to avoid conflicts)
DROP TRIGGER IF EXISTS trigger_maintain_child_count ON composition_child;

-- Trigger function to maintain child_count (DELETE and UPDATE only)
CREATE OR REPLACE FUNCTION maintain_composition_child_count()
RETURNS TRIGGER AS $$
BEGIN
    -- NOTE: INSERT is NOT handled here - bulk inserts set child_count directly
    -- This avoids double-counting when C++ inserts compositions with pre-computed counts

    IF TG_OP = 'DELETE' THEN
        -- Decrement child_count for the removed composition
        UPDATE composition SET child_count = child_count - 1 WHERE id = OLD.composition_id;
        RETURN OLD;

    ELSIF TG_OP = 'UPDATE' THEN
        -- If composition_id changed, adjust counts accordingly
        IF OLD.composition_id IS DISTINCT FROM NEW.composition_id THEN
            UPDATE composition SET child_count = child_count - 1 WHERE id = OLD.composition_id;
            UPDATE composition SET child_count = child_count + 1 WHERE id = NEW.composition_id;
        END IF;
        RETURN NEW;
    END IF;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger on composition_child table (DELETE and UPDATE only)
-- NOTE: INSERT is intentionally excluded - C++ handles counts during bulk insert
CREATE TRIGGER trigger_maintain_child_count
    AFTER DELETE OR UPDATE OF composition_id ON composition_child
    FOR EACH ROW EXECUTE FUNCTION maintain_composition_child_count();

-- Add check constraint to ensure child_count is never negative (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'check_child_count_non_negative'
        AND conrelid = 'composition'::regclass
    ) THEN
        ALTER TABLE composition ADD CONSTRAINT check_child_count_non_negative CHECK (child_count >= 0);
    END IF;
END $$;

-- =============================================================================
-- REPAIR FUNCTIONS
-- =============================================================================

-- Function to recalculate child_count for all compositions (for data repair)
CREATE OR REPLACE FUNCTION recalculate_all_child_counts()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    WITH updated AS (
        UPDATE composition c
        SET child_count = COALESCE(sub.actual_count, 0)
        FROM (
            SELECT composition_id, COUNT(*) as actual_count
            FROM composition_child
            GROUP BY composition_id
        ) sub
        WHERE c.id = sub.composition_id
          AND c.child_count != sub.actual_count
        RETURNING 1
    )
    SELECT COUNT(*) INTO updated_count FROM updated;

    -- Also handle compositions with no children
    UPDATE composition
    SET child_count = 0
    WHERE child_count != 0
      AND NOT EXISTS (SELECT 1 FROM composition_child WHERE composition_id = composition.id);

    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION recalculate_all_child_counts() IS
    'Repairs child_count for all compositions. Returns number of compositions updated.';

-- Function to find compositions with mismatched child_count
CREATE OR REPLACE FUNCTION find_child_count_mismatches()
RETURNS TABLE (
    composition_id_hex TEXT,
    label TEXT,
    expected_count BIGINT,
    actual_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        encode(c.id, 'hex'),
        c.label,
        COALESCE(cc.cnt, 0),
        c.child_count
    FROM composition c
    LEFT JOIN (
        SELECT composition_id, COUNT(*) as cnt
        FROM composition_child
        GROUP BY composition_id
    ) cc ON c.id = cc.composition_id
    WHERE c.child_count != COALESCE(cc.cnt, 0);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION find_child_count_mismatches() IS
    'Finds compositions where child_count does not match actual children in composition_child';