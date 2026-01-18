-- Function and trigger to maintain composition child_count integrity automatically
--
-- This file contains database constraints and triggers to ensure that the child_count
-- column in the composition table always matches the actual number of child records
-- in the composition_child table.
--
-- Requirements:
-- 1. Trigger function that updates child_count on INSERT/DELETE/UPDATE of composition_child
-- 2. Triggers on composition_child table for automatic maintenance
-- 3. Constraint to ensure child_count >= 0
-- 4. Function to recalculate child_count for data repair

-- Trigger function to maintain child_count
CREATE OR REPLACE FUNCTION maintain_composition_child_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Increment child_count for the new composition
        UPDATE composition SET child_count = child_count + 1 WHERE id = NEW.composition_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
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

-- Create the trigger on composition_child table
CREATE TRIGGER trigger_maintain_child_count
    AFTER INSERT OR DELETE OR UPDATE OF composition_id ON composition_child
    FOR EACH ROW EXECUTE FUNCTION maintain_composition_child_count();

-- Add check constraint to ensure child_count is never negative
ALTER TABLE composition ADD CONSTRAINT check_child_count_non_negative CHECK (child_count >= 0);

-- Function to recalculate child_count for all compositions (for data repair)
CREATE OR REPLACE FUNCTION recalculate_all_child_counts()
RETURNS VOID AS $$
BEGIN
    UPDATE composition SET child_count = COALESCE((
        SELECT COUNT(*) FROM composition_child WHERE composition_id = composition.id
    ), 0);
END;
$$ LANGUAGE plpgsql;