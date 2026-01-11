-- Function to validate composition child references
CREATE OR REPLACE FUNCTION validate_composition_child()
RETURNS TRIGGER AS $$
BEGIN
    -- Ensure the referenced child exists
    IF NEW.child_type = 'A' THEN
        IF NOT EXISTS (SELECT 1 FROM atom WHERE id = NEW.child_id) THEN
            RAISE EXCEPTION 'Atom child % does not exist', NEW.child_id;
        END IF;
    ELSIF NEW.child_type = 'C' THEN
        IF NOT EXISTS (SELECT 1 FROM composition WHERE id = NEW.child_id) THEN
            RAISE EXCEPTION 'Composition child % does not exist', NEW.child_id;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to refresh relation consensus materialized view
CREATE OR REPLACE FUNCTION refresh_relation_consensus()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW relation_consensus;
END;
$$ LANGUAGE plpgsql;