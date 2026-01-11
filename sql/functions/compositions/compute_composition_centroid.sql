-- =============================================================================
-- COMPUTE COMPOSITION CENTROID
-- =============================================================================
-- Computes composition centroid from its atom children
-- =============================================================================

CREATE OR REPLACE FUNCTION compute_composition_centroid(comp_id bytea)
RETURNS geometry AS $$
    SELECT st_centroid_4d(ST_Collect(a.geom))
    FROM composition_child cc
    JOIN atom a ON a.id = cc.child_id
    WHERE cc.composition_id = comp_id;
$$ LANGUAGE sql STABLE PARALLEL SAFE;