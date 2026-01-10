-- =============================================================================
-- PROJECTION QUALITY FUNCTIONS
-- =============================================================================
-- Functions for calculating and tracking 4D projection quality metrics.
-- Enables selective geometry writing based on manifold quality.
-- =============================================================================

-- =============================================================================
-- Quality Score Calculation Function
-- =============================================================================

CREATE OR REPLACE FUNCTION calculate_projection_quality(
    p_role TEXT,
    p_dtype TEXT,
    p_dim INTEGER,
    p_variance_explained REAL
)
RETURNS REAL AS $$
DECLARE
    score REAL := 0.0;
    dtype_bonus REAL := 0.0;
    role_bonus REAL := 0.0;
    spectrum_bonus REAL := 0.0;
BEGIN
    -- Dtype quality: higher precision = higher score
    CASE p_dtype
        WHEN 'FP32' THEN dtype_bonus := 2.0;
        WHEN 'BF16' THEN dtype_bonus := 1.5;
        WHEN 'FP16' THEN dtype_bonus := 1.0;
        WHEN 'FP8' THEN dtype_bonus := 0.5;
        ELSE dtype_bonus := 0.0;
    END CASE;

    -- Role importance: embeddings > attention > ffn
    CASE p_role
        WHEN 'embeddings' THEN role_bonus := 2.0;
        WHEN 'attention' THEN role_bonus := 1.5;
        WHEN 'ffn' THEN role_bonus := 1.0;
        ELSE role_bonus := 0.5;
    END CASE;

    -- Spectral quality: higher variance explained = higher score
    IF p_variance_explained IS NOT NULL THEN
        spectrum_bonus := p_variance_explained * 2.0;
    END IF;

    -- Dimensionality factor: log(dim) rewards high-dim embeddings
    score := LOG(p_dim) + dtype_bonus + role_bonus + spectrum_bonus;

    RETURN score;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- Trigger to Auto-Calculate Quality Score
-- =============================================================================

CREATE OR REPLACE FUNCTION update_projection_quality()
RETURNS TRIGGER AS $$
BEGIN
    NEW.quality_score := calculate_projection_quality(
        NEW.role,
        NEW.dtype,
        NEW.dim,
        NEW.variance_explained
    );
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_projection_quality
    BEFORE INSERT OR UPDATE ON projection_metadata
    FOR EACH ROW EXECUTE FUNCTION update_projection_quality();

-- =============================================================================
-- Champion Selection Function
-- =============================================================================

CREATE OR REPLACE FUNCTION get_champion_models(p_role TEXT)
RETURNS TABLE (
    model_id BIGINT,
    model_name TEXT,
    quality_score REAL
) AS $$
    SELECT
        pm.model_id,
        m.name,
        MAX(pm.quality_score) as quality_score
    FROM projection_metadata pm
    JOIN model m ON m.id = pm.model_id
    WHERE pm.role = p_role
      AND pm.converged = TRUE
      AND pm.variance_explained > 0.3  -- Threshold for good manifolds
    GROUP BY pm.model_id, m.name
    ORDER BY MAX(pm.quality_score) DESC
    LIMIT 3;  -- Top 3 champions per role
$$ LANGUAGE sql STABLE;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON FUNCTION calculate_projection_quality IS 'Computes quality score for 4D projection based on role, dtype, dimensionality, and spectral quality';
COMMENT ON FUNCTION update_projection_quality IS 'Trigger function to auto-calculate quality score on insert/update';
COMMENT ON FUNCTION get_champion_models IS 'Returns top 3 models for a given role based on projection quality';
