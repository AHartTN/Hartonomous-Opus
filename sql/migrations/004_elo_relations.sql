-- Migration 004: ELO-style relation evidence tracking
--
-- Enables multi-model consensus with per-model ELO ratings
--
-- Design:
-- - relation_evidence: Per-model observations with ELO ratings
-- - relation_consensus: Materialized view of consensus across models
-- - relation table becomes a view over consensus (backward compatible)

BEGIN;

-- =============================================================================
-- Per-Model Relation Evidence Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS relation_evidence (
    source_id BYTEA NOT NULL,
    target_id BYTEA NOT NULL,
    relation_type CHAR(1) NOT NULL CHECK (relation_type IN ('S', 'H', 'V', 'M', 'A', 'T')),
    source_model TEXT NOT NULL,
    layer INT NOT NULL,
    component TEXT NOT NULL,

    -- ELO-style rating system
    rating REAL NOT NULL DEFAULT 1500.0 CHECK (rating >= 0 AND rating <= 3000),
    k_factor REAL NOT NULL DEFAULT 32.0,
    observation_count INT NOT NULL DEFAULT 1 CHECK (observation_count >= 1),

    -- Raw evidence
    raw_weight REAL NOT NULL CHECK (raw_weight >= -1.0 AND raw_weight <= 1.0),
    normalized_weight REAL NOT NULL CHECK (normalized_weight >= -1.0 AND normalized_weight <= 1.0),

    -- Metadata
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (source_id, target_id, relation_type, source_model, layer, component),

    -- FK constraints (optional, can be deferred for bulk insert)
    FOREIGN KEY (source_id) REFERENCES composition(id) DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (target_id) REFERENCES composition(id) DEFERRABLE INITIALLY DEFERRED
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_rel_evidence_source ON relation_evidence(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_evidence_target ON relation_evidence(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_evidence_model ON relation_evidence(source_model);
CREATE INDEX IF NOT EXISTS idx_rel_evidence_rating ON relation_evidence(rating DESC);
CREATE INDEX IF NOT EXISTS idx_rel_evidence_updated ON relation_evidence(last_updated DESC);

-- =============================================================================
-- Consensus Materialized View
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS relation_consensus AS
SELECT
    source_id,
    target_id,
    relation_type,

    -- ELO-based consensus metrics
    AVG(rating)::REAL as avg_rating,
    STDDEV(rating)::REAL as rating_stddev,
    MIN(rating)::REAL as min_rating,
    MAX(rating)::REAL as max_rating,

    -- Confidence: inverse of disagreement
    -- High stddev → low confidence, low stddev → high confidence
    (1.0 / (1.0 + COALESCE(STDDEV(rating), 0.0) / 400.0))::REAL as confidence,

    -- Consensus weight: average rating mapped to [-1, 1] via tanh
    TANH((AVG(rating) - 1500.0) / 400.0)::REAL as consensus_weight,

    -- Evidence statistics
    COUNT(*)::INT as num_models,
    SUM(observation_count)::INT as total_observations,

    -- Temporal metadata
    MIN(first_seen) as first_seen,
    MAX(last_updated) as last_updated

FROM relation_evidence
GROUP BY source_id, target_id, relation_type;

-- Indexes on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_consensus_pk
    ON relation_consensus(source_id, target_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_rel_consensus_source ON relation_consensus(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_consensus_target ON relation_consensus(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_consensus_weight ON relation_consensus(consensus_weight DESC);
CREATE INDEX IF NOT EXISTS idx_rel_consensus_confidence ON relation_consensus(confidence DESC);

-- =============================================================================
-- Backward-Compatible Relation View
-- =============================================================================

-- Drop old relation table if it exists, recreate as view
-- WARNING: This will lose existing relation data!
-- Run migration script to convert existing relations to relation_evidence first

DO $$
BEGIN
    -- Check if relation is a table (not a view)
    IF EXISTS (
        SELECT 1 FROM pg_tables
        WHERE schemaname = 'public' AND tablename = 'relation'
    ) THEN
        -- Migrate existing data to relation_evidence
        INSERT INTO relation_evidence
            (source_id, target_id, relation_type, source_model, layer, component,
             rating, observation_count, raw_weight, normalized_weight, first_seen)
        SELECT
            source_id,
            target_id,
            relation_type,
            COALESCE(source_model, 'unknown'),
            COALESCE(layer, -1),
            COALESCE(component, 'legacy'),
            -- Infer initial rating from weight
            1500.0 + 400.0 * ATANH(LEAST(0.999, GREATEST(-0.999, weight))),
            COALESCE(source_count, 1),
            weight,
            weight,
            NOW()
        FROM relation
        ON CONFLICT DO NOTHING;

        -- Drop old table
        DROP TABLE relation CASCADE;

        RAISE NOTICE 'Migrated existing relations to relation_evidence';
    END IF;
END $$;

-- Create relation as view over consensus (backward compatible)
CREATE OR REPLACE VIEW relation AS
SELECT
    'C' as source_type,  -- Legacy: assume composition
    source_id,
    'C' as target_type,  -- Legacy: assume composition
    target_id,
    relation_type,
    consensus_weight as weight,
    'consensus' as source_model,
    total_observations as source_count,
    -1 as layer,  -- Legacy: no layer info in consensus
    'multi-model' as component
FROM relation_consensus;

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to refresh consensus view
CREATE OR REPLACE FUNCTION refresh_relation_consensus()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY relation_consensus;
END;
$$ LANGUAGE plpgsql;

-- Function to get relation confidence
CREATE OR REPLACE FUNCTION get_relation_confidence(
    p_source_id BYTEA,
    p_target_id BYTEA,
    p_relation_type CHAR(1)
)
RETURNS REAL AS $$
    SELECT confidence
    FROM relation_consensus
    WHERE source_id = p_source_id
      AND target_id = p_target_id
      AND relation_type = p_relation_type;
$$ LANGUAGE sql STABLE;

-- Function to get per-model ratings for a relation
CREATE OR REPLACE FUNCTION get_model_ratings(
    p_source_id BYTEA,
    p_target_id BYTEA,
    p_relation_type CHAR(1)
)
RETURNS TABLE(
    model TEXT,
    rating REAL,
    observations INT,
    last_updated TIMESTAMPTZ
) AS $$
    SELECT
        source_model,
        rating,
        observation_count,
        last_updated
    FROM relation_evidence
    WHERE source_id = p_source_id
      AND target_id = p_target_id
      AND relation_type = p_relation_type
    ORDER BY rating DESC;
$$ LANGUAGE sql STABLE;

COMMIT;

-- Post-migration: Refresh consensus view
SELECT refresh_relation_consensus();
