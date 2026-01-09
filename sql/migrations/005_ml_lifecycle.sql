-- Migration 005: ML Lifecycle Management
--
-- Complete MLOps infrastructure for experiment tracking,
-- model versioning, and artifact management
--
-- Design:
-- - ml_experiment: Top-level experiment containers
-- - ml_run: Individual training runs within experiments
-- - ml_model_version: Versioned model artifacts with lineage
-- - ml_model_registry: Central registry for deployed models
-- - ml_metric_log: Time-series metrics during training
-- - ml_artifact: Store paths and metadata for model artifacts
-- - ml_checkpoint: Training checkpoints for resumption
--
-- This enables:
-- - Reproducible ML experiments
-- - Model lineage tracking
-- - A/B testing and model comparison
-- - Production deployment tracking
-- - Rollback capabilities

BEGIN;

-- =============================================================================
-- Experiment Management
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml_experiment (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    tags TEXT[] DEFAULT '{}',

    -- Configuration (JSON blob for flexibility)
    model_config JSONB DEFAULT '{}',
    training_config JSONB DEFAULT '{}',
    hyperparameters JSONB DEFAULT '{}',

    -- Target metrics for this experiment
    target_metrics JSONB DEFAULT '{}',  -- e.g., {"accuracy": 0.95, "f1": 0.90}

    -- Ownership and metadata
    created_by TEXT NOT NULL DEFAULT CURRENT_USER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived', 'failed')),

    -- Link to hypercube compositions (optional)
    base_model_composition_id BYTEA REFERENCES composition(id) DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX idx_ml_experiment_status ON ml_experiment(status);
CREATE INDEX idx_ml_experiment_created ON ml_experiment(created_at DESC);
CREATE INDEX idx_ml_experiment_tags ON ml_experiment USING gin(tags);
CREATE INDEX idx_ml_experiment_base_model ON ml_experiment(base_model_composition_id) WHERE base_model_composition_id IS NOT NULL;

-- =============================================================================
-- Training Runs
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml_run (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES ml_experiment(id) ON DELETE CASCADE,
    run_name TEXT,

    -- Run lifecycle
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'stopped', 'crashed')),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration_seconds INT GENERATED ALWAYS AS (EXTRACT(EPOCH FROM (end_time - start_time))) STORED,

    -- Configuration for this specific run
    hyperparameters JSONB DEFAULT '{}',
    environment JSONB DEFAULT '{}',  -- Python version, CUDA version, etc.

    -- Results
    metrics JSONB DEFAULT '{}',  -- Final metrics
    artifacts JSONB DEFAULT '{}',  -- Paths to saved models, plots, etc.

    -- Execution metadata
    logs TEXT,  -- Training logs (or path to log file)
    error_message TEXT,
    stack_trace TEXT,

    -- Resource usage (optional)
    gpu_hours REAL,
    cpu_hours REAL,
    memory_peak_gb REAL,

    -- Reproducibility
    random_seed INT,
    git_commit_hash TEXT,

    -- Ownership
    created_by TEXT NOT NULL DEFAULT CURRENT_USER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Link to resulting model version
    output_model_version_id BIGINT REFERENCES ml_model_version(id) DEFERRABLE INITIALLY DEFERRED,

    CONSTRAINT unique_run_name_per_experiment UNIQUE (experiment_id, run_name)
);

CREATE INDEX idx_ml_run_experiment ON ml_run(experiment_id);
CREATE INDEX idx_ml_run_status ON ml_run(status);
CREATE INDEX idx_ml_run_start_time ON ml_run(start_time DESC);
CREATE INDEX idx_ml_run_output_model ON ml_run(output_model_version_id) WHERE output_model_version_id IS NOT NULL;

-- =============================================================================
-- Model Versioning and Registry
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml_model_version (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,

    -- Lineage
    parent_version_id BIGINT REFERENCES ml_model_version(id) DEFERRABLE INITIALLY DEFERRED,
    source_run_id BIGINT REFERENCES ml_run(id) DEFERRABLE INITIALLY DEFERRED,

    -- Model metadata
    model_type TEXT,  -- e.g., 'transformer', 'cnn', 'moe'
    architecture JSONB DEFAULT '{}',
    hyperparameters JSONB DEFAULT '{}',

    -- Training metadata
    training_data_info JSONB DEFAULT '{}',
    training_duration_seconds INT,

    -- Performance metrics
    performance_metrics JSONB DEFAULT '{}',
    validation_metrics JSONB DEFAULT '{}',
    test_metrics JSONB DEFAULT '{}',

    -- Artifact storage
    model_artifact_path TEXT NOT NULL,  -- Path or URI to model file
    model_size_bytes BIGINT,
    checksum_blake3 BYTEA,  -- BLAKE3 hash for integrity

    -- Status and approval workflow
    validation_status TEXT NOT NULL DEFAULT 'pending' CHECK (validation_status IN ('pending', 'approved', 'rejected', 'deprecated')),
    approval_notes TEXT,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,

    -- Deployment tracking
    is_deployed BOOLEAN DEFAULT FALSE,
    deployment_target TEXT,  -- e.g., 'production', 'staging', 'canary'
    deployed_at TIMESTAMPTZ,

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    description TEXT,
    created_by TEXT NOT NULL DEFAULT CURRENT_USER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Link to hypercube composition (for semantic integration)
    composition_id BYTEA REFERENCES composition(id) DEFERRABLE INITIALLY DEFERRED,

    CONSTRAINT unique_model_version UNIQUE (model_name, version)
);

CREATE INDEX idx_ml_model_version_name ON ml_model_version(model_name);
CREATE INDEX idx_ml_model_version_status ON ml_model_version(validation_status);
CREATE INDEX idx_ml_model_version_deployed ON ml_model_version(is_deployed) WHERE is_deployed = TRUE;
CREATE INDEX idx_ml_model_version_created ON ml_model_version(created_at DESC);
CREATE INDEX idx_ml_model_version_parent ON ml_model_version(parent_version_id) WHERE parent_version_id IS NOT NULL;
CREATE INDEX idx_ml_model_version_source_run ON ml_model_version(source_run_id) WHERE source_run_id IS NOT NULL;
CREATE INDEX idx_ml_model_version_composition ON ml_model_version(composition_id) WHERE composition_id IS NOT NULL;
CREATE INDEX idx_ml_model_version_tags ON ml_model_version USING gin(tags);

-- =============================================================================
-- Model Registry (Production Deployment Tracking)
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml_model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,

    -- Current production version
    production_version_id BIGINT NOT NULL REFERENCES ml_model_version(id),
    previous_version_id BIGINT REFERENCES ml_model_version(id) DEFERRABLE INITIALLY DEFERRED,

    -- Deployment metadata
    deployment_strategy TEXT DEFAULT 'blue-green' CHECK (deployment_strategy IN ('blue-green', 'canary', 'rolling', 'shadow')),
    traffic_percentage INT DEFAULT 100 CHECK (traffic_percentage >= 0 AND traffic_percentage <= 100),

    -- SLA and monitoring
    target_latency_ms INT,
    target_throughput_rps INT,
    target_availability_percent REAL,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    health_status TEXT DEFAULT 'healthy' CHECK (health_status IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
    last_health_check TIMESTAMPTZ,

    -- Ownership
    owner_team TEXT,
    oncall_contact TEXT,

    -- Audit trail
    deployed_by TEXT NOT NULL,
    deployed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ml_model_registry_active ON ml_model_registry(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_ml_model_registry_health ON ml_model_registry(health_status);
CREATE INDEX idx_ml_model_registry_production_version ON ml_model_registry(production_version_id);

-- =============================================================================
-- Metric Logging (Time-Series Training Metrics)
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml_metric_log (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES ml_run(id) ON DELETE CASCADE,

    -- Metric identification
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,

    -- Time-series tracking
    step INT NOT NULL,  -- Training step/iteration
    epoch INT,  -- Training epoch (if applicable)
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Metric metadata
    metric_type TEXT CHECK (metric_type IN ('train', 'validation', 'test', 'system')),
    tags JSONB DEFAULT '{}',

    CONSTRAINT unique_metric_per_step UNIQUE (run_id, metric_name, step)
);

CREATE INDEX idx_ml_metric_log_run ON ml_metric_log(run_id, metric_name, step);
CREATE INDEX idx_ml_metric_log_timestamp ON ml_metric_log(timestamp DESC);

-- =============================================================================
-- Artifact Storage
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml_artifact (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES ml_run(id) ON DELETE CASCADE,

    -- Artifact identification
    artifact_name TEXT NOT NULL,
    artifact_type TEXT NOT NULL CHECK (artifact_type IN ('model', 'checkpoint', 'plot', 'dataset', 'config', 'log', 'other')),

    -- Storage
    artifact_path TEXT NOT NULL,  -- Local path or cloud URI (s3://, gs://, etc.)
    size_bytes BIGINT,
    checksum_blake3 BYTEA,

    -- Metadata
    description TEXT,
    mime_type TEXT,
    tags JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_artifact_per_run UNIQUE (run_id, artifact_name)
);

CREATE INDEX idx_ml_artifact_run ON ml_artifact(run_id);
CREATE INDEX idx_ml_artifact_type ON ml_artifact(artifact_type);
CREATE INDEX idx_ml_artifact_created ON ml_artifact(created_at DESC);

-- =============================================================================
-- Training Checkpoints
-- =============================================================================

CREATE TABLE IF NOT EXISTS ml_checkpoint (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES ml_run(id) ON DELETE CASCADE,

    -- Checkpoint identification
    checkpoint_name TEXT,
    step INT NOT NULL,
    epoch INT,

    -- Storage
    checkpoint_path TEXT NOT NULL,
    size_bytes BIGINT,
    checksum_blake3 BYTEA,

    -- Checkpoint metadata
    metrics JSONB DEFAULT '{}',  -- Metrics at this checkpoint
    is_best BOOLEAN DEFAULT FALSE,  -- Best checkpoint so far

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_checkpoint_per_run UNIQUE (run_id, step)
);

CREATE INDEX idx_ml_checkpoint_run ON ml_checkpoint(run_id);
CREATE INDEX idx_ml_checkpoint_step ON ml_checkpoint(run_id, step DESC);
CREATE INDEX idx_ml_checkpoint_best ON ml_checkpoint(run_id, is_best) WHERE is_best = TRUE;

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to create a new experiment
CREATE OR REPLACE FUNCTION create_ml_experiment(
    p_name TEXT,
    p_description TEXT DEFAULT NULL,
    p_model_config JSONB DEFAULT '{}'::JSONB,
    p_training_config JSONB DEFAULT '{}'::JSONB
)
RETURNS BIGINT AS $$
DECLARE
    v_experiment_id BIGINT;
BEGIN
    INSERT INTO ml_experiment (name, description, model_config, training_config)
    VALUES (p_name, p_description, p_model_config, p_training_config)
    RETURNING id INTO v_experiment_id;

    RETURN v_experiment_id;
END;
$$ LANGUAGE plpgsql;

-- Function to start a training run
CREATE OR REPLACE FUNCTION start_ml_run(
    p_experiment_id BIGINT,
    p_run_name TEXT DEFAULT NULL,
    p_hyperparameters JSONB DEFAULT '{}'::JSONB
)
RETURNS BIGINT AS $$
DECLARE
    v_run_id BIGINT;
BEGIN
    INSERT INTO ml_run (experiment_id, run_name, hyperparameters, status, start_time)
    VALUES (p_experiment_id, p_run_name, p_hyperparameters, 'running', NOW())
    RETURNING id INTO v_run_id;

    RETURN v_run_id;
END;
$$ LANGUAGE plpgsql;

-- Function to complete a training run
CREATE OR REPLACE FUNCTION complete_ml_run(
    p_run_id BIGINT,
    p_metrics JSONB DEFAULT '{}'::JSONB,
    p_status TEXT DEFAULT 'completed'
)
RETURNS VOID AS $$
BEGIN
    UPDATE ml_run
    SET status = p_status,
        end_time = NOW(),
        metrics = p_metrics
    WHERE id = p_run_id;
END;
$$ LANGUAGE plpgsql;

-- Function to log metrics during training
CREATE OR REPLACE FUNCTION log_ml_metric(
    p_run_id BIGINT,
    p_metric_name TEXT,
    p_metric_value REAL,
    p_step INT,
    p_epoch INT DEFAULT NULL,
    p_metric_type TEXT DEFAULT 'train'
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO ml_metric_log (run_id, metric_name, metric_value, step, epoch, metric_type)
    VALUES (p_run_id, p_metric_name, p_metric_value, p_step, p_epoch, p_metric_type)
    ON CONFLICT (run_id, metric_name, step) DO UPDATE
    SET metric_value = EXCLUDED.metric_value,
        timestamp = NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to register a model version
CREATE OR REPLACE FUNCTION register_ml_model_version(
    p_model_name TEXT,
    p_version TEXT,
    p_artifact_path TEXT,
    p_source_run_id BIGINT DEFAULT NULL,
    p_performance_metrics JSONB DEFAULT '{}'::JSONB
)
RETURNS BIGINT AS $$
DECLARE
    v_version_id BIGINT;
BEGIN
    INSERT INTO ml_model_version (
        model_name, version, model_artifact_path, source_run_id, performance_metrics
    )
    VALUES (
        p_model_name, p_version, p_artifact_path, p_source_run_id, p_performance_metrics
    )
    RETURNING id INTO v_version_id;

    -- Update the source run if provided
    IF p_source_run_id IS NOT NULL THEN
        UPDATE ml_run
        SET output_model_version_id = v_version_id
        WHERE id = p_source_run_id;
    END IF;

    RETURN v_version_id;
END;
$$ LANGUAGE plpgsql;

-- Function to deploy a model to registry
CREATE OR REPLACE FUNCTION deploy_ml_model(
    p_model_name TEXT,
    p_version_id BIGINT,
    p_deployment_strategy TEXT DEFAULT 'blue-green'
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO ml_model_registry (
        model_name, production_version_id, deployment_strategy, deployed_by
    )
    VALUES (
        p_model_name, p_version_id, p_deployment_strategy, CURRENT_USER
    )
    ON CONFLICT (model_name) DO UPDATE
    SET previous_version_id = ml_model_registry.production_version_id,
        production_version_id = p_version_id,
        deployed_by = CURRENT_USER,
        deployed_at = NOW(),
        updated_at = NOW();

    -- Mark the version as deployed
    UPDATE ml_model_version
    SET is_deployed = TRUE,
        deployed_at = NOW()
    WHERE id = p_version_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get experiment leaderboard
CREATE OR REPLACE FUNCTION get_experiment_leaderboard(
    p_experiment_id BIGINT,
    p_metric_name TEXT,
    p_limit INT DEFAULT 10
)
RETURNS TABLE (
    run_id BIGINT,
    run_name TEXT,
    metric_value REAL,
    hyperparameters JSONB,
    duration_seconds INT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.run_name,
        (r.metrics->>p_metric_name)::REAL AS metric_value,
        r.hyperparameters,
        r.duration_seconds,
        r.created_at
    FROM ml_run r
    WHERE r.experiment_id = p_experiment_id
      AND r.status = 'completed'
      AND r.metrics ? p_metric_name
    ORDER BY (r.metrics->>p_metric_name)::REAL DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

COMMIT;

-- Post-migration: Create default experiment for testing
INSERT INTO ml_experiment (name, description, status)
VALUES ('default', 'Default experiment for ad-hoc runs', 'active')
ON CONFLICT (name) DO NOTHING;
