/**
 * ML Operations API
 * ==================
 *
 * High-level API for machine learning operations in Hartonomous-Opus
 * Integrates with the 4D hypercube substrate for geometric ML operations
 *
 * Key capabilities:
 * - Experiment tracking and lifecycle management
 * - Model training with metric logging
 * - Model versioning and deployment
 * - Fine-tuning and transfer learning
 * - Real-time inference with caching
 * - Distributed training coordination
 */

#pragma once

#include "hypercube/types.hpp"
#include "hypercube/coordinates.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <chrono>

// Forward declarations
struct PGconn;

namespace hypercube {
namespace ml {

// ============================================================================
// Core Types
// ============================================================================

struct Hyperparameters {
    double learning_rate = 0.001;
    int batch_size = 32;
    int num_epochs = 10;
    double weight_decay = 0.0;
    double dropout_rate = 0.1;
    std::string optimizer = "adam";
    std::unordered_map<std::string, std::string> custom;
};

struct TrainingMetrics {
    double loss = 0.0;
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;
    std::unordered_map<std::string, double> custom_metrics;
};

struct ModelInfo {
    int64_t model_version_id;
    std::string model_name;
    std::string version;
    std::string model_type;
    std::string artifact_path;
    TrainingMetrics performance_metrics;
    bool is_deployed;
    std::chrono::system_clock::time_point created_at;
};

struct ExperimentInfo {
    int64_t experiment_id;
    std::string name;
    std::string description;
    std::string status;
    int total_runs;
    std::chrono::system_clock::time_point created_at;
};

struct TrainingRunInfo {
    int64_t run_id;
    int64_t experiment_id;
    std::string run_name;
    std::string status;
    Hyperparameters hyperparameters;
    TrainingMetrics final_metrics;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    int duration_seconds;
};

// ============================================================================
// ML Operations Interface
// ============================================================================

class MLOperations {
public:
    explicit MLOperations(PGconn* conn);
    ~MLOperations();

    // Experiment Management
    // =====================

    /**
     * Create a new ML experiment
     * @param name Unique experiment name
     * @param description Experiment description
     * @param model_config Model configuration (JSON)
     * @param training_config Training configuration (JSON)
     * @return Experiment ID
     */
    int64_t create_experiment(
        const std::string& name,
        const std::string& description = "",
        const std::string& model_config = "{}",
        const std::string& training_config = "{}"
    );

    /**
     * Get experiment information
     * @param experiment_id Experiment ID
     * @return Experiment information
     */
    ExperimentInfo get_experiment(int64_t experiment_id);

    /**
     * List all experiments
     * @param active_only Only list active experiments
     * @return Vector of experiments
     */
    std::vector<ExperimentInfo> list_experiments(bool active_only = true);

    // Training Run Management
    // =======================

    /**
     * Start a new training run
     * @param experiment_id Parent experiment ID
     * @param run_name Optional run name
     * @param hyperparameters Training hyperparameters
     * @return Run ID
     */
    int64_t start_run(
        int64_t experiment_id,
        const std::string& run_name = "",
        const Hyperparameters& hyperparameters = {}
    );

    /**
     * Log metrics during training
     * @param run_id Training run ID
     * @param metric_name Metric name
     * @param metric_value Metric value
     * @param step Training step/iteration
     * @param epoch Training epoch (optional)
     * @param metric_type Type: train, validation, test
     */
    void log_metric(
        int64_t run_id,
        const std::string& metric_name,
        double metric_value,
        int step,
        int epoch = -1,
        const std::string& metric_type = "train"
    );

    /**
     * Log multiple metrics at once
     * @param run_id Training run ID
     * @param metrics Map of metric name -> value
     * @param step Training step
     * @param epoch Training epoch (optional)
     */
    void log_metrics(
        int64_t run_id,
        const std::unordered_map<std::string, double>& metrics,
        int step,
        int epoch = -1
    );

    /**
     * Complete a training run
     * @param run_id Training run ID
     * @param final_metrics Final metrics from training
     * @param status Run status (completed, failed, etc.)
     */
    void complete_run(
        int64_t run_id,
        const TrainingMetrics& final_metrics = {},
        const std::string& status = "completed"
    );

    /**
     * Save a training checkpoint
     * @param run_id Training run ID
     * @param checkpoint_path Path to checkpoint file
     * @param step Training step
     * @param epoch Training epoch
     * @param metrics Metrics at checkpoint
     * @param is_best Whether this is the best checkpoint so far
     * @return Checkpoint ID
     */
    int64_t save_checkpoint(
        int64_t run_id,
        const std::string& checkpoint_path,
        int step,
        int epoch,
        const TrainingMetrics& metrics,
        bool is_best = false
    );

    /**
     * Get run information
     * @param run_id Run ID
     * @return Run information
     */
    TrainingRunInfo get_run(int64_t run_id);

    /**
     * Get experiment leaderboard
     * @param experiment_id Experiment ID
     * @param metric_name Metric to rank by
     * @param limit Number of top runs to return
     * @return Ranked list of runs
     */
    std::vector<TrainingRunInfo> get_leaderboard(
        int64_t experiment_id,
        const std::string& metric_name = "accuracy",
        int limit = 10
    );

    // Model Versioning
    // ================

    /**
     * Register a new model version
     * @param model_name Model name
     * @param version Version string (e.g., "v1.0.0")
     * @param artifact_path Path to model artifact
     * @param source_run_id Source training run (optional)
     * @param performance_metrics Model performance metrics
     * @return Model version ID
     */
    int64_t register_model_version(
        const std::string& model_name,
        const std::string& version,
        const std::string& artifact_path,
        int64_t source_run_id = -1,
        const TrainingMetrics& performance_metrics = {}
    );

    /**
     * Get model version information
     * @param version_id Model version ID
     * @return Model information
     */
    ModelInfo get_model_version(int64_t version_id);

    /**
     * List all versions of a model
     * @param model_name Model name
     * @return Vector of model versions
     */
    std::vector<ModelInfo> list_model_versions(const std::string& model_name);

    /**
     * Approve a model version for deployment
     * @param version_id Model version ID
     * @param approval_notes Optional approval notes
     */
    void approve_model_version(
        int64_t version_id,
        const std::string& approval_notes = ""
    );

    // Model Deployment
    // ================

    /**
     * Deploy a model to production
     * @param model_name Model name
     * @param version_id Model version ID
     * @param deployment_strategy Deployment strategy (blue-green, canary, etc.)
     */
    void deploy_model(
        const std::string& model_name,
        int64_t version_id,
        const std::string& deployment_strategy = "blue-green"
    );

    /**
     * Get currently deployed model
     * @param model_name Model name
     * @return Deployed model information
     */
    ModelInfo get_deployed_model(const std::string& model_name);

    /**
     * Rollback to previous model version
     * @param model_name Model name
     */
    void rollback_model(const std::string& model_name);

    // Fine-Tuning Operations
    // ======================

    /**
     * Start fine-tuning from a base model
     * @param base_model_version_id Base model version ID
     * @param fine_tune_dataset_path Path to fine-tuning dataset
     * @param hyperparameters Fine-tuning hyperparameters
     * @param experiment_id Parent experiment ID
     * @return Fine-tuning run ID
     */
    int64_t start_fine_tuning(
        int64_t base_model_version_id,
        const std::string& fine_tune_dataset_path,
        const Hyperparameters& hyperparameters,
        int64_t experiment_id
    );

    // Inference Operations
    // ====================

    /**
     * Load a model for inference
     * @param version_id Model version ID
     * @return Success status
     */
    bool load_model_for_inference(int64_t version_id);

    /**
     * Run inference on input
     * @param model_version_id Model version ID
     * @param input_composition Input composition ID
     * @return Output composition ID
     */
    Blake3Hash run_inference(
        int64_t model_version_id,
        const Blake3Hash& input_composition
    );

    /**
     * Batch inference
     * @param model_version_id Model version ID
     * @param input_compositions Vector of input composition IDs
     * @return Vector of output composition IDs
     */
    std::vector<Blake3Hash> run_batch_inference(
        int64_t model_version_id,
        const std::vector<Blake3Hash>& input_compositions
    );

private:
    PGconn* conn_;

    // Helper methods
    std::string hyperparameters_to_json(const Hyperparameters& hp);
    std::string metrics_to_json(const TrainingMetrics& metrics);
    Hyperparameters json_to_hyperparameters(const std::string& json);
    TrainingMetrics json_to_metrics(const std::string& json);

    // Inference helpers
    Point4F get_composition_coordinates(const Blake3Hash& hash);
    Blake3Hash compute_hash_from_coordinates(const Point4D& coords);
};

} // namespace ml
} // namespace hypercube
