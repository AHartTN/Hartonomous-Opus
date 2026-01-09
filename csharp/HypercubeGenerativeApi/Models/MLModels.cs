/**
 * ML Operations Models
 * ====================
 *
 * Data models for ML lifecycle management API
 */

using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace HypercubeGenerativeApi.Models;

// ============================================================================
// Experiment Models
// ============================================================================

public class CreateExperimentRequest
{
    [Required]
    [StringLength(200)]
    public string Name { get; set; } = string.Empty;

    public string? Description { get; set; }

    [JsonPropertyName("model_config")]
    public Dictionary<string, object>? ModelConfig { get; set; }

    [JsonPropertyName("training_config")]
    public Dictionary<string, object>? TrainingConfig { get; set; }

    public List<string>? Tags { get; set; }
}

public class ExperimentResponse
{
    [JsonPropertyName("experiment_id")]
    public long ExperimentId { get; set; }

    public string Name { get; set; } = string.Empty;
    public string? Description { get; set; }
    public string Status { get; set; } = "active";

    [JsonPropertyName("total_runs")]
    public int TotalRuns { get; set; }

    [JsonPropertyName("created_at")]
    public DateTime CreatedAt { get; set; }

    [JsonPropertyName("created_by")]
    public string? CreatedBy { get; set; }

    public List<string>? Tags { get; set; }
}

// ============================================================================
// Training Run Models
// ============================================================================

public class StartRunRequest
{
    [Required]
    [JsonPropertyName("experiment_id")]
    public long ExperimentId { get; set; }

    [JsonPropertyName("run_name")]
    public string? RunName { get; set; }

    public HyperparametersRequest? Hyperparameters { get; set; }

    [JsonPropertyName("random_seed")]
    public int? RandomSeed { get; set; }

    [JsonPropertyName("git_commit")]
    public string? GitCommit { get; set; }
}

public class HyperparametersRequest
{
    [JsonPropertyName("learning_rate")]
    public double LearningRate { get; set; } = 0.001;

    [JsonPropertyName("batch_size")]
    public int BatchSize { get; set; } = 32;

    [JsonPropertyName("num_epochs")]
    public int NumEpochs { get; set; } = 10;

    [JsonPropertyName("weight_decay")]
    public double WeightDecay { get; set; } = 0.0;

    [JsonPropertyName("dropout_rate")]
    public double DropoutRate { get; set; } = 0.1;

    public string Optimizer { get; set; } = "adam";

    [JsonExtensionData]
    public Dictionary<string, object>? CustomParameters { get; set; }
}

public class TrainingRunResponse
{
    [JsonPropertyName("run_id")]
    public long RunId { get; set; }

    [JsonPropertyName("experiment_id")]
    public long ExperimentId { get; set; }

    [JsonPropertyName("run_name")]
    public string? RunName { get; set; }

    public string Status { get; set; } = "pending";

    public HyperparametersRequest? Hyperparameters { get; set; }

    [JsonPropertyName("start_time")]
    public DateTime? StartTime { get; set; }

    [JsonPropertyName("end_time")]
    public DateTime? EndTime { get; set; }

    [JsonPropertyName("duration_seconds")]
    public int? DurationSeconds { get; set; }

    public Dictionary<string, double>? Metrics { get; set; }
}

public class LogMetricRequest
{
    [Required]
    [JsonPropertyName("run_id")]
    public long RunId { get; set; }

    [Required]
    [JsonPropertyName("metric_name")]
    public string MetricName { get; set; } = string.Empty;

    [Required]
    [JsonPropertyName("metric_value")]
    public double MetricValue { get; set; }

    [Required]
    public int Step { get; set; }

    public int? Epoch { get; set; }

    [JsonPropertyName("metric_type")]
    public string MetricType { get; set; } = "train";
}

public class LogMetricsRequest
{
    [Required]
    [JsonPropertyName("run_id")]
    public long RunId { get; set; }

    [Required]
    public Dictionary<string, double> Metrics { get; set; } = new();

    [Required]
    public int Step { get; set; }

    public int? Epoch { get; set; }
}

public class CompleteRunRequest
{
    [Required]
    [JsonPropertyName("run_id")]
    public long RunId { get; set; }

    public Dictionary<string, double>? Metrics { get; set; }

    public string Status { get; set; } = "completed";

    [JsonPropertyName("error_message")]
    public string? ErrorMessage { get; set; }
}

public class SaveCheckpointRequest
{
    [Required]
    [JsonPropertyName("run_id")]
    public long RunId { get; set; }

    [Required]
    [JsonPropertyName("checkpoint_path")]
    public string CheckpointPath { get; set; } = string.Empty;

    [Required]
    public int Step { get; set; }

    [Required]
    public int Epoch { get; set; }

    public Dictionary<string, double>? Metrics { get; set; }

    [JsonPropertyName("is_best")]
    public bool IsBest { get; set; } = false;
}

public class CheckpointResponse
{
    [JsonPropertyName("checkpoint_id")]
    public long CheckpointId { get; set; }

    [JsonPropertyName("run_id")]
    public long RunId { get; set; }

    public int Step { get; set; }
    public int Epoch { get; set; }

    [JsonPropertyName("checkpoint_path")]
    public string CheckpointPath { get; set; } = string.Empty;

    [JsonPropertyName("is_best")]
    public bool IsBest { get; set; }

    [JsonPropertyName("created_at")]
    public DateTime CreatedAt { get; set; }
}

// ============================================================================
// Model Version Models
// ============================================================================

public class RegisterModelVersionRequest
{
    [Required]
    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = string.Empty;

    [Required]
    public string Version { get; set; } = string.Empty;

    [Required]
    [JsonPropertyName("artifact_path")]
    public string ArtifactPath { get; set; } = string.Empty;

    [JsonPropertyName("source_run_id")]
    public long? SourceRunId { get; set; }

    [JsonPropertyName("performance_metrics")]
    public Dictionary<string, double>? PerformanceMetrics { get; set; }

    [JsonPropertyName("model_type")]
    public string? ModelType { get; set; }

    public string? Description { get; set; }

    public List<string>? Tags { get; set; }
}

public class ModelVersionResponse
{
    [JsonPropertyName("version_id")]
    public long VersionId { get; set; }

    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = string.Empty;

    public string Version { get; set; } = string.Empty;

    [JsonPropertyName("model_type")]
    public string? ModelType { get; set; }

    [JsonPropertyName("artifact_path")]
    public string ArtifactPath { get; set; } = string.Empty;

    [JsonPropertyName("validation_status")]
    public string ValidationStatus { get; set; } = "pending";

    [JsonPropertyName("is_deployed")]
    public bool IsDeployed { get; set; }

    [JsonPropertyName("performance_metrics")]
    public Dictionary<string, double>? PerformanceMetrics { get; set; }

    [JsonPropertyName("created_at")]
    public DateTime CreatedAt { get; set; }

    [JsonPropertyName("created_by")]
    public string? CreatedBy { get; set; }

    public List<string>? Tags { get; set; }
}

public class ApproveModelRequest
{
    [Required]
    [JsonPropertyName("version_id")]
    public long VersionId { get; set; }

    [JsonPropertyName("approval_notes")]
    public string? ApprovalNotes { get; set; }
}

// ============================================================================
// Model Deployment Models
// ============================================================================

public class DeployModelRequest
{
    [Required]
    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = string.Empty;

    [Required]
    [JsonPropertyName("version_id")]
    public long VersionId { get; set; }

    [JsonPropertyName("deployment_strategy")]
    public string DeploymentStrategy { get; set; } = "blue-green";

    [JsonPropertyName("traffic_percentage")]
    [Range(0, 100)]
    public int TrafficPercentage { get; set; } = 100;
}

public class DeploymentResponse
{
    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = string.Empty;

    [JsonPropertyName("production_version_id")]
    public long ProductionVersionId { get; set; }

    [JsonPropertyName("previous_version_id")]
    public long? PreviousVersionId { get; set; }

    [JsonPropertyName("deployment_strategy")]
    public string DeploymentStrategy { get; set; } = "blue-green";

    [JsonPropertyName("deployed_at")]
    public DateTime DeployedAt { get; set; }

    [JsonPropertyName("deployed_by")]
    public string? DeployedBy { get; set; }

    [JsonPropertyName("health_status")]
    public string HealthStatus { get; set; } = "healthy";
}

// ============================================================================
// Inference Models
// ============================================================================

public class InferenceRequest
{
    [Required]
    [JsonPropertyName("model_version_id")]
    public long ModelVersionId { get; set; }

    [Required]
    public string Input { get; set; } = string.Empty;

    public Dictionary<string, object>? Parameters { get; set; }
}

public class InferenceResponse
{
    public string Output { get; set; } = string.Empty;

    [JsonPropertyName("inference_time_ms")]
    public double InferenceTimeMs { get; set; }

    public Dictionary<string, object>? Metadata { get; set; }
}

// ============================================================================
// Leaderboard Models
// ============================================================================

public class LeaderboardRequest
{
    [Required]
    [JsonPropertyName("experiment_id")]
    public long ExperimentId { get; set; }

    [JsonPropertyName("metric_name")]
    public string MetricName { get; set; } = "accuracy";

    public int Limit { get; set; } = 10;
}

public class LeaderboardEntry
{
    [JsonPropertyName("run_id")]
    public long RunId { get; set; }

    [JsonPropertyName("run_name")]
    public string? RunName { get; set; }

    [JsonPropertyName("metric_value")]
    public double MetricValue { get; set; }

    public HyperparametersRequest? Hyperparameters { get; set; }

    [JsonPropertyName("duration_seconds")]
    public int DurationSeconds { get; set; }

    [JsonPropertyName("created_at")]
    public DateTime CreatedAt { get; set; }
}

public class LeaderboardResponse
{
    [JsonPropertyName("experiment_id")]
    public long ExperimentId { get; set; }

    [JsonPropertyName("metric_name")]
    public string MetricName { get; set; } = string.Empty;

    public List<LeaderboardEntry> Entries { get; set; } = new();
}
