/**
 * ML Operations Controller
 * =========================
 *
 * REST API endpoints for ML lifecycle management
 * Provides experiment tracking, model versioning, and deployment
 *
 * OpenAI-compatible API style with extensions for MLOps
 */

using Microsoft.AspNetCore.Mvc;
using HypercubeGenerativeApi.Models;
using Npgsql;
using System.Text.Json;

namespace HypercubeGenerativeApi.Controllers;

[ApiController]
[Route("v1/ml")]
[Produces("application/json")]
public class MLOperationsController : ControllerBase
{
    private readonly ILogger<MLOperationsController> _logger;
    private readonly IConfiguration _configuration;
    private readonly string _connectionString;

    public MLOperationsController(
        ILogger<MLOperationsController> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        _connectionString = _configuration.GetConnectionString("HypercubeDatabase")
            ?? throw new InvalidOperationException("Database connection string not configured");
    }

    // ========================================================================
    // Experiment Management
    // ========================================================================

    /// <summary>
    /// Create a new ML experiment
    /// </summary>
    [HttpPost("experiments")]
    [ProducesResponseType(typeof(ExperimentResponse), 200)]
    [ProducesResponseType(typeof(ErrorResponse), 400)]
    public async Task<IActionResult> CreateExperiment([FromBody] CreateExperimentRequest request)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();

            var modelConfigJson = request.ModelConfig != null
                ? JsonSerializer.Serialize(request.ModelConfig)
                : "{}";

            var trainingConfigJson = request.TrainingConfig != null
                ? JsonSerializer.Serialize(request.TrainingConfig)
                : "{}";

            await using var cmd = new NpgsqlCommand(
                "SELECT create_ml_experiment($1, $2, $3::JSONB, $4::JSONB)",
                conn);

            cmd.Parameters.AddWithValue(request.Name);
            cmd.Parameters.AddWithValue(request.Description ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue(modelConfigJson);
            cmd.Parameters.AddWithValue(trainingConfigJson);

            var experimentId = (long)(await cmd.ExecuteScalarAsync() ?? 0);

            // Get experiment details
            var experiment = await GetExperimentDetails(conn, experimentId);

            _logger.LogInformation("Created experiment {ExperimentId}: {Name}", experimentId, request.Name);

            return Ok(experiment);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create experiment: {Name}", request.Name);
            return BadRequest(new ErrorResponse
            {
                Error = new ErrorDetails
                {
                    Message = ex.Message,
                    Type = "experiment_creation_error"
                }
            });
        }
    }

    /// <summary>
    /// Get experiment details
    /// </summary>
    [HttpGet("experiments/{experimentId}")]
    [ProducesResponseType(typeof(ExperimentResponse), 200)]
    [ProducesResponseType(404)]
    public async Task<IActionResult> GetExperiment(long experimentId)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();

            var experiment = await GetExperimentDetails(conn, experimentId);

            return Ok(experiment);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get experiment: {ExperimentId}", experimentId);
            return NotFound();
        }
    }

    /// <summary>
    /// List all experiments
    /// </summary>
    [HttpGet("experiments")]
    [ProducesResponseType(typeof(List<ExperimentResponse>), 200)]
    public async Task<IActionResult> ListExperiments([FromQuery] bool activeOnly = true)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();

            var query = "SELECT id, name, description, status, created_at, created_by FROM ml_experiment ";

            if (activeOnly)
            {
                query += "WHERE status = 'active' ";
            }

            query += "ORDER BY created_at DESC";

            await using var cmd = new NpgsqlCommand(query, conn);
            await using var reader = await cmd.ExecuteReaderAsync();

            var experiments = new List<ExperimentResponse>();

            while (await reader.ReadAsync())
            {
                experiments.Add(new ExperimentResponse
                {
                    ExperimentId = reader.GetInt64(0),
                    Name = reader.GetString(1),
                    Description = reader.IsDBNull(2) ? null : reader.GetString(2),
                    Status = reader.GetString(3),
                    CreatedAt = reader.GetDateTime(4),
                    CreatedBy = reader.IsDBNull(5) ? null : reader.GetString(5)
                });
            }

            return Ok(experiments);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to list experiments");
            return BadRequest(new ErrorResponse
            {
                Error = new ErrorDetails
                {
                    Message = ex.Message,
                    Type = "experiment_list_error"
                }
            });
        }
    }

    // ========================================================================
    // Training Run Management
    // ========================================================================

    /// <summary>
    /// Start a new training run
    /// </summary>
    [HttpPost("runs")]
    [ProducesResponseType(typeof(TrainingRunResponse), 200)]
    [ProducesResponseType(typeof(ErrorResponse), 400)]
    public async Task<IActionResult> StartRun([FromBody] StartRunRequest request)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();

            var hyperparametersJson = request.Hyperparameters != null
                ? JsonSerializer.Serialize(request.Hyperparameters)
                : "{}";

            await using var cmd = new NpgsqlCommand(
                "SELECT start_ml_run($1, $2, $3::JSONB)",
                conn);

            cmd.Parameters.AddWithValue(request.ExperimentId);
            cmd.Parameters.AddWithValue(request.RunName ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue(hyperparametersJson);

            var runId = (long)(await cmd.ExecuteScalarAsync() ?? 0);

            // Get run details
            var run = await GetRunDetails(conn, runId);

            _logger.LogInformation("Started run {RunId} for experiment {ExperimentId}", runId, request.ExperimentId);

            return Ok(run);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start run for experiment: {ExperimentId}", request.ExperimentId);
            return BadRequest(new ErrorResponse
            {
                Error = new ErrorDetails
                {
                    Message = ex.Message,
                    Type = "run_start_error"
                }
            });
        }
    }

    /// <summary>
    /// Log training metrics
    /// </summary>
    [HttpPost("metrics")]
    [ProducesResponseType(200)]
    [ProducesResponseType(typeof(ErrorResponse), 400)]
    public async Task<IActionResult> LogMetrics([FromBody] LogMetricsRequest request)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();

            foreach (var (metricName, metricValue) in request.Metrics)
            {
                await using var cmd = new NpgsqlCommand(
                    "SELECT log_ml_metric($1, $2, $3, $4, $5, $6)",
                    conn);

                cmd.Parameters.AddWithValue(request.RunId);
                cmd.Parameters.AddWithValue(metricName);
                cmd.Parameters.AddWithValue(metricValue);
                cmd.Parameters.AddWithValue(request.Step);
                cmd.Parameters.AddWithValue(request.Epoch ?? (object)DBNull.Value);
                cmd.Parameters.AddWithValue("train");

                await cmd.ExecuteNonQueryAsync();
            }

            _logger.LogInformation("Logged {Count} metrics for run {RunId} at step {Step}",
                request.Metrics.Count, request.RunId, request.Step);

            return Ok(new { success = true });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to log metrics for run: {RunId}", request.RunId);
            return BadRequest(new ErrorResponse
            {
                Error = new ErrorDetails
                {
                    Message = ex.Message,
                    Type = "metric_logging_error"
                }
            });
        }
    }

    /// <summary>
    /// Complete a training run
    /// </summary>
    [HttpPost("runs/complete")]
    [ProducesResponseType(200)]
    [ProducesResponseType(typeof(ErrorResponse), 400)]
    public async Task<IActionResult> CompleteRun([FromBody] CompleteRunRequest request)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();

            var metricsJson = request.Metrics != null
                ? JsonSerializer.Serialize(request.Metrics)
                : "{}";

            await using var cmd = new NpgsqlCommand(
                "SELECT complete_ml_run($1, $2::JSONB, $3)",
                conn);

            cmd.Parameters.AddWithValue(request.RunId);
            cmd.Parameters.AddWithValue(metricsJson);
            cmd.Parameters.AddWithValue(request.Status);

            await cmd.ExecuteNonQueryAsync();

            _logger.LogInformation("Completed run {RunId} with status {Status}", request.RunId, request.Status);

            return Ok(new { success = true });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to complete run: {RunId}", request.RunId);
            return BadRequest(new ErrorResponse
            {
                Error = new ErrorDetails
                {
                    Message = ex.Message,
                    Type = "run_completion_error"
                }
            });
        }
    }

    /// <summary>
    /// Get experiment leaderboard
    /// </summary>
    [HttpPost("leaderboard")]
    [ProducesResponseType(typeof(LeaderboardResponse), 200)]
    public async Task<IActionResult> GetLeaderboard([FromBody] LeaderboardRequest request)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();

            await using var cmd = new NpgsqlCommand(
                "SELECT * FROM get_experiment_leaderboard($1, $2, $3)",
                conn);

            cmd.Parameters.AddWithValue(request.ExperimentId);
            cmd.Parameters.AddWithValue(request.MetricName);
            cmd.Parameters.AddWithValue(request.Limit);

            await using var reader = await cmd.ExecuteReaderAsync();

            var entries = new List<LeaderboardEntry>();

            while (await reader.ReadAsync())
            {
                entries.Add(new LeaderboardEntry
                {
                    RunId = reader.GetInt64(0),
                    RunName = reader.IsDBNull(1) ? null : reader.GetString(1),
                    MetricValue = reader.GetDouble(2),
                    DurationSeconds = reader.GetInt32(4),
                    CreatedAt = reader.GetDateTime(5)
                });
            }

            return Ok(new LeaderboardResponse
            {
                ExperimentId = request.ExperimentId,
                MetricName = request.MetricName,
                Entries = entries
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get leaderboard for experiment: {ExperimentId}", request.ExperimentId);
            return BadRequest(new ErrorResponse
            {
                Error = new ErrorDetails
                {
                    Message = ex.Message,
                    Type = "leaderboard_error"
                }
            });
        }
    }

    // ========================================================================
    // Model Versioning
    // ========================================================================

    /// <summary>
    /// Register a new model version
    /// </summary>
    [HttpPost("models/versions")]
    [ProducesResponseType(typeof(ModelVersionResponse), 200)]
    [ProducesResponseType(typeof(ErrorResponse), 400)]
    public async Task<IActionResult> RegisterModelVersion([FromBody] RegisterModelVersionRequest request)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();

            var metricsJson = request.PerformanceMetrics != null
                ? JsonSerializer.Serialize(request.PerformanceMetrics)
                : "{}";

            await using var cmd = new NpgsqlCommand(
                "SELECT register_ml_model_version($1, $2, $3, $4, $5::JSONB)",
                conn);

            cmd.Parameters.AddWithValue(request.ModelName);
            cmd.Parameters.AddWithValue(request.Version);
            cmd.Parameters.AddWithValue(request.ArtifactPath);
            cmd.Parameters.AddWithValue(request.SourceRunId ?? (object)DBNull.Value);
            cmd.Parameters.AddWithValue(metricsJson);

            var versionId = (long)(await cmd.ExecuteScalarAsync() ?? 0);

            _logger.LogInformation("Registered model version {VersionId}: {ModelName} {Version}",
                versionId, request.ModelName, request.Version);

            return Ok(new ModelVersionResponse
            {
                VersionId = versionId,
                ModelName = request.ModelName,
                Version = request.Version,
                ArtifactPath = request.ArtifactPath,
                CreatedAt = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to register model version: {ModelName} {Version}",
                request.ModelName, request.Version);
            return BadRequest(new ErrorResponse
            {
                Error = new ErrorDetails
                {
                    Message = ex.Message,
                    Type = "model_registration_error"
                }
            });
        }
    }

    /// <summary>
    /// Deploy a model to production
    /// </summary>
    [HttpPost("models/deploy")]
    [ProducesResponseType(typeof(DeploymentResponse), 200)]
    [ProducesResponseType(typeof(ErrorResponse), 400)]
    public async Task<IActionResult> DeployModel([FromBody] DeployModelRequest request)
    {
        try
        {
            await using var conn = new NpgsqlConnection(_connectionString);
            await conn.OpenAsync();

            await using var cmd = new NpgsqlCommand(
                "SELECT deploy_ml_model($1, $2, $3)",
                conn);

            cmd.Parameters.AddWithValue(request.ModelName);
            cmd.Parameters.AddWithValue(request.VersionId);
            cmd.Parameters.AddWithValue(request.DeploymentStrategy);

            await cmd.ExecuteNonQueryAsync();

            _logger.LogInformation("Deployed model {ModelName} version {VersionId} with strategy {Strategy}",
                request.ModelName, request.VersionId, request.DeploymentStrategy);

            return Ok(new DeploymentResponse
            {
                ModelName = request.ModelName,
                ProductionVersionId = request.VersionId,
                DeploymentStrategy = request.DeploymentStrategy,
                DeployedAt = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to deploy model: {ModelName} {VersionId}",
                request.ModelName, request.VersionId);
            return BadRequest(new ErrorResponse
            {
                Error = new ErrorDetails
                {
                    Message = ex.Message,
                    Type = "model_deployment_error"
                }
            });
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    private async Task<ExperimentResponse> GetExperimentDetails(NpgsqlConnection conn, long experimentId)
    {
        await using var cmd = new NpgsqlCommand(
            "SELECT id, name, description, status, created_at, created_by, " +
            "       (SELECT COUNT(*) FROM ml_run WHERE experiment_id = $1) " +
            "FROM ml_experiment WHERE id = $1",
            conn);

        cmd.Parameters.AddWithValue(experimentId);

        await using var reader = await cmd.ExecuteReaderAsync();

        if (!await reader.ReadAsync())
        {
            throw new Exception($"Experiment {experimentId} not found");
        }

        return new ExperimentResponse
        {
            ExperimentId = reader.GetInt64(0),
            Name = reader.GetString(1),
            Description = reader.IsDBNull(2) ? null : reader.GetString(2),
            Status = reader.GetString(3),
            CreatedAt = reader.GetDateTime(4),
            CreatedBy = reader.IsDBNull(5) ? null : reader.GetString(5),
            TotalRuns = reader.GetInt32(6)
        };
    }

    private async Task<TrainingRunResponse> GetRunDetails(NpgsqlConnection conn, long runId)
    {
        await using var cmd = new NpgsqlCommand(
            "SELECT id, experiment_id, run_name, status, start_time, end_time, duration_seconds " +
            "FROM ml_run WHERE id = $1",
            conn);

        cmd.Parameters.AddWithValue(runId);

        await using var reader = await cmd.ExecuteReaderAsync();

        if (!await reader.ReadAsync())
        {
            throw new Exception($"Run {runId} not found");
        }

        return new TrainingRunResponse
        {
            RunId = reader.GetInt64(0),
            ExperimentId = reader.GetInt64(1),
            RunName = reader.IsDBNull(2) ? null : reader.GetString(2),
            Status = reader.GetString(3),
            StartTime = reader.IsDBNull(4) ? null : reader.GetDateTime(4),
            EndTime = reader.IsDBNull(5) ? null : reader.GetDateTime(5),
            DurationSeconds = reader.IsDBNull(6) ? null : reader.GetInt32(6)
        };
    }
}
