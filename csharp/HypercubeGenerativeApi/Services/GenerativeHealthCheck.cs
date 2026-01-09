using HypercubeGenerativeApi.Interfaces;
using Microsoft.Extensions.Diagnostics.HealthChecks;

namespace HypercubeGenerativeApi.Services;

/// <summary>
/// Health check for generative service
/// </summary>
public class GenerativeHealthCheck : IHealthCheck
{
    private readonly GenerativeService _generativeService;
    private readonly IConnectionRepository _connectionRepository;
    private readonly IDatabaseStatsRepository _databaseStatsRepository;

    public GenerativeHealthCheck(
        GenerativeService generativeService,
        IConnectionRepository connectionRepository,
        IDatabaseStatsRepository databaseStatsRepository)
    {
        _generativeService = generativeService;
        _connectionRepository = connectionRepository;
        _databaseStatsRepository = databaseStatsRepository;
    }

    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        var data = new Dictionary<string, object>();

        // Check if generative service is initialized
        var generativeHealthy = _generativeService.IsInitialized;
        data["cache_loaded"] = generativeHealthy;

        // Get cache statistics
        var (vocabCount, bigramCount, attentionCount) = _generativeService.GetCacheStats();
        data["vocab_count"] = vocabCount;
        data["bigram_count"] = bigramCount;
        data["attention_count"] = attentionCount;

        // Check database connectivity
        var dbHealthy = await _connectionRepository.CheckConnectionAsync();
        data["database_connected"] = dbHealthy;

        // Overall health
        var overallHealthy = generativeHealthy && dbHealthy;
        var status = overallHealthy ? HealthStatus.Healthy : HealthStatus.Degraded;

        // Generate description
        var description = overallHealthy
            ? "Hypercube generative service is healthy"
            : "Hypercube generative service has issues";

        return new HealthCheckResult(status, description, data: data);
    }
}