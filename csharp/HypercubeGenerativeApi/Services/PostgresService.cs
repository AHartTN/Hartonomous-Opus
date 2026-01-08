using Microsoft.Extensions.Logging;
using Npgsql;
using System.Data;

namespace HypercubeGenerativeApi.Services;

/// <summary>
/// Service for PostgreSQL database connectivity and hypercube operations
/// </summary>
public class PostgresService : IDisposable
{
    private readonly ILogger<PostgresService> _logger;
    private readonly string _connectionString;
    private NpgsqlConnection? _connection;
    private bool _disposed;

    public PostgresService(ILogger<PostgresService> logger, IConfiguration configuration)
    {
        _logger = logger;
        _connectionString = configuration.GetConnectionString("HypercubeDatabase")
            ?? "Host=localhost;Port=5432;Username=hartonomous;Password=hartonomous;Database=hypercube";
    }

    /// <summary>
    /// Initialize database connection
    /// </summary>
    public async Task InitializeAsync()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PostgresService));

        try
        {
            _connection = new NpgsqlConnection(_connectionString);
            await _connection.OpenAsync();
            _logger.LogInformation("PostgreSQL connection established successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to connect to PostgreSQL database");
            throw;
        }
    }

    /// <summary>
    /// Check database connectivity
    /// </summary>
    public async Task<bool> CheckConnectionAsync()
    {
        if (_connection == null || _disposed) return false;

        try
        {
            await using var cmd = new NpgsqlCommand("SELECT 1", _connection);
            await cmd.ExecuteScalarAsync();
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Database connectivity check failed");
            return false;
        }
    }

    /// <summary>
    /// Get database statistics
    /// </summary>
    public async Task<Dictionary<string, object>> GetDatabaseStatsAsync()
    {
        if (_connection == null || _disposed)
        {
            return new Dictionary<string, object> { ["error"] = "No database connection" };
        }

        try
        {
            var stats = new Dictionary<string, object>();

            await using var cmd = new NpgsqlCommand(@"
                SELECT 'atoms' as stat_name, COUNT(*) as stat_value FROM atom
                UNION ALL
                SELECT 'compositions', COUNT(*) FROM composition
                UNION ALL
                SELECT 'compositions_with_centroid', COUNT(*) FROM composition WHERE centroid IS NOT NULL
                UNION ALL
                SELECT 'relations', COUNT(*) FROM relation
                UNION ALL
                SELECT 'attention_edges', COUNT(*) FROM relation WHERE relation_type = 'A'
                UNION ALL
                SELECT 'sequence_edges', COUNT(*) FROM relation WHERE relation_type = 'S'
            ", _connection);

            await using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                var statName = reader.GetString(0);
                var statValue = reader.GetInt64(1);
                stats[statName] = statValue;
            }

            return stats;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving database statistics");
            return new Dictionary<string, object> { ["error"] = ex.Message };
        }
    }

    /// <summary>
    /// Check if a token exists in the hypercube vocabulary
    /// </summary>
    public async Task<bool> TokenExistsAsync(string token)
    {
        if (_connection == null || _disposed || string.IsNullOrWhiteSpace(token))
        {
            return false;
        }

        try
        {
            await using var cmd = new NpgsqlCommand(@"
                SELECT 1
                FROM composition
                WHERE label = @token
                  AND centroid IS NOT NULL
                LIMIT 1
            ", _connection);

            cmd.Parameters.AddWithValue("@token", token);

            var result = await cmd.ExecuteScalarAsync();
            return result != null;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error checking token existence '{Token}'", token);
            return false;
        }
    }

    /// <summary>
    /// Get all tokens that exist in the vocabulary from a prompt
    /// </summary>
    public async Task<string[]> GetValidTokensFromPromptAsync(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return Array.Empty<string>();
        }

        // Basic word tokenization
        var tokens = prompt.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':', '"', '\'' },
                                  StringSplitOptions.RemoveEmptyEntries)
                           .Select(w => w.Trim().ToLowerInvariant())
                           .Where(w => !string.IsNullOrEmpty(w))
                           .Distinct() // Remove duplicates
                           .Take(100) // Reasonable limit
                           .ToArray();

        var validTokens = new List<string>();
        foreach (var token in tokens)
        {
            if (await TokenExistsAsync(token))
            {
                validTokens.Add(token);
            }
        }

        _logger.LogDebug("Found {ValidCount} valid tokens from {TotalCount} in prompt",
            validTokens.Count, tokens.Length);

        return validTokens.ToArray();
    }

    /// <summary>
    /// Dispose database connection
    /// </summary>
    public void Dispose()
    {
        _connection?.Dispose();
        _connection = null;
    }
}