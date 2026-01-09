using System.Collections.Generic;
using System.Threading.Tasks;
using HypercubeGenerativeApi.Interfaces;
using HypercubeGenerativeApi.Services;

namespace HypercubeGenerativeApi.Repositories;

/// <summary>
/// PostgreSQL implementation of database stats repository
/// </summary>
public class PostgresDatabaseStatsRepository : IDatabaseStatsRepository
{
    private readonly PostgresService _postgresService;

    public PostgresDatabaseStatsRepository(PostgresService postgresService)
    {
        _postgresService = postgresService;
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, object>> GetDatabaseStatsAsync()
    {
        return await _postgresService.GetDatabaseStatsAsync();
    }
}