using System.Threading.Tasks;
using HypercubeGenerativeApi.Interfaces;
using HypercubeGenerativeApi.Services;

namespace HypercubeGenerativeApi.Repositories;

/// <summary>
/// PostgreSQL implementation of connection repository
/// </summary>
public class PostgresConnectionRepository : IConnectionRepository
{
    private readonly PostgresService _postgresService;

    public PostgresConnectionRepository(PostgresService postgresService)
    {
        _postgresService = postgresService;
    }

    /// <inheritdoc/>
    public async Task InitializeAsync()
    {
        await _postgresService.InitializeAsync();
    }

    /// <inheritdoc/>
    public async Task<bool> CheckConnectionAsync()
    {
        return await _postgresService.CheckConnectionAsync();
    }
}