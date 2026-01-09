using HypercubeGenerativeApi.Interfaces;
using HypercubeGenerativeApi.Services;

namespace HypercubeGenerativeApi.Repositories;

/// <summary>
/// PostgreSQL implementation of composition repository
/// </summary>
public class PostgresCompositionRepository : ICompositionRepository
{
    private readonly PostgresService _postgresService;

    public PostgresCompositionRepository(PostgresService postgresService)
    {
        _postgresService = postgresService;
    }

    /// <inheritdoc/>
    public async Task<bool> TokenExistsAsync(string token)
    {
        return await _postgresService.TokenExistsAsync(token);
    }

    /// <inheritdoc/>
    public async Task<string[]> GetValidTokensFromPromptAsync(string prompt)
    {
        return await _postgresService.GetValidTokensFromPromptAsync(prompt);
    }
}