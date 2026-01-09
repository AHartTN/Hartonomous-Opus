using System.Collections.Generic;
using System.Threading.Tasks;

namespace HypercubeGenerativeApi.Interfaces;

/// <summary>
/// Repository interface for database statistics operations
/// </summary>
public interface IDatabaseStatsRepository
{
    /// <summary>
    /// Gets database statistics
    /// </summary>
    Task<Dictionary<string, object>> GetDatabaseStatsAsync();
}