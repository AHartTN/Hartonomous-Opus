using System.Threading.Tasks;

namespace HypercubeGenerativeApi.Interfaces;

/// <summary>
/// Repository interface for database connection operations
/// </summary>
public interface IConnectionRepository
{
    /// <summary>
    /// Initializes database connection
    /// </summary>
    Task InitializeAsync();

    /// <summary>
    /// Checks database connectivity
    /// </summary>
    Task<bool> CheckConnectionAsync();
}