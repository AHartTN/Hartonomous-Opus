using System.Threading.Tasks;

namespace HypercubeGenerativeApi.Interfaces;

/// <summary>
/// Repository interface for composition-related database operations
/// </summary>
public interface ICompositionRepository
{
    /// <summary>
    /// Checks if a token exists in the hypercube vocabulary
    /// </summary>
    Task<bool> TokenExistsAsync(string token);

    /// <summary>
    /// Gets all tokens that exist in the vocabulary from a prompt
    /// </summary>
    Task<string[]> GetValidTokensFromPromptAsync(string prompt);

    /// <summary>
    /// Gets the composition ID (BLAKE3 hash) for a given token label
    /// </summary>
    /// <param name="label">Token label to look up</param>
    /// <returns>32-byte BLAKE3 hash, or null if not found</returns>
    Task<byte[]?> GetCompositionIdByLabelAsync(string label);
}