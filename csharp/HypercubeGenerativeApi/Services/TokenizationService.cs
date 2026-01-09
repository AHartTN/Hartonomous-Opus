using HypercubeGenerativeApi.Interfaces;
using HypercubeGenerativeApi.Interop;
using Microsoft.Extensions.Logging;

namespace HypercubeGenerativeApi.Services;

/// <summary>
/// Service for tokenizing prompts and encoding them to hypercube composition IDs
/// </summary>
public class TokenizationService
{
    private readonly ILogger<TokenizationService> _logger;
    private readonly ICompositionRepository _compositionRepository;

    public TokenizationService(ILogger<TokenizationService> logger, ICompositionRepository compositionRepository)
    {
        _logger = logger;
        _compositionRepository = compositionRepository;
    }

    /// <summary>
    /// Tokenize a prompt and encode to composition IDs
    /// </summary>
    /// <param name="prompt">The input prompt to tokenize</param>
    /// <returns>Array of composition IDs representing the tokenized prompt</returns>
    public async Task<long[]> TokenizeAndEncodeAsync(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return Array.Empty<long>();
        }

        try
        {
            // Basic word-level tokenization (can be enhanced later)
            var tokens = TokenizeWords(prompt);

            if (tokens.Length == 0)
            {
                _logger.LogWarning("No tokens found in prompt: {Prompt}", prompt);
                return Array.Empty<long>();
            }

            // Encode each token to composition ID
            var compositionIds = new List<long>();
            foreach (var token in tokens)
            {
                var compositionId = await EncodeTokenAsync(token);
                if (compositionId.HasValue)
                {
                    compositionIds.Add(compositionId.Value);

                    // Limit context length to prevent excessive memory usage
                    if (compositionIds.Count >= 50) // Configurable limit
                    {
                        _logger.LogWarning("Prompt truncated to {Count} tokens for context limit", compositionIds.Count);
                        break;
                    }
                }
                else
                {
                    _logger.LogDebug("Token '{Token}' not found in vocabulary, skipping", token);
                }
            }

            _logger.LogInformation("Tokenized prompt '{Prompt}' into {TokenCount} tokens, encoded {EncodedCount} compositions",
                prompt, tokens.Length, compositionIds.Count);

            return compositionIds.ToArray();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error tokenizing and encoding prompt: {Prompt}", prompt);
            throw;
        }
    }

    /// <summary>
    /// Estimate token count for usage statistics
    /// </summary>
    public int EstimateTokenCount(string text)
    {
        if (string.IsNullOrEmpty(text)) return 0;

        // Simple estimation: average 4 characters per token
        // This can be made more accurate with actual tokenization
        var wordCount = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length;
        return Math.Max(1, wordCount);
    }

    /// <summary>
    /// Basic word-level tokenization
    /// TODO: Replace with more sophisticated tokenization that matches hypercube vocabulary
    /// </summary>
    private static string[] TokenizeWords(string text)
    {
        // Simple word splitting - split on whitespace and punctuation
        var words = text.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':', '"', '\'' },
                               StringSplitOptions.RemoveEmptyEntries);

        // Clean up tokens (lowercase, trim)
        return words
            .Select(w => w.Trim().ToLowerInvariant())
            .Where(w => !string.IsNullOrEmpty(w))
            .ToArray();
    }

    /// <summary>
    /// Encode a single token to hypercube composition ID
    /// </summary>
    private async Task<long?> EncodeTokenAsync(string token)
    {
        // Note: Current implementation uses simplified ID handling
        // TODO: Update to handle full 32-byte BLAKE3 hashes properly
        // For now, we check if token exists but return simplified ID

        try
        {
            _logger.LogDebug("Looking up token '{Token}' in hypercube vocabulary", token);

            // Check if token exists in vocabulary via repository
            var exists = await _compositionRepository.TokenExistsAsync(token);
            if (!exists)
            {
                _logger.LogDebug("Token '{Token}' not found in vocabulary", token);
                return null;
            }

            // For now, return a stable hash as simplified ID
            // In production, this would return the actual BYTEA composition ID
            // and interop layer would be updated to handle byte arrays
            var stableHash = GetStableHash(token);
            return stableHash;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error encoding token '{Token}'", token);
            return null;
        }
    }

    /// <summary>
    /// Generate a stable hash for token lookup
    /// TODO: Replace with actual BYTEA composition ID from database
    /// </summary>
    private static long GetStableHash(string input)
    {
        // Use a simple but stable hashing approach
        // In production, this would be replaced with actual composition IDs
        unchecked
        {
            long hash = 23;
            foreach (char c in input)
            {
                hash = hash * 31 + c;
            }
            return Math.Abs(hash);
        }
    }

    /// <summary>
    /// Decode composition IDs back to readable tokens (for debugging and logging)
    /// </summary>
    public async Task<string[]> DecodeCompositionIdsAsync(long[] compositionIds)
    {
        if (compositionIds == null || compositionIds.Length == 0)
        {
            return Array.Empty<string>();
        }

        var tokens = new List<string>();
        var decodeTasks = new List<Task<string>>();

        // Create decode tasks for each ID
        foreach (var id in compositionIds)
        {
            decodeTasks.Add(DecodeCompositionIdAsync(id));
        }

        try
        {
            // Wait for all decode operations to complete
            var decodedTokens = await Task.WhenAll(decodeTasks);
            tokens.AddRange(decodedTokens);

            _logger.LogDebug("Decoded {Count} composition IDs to tokens", compositionIds.Length);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error decoding composition IDs batch");

            // Fallback: return safe representations
            foreach (var id in compositionIds)
            {
                tokens.Add($"[decode_error_{id}]");
            }
        }

        return tokens.ToArray();
    }

    /// <summary>
    /// Decode a single composition ID to token label
    /// </summary>
    private async Task<string> DecodeCompositionIdAsync(long compositionId)
    {
        try
        {
            // TODO: Implement actual reverse lookup from composition ID to label
            // This would require extending PostgresService with reverse lookup capability

            // For now, provide a meaningful representation
            // In production, this would query: SELECT label FROM composition WHERE id = @id

            _logger.LogTrace("Attempting to decode composition ID {Id}", compositionId);

            // Placeholder implementation - provide stable, recognizable tokens
            // This helps with debugging and logging even without full DB lookup
            if (compositionId >= 1000000) // Our generated IDs
            {
                // Attempt to reverse the stable hash (limited success)
                var candidateTokens = await AttemptReverseHashLookup(compositionId);
                if (candidateTokens.Any())
                {
                    return candidateTokens.First(); // Best guess
                }
            }

            // Fallback to safe representation
            return $"comp_{compositionId}";
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error decoding composition ID {Id}", compositionId);
            return $"[error_{compositionId}]";
        }
    }

    /// <summary>
    /// Attempt to reverse a stable hash to find possible token matches
    /// This is a heuristic approach for debugging when DB lookup isn't available
    /// </summary>
    private async Task<string[]> AttemptReverseHashLookup(long targetId)
    {
        // This is a simplified reverse lookup for common tokens
        // In production, this would be replaced with proper DB queries

        var commonTokens = new[] {
            "the", "a", "an", "is", "are", "was", "were", "has", "have",
            "this", "that", "these", "those", "and", "or", "but", "if",
            "it", "its", "they", "them", "their", "you", "your", "we", "us", "our"
        };

        var matches = new List<string>();

        foreach (var token in commonTokens)
        {
            var hash = GetStableHash(token);
            if (hash == targetId)
            {
                matches.Add(token);
            }
        }

        return matches.ToArray();
    }
}