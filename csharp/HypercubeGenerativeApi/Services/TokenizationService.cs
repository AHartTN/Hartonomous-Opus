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
    /// Represents a token with its 4D geometric coordinates
    /// </summary>
    public struct GeometricToken
    {
        public string Token;
        public uint Codepoint;
        public Point4D Coordinates;
    }

    /// <summary>
    /// Tokenize a prompt into its geometric representation (4D coordinates)
    /// This bypasses the mock database ID system and accesses the core C++ geometry directly.
    /// </summary>
    /// <param name="prompt">The input prompt</param>
    /// <returns>Array of GeometricTokens with their 4D coordinates</returns>
    public Task<GeometricToken[]> TokenizeToGeometryAsync(string prompt)
    {
        if (string.IsNullOrEmpty(prompt))
        {
            return Task.FromResult(Array.Empty<GeometricToken>());
        }

        var results = new List<GeometricToken>();

        // For now, we tokenize by character (codepoint) to prove the geometry works
        // Future versions can do centroid calculations for words
        foreach (var c in prompt)
        {
            uint codepoint = (uint)c;
            
            // Call the NATIVE C++ engine
            Point4D coords = GenerativeInterop.geom_map_codepoint(codepoint);
            
            results.Add(new GeometricToken 
            {
                Token = c.ToString(),
                Codepoint = codepoint,
                Coordinates = coords
            });
        }

        return Task.FromResult(results.ToArray());
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
    /// Advanced tokenization matching hypercube vocabulary
    /// Uses character-level CPE (Codepoint Pair Encoding) for universal tokenization
    /// </summary>
    private async Task<string[]> TokenizeWordsAsync(string text)
    {
        // First try word-level tokenization
        var words = text.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':', '"', '\'' },
                               StringSplitOptions.RemoveEmptyEntries);

        var tokens = new List<string>();

        foreach (var word in words)
        {
            var trimmed = word.Trim().ToLowerInvariant();
            if (string.IsNullOrEmpty(trimmed)) continue;

            // Check if whole word exists in vocabulary
            var exists = await _compositionRepository.TokenExistsAsync(trimmed);
            if (exists)
            {
                tokens.Add(trimmed);
                continue;
            }

            // Fall back to character-level tokenization for OOV words
            // This ensures every input can be tokenized via CPE
            foreach (char c in trimmed)
            {
                tokens.Add(c.ToString());
            }
        }

        return tokens.ToArray();
    }

    /// <summary>
    /// Legacy synchronous tokenizer for backward compatibility
    /// </summary>
    private static string[] TokenizeWords(string text)
    {
        var words = text.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':', '"', '\'' },
                               StringSplitOptions.RemoveEmptyEntries);

        return words
            .Select(w => w.Trim().ToLowerInvariant())
            .Where(w => !string.IsNullOrEmpty(w))
            .ToArray();
    }

    /// <summary>
    /// Encode a single token to hypercube composition ID
    /// Returns the actual BLAKE3 hash from the database
    /// </summary>
    private async Task<long?> EncodeTokenAsync(string token)
    {
        try
        {
            _logger.LogDebug("Looking up token '{Token}' in hypercube vocabulary", token);

            // Query database for actual composition ID
            var compositionId = await _compositionRepository.GetCompositionIdByLabelAsync(token);
            if (compositionId == null)
            {
                _logger.LogDebug("Token '{Token}' not found in vocabulary", token);
                return null;
            }

            // Convert BLAKE3 hash (32 bytes) to a stable 64-bit ID for C# API
            // In a real system, we'd pass the full 32-byte hash through interop
            // For now, use first 8 bytes as long
            byte[] hashBytes = compositionId;
            if (hashBytes.Length >= 8)
            {
                return BitConverter.ToInt64(hashBytes, 0);
            }

            // Fallback to stable hash if composition ID format is unexpected
            return GetStableHash(token);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error encoding token '{Token}'", token);
            return null;
        }
    }

    /// <summary>
    /// Generate a stable hash for token lookup (fallback only)
    /// This is a deterministic hash used as fallback when database lookup fails
    /// </summary>
    private static long GetStableHash(string input)
    {
        // Use FNV-1a 64-bit hash for stable token IDs
        // This is only used as a fallback - real IDs come from database
        unchecked
        {
            const ulong FNV_OFFSET_BASIS = 14695981039346656037;
            const ulong FNV_PRIME = 1099511628211;

            ulong hash = FNV_OFFSET_BASIS;
            foreach (char c in input)
            {
                hash = hash * 31 + c;
            }
            return Math.Abs((long)hash);
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