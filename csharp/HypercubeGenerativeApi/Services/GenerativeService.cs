using HypercubeGenerativeApi.Interop;
using HypercubeGenerativeApi.Models;
using Microsoft.Extensions.Logging;
using System.Runtime.InteropServices;

namespace HypercubeGenerativeApi.Services;

/// <summary>
/// Service for orchestrating hypercube generative operations
/// </summary>
public class GenerativeService
{
    private readonly ILogger<GenerativeService> _logger;
    private readonly TokenizationService _tokenizationService;
    private readonly PostgresService _postgresService;
    private bool _isInitialized;

    public GenerativeService(
        ILogger<GenerativeService> logger,
        TokenizationService tokenizationService,
        PostgresService postgresService)
    {
        _logger = logger;
        _tokenizationService = tokenizationService;
        _postgresService = postgresService;
        _isInitialized = false;
    }

    /// <summary>
    /// Initialize the generative engine caches
    /// </summary>
    public async Task InitializeAsync()
    {
        if (_isInitialized)
        {
            _logger.LogInformation("Generative service already initialized");
            return;
        }

        try
        {
            _logger.LogInformation("Loading hypercube generative caches...");

            // Load all caches in sequence
            var vocabCount = GenerativeInterop.gen_load_vocab();
            _logger.LogInformation("Loaded {Count} vocabulary entries", vocabCount);

            var bigramCount = GenerativeInterop.gen_load_bigrams();
            _logger.LogInformation("Loaded {Count} bigram scores", bigramCount);

            var attentionCount = GenerativeInterop.gen_load_attention();
            _logger.LogInformation("Loaded {Count} attention edges", attentionCount);

            // Set default configuration
            GenerativeInterop.gen_config_set_weights(0.4, 0.3, 0.2, 0.1); // centroid, pmi, attention, global
            GenerativeInterop.gen_config_set_policy(0, 0.7); // greedy=false, temperature=0.7
            GenerativeInterop.gen_config_set_filter(500, 0.1); // max_candidates, hilbert_range

            _isInitialized = true;
            _logger.LogInformation("Generative service initialized successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize generative service");
            throw;
        }
    }

    /// <summary>
    /// Generate text completion
    /// </summary>
    public async Task<CompletionResponse> GenerateCompletionAsync(CompletionRequest request)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Generative service not initialized");
        }

        var startTime = DateTimeOffset.UtcNow;

        try
        {
            // Validate request
            if (string.IsNullOrWhiteSpace(request.Prompt))
            {
                throw new ArgumentException("Prompt cannot be empty", nameof(request.Prompt));
            }

            if (request.MaxTokens <= 0 || request.MaxTokens > 2048)
            {
                throw new ArgumentException("MaxTokens must be between 1 and 2048", nameof(request.MaxTokens));
            }

            // Get valid tokens from the prompt
            var validTokens = await _postgresService.GetValidTokensFromPromptAsync(request.Prompt);

            if (validTokens.Length == 0)
            {
                _logger.LogWarning("No valid tokens found in prompt: {Prompt}", request.Prompt);
                // Fall back to a default token
                validTokens = new[] { "the" };
            }

            // Use the last valid token as the starting point for generation
            var startLabel = validTokens.Last();

            _logger.LogInformation("Generating completion for prompt: {Prompt} (found {TokenCount} valid tokens, starting with: {Start})",
                request.Prompt, validTokens.Length, startLabel);

            // Configure generation based on request
            var greedy = request.Temperature < 0.1 ? 1 : 0;
            GenerativeInterop.gen_config_set_policy(greedy, request.Temperature);

            // Generate tokens with stop sequence checking
            var tokens = new List<string>();
            var scores = new List<double>();
            var stopSequences = request.Stop ?? new[] { ".", "!", "?" };

            // Generate tokens one by one to check for stop sequences
            var remainingTokens = request.MaxTokens;
            var currentLabel = startLabel;

            while (remainingTokens > 0 && tokens.Count < request.MaxTokens)
            {
                // Use real C++ hypercube generation
                var results = new GenTokenResult[1];
                var tokenCount = GenerativeInterop.gen_generate(currentLabel, (UIntPtr)1, results);

                if (tokenCount == UIntPtr.Zero || results[0].token_index == UIntPtr.Zero)
                {
                    break; // No more tokens available
                }

                // Get the generated token label
                var labelPtr = GenerativeInterop.gen_vocab_get_label(results[0].token_index);
                if (labelPtr != IntPtr.Zero)
                {
                    var nextToken = Marshal.PtrToStringAnsi(labelPtr);
                    if (!string.IsNullOrEmpty(nextToken))
                    {
                        // Check for stop sequences
                        var shouldStop = stopSequences.Any(stop =>
                            nextToken.Contains(stop) ||
                            (tokens.Count > 0 && string.Join("", tokens.Skip(Math.Max(0, tokens.Count - 3)).Concat(new[] { nextToken })).Contains(stop))
                        );

                        if (shouldStop)
                        {
                            _logger.LogDebug("Stopping generation due to stop sequence in token: {Token}", nextToken);
                            break;
                        }

                        tokens.Add(nextToken);
                        scores.Add(results[0].score_total); // Combined geometric + PMI + attention score
                        currentLabel = nextToken;
                        remainingTokens--;

                        _logger.LogTrace("Generated token '{Token}' with geometric score {Score:F4}",
                            nextToken, results[0].score_total);
                    }
                    else
                    {
                        break; // Invalid token label
                    }
                }
                else
                {
                    break; // Invalid token index
                }
            }

            // Determine finish reason
            var finishReason = "stop";
            if (tokens.Count >= request.MaxTokens)
            {
                finishReason = "length";
            }

            // Build completion text
            var completionText = string.Join(" ", tokens.Where(t => !string.IsNullOrWhiteSpace(t)));

            // Create response
            var response = new CompletionResponse
            {
                Id = $"gen-hc-{Guid.NewGuid().ToString("N")}",
                Created = startTime.ToUnixTimeSeconds(),
                Model = request.Model,
                Choices = new List<CompletionChoice>
                {
                    new CompletionChoice
                    {
                        Text = completionText,
                        Index = 0,
                        FinishReason = finishReason
                    }
                },
                Usage = new UsageInfo
                {
                    PromptTokens = _tokenizationService.EstimateTokenCount(request.Prompt),
                    CompletionTokens = tokens.Count,
                    TotalTokens = _tokenizationService.EstimateTokenCount(request.Prompt) + tokens.Count
                }
            };

            var duration = DateTimeOffset.UtcNow - startTime;
            _logger.LogInformation("Generated completion in {Duration}ms: {TokenCount} tokens, finished: {Reason}",
                duration.TotalMilliseconds, tokens.Count, finishReason);

            return response;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating completion for prompt: {Prompt}", request.Prompt);
            throw;
        }
    }

    /// <summary>
    /// Check if caches are loaded
    /// </summary>
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Get cache statistics
    /// </summary>
    public (int VocabCount, int BigramCount, int AttentionCount) GetCacheStats()
    {
        return (
            GenerativeInterop.gen_vocab_count(),
            GenerativeInterop.gen_bigram_count(),
            GenerativeInterop.gen_attention_count()
        );
    }


}