namespace HypercubeGenerativeApi.Models;

/// <summary>
/// OpenAI-compatible completion request
/// </summary>
public class CompletionRequest
{
    /// <summary>
    /// The model to use for generation
    /// </summary>
    public string Model { get; set; } = "hypercube-generative";

    /// <summary>
    /// The prompt to generate completions for
    /// </summary>
    public string Prompt { get; set; } = string.Empty;

    /// <summary>
    /// The maximum number of tokens to generate
    /// </summary>
    public int MaxTokens { get; set; } = 16;

    /// <summary>
    /// Sampling temperature between 0 and 2
    /// </summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>
    /// Nucleus sampling parameter
    /// </summary>
    public double? TopP { get; set; }

    /// <summary>
    /// Whether to stream the response
    /// </summary>
    public bool Stream { get; set; } = false;

    /// <summary>
    /// Sequences where the API will stop generating
    /// </summary>
    public string[]? Stop { get; set; }

    /// <summary>
    /// Number of completions to generate
    /// </summary>
    public int N { get; set; } = 1;

    /// <summary>
    /// Whether to return log probabilities
    /// </summary>
    public bool? LogProbs { get; set; }

    /// <summary>
    /// Echo the prompt back in the response
    /// </summary>
    public bool Echo { get; set; } = false;

    /// <summary>
    /// Random seed for deterministic generation
    /// </summary>
    public int? Seed { get; set; }
}