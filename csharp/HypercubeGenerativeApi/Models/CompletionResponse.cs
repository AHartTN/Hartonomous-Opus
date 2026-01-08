namespace HypercubeGenerativeApi.Models;

/// <summary>
/// OpenAI-compatible completion response
/// </summary>
public class CompletionResponse
{
    /// <summary>
    /// Unique identifier for the completion
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Object type (always "text_completion")
    /// </summary>
    public string Object { get; set; } = "text_completion";

    /// <summary>
    /// Unix timestamp of when the completion was created
    /// </summary>
    public long Created { get; set; }

    /// <summary>
    /// The model used for generation
    /// </summary>
    public string Model { get; set; } = string.Empty;

    /// <summary>
    /// Array of completion choices
    /// </summary>
    public List<CompletionChoice> Choices { get; set; } = new();

    /// <summary>
    /// Token usage statistics
    /// </summary>
    public UsageInfo Usage { get; set; } = new();
}

/// <summary>
/// Individual completion choice
/// </summary>
public class CompletionChoice
{
    /// <summary>
    /// The generated text
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Index of this choice in the array
    /// </summary>
    public int Index { get; set; }

    /// <summary>
    /// Log probabilities (null for now)
    /// </summary>
    public object? LogProbs { get; set; }

    /// <summary>
    /// Reason why generation stopped
    /// </summary>
    public string FinishReason { get; set; } = "stop";
}

/// <summary>
/// Token usage information
/// </summary>
public class UsageInfo
{
    /// <summary>
    /// Number of tokens in the prompt
    /// </summary>
    public int PromptTokens { get; set; }

    /// <summary>
    /// Number of tokens in the completion
    /// </summary>
    public int CompletionTokens { get; set; }

    /// <summary>
    /// Total number of tokens
    /// </summary>
    public int TotalTokens { get; set; }
}

/// <summary>
/// Models list response
/// </summary>
public class ModelsResponse
{
    /// <summary>
    /// Object type (always "list")
    /// </summary>
    public string Object { get; set; } = "list";

    /// <summary>
    /// Array of available models
    /// </summary>
    public List<ModelInfo> Data { get; set; } = new();
}

/// <summary>
/// Model information
/// </summary>
public class ModelInfo
{
    /// <summary>
    /// Model identifier
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Object type (always "model")
    /// </summary>
    public string Object { get; set; } = "model";

    /// <summary>
    /// Unix timestamp when model was created
    /// </summary>
    public long Created { get; set; }

    /// <summary>
    /// Organization that owns the model
    /// </summary>
    public string OwnedBy { get; set; } = string.Empty;
}