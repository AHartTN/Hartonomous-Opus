# GenerativeService Documentation

## Overview

The `GenerativeService` is the core orchestration service of the Hypercube Generative API. It manages the complete lifecycle of text generation requests, bridging the gap between OpenAI-compatible HTTP requests and the underlying C++ hypercube generation engine.

## Class Structure

```csharp
public class GenerativeService
{
    // Dependencies
    private readonly ILogger<GenerativeService> _logger;
    private readonly TokenizationService _tokenizationService;
    private readonly PostgresService _postgresService;

    // State
    private bool _isInitialized;

    // Core Methods
    public Task InitializeAsync()
    public Task<CompletionResponse> GenerateCompletionAsync(CompletionRequest request)
    public (int VocabCount, int BigramCount, int AttentionCount) GetCacheStats()

    // Properties
    public bool IsInitialized => _isInitialized;
}
```

## Lifecycle Management

### Initialization Process

```csharp
public async Task InitializeAsync()
{
    // 1. Check if already initialized
    if (_isInitialized) return;

    // 2. Load hypercube caches via C++ interop
    var vocabCount = GenerativeInterop.gen_load_vocab();
    var bigramCount = GenerativeInterop.gen_load_bigrams();
    var attentionCount = GenerativeInterop.gen_load_attention();

    // 3. Configure default generation parameters
    GenerativeInterop.gen_config_set_weights(0.4, 0.3, 0.2, 0.1);
    GenerativeInterop.gen_config_set_policy(0, 0.7);
    GenerativeInterop.gen_config_set_filter(500, 0.1);

    // 4. Mark as ready
    _isInitialized = true;
}
```

**Cache Loading Details**:
- **Vocabulary**: ~50,000+ compositions with 4D centroids
- **Bigrams**: PMI (Pointwise Mutual Information) scores between token pairs
- **Attention**: Pre-computed attention weights from model ingestion
- **Memory Impact**: ~500MB loaded into C++ process memory

### Service State

- **`_isInitialized`**: Guards against uninitialized generation attempts
- **Thread Safety**: Single initialization, read-only state after init
- **Health Check**: `IsInitialized` property used by health monitoring

## Generation Workflow

### Main Entry Point

```csharp
public async Task<CompletionResponse> GenerateCompletionAsync(CompletionRequest request)
{
    // Phase 1: Validation
    ValidateRequest(request);

    // Phase 2: Tokenization
    var validTokens = await _postgresService.GetValidTokensFromPromptAsync(request.Prompt);

    // Phase 3: Generation Setup
    ConfigureGeneration(request);

    // Phase 4: Token-by-Token Generation
    var tokens = await GenerateWithStopSequences(validTokens.Last(), request);

    // Phase 5: Response Formatting
    return FormatResponse(tokens, request);
}
```

### Phase 1: Request Validation

**Input Validation**:
- Model must be "hypercube-generative"
- Prompt cannot be null/empty or >10,000 characters
- MaxTokens: 1-2048 range
- Temperature: 0.0-2.0 range (maps to greedy vs stochastic)

**Service Readiness**:
- Checks `_isInitialized` flag
- Throws `InvalidOperationException` if not ready

### Phase 2: Tokenization & Encoding

**Process Flow**:
```csharp
// Get tokens that exist in hypercube vocabulary
var validTokens = await _postgresService.GetValidTokensFromPromptAsync(request.Prompt);

// Fallback for prompts with no valid tokens
if (validTokens.Length == 0) {
    validTokens = new[] { "the" }; // Default starting token
}

// Use last valid token as generation starting point
var startLabel = validTokens.Last();
```

**Tokenization Strategy**:
- Simple whitespace/punctuation splitting
- Database validation against composition.label
- Last-token continuation (maintains context flow)
- Fallback to common token if no matches

### Phase 3: Generation Configuration

**OpenAI → Hypercube Parameter Mapping**:

| OpenAI Parameter | Hypercube Mapping | C++ Function |
|------------------|-------------------|--------------|
| `temperature = 0` | Greedy selection | `gen_config_set_policy(1, temp)` |
| `temperature > 0` | Stochastic sampling | `gen_config_set_policy(0, temp)` |
| `max_tokens` | Generation length limit | Loop control in C# |
| `stop` | Early termination sequences | String matching in generation loop |

### Phase 4: Token-by-Token Generation

**Stop Sequence Implementation**:
```csharp
while (remainingTokens > 0 && tokens.Count < request.MaxTokens) {
    // Generate one token
    var resultPtr = GenerativeInterop.gen_generate(currentLabel, 1, out var tokenCount);

    if (resultPtr == IntPtr.Zero) break;

    try {
        // Extract generated token
        var token = ExtractTokenFromResult(resultPtr);

        // Check stop sequences
        var shouldStop = stopSequences.Any(stop =>
            token.Contains(stop) ||
            (tokens.Count > 0 && CombinedTextEndsWith(tokens, token, stop))
        );

        if (shouldStop) {
            finishReason = "stop";
            break;
        }

        tokens.Add(token);
        currentLabel = token;
        remainingTokens--;
    }
    finally {
        GenerativeInterop.gen_free_results(resultPtr);
    }
}
```

**Stop Sequence Logic**:
- Exact token matching: `"."` matches period tokens
- Compound matching: `"end of"` matches across token boundaries
- Early termination prevents over-generation
- Proper `finish_reason` reporting

### Phase 5: Response Formatting

**OpenAI-Compatible Response**:
```csharp
return new CompletionResponse {
    Id = $"gen-hc-{Guid.NewGuid().ToString("N")}",
    Object = "text_completion",
    Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
    Model = request.Model,
    Choices = new[] {
        new CompletionChoice {
            Text = string.Join(" ", tokens),
            Index = 0,
            FinishReason = finishReason
        }
    },
    Usage = new UsageInfo {
        PromptTokens = _tokenizationService.EstimateTokenCount(request.Prompt),
        CompletionTokens = tokens.Count,
        TotalTokens = promptTokens + tokens.Count
    }
};
```

## Error Handling

### Exception Hierarchy

- **`ArgumentException`**: Invalid request parameters
- **`InvalidOperationException`**: Service not initialized
- **`NpgsqlException`**: Database connectivity issues
- **`Exception`**: Unexpected errors (logged, safe message returned)

### Error Response Mapping

| Exception Type | HTTP Status | Error Code | User Message |
|----------------|-------------|------------|--------------|
| ArgumentException | 400 | invalid_parameter | Parameter validation message |
| InvalidOperationException | 503 | model_not_loaded | Service initializing |
| NpgsqlException | 503 | database_unavailable | Database temporarily unavailable |
| Exception | 500 | internal_error | Unexpected error occurred |

## Performance Characteristics

### Latency Breakdown

| Phase | Typical Time | Notes |
|-------|--------------|-------|
| Tokenization | 10-50ms | Database lookups |
| Cache Validation | 1-5ms | In-memory checks |
| Generation Setup | 1ms | Parameter configuration |
| Token Generation | 50-200ms | C++ computation per token |
| Response Formatting | 1-5ms | JSON serialization |

### Memory Usage

- **Per Request**: ~1-5KB for token storage and marshalling
- **Cache Overhead**: ~500MB shared across all requests
- **Connection Pooling**: Future - reduce DB connection overhead

### Scaling Considerations

- **Stateless Design**: Each request independent
- **Horizontal Scaling**: Add instances behind load balancer
- **Cache Sharing**: All instances load same C++ caches
- **Database Load**: Token validation queries per request

## Monitoring & Diagnostics

### Health Check Integration

```csharp
public Task<HealthCheckResult> CheckHealthAsync() {
    var cacheHealthy = _generativeService.IsInitialized;
    var vocabCount = _generativeService.GetCacheStats().VocabCount;

    return HealthCheckResult.Create(
        cacheHealthy ? HealthStatus.Healthy : HealthStatus.Degraded,
        $"Vocab: {vocabCount}, Initialized: {cacheHealthy}"
    );
}
```

### Logging Events

- **Request Start**: Model, prompt length, parameters
- **Tokenization Results**: Valid tokens found, fallbacks used
- **Generation Progress**: Token count, stop sequence hits
- **Performance**: Total latency, token generation rate
- **Errors**: Detailed exception information with context

### Metrics Collected

- **Request Count**: Per model, per endpoint
- **Generation Time**: End-to-end latency distribution
- **Token Counts**: Prompt and completion size distributions
- **Error Rates**: Per error type and endpoint
- **Cache Hit Rates**: Token validation success rates

## Configuration Options

### Generation Parameters

```json
{
  "Hypercube": {
    "Generative": {
      "CacheLoadTimeout": 300,      // Cache loading timeout (seconds)
      "MaxGenerationTokens": 2048,  // Absolute maximum tokens
      "DefaultTemperature": 0.7     // Default sampling temperature
    }
  }
}
```

### C++ Engine Weights

```csharp
// Default scoring weights (configurable)
GenerativeInterop.gen_config_set_weights(
    centroidWeight: 0.4,    // 4D geometric similarity
    pmiWeight: 0.3,         // Co-occurrence statistics
    attentionWeight: 0.2,   // Pre-computed attention
    globalWeight: 0.1       // Frequency priors
);
```

## Testing Strategy

### Unit Tests

```csharp
[TestMethod]
public async Task GenerateCompletionAsync_ValidRequest_ReturnsCompletionResponse()
{
    // Arrange
    var service = new GenerativeService(logger, tokenizer, postgres);
    await service.InitializeAsync();

    var request = new CompletionRequest {
        Model = "hypercube-generative",
        Prompt = "Hello world",
        MaxTokens = 10
    };

    // Act
    var response = await service.GenerateCompletionAsync(request);

    // Assert
    Assert.IsNotNull(response);
    Assert.AreEqual("text_completion", response.Object);
    Assert.AreEqual("hypercube-generative", response.Model);
    Assert.IsTrue(response.Choices.Any());
}
```

### Integration Tests

- **Full Workflow**: HTTP request → Tokenization → Generation → Response
- **Database Dependency**: Real PostgreSQL with test data
- **C++ Interop**: Mock or real DLL calls
- **Performance**: Latency and throughput validation

### Error Scenario Tests

- Service not initialized
- Invalid parameters
- Database disconnection
- C++ engine failures
- Memory allocation errors

## Future Enhancements

### Streaming Support
- **Server-Sent Events**: Real-time token streaming
- **Cancellation Tokens**: Request cancellation support
- **Partial Responses**: Incremental result delivery

### Advanced Generation
- **Context Window**: Multi-token prompt continuation
- **Conversation Memory**: Chat history integration
- **Custom Stop Sequences**: Dynamic termination rules

### Performance Optimizations
- **Batch Generation**: Multiple prompts in single call
- **Cache Prefetching**: Proactive token validation
- **Async Generation**: Non-blocking C++ calls

### Monitoring Enhancements
- **Detailed Metrics**: Per-token generation times
- **Quality Metrics**: Semantic coherence scoring
- **Usage Analytics**: Token consumption patterns

This service provides the critical orchestration layer that transforms OpenAI-compatible requests into hypercube-powered text generation, maintaining both API compatibility and access to advanced semantic capabilities.