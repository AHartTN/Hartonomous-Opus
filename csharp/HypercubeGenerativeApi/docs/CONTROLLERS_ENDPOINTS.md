# Controllers & Endpoints Documentation

## Overview

The Controllers layer implements OpenAI-compatible REST endpoints, handling HTTP requests, validation, and response formatting. This layer provides the external API interface that any OpenAI client can use seamlessly.

## Architecture

### Minimal API Design

```csharp
// Program.cs - Route Registration
app.MapPost("/v1/completions", CompletionsEndpoints.CreateCompletion);
app.MapGet("/v1/models", ModelsEndpoints.ListModels);
app.MapHealthChecks("/health");
```

### Controller Organization

- **CompletionsEndpoints**: Text completion generation
- **ModelsEndpoints**: Model information and listing
- **Health Checks**: System status monitoring

## CompletionsEndpoints

### CreateCompletion Method

```csharp
public static async Task<IResult> CreateCompletion(
    CompletionRequest request,
    GenerativeService generativeService,
    ILogger<GenerativeService> logger)
```

#### Request Flow

1. **Parameter Logging**: Record incoming request details
2. **Model Validation**: Ensure `hypercube-generative` model requested
3. **Prompt Validation**: Check for empty/null prompts and length limits
4. **Parameter Validation**: Validate temperature, max_tokens ranges
5. **Service Readiness**: Verify generative engine is initialized
6. **Generation**: Call GenerativeService with validated parameters
7. **Response**: Return OpenAI-compatible JSON

#### Validation Rules

**Model Validation**:
```csharp
if (request.Model != "hypercube-generative") {
    return Results.NotFound(ErrorResponseFactory.NotFound(
        $"Model '{request.Model}' not found"));
}
```

**Prompt Validation**:
```csharp
if (string.IsNullOrWhiteSpace(request.Prompt)) {
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        "Prompt cannot be empty", ErrorCodes.MissingRequiredParameter, "prompt"));
}

if (request.Prompt.Length > 10000) {
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        "Prompt is too large", ErrorCodes.ParameterTooLarge, "prompt"));
}
```

**Parameter Validation**:
```csharp
if (request.MaxTokens <= 0 || request.MaxTokens > 2048) {
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        "max_tokens must be between 1 and 2048", ErrorCodes.InvalidParameter, "max_tokens"));
}

if (request.Temperature < 0 || request.Temperature > 2) {
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        "temperature must be between 0 and 2", ErrorCodes.InvalidParameter, "temperature"));
}
```

#### Error Handling

**Structured Error Responses**:
```csharp
catch (ArgumentException ex) {
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        ex.Message, ErrorCodes.InvalidParameter));
}

catch (InvalidOperationException ex) when (ex.Message.Contains("not initialized")) {
    return TypedResults.Json(ErrorResponseFactory.ServiceUnavailable(
        "Service is not ready", ErrorCodes.ModelNotLoaded), statusCode: 503);
}

catch (NpgsqlException ex) {
    return TypedResults.Json(ErrorResponseFactory.ServiceUnavailable(
        "Database temporarily unavailable", ErrorCodes.DatabaseUnavailable), statusCode: 503);
}
```

**Error Response Format**:
```json
{
  "error": {
    "message": "max_tokens must be between 1 and 2048",
    "type": "invalid_request_error",
    "code": "invalid_parameter",
    "param": "max_tokens"
  }
}
```

### OpenAI Compatibility

**Supported Parameters**:
- `model`: Model identifier (validated)
- `prompt`: Input text (required, validated)
- `max_tokens`: Generation length limit (1-2048)
- `temperature`: Sampling temperature (0.0-2.0)
- `stop`: Stop sequences (implemented)
- `stream`: Streaming support (not yet implemented)

**Response Format**:
```json
{
  "id": "gen-hc-abc123",
  "object": "text_completion",
  "created": 1677652288,
  "model": "hypercube-generative",
  "choices": [{
    "text": " generated text...",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 10,
    "total_tokens": 15
  }
}
```

## ModelsEndpoints

### ListModels Method

```csharp
public static IResult ListModels()
```

#### Implementation

```csharp
return Results.Ok(new ModelsResponse {
    Data = new[] {
        new ModelInfo {
            Id = "hypercube-generative",
            Created = 1677652288,  // Approximate creation timestamp
            OwnedBy = "hartonomous-opus"
        }
    }
});
```

#### Response Format

```json
{
  "object": "list",
  "data": [{
    "id": "hypercube-generative",
    "object": "model",
    "created": 1677652288,
    "owned_by": "hartonomous-opus"
  }]
}
```

## Health Checks

### GenerativeHealthCheck

```csharp
public class GenerativeHealthCheck : IHealthCheck
{
    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, CancellationToken cancellationToken)
    {
        var cacheHealthy = _generativeService.IsInitialized;
        var dbHealthy = await _postgresService.CheckConnectionAsync();
        var stats = await _postgresService.GetDatabaseStatsAsync();

        var overallHealthy = cacheHealthy && dbHealthy;
        var status = overallHealthy ? HealthStatus.Healthy : HealthStatus.Degraded;

        return new HealthCheckResult(status, "Hypercube generative health check",
            data: new Dictionary<string, object> {
                ["cache_loaded"] = cacheHealthy,
                ["database_connected"] = dbHealthy,
                ["vocab_count"] = stats.GetValueOrDefault("compositions", 0L),
                ["atoms_count"] = stats.GetValueOrDefault("atoms", 0L)
            });
    }
}
```

#### Health Check Endpoints

- **`/health`**: Overall system health
- **`/health/ready`**: Readiness for traffic
- **`/health/live`**: Liveness probe

#### Health Response Example

```json
{
  "status": "Healthy",
  "description": "Hypercube generative health check",
  "data": {
    "cache_loaded": true,
    "database_connected": true,
    "vocab_count": 50000,
    "atoms_count": 1100000
  }
}
```

## Request Processing Pipeline

### HTTP Request Lifecycle

1. **Routing**: Minimal API routes to appropriate endpoint method
2. **Model Binding**: JSON deserialization to request DTOs
3. **Dependency Injection**: Services injected via DI container
4. **Validation**: Business logic validation (model, parameters, service state)
5. **Processing**: Call generative service with validated parameters
6. **Response**: Format OpenAI-compatible JSON response
7. **Error Handling**: Structured error responses on failures

### Parameter Mapping

| OpenAI Parameter | Validation | Mapping |
|------------------|------------|---------|
| `model` | Must be "hypercube-generative" | Passed to response |
| `prompt` | Required, <10K chars | Tokenized and validated |
| `max_tokens` | 1-2048 | Generation length limit |
| `temperature` | 0.0-2.0 | Sampling randomness |
| `stop` | String array | Early termination sequences |
| `stream` | Boolean | Future: real-time streaming |

## Security Considerations

### Input Validation

- **SQL Injection Prevention**: Parameterized queries in PostgresService
- **Prompt Length Limits**: Prevent excessive resource usage
- **Parameter Range Checks**: Prevent invalid C++ engine states
- **Content Sanitization**: Basic input cleaning

### Error Information Disclosure

- **Production Mode**: Generic error messages
- **Development Mode**: Detailed exception information
- **Logging**: Sensitive data not logged
- **Stack Traces**: Not exposed in API responses

## Performance Monitoring

### Request Metrics

- **Response Times**: End-to-end latency tracking
- **Error Rates**: Per endpoint failure percentages
- **Usage Patterns**: Token consumption analytics
- **Concurrency**: Active request monitoring

### Logging Integration

```csharp
// Request logging
logger.LogInformation("Received completion request for model {Model}", request.Model);

// Error logging
logger.LogError(ex, "Error processing completion request");

// Performance logging
logger.LogInformation("Generated completion in {Duration}ms: {TokenCount} tokens",
    duration.TotalMilliseconds, tokens.Count);
```

## Testing Strategy

### Endpoint Tests

```csharp
[TestClass]
public class CompletionsEndpointTests : WebApplicationFactory<Program>
{
    [TestMethod]
    public async Task CreateCompletion_ValidRequest_ReturnsSuccess()
    {
        var client = CreateClient();
        var request = new { model = "hypercube-generative", prompt = "test", max_tokens = 10 };

        var response = await client.PostAsJsonAsync("/v1/completions", request);

        Assert.AreEqual(HttpStatusCode.OK, response.StatusCode);
        var result = await response.Content.ReadFromJsonAsync<CompletionResponse>();
        Assert.IsNotNull(result);
    }

    [TestMethod]
    public async Task CreateCompletion_InvalidModel_ReturnsNotFound()
    {
        var client = CreateClient();
        var request = new { model = "invalid", prompt = "test", max_tokens = 10 };

        var response = await client.PostAsJsonAsync("/v1/completions", request);

        Assert.AreEqual(HttpStatusCode.NotFound, response.StatusCode);
    }
}
```

### Error Response Tests

```csharp
[TestMethod]
public async Task CreateCompletion_EmptyPrompt_ReturnsBadRequest()
{
    var client = CreateClient();
    var request = new { model = "hypercube-generative", prompt = "", max_tokens = 10 };

    var response = await client.PostAsJsonAsync("/v1/completions", request);

    Assert.AreEqual(HttpStatusCode.BadRequest, response.StatusCode);

    var error = await response.Content.ReadFromJsonAsync<ErrorResponse>();
    Assert.AreEqual("missing_required_parameter", error.Error.Code);
    Assert.AreEqual("prompt", error.Error.Param);
}
```

## Future Enhancements

### Streaming Support

```csharp
// Future implementation
app.MapPost("/v1/completions", async (CompletionRequest request) => {
    if (request.Stream) {
        return StreamingResponse(request);
    } else {
        return SynchronousResponse(request);
    }
});
```

### Chat Completions

```csharp
// Future endpoint
app.MapPost("/v1/chat/completions", ChatEndpoints.CreateChatCompletion);
```

### Advanced Validation

- **Rate Limiting**: Request throttling per client
- **Authentication**: API key validation
- **Content Filtering**: Harmful content detection
- **Usage Quotas**: Per-user limits

### Monitoring Integration

- **OpenTelemetry**: Distributed tracing
- **Prometheus Metrics**: Performance monitoring
- **Structured Logging**: JSON log format
- **Alert Integration**: Error rate alerting

## Configuration

### Endpoint Configuration

```json
{
  "ApiSettings": {
    "MaxPromptLength": 10000,
    "MaxTokensLimit": 2048,
    "DefaultTemperature": 0.7,
    "RequestTimeoutSeconds": 30
  }
}
```

### CORS Configuration

```csharp
// Future: Configure CORS for web clients
builder.Services.AddCors(options => {
    options.AddPolicy("AllowOpenAI", policy => {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});
```

This controllers layer provides a robust, OpenAI-compatible API interface that handles validation, error management, and response formatting while orchestrating the underlying hypercube generation capabilities.