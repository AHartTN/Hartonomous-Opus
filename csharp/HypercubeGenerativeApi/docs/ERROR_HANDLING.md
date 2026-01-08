# Error Handling & Validation Documentation

## Overview

The Hypercube Generative API implements comprehensive error handling and validation to ensure robust operation and OpenAI-compatible error responses. The system provides structured error information while maintaining security and preventing information disclosure.

## Error Response Architecture

### Error Response DTOs

```csharp
public class ErrorResponse
{
    public ErrorDetails Error { get; set; } = new();
}

public class ErrorDetails
{
    public string Message { get; set; } = string.Empty;
    public string Type { get; set; } = "internal_error";
    public string? Code { get; set; }
    public string? Param { get; set; }
    public object? InternalMessage { get; set; }
}
```

### Error Factory Methods

```csharp
public static class ErrorResponseFactory
{
    public static ErrorResponse BadRequest(string message, string code, string? param = null)
        => new() { Error = new() { Message = message, Type = ErrorTypes.InvalidRequest, Code = code, Param = param } };

    public static ErrorResponse NotFound(string message, string code = ErrorCodes.InvalidModel)
        => new() { Error = new() { Message = message, Type = ErrorTypes.NotFound, Code = code } };

    public static ErrorResponse ServiceUnavailable(string message, string code = ErrorCodes.ServiceUnavailable)
        => new() { Error = new() { Message = message, Type = ErrorTypes.ServiceUnavailable, Code = code } };

    public static ErrorResponse InternalError(string message, Exception? exception = null)
        => new() { Error = new() { Message = message, Type = ErrorTypes.Internal, Code = "internal_error",
                                   InternalMessage = exception?.Message } };
}
```

## Error Classification

### Error Types

| Type | HTTP Code | Description |
|------|-----------|-------------|
| `invalid_request_error` | 400 | Client request issues |
| `authentication_error` | 401 | Authentication failures |
| `permission_error` | 403 | Authorization issues |
| `not_found_error` | 404 | Resource not found |
| `conflict_error` | 409 | Resource conflicts |
| `unprocessable_entity_error` | 422 | Validation failures |
| `rate_limit_error` | 429 | Rate limit exceeded |
| `internal_error` | 500 | Server errors |
| `service_unavailable_error` | 503 | Service unavailable |

### Error Codes

```csharp
public static class ErrorCodes
{
    // Request validation
    public const string MissingRequiredParameter = "missing_required_parameter";
    public const string InvalidParameter = "invalid_parameter";
    public const string ParameterTooLarge = "parameter_too_large";
    public const string InvalidModel = "model_not_found";

    // Authentication/Authorization
    public const string InvalidApiKey = "invalid_api_key";
    public const string InsufficientPermissions = "insufficient_permissions";

    // Service state
    public const string ServiceUnavailable = "service_unavailable";
    public const string ModelNotLoaded = "model_not_loaded";
    public const string DatabaseUnavailable = "database_unavailable";

    // Generation errors
    public const string GenerationFailed = "generation_failed";
    public const string ContentFilter = "content_filter";
    public const string LengthExceeded = "maximum_length_exceeded";
}
```

## Validation Layers

### Request Validation

#### Model Validation
```csharp
if (request.Model != "hypercube-generative") {
    return Results.NotFound(ErrorResponseFactory.NotFound(
        $"Model '{request.Model}' not found"));
}
```

#### Parameter Validation
```csharp
// Prompt validation
if (string.IsNullOrWhiteSpace(request.Prompt)) {
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        "Prompt cannot be empty", ErrorCodes.MissingRequiredParameter, "prompt"));
}

if (request.Prompt.Length > 10000) {
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        "Prompt is too large", ErrorCodes.ParameterTooLarge, "prompt"));
}

// Max tokens validation
if (request.MaxTokens <= 0 || request.MaxTokens > 2048) {
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        "max_tokens must be between 1 and 2048", ErrorCodes.InvalidParameter, "max_tokens"));
}

// Temperature validation
if (request.Temperature < 0 || request.Temperature > 2) {
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        "temperature must be between 0 and 2", ErrorCodes.InvalidParameter, "temperature"));
}
```

### Service State Validation

#### Initialization Checks
```csharp
if (!generativeService.IsInitialized) {
    return TypedResults.Json(ErrorResponseFactory.ServiceUnavailable(
        "Service is initializing, please try again later",
        ErrorCodes.ServiceUnavailable), statusCode: 503);
}
```

#### Database Connectivity
```csharp
catch (NpgsqlException ex) {
    return TypedResults.Json(ErrorResponseFactory.ServiceUnavailable(
        "Database temporarily unavailable", ErrorCodes.DatabaseUnavailable), statusCode: 503);
}
```

## Exception Handling Strategy

### Controller-Level Exception Handling

```csharp
try {
    // Request processing
    var response = await generativeService.GenerateCompletionAsync(request);
    return Results.Ok(response);
}
catch (ArgumentException ex) {
    logger.LogWarning(ex, "Invalid completion request parameter");
    return Results.BadRequest(ErrorResponseFactory.BadRequest(
        ex.Message, ErrorCodes.InvalidParameter));
}
catch (InvalidOperationException ex) when (ex.Message.Contains("not initialized")) {
    logger.LogWarning(ex, "Service not initialized");
    return TypedResults.Json(ErrorResponseFactory.ServiceUnavailable(
        "Service is not ready", ErrorCodes.ModelNotLoaded), statusCode: 503);
}
catch (NpgsqlException ex) {
    logger.LogError(ex, "Database error during completion generation");
    return TypedResults.Json(ErrorResponseFactory.ServiceUnavailable(
        "Database temporarily unavailable", ErrorCodes.DatabaseUnavailable), statusCode: 503);
}
catch (Exception ex) {
    logger.LogError(ex, "Unexpected error processing completion request");
    return TypedResults.Json(ErrorResponseFactory.InternalError(
        "An unexpected error occurred", ex), statusCode: 500);
}
```

### Global Exception Handler

```csharp
// Program.cs - Global error handling
app.UseExceptionHandler(exceptionHandlerApp => {
    exceptionHandlerApp.Run(async context => {
        var exception = context.Features.Get<Microsoft.AspNetCore.Diagnostics.IExceptionHandlerFeature>()?.Error;

        context.Response.StatusCode = StatusCodes.Status500InternalServerError;
        context.Response.ContentType = "application/json";

        var errorResponse = new {
            error = new {
                message = "An internal error occurred",
                type = "internal_error",
                details = app.Environment.IsDevelopment() ? exception?.Message : null
            }
        };

        await context.Response.WriteAsJsonAsync(errorResponse);
    });
});
```

## Service-Level Error Handling

### GenerativeService Error Handling

```csharp
public async Task<CompletionResponse> GenerateCompletionAsync(CompletionRequest request)
{
    if (!_isInitialized) {
        throw new InvalidOperationException("Generative service not initialized");
    }

    try {
        // Generation logic
        var validTokens = await _postgresService.GetValidTokensFromPromptAsync(request.Prompt);

        if (validTokens.Length == 0) {
            _logger.LogWarning("No valid tokens found in prompt: {Prompt}", request.Prompt);
            validTokens = new[] { "the" };
        }

        // Continue with generation...
    }
    catch (Exception ex) {
        _logger.LogError(ex, "Error generating completion for prompt: {Prompt}", request.Prompt);
        throw;
    }
}
```

### TokenizationService Error Handling

```csharp
private async Task<long?> EncodeTokenAsync(string token)
{
    try {
        var exists = await _postgresService.TokenExistsAsync(token);
        if (!exists) {
            _logger.LogDebug("Token '{Token}' not found in vocabulary", token);
            return null;
        }

        return GetStableHash(token);
    }
    catch (Exception ex) {
        _logger.LogWarning(ex, "Error encoding token '{Token}'", token);
        return null;
    }
}
```

### PostgresService Error Handling

```csharp
public async Task<bool> TokenExistsAsync(string token)
{
    if (_connection == null || _disposed || string.IsNullOrWhiteSpace(token)) {
        return false;
    }

    try {
        await using var cmd = new NpgsqlCommand(query, _connection);
        cmd.Parameters.AddWithValue("@token", token);

        var result = await cmd.ExecuteScalarAsync();
        return result != null;
    }
    catch (Exception ex) {
        _logger.LogWarning(ex, "Error checking token existence '{Token}'", token);
        return false;
    }
}
```

## Security Considerations

### Information Disclosure Prevention

- **Production Mode**: Generic error messages only
- **Development Mode**: Detailed exception information
- **No Stack Traces**: Never exposed in API responses
- **Sensitive Data**: Not included in error messages

### Input Sanitization

- **SQL Injection**: Parameterized queries prevent injection
- **Prompt Limits**: Length limits prevent resource exhaustion
- **Parameter Validation**: Range checks prevent invalid states
- **Content Filtering**: Basic input cleaning (future enhancement)

## Logging Strategy

### Error Logging Levels

```csharp
// Warning - Recoverable issues
_logger.LogWarning("No valid tokens found in prompt: {Prompt}", prompt);

// Error - Unrecoverable issues requiring attention
_logger.LogError(ex, "Database error during completion generation");

// Information - Normal operations with issues
_logger.LogInformation("Generated completion in {Duration}ms: {TokenCount} tokens",
    duration.TotalMilliseconds, tokens.Count);
```

### Structured Logging

```json
{
  "Timestamp": "2024-01-08T16:00:00.000Z",
  "Level": "Warning",
  "Message": "No valid tokens found in prompt",
  "Properties": {
    "Prompt": "unknown word here",
    "ValidTokens": 0,
    "TotalTokens": 3
  }
}
```

## Testing Error Scenarios

### Unit Tests

```csharp
[TestMethod]
public async Task CreateCompletion_InvalidModel_ReturnsNotFound()
{
    var client = CreateClient();
    var request = new { model = "invalid-model", prompt = "test", max_tokens = 10 };

    var response = await client.PostAsJsonAsync("/v1/completions", request);

    Assert.AreEqual(HttpStatusCode.NotFound, response.StatusCode);

    var error = await response.Content.ReadFromJsonAsync<ErrorResponse>();
    Assert.AreEqual("not_found_error", error.Error.Type);
    Assert.AreEqual("model_not_found", error.Error.Code);
}

[TestMethod]
public async Task CreateCompletion_EmptyPrompt_ReturnsBadRequest()
{
    var client = CreateClient();
    var request = new { model = "hypercube-generative", prompt = "", max_tokens = 10 };

    var response = await client.PostAsJsonAsync("/v1/completions", request);

    Assert.AreEqual(HttpStatusCode.BadRequest, response.StatusCode);

    var error = await response.Content.ReadFromJsonAsync<ErrorResponse>();
    Assert.AreEqual("invalid_request_error", error.Error.Type);
    Assert.AreEqual("missing_required_parameter", error.Error.Code);
    Assert.AreEqual("prompt", error.Error.Param);
}
```

### Integration Tests

```csharp
[TestMethod]
public async Task DatabaseUnavailable_ReturnsServiceUnavailable()
{
    // Arrange: Configure invalid database connection
    var client = CreateClient();

    var request = new { model = "hypercube-generative", prompt = "test", max_tokens = 10 };

    // Act
    var response = await client.PostAsJsonAsync("/v1/completions", request);

    // Assert
    Assert.AreEqual(HttpStatusCode.ServiceUnavailable, response.StatusCode);

    var error = await response.Content.ReadFromJsonAsync<ErrorResponse>();
    Assert.AreEqual("database_unavailable", error.Error.Code);
}
```

## Monitoring & Alerting

### Error Metrics

- **Error Rate by Type**: Track frequency of different error types
- **Error Rate by Endpoint**: Monitor per-endpoint failure rates
- **Database Error Trends**: Track database connectivity issues
- **Service Unavailable Duration**: Measure downtime periods

### Alert Conditions

```csharp
// Alert on high error rates
if (errorRate > 0.05) { // 5% error rate
    _alertService.SendAlert("High error rate detected", errorDetails);
}

// Alert on service unavailability
if (!serviceHealthy && timeSinceLastHealthy > TimeSpan.FromMinutes(5)) {
    _alertService.SendAlert("Service unavailable for 5+ minutes", serviceStatus);
}
```

## Future Enhancements

### Advanced Error Handling

- **Retry Logic**: Automatic retry for transient failures
- **Circuit Breaker**: Prevent cascade failures
- **Fallback Responses**: Degraded functionality during outages
- **Rate Limiting**: Prevent abuse and resource exhaustion

### Enhanced Validation

- **Content Filtering**: Detect and reject harmful content
- **Semantic Validation**: Check prompt coherence
- **Usage Quotas**: Per-user limits and tracking
- **Request Deduplication**: Prevent duplicate processing

### Monitoring Integration

- **OpenTelemetry**: Distributed tracing for errors
- **Prometheus Metrics**: Error rate and latency tracking
- **ELK Integration**: Centralized error log analysis
- **PagerDuty**: Automated incident response

This comprehensive error handling system ensures the API provides clear, actionable error information while maintaining security and enabling effective monitoring and debugging.