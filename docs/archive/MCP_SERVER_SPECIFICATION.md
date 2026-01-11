# MCP Server Implementation Specification

## Overview
This document provides detailed technical specifications for implementing an MCP server within the Hartonomous-Opus hypercube API. The server will expose semantic operations as MCP tools and resources, enabling external MCP clients to leverage hypercube capabilities.

## Architecture Design

### Server Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │────│   MCP Server     │────│  Business Logic │
│                 │    │  (HTTP/Stdio)   │    │  (Services)     │
│ - Tool Calls    │    │                  │    │                 │
│ - Resource Req  │    │ - Tool Dispatch  │    │ - Semantic Ops  │
│ - Subscriptions │    │ - Resource Serve │    │ - Generation    │
└─────────────────┘    │ - Auth/Rate Lim  │    │ - Ingestion     │
                       └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  PostgreSQL     │
                       │  Hypercube DB   │
                       └─────────────────┘
```

### Transport Options

#### HTTP Transport (Primary)
- **Protocol**: HTTP/1.1 with JSON-RPC 2.0 over POST
- **Endpoint**: `/mcp` (configurable)
- **Authentication**: Bearer token in Authorization header
- **TLS**: Required for production
- **CORS**: Configurable for cross-origin access

#### Stdio Transport (Secondary)
- **Use Case**: Local CLI tools, background processes
- **Implementation**: Process-based communication
- **Security**: OS-level process isolation
- **Deployment**: Not suitable for network access

## Tool Specifications

### 1. Semantic Search Tool

#### Interface Definition
```csharp
[McpServerTool]
[Description("Perform semantic similarity search using hypercube relationships")]
public async Task<SemanticSearchResult> SemanticSearch(
    [Description("Search query text")] string query,
    [Description("Maximum number of results")] int limit = 10,
    [Description("Similarity threshold (0.0-1.0)")] double threshold = 0.1,
    [Description("Search domain filter")] string? domain = null)
```

#### Input Validation Rules
```csharp
private readonly ValidationRule[] _semanticSearchRules = new[]
{
    new ValidationRule
    {
        Parameter = "query",
        Required = true,
        Type = typeof(string),
        MinLength = 1,
        MaxLength = 1000,
        Pattern = @"^[\w\s\p{P}]+$" // Word characters, spaces, punctuation
    },
    new ValidationRule
    {
        Parameter = "limit",
        Required = false,
        Type = typeof(int),
        MinValue = 1,
        MaxValue = 100,
        DefaultValue = 10
    },
    new ValidationRule
    {
        Parameter = "threshold",
        Required = false,
        Type = typeof(double),
        MinValue = 0.0,
        MaxValue = 1.0,
        DefaultValue = 0.1
    }
};
```

#### Implementation Logic
```csharp
public async Task<SemanticSearchResult> SemanticSearch(
    string query, int limit, double threshold, string? domain)
{
    // Input validation
    var validation = _validator.ValidateToolCall("SemanticSearch",
        new Dictionary<string, object>
        {
            ["query"] = query,
            ["limit"] = limit,
            ["threshold"] = threshold,
            ["domain"] = domain
        });

    if (!validation.IsValid)
        throw new McpException($"Validation failed: {string.Join(", ", validation.Errors)}");

    // Authorization check
    await _authorization.CheckPermissionAsync(_currentUser, "semantic:search");

    // Rate limiting
    if (!await _rateLimiter.CheckRateLimitAsync(_currentUser.Id, "SemanticSearch"))
        throw new McpException("Rate limit exceeded");

    // Business logic execution
    var semanticService = _serviceProvider.GetRequiredService<SemanticQueryService>();
    var results = await semanticService.QuerySemantic(new SemanticQueryRequest
    {
        Query = query,
        Limit = limit,
        Threshold = threshold,
        Domain = domain
    });

    // Response mapping
    return new SemanticSearchResult
    {
        Query = query,
        Results = results.Select(r => new SemanticResult
        {
            Term = r.Term,
            Similarity = r.Similarity,
            Relationships = r.Relationships?.ToArray() ?? Array.Empty<string>(),
            Metadata = new ResultMetadata
            {
                Confidence = r.Confidence,
                Source = r.Source,
                Timestamp = DateTime.UtcNow
            }
        }).ToArray(),
        TotalFound = results.Length,
        ExecutionTime = DateTime.UtcNow - startTime
    };
}
```

#### Response Schema
```csharp
public class SemanticSearchResult
{
    [JsonPropertyName("query")]
    public string Query { get; set; }

    [JsonPropertyName("results")]
    public SemanticResult[] Results { get; set; }

    [JsonPropertyName("total_found")]
    public int TotalFound { get; set; }

    [JsonPropertyName("execution_time")]
    public TimeSpan ExecutionTime { get; set; }
}

public class SemanticResult
{
    [JsonPropertyName("term")]
    public string Term { get; set; }

    [JsonPropertyName("similarity")]
    public double Similarity { get; set; }

    [JsonPropertyName("relationships")]
    public string[] Relationships { get; set; }

    [JsonPropertyName("metadata")]
    public ResultMetadata Metadata { get; set; }
}

public class ResultMetadata
{
    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("source")]
    public string Source { get; set; }

    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; }
}
```

#### Error Handling
```csharp
try
{
    // Tool implementation
}
catch (ValidationException ex)
{
    _logger.LogWarning(ex, "Validation failed for SemanticSearch");
    throw new McpException(McpErrorCodes.InvalidParams, ex.Message);
}
catch (AuthorizationException ex)
{
    _logger.LogWarning(ex, "Authorization failed for SemanticSearch");
    throw new McpException(McpErrorCodes.Unauthorized, "Insufficient permissions");
}
catch (RateLimitException ex)
{
    _logger.LogWarning(ex, "Rate limit exceeded for SemanticSearch");
    throw new McpException(McpErrorCodes.ServerError, "Rate limit exceeded");
}
catch (Exception ex)
{
    _logger.LogError(ex, "Unexpected error in SemanticSearch");
    throw new McpException(McpErrorCodes.InternalError, "Internal server error");
}
```

### 2. Analogy Generation Tool

#### Interface Definition
```csharp
[McpServerTool]
[Description("Generate analogies using geometric relationships in 4D space")]
public async Task<AnalogyResult[]> FindAnalogies(
    [Description("First term in analogy")] string termA,
    [Description("Second term in analogy")] string termB,
    [Description("Third term in analogy")] string termC,
    [Description("Maximum analogies to return")] int limit = 5,
    [Description("Minimum confidence threshold")] double minConfidence = 0.3)
```

#### Implementation Logic
```csharp
public async Task<AnalogyResult[]> FindAnalogies(
    string termA, string termB, string termC, int limit, double minConfidence)
{
    var analogyService = _serviceProvider.GetRequiredService<SemanticQueryService>();
    var results = await analogyService.QueryAnalogies(new AnalogyRequest
    {
        TermA = termA,
        TermB = termB,
        TermC = termC,
        Limit = limit,
        MinConfidence = minConfidence
    });

    return results.Select(r => new AnalogyResult
    {
        Term = r.Result,
        Analogy = $"{termA} : {termB} :: {termC} : {r.Result}",
        Confidence = r.Confidence,
        Explanation = r.Explanation,
        GeometricDistance = r.GeometricDistance
    }).ToArray();
}
```

### 3. Geometric Neighbors Tool

#### Interface Definition
```csharp
[McpServerTool]
[Description("Find geometric neighbors in 4D hypersphere space")]
public async Task<GeometricNeighborResult[]> GeometricNeighbors(
    [Description("Entity identifier")] string entityId,
    [Description("Search radius in 4D space")] double radius = 1.0,
    [Description("Maximum neighbors to return")] int limit = 20,
    [Description("Coordinate system")] string coordinateSystem = "spherical")
```

#### Implementation Logic
```csharp
public async Task<GeometricNeighborResult[]> GeometricNeighbors(
    string entityId, double radius, int limit, string coordinateSystem)
{
    var geometricService = _serviceProvider.GetRequiredService<GeometricController>();
    var results = await geometricService.FindNeighbors(new GeometricQuery
    {
        EntityId = entityId,
        Radius = radius,
        Limit = limit,
        CoordinateSystem = Enum.Parse<CoordinateSystem>(coordinateSystem)
    });

    return results.Select(r => new GeometricNeighborResult
    {
        EntityId = r.EntityId,
        Distance = r.Distance,
        Coordinates = r.Coordinates,
        CoordinateSystem = coordinateSystem
    }).ToArray();
}
```

### 4. Text Generation Tool

#### Interface Definition
```csharp
[McpServerTool]
[Description("Generate text using hypercube semantic walking")]
public async Task<GenerationResult> GenerateText(
    [Description("Prompt text to continue")] string prompt,
    [Description("Maximum tokens to generate")] int maxTokens = 100,
    [Description("Creativity parameter")] double temperature = 0.7,
    [Description("Stop sequences")] string[] stopSequences = null,
    [Description("Random seed for reproducibility")] int? seed = null)
```

#### Content Filtering Rules
```csharp
private readonly ContentFilter[] _generationFilters = new[]
{
    new ContentFilter { Pattern = @"profanity|offensive", Action = FilterAction.Reject },
    new ContentFilter { Pattern = @"personal.*information", Action = FilterAction.Redact },
    new ContentFilter { MaxLength = 10000, Action = FilterAction.Truncate }
};
```

#### Implementation Logic
```csharp
public async Task<GenerationResult> GenerateText(
    string prompt, int maxTokens, double temperature,
    string[] stopSequences, int? seed)
{
    // Apply content filtering
    var filteredPrompt = await _contentFilter.FilterAsync(prompt);
    if (filteredPrompt.Rejected)
        throw new McpException("Content filter rejected prompt");

    var generativeService = _serviceProvider.GetRequiredService<GenerativeService>();
    var result = await generativeService.GenerateCompletion(new CompletionRequest
    {
        Prompt = filteredPrompt.Text,
        MaxTokens = Math.Min(maxTokens, _maxGenerationTokens),
        Temperature = Math.Clamp(temperature, 0.1, 2.0),
        StopSequences = stopSequences,
        Seed = seed
    });

    // Post-generation filtering
    var filteredResult = await _contentFilter.FilterAsync(result.Text);

    return new GenerationResult
    {
        Text = filteredResult.Text,
        TokensGenerated = result.TokensGenerated,
        FinishReason = result.FinishReason.ToString(),
        Seed = seed,
        Filtered = filteredResult.Modified
    };
}
```

### 5. Content Ingestion Tool

#### Interface Definition
```csharp
[McpServerTool]
[Description("Ingest content into the hypercube semantic database")]
public async Task<IngestionResult> IngestContent(
    [Description("Content to ingest")] string content,
    [Description("Content type (text, code, json)")] string contentType = "text",
    [Description("Content source identifier")] string source = null,
    [Description("Processing priority")] string priority = "normal",
    [Description("Metadata key-value pairs")] Dictionary<string, string> metadata = null)
```

#### Content Type Validation
```csharp
private readonly Dictionary<string, ContentTypeSpec> _contentTypeSpecs = new()
{
    ["text"] = new ContentTypeSpec
    {
        MaxSize = 1000000, // 1MB
        AllowedEncodings = new[] { "utf-8", "ascii" },
        Sanitizers = new[] { "html-escape", "normalize-whitespace" }
    },
    ["code"] = new ContentTypeSpec
    {
        MaxSize = 500000, // 500KB
        AllowedEncodings = new[] { "utf-8" },
        Sanitizers = new[] { "syntax-highlight", "remove-comments" }
    },
    ["json"] = new ContentTypeSpec
    {
        MaxSize = 100000, // 100KB
        AllowedEncodings = new[] { "utf-8" },
        Validators = new[] { "json-schema" }
    }
};
```

#### Implementation Logic
```csharp
public async Task<IngestionResult> IngestContent(
    string content, string contentType, string source,
    string priority, Dictionary<string, string> metadata)
{
    // Content type validation
    if (!_contentTypeSpecs.TryGetValue(contentType, out var spec))
        throw new McpException($"Unsupported content type: {contentType}");

    if (content.Length > spec.MaxSize)
        throw new McpException($"Content exceeds maximum size: {spec.MaxSize}");

    // Content sanitization
    var sanitizedContent = await _contentProcessor.SanitizeAsync(content, spec.Sanitizers);

    // Priority handling
    var processingPriority = Enum.Parse<ProcessingPriority>(priority, true);
    var queueName = GetQueueName(processingPriority);

    var ingestionService = _serviceProvider.GetRequiredService<IngestionService>();
    var result = await ingestionService.IngestDocument(new IngestionRequest
    {
        Content = sanitizedContent,
        ContentType = contentType,
        Source = source,
        Priority = processingPriority,
        Metadata = metadata ?? new Dictionary<string, string>()
    });

    return new IngestionResult
    {
        Success = result.Success,
        CompositionId = result.CompositionId,
        AtomsProcessed = result.AtomsProcessed,
        ErrorMessage = result.ErrorMessage,
        ProcessingTime = result.ProcessingTime,
        QueuePosition = await GetQueuePositionAsync(queueName, result.CompositionId)
    };
}
```

## Resource Specifications

### Semantic Knowledge Resource

#### Interface Definition
```csharp
[McpServerResource("hypercube://knowledge/{topic}")]
[Description("Access semantic knowledge about topics")]
public async Task<ResourceContent> GetKnowledgeResource(
    [FromRoute] string topic,
    [FromQuery] int depth = 2,
    [FromQuery] string format = "json")
```

#### Implementation Logic
```csharp
public async Task<ResourceContent> GetKnowledgeResource(string topic, int depth, string format)
{
    var knowledgeService = _serviceProvider.GetRequiredService<KnowledgeService>();
    var knowledge = await knowledgeService.GetTopicKnowledgeAsync(topic, depth);

    string content;
    string mimeType;

    switch (format.ToLower())
    {
        case "json":
            content = JsonSerializer.Serialize(knowledge, _jsonOptions);
            mimeType = "application/json";
            break;
        case "xml":
            content = SerializeToXml(knowledge);
            mimeType = "application/xml";
            break;
        default:
            throw new McpException($"Unsupported format: {format}");
    }

    return new ResourceContent
    {
        Uri = $"hypercube://knowledge/{topic}?depth={depth}&format={format}",
        MimeType = mimeType,
        Text = content
    };
}
```

### Geometric Space Resource

#### Interface Definition
```csharp
[McpServerResource("hypercube://geometry/{entity}")]
[Description("Access geometric coordinates and relationships")]
public async Task<ResourceContent> GetGeometricResource(
    [FromRoute] string entity,
    [FromQuery] string coordinateSystem = "spherical",
    [FromQuery] bool includeNeighbors = false)
```

#### Implementation Logic
```csharp
public async Task<ResourceContent> GetGeometricResource(
    string entity, string coordinateSystem, bool includeNeighbors)
{
    var geometricService = _serviceProvider.GetRequiredService<GeometricController>();
    var geometry = await geometricService.GetEntityGeometryAsync(entity, coordinateSystem);

    if (includeNeighbors)
    {
        geometry.Neighbors = await geometricService.FindNeighborsAsync(
            entity, radius: 2.0, limit: 10);
    }

    return new ResourceContent
    {
        Uri = $"hypercube://geometry/{entity}?system={coordinateSystem}&neighbors={includeNeighbors}",
        MimeType = "application/json",
        Text = JsonSerializer.Serialize(geometry)
    };
}
```

## Middleware and Infrastructure

### Authentication Middleware
```csharp
public class McpAuthenticationMiddleware
{
    private readonly IJwtTokenValidator _tokenValidator;
    private readonly IUserRepository _userRepository;

    public async Task InvokeAsync(McpRequestContext context)
    {
        var authHeader = context.Request.Headers.GetValueOrDefault("Authorization");
        if (string.IsNullOrEmpty(authHeader) || !authHeader.StartsWith("Bearer "))
        {
            context.Response = CreateErrorResponse(McpErrorCodes.Unauthorized, "Missing or invalid token");
            return;
        }

        var token = authHeader.Substring("Bearer ".Length);
        var validationResult = await _tokenValidator.ValidateAsync(token);

        if (!validationResult.IsValid)
        {
            context.Response = CreateErrorResponse(McpErrorCodes.Unauthorized, validationResult.Error);
            return;
        }

        context.User = await _userRepository.GetByIdAsync(validationResult.UserId);
        await _next(context);
    }
}
```

### Authorization Middleware
```csharp
public class McpAuthorizationMiddleware
{
    private readonly IAuthorizationService _authService;

    public async Task InvokeAsync(McpRequestContext context)
    {
        var toolAttribute = context.ToolMethod.GetCustomAttribute<McpServerToolAttribute>();
        if (toolAttribute?.RequiresPermission != null)
        {
            var hasPermission = await _authService.HasPermissionAsync(
                context.User, toolAttribute.RequiresPermission);

            if (!hasPermission)
            {
                context.Response = CreateErrorResponse(
                    McpErrorCodes.Unauthorized, "Insufficient permissions");
                return;
            }
        }

        await _next(context);
    }
}
```

### Rate Limiting Middleware
```csharp
public class McpRateLimitMiddleware
{
    private readonly IRateLimitService _rateLimitService;

    public async Task InvokeAsync(McpRequestContext context)
    {
        var userId = context.User?.Id ?? "anonymous";
        var operation = context.Request.Method;

        if (!await _rateLimitService.CheckLimitAsync(userId, operation))
        {
            context.Response = CreateErrorResponse(
                McpErrorCodes.ServerError, "Rate limit exceeded");
            return;
        }

        await _next(context);
    }
}
```

### Logging Middleware
```csharp
public class McpLoggingMiddleware
{
    private readonly ILogger _logger;
    private readonly IMetricsService _metrics;

    public async Task InvokeAsync(McpRequestContext context)
    {
        var startTime = DateTime.UtcNow;
        var requestId = Guid.NewGuid().ToString();

        _logger.LogInformation(
            "MCP Request: {RequestId}, User: {UserId}, Tool: {ToolName}",
            requestId, context.User?.Id, context.Request.Method);

        try
        {
            await _next(context);

            var duration = DateTime.UtcNow - startTime;
            _metrics.RecordRequest(duration, context.Response.Success);

            _logger.LogInformation(
                "MCP Response: {RequestId}, Duration: {Duration}ms, Success: {Success}",
                requestId, duration.TotalMilliseconds, context.Response.Success);
        }
        catch (Exception ex)
        {
            var duration = DateTime.UtcNow - startTime;
            _metrics.RecordError();

            _logger.LogError(ex,
                "MCP Error: {RequestId}, Duration: {Duration}ms, Error: {Message}",
                requestId, duration.TotalMilliseconds, ex.Message);
            throw;
        }
    }
}
```

## Configuration Schema

### Server Configuration
```csharp
public class McpServerConfig
{
    public string Name { get; set; } = "Hartonomous-Opus Hypercube";
    public string Version { get; set; } = "1.0.0";
    public ServerCapabilities Capabilities { get; set; } = new();
    public TransportConfig Transport { get; set; } = new();
    public SecurityConfig Security { get; set; } = new();
    public Dictionary<string, ToolConfig> Tools { get; set; } = new();
}

public class TransportConfig
{
    public string Type { get; set; } = "http";
    public HttpTransportConfig Http { get; set; } = new();
    public StdioTransportConfig Stdio { get; set; } = new();
}

public class HttpTransportConfig
{
    public string Path { get; set; } = "/mcp";
    public int Port { get; set; } = 5001;
    public bool RequireHttps { get; set; } = true;
    public CorsConfig Cors { get; set; } = new();
}

public class SecurityConfig
{
    public bool RequireAuthentication { get; set; } = true;
    public RateLimitConfig RateLimiting { get; set; } = new();
    public ContentFilterConfig ContentFiltering { get; set; } = new();
}
```

### Tool Configuration
```csharp
public class ToolConfig
{
    public bool Enabled { get; set; } = true;
    public string[] RequiredPermissions { get; set; } = Array.Empty<string>();
    public RateLimitConfig RateLimit { get; set; } = new();
    public ValidationConfig Validation { get; set; } = new();
    public ContentFilterConfig ContentFilter { get; set; } = new();
}
```

## Error Handling

### Error Code Mapping
```csharp
public static class McpErrorCodes
{
    public const int ParseError = -32700;
    public const int InvalidRequest = -32600;
    public const int MethodNotFound = -32601;
    public const int InvalidParams = -32602;
    public const int InternalError = -32603;

    // MCP-specific error codes
    public const int Unauthorized = 1;
    public const int Forbidden = 2;
    public const int NotFound = 3;
    public const int RateLimited = 4;
    public const int ContentFiltered = 5;
}
```

### Exception Handling
```csharp
public class McpException : Exception
{
    public int ErrorCode { get; }
    public object ErrorData { get; }

    public McpException(int errorCode, string message, object errorData = null)
        : base(message)
    {
        ErrorCode = errorCode;
        ErrorData = errorData;
    }
}

public static class McpErrorResponse
{
    public static McpResponse Create(McpException ex)
    {
        return new McpResponse
        {
            Success = false,
            Error = new McpError
            {
                Code = ex.ErrorCode,
                Message = ex.Message,
                Data = ex.ErrorData
            }
        };
    }
}
```

## Performance Optimization

### Caching Strategy
```csharp
public class McpCacheService
{
    private readonly IMemoryCache _cache;
    private readonly DistributedCacheEntryOptions _defaultOptions;

    public McpCacheService(IMemoryCache cache)
    {
        _cache = cache;
        _defaultOptions = new DistributedCacheEntryOptions
        {
            AbsoluteExpirationRelativeToNow = TimeSpan.FromMinutes(5),
            SlidingExpiration = TimeSpan.FromMinutes(2)
        };
    }

    public async Task<T> GetOrCreateAsync<T>(string key, Func<Task<T>> factory)
    {
        return await _cache.GetOrCreateAsync(key, entry =>
        {
            entry.SetOptions(_defaultOptions);
            return factory();
        });
    }

    public void InvalidatePattern(string pattern)
    {
        // Implementation for pattern-based cache invalidation
    }
}
```

### Connection Pooling
```csharp
public class McpConnectionPool
{
    private readonly ConcurrentQueue<IMcpConnection> _pool = new();
    private readonly SemaphoreSlim _semaphore;
    private readonly McpConnectionFactory _factory;

    public McpConnectionPool(int maxConnections, McpConnectionFactory factory)
    {
        _semaphore = new SemaphoreSlim(maxConnections, maxConnections);
        _factory = factory;
    }

    public async Task<IMcpConnection> GetConnectionAsync()
    {
        await _semaphore.WaitAsync();

        if (_pool.TryDequeue(out var connection))
        {
            if (connection.IsHealthy)
            {
                return new PooledConnection(connection, this);
            }
            else
            {
                await connection.DisposeAsync();
            }
        }

        return new PooledConnection(await _factory.CreateAsync(), this);
    }

    public void ReturnConnection(IMcpConnection connection)
    {
        if (connection.IsHealthy)
        {
            _pool.Enqueue(connection);
        }
        else
        {
            connection.DisposeAsync().GetAwaiter().GetResult();
        }

        _semaphore.Release();
    }
}
```

## Monitoring and Metrics

### Metrics Collection
```csharp
public class McpMetricsCollector
{
    private readonly Meter _meter = new("Hartonomous-Opus.MCP.Server");
    private readonly Counter<long> _requestsTotal;
    private readonly Counter<long> _errorsTotal;
    private readonly Histogram<double> _requestDuration;
    private readonly UpDownCounter<long> _activeConnections;

    public McpMetricsCollector()
    {
        _requestsTotal = _meter.CreateCounter<long>(
            "mcp_requests_total",
            description: "Total number of MCP requests");

        _errorsTotal = _meter.CreateCounter<long>(
            "mcp_errors_total",
            description: "Total number of MCP errors");

        _requestDuration = _meter.CreateHistogram<double>(
            "mcp_request_duration_seconds",
            description: "Duration of MCP requests in seconds");

        _activeConnections = _meter.CreateUpDownCounter<long>(
            "mcp_active_connections",
            description: "Number of active MCP connections");
    }

    public void RecordRequest(string method, TimeSpan duration, bool success)
    {
        _requestsTotal.Add(1, new KeyValuePair<string, object>("method", method));
        _requestDuration.Record(duration.TotalSeconds,
            new KeyValuePair<string, object>("method", method));

        if (!success)
        {
            _errorsTotal.Add(1, new KeyValuePair<string, object>("method", method));
        }
    }
}
```

### Health Checks
```csharp
public class McpServerHealthCheck : IHealthCheck
{
    private readonly IMcpServer _server;

    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        var healthData = new Dictionary<string, object>();

        try
        {
            // Check server responsiveness
            var serverInfo = await _server.GetServerInfoAsync(cancellationToken);
            healthData["server_info"] = serverInfo;
            healthData["server_status"] = "healthy";
        }
        catch (Exception ex)
        {
            healthData["server_status"] = "unhealthy";
            healthData["server_error"] = ex.Message;
        }

        // Check tool availability
        try
        {
            var tools = await _server.ListToolsAsync(cancellationToken);
            healthData["tools_count"] = tools.Length;
            healthData["tools_status"] = "healthy";
        }
        catch (Exception ex)
        {
            healthData["tools_status"] = "unhealthy";
            healthData["tools_error"] = ex.Message;
        }

        var isHealthy = healthData.Values.All(v =>
            v as string != "unhealthy");

        return isHealthy
            ? HealthCheckResult.Healthy("MCP server is healthy", healthData)
            : HealthCheckResult.Unhealthy("MCP server has issues", null, healthData);
    }
}
```

This specification provides comprehensive details for implementing a production-ready MCP server within the Hartonomous-Opus system, including tool definitions, middleware, security, performance optimization, and monitoring capabilities.