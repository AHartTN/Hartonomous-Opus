# MCP Integration Plan

## Overview
This document outlines the detailed integration plan for adding MCP server and client functionality to the Hartonomous-Opus app layer. The plan covers code changes, dependencies, testing, deployment, and rollout strategy.

## Current Architecture Assessment

### C# API Structure
```
HypercubeGenerativeApi/
├── Program.cs                    # Application entry point
├── Controllers/                  # HTTP endpoints
│   ├── CompletionsController.cs
│   ├── SemanticQueryController.cs
│   └── GeometricController.cs
├── Services/                    # Business logic
│   ├── GenerativeService.cs
│   ├── SemanticQueryService.cs
│   └── PostgresService.cs
├── Models/                      # Request/response DTOs
└── Interop/                     # C++ P/Invoke interfaces
```

### Integration Points
1. **Program.cs**: Add MCP services and middleware
2. **Services Layer**: Wrap existing services as MCP tools
3. **Controllers**: Add MCP endpoints alongside REST APIs
4. **Configuration**: Extend appsettings.json for MCP settings
5. **Dependencies**: Add MCP SDK NuGet package

## Implementation Phases

### Phase 1: Foundation Setup (Week 1)

#### 1.1 Add MCP SDK Dependency
**Files Modified**:
- `HypercubeGenerativeApi.csproj`

**Changes**:
```xml
<PackageReference Include="ModelContextProtocol" Version="1.0.0" />
```

**Rationale**: Adds official MCP C# SDK for server and client functionality.

#### 1.2 Create MCP Configuration Models
**New Files**:
- `Models/McpConfig.cs`
- `Models/McpServerConfig.cs`
- `Models/McpClientConfig.cs`

**Content** (McpConfig.cs):
```csharp
public class McpConfig
{
    public McpServerConfig Server { get; set; } = new();
    public Dictionary<string, McpClientConfig> Clients { get; set; } = new();
    public SecurityConfig Security { get; set; } = new();
    public ResilienceConfig Resilience { get; set; } = new();
}

public class McpServerConfig
{
    public bool Enabled { get; set; } = true;
    public TransportConfig Transport { get; set; } = new();
    public Dictionary<string, ToolConfig> Tools { get; set; } = new();
}

public class McpClientConfig
{
    public string Transport { get; set; } = "http";
    public string BaseUrl { get; set; }
    public Dictionary<string, string> Headers { get; set; } = new();
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
}

public class TransportConfig
{
    public string Type { get; set; } = "http";
    public string Path { get; set; } = "/mcp";
    public int Port { get; set; } = 5001;
}

public class ToolConfig
{
    public bool Enabled { get; set; } = true;
    public string[] RequiredPermissions { get; set; } = Array.Empty<string>();
    public RateLimitConfig RateLimit { get; set; } = new();
}

public class SecurityConfig
{
    public bool RequireAuthentication { get; set; } = true;
    public RateLimitConfig GlobalRateLimit { get; set; } = new();
}

public class ResilienceConfig
{
    public int MaxRetries { get; set; } = 3;
    public CircuitBreakerConfig CircuitBreaker { get; set; } = new();
}

public class RateLimitConfig
{
    public int RequestsPerMinute { get; set; } = 60;
    public int RequestsPerHour { get; set; } = 1000;
}

public class CircuitBreakerConfig
{
    public int FailureThreshold { get; set; } = 5;
    public TimeSpan RecoveryTimeout { get; set; } = TimeSpan.FromMinutes(1);
}
```

**Rationale**: Defines configuration structure for MCP server and client settings.

#### 1.3 Update appsettings.json
**Files Modified**:
- `appsettings.json`

**Changes**:
```json
{
  "Mcp": {
    "Server": {
      "Enabled": true,
      "Transport": {
        "Type": "http",
        "Path": "/mcp",
        "Port": 5001
      },
      "Tools": {
        "SemanticSearch": {
          "Enabled": true,
          "RequiredPermissions": ["semantic:read"],
          "RateLimit": {
            "RequestsPerMinute": 100
          }
        },
        "GenerateText": {
          "Enabled": true,
          "RequiredPermissions": ["generation:execute"],
          "RateLimit": {
            "RequestsPerMinute": 10
          }
        }
      }
    },
    "Clients": {
      "externalTools": {
        "Transport": "http",
        "BaseUrl": "https://api.example.com/mcp",
        "Headers": {
          "Authorization": "Bearer ${EXTERNAL_API_KEY}"
        }
      }
    },
    "Security": {
      "RequireAuthentication": true,
      "GlobalRateLimit": {
        "RequestsPerMinute": 1000,
        "RequestsPerHour": 10000
      }
    },
    "Resilience": {
      "MaxRetries": 3,
      "CircuitBreaker": {
        "FailureThreshold": 5,
        "RecoveryTimeout": "00:01:00"
      }
    }
  }
}
```

**Rationale**: Provides runtime configuration for MCP functionality.

#### 1.4 Create MCP Core Services
**New Files**:
- `Services/Interfaces/IMcpServerService.cs`
- `Services/Interfaces/IMcpClientService.cs`
- `Services/McpServerService.cs`
- `Services/McpClientService.cs`

**Content** (IMcpServerService.cs):
```csharp
public interface IMcpServerService
{
    Task StartAsync();
    Task StopAsync();
    Task<McpServerInfo> GetServerInfoAsync();
    Task<IEnumerable<string>> GetAvailableToolsAsync();
}
```

**Content** (IMcpClientService.cs):
```csharp
public interface IMcpClientService
{
    Task ConnectAsync(string serverId);
    Task DisconnectAsync(string serverId);
    Task<ToolResult> InvokeToolAsync(string serverId, string toolName, Dictionary<string, object> parameters);
    Task<IEnumerable<Tool>> GetAvailableToolsAsync(string serverId);
    Task<bool> IsConnectedAsync(string serverId);
}
```

**Rationale**: Defines core MCP service interfaces.

### Phase 2: MCP Server Implementation (Week 2)

#### 2.1 Create Tool Implementations
**New Files**:
- `Services/McpTools/SemanticSearchTool.cs`
- `Services/McpTools/GenerateTextTool.cs`
- `Services/McpTools/GeometricNeighborsTool.cs`
- `Services/McpTools/IngestContentTool.cs`

**Content** (SemanticSearchTool.cs):
```csharp
[McpServerTool, Description("Perform semantic similarity search using hypercube relationships")]
public class SemanticSearchTool
{
    private readonly ISemanticQueryService _semanticService;
    private readonly IAuthorizationService _authService;
    private readonly IRateLimiter _rateLimiter;

    public SemanticSearchTool(
        ISemanticQueryService semanticService,
        IAuthorizationService authService,
        IRateLimiter rateLimiter)
    {
        _semanticService = semanticService;
        _authService = authService;
        _rateLimiter = rateLimiter;
    }

    [McpServerToolMethod]
    public async Task<SemanticSearchResult> ExecuteAsync(
        [Inject] IUser user,
        string query,
        int limit = 10,
        double threshold = 0.1)
    {
        // Authorization
        await _authService.CheckPermissionAsync(user, "semantic:search");

        // Rate limiting
        if (!await _rateLimiter.CheckRateLimitAsync(user.Id, "SemanticSearch"))
            throw new McpException("Rate limit exceeded");

        // Business logic
        var results = await _semanticService.QuerySemantic(new SemanticQueryRequest
        {
            Query = query,
            Limit = limit,
            Threshold = threshold
        });

        return new SemanticSearchResult
        {
            Query = query,
            Results = results.Select(r => new SemanticResult
            {
                Term = r.Term,
                Similarity = r.Similarity
            }).ToArray()
        };
    }
}
```

**Rationale**: Implements MCP tools that wrap existing business logic with security and rate limiting.

#### 2.2 Create MCP Server Middleware
**New Files**:
- `Middleware/McpAuthenticationMiddleware.cs`
- `Middleware/McpAuthorizationMiddleware.cs`
- `Middleware/McpRateLimitMiddleware.cs`
- `Middleware/McpLoggingMiddleware.cs`

**Content** (McpAuthenticationMiddleware.cs):
```csharp
public class McpAuthenticationMiddleware
{
    private readonly IJwtTokenValidator _tokenValidator;

    public async Task InvokeAsync(McpRequestContext context)
    {
        var authHeader = context.Request.Headers.GetValueOrDefault("Authorization");
        if (string.IsNullOrEmpty(authHeader) || !authHeader.StartsWith("Bearer "))
        {
            context.Response = CreateErrorResponse(McpErrorCodes.Unauthorized, "Missing token");
            return;
        }

        var token = authHeader.Substring("Bearer ".Length);
        var validationResult = await _tokenValidator.ValidateAsync(token);

        if (!validationResult.IsValid)
        {
            context.Response = CreateErrorResponse(McpErrorCodes.Unauthorized, validationResult.Error);
            return;
        }

        context.User = validationResult.User;
        await _next(context);
    }
}
```

**Rationale**: Implements security middleware for MCP requests.

#### 2.3 Update Program.cs for MCP Server
**Files Modified**:
- `Program.cs`

**Changes**:
```csharp
// Add MCP services
builder.Services.AddMcpServer();
builder.Services.AddMcpToolsFromAssembly(typeof(SemanticSearchTool).Assembly);

// Add MCP middleware
app.UseMcpAuthentication();
app.UseMcpAuthorization();
app.UseMcpRateLimiting();
app.UseMcpLogging();

// Map MCP endpoint
app.MapMcp("/mcp");

// Initialize MCP server
var mcpServer = app.Services.GetRequiredService<IMcpServer>();
await mcpServer.StartAsync();
```

**Rationale**: Integrates MCP server into the ASP.NET Core pipeline.

#### 2.4 Add MCP Controllers (Optional)
**New Files**:
- `Controllers/McpController.cs`

**Content**:
```csharp
[ApiController]
[Route("mcp")]
public class McpController : ControllerBase
{
    private readonly IMcpServerService _mcpServer;

    public McpController(IMcpServerService mcpServer)
    {
        _mcpServer = mcpServer;
    }

    [HttpGet("tools")]
    public async Task<IActionResult> GetTools()
    {
        var tools = await _mcpServer.GetAvailableToolsAsync();
        return Ok(tools);
    }

    [HttpGet("info")]
    public async Task<IActionResult> GetServerInfo()
    {
        var info = await _mcpServer.GetServerInfoAsync();
        return Ok(info);
    }
}
```

**Rationale**: Provides REST API access to MCP server information for debugging.

### Phase 3: MCP Client Implementation (Week 3)

#### 3.1 Implement Transport Layer
**New Files**:
- `Services/Transports/HttpMcpTransport.cs`
- `Services/Transports/StdioMcpTransport.cs`
- `Services/Transports/IMcpTransport.cs`

**Content** (HttpMcpTransport.cs):
```csharp
public class HttpMcpTransport : IMcpTransport
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

    public HttpMcpTransport(string baseUrl, Dictionary<string, string> headers = null)
    {
        _baseUrl = baseUrl;
        _httpClient = new HttpClient();

        if (headers != null)
        {
            foreach (var (key, value) in headers)
            {
                _httpClient.DefaultRequestHeaders.Add(key, value);
            }
        }
    }

    public async Task<McpResponse> SendRequestAsync(McpRequest request)
    {
        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync(_baseUrl, content);
        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<McpResponse>(responseJson);
    }
}
```

**Rationale**: Implements HTTP transport for MCP client communications.

#### 3.2 Implement Connection Pooling
**New Files**:
- `Services/McpConnectionPool.cs`

**Content**:
```csharp
public class McpConnectionPool
{
    private readonly ConcurrentDictionary<string, ConnectionPool> _pools = new();
    private readonly IMcpTransportFactory _transportFactory;

    public async Task<IMcpConnection> GetConnectionAsync(string serverId)
    {
        var pool = _pools.GetOrAdd(serverId, _ => new ConnectionPool(_transportFactory, serverId));
        return await pool.GetConnectionAsync();
    }

    public async Task ReleaseConnectionAsync(string serverId, IMcpConnection connection)
    {
        if (_pools.TryGetValue(serverId, out var pool))
        {
            await pool.ReleaseConnectionAsync(connection);
        }
    }

    private class ConnectionPool
    {
        private readonly ConcurrentQueue<IMcpConnection> _connections = new();
        private readonly SemaphoreSlim _semaphore;
        private readonly IMcpTransportFactory _transportFactory;
        private readonly string _serverId;

        public ConnectionPool(IMcpTransportFactory transportFactory, string serverId, int maxConnections = 10)
        {
            _transportFactory = transportFactory;
            _serverId = serverId;
            _semaphore = new SemaphoreSlim(maxConnections, maxConnections);
        }

        public async Task<IMcpConnection> GetConnectionAsync()
        {
            await _semaphore.WaitAsync();

            if (_connections.TryDequeue(out var connection))
            {
                if (await connection.IsHealthyAsync())
                {
                    return new PooledConnection(connection, this);
                }
                else
                {
                    await connection.DisposeAsync();
                }
            }

            var transport = await _transportFactory.CreateTransportAsync(_serverId);
            var newConnection = new McpConnection(transport);
            return new PooledConnection(newConnection, this);
        }

        public async Task ReleaseConnectionAsync(IMcpConnection connection)
        {
            if (connection is PooledConnection pooled)
            {
                var innerConnection = pooled.GetInnerConnection();
                if (await innerConnection.IsHealthyAsync())
                {
                    _connections.Enqueue(innerConnection);
                }
                else
                {
                    await innerConnection.DisposeAsync();
                }
            }

            _semaphore.Release();
        }
    }
}
```

**Rationale**: Provides efficient connection management for MCP clients.

#### 3.3 Implement Client Service
**Files Modified**:
- `Services/McpClientService.cs`

**Content**:
```csharp
public class McpClientService : IMcpClientService
{
    private readonly IMcpConnectionPool _connectionPool;
    private readonly IMcpToolDiscoveryService _toolDiscovery;
    private readonly ILogger<McpClientService> _logger;

    public async Task ConnectAsync(string serverId)
    {
        try
        {
            var connection = await _connectionPool.GetConnectionAsync(serverId);
            // Connection is pooled and will be reused
            _logger.LogInformation("Connected to MCP server: {ServerId}", serverId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to connect to MCP server: {ServerId}", serverId);
            throw;
        }
    }

    public async Task<ToolResult> InvokeToolAsync(
        string serverId, string toolName, Dictionary<string, object> parameters)
    {
        var connection = await _connectionPool.GetConnectionAsync(serverId);

        try
        {
            var toolCall = new ToolCallRequest
            {
                Name = toolName,
                Arguments = parameters
            };

            var response = await connection.SendRequestAsync(new McpRequest
            {
                Method = "tools/call",
                Params = toolCall
            });

            if (!response.Success)
            {
                throw new McpException($"Tool call failed: {response.Error?.Message}");
            }

            return response.Result as ToolResult;
        }
        finally
        {
            await _connectionPool.ReleaseConnectionAsync(serverId, connection);
        }
    }

    public async Task<IEnumerable<Tool>> GetAvailableToolsAsync(string serverId)
    {
        return await _toolDiscovery.DiscoverToolsAsync(serverId);
    }
}
```

**Rationale**: Implements the core MCP client functionality.

#### 3.4 Integrate Client into Existing Services
**Files Modified**:
- `Services/GenerativeService.cs`

**Changes**:
```csharp
public class EnhancedGenerativeService
{
    private readonly GenerativeService _baseService;
    private readonly IMcpClientService _mcpClient;

    public async Task<CompletionResult> GenerateCompletion(CompletionRequest request)
    {
        // Base generation
        var result = await _baseService.GenerateCompletion(request);

        // Augment with external tools if configured
        if (_configuration.EnableExternalAugmentation)
        {
            try
            {
                var externalContext = await _mcpClient.InvokeToolAsync(
                    "externalTools",
                    "get_relevant_context",
                    new Dictionary<string, object>
                    {
                        ["query"] = request.Prompt,
                        ["max_tokens"] = 500
                    });

                result = await CombineResultsAsync(result, externalContext);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "External augmentation failed");
            }
        }

        return result;
    }
}
```

**Rationale**: Enhances existing services with MCP client capabilities.

### Phase 4: Testing and Validation (Week 4)

#### 4.1 Unit Tests
**New Files**:
- `Tests/McpTools/SemanticSearchToolTests.cs`
- `Tests/McpServices/McpClientServiceTests.cs`

**Content** (SemanticSearchToolTests.cs):
```csharp
[TestClass]
public class SemanticSearchToolTests
{
    private Mock<ISemanticQueryService> _semanticServiceMock;
    private Mock<IAuthorizationService> _authServiceMock;
    private Mock<IRateLimiter> _rateLimiterMock;
    private SemanticSearchTool _tool;

    [TestInitialize]
    public void Setup()
    {
        _semanticServiceMock = new Mock<ISemanticQueryService>();
        _authServiceMock = new Mock<IAuthorizationService>();
        _rateLimiterMock = new Mock<IRateLimiter>();

        _tool = new SemanticSearchTool(
            _semanticServiceMock.Object,
            _authServiceMock.Object,
            _rateLimiterMock.Object);
    }

    [TestMethod]
    public async Task ExecuteAsync_ValidRequest_ReturnsResults()
    {
        // Arrange
        var user = new Mock<IUser>();
        var results = new[]
        {
            new SemanticResult { Term = "test", Similarity = 0.8 }
        };

        _authServiceMock.Setup(a => a.CheckPermissionAsync(user.Object, "semantic:search"))
            .ReturnsAsync(true);
        _rateLimiterMock.Setup(r => r.CheckRateLimitAsync(user.Object.Id, "SemanticSearch"))
            .ReturnsAsync(true);
        _semanticServiceMock.Setup(s => s.QuerySemantic(It.IsAny<SemanticQueryRequest>()))
            .ReturnsAsync(results);

        // Act
        var result = await _tool.ExecuteAsync(user.Object, "test query", 10, 0.1);

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("test query", result.Query);
        Assert.AreEqual(1, result.Results.Length);
    }
}
```

**Rationale**: Ensures MCP tools work correctly in isolation.

#### 4.2 Integration Tests
**New Files**:
- `Tests/Integration/McpServerIntegrationTests.cs`

**Content**:
```csharp
[TestClass]
public class McpServerIntegrationTests : TestServerFixture
{
    [TestMethod]
    public async Task EndToEnd_SemanticSearch_Success()
    {
        // Arrange
        var client = CreateMcpClient();

        // Act
        var response = await client.SendRequestAsync(new McpRequest
        {
            Method = "tools/call",
            Params = new ToolCallRequest
            {
                Name = "SemanticSearch",
                Arguments = new Dictionary<string, object>
                {
                    ["query"] = "artificial intelligence",
                    ["limit"] = 5
                }
            }
        });

        // Assert
        Assert.IsTrue(response.Success);
        var result = response.Result as SemanticSearchResult;
        Assert.IsNotNull(result);
        Assert.AreEqual("artificial intelligence", result.Query);
    }
}
```

**Rationale**: Tests complete MCP server functionality.

#### 4.3 Load Tests
**New Files**:
- `Tests/Load/McpLoadTests.cs`

**Content**:
```csharp
[TestClass]
public class McpLoadTests
{
    [TestMethod]
    public async Task HighConcurrency_LoadTest()
    {
        var clients = Enumerable.Range(0, 100)
            .Select(_ => CreateMcpClient())
            .ToArray();

        var tasks = clients.Select(client =>
            client.SendRequestAsync(CreateSemanticSearchRequest()));

        var stopwatch = Stopwatch.StartNew();
        var results = await Task.WhenAll(tasks);
        stopwatch.Stop();

        Assert.AreEqual(100, results.Length);
        Assert.IsTrue(results.All(r => r.Success));
        Assert.IsTrue(stopwatch.Elapsed < TimeSpan.FromSeconds(30));
    }
}
```

**Rationale**: Validates performance under load.

### Phase 5: Deployment and Monitoring (Week 4)

#### 5.1 Update Dockerfile
**Files Modified**:
- `Dockerfile`

**Changes**:
```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443
EXPOSE 5001  # MCP port

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["HypercubeGenerativeApi.csproj", "."]
RUN dotnet restore
COPY . .
RUN dotnet build -c Release -o /app/build

FROM build AS publish
RUN dotnet publish -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "HypercubeGenerativeApi.dll"]
```

**Rationale**: Exposes MCP port in container.

#### 5.2 Add Health Checks
**Files Modified**:
- `Program.cs`

**Changes**:
```csharp
builder.Services.AddHealthChecks()
    .AddCheck<McpServerHealthCheck>("mcp-server")
    .AddCheck<McpClientHealthCheck>("mcp-client");
```

**Rationale**: Provides health monitoring for MCP services.

#### 5.3 Add Metrics and Monitoring
**New Files**:
- `Services/McpMetricsService.cs`

**Content**:
```csharp
public class McpMetricsService
{
    private readonly Meter _meter = new("Hartonomous-Opus.MCP");

    public McpMetricsService()
    {
        _meter.CreateCounter<long>("mcp_requests_total", "Total MCP requests");
        _meter.CreateCounter<long>("mcp_errors_total", "Total MCP errors");
        _meter.CreateHistogram<double>("mcp_request_duration_seconds", "Request duration");
    }

    public void RecordRequest(string toolName, TimeSpan duration, bool success)
    {
        // Record metrics
    }
}
```

**Rationale**: Enables observability for MCP operations.

## Rollout Strategy

### Phase 1: Development Environment (Week 1-2)
- Deploy to development environment
- Enable MCP server with basic tools
- Test with internal clients
- Validate security measures

### Phase 2: Staging Environment (Week 3)
- Deploy to staging with full MCP functionality
- Enable MCP client connections
- Conduct integration testing
- Performance benchmarking

### Phase 3: Production Rollout (Week 4)
- **Canary Deployment**: 10% traffic to MCP-enabled instances
- **Feature Flags**: Gradually enable MCP features
- **Monitoring**: Close monitoring of metrics and errors
- **Rollback Plan**: Ability to disable MCP features immediately

### Phase 4: Full Production (Week 5+)
- **100% Deployment**: All instances MCP-enabled
- **External Access**: Open MCP endpoints to external clients
- **Documentation**: Publish MCP API documentation
- **Support**: Train support team on MCP functionality

## Risk Mitigation

### Technical Risks
1. **Performance Impact**
   - Mitigation: Comprehensive performance testing, resource monitoring

2. **Security Vulnerabilities**
   - Mitigation: Security audit, penetration testing, gradual rollout

3. **Compatibility Issues**
   - Mitigation: Extensive integration testing, feature flags

### Operational Risks
1. **Increased Complexity**
   - Mitigation: Comprehensive documentation, training

2. **Monitoring Gaps**
   - Mitigation: Implement comprehensive metrics and alerting

3. **Rollback Challenges**
   - Mitigation: Feature flags, canary deployment strategy

## Success Criteria

### Functional Requirements
- [ ] MCP server exposes all specified tools
- [ ] MCP client can connect to external servers
- [ ] Authentication and authorization work correctly
- [ ] Rate limiting prevents abuse
- [ ] Error handling is robust

### Non-Functional Requirements
- [ ] Response times under 500ms for typical operations
- [ ] 99.9% uptime for MCP services
- [ ] No security vulnerabilities
- [ ] Comprehensive test coverage
- [ ] Complete documentation

## Dependencies and Prerequisites

### External Dependencies
- ModelContextProtocol NuGet package
- ASP.NET Core 8.0
- PostgreSQL with hypercube extensions

### Internal Dependencies
- Existing semantic query services
- Authentication and authorization systems
- Database connectivity
- Logging and monitoring infrastructure

## Timeline Summary

| Phase | Duration | Key Activities | Deliverables |
|-------|----------|----------------|--------------|
| Foundation | Week 1 | SDK integration, basic setup | Config models, basic services |
| Server Implementation | Week 2 | Tool development, middleware | MCP server with tools |
| Client Implementation | Week 3 | Transport layer, integration | MCP client functionality |
| Testing & Deployment | Week 4 | Testing, monitoring, rollout | Production deployment |
| Full Production | Week 5+ | External access, support | Complete MCP ecosystem |

This integration plan provides a structured approach to adding MCP functionality while minimizing risk and ensuring robust implementation.