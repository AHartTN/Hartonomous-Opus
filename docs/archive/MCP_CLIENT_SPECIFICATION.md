# MCP Client Implementation Specification

## Overview
This document provides detailed technical specifications for implementing MCP client functionality within the Hartonomous-Opus system. The client enables connection to external MCP servers, discovery and invocation of remote tools, and integration with local services.

## Architecture Design

### Client Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local Service │────│   MCP Client     │────│   MCP Server    │
│                 │    │                  │    │  (External)     │
│ - Generation    │    │ - Connection Mgmt│    │                 │
│ - Ingestion     │    │ - Tool Discovery │    │ - Tools         │
│ - Analysis      │    │ - Call Routing   │    │ - Resources     │
└─────────────────┘    │ - Error Handling │    └─────────────────┘
                       │ - Load Balancing │
                       └──────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Circuit Breaker │
                       │  Rate Limiting   │
                       └─────────────────┘
```

### Client Components

#### Connection Manager
```csharp
public interface IMcpConnectionManager
{
    Task<IMcpConnection> GetConnectionAsync(string serverId);
    Task ReleaseConnectionAsync(string serverId, IMcpConnection connection);
    Task HealthCheckAsync(string serverId);
    Task<IEnumerable<string>> GetConnectedServerIdsAsync();
}
```

#### Tool Discovery Service
```csharp
public interface IMcpToolDiscoveryService
{
    Task<IEnumerable<Tool>> DiscoverToolsAsync(string serverId);
    Task<Tool> GetToolAsync(string serverId, string toolName);
    Task<bool> IsToolAvailableAsync(string serverId, string toolName);
    Task UpdateToolCacheAsync(string serverId);
}
```

#### Tool Invocation Service
```csharp
public interface IMcpToolInvocationService
{
    Task<ToolResult> InvokeToolAsync(string serverId, string toolName, Dictionary<string, object> parameters);
    Task<ToolResult> InvokeToolWithRetryAsync(string serverId, string toolName, Dictionary<string, object> parameters, int maxRetries = 3);
    Task<BatchToolResult> InvokeToolsBatchAsync(IEnumerable<ToolCallRequest> requests);
}
```

## Transport Implementations

### HTTP Transport

#### Configuration
```csharp
public class HttpMcpTransportConfig
{
    public string BaseUrl { get; set; }
    public Dictionary<string, string> Headers { get; set; } = new();
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
    public int MaxRetries { get; set; } = 3;
    public TimeSpan RetryDelay { get; set; } = TimeSpan.FromSeconds(1);
    public bool UseHttps { get; set; } = true;
}
```

#### Implementation
```csharp
public class HttpMcpTransport : IMcpTransport
{
    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly HttpMcpTransportConfig _config;

    public HttpMcpTransport(HttpMcpTransportConfig config)
    {
        _config = config;
        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(config.BaseUrl),
            Timeout = config.Timeout
        };

        foreach (var (key, value) in config.Headers)
        {
            _httpClient.DefaultRequestHeaders.Add(key, value);
        }

        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
    }

    public async Task<McpResponse> SendRequestAsync(McpRequest request)
    {
        var retryPolicy = Policy
            .Handle<HttpRequestException>()
            .Or<TaskCanceledException>()
            .WaitAndRetryAsync(_config.MaxRetries,
                retryAttempt => _config.RetryDelay * Math.Pow(2, retryAttempt));

        return await retryPolicy.ExecuteAsync(async () =>
        {
            var jsonContent = JsonSerializer.Serialize(request, _jsonOptions);
            var httpContent = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync("", httpContent);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<McpResponse>(responseJson, _jsonOptions);
        });
    }
}
```

### Stdio Transport

#### Configuration
```csharp
public class StdioMcpTransportConfig
{
    public string Command { get; set; }
    public string[] Arguments { get; set; } = Array.Empty<string>();
    public Dictionary<string, string> Environment { get; set; } = new();
    public string WorkingDirectory { get; set; }
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
}
```

#### Implementation
```csharp
public class StdioMcpTransport : IMcpTransport, IDisposable
{
    private readonly Process _process;
    private readonly StreamReader _reader;
    private readonly StreamWriter _writer;
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly JsonSerializerOptions _jsonOptions;

    public StdioMcpTransport(StdioMcpTransportConfig config)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = config.Command,
            Arguments = string.Join(" ", config.Arguments),
            WorkingDirectory = config.WorkingDirectory ?? Directory.GetCurrentDirectory(),
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        foreach (var (key, value) in config.Environment)
        {
            startInfo.Environment[key] = value;
        }

        _process = Process.Start(startInfo);
        _reader = _process.StandardOutput;
        _writer = _process.StandardInput;

        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
    }

    public async Task<McpResponse> SendRequestAsync(McpRequest request)
    {
        await _writeLock.WaitAsync();
        try
        {
            var jsonRequest = JsonSerializer.Serialize(request, _jsonOptions);
            await _writer.WriteLineAsync(jsonRequest);
            await _writer.FlushAsync();

            var jsonResponse = await _reader.ReadLineAsync();
            if (jsonResponse == null)
                throw new InvalidOperationException("Process terminated unexpectedly");

            return JsonSerializer.Deserialize<McpResponse>(jsonResponse, _jsonOptions);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    public void Dispose()
    {
        _writer?.Dispose();
        _reader?.Dispose();
        _process?.Kill();
        _process?.Dispose();
    }
}
```

## Connection Pooling

### Connection Pool Implementation
```csharp
public class McpConnectionPool
{
    private readonly ConcurrentDictionary<string, ConnectionPool> _pools = new();
    private readonly IMcpTransportFactory _transportFactory;

    public McpConnectionPool(IMcpTransportFactory transportFactory)
    {
        _transportFactory = transportFactory;
    }

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

    private class PooledConnection : IMcpConnection
    {
        private readonly IMcpConnection _inner;
        private readonly ConnectionPool _pool;

        public PooledConnection(IMcpConnection inner, ConnectionPool pool)
        {
            _inner = inner;
            _pool = pool;
        }

        public Task<McpResponse> SendRequestAsync(McpRequest request)
            => _inner.SendRequestAsync(request);

        public Task<bool> IsHealthyAsync()
            => _inner.IsHealthyAsync();

        public async ValueTask DisposeAsync()
        {
            await _pool.ReleaseConnectionAsync(this);
        }

        public IMcpConnection GetInnerConnection() => _inner;
    }
}
```

## Error Handling and Resilience

### Circuit Breaker Pattern
```csharp
public class McpCircuitBreaker
{
    private CircuitState _state = CircuitState.Closed;
    private int _failureCount = 0;
    private DateTime _lastFailureTime;
    private readonly int _failureThreshold;
    private readonly TimeSpan _timeoutPeriod;

    public McpCircuitBreaker(int failureThreshold = 5, TimeSpan? timeoutPeriod = null)
    {
        _failureThreshold = failureThreshold;
        _timeoutPeriod = timeoutPeriod ?? TimeSpan.FromMinutes(1);
    }

    public async Task<T> ExecuteAsync<T>(Func<Task<T>> operation)
    {
        if (_state == CircuitState.Open)
        {
            if (DateTime.UtcNow - _lastFailureTime > _timeoutPeriod)
            {
                _state = CircuitState.HalfOpen;
            }
            else
            {
                throw new CircuitBreakerOpenException();
            }
        }

        try
        {
            var result = await operation();
            OnSuccess();
            return result;
        }
        catch (Exception ex)
        {
            OnFailure();
            throw;
        }
    }

    private void OnSuccess()
    {
        _failureCount = 0;
        _state = CircuitState.Closed;
    }

    private void OnFailure()
    {
        _failureCount++;
        _lastFailureTime = DateTime.UtcNow;

        if (_failureCount >= _failureThreshold)
        {
            _state = CircuitState.Open;
        }
    }

    private enum CircuitState { Closed, Open, HalfOpen }
}
```

### Retry Policy
```csharp
public class McpRetryPolicy
{
    private readonly int _maxRetries;
    private readonly TimeSpan _initialDelay;
    private readonly double _backoffMultiplier;

    public McpRetryPolicy(int maxRetries = 3, TimeSpan? initialDelay = null, double backoffMultiplier = 2.0)
    {
        _maxRetries = maxRetries;
        _initialDelay = initialDelay ?? TimeSpan.FromSeconds(1);
        _backoffMultiplier = backoffMultiplier;
    }

    public async Task<T> ExecuteAsync<T>(Func<Task<T>> operation)
    {
        var exceptions = new List<Exception>();

        for (int attempt = 0; attempt <= _maxRetries; attempt++)
        {
            try
            {
                return await operation();
            }
            catch (Exception ex) when (IsRetryableException(ex))
            {
                exceptions.Add(ex);

                if (attempt < _maxRetries)
                {
                    var delay = TimeSpan.FromTicks((long)(_initialDelay.Ticks * Math.Pow(_backoffMultiplier, attempt)));
                    await Task.Delay(delay);
                }
            }
        }

        throw new AggregateException("Operation failed after retries", exceptions);
    }

    private bool IsRetryableException(Exception ex)
    {
        return ex is HttpRequestException ||
               ex is TaskCanceledException ||
               ex is IOException;
    }
}
```

### Fallback Strategy
```csharp
public class McpFallbackStrategy
{
    private readonly Dictionary<string, Func<Task<object>>> _fallbacks = new();

    public void RegisterFallback(string operation, Func<Task<object>> fallback)
    {
        _fallbacks[operation] = fallback;
    }

    public async Task<T> ExecuteWithFallbackAsync<T>(string operation, Func<Task<T>> primary, Func<T> defaultValue = null)
    {
        try
        {
            return await primary();
        }
        catch (Exception ex)
        {
            if (_fallbacks.TryGetValue(operation, out var fallback))
            {
                try
                {
                    return (T)await fallback();
                }
                catch (Exception fallbackEx)
                {
                    // Log fallback failure but don't throw
                }
            }

            return defaultValue != null ? defaultValue() : default;
        }
    }
}
```

## Tool Augmentation Framework

### Tool Augmentation Service
```csharp
public class McpToolAugmentationService
{
    private readonly IMcpClientService _mcpClient;
    private readonly Dictionary<string, IToolAugmenter> _augmenters = new();

    public void RegisterAugmenter(string operation, IToolAugmenter augmenter)
    {
        _augmenters[operation] = augmenter;
    }

    public async Task<T> AugmentOperationAsync<T>(string operation, Func<Task<T>> primaryOperation, object context = null)
    {
        var result = await primaryOperation();

        if (_augmenters.TryGetValue(operation, out var augmenter))
        {
            result = await augmenter.AugmentAsync(result, context);
        }

        return result;
    }
}

public interface IToolAugmenter
{
    Task<T> AugmentAsync<T>(T input, object context = null);
}
```

### Generation Augmenter Example
```csharp
public class GenerationAugmenter : IToolAugmenter
{
    private readonly IMcpClientService _mcpClient;

    public async Task<T> AugmentAsync<T>(T input, object context = null)
    {
        if (input is CompletionResult completion)
        {
            try
            {
                // Use external MCP tool for fact-checking
                var factCheckResult = await _mcpClient.InvokeToolAsync(
                    "externalTools",
                    "fact_check",
                    new Dictionary<string, object> { ["text"] = completion.Text });

                if (factCheckResult.Success && factCheckResult.Content is Dictionary<string, object> facts)
                {
                    completion.Facts = facts;
                }
            }
            catch (Exception ex)
            {
                // Log but don't fail the operation
            }
        }

        return input;
    }
}
```

### Ingestion Augmenter Example
```csharp
public class IngestionAugmenter : IToolAugmenter
{
    private readonly IMcpClientService _mcpClient;

    public async Task<T> AugmentAsync<T>(T input, object context = null)
    {
        if (input is IngestionResult ingestion)
        {
            try
            {
                // Use external MCP tool for content analysis
                var analysisResult = await _mcpClient.InvokeToolAsync(
                    "externalTools",
                    "analyze_content",
                    new Dictionary<string, object> { ["content"] = ingestion.OriginalContent });

                if (analysisResult.Success && analysisResult.Content is Dictionary<string, object> analysis)
                {
                    ingestion.Analysis = analysis;
                }
            }
            catch (Exception ex)
            {
                // Log but don't fail the operation
            }
        }

        return input;
    }
}
```

## Configuration Management

### Client Configuration Schema
```csharp
public class McpClientConfiguration
{
    public Dictionary<string, McpServerConfig> Servers { get; set; } = new();
    public ClientSettings Settings { get; set; } = new();
}

public class McpServerConfig
{
    public string Transport { get; set; } // "http" or "stdio"
    public HttpTransportConfig Http { get; set; }
    public StdioTransportConfig Stdio { get; set; }
    public ResilienceConfig Resilience { get; set; } = new();
    public AuthenticationConfig Authentication { get; set; }
}

public class HttpTransportConfig
{
    public string BaseUrl { get; set; }
    public Dictionary<string, string> Headers { get; set; } = new();
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
}

public class StdioTransportConfig
{
    public string Command { get; set; }
    public string[] Arguments { get; set; } = Array.Empty<string>();
    public Dictionary<string, string> Environment { get; set; } = new();
    public string WorkingDirectory { get; set; }
}

public class ResilienceConfig
{
    public int MaxRetries { get; set; } = 3;
    public TimeSpan RetryDelay { get; set; } = TimeSpan.FromSeconds(1);
    public int CircuitBreakerThreshold { get; set; } = 5;
    public TimeSpan CircuitBreakerTimeout { get; set; } = TimeSpan.FromMinutes(1);
}

public class AuthenticationConfig
{
    public string Type { get; set; } // "bearer", "basic", "api-key"
    public string Token { get; set; }
    public string Username { get; set; }
    public string Password { get; set; }
    public string ApiKey { get; set; }
    public string ApiKeyHeader { get; set; } = "X-API-Key";
}

public class ClientSettings
{
    public int ConnectionPoolSize { get; set; } = 10;
    public TimeSpan ConnectionIdleTimeout { get; set; } = TimeSpan.FromMinutes(5);
    public TimeSpan ToolCacheExpiration { get; set; } = TimeSpan.FromMinutes(10);
    public bool EnableAugmentation { get; set; } = true;
    public string[] EnabledAugmenters { get; set; } = Array.Empty<string>();
}
```

### Configuration Loading
```csharp
public class McpConfigurationLoader
{
    private readonly IConfiguration _configuration;

    public McpConfigurationLoader(IConfiguration configuration)
    {
        _configuration = configuration;
    }

    public McpClientConfiguration LoadConfiguration()
    {
        var config = new McpClientConfiguration();

        // Load server configurations
        var serversSection = _configuration.GetSection("McpClients");
        foreach (var serverSection in serversSection.GetChildren())
        {
            var serverConfig = serverSection.Get<McpServerConfig>();
            if (serverConfig != null)
            {
                config.Servers[serverSection.Key] = serverConfig;
            }
        }

        // Load client settings
        config.Settings = _configuration.GetSection("McpClient").Get<ClientSettings>() ?? new ClientSettings();

        return config;
    }
}
```

## Monitoring and Metrics

### Client Metrics Collector
```csharp
public class McpClientMetricsCollector
{
    private readonly Meter _meter = new("Hartonomous-Opus.MCP.Client");
    private readonly Counter<long> _requestsTotal;
    private readonly Counter<long> _errorsTotal;
    private readonly Histogram<double> _requestDuration;
    private readonly UpDownCounter<long> _activeConnections;

    public McpClientMetricsCollector()
    {
        _requestsTotal = _meter.CreateCounter<long>(
            "mcp_client_requests_total",
            description: "Total number of MCP client requests");

        _errorsTotal = _meter.CreateCounter<long>(
            "mcp_client_errors_total",
            description: "Total number of MCP client errors");

        _requestDuration = _meter.CreateHistogram<double>(
            "mcp_client_request_duration_seconds",
            description: "Duration of MCP client requests in seconds");

        _activeConnections = _meter.CreateUpDownCounter<long>(
            "mcp_client_active_connections",
            description: "Number of active MCP client connections");
    }

    public void RecordRequest(string serverId, string toolName, TimeSpan duration, bool success)
    {
        var tags = new[]
        {
            new KeyValuePair<string, object>("server_id", serverId),
            new KeyValuePair<string, object>("tool_name", toolName)
        };

        _requestsTotal.Add(1, tags);
        _requestDuration.Record(duration.TotalSeconds, tags);

        if (!success)
        {
            _errorsTotal.Add(1, tags);
        }
    }

    public void UpdateActiveConnections(int delta)
    {
        _activeConnections.Add(delta);
    }
}
```

### Health Monitoring
```csharp
public class McpClientHealthMonitor
{
    private readonly IMcpClientService _clientService;
    private readonly ILogger<McpClientHealthMonitor> _logger;

    public async Task<HealthReport> CheckHealthAsync()
    {
        var healthReport = new HealthReport();
        var connectedServers = await _clientService.GetConnectedServerIdsAsync();

        foreach (var serverId in connectedServers)
        {
            try
            {
                var isHealthy = await _clientService.HealthCheckAsync(serverId);
                healthReport.ServerHealth[serverId] = isHealthy;

                if (!isHealthy)
                {
                    healthReport.UnhealthyServers++;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Health check failed for server {ServerId}", serverId);
                healthReport.ServerHealth[serverId] = false;
                healthReport.UnhealthyServers++;
            }
        }

        healthReport.IsHealthy = healthReport.UnhealthyServers == 0;
        return healthReport;
    }
}

public class HealthReport
{
    public bool IsHealthy { get; set; }
    public int UnhealthyServers { get; set; }
    public Dictionary<string, bool> ServerHealth { get; set; } = new();
}
```

## Self-Connection Handling

### Self-Connection Detection
```csharp
public class SelfConnectionDetector
{
    private readonly string _localServerId;
    private readonly IMcpServerInfoProvider _serverInfoProvider;

    public SelfConnectionDetector(string localServerId, IMcpServerInfoProvider serverInfoProvider)
    {
        _localServerId = localServerId;
        _serverInfoProvider = serverInfoProvider;
    }

    public async Task<bool> IsSelfConnectionAsync(string serverId, IMcpConnection connection)
    {
        try
        {
            var serverInfo = await connection.GetServerInfoAsync();

            // Check if server info matches local server
            if (serverInfo.Name == _serverInfoProvider.GetLocalServerName() &&
                serverInfo.Version == _serverInfoProvider.GetLocalServerVersion())
            {
                // Additional checks could include:
                // - IP address comparison
                // - Server capabilities comparison
                // - Unique server identifier comparison

                return true;
            }
        }
        catch (Exception ex)
        {
            // If we can't get server info, assume it's not a self-connection
        }

        return false;
    }
}
```

### Loop Prevention
```csharp
public class LoopPreventionService
{
    private readonly AsyncLocal<Stack<OperationContext>> _operationStack = new();

    public IDisposable EnterOperation(string operationId, string serverId, string toolName)
    {
        var stack = _operationStack.Value ??= new Stack<OperationContext>();

        // Check for potential loops
        if (stack.Any(ctx => ctx.ServerId == serverId && ctx.ToolName == toolName))
        {
            throw new InvalidOperationException($"Potential loop detected for {toolName} on {serverId}");
        }

        var context = new OperationContext
        {
            OperationId = operationId,
            ServerId = serverId,
            ToolName = toolName,
            Timestamp = DateTime.UtcNow
        };

        stack.Push(context);
        return new OperationScope(() => stack.Pop());
    }

    public bool IsInLoop(string serverId, string toolName, int maxDepth = 10)
    {
        var stack = _operationStack.Value;
        if (stack == null) return false;

        var matchingOperations = stack.Count(ctx =>
            ctx.ServerId == serverId && ctx.ToolName == toolName);

        return matchingOperations >= maxDepth;
    }

    private class OperationContext
    {
        public string OperationId { get; set; }
        public string ServerId { get; set; }
        public string ToolName { get; set; }
        public DateTime Timestamp { get; set; }
    }

    private class OperationScope : IDisposable
    {
        private readonly Action _onDispose;

        public OperationScope(Action onDispose) => _onDispose = onDispose;
        public void Dispose() => _onDispose();
    }
}
```

## Testing Framework

### Unit Testing
```csharp
[TestClass]
public class McpClientServiceTests
{
    private Mock<IMcpConnectionManager> _connectionManagerMock;
    private Mock<IMcpToolDiscoveryService> _toolDiscoveryMock;
    private McpClientService _clientService;

    [TestInitialize]
    public void Setup()
    {
        _connectionManagerMock = new Mock<IMcpConnectionManager>();
        _toolDiscoveryMock = new Mock<IMcpToolDiscoveryService>();
        _clientService = new McpClientService(
            _connectionManagerMock.Object,
            _toolDiscoveryMock.Object);
    }

    [TestMethod]
    public async Task InvokeToolAsync_ValidRequest_ReturnsResult()
    {
        // Arrange
        var expectedResult = new ToolResult { Success = true };
        _toolDiscoveryMock
            .Setup(s => s.IsToolAvailableAsync("server1", "testTool"))
            .ReturnsAsync(true);

        // Act
        var result = await _clientService.InvokeToolAsync(
            "server1", "testTool", new Dictionary<string, object>());

        // Assert
        Assert.IsNotNull(result);
        Assert.IsTrue(result.Success);
    }
}
```

### Integration Testing
```csharp
[TestClass]
public class McpIntegrationTests : TestServerFixture
{
    [TestMethod]
    public async Task EndToEnd_ToolInvocation_Success()
    {
        // Arrange
        var client = CreateMcpClient();

        // Act
        var result = await client.InvokeToolAsync(
            "testServer",
            "echo",
            new Dictionary<string, object> { ["message"] = "test" });

        // Assert
        Assert.IsTrue(result.Success);
        Assert.AreEqual("test", result.Content?["message"]);
    }
}
```

### Load Testing
```csharp
[TestClass]
public class McpLoadTests
{
    [TestMethod]
    public async Task HighConcurrency_LoadTest()
    {
        // Arrange
        var clients = Enumerable.Range(0, 100)
            .Select(_ => CreateMcpClient())
            .ToArray();

        var requests = clients.Select(client =>
            client.InvokeToolAsync("server", "testTool", new Dictionary<string, object>()));

        // Act
        var stopwatch = Stopwatch.StartNew();
        var results = await Task.WhenAll(requests);
        stopwatch.Stop();

        // Assert
        Assert.AreEqual(100, results.Length);
        Assert.IsTrue(results.All(r => r.Success));
        Assert.IsTrue(stopwatch.Elapsed < TimeSpan.FromSeconds(30));
    }
}
```

This specification provides comprehensive details for implementing a robust MCP client within the Hartonomous-Opus system, including transport implementations, resilience patterns, tool augmentation, and comprehensive testing strategies.