# MCP Testing Strategy

## Overview
This document outlines the comprehensive testing strategy for MCP server and client functionality in the Hartonomous-Opus system. Testing covers unit tests, integration tests, performance tests, security tests, and end-to-end validation.

## Testing Pyramid

```
End-to-End Tests (E2E)
    ↕️
Integration Tests
    ↕️
Component Tests
    ↕️
Unit Tests
```

## Unit Testing Strategy

### Test Categories

#### 1. Tool Implementation Tests
**Scope**: Individual MCP tool classes
**Coverage**: Parameter validation, business logic, error handling

**Example Test Structure**:
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
        var user = CreateMockUser();
        SetupValidAuthorization();
        SetupValidRateLimit();
        SetupSemanticServiceResponse();

        // Act
        var result = await _tool.ExecuteAsync(user.Object, "test query", 10, 0.1);

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("test query", result.Query);
        VerifySemanticServiceCalled();
    }

    [TestMethod]
    [ExpectedException(typeof(McpException))]
    public async Task ExecuteAsync_Unauthorized_ThrowsException()
    {
        // Arrange
        var user = CreateMockUser();
        SetupUnauthorizedAccess();

        // Act & Assert
        await _tool.ExecuteAsync(user.Object, "test query", 10, 0.1);
    }

    [TestMethod]
    public async Task ExecuteAsync_RateLimitExceeded_ThrowsException()
    {
        // Arrange
        var user = CreateMockUser();
        SetupValidAuthorization();
        SetupRateLimitExceeded();

        // Act & Assert
        var exception = await Assert.ThrowsExceptionAsync<McpException>(
            () => _tool.ExecuteAsync(user.Object, "test query", 10, 0.1));
        Assert.AreEqual("Rate limit exceeded", exception.Message);
    }

    [TestMethod]
    [DataRow("", DisplayName = "Empty query")]
    [DataRow(null, DisplayName = "Null query")]
    [DataRow("a", DisplayName = "Query too short")]
    [ExpectedException(typeof(McpException))]
    public async Task ExecuteAsync_InvalidQuery_ThrowsException(string invalidQuery)
    {
        // Arrange
        var user = CreateMockUser();
        SetupValidAuthorization();
        SetupValidRateLimit();

        // Act & Assert
        await _tool.ExecuteAsync(user.Object, invalidQuery, 10, 0.1);
    }

    [TestMethod]
    [DataRow(-1, DisplayName = "Negative limit")]
    [DataRow(0, DisplayName = "Zero limit")]
    [DataRow(101, DisplayName = "Limit too high")]
    [ExpectedException(typeof(McpException))]
    public async Task ExecuteAsync_InvalidLimit_ThrowsException(int invalidLimit)
    {
        // Arrange
        var user = CreateMockUser();
        SetupValidAuthorization();
        SetupValidRateLimit();

        // Act & Assert
        await _tool.ExecuteAsync(user.Object, "test query", invalidLimit, 0.1);
    }

    [TestMethod]
    [DataRow(-0.1, DisplayName = "Negative threshold")]
    [DataRow(1.1, DisplayName = "Threshold too high")]
    [ExpectedException(typeof(McpException))]
    public async Task ExecuteAsync_InvalidThreshold_ThrowsException(double invalidThreshold)
    {
        // Arrange
        var user = CreateMockUser();
        SetupValidAuthorization();
        SetupValidRateLimit();

        // Act & Assert
        await _tool.ExecuteAsync(user.Object, "test query", 10, invalidThreshold);
    }
}
```

#### 2. Service Layer Tests
**Scope**: MCP services (server, client, transport)
**Coverage**: Service initialization, configuration, error handling

#### 3. Middleware Tests
**Scope**: Authentication, authorization, rate limiting middleware
**Coverage**: Request processing, security enforcement

#### 4. Validation Tests
**Scope**: Input validation, sanitization, constraint checking
**Coverage**: Parameter validation, content filtering

## Integration Testing Strategy

### Test Environment Setup
```csharp
public class McpIntegrationTestFixture : IDisposable
{
    private readonly TestServer _testServer;
    private readonly HttpClient _client;
    private readonly IServiceScope _scope;

    public McpIntegrationTestFixture()
    {
        var builder = WebApplication.CreateBuilder();
        // Configure test services
        builder.Services.AddMcpServer();
        builder.Services.AddMcpToolsFromAssembly(typeof(TestMcpTools).Assembly);

        var app = builder.Build();
        app.MapMcp("/mcp");

        _testServer = new TestServer(app);
        _client = _testServer.CreateClient();
        _scope = app.Services.CreateScope();
    }

    public async Task<McpResponse> SendMcpRequestAsync(McpRequest request)
    {
        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await _client.PostAsync("/mcp", content);
        var responseJson = await response.Content.ReadAsStringAsync();

        return JsonSerializer.Deserialize<McpResponse>(responseJson);
    }

    public T GetService<T>() => _scope.ServiceProvider.GetRequiredService<T>();

    public void Dispose()
    {
        _scope?.Dispose();
        _testServer?.Dispose();
        _client?.Dispose();
    }
}
```

### Server Integration Tests
```csharp
[TestClass]
public class McpServerIntegrationTests : IClassFixture<McpIntegrationTestFixture>
{
    private readonly McpIntegrationTestFixture _fixture;

    public McpServerIntegrationTests(McpIntegrationTestFixture fixture)
    {
        _fixture = fixture;
    }

    [TestMethod]
    public async Task Initialize_ServerStartsSuccessfully()
    {
        // Arrange & Act
        var serverInfo = await GetServerInfoAsync();

        // Assert
        Assert.IsNotNull(serverInfo);
        Assert.AreEqual("Hartonomous-Opus Hypercube", serverInfo.Name);
        Assert.IsTrue(serverInfo.Capabilities.Tools);
    }

    [TestMethod]
    public async Task ListTools_ReturnsAvailableTools()
    {
        // Arrange & Act
        var tools = await ListToolsAsync();

        // Assert
        Assert.IsNotNull(tools);
        Assert.IsTrue(tools.Length > 0);
        Assert.IsTrue(tools.Any(t => t.Name == "SemanticSearch"));
        Assert.IsTrue(tools.Any(t => t.Name == "GenerateText"));
    }

    [TestMethod]
    public async Task CallTool_SemanticSearch_Success()
    {
        // Arrange
        var request = new McpRequest
        {
            Id = Guid.NewGuid().ToString(),
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
        };

        // Act
        var response = await _fixture.SendMcpRequestAsync(request);

        // Assert
        Assert.IsTrue(response.Success);
        var result = response.Result as SemanticSearchResult;
        Assert.IsNotNull(result);
        Assert.AreEqual("artificial intelligence", result.Query);
        Assert.IsTrue(result.TotalFound >= 0);
    }

    [TestMethod]
    public async Task CallTool_Unauthenticated_ReturnsError()
    {
        // Arrange - no auth header
        var request = new McpRequest
        {
            Id = Guid.NewGuid().ToString(),
            Method = "tools/call",
            Params = new ToolCallRequest
            {
                Name = "SemanticSearch",
                Arguments = new Dictionary<string, object>
                {
                    ["query"] = "test"
                }
            }
        };

        // Act
        var response = await _fixture.SendMcpRequestAsync(request);

        // Assert
        Assert.IsFalse(response.Success);
        Assert.AreEqual(McpErrorCodes.Unauthorized, response.Error.Code);
    }

    [TestMethod]
    public async Task CallTool_RateLimitExceeded_ReturnsError()
    {
        // Arrange - make multiple rapid calls
        var requests = Enumerable.Range(0, 100).Select(_ => CreateRateLimitTestRequest());

        // Act
        var responses = await Task.WhenAll(requests.Select(r => _fixture.SendMcpRequestAsync(r)));

        // Assert
        var rateLimitedResponses = responses.Where(r => !r.Success &&
            r.Error.Code == McpErrorCodes.ServerError &&
            r.Error.Message.Contains("rate limit")).ToArray();

        Assert.IsTrue(rateLimitedResponses.Length > 0);
    }

    [TestMethod]
    public async Task CallTool_InvalidParameters_ReturnsError()
    {
        // Arrange
        var request = new McpRequest
        {
            Id = Guid.NewGuid().ToString(),
            Method = "tools/call",
            Params = new ToolCallRequest
            {
                Name = "SemanticSearch",
                Arguments = new Dictionary<string, object>
                {
                    ["query"] = "", // Invalid: empty query
                    ["limit"] = -1 // Invalid: negative limit
                }
            }
        };

        // Act
        var response = await _fixture.SendMcpRequestAsync(request);

        // Assert
        Assert.IsFalse(response.Success);
        Assert.AreEqual(McpErrorCodes.InvalidParams, response.Error.Code);
    }

    [TestMethod]
    public async Task CallTool_UnknownTool_ReturnsError()
    {
        // Arrange
        var request = new McpRequest
        {
            Id = Guid.NewGuid().ToString(),
            Method = "tools/call",
            Params = new ToolCallRequest
            {
                Name = "UnknownTool",
                Arguments = new Dictionary<string, object>()
            }
        };

        // Act
        var response = await _fixture.SendMcpRequestAsync(request);

        // Assert
        Assert.IsFalse(response.Success);
        Assert.AreEqual(McpErrorCodes.MethodNotFound, response.Error.Code);
    }

    private async Task<ServerInfo> GetServerInfoAsync()
    {
        var request = new McpRequest { Method = "initialize" };
        var response = await _fixture.SendMcpRequestAsync(request);
        return response.Result as ServerInfo;
    }

    private async Task<Tool[]> ListToolsAsync()
    {
        var request = new McpRequest { Method = "tools/list" };
        var response = await _fixture.SendMcpRequestAsync(request);
        var result = response.Result as ToolListResult;
        return result.Tools;
    }

    private McpRequest CreateRateLimitTestRequest() => new McpRequest
    {
        Id = Guid.NewGuid().ToString(),
        Method = "tools/call",
        Params = new ToolCallRequest
        {
            Name = "SemanticSearch",
            Arguments = new Dictionary<string, object> { ["query"] = "rate limit test" }
        }
    };
}
```

### Client Integration Tests
```csharp
[TestClass]
public class McpClientIntegrationTests : IClassFixture<McpClientTestFixture>
{
    private readonly McpClientTestFixture _fixture;

    public McpClientIntegrationTests(McpClientTestFixture fixture)
    {
        _fixture = fixture;
    }

    [TestMethod]
    public async Task Connect_ValidServer_Success()
    {
        // Arrange
        var serverId = "testServer";

        // Act
        await _fixture.ClientService.ConnectAsync(serverId);

        // Assert
        Assert.IsTrue(await _fixture.ClientService.IsConnectedAsync(serverId));
    }

    [TestMethod]
    public async Task GetAvailableTools_ReturnsTools()
    {
        // Arrange
        var serverId = "testServer";
        await _fixture.ClientService.ConnectAsync(serverId);

        // Act
        var tools = await _fixture.ClientService.GetAvailableToolsAsync(serverId);

        // Assert
        Assert.IsNotNull(tools);
        Assert.IsTrue(tools.Any());
    }

    [TestMethod]
    public async Task InvokeTool_SemanticSearch_Success()
    {
        // Arrange
        var serverId = "testServer";
        await _fixture.ClientService.ConnectAsync(serverId);

        // Act
        var result = await _fixture.ClientService.InvokeToolAsync(
            serverId,
            "SemanticSearch",
            new Dictionary<string, object>
            {
                ["query"] = "test query",
                ["limit"] = 5
            });

        // Assert
        Assert.IsNotNull(result);
        Assert.IsTrue(result.Success);
    }

    [TestMethod]
    public async Task InvokeTool_NetworkFailure_RetrySuccess()
    {
        // Arrange - simulate network failures
        var serverId = "unreliableServer";
        await _fixture.ClientService.ConnectAsync(serverId);

        // Act
        var result = await _fixture.ClientService.InvokeToolAsync(
            serverId,
            "TestTool",
            new Dictionary<string, object>());

        // Assert
        Assert.IsNotNull(result);
        // Verify retries occurred and succeeded
    }

    [TestMethod]
    [ExpectedException(typeof(McpException))]
    public async Task InvokeTool_ServerUnavailable_ThrowsException()
    {
        // Arrange
        var serverId = "offlineServer";

        // Act
        await _fixture.ClientService.InvokeToolAsync(
            serverId,
            "TestTool",
            new Dictionary<string, object>());
    }
}
```

## Performance Testing Strategy

### Load Testing Framework
```csharp
[TestClass]
public class McpPerformanceTests
{
    private const int ConcurrentUsers = 100;
    private const int RequestsPerUser = 10;
    private const int MaxResponseTimeMs = 5000;

    [TestMethod]
    public async Task HighConcurrency_LoadTest()
    {
        // Arrange
        var clients = await CreateConcurrentClientsAsync(ConcurrentUsers);
        var testDuration = TimeSpan.FromMinutes(5);

        // Act
        var results = await RunLoadTestAsync(clients, testDuration);

        // Assert
        AssertPerformanceMetrics(results, ConcurrentUsers * RequestsPerUser);
    }

    [TestMethod]
    public async Task SpikeTest_SuddenLoadIncrease()
    {
        // Arrange
        var baselineClients = await CreateConcurrentClientsAsync(10);

        // Act - Phase 1: Baseline load
        var baselineResults = await RunLoadTestAsync(baselineClients, TimeSpan.FromMinutes(1));

        // Phase 2: Spike load
        var spikeClients = await CreateConcurrentClientsAsync(50);
        var spikeResults = await RunLoadTestAsync(spikeClients, TimeSpan.FromMinutes(1));

        // Assert
        AssertSpikePerformance(baselineResults, spikeResults);
    }

    [TestMethod]
    public async Task EnduranceTest_LongRunningLoad()
    {
        // Arrange
        var clients = await CreateConcurrentClientsAsync(20);

        // Act
        var results = await RunLoadTestAsync(clients, TimeSpan.FromHours(1));

        // Assert
        AssertEndurancePerformance(results);
    }

    private async Task<List<PerformanceResult>> RunLoadTestAsync(
        List<IMcpClient> clients, TimeSpan duration)
    {
        var results = new ConcurrentBag<PerformanceResult>();
        var tasks = new List<Task>();

        using var cts = new CancellationTokenSource(duration);

        foreach (var client in clients)
        {
            tasks.Add(Task.Run(async () =>
            {
                while (!cts.Token.IsCancellationRequested)
                {
                    var startTime = DateTime.UtcNow;

                    try
                    {
                        var result = await client.SendRequestAsync(CreateTestRequest());
                        var responseTime = DateTime.UtcNow - startTime;

                        results.Add(new PerformanceResult
                        {
                            Success = result.Success,
                            ResponseTime = responseTime,
                            ErrorType = result.Success ? null : result.Error?.Code
                        });
                    }
                    catch (Exception ex)
                    {
                        var responseTime = DateTime.UtcNow - startTime;
                        results.Add(new PerformanceResult
                        {
                            Success = false,
                            ResponseTime = responseTime,
                            ErrorType = ex.GetType().Name
                        });
                    }

                    await Task.Delay(100); // Rate limiting
                }
            }, cts.Token));
        }

        await Task.WhenAll(tasks);
        return results.ToList();
    }

    private void AssertPerformanceMetrics(List<PerformanceResult> results, int expectedRequests)
    {
        var totalRequests = results.Count;
        var successfulRequests = results.Count(r => r.Success);
        var successRate = (double)successfulRequests / totalRequests;

        var avgResponseTime = results.Average(r => r.ResponseTime.TotalMilliseconds);
        var p95ResponseTime = CalculatePercentile(results.Select(r => r.ResponseTime.TotalMilliseconds), 95);
        var p99ResponseTime = CalculatePercentile(results.Select(r => r.ResponseTime.TotalMilliseconds), 99);

        // Assert success rate
        Assert.IsTrue(successRate >= 0.95, $"Success rate too low: {successRate:P2}");

        // Assert response times
        Assert.IsTrue(avgResponseTime < MaxResponseTimeMs, $"Average response time too high: {avgResponseTime}ms");
        Assert.IsTrue(p95ResponseTime < MaxResponseTimeMs * 2, $"P95 response time too high: {p95ResponseTime}ms");

        // Assert error distribution
        var errorsByType = results.Where(r => !r.Success)
            .GroupBy(r => r.ErrorType)
            .ToDictionary(g => g.Key, g => g.Count());

        Assert.IsFalse(errorsByType.ContainsKey("TimeoutException"), "Timeouts should not occur under normal load");
    }

    private double CalculatePercentile(IEnumerable<double> values, double percentile)
    {
        var sortedValues = values.OrderBy(v => v).ToArray();
        var index = (int)Math.Ceiling((percentile / 100.0) * sortedValues.Length) - 1;
        return sortedValues[Math.Max(0, Math.Min(index, sortedValues.Length - 1))];
    }

    private McpRequest CreateTestRequest() => new McpRequest
    {
        Id = Guid.NewGuid().ToString(),
        Method = "tools/call",
        Params = new ToolCallRequest
        {
            Name = "SemanticSearch",
            Arguments = new Dictionary<string, object>
            {
                ["query"] = "performance test query",
                ["limit"] = 5
            }
        }
    };
}

public class PerformanceResult
{
    public bool Success { get; set; }
    public TimeSpan ResponseTime { get; set; }
    public string ErrorType { get; set; }
}
```

### Memory Leak Testing
```csharp
[TestClass]
public class McpMemoryLeakTests
{
    [TestMethod]
    public async Task LongRunningOperations_NoMemoryLeaks()
    {
        // Arrange
        var initialMemory = GC.GetTotalMemory(true);
        var iterations = 1000;

        // Act
        for (int i = 0; i < iterations; i++)
        {
            using var client = CreateMcpClient();
            await client.SendRequestAsync(CreateTestRequest());

            if (i % 100 == 0)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        // Allow for some memory growth but not unbounded
        var finalMemory = GC.GetTotalMemory(true);
        var memoryGrowth = finalMemory - initialMemory;
        var acceptableGrowth = 10 * 1024 * 1024; // 10MB

        // Assert
        Assert.IsTrue(memoryGrowth < acceptableGrowth,
            $"Memory growth too high: {memoryGrowth / 1024.0 / 1024.0:F2}MB");
    }
}
```

## Security Testing Strategy

### Penetration Testing
```csharp
[TestClass]
public class McpSecurityTests
{
    [TestMethod]
    [DataRow("../../../etc/passwd", DisplayName = "Path traversal")]
    [DataRow("<script>alert('xss')</script>", DisplayName = "XSS attempt")]
    [DataRow("'; DROP TABLE users; --", DisplayName = "SQL injection")]
    [ExpectedException(typeof(McpException))]
    public async Task CallTool_MaliciousInput_Rejected(string maliciousInput)
    {
        // Arrange
        var request = new McpRequest
        {
            Method = "tools/call",
            Params = new ToolCallRequest
            {
                Name = "SemanticSearch",
                Arguments = new Dictionary<string, object>
                {
                    ["query"] = maliciousInput
                }
            }
        };

        // Act & Assert
        await SendAuthenticatedRequestAsync(request);
    }

    [TestMethod]
    public async Task Authentication_BruteForce_AccountLocked()
    {
        // Arrange
        var invalidCredentials = new[]
        {
            ("invalid", "password1"),
            ("invalid", "password2"),
            ("invalid", "password3")
        };

        // Act
        foreach (var (username, password) in invalidCredentials)
        {
            await SendRequestWithCredentialsAsync(username, password, CreateTestRequest());
        }

        // Assert
        var finalRequest = await SendRequestWithCredentialsAsync("invalid", "password4", CreateTestRequest());
        Assert.IsFalse(finalRequest.Success);
        Assert.AreEqual(McpErrorCodes.Unauthorized, finalRequest.Error.Code);
        // Verify account is locked
    }

    [TestMethod]
    public async Task RateLimit_BypassAttempt_Detected()
    {
        // Arrange - attempt various bypass techniques
        var bypassAttempts = new[]
        {
            CreateRateLimitBypassRequest("X-Forwarded-For", "1.2.3.4"),
            CreateRateLimitBypassRequest("X-Real-IP", "1.2.3.4"),
            CreateRateLimitBypassRequest("CF-Connecting-IP", "1.2.3.4")
        };

        // Act
        var responses = await Task.WhenAll(bypassAttempts.Select(SendAuthenticatedRequestAsync));

        // Assert - all should be rate limited
        Assert.IsTrue(responses.All(r => !r.Success));
        Assert.IsTrue(responses.All(r => r.Error.Message.Contains("rate limit")));
    }

    [TestMethod]
    public async Task Encryption_InTransit_Enforced()
    {
        // Arrange - attempt HTTP connection to HTTPS-only endpoint
        var httpClient = new HttpClient();

        // Act & Assert
        await Assert.ThrowsExceptionAsync<HttpRequestException>(
            () => httpClient.PostAsync("http://localhost/mcp", CreateTestContent()));
    }

    [TestMethod]
    public async Task CertificateValidation_InvalidCert_Rejected()
    {
        // Arrange - use self-signed certificate
        var handler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (message, cert, chain, errors) => false
        };
        var client = new HttpClient(handler);

        // Act & Assert
        var exception = await Assert.ThrowsExceptionAsync<HttpRequestException>(
            () => client.PostAsync("https://localhost/mcp", CreateTestContent()));
        Assert.IsTrue(exception.Message.Contains("certificate"));
    }
}
```

### Fuzz Testing
```csharp
[TestClass]
public class McpFuzzTests
{
    [TestMethod]
    public async Task Fuzz_ToolParameters_RobustHandling()
    {
        // Arrange
        var fuzzer = new ParameterFuzzer();
        var iterations = 1000;

        // Act & Assert - No exceptions should be thrown
        for (int i = 0; i < iterations; i++)
        {
            var parameters = fuzzer.GenerateRandomParameters();
            try
            {
                var result = await CallToolWithParametersAsync("SemanticSearch", parameters);
                // Result can be error, but no exceptions
            }
            catch (Exception ex)
            {
                Assert.Fail($"Unexpected exception on iteration {i}: {ex.Message}");
            }
        }
    }

    [TestMethod]
    public async Task Fuzz_JSONPayload_RobustParsing()
    {
        // Arrange
        var jsonFuzzer = new JsonFuzzer();
        var iterations = 500;

        // Act & Assert
        for (int i = 0; i < iterations; i++)
        {
            var malformedJson = jsonFuzzer.GenerateMalformedJson();
            try
            {
                var result = await SendRawJsonAsync(malformedJson);
                // Should return proper error, not crash
                Assert.IsFalse(result.Success);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Unexpected exception parsing malformed JSON: {ex.Message}");
            }
        }
    }
}

public class ParameterFuzzer
{
    private readonly Random _random = new Random();

    public Dictionary<string, object> GenerateRandomParameters()
    {
        var parameters = new Dictionary<string, object>();

        // Random string lengths
        parameters["query"] = GenerateRandomString(_random.Next(0, 10000));

        // Random numbers
        parameters["limit"] = _random.Next(-1000, 1000);
        parameters["threshold"] = _random.NextDouble() * 2 - 0.5; // -0.5 to 1.5

        // Random nested objects
        if (_random.Next(2) == 1)
        {
            parameters["nested"] = new Dictionary<string, object>
            {
                ["deep"] = GenerateRandomString(100),
                ["number"] = _random.Next()
            };
        }

        return parameters;
    }

    private string GenerateRandomString(int length)
    {
        const string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()";
        return new string(Enumerable.Repeat(chars, length)
            .Select(s => s[_random.Next(s.Length)]).ToArray());
    }
}
```

## End-to-End Testing Strategy

### Full System Tests
```csharp
[TestClass]
public class McpEndToEndTests
{
    [TestMethod]
    public async Task CompleteWorkflow_SemanticAnalysisToGeneration()
    {
        // Arrange - Set up MCP server and client
        using var serverFixture = new McpServerTestFixture();
        using var clientFixture = new McpClientTestFixture();

        await serverFixture.StartServerAsync();
        await clientFixture.ConnectToServerAsync(serverFixture.ServerUrl);

        // Act - Complete semantic analysis workflow
        // 1. Ingest content
        var ingestResult = await clientFixture.CallToolAsync("IngestContent", new Dictionary<string, object>
        {
            ["content"] = "Artificial intelligence is transforming technology.",
            ["contentType"] = "text"
        });
        Assert.IsTrue(ingestResult.Success);

        // 2. Perform semantic search
        var searchResult = await clientFixture.CallToolAsync("SemanticSearch", new Dictionary<string, object>
        {
            ["query"] = "artificial intelligence",
            ["limit"] = 5
        });
        Assert.IsTrue(searchResult.Success);

        // 3. Generate related content
        var generationResult = await clientFixture.CallToolAsync("GenerateText", new Dictionary<string, object>
        {
            ["prompt"] = "The impact of AI on society",
            ["maxTokens"] = 100
        });
        Assert.IsTrue(generationResult.Success);

        // 4. Find analogies
        var analogyResult = await clientFixture.CallToolAsync("FindAnalogies", new Dictionary<string, object>
        {
            ["termA"] = "AI",
            ["termB"] = "electricity",
            ["termC"] = "society"
        });
        Assert.IsTrue(analogyResult.Success);

        // Assert - Verify end-to-end consistency
        var searchData = searchResult.Result as SemanticSearchResult;
        var generationData = generationResult.Result as GenerationResult;
        var analogyData = analogyResult.Result as AnalogyResult[];

        Assert.IsNotNull(searchData);
        Assert.IsNotNull(generationData);
        Assert.IsNotNull(analogyData);

        // Verify semantic relationships
        Assert.IsTrue(searchData.TotalFound > 0);
        Assert.IsFalse(string.IsNullOrEmpty(generationData.Text));
        Assert.IsTrue(analogyData.Length > 0);
    }

    [TestMethod]
    public async Task MultiClient_ConcurrentAccess_Consistency()
    {
        // Arrange
        using var serverFixture = new McpServerTestFixture();
        await serverFixture.StartServerAsync();

        var clients = await Task.WhenAll(
            Enumerable.Range(0, 10).Select(_ => CreateClientAndConnectAsync(serverFixture.ServerUrl)));

        // Act - All clients perform same operation simultaneously
        var tasks = clients.Select(client => client.CallToolAsync("SemanticSearch", new Dictionary<string, object>
        {
            ["query"] = "concurrent access test",
            ["limit"] = 5
        }));

        var results = await Task.WhenAll(tasks);

        // Assert - All results should be consistent
        Assert.IsTrue(results.All(r => r.Success));
        var firstResult = results.First().Result as SemanticSearchResult;

        foreach (var result in results.Skip(1))
        {
            var resultData = result.Result as SemanticSearchResult;
            Assert.AreEqual(firstResult.TotalFound, resultData.TotalFound);
            // Additional consistency checks
        }
    }

    [TestMethod]
    public async Task FailureRecovery_AutomaticRetry_Success()
    {
        // Arrange - Set up server that fails occasionally
        using var unreliableServer = new UnreliableMcpServerFixture();
        using var clientFixture = new McpClientTestFixture();

        await unreliableServer.StartServerAsync();
        await clientFixture.ConnectToServerAsync(unreliableServer.ServerUrl);

        // Configure client for retries
        clientFixture.ConfigureRetryPolicy(maxRetries: 3, backoffSeconds: 1);

        // Act - Make requests that may fail
        var results = new List<ToolResult>();
        for (int i = 0; i < 10; i++)
        {
            var result = await clientFixture.CallToolAsync("UnreliableTool", new Dictionary<string, object>());
            results.Add(result);
        }

        // Assert - Despite failures, most requests should succeed due to retries
        var successCount = results.Count(r => r.Success);
        Assert.IsTrue(successCount >= 7); // At least 70% success rate
    }

    [TestMethod]
    public async Task ResourceCleanup_AfterTest_NoMemoryLeaks()
    {
        // Arrange
        var initialMemory = GC.GetTotalMemory(true);

        // Act - Run multiple E2E tests
        await CompleteWorkflow_SemanticAnalysisToGeneration();
        await MultiClient_ConcurrentAccess_Consistency();

        // Force garbage collection
        GC.Collect();
        GC.WaitForPendingFinalizers();

        // Assert - Memory usage should return to baseline
        var finalMemory = GC.GetTotalMemory(true);
        var memoryGrowth = finalMemory - initialMemory;
        var acceptableGrowth = 50 * 1024 * 1024; // 50MB acceptable for test framework

        Assert.IsTrue(memoryGrowth < acceptableGrowth,
            $"Memory leak detected: {memoryGrowth / 1024.0 / 1024.0:F2}MB growth");
    }
}
```

## Test Automation and CI/CD

### Continuous Integration Pipeline
```yaml
# .github/workflows/mcp-tests.yml
name: MCP Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup .NET
      uses: actions/setup-dotnet@v3
      with:
        dotnet-version: 8.0.x
    - name: Restore dependencies
      run: dotnet restore
    - name: Run unit tests
      run: dotnet test --filter "TestCategory=Unit" --logger "trx;LogFileName=unit-tests.trx"
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: unit-test-results
        path: "**/unit-tests.trx"

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
    - uses: actions/checkout@v3
    - name: Setup .NET
      uses: actions/setup-dotnet@v3
      with:
        dotnet-version: 8.0.x
    - name: Setup database
      run: |
        dotnet tool install --global dotnet-ef
        dotnet ef database update
    - name: Run integration tests
      run: dotnet test --filter "TestCategory=Integration" --logger "trx;LogFileName=integration-tests.trx"
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: integration-test-results
        path: "**/integration-tests.trx"

  performance-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
    steps:
    - uses: actions/checkout@v3
    - name: Setup .NET
      uses: actions/setup-dotnet@v3
      with:
        dotnet-version: 8.0.x
    - name: Run performance tests
      run: dotnet test --filter "TestCategory=Performance"
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: "performance-results.json"

  security-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup .NET
      uses: actions/setup-dotnet@v3
      with:
        dotnet-version: 8.0.x
    - name: Run security tests
      run: dotnet test --filter "TestCategory=Security"
    - name: Run vulnerability scan
      uses: github/super-linter/slim@v5
      env:
        DEFAULT_BRANCH: main
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  e2e-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
    steps:
    - uses: actions/checkout@v3
    - name: Setup .NET
      uses: actions/setup-dotnet@v3
      with:
        dotnet-version: 8.0.x
    - name: Run E2E tests
      run: dotnet test --filter "TestCategory=E2E"
    - name: Upload E2E results
      uses: actions/upload-artifact@v3
      with:
        name: e2e-test-results
        path: "**/e2e-tests.trx"
```

### Test Data Management
```csharp
public static class TestDataFactory
{
    private static readonly Random _random = new Random(42); // Deterministic seed

    public static string GenerateRandomText(int minLength = 10, int maxLength = 100)
    {
        const string words = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua";
        var wordList = words.Split(' ');
        var length = _random.Next(minLength, maxLength);
        var result = new List<string>();

        for (int i = 0; i < length; i++)
        {
            result.Add(wordList[_random.Next(wordList.Length)]);
        }

        return string.Join(" ", result);
    }

    public static SemanticTestData GenerateSemanticTestData()
    {
        return new SemanticTestData
        {
            Query = GenerateRandomText(3, 10),
            ExpectedResults = _random.Next(0, 20),
            Threshold = _random.NextDouble() * 0.5 + 0.1, // 0.1 to 0.6
            Domain = _random.Next(2) == 0 ? "general" : "technical"
        };
    }

    public static GenerationTestData GenerateGenerationTestData()
    {
        return new GenerationTestData
        {
            Prompt = GenerateRandomText(5, 15),
            MaxTokens = _random.Next(50, 200),
            Temperature = _random.NextDouble() * 1.5 + 0.1, // 0.1 to 1.6
            ExpectedLength = _random.Next(20, 150)
        };
    }
}

public class SemanticTestData
{
    public string Query { get; set; }
    public int ExpectedResults { get; set; }
    public double Threshold { get; set; }
    public string Domain { get; set; }
}

public class GenerationTestData
{
    public string Prompt { get; set; }
    public int MaxTokens { get; set; }
    public double Temperature { get; set; }
    public int ExpectedLength { get; set; }
}
```

This comprehensive testing strategy ensures that MCP server and client functionality is thoroughly validated across all dimensions: functionality, performance, security, and reliability. The strategy includes automated testing, comprehensive coverage, and continuous validation throughout the development lifecycle.