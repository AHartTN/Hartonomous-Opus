# OpenAI-Compatible API for Hartonomous-Opus Hypercube Generative Engine

## Executive Summary

This document outlines the implementation of an OpenAI-compatible REST API for the Hartonomous-Opus hypercube generative engine. The solution uses a C# ASP.NET Core orchestrator that interops with the high-performance C++ generative engine, enabling seamless integration with existing LLM tooling while maintaining the unique 4D semantic substrate capabilities.

## Architecture Overview

### Core Components

```
┌─────────────────┐    HTTP/JSON    ┌──────────────────────┐
│   Client Apps   │◄──────────────►│  C# ASP.NET Core API │
│ (OpenAI compat) │                 │     Orchestrator     │
└─────────────────┘                 └──────────────────────┘
                                           │
                                           │ P/Invoke Interop
                                           ▼
┌─────────────────┐    C API        ┌──────────────────────┐
│  C++ Generative │◄──────────────►│  hypercube_generative │
│     Engine      │                 │         DLL          │
└─────────────────┘                 └──────────────────────┘
                                           │
                                           │ PostgreSQL/libpq
                                           ▼
┌─────────────────┐                        ┌──────────────────────┐
│   PostgreSQL    │◄──────────────────────►│  4D Hypercube DB     │
│                 │                        │  (Atoms, Compositions│
│                 │                        │   Relations)         │
└─────────────────┘                        └──────────────────────┘
```

### Technology Stack

- **Orchestrator**: ASP.NET Core 8+ (C#) - High-performance web API
- **Engine**: C++20 with PostgreSQL extensions - Performance-critical generation
- **Interop**: P/Invoke - Native C# to C++ bridging
- **Database**: PostgreSQL 18+ with PostGIS - 4D geometric storage
- **Serialization**: System.Text.Json - OpenAI-compatible JSON
- **Future AST**: TreeSitter (C++), Roslyn (.NET) - Code understanding

## OpenAI API Compatibility

### Supported Endpoints

- `POST /v1/completions` - Text completion generation
- `GET /v1/models` - List available models
- `GET /health` - Health check endpoint

### Request/Response Format

**Request (POST /v1/completions)**:
```json
{
  "model": "hypercube-generative",
  "prompt": "The meaning of life",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 1.0,
  "stop": ["\n", "."],
  "stream": false
}
```

**Response**:
```json
{
  "id": "gen-hc-abc123",
  "object": "text_completion",
  "created": 1677652288,
  "model": "hypercube-generative",
  "choices": [{
    "text": " is to understand the universe through geometric relationships and semantic connections.",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 12,
    "total_tokens": 17
  }
}
```

### Parameter Mapping

| OpenAI Parameter | Hypercube Mapping | Notes |
|------------------|-------------------|-------|
| `model` | Fixed to "hypercube-generative" | Single model for now |
| `prompt` | Tokenized to start labels | Last token used for generation |
| `max_tokens` | Direct mapping | Generation length limit |
| `temperature` | Stochastic vs greedy selection | 0 = greedy, >0 = temperature sampling |
| `top_p` | Not implemented | Future enhancement |
| `stop` | Stop token sequences | Match against generated tokens |
| `stream` | Not implemented | Future enhancement for SSE |

## C# Orchestrator Design

### Project Structure

```
csharp/
├── HypercubeGenerativeApi/
│   ├── HypercubeGenerativeApi.csproj
│   ├── Program.cs                    # App startup + DI
│   ├── Controllers/
│   │   ├── CompletionsController.cs  # OpenAI /v1/completions
│   │   └── ModelsController.cs       # /v1/models
│   ├── Models/
│   │   ├── CompletionRequest.cs      # OpenAI request DTOs
│   │   ├── CompletionResponse.cs     # OpenAI response DTOs
│   │   └── HypercubeModels.cs        # Internal types
│   ├── Services/
│   │   ├── GenerativeService.cs      # Business logic orchestrator
│   │   └── PostgresService.cs        # DB connection management
│   ├── Interop/
│   │   ├── GenerativeInterop.cs      # P/Invoke declarations
│   │   ├── InteropTypes.cs           # Marshalled structs
│   │   └── NativeMethods.cs          # Helper methods
│   └── appsettings.json              # Configuration
└── HypercubeAst/                     # Future AST processing
    ├── TreeSitter/                   # TreeSitter C++ bindings
    └── Roslyn/                       # Roslyn AST analysis
```

### Key Classes

**GenerativeService.cs**:
```csharp
public class GenerativeService
{
    public async Task<CompletionResponse> GenerateCompletionAsync(CompletionRequest request)
    {
        // 1. Validate request
        // 2. Load caches if needed
        // 3. Tokenize prompt
        // 4. Call C++ generation via interop
        // 5. Format OpenAI response
        // 6. Return result
    }
}
```

**GenerativeInterop.cs**:
```csharp
public static class GenerativeInterop
{
    [DllImport("hypercube_generative.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int gen_load_all();

    [DllImport("hypercube_generative.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr gen_generate(
        [MarshalAs(UnmanagedType.LPStr)] string startLabel,
        int maxTokens,
        out int tokenCount);

    [DllImport("hypercube_generative.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void gen_free_result(IntPtr result);
}
```

## C++ to C# Interop Design

### P/Invoke Strategy

#### Calling Convention & Platform Compatibility
- **Calling Convention**: `__cdecl` (Cdecl) for cross-platform Windows/Linux/macOS compatibility
- **Character Set**: UTF-8 encoding for all strings (no ANSI limitations)
- **Exception Handling**: C++ exceptions converted to HRESULT return codes
- **Threading Model**: STA (Single-Threaded Apartment) - no concurrent calls to same instance

#### Function Signature Standards
```cpp
// C++ side signature
GENERATIVE_C_API int gen_function_name(
    const char* input_param,
    size_t input_length,
    void* output_buffer,
    size_t* output_length
);

// C# side P/Invoke
[DllImport("hypercube_generative.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
private static extern int gen_function_name(
    [MarshalAs(UnmanagedType.LPStr)] string inputParam,
    UIntPtr inputLength,
    IntPtr outputBuffer,
    ref UIntPtr outputLength
);
```

### Data Structures & Marshalling

#### Primitive Types Mapping
| C++ Type | C# Type | Marshalling |
|----------|---------|-------------|
| `int32_t` | `int` | Direct |
| `int64_t` | `long` | Direct |
| `size_t` | `UIntPtr` | Platform-aware |
| `double` | `double` | Direct |
| `const char*` | `string` | LPStr (null-terminated) |
| `uint8_t*` | `IntPtr` | Manual array marshaling |

#### Complex Structures

**C# Side Marshalled Structs**:
```csharp
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 8)]
public struct GeneratedTokenResult
{
    public UIntPtr token_index;
    public double score_centroid;
    public double score_pmi;
    public double score_attn;
    public double score_global;
    public double score_total;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 8)]
public struct VocabEntry
{
    [MarshalAs(UnmanagedType.LPStr)]
    public string label;
    public long depth;
    public double frequency;
    public double hilbert_index;
    public Centroid4D centroid;
}

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 8)]
public struct Centroid4D
{
    public double x, y, z, m;
}
```

**C++ Side (generative_c.h)**:
```cpp
typedef struct {
    size_t token_index;
    double score_centroid, score_pmi, score_attn, score_global, score_total;
} GenTokenResult;

typedef struct {
    const char* label;
    int64_t depth;
    double frequency, hilbert_index;
    struct { double x, y, z, m; } centroid;
} VocabEntry;
```

#### Array Marshalling Strategy
```csharp
// For fixed-size arrays
[DllImport("...")]
private static extern int gen_generate_fixed(
    string startLabel,
    int maxTokens,
    [Out] GeneratedTokenResult[] results,  // Fixed size array
    out int actualCount
);

// For variable-size arrays (two-phase)
[DllImport("...")]
private static extern int gen_generate_alloc(
    string startLabel,
    int maxTokens,
    out IntPtr resultPtr,  // C++ allocates
    out int count
);

[DllImport("...")]
private static extern void gen_free_results(IntPtr ptr);  // C# frees
```

### Memory Management Protocol

#### Ownership Rules
1. **C# allocates primitive types and strings**: C++ borrows, never frees
2. **C++ allocates result arrays**: C# receives pointer, must call free function
3. **Cache lifetime**: Static C++ objects persist for app lifetime
4. **String encoding**: All strings are UTF-8 byte arrays

#### Safe Memory Patterns
```csharp
public class SafeInterop
{
    public static GeneratedTokenResult[] GenerateTokens(string startLabel, int maxTokens)
    {
        IntPtr resultPtr;
        int count;

        int hr = gen_generate_alloc(startLabel, maxTokens, out resultPtr, out count);
        if (hr != 0) throw new ExternalException("Generation failed", hr);

        try {
            // Marshal array from unmanaged memory
            var results = new GeneratedTokenResult[count];
            Marshal.PtrToStructure(resultPtr, results);
            return results;
        }
        finally {
            gen_free_results(resultPtr);  // Always free C++ allocation
        }
    }
}
```

### Connection & Database Management

#### PostgreSQL Connection Strategy
```csharp
public class PostgresInterop
{
    [DllImport("hypercube_generative.dll")]
    private static extern int gen_connect_db(
        [MarshalAs(UnmanagedType.LPStr)] string connectionString
    );

    [DllImport("hypercube_generative.dll")]
    private static extern int gen_disconnect_db();

    [DllImport("hypercube_generative.dll")]
    private static extern int gen_is_connected();
}
```

#### Connection Lifecycle
1. **App Startup**: Parse connection string from appsettings.json
2. **Connect**: Call `gen_connect_db()` once
3. **Verify**: Periodic `gen_is_connected()` checks
4. **App Shutdown**: Call `gen_disconnect_db()`

### Required Queries & Database Operations

#### Cache Loading Queries
```sql
-- Load vocabulary with 4D centroids
SELECT c.id, c.label, c.depth,
       COALESCE(child_count, 0) as frequency,
       ST_X(c.centroid) cx, ST_Y(c.centroid) cy,
       ST_Z(c.centroid) cz, ST_M(c.centroid) cm,
       (c.hilbert_lo::float8 / 9223372036854775807.0) hilbert
FROM composition c
LEFT JOIN (SELECT child_id, COUNT(*) child_count FROM composition_child GROUP BY child_id) stats
    ON stats.child_id = c.id
WHERE c.label IS NOT NULL
  AND c.label NOT LIKE '[%'
  AND c.centroid IS NOT NULL
ORDER BY c.label;

-- Load bigram PMI scores
SELECT source_id, target_id, weight
FROM relation
WHERE relation_type = 'S'
  AND weight > 0.3;

-- Load attention edges
SELECT source_id, target_id, weight
FROM relation
WHERE relation_type IN ('A', 'W', 'S')
  AND ABS(weight) > 0.1;
```

#### Runtime Queries
```sql
-- Tokenize prompt to composition IDs
SELECT c.id
FROM composition c
WHERE c.label = ANY($1)  -- Array of tokenized words
ORDER BY c.label;

-- Get composition details for start token
SELECT c.id, c.label, c.centroid,
       (c.hilbert_lo::float8 / 9223372036854775807.0) hilbert
FROM composition c
WHERE c.label = $1
LIMIT 1;

-- Semantic similarity for debugging
SELECT c.label,
       centroid_similarity(c.centroid, $1) similarity
FROM composition c
WHERE c.centroid IS NOT NULL
  AND c.label IS NOT NULL
  AND c.label != $2
ORDER BY centroid_distance(c.centroid, $1)
LIMIT $3;
```

### CLI Functionality & Management Tools

#### Database Setup CLI
```bash
# Initialize hypercube database
./hypercube-cli db init --connection-string "..." --seed-atoms

# Load AI model embeddings
./hypercube-cli models ingest --path /models/minilm/ --type safetensor

# Verify database integrity
./hypercube-cli db verify --check-relations --check-geometries
```

#### Cache Management CLI
```bash
# Pre-load all caches for production
./hypercube-cli cache load --vocab --bigrams --attention

# Check cache status
./hypercube-cli cache status

# Clear caches (for debugging)
./hypercube-cli cache clear --all
```

#### API Testing CLI
```bash
# Test completion endpoint
./hypercube-cli api test-completion --prompt "Hello world" --max-tokens 10

# Benchmark generation performance
./hypercube-cli api benchmark --requests 100 --concurrency 4

# Validate OpenAI compatibility
./hypercube-cli api validate-openai --test-file openai_test_cases.json
```

#### Docker & Deployment CLI
```bash
# Build production images
./hypercube-cli docker build --tag latest

# Deploy to Kubernetes
./hypercube-cli k8s deploy --namespace hypercube --replicas 3

# Update configuration
./hypercube-cli config update --db-connection "..." --api-port 5000
```

### API Endpoints Implementation

#### Core Endpoints
```csharp
// /v1/completions (POST)
[HttpPost("completions")]
public async Task<IActionResult> CreateCompletion([FromBody] CompletionRequest request)
{
    // 1. Validate request parameters
    // 2. Check cache loaded status
    // 3. Tokenize prompt if multi-token
    // 4. Call interop generation
    // 5. Format OpenAI response
    // 6. Track usage metrics
}

// /v1/models (GET)
[HttpGet("models")]
public IActionResult ListModels()
{
    return Ok(new {
        @object = "list",
        data = new[] {
            new {
                id = "hypercube-generative",
                @object = "model",
                created = 1677652288,
                owned_by = "hartonomous-opus"
            }
        }
    });
}

// Health check
[HttpGet("health")]
public IActionResult HealthCheck()
{
    var cacheStatus = await _generativeService.GetCacheStatusAsync();
    var dbStatus = await _postgresService.CheckConnectionAsync();

    return Ok(new {
        status = cacheStatus && dbStatus ? "healthy" : "degraded",
        cache_loaded = cacheStatus,
        database_connected = dbStatus,
        timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
    });
}
```

#### Administrative Endpoints
```csharp
// /admin/cache/reload (POST)
[HttpPost("admin/cache/reload")]
public async Task<IActionResult> ReloadCache()
{
    await _generativeService.ReloadAllCachesAsync();
    return Ok("Cache reloaded");
}

// /admin/db/stats (GET)
[HttpGet("admin/db/stats")]
public async Task<IActionResult> GetDatabaseStats()
{
    var stats = await _postgresService.GetDatabaseStatsAsync();
    return Ok(stats);
}

// /admin/generation/test (POST)
[HttpPost("admin/generation/test")]
public async Task<IActionResult> TestGeneration([FromBody] TestGenerationRequest request)
{
    var result = await _generativeService.TestGenerateAsync(request);
    return Ok(result);
}
```

### Error Handling & Diagnostics

#### Error Response Format
```csharp
public class ErrorResponse
{
    public string error { get; set; }
    public string message { get; set; }
    public int code { get; set; }
    public Dictionary<string, object> details { get; set; }
}
```

#### Exception Mapping
```csharp
public class InteropException : Exception
{
    public int HResult { get; }
    public string InteropDetails { get; }

    public static InteropException FromHResult(int hr, string operation)
    {
        var details = MapHResultToDetails(hr);
        return new InteropException(details.message, hr)
        {
            InteropDetails = details.details
        };
    }
}
```

#### Logging Integration
```csharp
public class InteropLogger
{
    private readonly ILogger _logger;

    public void LogInteropCall(string function, TimeSpan duration, int result)
    {
        _logger.LogInformation(
            "Interop call {Function} completed in {Duration}ms with result {Result}",
            function, duration.TotalMilliseconds, result);
    }

    public void LogCacheOperation(string operation, int itemCount)
    {
        _logger.LogInformation(
            "Cache {Operation} affected {Count} items",
            operation, itemCount);
    }
}
```

### Configuration Management

#### appsettings.json Structure
```json
{
  "Hypercube": {
    "Database": {
      "ConnectionString": "Host=localhost;Port=5432;Username=...;Password=...;Database=hypercube",
      "ConnectionTimeout": 30,
      "MaxPoolSize": 10
    },
    "Generative": {
      "CacheLoadTimeout": 300,
      "MaxGenerationTokens": 2048,
      "DefaultTemperature": 0.7,
      "EnableProfiling": false
    },
    "Interop": {
      "DllPath": "hypercube_generative.dll",
      "MemoryLimit": "1GB",
      "ThreadPoolSize": 4
    }
  },
  "Serilog": {
    "MinimumLevel": "Information",
    "WriteTo": [
      {
        "Name": "File",
        "Args": {
          "path": "logs/hypercube-api-.log",
          "rollingInterval": "Day"
        }
      }
    ]
  }
}
```

#### Environment-Specific Configs
- `appsettings.Development.json`: Debug logging, local DB
- `appsettings.Production.json`: Structured logging, production DB
- Environment variables for secrets: `HYPERCUBE_DB_PASSWORD`, etc.

### Monitoring & Observability

#### Metrics Collection
```csharp
public class MetricsService
{
    private readonly Meter _meter = new("HypercubeGenerative");

    private readonly Counter<long> _completionsTotal =
        _meter.CreateCounter<long>("completions_total");

    private readonly Histogram<double> _generationDuration =
        _meter.CreateHistogram<double>("generation_duration_seconds");

    private readonly UpDownCounter<long> _activeConnections =
        _meter.CreateUpDownCounter<long>("active_db_connections");

    public void RecordCompletion(string model, int tokensGenerated, TimeSpan duration)
    {
        _completionsTotal.Add(1, new KeyValuePair<string, object>("model", model));
        _generationDuration.Record(duration.TotalSeconds,
            new KeyValuePair<string, object>("model", model));
    }
}
```

#### Health Checks
```csharp
public class GenerativeHealthCheck : IHealthCheck
{
    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        var cacheStatus = await CheckCacheLoadedAsync();
        var dbStatus = await CheckDatabaseConnectionAsync();
        var interopStatus = await CheckInteropAvailableAsync();

        var status = cacheStatus && dbStatus && interopStatus
            ? HealthStatus.Healthy
            : HealthStatus.Degraded;

        return new HealthCheckResult(status, "Hypercube generative health check", data: new Dictionary<string, object>
        {
            ["cache_loaded"] = cacheStatus,
            ["database_connected"] = dbStatus,
            ["interop_available"] = interopStatus
        });
    }
}
```

### Testing Strategy

#### Unit Tests
```csharp
[TestClass]
public class InteropTests
{
    [TestMethod]
    public void GenerateTokens_ValidInput_ReturnsExpectedCount()
    {
        // Arrange
        var service = new GenerativeService();

        // Act
        var result = service.Generate("hello", 5);

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual(5, result.Tokens.Length);
        Assert.IsTrue(result.Tokens.All(t => !string.IsNullOrEmpty(t.Label)));
    }
}
```

#### Integration Tests
```csharp
[TestClass]
public class ApiIntegrationTests : WebApplicationFactory<Program>
{
    [TestMethod]
    public async Task CompletionsEndpoint_ReturnsOpenAICompatibleResponse()
    {
        // Arrange
        var client = CreateClient();
        var request = new
        {
            model = "hypercube-generative",
            prompt = "The meaning of life",
            max_tokens = 10
        };

        // Act
        var response = await client.PostAsJsonAsync("/v1/completions", request);
        var result = await response.Content.ReadFromJsonAsync<CompletionResponse>();

        // Assert
        Assert.AreEqual(HttpStatusCode.OK, response.StatusCode);
        Assert.AreEqual("hypercube-generative", result.Model);
        Assert.IsNotNull(result.Choices);
        Assert.AreEqual(1, result.Choices.Count);
        Assert.IsFalse(string.IsNullOrEmpty(result.Choices[0].Text));
    }
}
```

#### Load Testing
```csharp
[TestClass]
public class LoadTests
{
    [TestMethod]
    public async Task HighConcurrency_GenerationRequests_Succeeds()
    {
        var tasks = Enumerable.Range(0, 100).Select(async i =>
        {
            var response = await _client.PostAsJsonAsync("/v1/completions",
                new { prompt = $"Test prompt {i}", max_tokens = 20 });
            response.EnsureSuccessStatusCode();
        });

        await Task.WhenAll(tasks);
    }
}
```

### Deployment Architecture

#### Docker Multi-Stage Build
```dockerfile
# Build C++ dependencies
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS cpp-builder
COPY cpp/ ./cpp/
RUN ./cpp/build.sh --release --shared

# Build C# API
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS dotnet-builder
COPY csharp/ ./csharp/
COPY --from=cpp-builder /app/cpp/bin/hypercube_generative.dll ./csharp/HypercubeGenerativeApi/
RUN dotnet publish ./csharp/HypercubeGenerativeApi/ -c Release -o /app/publish

# Runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0
COPY --from=dotnet-builder /app/publish ./
EXPOSE 5000
ENTRYPOINT ["dotnet", "HypercubeGenerativeApi.dll"]
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypercube-generative-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hypercube-generative-api
  template:
    metadata:
      labels:
        app: hypercube-generative-api
    spec:
      containers:
      - name: api
        image: hypercube-generative:latest
        ports:
        - containerPort: 5000
        env:
        - name: ASPNETCORE_URLS
          value: "http://*:5000"
        - name: ConnectionStrings__HypercubeDatabase
          valueFrom:
            secretKeyRef:
              name: hypercube-db-secret
              key: connection-string
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
```

#### Ingress Configuration
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hypercube-generative-ingress
spec:
  rules:
  - host: api.hypercube.example.com
    http:
      paths:
      - path: /v1
        pathType: Prefix
        backend:
          service:
            name: hypercube-generative-service
            port:
              number: 80
```

### Security Considerations

#### API Security
- **Authentication**: Bearer token validation (optional for compatibility)
- **Rate Limiting**: Request throttling per IP/key
- **Input Validation**: Strict prompt length/content limits
- **HTTPS Only**: TLS 1.3 encryption in production

#### Database Security
- **Connection Encryption**: SSL/TLS for PostgreSQL connections
- **Parameterized Queries**: Prevent SQL injection
- **Least Privilege**: API uses read-only credentials where possible
- **Audit Logging**: All database operations logged

#### Interop Security
- **DLL Verification**: Check DLL hash on load
- **Memory Bounds**: Validate all buffer sizes
- **Exception Isolation**: C++ exceptions don't crash C# process
- **Resource Limits**: CPU/memory limits on native calls

This comprehensive interop, connectivity, and API specification provides the complete technical foundation for integrating the hypercube generative engine with OpenAI-compatible clients and vLLM-style serving infrastructure.

## AST Integration Concepts

### TreeSitter Integration (C++)

**Purpose**: Parse any programming language into concrete syntax trees (CSTs) for hypercube ingestion.

**Architecture**:
```
Source Code → TreeSitter Parser → CST → Token Extraction → Hypercube Atoms
                                      ↓
                            Relation Edges (AST structure)
```

**Implementation**:
- Bind TreeSitter grammars for target languages
- Extract tokens + positional relationships
- Map to hypercube compositions with geometric coordinates
- Store AST edges as relations for structural queries

**Use Cases**:
- Code search: "Find functions similar to this one"
- Code generation: Generate syntactically valid code
- Refactoring: Understand code structure changes

### Roslyn Integration (C#)

**Purpose**: Deep semantic analysis of C# code using .NET compiler platform.

**Architecture**:
```
C# Source → Roslyn Compilation → Semantic Model → Symbol Analysis
                                                        ↓
                                           Type Relationships → Hypercube Relations
```

**Implementation**:
- Use Roslyn APIs for syntax + semantic analysis
- Extract symbol relationships (inheritance, calls, references)
- Map semantic edges to hypercube relations
- Enable "intelligent" code completion and analysis

**Use Cases**:
- Symbol search: "Find all implementations of this interface"
- Code understanding: Explain what a method does semantically
- Cross-references: Navigate related code elements

### Unified AST Processing

**Combined Pipeline**:
1. **Parsing**: TreeSitter for CST, Roslyn for C# semantic model
2. **Normalization**: Convert ASTs to unified representation
3. **Embedding**: Project AST structures to 4D coordinates
4. **Storage**: Store in hypercube with geometric + relational data
5. **Query**: Semantic code search and generation

## Implementation Roadmap

### Phase 1: Core API (8 weeks)
- [ ] ASP.NET Core project setup
- [ ] P/Invoke interop layer
- [ ] /v1/completions endpoint
- [ ] Basic generation + JSON formatting
- [ ] Unit tests + integration tests
- [ ] Docker containerization

### Phase 2: Enhanced Features (6 weeks)
- [ ] Tokenization for multi-token prompts
- [ ] Usage statistics calculation
- [ ] Error handling + validation
- [ ] Stop sequences support
- [ ] Streaming responses (SSE)
- [ ] Authentication/authorization

### Phase 3: AST Integration (12 weeks)
- [ ] TreeSitter C++ bindings
- [ ] Basic language parsers (Python, JavaScript, C#)
- [ ] AST ingestion pipeline
- [ ] Roslyn semantic analysis
- [ ] Code generation capabilities
- [ ] AST query API endpoints

### Phase 4: Production (6 weeks)
- [ ] Performance optimization
- [ ] Monitoring + metrics
- [ ] Load testing
- [ ] Documentation
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline

## API Reference

### POST /v1/completions

Generate text completions using hypercube semantic walking.

**Parameters**:
- `model` (string): Model name (currently "hypercube-generative")
- `prompt` (string): Input text to complete
- `max_tokens` (int): Maximum tokens to generate (default: 16)
- `temperature` (float): Sampling temperature 0.0-1.0 (default: 0.7)
- `stop` (string[]): Stop sequences (default: [".", "!", "?"])
- `stream` (bool): Stream responses (not implemented)

**Response**: CompletionResponse object

### GET /v1/models

List available models.

**Response**:
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

## Implementation Notes

### Performance Considerations

- **Cache Loading**: ~30-60 seconds on first startup
- **Generation Speed**: 100-500ms per completion
- **Memory Usage**: ~500MB for loaded vocabularies
- **Concurrent Requests**: Single-threaded initially, can scale to multi-threaded

### Error Handling

- **400 Bad Request**: Invalid parameters
- **500 Internal Server Error**: C++ engine errors, DB connection issues
- **503 Service Unavailable**: Cache not loaded

### Security

- Input validation on all endpoints
- Rate limiting (future)
- CORS configuration for web clients
- API key authentication (future)

### Testing Strategy

- **Unit Tests**: P/Invoke wrappers, JSON serialization
- **Integration Tests**: Full request/response cycle
- **Load Tests**: Concurrent generation requests
- **Compatibility Tests**: OpenAI client library compatibility

## Conclusion

This OpenAI-compatible API bridges the powerful hypercube semantic substrate with standard LLM interfaces, enabling integration with existing tools while opening new possibilities for AST-aware code intelligence. The C# orchestrator provides the web interface and ecosystem integration, while the C++ engine delivers the high-performance 4D geometric generation.