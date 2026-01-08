# Hypercube Generative API - Architecture Documentation

## System Overview

The Hypercube Generative API is an OpenAI-compatible REST service that provides access to Hartonomous-Opus's 4D hypercube semantic substrate for text generation. It serves as a bridge between standard LLM APIs and the unique geometric computation capabilities of the hypercube system.

## Core Architecture Principles

### 1. OpenAI Compatibility
- **Drop-in Replacement**: Any client using OpenAI's `/v1/completions` API can use this service
- **Standard Responses**: JSON responses match OpenAI's format exactly
- **Parameter Mapping**: OpenAI parameters map to hypercube generation controls

### 2. C# Orchestration with C++ Performance
- **C# Layer**: Handles HTTP, JSON, validation, and orchestration
- **C++ Layer**: Provides high-performance hypercube computations
- **Clean Interop**: P/Invoke bridge with proper memory management

### 3. Semantic-First Generation
- **Vocabulary Validation**: Prompts validated against hypercube's semantic space
- **Geometric Relationships**: Generation uses 4D centroid similarity
- **Stop Sequence Awareness**: Early termination at semantic boundaries

## Component Architecture

```
┌─────────────────┐    HTTP/JSON    ┌──────────────────────┐
│   API Clients   │◄──────────────►│  ASP.NET Core API    │
│ (OpenAI compat) │                 │    Controllers       │
└─────────────────┘                 └──────────────────────┘
         ▲                                       │
         │                                       ▼
         │ REST                                  ┌──────────────────────┐
         │                                       │   Service Layer      │
         │                                       │ • GenerativeService  │
         │                                       │ • TokenizationService│
         │                                       │ • PostgresService    │
         └──────────────────────────────────────►│ • HealthCheck        │
                                                 └──────────────────────┘
                                                         │
                                                         ▼ P/Invoke
                                                 ┌──────────────────────┐
                                                 │  C++ Hypercube Engine │
                                                 │ • Vocabulary Cache    │
                                                 │ • Bigram Relations    │
                                                 │ • Attention Weights   │
                                                 │ • 4D Generation       │
                                                 └──────────────────────┘
                                                         │
                                                         ▼ PostgreSQL
                                                 ┌──────────────────────┐
                                                 │   Hypercube Database  │
                                                 │ • Atom table (1.1M)  │
                                                 │ • Composition table   │
                                                 │ • Relation edges      │
                                                 │ • 4D geometries       │
                                                 └──────────────────────┘
```

## Request Flow

### 1. HTTP Request Reception
- **Controller**: `CompletionsController.CreateCompletion()`
- **Validation**: Model, prompt, parameters validated
- **Authentication**: Future - API key validation

### 2. Tokenization & Encoding
- **Service**: `TokenizationService.TokenizeAndEncodeAsync()`
- **Process**: Split prompt → Validate against DB → Get composition IDs
- **Fallback**: Default to "the" if no valid tokens found

### 3. Generation Orchestration
- **Service**: `GenerativeService.GenerateCompletionAsync()`
- **Configuration**: Map OpenAI params to C++ settings
- **Iteration**: Generate tokens one-by-one with stop sequence checking

### 4. C++ Engine Execution
- **Interop**: P/Invoke calls to `hypercube_generative.dll`
- **Computation**: 4D centroid similarity, PMI scoring, attention weights
- **Memory**: Proper allocation/deallocation across language boundary

### 5. Response Formatting
- **DTOs**: Map C++ results to OpenAI-compatible JSON
- **Metadata**: Usage statistics, finish reasons, timestamps
- **Serialization**: System.Text.Json with property preservation

## Service Layer Design

### GenerativeService
**Purpose**: Orchestrates the complete generation pipeline

**Key Methods**:
- `InitializeAsync()`: Load C++ caches (vocab, bigrams, attention)
- `GenerateCompletionAsync()`: Main generation workflow
- `GetCacheStats()`: Monitoring and diagnostics

**Responsibilities**:
- Service lifecycle management
- Parameter validation and mapping
- Stop sequence implementation
- Error handling and logging
- Performance monitoring

### TokenizationService
**Purpose**: Convert natural language prompts to hypercube-compatible tokens

**Key Methods**:
- `TokenizeAndEncodeAsync()`: Main tokenization workflow
- `EstimateTokenCount()`: Usage statistics calculation
- `GetStableHash()`: Deterministic token ID generation

**Responsibilities**:
- Text preprocessing and splitting
- Database validation of token existence
- Fallback handling for unknown tokens
- Token counting for billing/metrics

### PostgresService
**Purpose**: Database connectivity and hypercube data access

**Key Methods**:
- `InitializeAsync()`: Establish database connection
- `TokenExistsAsync()`: Validate token in vocabulary
- `GetValidTokensFromPromptAsync()`: Extract known tokens
- `GetDatabaseStatsAsync()`: Monitoring information

**Responsibilities**:
- Connection management and pooling (future)
- Query execution with proper error handling
- BYTEA composition ID handling (future)
- Database health monitoring

### GenerativeHealthCheck
**Purpose**: ASP.NET Core health check implementation

**Key Methods**:
- `CheckHealthAsync()`: Comprehensive system health assessment

**Checks Performed**:
- Cache initialization status
- Database connectivity
- Memory usage (future)
- Generation capability

## Interop Layer Architecture

### P/Invoke Design Principles

#### Memory Ownership
- **C# Allocates**: Primitive types, strings, arrays
- **C++ Allocates**: Result structures, complex objects
- **Clear Handover**: Documented ownership transfer points

#### Error Handling
- **HRESULT Pattern**: C++ returns error codes, C# throws exceptions
- **Exception Isolation**: C++ exceptions don't crash C# process
- **Resource Cleanup**: Proper disposal even on errors

#### Threading Model
- **STA Compliance**: Single-threaded access to C++ objects
- **No Concurrency**: Serialize all interop calls
- **Async Safety**: C# async/await with synchronous P/Invoke

### Function Categories

#### Cache Management
```csharp
[DllImport(DllName)]
public static extern int gen_load_vocab();  // Load Unicode atoms

[DllImport(DllName)]
public static extern int gen_load_bigrams(); // Load PMI relations

[DllImport(DllName)]
public static extern int gen_load_attention(); // Load attention weights
```

#### Configuration
```csharp
[DllImport(DllName)]
public static extern void gen_config_set_weights(double centroid, double pmi, double attn, double global);

[DllImport(DllName)]
public static extern void gen_config_set_policy(int greedy, double temperature);
```

#### Generation
```csharp
[DllImport(DllName)]
public static extern IntPtr gen_generate(string startLabel, UIntPtr maxTokens, out int tokenCount);

[DllImport(DllName)]
public static extern void gen_free_results(IntPtr resultsPtr);
```

## Data Flow & State Management

### Application Startup
1. **DI Container Setup**: Register services and dependencies
2. **Database Connection**: Establish PostgreSQL connection
3. **Cache Loading**: Initialize C++ vocab/bigrams/attention caches
4. **Health Check Registration**: Enable monitoring endpoints
5. **Server Start**: Begin accepting HTTP requests

### Request Processing
1. **HTTP Reception**: Minimal API routes request to controller
2. **Validation**: Comprehensive parameter and state checks
3. **Tokenization**: Prompt → Valid hypercube tokens
4. **Generation**: C# orchestrates C++ generation with stop sequences
5. **Formatting**: Results → OpenAI-compatible JSON response

### Error Scenarios
- **Invalid Parameters**: 400 Bad Request with detailed error
- **Service Unavailable**: 503 with initialization status
- **Database Issues**: 503 with connection retry guidance
- **Generation Failures**: 500 with safe error messages

## Configuration Management

### appsettings.json Structure
```json
{
  "Hypercube": {
    "Database": {
      "ConnectionString": "Host=...;Database=hypercube",
      "ConnectionTimeout": 30,
      "MaxPoolSize": 10
    },
    "Generative": {
      "CacheLoadTimeout": 300,
      "MaxGenerationTokens": 2048,
      "DefaultTemperature": 0.7
    },
    "Interop": {
      "DllPath": "hypercube_generative.dll"
    }
  }
}
```

### Environment Overrides
- **Production DB**: Environment variables for secrets
- **Performance Tuning**: GC and thread pool settings
- **Logging Levels**: Structured logging configuration

## Monitoring & Observability

### Health Endpoints
- **`/health`**: Overall system health
- **`/health/ready`**: Readiness for traffic
- **`/health/live`**: Liveness probe

### Metrics Collection
- **Generation Latency**: End-to-end request time
- **Token Counts**: Prompt and completion sizes
- **Error Rates**: Per endpoint failure rates
- **Cache Statistics**: Vocabulary and relation counts

### Logging Strategy
- **Structured Logs**: JSON format with correlation IDs
- **Log Levels**: Debug (dev), Information (prod), Warning/Error
- **Performance Logs**: Timing information for bottlenecks
- **Security Logs**: Authentication and authorization events

## Testing Strategy

### Unit Tests
- **Service Logic**: Individual service method testing
- **Validation**: Parameter validation edge cases
- **Error Handling**: Exception scenarios and recovery

### Integration Tests
- **API Endpoints**: Full HTTP request/response cycles
- **Database**: Real PostgreSQL interactions
- **Interop**: C++ function call verification

### End-to-End Tests
- **OpenAI Compatibility**: Real client integration testing
- **Performance**: Load testing with realistic workloads
- **Reliability**: Chaos engineering and failure simulation

## Deployment Architecture

### Docker Containerization
- **Multi-stage Build**: Separate build and runtime stages
- **Native Dependencies**: Include C++ runtime libraries
- **Security Hardening**: Non-root user, minimal attack surface
- **Health Checks**: Container-level health monitoring

### Kubernetes Deployment
- **Horizontal Scaling**: Replica sets with load balancing
- **Resource Limits**: CPU and memory constraints
- **Config Maps**: Environment-specific configuration
- **Secrets Management**: Database credentials and API keys

### Production Considerations
- **Load Balancing**: Distribute requests across instances
- **Caching Layer**: Redis for hot data and session storage
- **Database Scaling**: Read replicas and connection pooling
- **Monitoring Stack**: Prometheus, Grafana, ELK stack

## Future Enhancements

### Streaming Support
- **Server-Sent Events**: Real-time token streaming
- **Client Compatibility**: OpenAI streaming client support
- **Cancellation**: Request cancellation and cleanup

### Advanced Features
- **Chat Completions**: Multi-turn conversation support
- **Embeddings API**: Semantic vector representations
- **Fine-tuning**: Custom model adaptation
- **Batch Processing**: Large-scale request handling

### AST Integration
- **TreeSitter**: Multi-language syntax parsing
- **Roslyn**: C# semantic analysis
- **Code Generation**: Syntactically valid output
- **Intelligent Completion**: Context-aware suggestions

This architecture provides a solid foundation for production deployment while maintaining the flexibility to add advanced features like AST integration and streaming support.