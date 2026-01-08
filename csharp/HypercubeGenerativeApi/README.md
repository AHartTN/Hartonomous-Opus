# Hypercube Generative API

OpenAI-compatible REST API for the Hartonomous-Opus hypercube generative engine. This API serves as a drop-in replacement for OpenAI's completion endpoints, enabling any OpenAI-compatible client to leverage hypercube's unique 4D semantic generation capabilities.

## 🚀 Quick Start

### Prerequisites
- .NET 8.0 SDK
- PostgreSQL with hypercube database
- Built `hypercube_generative.dll` (from C++ build)

### Running Locally

```bash
# Clone and navigate to the API project
cd csharp/HypercubeGenerativeApi

# Build the project
dotnet build

# Copy the native DLL to output directory
cp /path/to/hypercube_generative.dll bin/Debug/net8.0/

# Run the API server
dotnet run
```

**API available at**: `http://localhost:5000`

### Test with OpenAI-compatible Client

```bash
# Test completions endpoint (compatible with any OpenAI client)
curl -X POST http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hypercube-generative",
    "prompt": "The meaning of life is",
    "max_tokens": 20,
    "temperature": 0.7
  }'
```

### Docker Deployment

```bash
# Build container
docker build -t hypercube-generative-api .

# Run with database connection
docker run -p 5000:5000 \
  -e ConnectionStrings__HypercubeDatabase="Host=host.docker.internal;Database=hypercube" \
  hypercube-generative-api
```

## 📖 Documentation Structure

This API is extensively documented. See the detailed guides:

### 🏗️ **Architecture & Design**
- **[API Architecture](docs/API_ARCHITECTURE.md)** - System design, component relationships, data flow
- **[Error Handling](docs/ERROR_HANDLING.md)** - Comprehensive error management and validation
- **[Controllers & Endpoints](docs/CONTROLLERS_ENDPOINTS.md)** - API endpoints and HTTP handling

### 🔧 **Core Services**
- **[GenerativeService](docs/GENERATIVE_SERVICE.md)** - Main orchestration service and generation workflow
- **[TokenizationService](docs/TOKENIZATION_SERVICE.md)** - Prompt processing and vocabulary validation
- **[PostgresService](docs/POSTGRES_SERVICE.md)** - Database connectivity and hypercube data access

### 📊 **Additional Resources**
- **[Production Readiness Report](../PRODUCTION_READINESS_REPORT.md)** - Current status and remaining work
- **[OpenAI API Roadmap](../../plans/openai-compatible-api-implementation.md)** - Future enhancements plan

## 🎯 Key Features

### OpenAI Compatibility
- **Drop-in Replacement**: Works with any OpenAI-compatible client
- **Standard Endpoints**: `/v1/completions`, `/v1/models`, `/health`
- **Structured Responses**: OpenAI-format JSON with proper error handling
- **Parameter Mapping**: Temperature, max_tokens, stop sequences

### Hypercube Intelligence
- **4D Semantic Generation**: Uses geometric relationships for coherent text
- **Vocabulary Validation**: Ensures prompts contain known semantic tokens
- **Stop Sequence Support**: Early termination at natural boundaries
- **Context-Aware**: Maintains semantic flow through token relationships

### Production Ready
- **Comprehensive Validation**: Input sanitization and parameter checking
- **Health Monitoring**: Built-in health checks and metrics
- **Error Resilience**: Structured error responses with proper HTTP codes
- **Logging**: Detailed request/response logging for debugging
- **Docker Support**: Containerized deployment with security hardening

## 🔌 API Endpoints

### POST /v1/completions
Generate text completions using hypercube semantic walking.

**Request**:
```json
{
  "model": "hypercube-generative",
  "prompt": "The future of AI",
  "max_tokens": 50,
  "temperature": 0.7,
  "stop": [".", "!", "?"]
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
    "text": " involves geometric computation and semantic relationships...",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 8,
    "total_tokens": 12
  }
}
```

### GET /v1/models
List available models.

### GET /health
System health check with cache and database status.

## ⚙️ Configuration

### appsettings.json
```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "HypercubeGenerativeApi": "Information"
    }
  },
  "ConnectionStrings": {
    "HypercubeDatabase": "Host=localhost;Port=5432;Username=user;Password=pass;Database=hypercube"
  },
  "Hypercube": {
    "Database": {
      "ConnectionTimeout": 30,
      "MaxPoolSize": 10
    },
    "Generative": {
      "CacheLoadTimeout": 300,
      "MaxGenerationTokens": 2048,
      "DefaultTemperature": 0.7
    }
  }
}
```

### Environment Variables
```bash
# Database connection (secure production setup)
ConnectionStrings__HypercubeDatabase="Host=prod-db;Username=app;Password=${DB_PASSWORD}"

# Logging configuration
Logging__LogLevel__HypercubeGenerativeApi="Debug"

# Performance tuning
Hypercube__Generative__MaxGenerationTokens="4096"
```

## 🧪 Development & Testing

### Running Tests
```bash
# Unit tests
dotnet test --filter "TestCategory=Unit"

# Integration tests (requires database)
dotnet test --filter "TestCategory=Integration"

# All tests
dotnet test
```

### Test Coverage
- **API Endpoints**: HTTP request/response validation
- **Error Handling**: Exception scenarios and responses
- **Parameter Validation**: Edge cases and invalid inputs
- **Service Integration**: Component interaction testing

### Debugging
```bash
# Enable debug logging
export Logging__LogLevel__HypercubeGenerativeApi="Debug"

# View health status
curl http://localhost:5000/health

# Test with verbose output
curl -v -X POST http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hypercube-generative", "prompt": "test", "max_tokens": 5}'
```

## 🔧 Troubleshooting

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **DLL Load Failure** | `Unable to load DLL 'hypercube_generative.dll'` | Ensure native DLL is in output directory and has execute permissions |
| **Database Connection** | `Database temporarily unavailable` | Check PostgreSQL is running and connection string is correct |
| **Cache Not Loaded** | `Service is not ready` | Wait for application startup to complete cache loading |
| **Invalid Tokens** | Short or nonsensical output | Prompt contains unknown words; use common vocabulary |
| **Generation Timeout** | Request hangs | Reduce `max_tokens` or check C++ engine performance |

### Diagnostic Commands
```bash
# Check service health
curl http://localhost:5000/health

# Test basic generation
curl -X POST http://localhost:5000/v1/completions \
  -d '{"model": "hypercube-generative", "prompt": "the", "max_tokens": 5}'

# Check database connectivity
curl http://localhost:5000/health | jq '.data.database_connected'
```

### Performance Monitoring
- **Health Endpoint**: Cache status, DB connectivity, vocabulary counts
- **Response Times**: Typical generation: 100-500ms
- **Error Rates**: Monitor 4xx/5xx response codes
- **Token Throughput**: Track tokens generated per second

## 🔗 Integration Examples

### Roo Code Configuration
```json
{
  "apiBaseUrl": "http://localhost:5000/v1",
  "model": "hypercube-generative",
  "temperature": 0.7,
  "maxTokens": 100
}
```

### Python OpenAI Client
```python
import openai

# Configure for hypercube API
openai.api_base = "http://localhost:5000/v1"
openai.api_key = "not-needed"  # API key validation not implemented yet

response = openai.Completion.create(
    model="hypercube-generative",
    prompt="The essence of computation",
    max_tokens=50,
    temperature=0.7
)

print(response.choices[0].text)
```

### Generic HTTP Client
```javascript
// Any HTTP client can be used
const response = await fetch('http://localhost:5000/v1/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'hypercube-generative',
    prompt: 'The nature of intelligence',
    max_tokens: 30
  })
});

const result = await response.json();
console.log(result.choices[0].text);
```

## 🏭 Production Deployment

### Docker Compose Example
```yaml
version: '3.8'
services:
  hypercube-api:
    image: hypercube-generative-api:latest
    ports:
      - "5000:5000"
    environment:
      - ConnectionStrings__HypercubeDatabase=Host=db;Database=hypercube
      - ASPNETCORE_ENVIRONMENT=Production
    depends_on:
      - db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=hypercube
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Kubernetes Deployment
See future roadmap for complete K8s manifests with horizontal scaling and load balancing.

## 📈 Roadmap & Future Features

### Phase 2 (Immediate Next)
- **Streaming Responses**: Real-time token generation (SSE)
- **Authentication**: API key validation and rate limiting
- **Chat Completions**: `/v1/chat/completions` endpoint
- **Enhanced Monitoring**: Metrics and alerting

### Phase 3 (Advanced Features)
- **TreeSitter Integration**: Multi-language AST parsing
- **Roslyn Analysis**: Deep C# semantic understanding
- **Code Generation**: Syntactically valid output
- **Intelligent Completion**: Context-aware suggestions

### Phase 4 (Enterprise Scale)
- **Kubernetes Orchestration**: Auto-scaling and high availability
- **Multi-Platform**: Linux/Windows/macOS native builds
- **Client SDKs**: Python, JavaScript, Go libraries
- **Advanced Monitoring**: Distributed tracing and performance profiling

## 🤝 Contributing

### Development Setup
1. **Prerequisites**: .NET 8.0, PostgreSQL, C++ build tools
2. **Clone**: `git clone <repository>`
3. **Database**: Set up hypercube database schema
4. **Build C++**: Compile hypercube_generative.dll
5. **Run Tests**: `dotnet test`
6. **Start API**: `dotnet run`

### Code Organization
- **`Controllers/`**: HTTP endpoint handlers
- **`Models/`**: Request/response DTOs and validation
- **`Services/`**: Business logic and external integrations
- **`Interop/`**: C++ P/Invoke declarations
- **`docs/`**: Comprehensive documentation

### Testing Guidelines
- **Unit Tests**: Service logic, validation, error handling
- **Integration Tests**: Database operations, API endpoints
- **Performance Tests**: Latency, throughput, memory usage
- **Compatibility Tests**: OpenAI client integration

## 📄 License

Apache License 2.0 - See LICENSE file for details.

## 📞 Support

- **Documentation**: Comprehensive guides in `docs/` directory
- **Health Checks**: Built-in `/health` endpoint for monitoring
- **Logs**: Structured logging with configurable levels
- **Issues**: GitHub issues for bugs and feature requests

---

**Built with ❤️ for the Hartonomous-Opus hypercube semantic substrate**

*Transforming geometric computation into human language through 4D semantic relationships.*