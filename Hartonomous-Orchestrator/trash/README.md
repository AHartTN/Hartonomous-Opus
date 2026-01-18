# Hartonomous Orchestrator

An enterprise-grade RAG (Retrieval-Augmented Generation) orchestrator for managing generative AI workflows with OpenAI-compatible endpoints.

## 🚀 Quick Start

```bash
# Build and run with Docker (recommended)
docker build -t hartonomous-orchestrator .
docker run -p 8080:8080 -p 9090:9090 hartonomous-orchestrator

# Or use Docker Compose (includes Qdrant)
docker-compose up -d

# Or build locally and run
./scripts/build.sh
./scripts/launch.sh start

# Run basic tests
./scripts/test-basic.sh

# Check health
curl http://localhost:8080/health
```

## Overview

The Hartonomous Orchestrator is a production-ready C++ solution for enterprise RAG operations that provides a unified interface for managing generative, embedding, reranking, and vector search capabilities. It serves as a central hub that coordinates multiple AI services and databases to deliver sophisticated knowledge retrieval and generation workflows.

## Key Features

- **Multi-Service Orchestration**: Coordinates embedding, reranking, generative, and vector search services
- **OpenAI Compatibility**: Supports OpenAI-compatible endpoints with proxy capabilities
- **Enterprise-Grade RAG**: Full RAG pipeline with ingest and query capabilities
- **Scalable Architecture**: Modular design with service clients and middleware
- **Monitoring & Metrics**: Prometheus-compatible metrics collection
- **Security**: Authentication middleware and HTTPS support
- **Performance**: Async operations, caching, and optimization
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Container Ready**: Optimized for Docker/Kubernetes deployment

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   HTTP Server   │    │   Orchestrator │
│                 │    │                 │
│  (Endpoints)      │───▶│  (Logic)      │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
       │                      │
       ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│   Embedding     │    │   Reranking   │
│   Service       │    │   Service     │
│                 │    │                 │
│  (text-embedding)│    │  (rerank)     │
└─────────────────┘    └─────────────────┘
       │                      │
       ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│   Generative    │    │   Qdrant      │
│   Service       │    │   Vector DB   │
│                 │    │                 │
│  (chat/generation)│   │  (vector search)  │
└─────────────────┘    └─────────────────┘
```

## Components

1. **HTTP Server**: REST API endpoints for RAG operations
2. **Orchestrator**: Core RAG pipeline coordinator
3. **Service Clients**: 
   - Embedding Client (OpenAI-compatible)
   - Reranking Client (OpenAI-compatible)
   - Generative Client (OpenAI-compatible)
   - Qdrant Client (Vector database)
4. **Middleware**: 
   - Authentication
   - Rate Limiting
   - Logging
   - Tracing
5. **Core Infrastructure**:
   - Configuration Management (YAML)
   - Logging (spdlog)
   - Metrics Collection (Prometheus-compatible)
   - HTTP Client (Boost.Beast)

## Configuration

The orchestrator uses YAML configuration files for service endpoints and settings:

```yaml
# config.yaml
services:
  embedding:
    endpoint: "http://localhost:8711"
    api_key: "your-api-key"
    model: "text-embedding-3"
    
  reranking:
    endpoint: "http://localhost:8711"
    api_key: "your-api-key"
    model: "rerank-lite"
    
  generative:
    endpoint: "http://localhost:8711"
    api_key: "your-api-key"
    model: "gpt-3.5-turbo"
    
  vector_db:
    endpoint: "http://localhost:6333"
    api_key: "your-qdrant-key"

metrics:
  enabled: true
  port: 9090

logging:
  level: "info"
  file: "logs/orchestrator.log"
```

## Endpoints

### RAG Ingest
```
POST /ingest
{
  "id": "document-1",
  "content": "Your document text here",
  "metadata": { "category": "technical" }
}
```

### RAG Query
```
POST /query
{
  "query": "What is the capital of France?",
  "collection": "default"
}
```

### Batch Query
```
POST /batch-query
{
  "queries": [
    "What is the capital of France?",
    "What is the population of Paris?"
  ]
}
```

### Health Check
```
GET /health
```

### Metrics
```
GET /metrics
```

## Security

- API Key Authentication
- HTTPS support with SSL/TLS certificates
- Rate limiting
- Request validation

## Performance Features

- Async operations with Boost.Beast
- Connection pooling
- In-memory caching with TTL
- Circuit breaker pattern
- Retry logic with exponential backoff
- Prometheus metrics
- Graceful shutdown handling

## Deployment

### Build Requirements

- C++20 compiler
- CMake 3.20+
- vcpkg package manager
- Boost libraries (Beast, JSON)
- spdlog
- rapidyaml
- Prometheus C++ client

## 🏗️ Building and Running

### Docker (Recommended)

The easiest way to run the orchestrator is using Docker:

```bash
# Build the image
docker build -t hartonomous-orchestrator .

# Run with default config
docker run -p 8080:8080 -p 9090:9090 hartonomous-orchestrator

# Run with custom config
docker run -p 8080:8080 -p 9090:9090 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  hartonomous-orchestrator
```

### Local Build Scripts

Cross-platform build scripts are provided for development:

#### Linux/macOS
```bash
# Build everything
./scripts/build.sh

# Build in debug mode
BUILD_TYPE=Debug ./scripts/build.sh

# Build without tests
ENABLE_TESTS=false ./scripts/build.sh

# Launch after building
./scripts/launch.sh start
```

#### Windows
```batch
REM Build everything
scripts\build.bat

REM Launch after building
scripts\launch.bat start
```

### Manual Build

For advanced users or CI/CD:

#### Prerequisites
- C++20 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.20+
- Dependencies: Boost, spdlog, yaml-cpp, prometheus-cpp

#### Build Steps
```bash
# Create build directory
mkdir build && cd build

# Configure (with tests)
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# Build
cmake --build . --config Release --parallel

# Run tests
ctest --output-on-failure

# Install (optional)
cmake --install . --prefix /usr/local
```

## 📊 Management & Monitoring

### Launch Scripts

Cross-platform management scripts are provided:

#### Linux/macOS
```bash
# Start the orchestrator
./scripts/launch.sh start

# Check status
./scripts/launch.sh status

# View logs in real-time
./scripts/launch.sh logs -f

# Restart
./scripts/launch.sh restart

# Stop
./scripts/launch.sh stop
```

#### Windows
```batch
REM Start the orchestrator
scripts\launch.bat start

REM Check status
scripts\launch.bat status

REM Stop
scripts\launch.bat stop
```

### Health Checks

Run comprehensive health checks:

```bash
# Basic health check
./scripts/health-check.sh

# Verbose output
./scripts/health-check.sh --verbose

# Custom config file
./scripts/health-check.sh --config /path/to/config.yaml
```

The health check verifies:
- Process status and responsiveness
- Service endpoint availability
- System resource usage
- Configuration validity

### API Endpoints

#### Query Endpoint
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "collection": "default"}'
```

#### Document Ingestion
```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "id": "doc1",
      "content": "Machine learning is...",
      "metadata": {"source": "wikipedia"}
    }]
  }'
```

#### Health Check
```bash
curl http://localhost:8080/health
```

#### Metrics (Prometheus format)
```bash
curl http://localhost:9090/metrics
```

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run specific test suite
./build/bin/hartonomous-tests --gtest_filter="*OrchestratorTest*"

# Run integration tests
./build/bin/hartonomous-tests --gtest_filter="*IntegrationTest*"

# Generate test report
./build/bin/hartonomous-tests --gtest_output=xml:test_results.xml
```

## 🚀 Deployment

### Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: '3.8'
services:
  orchestrator:
    build: .
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./logs:/app/logs
    depends_on:
      - qdrant
      - embedding-service
      - generative-service

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"

  # Add your AI service containers here
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml

# Check deployment
kubectl get pods -l app=hartonomous-orchestrator

# View logs
kubectl logs -l app=hartonomous-orchestrator
```

## 📖 Documentation

- **[Launch Guide](LAUNCH_GUIDE.md)**: Comprehensive launch and management guide
- **[Architecture](architecture.md)**: System architecture and design details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [Launch Guide](LAUNCH_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/hartonomous-orchestrator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/hartonomous-orchestrator/discussions)

For enterprise support and custom deployments, contact support@hartonomous.com
```

## Next Steps

Now that I've implemented the core components of the Hartonomous Orchestrator, I'll create a comprehensive Docker configuration to make it production-ready for deployment. Let me implement Docker-related files:
<tool_call>
<function=write_to_file>
<parameter=path>
Dockerfile