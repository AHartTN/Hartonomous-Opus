# Hartonomous Orchestrator Launch and Management Guide

This guide provides comprehensive instructions for launching, managing, and maintaining the Hartonomous Orchestrator RAG system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Building from Source](#building-from-source)
4. [Configuration](#configuration)
5. [Launching the System](#launching-the-system)
6. [Management and Monitoring](#management-and-monitoring)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)
10. [Performance Tuning](#performance-tuning)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Compiler**: GCC 9+ or Clang 10+ (Linux/macOS), MSVC 2019+ (Windows)
- **CMake**: 3.20 or later
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: 2GB free space for build artifacts

### Dependencies

The system requires the following external services and libraries:

#### External Services
- **Embedding Service**: OpenAI-compatible API (e.g., text-embedding-3-large)
- **Reranking Service**: Cross-encoder model API
- **Generative Service**: OpenAI-compatible chat API (e.g., GPT-4)
- **Vector Database**: Qdrant vector database

#### System Libraries
- **Boost**: 1.74+ (Beast, JSON, ASIO)
- **spdlog**: 1.9+ (Logging)
- **yaml-cpp**: 0.7+ (Configuration)
- **prometheus-cpp**: 1.0+ (Metrics)
- **GTest**: 1.10+ (Testing, optional)

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libprometheus-cpp-dev \
    libgtest-dev \
    pkg-config \
    curl
```

#### macOS (with Homebrew)
```bash
brew install \
    cmake \
    boost \
    spdlog \
    yaml-cpp \
    prometheus-cpp \
    googletest \
    pkg-config \
    curl
```

#### Windows (with vcpkg)
```powershell
# Install vcpkg if not already installed
git clone https://github.com/Microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat

# Install dependencies
.\vcpkg\vcpkg install boost beast json rapidyaml spdlog prometheus-cpp
```

## Quick Start

For the impatient, here's how to get started quickly:

```bash
# 1. Clone and build
git clone <repository-url>
cd hartonomous-orchestrator
./scripts/build.sh

# 2. Configure services (edit config.yaml)
# Make sure your external services are running and accessible

# 3. Launch
./scripts/launch.sh start

# 4. Check health
./scripts/health-check.sh

# 5. Test basic functionality
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

## Building from Source

### Using the Build Script

The easiest way to build is using the provided build script:

```bash
# Build everything (release mode)
./scripts/build.sh

# Debug build
BUILD_TYPE=Debug ./scripts/build.sh

# Build without tests
ENABLE_TESTS=false ./scripts/build.sh

# Verbose build
VERBOSE_BUILD=1 ./scripts/build.sh
```

### Manual Build

If you prefer manual control:

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### Build Options

- `BUILD_TYPE`: `Debug`, `Release`, `RelWithDebInfo`
- `ENABLE_TESTS`: `true`/`false` - Build test suite
- `CMAKE_EXTRA_ARGS`: Additional CMake arguments
- `INSTALL_PREFIX`: Custom install prefix

## Configuration

### Configuration File Structure

The system uses a YAML configuration file (`config.yaml`):

```yaml
# Server Configuration
server:
  host: "0.0.0.0"
  port: 8080
  max_connections: 1000
  request_timeout: 30

# External Services
services:
  embedding:
    endpoint: "http://embedding-service:8711"
    api_key: "your-embedding-key"
    model: "text-embedding-3-large"
    timeout: 30

  reranking:
    endpoint: "http://reranking-service:8711"
    api_key: "your-reranking-key"
    model: "rerank-lite"
    timeout: 30

  generative:
    endpoint: "http://generative-service:8711"
    api_key: "your-generative-key"
    model: "gpt-4-turbo"
    timeout: 60

  vector_db:
    endpoint: "http://qdrant-service:6333"
    api_key: "your-qdrant-key"
    timeout: 30

# Monitoring and Logging
logging:
  level: "info"
  file: "/var/log/hartonomous/orchestrator.log"
  max_size: 10485760
  max_files: 5

metrics:
  enabled: true
  port: 9090
  collection_interval: 30
```

### Environment Variables

Override configuration with environment variables:

```bash
# Server
export SERVER_HOST="127.0.0.1"
export SERVER_PORT="9090"

# Service endpoints
export EMBEDDING_ENDPOINT="http://localhost:8711"
export QDRANT_ENDPOINT="http://localhost:6333"

# API Keys
export EMBEDDING_API_KEY="your-key"
export GENERATIVE_API_KEY="your-key"
```

## Launching the System

### Using Launch Scripts

#### Linux/macOS
```bash
# Start the orchestrator
./scripts/launch.sh start

# Check status
./scripts/launch.sh status

# View logs
./scripts/launch.sh logs

# Restart
./scripts/launch.sh restart

# Stop
./scripts/launch.sh stop
```

#### Windows
```batch
# Start the orchestrator
scripts\launch.bat start

# Check status
scripts\launch.bat status

# Stop
scripts\launch.bat stop
```

### Manual Launch

```bash
# Direct execution
./build/bin/hartonomous-orchestrator --config config.yaml

# With environment variable overrides
SERVER_PORT=9090 ./build/bin/hartonomous-orchestrator
```

### Service Dependencies

Ensure these services are running before starting the orchestrator:

1. **Qdrant Vector Database**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Embedding Service** (OpenAI-compatible)
   ```bash
   # Could be a local model server or cloud API
   ```

3. **Generative Service** (OpenAI-compatible)
   ```bash
   # Could be a local model server or cloud API
   ```

## Management and Monitoring

### Health Checks

Run comprehensive health checks:

```bash
# Basic health check
./scripts/health-check.sh

# Verbose health check
./scripts/health-check.sh --verbose

# With custom config
./scripts/health-check.sh --config /path/to/config.yaml
```

The health check verifies:
- Process status
- Service endpoints
- System resources
- Configuration validity

### Monitoring

#### Metrics Endpoint

Access Prometheus metrics at `http://localhost:9090/metrics`:

```
# HELP query_count_total Total number of queries processed
# TYPE query_count_total counter
query_count_total 42

# HELP query_latency_seconds Query processing latency
# TYPE query_latency_seconds histogram
query_latency_seconds_bucket{le="0.1"} 10
```

#### Log Monitoring

```bash
# Follow logs in real-time
./scripts/launch.sh logs -f

# View last 100 lines
./scripts/launch.sh logs 100

# Search logs for errors
grep "ERROR" logs/orchestrator.log
```

### API Endpoints

#### Query Endpoint
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "collection": "default",
    "limit": 5
  }'
```

#### Document Ingestion
```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "doc1",
        "content": "Machine learning is a subset of AI...",
        "metadata": {"source": "wikipedia"}
      }
    ]
  }'
```

#### Health Check
```bash
curl http://localhost:8080/health
```

## Testing

### Running Tests

```bash
# Run all tests
cd build && ctest

# Run specific test
./build/bin/hartonomous-tests --gtest_filter="*OrchestratorTest*"

# Run with verbose output
./build/bin/hartonomous-tests --gtest_output=xml:test_results.xml
```

### Test Coverage

The test suite includes:

- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end functionality testing
- **Performance Tests**: Latency and throughput benchmarks
- **Configuration Tests**: Config file parsing and validation

### Adding Tests

Tests are located in `tests/` directory. Example test structure:

```cpp
// tests/test_your_component.cpp
#include <gtest/gtest.h>
#include "your_component.hpp"

TEST(YourComponentTest, BasicFunctionality) {
    // Test implementation
}
```

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t hartonomous-orchestrator .

# Run container
docker run -p 8080:8080 -p 9090:9090 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  hartonomous-orchestrator
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -l app=hartonomous-orchestrator

# View logs
kubectl logs -l app=hartonomous-orchestrator
```

### Production Considerations

1. **Security**: Use HTTPS, API key authentication
2. **Monitoring**: Set up Prometheus and Grafana
3. **Logging**: Configure log aggregation (ELK stack)
4. **Scaling**: Use Kubernetes HPA for autoscaling
5. **Backup**: Regular backups of vector database

## Troubleshooting

### Common Issues

#### Build Failures

**Problem**: Missing dependencies
```
Solution: ./scripts/build.sh --verbose
Check that all required libraries are installed
```

**Problem**: CMake configuration fails
```
Solution: rm -rf build && ./scripts/build.sh
Check CMake version (3.20+ required)
```

#### Runtime Issues

**Problem**: Services not reachable
```
Check: ./scripts/health-check.sh
Verify service endpoints in config.yaml
Check network connectivity
```

**Problem**: High memory usage
```
Check: ./scripts/health-check.sh
Monitor with: htop or top
Consider reducing batch sizes in config
```

**Problem**: Slow query responses
```
Check: ./scripts/health-check.sh
Verify service response times
Consider optimizing embedding dimensions
```

### Log Analysis

```bash
# Search for errors
grep "ERROR" logs/orchestrator.log | tail -10

# Check for warnings
grep "WARNING" logs/orchestrator.log | tail -10

# Monitor performance
grep "latency" logs/orchestrator.log | tail -20
```

### Debug Mode

Run in debug mode for detailed logging:

```bash
# Set log level to debug
export LOG_LEVEL=debug

# Start with debug build
BUILD_TYPE=Debug ./scripts/launch.sh start
```

## Performance Tuning

### Configuration Tuning

```yaml
# Optimized settings
server:
  max_connections: 500
  request_timeout: 45

services:
  embedding:
    timeout: 20
    max_retries: 2

  generative:
    timeout: 90
    max_retries: 1
```

### System Optimization

1. **Memory**: Increase system memory for larger models
2. **CPU**: Use multi-core systems for parallel processing
3. **Network**: Ensure low-latency connections to services
4. **Storage**: Use SSD storage for vector database

### Monitoring Performance

```bash
# Check system resources
./scripts/health-check.sh

# Monitor query latency
curl http://localhost:9090/metrics | grep query_latency

# Profile with perf (Linux)
perf record -g ./build/bin/hartonomous-orchestrator
perf report
```

## Support and Contributing

### Getting Help

1. Check the [README.md](README.md) for basic information
2. Review [architecture.md](architecture.md) for design details
3. Check logs and health status
4. Open an issue with detailed information

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd hartonomous-orchestrator

# Set up development environment
./scripts/build.sh --build-type Debug

# Run tests continuously during development
watch -n 30 ctest