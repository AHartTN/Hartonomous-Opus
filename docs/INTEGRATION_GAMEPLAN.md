# THJIS DOCUMENT IS ALREADY OUT OF DATE BECAUSE IT INSULTS MY VISION BY THINKING WE ARE CALLING LLAMA.CPP INSTEAD OF REPLACING IT!

# Hartonomous Integration Gameplan

**Date**: 2026-01-18
**Projects**: Hartonomous-Opus (main), Hartonomous-Benchmark, Hartonomous-Orchestrator

## Executive Summary

This document outlines the strategic integration of three related projects into a cohesive system:
1. **Hartonomous-Opus** - Core C++ hypercube/semantic engine with PostgreSQL backend
2. **Hartonomous-Benchmark** - Hardware detection, SIMD benchmarking, performance validation
3. **Hartonomous-Orchestrator** - OpenAI-compatible API gateway with RAG orchestration

## Current State Analysis

### Hartonomous-Opus (cpp/)
**Purpose**: Core semantic engine with hypercube mathematics
- **Strengths**:
  - Robust SIMD dispatch system (AVX2, AVX-512, VNNI)
  - PostgreSQL integration for relation storage
  - Multi-model ingestion (safetensor, embedding relations, temporal, multimodal)
  - Runtime ISA detection and kernel selection
- **Weaknesses**:
  - CPU feature detection flaky on Windows/MSVC (FIXED in this session)
  - No performance benchmarking framework
  - No external API for agent integration
  - Limited testing/validation infrastructure

### Hartonomous-Benchmark (Hartonomous-Benchmark/)
**Purpose**: Enterprise benchmarking suite for HPC/ML workloads
- **Strengths**:
  - Comprehensive hardware detection (CPU, GPU, memory)
  - Google Benchmark integration
  - SIMD/AVX micro-benchmarks
  - Matrix ops, HNSW, FFT, RNG benchmarks
  - Automated result collection/analysis
- **Weaknesses**:
  - Linux-focused (Ubuntu primarily)
  - Hardcoded hardware detection (uses <cpuid.h> GCC builtin)
  - Standalone project, not integrated with Opus
  - No Windows MSVC support in hardware detection

### Hartonomous-Orchestrator (Hartonomous-Orchestrator/)
**Purpose**: OpenAI-compatible API gateway with automatic RAG
- **Strengths**:
  - Full OpenAI API compatibility (drop-in replacement)
  - Automatic RAG orchestration (search → rerank → context injection)
  - FastAPI with OpenAPI docs
  - Multi-model support (generative, embedding, reranker)
  - Qdrant vector store integration
  - Streaming support
- **Weaknesses**:
  - Python-based (potential performance bottleneck)
  - Uses llama.cpp backends (not Opus engine)
  - No connection to Opus PostgreSQL database
  - No hypercube/semantic features

## Integration Vision

### Phase 1: Benchmark Integration (IMMEDIATE)
**Goal**: Use Benchmark project to validate Opus SIMD/CPU detection

#### 1.1 CPU Feature Detection Unification
- **Action**: Extract Benchmark's `HardwareDetector` class
- **Location**: Move to `cpp/src/core/hardware_detection.cpp`
- **Benefits**:
  - Replace flaky cmake `try_run()` detection
  - Runtime validation of SIMD capabilities
  - Cross-platform (need Windows adaptation)
  - GPU detection capability

**Implementation Steps**:
1. Create `cpp/include/hypercube/hardware_detection.hpp`
2. Port `HardwareDetector` from Benchmark
3. Add MSVC `__cpuidex()` support (currently uses GCC `__get_cpuid`)
4. Add OS support check (XGETBV for AVX state saving)
5. Integrate with existing `cpu_features.cpp`
6. Add `hardware_check` CLI tool to Opus (like Benchmark)

**Files to Create/Modify**:
```
cpp/include/hypercube/hardware_detection.hpp     (NEW - unified detector)
cpp/src/core/hardware_detection.cpp              (NEW - implementation)
cpp/src/tools/hardware_check.cpp                 (NEW - CLI tool)
cpp/cmake/CompilerFlags.cmake                    (MODIFY - use runtime detection)
```

#### 1.2 SIMD Benchmarking Framework
- **Action**: Port key Benchmark micro-benchmarks to Opus
- **Target Operations**:
  - Dot product (float/double) - test dispatch.cpp kernels
  - Matrix multiply - test ops.cpp GEMM
  - HNSW operations - test query performance
  - Memory bandwidth - validate cache usage

**Implementation Steps**:
1. Add Google Benchmark dependency to Opus CMakeLists.txt
2. Create `cpp/benchmarks/` directory
3. Port micro-benchmarks for:
   - `simd_kernels_*.cpp` (AVX2, AVX-512, VNNI)
   - `ops.cpp` distance functions
   - HNSW query operations
4. Add benchmark target: `cmake --build build --target benchmarks`
5. Generate baseline results for i9-14900

**Files to Create**:
```
cpp/benchmarks/CMakeLists.txt                    (NEW)
cpp/benchmarks/bench_simd_dispatch.cpp           (NEW)
cpp/benchmarks/bench_distance_ops.cpp            (NEW)
cpp/benchmarks/bench_hnsw_query.cpp              (NEW)
cpp/benchmarks/bench_memory.cpp                  (NEW)
```

#### 1.3 Performance Regression Testing
- **Action**: Integrate benchmarks into CI/CD
- **Metrics**:
  - AVX2 vs AVX-512 speedup ratios
  - VNNI vs non-VNNI performance
  - Cache miss rates
  - Memory bandwidth utilization

**Implementation Steps**:
1. Add `scripts/run_benchmarks.sh` (Windows: `.bat`)
2. Store baseline results in `benchmarks/baselines/`
3. Add GitHub Actions workflow for benchmark runs
4. Generate performance reports (JSON/CSV)

### Phase 2: Orchestrator-Opus Bridge (HIGH PRIORITY)
**Goal**: Connect Orchestrator to Opus database and engine

#### 2.1 PostgreSQL Integration
**Current**: Orchestrator uses Qdrant vector store
**Target**: Orchestrator queries Opus PostgreSQL for semantic relations

**Architecture**:
```
Orchestrator (Python FastAPI)
    ↓
PostgreSQL (relation_evidence, composition, etc.)
    ↑
Opus Ingesters (C++)
```

**Implementation Steps**:
1. Add `psycopg2` to Orchestrator requirements.txt
2. Create `openai_gateway/clients/postgres_client.py`
3. Add config for PostgreSQL connection (reuse Opus DB)
4. Modify RAG search to query:
   - `composition` table for embeddings
   - `relation_evidence` for semantic relations
   - Use Opus's HNSW indexes
5. Deprecate Qdrant dependency (optional)

**SQL Queries Needed**:
```sql
-- Embedding search (replace Qdrant)
SELECT id, embedding <-> query_embedding AS distance
FROM composition
ORDER BY embedding <-> query_embedding
LIMIT :top_k;

-- Relation traversal (new capability)
SELECT source_id, target_id, relation_type, rating, raw_weight
FROM relation_evidence
WHERE source_id = :entity_id
ORDER BY rating DESC
LIMIT :max_relations;

-- Semantic context retrieval
SELECT c.id, c.model, c.layer, c.component, c.metadata
FROM composition c
JOIN relation_evidence r ON (c.id = r.source_id OR c.id = r.target_id)
WHERE r.source_id = :query_entity
ORDER BY r.rating DESC;
```

**Benefits**:
- Unified data store (no Qdrant dependency)
- Access to Opus relation graphs
- ELO-rated semantic relations
- Hypercube spatial queries

#### 2.2 C++ Engine Exposure
**Goal**: Expose Opus C++ engine to Orchestrator via C bridge

**Current**: Orchestrator calls llama.cpp servers
**Target**: Orchestrator calls Opus C bridges for:
- Embedding generation (via `embedding_c.dll`)
- Semantic search (via `hypercube_c.dll`)
- Relation traversal

**Implementation Steps**:
1. Create `cpp/src/bridge/rag_c.cpp` - RAG-specific C API
2. Functions to expose:
   ```cpp
   // Semantic search via hypercube
   extern "C" EXPORT void* hc_semantic_search(
       const char* query_text,
       int top_k,
       float* distances,
       void** result_ids
   );

   // Relation expansion
   extern "C" EXPORT void* hc_expand_relations(
       const void* entity_id,
       const char* relation_type,
       int max_depth,
       void** results
   );

   // Context assembly
   extern "C" EXPORT char* hc_assemble_context(
       const void** entity_ids,
       int count,
       int max_tokens
   );
   ```
3. Add Python ctypes bindings in Orchestrator:
   ```python
   # openai_gateway/clients/hypercube_client.py
   from ctypes import *

   class HypercubeClient:
       def __init__(self, dll_path):
           self.lib = CDLL(dll_path)
           # Load functions...

       def semantic_search(self, query: str, top_k: int):
           # Call hc_semantic_search

       def expand_relations(self, entity_id, relation_type):
           # Call hc_expand_relations
   ```
4. Modify Orchestrator RAG pipeline:
   ```python
   # openai_gateway/rag/search.py
   from ..clients.hypercube_client import hypercube_client

   async def rag_search(query: str, top_k: int, rerank_top_n: int):
       # Use Opus engine instead of Qdrant
       results = hypercube_client.semantic_search(query, top_k)
       # Rerank via existing reranker
       ranked = await rerank_results(query, results, rerank_top_n)
       return ranked
   ```

**Benefits**:
- Direct access to Opus performance (C++ SIMD)
- Hypercube semantic search
- No duplication of embedding generation
- Unified relation graph

#### 2.3 Agentic Extensions
**Goal**: Add agent-specific endpoints to Orchestrator

**New Endpoints**:
```python
# Hypercube spatial queries
POST /v1/hypercube/spatial_query
{
    "center_id": "entity_id",
    "radius": 0.5,
    "dimensions": [0, 1, 2, 3]  # 4D hypercube coords
}

# Relation graph traversal
POST /v1/relations/traverse
{
    "start_id": "entity_id",
    "relation_types": ["temporal", "semantic", "embedding"],
    "max_depth": 3,
    "min_rating": 1400.0  # ELO threshold
}

# Multi-hop semantic reasoning
POST /v1/semantic/multi_hop
{
    "query": "How does X relate to Y?",
    "max_hops": 5,
    "beam_width": 10
}
```

**Implementation**:
1. Create `openai_gateway/routes/hypercube.py`
2. Create `openai_gateway/routes/relations.py`
3. Implement graph traversal with ELO ranking
4. Add beam search for multi-hop reasoning

### Phase 3: Unified Build System (MEDIUM PRIORITY)
**Goal**: Single build system for all components

#### 3.1 CMake Superbuild
**Structure**:
```
D:/Repositories/Hartonomous-Opus/
├── CMakeLists.txt              (Superbuild - orchestrates all)
├── cpp/                        (Opus core)
│   └── CMakeLists.txt
├── Hartonomous-Benchmark/      (Optional add_subdirectory)
│   └── CMakeLists.txt
├── Hartonomous-Orchestrator/   (Python packaging)
│   ├── setup.py
│   └── pyproject.toml
└── scripts/
    ├── build-all.sh
    └── build-all.bat
```

**Superbuild CMakeLists.txt**:
```cmake
cmake_minimum_required(VERSION 3.24)
project(Hartonomous VERSION 1.0.0)

option(BUILD_OPUS "Build Opus C++ engine" ON)
option(BUILD_BENCHMARKS "Build benchmark suite" ON)
option(BUILD_ORCHESTRATOR "Install Orchestrator Python package" ON)

if(BUILD_OPUS)
    add_subdirectory(cpp)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(Hartonomous-Benchmark)
endif()

if(BUILD_ORCHESTRATOR)
    # Install Python package
    install(DIRECTORY Hartonomous-Orchestrator/openai_gateway
            DESTINATION lib/python/site-packages)
endif()

# Create unified install target
install(TARGETS hypercube_core hypercube_c embedding_c generative_c
        DESTINATION bin)
```

#### 3.2 Unified Testing
**Goal**: Single test runner for all components

```bash
# Run all tests (C++ + Python + Benchmarks)
./scripts/test-all.sh

# C++ unit tests
ctest -C Release

# Python tests
pytest Hartonomous-Orchestrator/

# Benchmarks (regression)
./scripts/run_benchmarks.sh --compare baseline
```

### Phase 4: Documentation & Deployment (ONGOING)
**Goal**: Comprehensive documentation for integrated system

#### 4.1 Architecture Documentation
**Create**:
```
docs/
├── architecture/
│   ├── system_overview.md
│   ├── data_flow.md
│   ├── api_reference.md
│   └── performance_tuning.md
├── integration/
│   ├── benchmark_integration.md
│   ├── orchestrator_integration.md
│   └── database_schema.md
└── deployment/
    ├── docker_compose.yml
    ├── kubernetes/
    └── systemd/
```

#### 4.2 Docker Compose Deployment
**Complete Stack**:
```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: hypercube
      POSTGRES_USER: hypercube
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - ./sql/deploy/full_schema.sql:/docker-entrypoint-initdb.d/schema.sql
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  opus-ingest:
    build:
      context: ./cpp
      dockerfile: Dockerfile.ingest
    depends_on:
      - postgres
    environment:
      DB_CONNECTION: postgres://postgres:5432/hypercube
    volumes:
      - ./data:/data

  orchestrator:
    build:
      context: ./Hartonomous-Orchestrator
    depends_on:
      - postgres
      - opus-ingest
    environment:
      DATABASE_URL: postgres://postgres:5432/hypercube
      HYPERCUBE_DLL: /usr/local/lib/hypercube_c.so
    ports:
      - "8700:8700"
    volumes:
      - ./cpp/build/lib:/usr/local/lib

volumes:
  pgdata:
```

## Implementation Priority Matrix

| Task | Impact | Effort | Priority | Timeline |
|------|--------|--------|----------|----------|
| CPU Detection Unification | High | Low | **P0** | Week 1 |
| SIMD Micro-benchmarks | High | Medium | **P0** | Week 1-2 |
| PostgreSQL Integration (Orchestrator) | High | Medium | **P1** | Week 2-3 |
| C++ Bridge for RAG | High | High | **P1** | Week 3-4 |
| Agentic Extensions | Medium | Medium | **P2** | Week 4-5 |
| Unified Build System | Medium | Low | **P2** | Week 5 |
| Docker Compose Stack | Medium | Low | **P3** | Week 6 |
| Documentation | High | Medium | **P3** | Ongoing |

## Success Metrics

### Performance
- [ ] Opus SIMD kernels show expected speedup (2-4x AVX2 vs scalar, 1.5-2x AVX-512 vs AVX2)
- [ ] Orchestrator RAG latency < 300ms (from Opus database)
- [ ] End-to-end query: < 500ms (search) + LLM generation time

### Functionality
- [ ] Orchestrator uses Opus PostgreSQL for all vector operations
- [ ] Agentic workflows can traverse relation graphs
- [ ] Hypercube spatial queries available via API
- [ ] Benchmark suite validates all SIMD paths

### Integration
- [ ] Single `docker compose up` starts entire stack
- [ ] Unified test suite (C++ + Python) passes
- [ ] OpenAPI docs reflect Opus extensions
- [ ] Performance baselines established

## Risk Mitigation

### Risk 1: Performance Regression from Python
**Mitigation**: Use C++ bridges for hot paths (search, embed)
**Fallback**: Keep llama.cpp backends as option

### Risk 2: PostgreSQL Vector Search Performance
**Mitigation**: Ensure HNSW indexes properly configured
**Fallback**: Keep Qdrant as alternative backend
**Test**: Benchmark both on realistic datasets

### Risk 3: Build Complexity
**Mitigation**: Incremental integration, keep projects buildable standalone
**Fallback**: Document manual build steps clearly

### Risk 4: API Breaking Changes
**Mitigation**: Version API endpoints (/v1/, /v2/)
**Rollback**: Keep old endpoints during transition

## Next Steps (Immediate Actions)

1. **This Session** (COMPLETED):
   - ✅ Fixed CPU feature detection in Opus
   - ✅ Analyzed Benchmark and Orchestrator codebases
   - ✅ Created this integration gameplan

2. **Next Session**:
   - [ ] Extract HardwareDetector from Benchmark
   - [ ] Add MSVC support to CPU detection
   - [ ] Create hardware_check CLI tool for Opus
   - [ ] Port first micro-benchmark (SIMD dot product)

3. **Week 1**:
   - [ ] Complete CPU detection unification
   - [ ] Port key micro-benchmarks
   - [ ] Establish performance baselines
   - [ ] Create PostgreSQL client in Orchestrator

4. **Week 2**:
   - [ ] Implement Opus PostgreSQL queries in Orchestrator
   - [ ] Begin C++ bridge development
   - [ ] Design agentic API extensions

## Conclusion

This integration creates a **unified agentic system**:
- **Performance**: Opus C++ engine with SIMD optimization
- **Validation**: Benchmark suite for continuous performance testing
- **Interface**: Orchestrator provides OpenAI-compatible API
- **Data**: Single PostgreSQL database with relation graphs
- **Capability**: Hypercube semantic search + RAG + relation traversal

The system will enable sophisticated agentic workflows while maintaining high performance and OpenAI compatibility.
