# Hartonomous Deployment Package

Pre-built deployment package for Hartonomous Orchestrator - an OpenAI-compatible API gateway for database-native AI inference.

## What's Included

```
deployment/
├── windows/
│   ├── bin/                    # DLLs for Windows
│   │   ├── embedding_c.dll
│   │   ├── generative_c.dll
│   │   ├── hypercube_c.dll
│   │   ├── hypercube.dll
│   │   ├── generative.dll
│   │   ├── libpq.dll
│   │   └── ...
│   ├── orchestrator/           # Python FastAPI gateway
│   │   ├── openai_gateway/
│   │   └── requirements.txt
│   └── INSTALL.md             # Windows installation guide
│
├── linux/
│   ├── bin/                    # Shared libraries for Linux (to be built)
│   │   ├── libembedding_c.so
│   │   ├── libgenerative_c.so
│   │   ├── libhypercube_c.so
│   │   └── ...
│   ├── orchestrator/           # Python FastAPI gateway
│   │   ├── openai_gateway/
│   │   └── requirements.txt
│   └── INSTALL.md             # Linux installation guide
│
└── shared/
    └── config/
        └── .env.template       # Environment configuration template
```

## Quick Start

### Windows

1. Navigate to `windows/INSTALL.md`
2. Follow step-by-step instructions
3. Key requirement: Add `windows/bin` to PATH

### Linux

1. Navigate to `linux/INSTALL.md`
2. Follow step-by-step instructions
3. Key requirement: Set `LD_LIBRARY_PATH` to `linux/bin`

## Architecture Overview

```
┌─────────────────────────────────────────┐
│ OpenAI-Compatible Client                │
│ (Cursor, ChatGPT, custom app)           │
└───────────────┬─────────────────────────┘
                │ HTTP POST /v1/chat/completions
                ↓
┌─────────────────────────────────────────┐
│ Hartonomous Orchestrator (Port 8700)    │
│ - FastAPI Gateway                       │
│ - Python ctypes bridge                  │
└───────────────┬─────────────────────────┘
                │ ctypes function calls
                ↓
┌─────────────────────────────────────────┐
│ C++ DLLs / Shared Libraries             │
│ - embedding_c: SIMD vector ops          │
│ - generative_c: Token generation        │
│ - hypercube_c: 4D geometry              │
└───────────────┬─────────────────────────┘
                │ SQL queries (libpq)
                ↓
┌─────────────────────────────────────────┐
│ PostgreSQL (hart-server:5432)           │
│ - composition table (4D coordinates)    │
│ - relation table (ELO-rated edges)      │
│ - B-tree indexes (Hilbert curves)       │
│ - R-tree indexes (spatial queries)      │
└─────────────────────────────────────────┘

THE DATABASE IS THE MODEL
```

## Key Concepts

### Database-Native AI

Traditional AI uses forward passes through neural network weights. Hartonomous uses **spatial queries and graph traversal** through a database:

- **No neural networks**: No matrix multiplication, no softmax
- **No forward passes**: No GPU inference
- **No training**: Ingestion IS training
- **O(log N) complexity**: Spatial indexing beats O(N²) attention

### 4D Spatial Geometry

Compositions exist as points in a 4D hypercube:
- **Coordinates**: (x, y, z, w) from Laplacian projection
- **Hilbert curves**: Derived FROM coordinates for B-tree locality
- **Spatial queries**: ST_Distance, ST_Intersects in 4D
- **R-tree indexes**: O(log N) nearest neighbor search

### ELO-Rated Relations

Intelligence emerges from **relations between compositions**:
- **Edges with ratings**: ELO scores (800-1600) indicate quality
- **Multiple types**: Embedding, Temporal, Semantic, Modal
- **Graph traversal**: Multi-hop queries through high-ELO edges
- **Cached**: Bigrams (PMI) and attention loaded into C++ on startup

### Token Generation

No forward pass. Instead, score candidates by:
1. **Centroid distance**: 4D Euclidean proximity
2. **PMI scores**: Statistical co-occurrence from bigrams
3. **Attention weights**: Learned associations from relations
4. **Global factors**: Frequency, Hilbert locality

Select next token via weighted combination + temperature sampling.

## Remote PostgreSQL Setup

This deployment assumes PostgreSQL runs on a remote server:

```
PostgreSQL Server: hart-server:5432
Database: hypercube
User: postgres
Password: postgres
```

### Required PostgreSQL Configuration

On `hart-server`, PostgreSQL must:

1. **Listen on network** (`postgresql.conf`):
   ```ini
   listen_addresses = '*'
   ```

2. **Allow remote connections** (`pg_hba.conf`):
   ```
   host    all    all    <client-ip>/32    md5
   ```

3. **Have required data**:
   - `composition` table with 4D coordinates and labels
   - `relation` table with ELO-rated edges
   - Spatial indexes (B-tree on Hilbert, R-tree on geometry)

### Verify Database

SSH to hart-server and check:

```sql
-- Check compositions
SELECT COUNT(*) as total,
       COUNT(*) FILTER (WHERE label IS NOT NULL) as with_labels,
       COUNT(*) FILTER (WHERE position_x IS NOT NULL) as with_coords
FROM composition;

-- Check relations
SELECT COUNT(*) as total,
       AVG(weight) as avg_elo
FROM relation;

-- Check indexes
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('composition', 'relation')
AND (indexname LIKE '%hilbert%' OR indexname LIKE '%geom%');
```

Expected:
- Compositions: Thousands+ with labels and 4D coordinates
- Relations: Thousands+ with ELO ratings
- Indexes: B-tree on hilbert_hi/lo, R-tree (GIST) on geometry

If empty, run ingestion on hart-server to populate data.

## Environment Variables

Copy `shared/config/.env.template` to `orchestrator/.env` and configure:

### Required

```ini
# PostgreSQL connection
POSTGRES_URL=postgresql://postgres:postgres@hart-server:5432/hypercube

# Use Hartonomous (not llama.cpp)
USE_OPUS_DB=true

# DLL/SO path (absolute)
HARTONOMOUS_BIN_PATH=/path/to/deployment/windows/bin  # or linux/bin
```

### Optional

```ini
# Orchestrator
ORCHESTRATOR_PORT=8700
LOG_LEVEL=INFO

# RAG
RAG_ENABLED=true
RAG_TOP_K=10
RAG_RERANK_TOP_N=3

# Cache thresholds
HARTONOMOUS_MIN_BIGRAM_RATING=1000.0
HARTONOMOUS_MIN_ATTENTION_RATING=1200.0
```

## Testing

### 1. Database Connection

```bash
# Windows
python -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:postgres@hart-server:5432/hypercube'); print('Connected'); conn.close()"

# Linux
python3 -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:postgres@hart-server:5432/hypercube'); print('Connected'); conn.close()"
```

### 2. DLL/Library Loading

```bash
# Windows
python -c "from openai_gateway.clients.hartonomous_client import get_hartonomous_client; client = get_hartonomous_client(); print('SUCCESS')"

# Linux
python3 -c "from openai_gateway.clients.hartonomous_client import get_hartonomous_client; client = get_hartonomous_client(); print('SUCCESS')"
```

### 3. Cache Loading

```bash
# Windows/Linux
python -c "from openai_gateway.clients.hartonomous_loader import initialize_hartonomous_caches, get_cache_stats; initialize_hartonomous_caches(); print(get_cache_stats())"
```

### 4. API Request

```bash
curl http://localhost:8700/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hartonomous", "messages": [{"role": "user", "content": "What is a whale?"}], "max_tokens": 50}'
```

## Using with AI Clients

### Cursor

1. Open Settings → Features → Language Models
2. Add custom OpenAI endpoint: `http://localhost:8700`
3. Set API key to any value (not validated)
4. Select "hartonomous" model

### ChatGPT (via OpenAI SDK)

```python
import openai

openai.api_base = "http://localhost:8700/v1"
openai.api_key = "not-needed"

response = openai.ChatCompletion.create(
    model="hartonomous",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Custom Application

Any OpenAI-compatible client can connect to `http://localhost:8700` and use model name `"hartonomous"`.

## Performance Characteristics

### Latency

- **Spatial query**: 1-10ms (R-tree traversal)
- **Relation lookup**: <1ms (B-tree + cache)
- **Multi-hop expansion**: 10-50ms (depends on hops and cache hits)
- **Token generation**: 5-20ms per token
- **Total**: 20-100ms per request (no GPU!)

### Throughput

- **Single instance**: 100-1000 requests/second (CPU-bound)
- **With PostgreSQL replicas**: Scales horizontally
- **Bottleneck**: Database I/O (use SSDs, add read replicas)

### Memory

- **Orchestrator**: 100-500 MB (Python + caches)
- **Vocabulary cache**: ~10 MB per 10K tokens
- **Bigram cache**: ~50 MB per 100K relations
- **Attention cache**: ~50 MB per 100K relations

### Scaling

- **Vertical**: More RAM for larger caches
- **Horizontal**: Multiple Orchestrator instances + PostgreSQL read replicas
- **Database**: Shard by Hilbert range or composition ID

## Troubleshooting

See platform-specific INSTALL.md files for detailed troubleshooting:
- `windows/INSTALL.md`
- `linux/INSTALL.md`

Common issues:
1. **DLL/Library not found**: Check PATH or LD_LIBRARY_PATH
2. **Database connection refused**: Check PostgreSQL configuration and firewall
3. **Empty vocabulary**: Run ingestion to populate database
4. **Slow queries**: Check database has spatial indexes
5. **Missing dependencies**: Install Visual C++ Redistributables (Windows) or libpq/openssl (Linux)

## Support

For issues or questions:
1. Check INSTALL.md for your platform
2. Review architecture documentation in main repository
3. Verify database schema and data on hart-server
4. Check Orchestrator logs for detailed error messages

## License

See main repository for license information.

## The Vision

**Traditional AI**: Data → Training → Model Weights → Inference

**Hartonomous**: Data → Ingestion → Database → Queries

No neural networks. No forward passes. No GPUs.

Just spatial queries (O(log N)) + relation traversal (O(K)).

**The database IS the model. The relations ARE the intelligence.**
