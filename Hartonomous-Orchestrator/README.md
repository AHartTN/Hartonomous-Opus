# OpenAI-compatible RAG Gateway

**OpenAI-compatible API gateway with automatic RAG orchestration and OpenAPI documentation**

Turns your llama.cpp servers into a full-featured RAG system. Chat completions automatically search your knowledge base, rerank results, and inject context - no extra API calls needed.

## Features

- **Automatic RAG** - Chat completions intelligently search knowledge base and inject context
- **OpenAI-compatible** - Drop-in replacement for OpenAI API
- **Full parameter support** - seed, logprobs, penalties, streaming, JSON mode, etc.
- **Vector search** - Qdrant integration for fast semantic search
- **Smart reranking** - Uses your reranker model to surface best results
- **Document ingestion** - Auto-chunking with tiktoken
- **Minimal dependencies** - ~100MB total (FastAPI, Qdrant client, etc.)
- **~600 lines of code** - Readable, hackable, no magic

## Quick Start

```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

Gateway runs on `http://localhost:8700`

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8700/docs
- **ReDoc**: http://localhost:8700/redoc
- **OpenAPI JSON**: http://localhost:8700/openapi.json

All endpoints are fully documented with request/response schemas, parameter descriptions, and example usage.

## Prerequisites

Running services on your host:
- **Llama.cpp generative** - Port 8710
- **Llama.cpp embedding** - Port 8711
- **Llama.cpp reranker** - Port 8712
- **Qdrant vector DB** - Port 6333

## Environment Variables

```bash
# Llama.cpp backends
GENERATIVE_URL=http://localhost:8710
EMBEDDING_URL=http://localhost:8711
RERANKER_URL=http://localhost:8712
BACKEND_API_KEY=Welcome!123

# Qdrant vector store
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# RAG configuration
RAG_ENABLED=true              # Enable/disable auto-RAG
RAG_TOP_K=10                  # Vector search results
RAG_RERANK_TOP_N=3            # Top results after reranking
COLLECTION_NAME=knowledge_base
CHUNK_SIZE=512                # Tokens per chunk
CHUNK_OVERLAP=50              # Overlap between chunks
VECTOR_SIZE=4096              # Embedding dimension
```

## How It Works

### Automatic RAG in Chat Completions

When you send a chat completion request:

1. **Query extraction** - Extracts last user message as search query
2. **Embedding** - Generates embedding via your embedding server (8711)
3. **Vector search** - Searches Qdrant for top-k similar documents
4. **Reranking** - Reranks results via your reranker server (8712)
5. **Context injection** - Injects top-n documents as system message
6. **Generation** - Generates response via your generative server (8710)

All of this happens **automatically** in a single `/v1/chat/completions` call.

### Control RAG Behavior

```python
# Disable RAG for specific request
response = client.chat.completions.create(
    model="qwen3-coder-30b",
    messages=[{"role": "user", "content": "Hello"}],
    rag_enabled=False  # Skip RAG for this request
)

# Customize RAG parameters
response = client.chat.completions.create(
    model="qwen3-coder-30b",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    rag_top_k=20,        # Retrieve more candidates
    rag_rerank_top_n=5   # Inject more context
)
```

## API Endpoints

### Chat Completions (with Auto-RAG)

```bash
curl http://localhost:8700/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "temperature": 0.7,
    "max_tokens": 500,
    "stream": false
  }'
```

**Supported parameters:**
- `model`, `messages`, `temperature`, `max_tokens`, `max_completion_tokens`
- `stream`, `stop`, `top_p`, `frequency_penalty`, `presence_penalty`
- `seed` (best-effort determinism)
- `logprobs`, `top_logprobs` (token probabilities)
- `n` (multiple completions)
- `user` (tracking)
- `tools`, `tool_choice`, `parallel_tool_calls` (function calling - structured)
- `response_format` (JSON mode)
- `rag_enabled`, `rag_top_k`, `rag_rerank_top_n` (RAG control)

### Embeddings

```bash
curl http://localhost:8700/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-4b",
    "input": "Your text here",
    "dimensions": 2048
  }'
```

**Supported parameters:**
- `input` (string or array)
- `dimensions` (reduce embedding size)
- `encoding_format` (float or base64)
- `user`

### Reranking

```bash
curl http://localhost:8700/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "documents": ["doc1", "doc2", "doc3"],
    "top_n": 2
  }'
```

### Document Ingestion

**Ingest text documents:**
```bash
curl http://localhost:8700/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["Your document text here..."],
    "metadata": [{"source": "manual", "topic": "AI"}],
    "chunk": true
  }'
```

**Ingest files:**
```bash
curl http://localhost:8700/v1/ingest/file \
  -F "file=@document.txt"
```

### Vector Search

```bash
curl http://localhost:8700/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "top_k": 5,
    "filter": {"topic": "AI"}
  }'
```

### Collection Management

**Get stats:**
```bash
curl http://localhost:8700/v1/collection/stats
```

**Clear collection:**
```bash
curl -X DELETE http://localhost:8700/v1/collection
```

### Other Endpoints

**List models:**
```bash
curl http://localhost:8700/v1/models
```

**Health check:**
```bash
curl http://localhost:8700/health
```

## Usage with Roo Code / Agentic Systems

### Roo Code Configuration

```
API Endpoint: http://localhost:8700
Model: qwen3-coder-30b
API Key: (leave empty)
```

### Compatible Tools

Works with any OpenAI-compatible client:
- **Roo Code** - Agentic coding assistant
- **Cursor** - AI code editor
- **Continue.dev** - VS Code extension
- **LangChain** - LLM framework
- **LlamaIndex** - RAG framework
- **OpenAI Python SDK** - Drop-in replacement

## Example Workflows

### 1. Build a Knowledge Base

```python
import httpx

BASE_URL = "http://localhost:8700"

# Ingest documents
docs = [
    "Python is a high-level programming language...",
    "Machine learning is a subset of AI...",
    "Neural networks are computing systems..."
]

response = httpx.post(f"{BASE_URL}/v1/ingest", json={
    "documents": docs,
    "metadata": [{"source": "docs"}] * len(docs),
    "chunk": True
})

print(response.json())
# {"status": "success", "chunks_created": 15, "collection_size": 15}
```

### 2. Query with Auto-RAG

```python
# Chat completion automatically searches knowledge base
response = httpx.post(f"{BASE_URL}/v1/chat/completions", json={
    "model": "qwen3-coder-30b",
    "messages": [
        {"role": "user", "content": "What is machine learning?"}
    ]
})

# Response includes context from ingested documents
print(response.json()["choices"][0]["message"]["content"])
```

### 3. Manual RAG Pipeline

```python
# 1. Search
search_response = httpx.post(f"{BASE_URL}/v1/search", json={
    "query": "neural networks",
    "top_k": 10
})

docs = [r["document"] for r in search_response.json()["results"]]

# 2. Rerank
rerank_response = httpx.post(f"{BASE_URL}/v1/rerank", json={
    "query": "neural networks",
    "documents": docs,
    "top_n": 3
})

top_docs = [r["document"] for r in rerank_response.json()["results"]]

# 3. Generate with context
response = httpx.post(f"{BASE_URL}/v1/chat/completions", json={
    "model": "qwen3-coder-30b",
    "messages": [
        {"role": "system", "content": f"Context: {' '.join(top_docs)}"},
        {"role": "user", "content": "Explain neural networks"}
    ],
    "rag_enabled": False  # Skip auto-RAG since we manually provided context
})
```

### 4. Streaming with RAG

```python
import httpx

with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions", json={
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "Tell me about Python"}],
    "stream": True
}, timeout=None) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            if line == "data: [DONE]":
                break
            import json
            chunk = json.loads(line[6:])
            content = chunk["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)
```

## Architecture

```
┌─────────────┐
│  Roo Code   │
│   Cursor    │  ← OpenAI-compatible API
│  LangChain  │
└──────┬──────┘
       │
       ↓
┌──────────────────────────────────────┐
│  RAG Gateway (port 8700)             │
│  ┌────────────────────────────────┐  │
│  │  /v1/chat/completions          │  │
│  │  1. Extract query              │  │
│  │  2. Embed → Qdrant search      │  │
│  │  3. Rerank results             │  │
│  │  4. Inject context             │  │
│  │  5. Generate response          │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Llama.cpp Servers (host)            │
│  ├─ Generative (8710)                │
│  ├─ Embedding  (8711)                │
│  └─ Reranker   (8712)                │
└──────────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Qdrant Vector DB (6333)             │
│  └─ knowledge_base collection        │
└──────────────────────────────────────┘
```

## Why This Is Better Than The Trash

| Trash Folder | This Gateway |
|--------------|--------------|
| 5GB+ dependencies (vcpkg, Boost) | ~100MB (FastAPI, Qdrant client) |
| C++ compilation hell | Pure Python, pip install |
| ~150 lines doing nothing | ~600 lines with full RAG |
| Single-purpose proxy | Multi-model orchestration |
| No vector search | Qdrant integration |
| No reranking | Smart reranking pipeline |
| No context injection | Automatic RAG |

## Performance Notes

- **Embedding**: ~50-200ms per query (depends on your hardware)
- **Vector search**: ~5-20ms (Qdrant is fast)
- **Reranking**: ~100-500ms for 10 docs (depends on model)
- **Total RAG overhead**: ~200-700ms before generation starts

Disable RAG (`rag_enabled=false`) for low-latency requests.

## Advanced Configuration

### Custom Chunking

Adjust chunk size and overlap based on your model's context window:

```bash
CHUNK_SIZE=1024      # Larger chunks for more context
CHUNK_OVERLAP=100    # More overlap for continuity
```

### Vector Dimensions

If using a different embedding model, update vector size:

```bash
VECTOR_SIZE=1536     # For text-embedding-ada-002
VECTOR_SIZE=768      # For smaller models
VECTOR_SIZE=4096     # For Qwen3-Embedding-4B (default)
```

### Multiple Collections

Use different collections for different knowledge domains:

```bash
COLLECTION_NAME=code_docs        # For code documentation
COLLECTION_NAME=research_papers  # For research papers
```

## Troubleshooting

**Gateway can't connect to Qdrant:**
- Check Qdrant is running: `curl http://localhost:6333/health`
- Verify `QDRANT_URL` in docker-compose.yml

**No RAG results found:**
- Check collection has documents: `curl http://localhost:8700/v1/collection/stats`
- Ingest documents via `/v1/ingest` endpoint

**Slow responses:**
- Reduce `RAG_TOP_K` to retrieve fewer candidates
- Reduce `RAG_RERANK_TOP_N` to rerank fewer documents
- Disable RAG for specific requests with `rag_enabled=false`

**Embedding dimension mismatch:**
- Set `VECTOR_SIZE` to match your embedding model's output dimension
- Delete and recreate collection if dimension changed

## License

MIT - Do whatever you want with it

## Credits

Built with:
- FastAPI - Web framework
- Qdrant - Vector database
- httpx - HTTP client
- tiktoken - Tokenization
- Your llama.cpp servers - The actual AI
