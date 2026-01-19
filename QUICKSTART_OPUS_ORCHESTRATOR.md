# Quickstart: Opus + Orchestrator Integration

**Get your AI talking to your database in 5 steps**

## What You'll Get

Talk to your database like you're talking to ChatGPT/Claude:
- Ingest MiniLM model + Moby Dick text
- Query via OpenAI-compatible API
- Get intelligent responses from your own data substrate

## Prerequisites

1. **PostgreSQL** running with Opus schema
2. **llama.cpp servers** (or compatible):
   - Generative model (port 8710)
   - Embedding model (port 8711)
   - Reranker model (port 8712)

## Step 1: Set Up Database

```bash
# Ensure PostgreSQL is running
psql -U postgres -h localhost -p 5432

# Deploy Opus schema
psql -U postgres -h localhost -p 5432 -d hypercube -f sql/deploy/full_schema.sql
```

**Verify**:
```sql
-- Should see composition, relation_evidence, atom_calculator tables
\dt
```

## Step 2: Build Opus Ingesters

```bash
cd cpp

# Clean build with proper AVX detection
rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Verify executables exist
ls build/bin/Release/  # Windows: ingest_safetensor.exe, etc.
```

## Step 3: Ingest Content

### 3a. Ingest MiniLM Model

```bash
# Download MiniLM safetensor (if you don't have it)
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Ingest the model
cd build/bin/Release
./ingest_safetensor.exe \
  --model-path "C:/models/all-MiniLM-L6-v2/model.safetensors" \
  --model-name "minilm" \
  --db-connection "postgresql://postgres:postgres@localhost:5432/hypercube"
```

Expected output:
```
[INGEST] Processing safetensor: all-MiniLM-L6-v2
[INGEST] Found 85 tensors
[INGEST] Inserted 85 compositions
[INGEST] Created 1200 embedding relations
[INGEST] Complete!
```

### 3b. Ingest Moby Dick Text

**Option 1: Via Opus Ingest Tool** (if you have text ingester built):
```bash
./ingest_text.exe \
  --file "path/to/moby_dick.txt" \
  --model-name "minilm" \
  --embedding-server "http://localhost:8711" \
  --chunk-size 512 \
  --db-connection "postgresql://postgres:postgres@localhost:5432/hypercube"
```

**Option 2: Via Python Script** (manual):
```python
import psycopg2
import httpx

# Connect to database
conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/hypercube")

# Read Moby Dick
with open("moby_dick.txt") as f:
    text = f.read()

# Chunk it (simple sentence-based chunking)
chunks = text.split('. ')

# Get embeddings and insert
for i, chunk in enumerate(chunks[:100]):  # First 100 chunks for testing
    if not chunk.strip():
        continue

    # Get embedding from llama.cpp
    response = httpx.post("http://localhost:8711/v1/embeddings", json={
        "input": chunk,
        "model": "minilm"
    })
    embedding = response.json()["data"][0]["embedding"]

    # Insert into composition table
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO composition (model, layer, component, embedding, metadata)
            VALUES ('moby-dick', 0, 'chunk', %s::vector, %s::jsonb)
        """, (
            '[' + ','.join(map(str, embedding)) + ']',
            '{"text": "' + chunk.replace('"', '\\"') + '", "source": "moby_dick", "chunk_id": ' + str(i) + '}'
        ))

conn.commit()
print("Ingested Moby Dick chunks!")
```

**Verify Ingestion**:
```sql
-- Check composition count
SELECT COUNT(*) FROM composition;
-- Should see: MiniLM tensors + Moby Dick chunks

-- Check that embeddings exist
SELECT COUNT(*) FROM composition WHERE embedding IS NOT NULL;

-- Sample a Moby Dick chunk
SELECT model, metadata->>'text' AS text
FROM composition
WHERE model = 'moby-dick'
LIMIT 1;
```

## Step 4: Set Up Orchestrator

```bash
cd Hartonomous-Orchestrator

# Install dependencies (includes psycopg2-binary now)
pip install -r requirements.txt

# Set environment variables
cat > .env << EOF
# Opus PostgreSQL database
POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/hypercube
USE_OPUS_DB=true

# llama.cpp backends
GENERATIVE_URL=http://localhost:8710
EMBEDDING_URL=http://localhost:8711
RERANKER_URL=http://localhost:8712

# RAG configuration
RAG_ENABLED=true
RAG_TOP_K=10
RAG_RERANK_TOP_N=3
RAG_MIN_RATING=1000.0
RAG_MAX_HOPS=2
EOF

# Start the gateway
python openai_gateway.py
```

Expected output:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Opus PostgreSQL client initialized and ready
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8700
```

## Step 5: Test Query Flow

### 5a. Health Check

```bash
curl http://localhost:8700/health
```

Expected:
```json
{
  "status": "healthy",
  "backends": {
    "generative": "up",
    "embedding": "up",
    "reranker": "up"
  },
  "vector_store": {}
}
```

### 5b. Database Stats

```bash
curl http://localhost:8700/v1/opus/stats
```

Add this endpoint to `openai_gateway/routes/collections.py`:
```python
@router.get("/v1/opus/stats")
async def get_opus_stats():
    """Get Opus database statistics"""
    from ..clients.opus_postgres_client import get_opus_client
    opus_client = get_opus_client()
    return opus_client.get_stats()
```

### 5c. Chat Completion (THE MOMENT OF TRUTH)

```bash
curl http://localhost:8700/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder",
    "messages": [
      {"role": "user", "content": "Who is Captain Ahab?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

**What Should Happen**:
1. Orchestrator gets your query: "Who is Captain Ahab?"
2. Calls embedding server → gets query embedding
3. Queries Opus PostgreSQL composition table → semantic search
4. Finds Moby Dick chunks about Ahab
5. Optionally expands via relations
6. Reranks with reranker model
7. Injects context into generative model
8. Returns response with Opus database context

Expected response:
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "qwen3-coder",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Captain Ahab is the monomaniacal captain of the whaling ship Pequod in Herman Melville's novel Moby-Dick. He is obsessed with hunting the white whale Moby Dick, who took his leg in a previous encounter..."
    }
  }]
}
```

### 5d. Verify RAG Is Working

Check the Orchestrator logs - you should see:
```
INFO: Using Opus PostgreSQL for RAG search
INFO: Got query embedding: 384d
INFO: Opus semantic search returned 15 results
INFO: Expanding via relations (max 2 hops)
INFO: Extracted 15 documents for reranking
INFO: Reranked to top 3 documents
```

## Step 6: Use with Roo Code / Cursor

Configure your AI coding assistant:

**Roo Code**:
```
API Endpoint: http://localhost:8700
Model: qwen3-coder
API Key: (leave empty)
```

**Cursor**:
```json
{
  "models": [{
    "provider": "openai",
    "apiBase": "http://localhost:8700",
    "model": "qwen3-coder"
  }]
}
```

Now when you ask "What's in my database?" or "Explain the Moby Dick content", it queries YOUR Opus substrate!

## Troubleshooting

### Issue: "Opus database client not initialized"
**Fix**: Check POSTGRES_URL is correct and database is accessible
```bash
psql "postgresql://postgres:postgres@localhost:5432/hypercube" -c "SELECT 1"
```

### Issue: "No results from Opus semantic search"
**Possible causes**:
1. No embeddings in database
   ```sql
   SELECT COUNT(*) FROM composition WHERE embedding IS NOT NULL;
   ```
2. Embedding dimension mismatch
   ```sql
   -- Check vector dimension
   SELECT vector_dims(embedding) FROM composition LIMIT 1;
   ```

**Fix**: Ensure ingestion completed and embeddings match your model dimensions

### Issue: "Failed to get embedding"
**Fix**: Check embedding server is running
```bash
curl http://localhost:8711/health
```

### Issue: "No text content found in results"
**Fix**: Ensure metadata has 'text' or 'content' field
```sql
SELECT metadata FROM composition WHERE model = 'moby-dick' LIMIT 1;
-- Should see: {"text": "...", "source": "moby_dick"}
```

## Next Steps

1. **Ingest More Content**:
   - More books/documents
   - Code repositories
   - Research papers
   - Model weights

2. **Enable Relation Traversal**:
   - Set `RAG_MAX_HOPS=3` for deeper searches
   - Adjust `RAG_MIN_RATING` to filter low-quality relations

3. **Add Agentic Endpoints**:
   - Hypercube spatial queries
   - Multi-hop reasoning
   - Graph visualization

4. **Performance Tuning**:
   - Add HNSW indexes on embeddings
   - Tune `RAG_TOP_K` and `RAG_RERANK_TOP_N`
   - Consider connection pooling

## Success Criteria

- ✅ Database has compositions with embeddings
- ✅ Orchestrator connects to Opus PostgreSQL
- ✅ Chat completions include Moby Dick context
- ✅ Logs show "Opus semantic search" working
- ✅ Responses reference your ingested content

**You're now talking to your database substrate!**
