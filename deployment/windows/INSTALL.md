# Hartonomous Orchestrator - Windows Installation

## Prerequisites

- **Python 3.10+** installed and in PATH
- **PostgreSQL access** to remote server (hart-server:5432)
- **Network connectivity** to hart-server

## Installation Steps

### 1. Verify DLLs

Check that all required DLLs are present in `bin/`:

```cmd
dir bin\*.dll
```

**Required DLLs**:
- `embedding_c.dll` - SIMD vector operations
- `generative_c.dll` - Text generation engine
- `hypercube_c.dll` - 4D geometry operations
- `hypercube.dll` - Core hypercube library
- `generative.dll` - Core generative library
- `libpq.dll` - PostgreSQL client
- `libcrypto-3-x64.dll` - OpenSSL crypto
- `libssl-3-x64.dll` - OpenSSL SSL

### 2. Add DLLs to System PATH

**Option A: Set Environment Variable (Recommended)**

```cmd
# Open System Properties > Environment Variables
# Add to User or System PATH:
C:\path\to\deployment\windows\bin
```

**Option B: Copy to System Directory**

```cmd
# Copy DLLs to Windows System32 (requires admin)
copy bin\*.dll C:\Windows\System32\
```

**Option C: Run from Deployment Directory**

Set PATH temporarily in each session:
```cmd
set PATH=C:\path\to\deployment\windows\bin;%PATH%
```

### 3. Set Up Python Environment

```cmd
cd orchestrator

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment

```cmd
# Copy config template
copy ..\..\..\shared\config\.env.template .env

# Edit .env with your settings
notepad .env
```

**Update these values in `.env`**:

```ini
POSTGRES_HOST=hart-server
POSTGRES_PORT=5432
POSTGRES_DB=hypercube
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_URL=postgresql://postgres:postgres@hart-server:5432/hypercube

# Set to absolute path of bin directory
HARTONOMOUS_BIN_PATH=C:\path\to\deployment\windows\bin

USE_OPUS_DB=true
ORCHESTRATOR_PORT=8700
RAG_ENABLED=true
```

### 5. Test Database Connection

```cmd
# Test PostgreSQL connectivity
python -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:postgres@hart-server:5432/hypercube'); print('Connected!'); conn.close()"
```

If connection fails:
- Verify `hart-server` is reachable: `ping hart-server`
- Check PostgreSQL is listening on 5432
- Verify PostgreSQL `pg_hba.conf` allows remote connections
- Check firewall allows port 5432

### 6. Test DLL Loading

```cmd
python -c "from openai_gateway.clients.hartonomous_client import get_hartonomous_client; client = get_hartonomous_client(); print('SUCCESS: DLLs loaded')"
```

**If DLL loading fails**:

1. Verify PATH includes bin directory:
   ```cmd
   echo %PATH%
   ```

2. Check for missing dependencies:
   ```cmd
   # Use Dependency Walker or similar tool
   # Download from: https://www.dependencywalker.com/
   depends.exe bin\hypercube_c.dll
   ```

3. Verify Visual C++ Redistributables installed:
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and reboot

### 7. Test Cache Loading

```cmd
python -c "from openai_gateway.clients.hartonomous_loader import initialize_hartonomous_caches, get_cache_stats; initialize_hartonomous_caches(); print(get_cache_stats())"
```

**Expected output**:
```
INFO: Loading vocabulary from database...
INFO: Loaded XXX vocabulary entries (vocab size: XXX)
INFO: Loading bigram (PMI) cache...
INFO: Loaded XXX bigram entries
INFO: Loading attention cache...
INFO: Loaded XXX attention entries
{'loaded': True, 'vocab_size': XXX, ...}
```

**If vocab_size is 0**:
- Compositions in database lack `label` field
- Run ingestion to populate composition metadata
- Check: `SELECT COUNT(*) FROM composition WHERE label IS NOT NULL;` on hart-server

### 8. Start Orchestrator

```cmd
# Make sure virtual environment is activated
venv\Scripts\activate

# Start server
python openai_gateway.py
```

**Expected startup logs**:
```
INFO: Initializing OpenAI Gateway...
INFO: Found embedding_c at ...
INFO: Found generative_c at ...
INFO: Found hypercube_c at ...
INFO: Loading Hartonomous caches from database...
INFO: Loaded XXX vocabulary entries (vocab size: XXX)
INFO: Hartonomous caches loaded successfully
INFO: Uvicorn running on http://0.0.0.0:8700
```

### 9. Test API Endpoint

Open a new terminal:

```cmd
curl http://localhost:8700/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"hartonomous\", \"messages\": [{\"role\": \"user\", \"content\": \"What is a whale?\"}], \"max_tokens\": 50}"
```

**Expected response**:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "hartonomous",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "[Generated text...]"
    },
    "finish_reason": "stop"
  }]
}
```

### 10. Run as Windows Service (Optional)

Create `hartonomous-orchestrator.xml` for NSSM:

```xml
<service>
  <id>HartonomousOrchestrator</id>
  <name>Hartonomous Orchestrator</name>
  <description>Hartonomous OpenAI-compatible gateway</description>
  <executable>C:\path\to\deployment\windows\orchestrator\venv\Scripts\python.exe</executable>
  <arguments>openai_gateway.py</arguments>
  <workingdirectory>C:\path\to\deployment\windows\orchestrator</workingdirectory>
  <env name="PATH" value="C:\path\to\deployment\windows\bin;%PATH%"/>
  <logmode>rotate</logmode>
</service>
```

Install with NSSM:
```cmd
nssm install HartonomousOrchestrator C:\path\to\orchestrator\venv\Scripts\python.exe openai_gateway.py
nssm set HartonomousOrchestrator AppDirectory C:\path\to\orchestrator
nssm set HartonomousOrchestrator AppEnvironmentExtra PATH=C:\path\to\bin;%PATH%
nssm start HartonomousOrchestrator
```

## Troubleshooting

### DLL Load Errors

**"Could not find module ... (or one of its dependencies)"**

1. Run `where <dll_name>.dll` to verify PATH resolution
2. Use Dependency Walker to find missing dependencies
3. Ensure Visual C++ Redistributables installed
4. Copy `libpq.dll`, `libcrypto-3-x64.dll`, `libssl-3-x64.dll` to same directory as C bridge DLLs

### Database Connection Errors

**"could not connect to server"**

1. Verify network: `ping hart-server`
2. Check PostgreSQL running: `ssh ahart@hart-server "systemctl status postgresql"`
3. Check port open: `Test-NetConnection -ComputerName hart-server -Port 5432`
4. Verify `pg_hba.conf` on hart-server allows your IP

### Empty Vocabulary Cache

**"vocab_size: 0"**

Database compositions need labels. On hart-server:

```sql
-- Check composition labels
SELECT COUNT(*) as total,
       COUNT(*) FILTER (WHERE label IS NOT NULL) as with_labels
FROM composition;

-- Check metadata
SELECT COUNT(*) FROM composition WHERE metadata->>'text' IS NOT NULL;
```

If empty, run ingestion on hart-server to populate data.

### API Returns Errors

Check logs in `openai_gateway.log` for detailed error messages.

## Next Steps

- Configure Cursor/ChatGPT to use http://localhost:8700 as OpenAI endpoint
- Test chat completions with various queries
- Monitor performance and tune cache settings
- Scale: Add read replicas to PostgreSQL for higher throughput

## Architecture

```
Windows Client
    ↓
Orchestrator (FastAPI on 8700)
    ↓ (ctypes)
C++ DLLs (embedding_c, generative_c, hypercube_c)
    ↓ (libpq)
PostgreSQL on hart-server:5432
```

**THE DATABASE IS THE MODEL**
- No llama.cpp
- No neural networks
- No forward passes
- Pure spatial queries + relation traversal
- O(log N) + O(K) complexity
