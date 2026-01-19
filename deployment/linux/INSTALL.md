# Hartonomous Orchestrator - Linux Installation

## Prerequisites

- **Python 3.10+** installed
- **PostgreSQL access** to remote server (hart-server:5432) or local
- **Build tools** if compiling from source: `gcc`, `g++`, `cmake`, `make`

## Installation Steps

### 1. Build C++ Shared Libraries (if not pre-built)

```bash
cd ../../Hartonomous-Opus/cpp

# Clean build
rm -rf build
mkdir build
cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
make -j$(nproc)

# Copy shared libraries to deployment
cp bin/*.so ../../../deployment/linux/bin/
```

**Or copy pre-built libraries**:
```bash
# If .so files are already built
cp /path/to/built/*.so deployment/linux/bin/
```

### 2. Verify Shared Libraries

```bash
cd deployment/linux
ls -lh bin/*.so
```

**Required libraries**:
- `libembedding_c.so` - SIMD vector operations
- `libgenerative_c.so` - Text generation engine
- `libhypercube_c.so` - 4D geometry operations
- `libhypercube.so` - Core hypercube library
- `libgenerative.so` - Core generative library

### 3. Set LD_LIBRARY_PATH

**Option A: System-wide (requires root)**

```bash
# Copy to system lib directory
sudo cp bin/*.so /usr/local/lib/
sudo ldconfig
```

**Option B: User-specific (Recommended)**

Add to `~/.bashrc` or `~/.profile`:

```bash
export HARTONOMOUS_BIN=/path/to/deployment/linux/bin
export LD_LIBRARY_PATH=$HARTONOMOUS_BIN:$LD_LIBRARY_PATH
```

Then reload:
```bash
source ~/.bashrc
```

**Option C: Per-session**

```bash
export LD_LIBRARY_PATH=/path/to/deployment/linux/bin:$LD_LIBRARY_PATH
```

### 4. Set Up Python Environment

```bash
cd orchestrator

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure Environment

```bash
# Copy config template
cp ../../shared/config/.env.template .env

# Edit configuration
nano .env  # or vim, emacs, etc.
```

**Update these values in `.env`**:

```ini
# PostgreSQL on remote server
POSTGRES_HOST=hart-server
POSTGRES_PORT=5432
POSTGRES_DB=hypercube
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_URL=postgresql://postgres:postgres@hart-server:5432/hypercube

# Or if running locally on hart-server itself:
# POSTGRES_HOST=localhost
# POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/hypercube

# Absolute path to bin directory
HARTONOMOUS_BIN_PATH=/path/to/deployment/linux/bin

USE_OPUS_DB=true
ORCHESTRATOR_HOST=0.0.0.0
ORCHESTRATOR_PORT=8700
RAG_ENABLED=true
LOG_LEVEL=INFO
```

### 6. Test Database Connection

```bash
# Test PostgreSQL connectivity
python3 -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:postgres@hart-server:5432/hypercube'); print('Connected!'); conn.close()"
```

If connection fails:

**From remote machine**:
```bash
# Test network connectivity
ping hart-server

# Test PostgreSQL port
nc -zv hart-server 5432
# or
telnet hart-server 5432

# SSH to hart-server and check PostgreSQL
ssh ahart@hart-server
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"
```

**On hart-server** (if PostgreSQL refuses connections):

1. Edit `/etc/postgresql/*/main/postgresql.conf`:
   ```ini
   listen_addresses = '*'  # or specific IPs
   ```

2. Edit `/etc/postgresql/*/main/pg_hba.conf`:
   ```
   # Allow connections from local network
   host    all             all             192.168.0.0/16          md5
   # Or allow from specific IP
   host    all             all             <your-client-ip>/32     md5
   ```

3. Restart PostgreSQL:
   ```bash
   sudo systemctl restart postgresql
   ```

4. Check firewall:
   ```bash
   sudo ufw allow 5432/tcp
   # or
   sudo iptables -A INPUT -p tcp --dport 5432 -j ACCEPT
   ```

### 7. Test Shared Library Loading

```bash
# Ensure LD_LIBRARY_PATH is set
echo $LD_LIBRARY_PATH

# Test loading
python3 -c "from openai_gateway.clients.hartonomous_client import get_hartonomous_client; client = get_hartonomous_client(); print('SUCCESS: Libraries loaded')"
```

**If library loading fails**:

1. Check dependencies:
   ```bash
   ldd bin/libhypercube_c.so
   ```

   Should show all dependencies resolved (not "not found").

2. Install missing system libraries:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y libpq5 libssl3 libstdc++6

   # RHEL/CentOS/Fedora
   sudo yum install -y postgresql-libs openssl-libs libstdc++

   # Arch
   sudo pacman -S postgresql-libs openssl
   ```

3. Check for missing symbols:
   ```bash
   nm -D bin/libhypercube_c.so | grep "U "
   ```

### 8. Test Cache Loading

```bash
python3 -c "from openai_gateway.clients.hartonomous_loader import initialize_hartonomous_caches, get_cache_stats; initialize_hartonomous_caches(); print(get_cache_stats())"
```

**Expected output**:
```
INFO: Loading vocabulary from database...
INFO: Loaded XXX vocabulary entries (vocab size: XXX)
INFO: Loading bigram (PMI) cache...
INFO: Loaded XXX bigram entries
INFO: Loading attention cache...
INFO: Loaded XXX attention entries
{'loaded': True, 'vocab_size': XXX, 'bigram_count': XXX, 'attention_edges': XXX}
```

**If vocab_size is 0**:

SSH to hart-server and check database:

```bash
ssh ahart@hart-server
sudo -u postgres psql hypercube

-- Check composition labels
SELECT COUNT(*) as total,
       COUNT(*) FILTER (WHERE label IS NOT NULL) as with_labels
FROM composition;

-- Check metadata
SELECT COUNT(*) FROM composition WHERE metadata->>'text' IS NOT NULL;

-- Check relations
SELECT COUNT(*) FROM relation WHERE weight >= 1000.0;
```

If data is empty, run ingestion on hart-server to populate database.

### 9. Start Orchestrator

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Verify LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Start server
python3 openai_gateway.py
```

**Expected startup logs**:
```
INFO: Initializing OpenAI Gateway...
INFO: Found embedding_c at /path/to/deployment/linux/bin/libembedding_c.so
INFO: Found generative_c at /path/to/deployment/linux/bin/libgenerative_c.so
INFO: Found hypercube_c at /path/to/deployment/linux/bin/libhypercube_c.so
INFO: Loading Hartonomous caches from database...
INFO: Loaded XXX vocabulary entries (vocab size: XXX)
INFO: Loaded XXX bigram entries
INFO: Loaded XXX attention entries
INFO: Hartonomous caches loaded successfully
INFO: Uvicorn running on http://0.0.0.0:8700
```

### 10. Test API Endpoint

Open a new terminal:

```bash
curl http://localhost:8700/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hartonomous",
    "messages": [
      {"role": "user", "content": "What is a whale?"}
    ],
    "max_tokens": 50
  }'
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
      "content": "[Generated text from database...]"
    },
    "finish_reason": "stop"
  }]
}
```

### 11. Run as systemd Service (Optional)

Create `/etc/systemd/system/hartonomous-orchestrator.service`:

```ini
[Unit]
Description=Hartonomous Orchestrator - OpenAI-compatible Gateway
After=network.target

[Service]
Type=simple
User=ahart
Group=ahart
WorkingDirectory=/path/to/deployment/linux/orchestrator
Environment="LD_LIBRARY_PATH=/path/to/deployment/linux/bin"
Environment="PATH=/path/to/deployment/linux/orchestrator/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/path/to/deployment/linux/orchestrator/venv/bin/python3 openai_gateway.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable hartonomous-orchestrator
sudo systemctl start hartonomous-orchestrator
sudo systemctl status hartonomous-orchestrator

# View logs
sudo journalctl -u hartonomous-orchestrator -f
```

## If Running on hart-server Itself

If deploying directly on hart-server (the PostgreSQL host):

1. Use `localhost` for database connection:
   ```ini
   POSTGRES_HOST=localhost
   POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/hypercube
   ```

2. No need to configure remote access in `pg_hba.conf`

3. Shared libraries may already be built in `/path/to/Hartonomous-Opus/cpp/build/bin`

4. Expose port 8700 to network for clients:
   ```bash
   sudo ufw allow 8700/tcp
   ```

## Troubleshooting

### Library Load Errors

**"cannot open shared object file: No such file or directory"**

1. Verify LD_LIBRARY_PATH:
   ```bash
   echo $LD_LIBRARY_PATH
   ```

2. Check library exists:
   ```bash
   ls -l $LD_LIBRARY_PATH/*.so
   ```

3. Check dependencies:
   ```bash
   ldd bin/libhypercube_c.so
   ```

4. Add to ldconfig (system-wide):
   ```bash
   echo "/path/to/deployment/linux/bin" | sudo tee /etc/ld.so.conf.d/hartonomous.conf
   sudo ldconfig
   ```

### Database Connection Errors

**"could not connect to server"**

See step 6 above for detailed troubleshooting.

Quick checks:
```bash
# Network
ping hart-server

# PostgreSQL port
nc -zv hart-server 5432

# PostgreSQL service
ssh ahart@hart-server "systemctl status postgresql"

# Firewall
ssh ahart@hart-server "sudo ufw status | grep 5432"
```

### Empty Vocabulary Cache

**"vocab_size: 0"**

Database needs ingestion. On hart-server:

```bash
cd /path/to/Hartonomous-Opus
./scripts/linux/ingest-testdata.sh
```

### Permission Errors

If running as non-root and hitting permission issues:

```bash
# Make sure user owns deployment directory
sudo chown -R $USER:$USER /path/to/deployment

# Make libraries readable
chmod +r bin/*.so
```

## Performance Tuning

### PostgreSQL Connection Pool

Edit `.env`:
```ini
# Increase connection pool size for high traffic
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
```

### Cache Settings

Adjust ELO rating thresholds:
```ini
# Lower thresholds = more relations cached (more memory, richer context)
HARTONOMOUS_MIN_BIGRAM_RATING=800.0
HARTONOMOUS_MIN_ATTENTION_RATING=1000.0

# Higher thresholds = fewer relations cached (less memory, higher quality)
HARTONOMOUS_MIN_BIGRAM_RATING=1200.0
HARTONOMOUS_MIN_ATTENTION_RATING=1400.0
```

### Linux Kernel Tuning

For high-throughput production:

```bash
# Increase max connections
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535

# Increase file descriptors
ulimit -n 65535
```

## Next Steps

- Configure AI clients (Cursor, ChatGPT, etc.) to use `http://<server>:8700`
- Test chat completions with various queries
- Monitor PostgreSQL performance with `pg_stat_statements`
- Scale: Add PostgreSQL read replicas for horizontal scaling
- Optimize: Add Redis cache layer for hot queries

## Architecture

```
Linux Client
    ↓
Orchestrator (FastAPI on 8700)
    ↓ (ctypes via LD_LIBRARY_PATH)
C++ Shared Libraries (.so)
    ↓ (libpq)
PostgreSQL on hart-server:5432 (or localhost)
    ↓ (B-tree + R-tree indexes)
4D Spatial Data + ELO-Rated Relations
```

**THE DATABASE IS THE MODEL**
- No llama.cpp
- No neural networks
- No forward passes
- Pure spatial queries (O(log N)) + relation traversal (O(K))
- Intelligence emerges from ELO-rated edges
