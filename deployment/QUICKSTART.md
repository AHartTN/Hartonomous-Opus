# Hartonomous Orchestrator - Quick Start

Get Hartonomous running in 5 minutes.

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] Network access to hart-server (PostgreSQL at hart-server:5432)
- [ ] Windows: Visual C++ Redistributables installed
- [ ] Linux: gcc, libpq, openssl installed

## Windows Quick Start

```cmd
# 1. Add DLLs to PATH
set PATH=C:\path\to\deployment\windows\bin;%PATH%

# 2. Set up Python environment
cd windows\orchestrator
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure
copy ..\..\shared\config\.env.template .env
notepad .env
```

Edit `.env`:
```ini
POSTGRES_URL=postgresql://postgres:postgres@hart-server:5432/hypercube
USE_OPUS_DB=true
HARTONOMOUS_BIN_PATH=C:\path\to\deployment\windows\bin
```

```cmd
# 4. Start (or use start-orchestrator.cmd)
cd windows
start-orchestrator.cmd
```

## Linux Quick Start

```bash
# 1. Build or copy shared libraries
# If building from source:
cd ../../Hartonomous-Opus/cpp/build
make -j$(nproc)
cp bin/*.so ../../../deployment/linux/bin/

# 2. Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/deployment/linux/bin:$LD_LIBRARY_PATH

# 3. Set up Python environment
cd linux/orchestrator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Configure
cp ../../shared/config/.env.template .env
nano .env
```

Edit `.env`:
```ini
POSTGRES_URL=postgresql://postgres:postgres@hart-server:5432/hypercube
USE_OPUS_DB=true
HARTONOMOUS_BIN_PATH=/path/to/deployment/linux/bin
```

```bash
# 5. Start (or use start-orchestrator.sh)
cd linux
./start-orchestrator.sh
```

## Verify Installation

### Test 1: Database Connection

```bash
python -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:postgres@hart-server:5432/hypercube'); print('Connected!'); conn.close()"
```

✅ **Success**: "Connected!"
❌ **Failure**: Check network, PostgreSQL service, firewall

### Test 2: Library Loading

```bash
python -c "from openai_gateway.clients.hartonomous_client import get_hartonomous_client; get_hartonomous_client(); print('Libraries loaded!')"
```

✅ **Success**: "Libraries loaded!"
❌ **Failure**:
- Windows: Check PATH includes bin directory, install VC++ Redistributables
- Linux: Check LD_LIBRARY_PATH, install libpq/openssl

### Test 3: Cache Loading

```bash
python -c "from openai_gateway.clients.hartonomous_loader import initialize_hartonomous_caches, get_cache_stats; initialize_hartonomous_caches(); print(get_cache_stats())"
```

✅ **Success**: Shows vocab_size > 0, bigram_count > 0
⚠️ **Warning**: vocab_size = 0 means database needs ingestion

### Test 4: API Request

```bash
curl http://localhost:8700/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hartonomous","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
```

✅ **Success**: JSON response with generated text
❌ **Failure**: Check logs in orchestrator directory

## Common Issues

### "DLL not found" / "cannot open shared object"

**Windows**:
```cmd
# Verify PATH
echo %PATH%

# Add permanently via System Properties > Environment Variables
# Or temporarily:
set PATH=C:\path\to\deployment\windows\bin;%PATH%
```

**Linux**:
```bash
# Verify LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Add permanently to ~/.bashrc:
echo 'export LD_LIBRARY_PATH=/path/to/deployment/linux/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### "could not connect to server"

```bash
# Test network
ping hart-server

# Test PostgreSQL port
# Windows:
Test-NetConnection -ComputerName hart-server -Port 5432
# Linux:
nc -zv hart-server 5432

# SSH to hart-server and check
ssh ahart@hart-server
systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"
```

### "vocab_size: 0"

Database needs data. SSH to hart-server:

```bash
ssh ahart@hart-server
cd /path/to/Hartonomous-Opus
./scripts/linux/ingest-testdata.sh
```

Or check database:
```sql
sudo -u postgres psql hypercube
SELECT COUNT(*) FROM composition WHERE label IS NOT NULL;
```

### Port 8700 already in use

```bash
# Find process using port
# Windows:
netstat -ano | findstr :8700
# Linux:
lsof -i :8700

# Change port in .env:
ORCHESTRATOR_PORT=8701
```

## Next Steps

1. **Test with AI client**: Configure Cursor/ChatGPT to use `http://localhost:8700`
2. **Review logs**: Check `orchestrator/openai_gateway.log` for errors
3. **Tune performance**: Adjust cache settings in `.env`
4. **Production setup**: See INSTALL.md for systemd/Windows service configuration
5. **Scale**: Add PostgreSQL read replicas for higher throughput

## Architecture Reminder

```
Your AI Client
    ↓ HTTP
Orchestrator (Port 8700)
    ↓ ctypes
C++ DLLs/Libraries
    ↓ SQL
PostgreSQL (hart-server:5432)
```

**No llama.cpp. No neural networks. Database-native intelligence.**

## Getting Help

1. Read full installation guide: `windows/INSTALL.md` or `linux/INSTALL.md`
2. Check architecture docs: `README.md`
3. Review main repository documentation
4. Verify database schema and data on hart-server

The system is designed to be transparent and debuggable. Every query is SQL. Every operation is traceable. Check logs, check database, trace the flow.
