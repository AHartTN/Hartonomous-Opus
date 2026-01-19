# PostgreSQL Extensions Build & Deploy Guide

Complete guide for building and deploying Hartonomous PostgreSQL extensions from Windows to hart-server.

## The Problem

PostgreSQL extensions must be compiled against the **exact PostgreSQL version** running on the target server. You can't build on Windows and deploy to Linux, and you can't build against PostgreSQL 14 and deploy to PostgreSQL 16.

## The Solution

1. **Package PostgreSQL development files from hart-server** (headers, libraries, pg_config)
2. **Build extensions on your dev machine** using the packaged files
3. **Deploy compiled extensions back to hart-server**

This ensures version compatibility while allowing cross-compilation.

---

## Step 1: Package PostgreSQL Dev Files from hart-server

On **hart-server**, run:

```bash
ssh ahart@hart-server

# Download the packaging script (or create it manually)
cd ~
# ... create package-pg-dev-from-server.sh with the provided content ...

# Make executable
chmod +x package-pg-dev-from-server.sh

# Run it
./package-pg-dev-from-server.sh
```

**What this does:**
- Finds your PostgreSQL installation (`pg_config`)
- Copies headers from `/usr/include/postgresql/`
- Copies server headers from `/usr/include/postgresql/[version]/server/`
- Copies `libpq.so` and related libraries
- Packages everything into `~/pg-dev-package.tar.gz`

**Output:**
```
~/pg-dev-package.tar.gz (typically 5-20 MB)
```

---

## Step 2: Copy Package to Your Dev Machine

On your **Windows dev machine**:

```bash
# Copy from hart-server
scp ahart@hart-server:~/pg-dev-package.tar.gz .

# Extract
tar -xzf pg-dev-package.tar.gz
# Creates: pg-dev-package/ directory
```

**Package contents:**
```
pg-dev-package/
├── bin/
│   └── pg_config          # PostgreSQL configuration utility
├── include/               # Client headers (libpq)
├── include-server/        # Server headers (for extensions)
├── lib/                   # PostgreSQL libraries
│   ├── libpq.so
│   └── ...
├── pg-config.env          # Environment setup script
└── README.md              # Detailed instructions
```

---

## Step 3: Build Extensions

On your **Windows dev machine** (using WSL or Git Bash):

```bash
cd /d/Repositories/Hartonomous-Opus/deployment

# Run the build script
./build-extensions-with-pg-dev.sh
```

**What this does:**
1. Sources `pg-dev-package/pg-config.env` to set environment variables
2. Configures CMake with `-DHYPERCUBE_ENABLE_PG_EXTENSIONS=ON`
3. Builds only the PostgreSQL extension targets:
   - `hypercube.so`
   - `generative.so`
   - `hypercube_ops.so`
   - `embedding_ops.so`
   - `semantic_ops.so`
4. Packages extensions + SQL files into `~/hartonomous-pg-extensions.tar.gz`

**Output:**
```
~/hartonomous-pg-extensions.tar.gz
```

Contains:
- `*.so` files (compiled extensions)
- `*.control` files (extension metadata)
- `*--*.sql` files (SQL definitions)
- `install-on-hart-server.sh` (installation script)

---

## Step 4: Deploy to hart-server

Copy the extension package back to **hart-server**:

```bash
# From your dev machine
scp ~/hartonomous-pg-extensions.tar.gz ahart@hart-server:~/
```

On **hart-server**, install:

```bash
ssh ahart@hart-server

# Extract
cd ~
tar -xzf hartonomous-pg-extensions.tar.gz
cd hartonomous-pg-extensions

# Run installation script (uses sudo if needed)
./install-on-hart-server.sh
```

**What this does:**
- Copies `*.so` files to PostgreSQL library directory (usually `/usr/lib/postgresql/[version]/lib/`)
- Copies `*.control` and `*.sql` files to extension directory (usually `/usr/share/postgresql/[version]/extension/`)

---

## Step 5: Enable Extensions in Database

On **hart-server**, in PostgreSQL:

```bash
sudo -u postgres psql hypercube
```

```sql
-- Create extensions
CREATE EXTENSION IF NOT EXISTS hypercube;
CREATE EXTENSION IF NOT EXISTS hypercube_ops;
CREATE EXTENSION IF NOT EXISTS embedding_ops;
CREATE EXTENSION IF NOT EXISTS generative;
CREATE EXTENSION IF NOT EXISTS semantic_ops;

-- List loaded extensions
\dx

-- Test a function
SELECT hc_map_codepoint(65);  -- Should return 4D coordinates for 'A'
```

---

## What Gets Built

### PostgreSQL Extensions (`.so` files)

These are loaded by PostgreSQL and provide UDFs:

| Extension | Source | Purpose |
|-----------|--------|---------|
| `hypercube.so` | `src/pg/hypercube_pg.c` | Core 4D geometry functions |
| `generative.so` | `src/pg/generative_pg.c` | Text generation, token scoring |
| `hypercube_ops.so` | `src/pg/hypercube_ops_pg.c` | Operators for 4D types |
| `embedding_ops.so` | `src/pg/embedding_ops_pg.c` | Vector similarity operations |
| `semantic_ops.so` | `src/pg/semantic_ops_pg.c` | Semantic search operations |

### C Bridge Libraries

These are linked INTO the extensions (not separate):

- `libhypercube_c.so` - Core 4D coordinate system
- `libgenerative_c.so` - Generation engine
- `libembedding_c.so` - SIMD vector operations

The extensions link these statically or dynamically.

---

## Architecture

```
┌─────────────────────────────────────────┐
│ PostgreSQL (hart-server:5432)           │
├─────────────────────────────────────────┤
│ SQL Query:                              │
│   SELECT hc_map_codepoint(65);          │
│          ↓                               │
│ Extension: hypercube.so                 │
│          ↓                               │
│ C Bridge: libhypercube_c.so             │
│          ↓                               │
│ Core C++: Laplacian projection, Hilbert│
│          ↓                               │
│ Returns: 4D coordinates (x,y,z,w)       │
└─────────────────────────────────────────┘
```

**Key Points:**
- PostgreSQL calls UDFs defined in `.so` extensions
- Extensions use C bridge libraries for heavy computation
- ALL heavy lifting (SIMD, math, walks) happens in C++
- Database stores results (compositions, relations)

---

## Troubleshooting

### "ERROR: pg_config not found"

On **hart-server**, install PostgreSQL development package:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql-server-dev-all

# RHEL/CentOS/Fedora
sudo yum install postgresql-devel
```

### "ERROR: extension not found after install"

Check installation paths:

```bash
# On hart-server
pg_config --pkglibdir    # Should contain *.so files
pg_config --sharedir     # Should contain extension/*.control files

ls -lh $(pg_config --pkglibdir)/hypercube.so
ls -lh $(pg_config --sharedir)/extension/hypercube.control
```

### "ERROR: incompatible version"

Extensions must match PostgreSQL version exactly. Rebuild against current hart-server PostgreSQL version:

```bash
# On hart-server
pg_config --version

# Compare to version in pg-dev-package
# If different, re-run package-pg-dev-from-server.sh
```

### "ERROR: undefined symbol: ..."

Extension depends on a C++ library that wasn't linked properly. Check `cpp/cmake/TargetsPgExtensions.cmake`:

```cmake
# Each extension must link its dependencies
hc_add_pg_extension(hypercube src/pg/hypercube_pg.c
    C_BRIDGE hypercube_c  # Links libhypercube_c
    OUTPUT_NAME "hypercube")
```

### Build fails with "PostgreSQL not found"

Ensure you sourced the environment:

```bash
source pg-dev-package/pg-config.env
pg_config --version  # Should work
```

---

## Directory Structure

```
Hartonomous-Opus/
├── cpp/
│   ├── CMakeLists.txt
│   ├── cmake/
│   │   └── TargetsPgExtensions.cmake  # Extension build definitions
│   ├── src/
│   │   ├── pg/                        # PostgreSQL extension C code
│   │   │   ├── hypercube_pg.c
│   │   │   ├── generative_pg.c
│   │   │   └── ...
│   │   └── bridge/                    # C bridge implementations
│   │       ├── hypercube_c.cpp
│   │       ├── generative_c.cpp
│   │       └── embedding_c.cpp
│   ├── sql/                           # SQL definitions
│   │   ├── hypercube.control
│   │   ├── hypercube--1.0.sql
│   │   └── ...
│   └── build-pg-extensions/           # Build output
│       └── bin/
│           ├── hypercube.so
│           ├── generative.so
│           └── ...
│
└── deployment/
    ├── package-pg-dev-from-server.sh      # Run on hart-server
    ├── build-extensions-with-pg-dev.sh    # Run on dev machine
    └── pg-dev-package/                    # Extracted package
        ├── bin/pg_config
        ├── include/
        ├── include-server/
        └── lib/
```

---

## Next Steps After Installation

1. **Test UDFs:**
   ```sql
   -- 4D coordinate mapping
   SELECT hc_map_codepoint(65);
   SELECT hc_map_codepoint(32768);

   -- Text generation (if vocab loaded)
   SELECT gen_generate('The', 10);

   -- Spatial queries
   SELECT * FROM composition
   ORDER BY ST_Distance(centroid,
                        ST_MakePoint(0.5, 0.5, 0.5, 0.5))
   LIMIT 10;
   ```

2. **Load Caches:**
   - Vocabulary from compositions
   - Bigrams from high-ELO relations
   - Attention from very high-ELO relations

3. **Build Application Layer:**
   - C# API gateway
   - Query orchestration
   - Response streaming

4. **Optimize:**
   - Add indexes if missing
   - Tune PostgreSQL configuration
   - Profile UDF performance

---

## Summary

**To rebuild and redeploy:**

```bash
# 1. On hart-server: Package dev files
./package-pg-dev-from-server.sh

# 2. On dev machine: Get package
scp ahart@hart-server:~/pg-dev-package.tar.gz .
tar -xzf pg-dev-package.tar.gz

# 3. On dev machine: Build extensions
cd /d/Repositories/Hartonomous-Opus/deployment
./build-extensions-with-pg-dev.sh

# 4. On hart-server: Deploy
scp ~/hartonomous-pg-extensions.tar.gz ahart@hart-server:~/
ssh ahart@hart-server
tar -xzf ~/hartonomous-pg-extensions.tar.gz
cd hartonomous-pg-extensions
./install-on-hart-server.sh

# 5. On hart-server: Enable
sudo -u postgres psql hypercube -c "CREATE EXTENSION IF NOT EXISTS hypercube CASCADE;"
```

**The core is now:**
- PostgreSQL database (data layer)
- C++ UDFs (computation layer)
- SQL queries (orchestration)

Everything else (Python, C#) is just presentation/API layers on top.
