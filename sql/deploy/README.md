# Hypercube Database Deployment

This folder contains deployment scripts for the Hypercube database.

## Quick Deploy

From Windows (PowerShell):
```powershell
.\scripts\windows\deploy.ps1
```

From Linux/Mac:
```bash
psql -h $HC_DB_HOST -U $HC_DB_USER -d $HC_DB_NAME -f sql/deploy/full_schema.sql
```

## Files

- `full_schema.sql` - Complete schema in one file (no dependencies, can be run directly)
- `rebuild.ps1` - Script to regenerate full_schema.sql from source files

## C++ Extensions (Server-Side Only)

If you need the C++ extensions (hypercube, hypercube_ops, etc.), they must be
compiled and installed **on the PostgreSQL server**:

1. Copy the `cpp/` and `sql/extensions/` folders to the server
2. On the server:
   ```bash
   cd cpp && mkdir build && cd build
   cmake .. && make && sudo make install
   ```
3. Then in psql:
   ```sql
   CREATE EXTENSION hypercube;
   CREATE EXTENSION hypercube_ops;
   ```

The core functionality works without extensions - they only add performance
optimizations and some advanced features.
