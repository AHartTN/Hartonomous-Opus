#!/bin/bash
# Package PostgreSQL development files from hart-server
# Run this ON hart-server to create a tarball you can copy to your dev machine

set -e

echo "============================================"
echo "PostgreSQL Development Package Builder"
echo "============================================"
echo ""

# Detect PostgreSQL installation
if ! command -v pg_config &> /dev/null; then
    echo "ERROR: pg_config not found"
    echo "Install PostgreSQL development package:"
    echo "  Ubuntu/Debian: sudo apt-get install postgresql-server-dev-all"
    echo "  RHEL/CentOS: sudo yum install postgresql-devel"
    exit 1
fi

PG_VERSION=$(pg_config --version | awk '{print $2}')
PG_INCLUDEDIR=$(pg_config --includedir)
PG_INCLUDEDIR_SERVER=$(pg_config --includedir-server)
PG_LIBDIR=$(pg_config --libdir)
PG_PKGLIBDIR=$(pg_config --pkglibdir)
PG_BINDIR=$(pg_config --bindir)

echo "PostgreSQL Version: $PG_VERSION"
echo "Include Dir: $PG_INCLUDEDIR"
echo "Server Include Dir: $PG_INCLUDEDIR_SERVER"
echo "Library Dir: $PG_LIBDIR"
echo "Package Library Dir: $PG_PKGLIBDIR"
echo "Binary Dir: $PG_BINDIR"
echo ""

# Create temporary directory for packaging
PACKAGE_DIR="$HOME/pg-dev-package"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

echo "Creating package in: $PACKAGE_DIR"
echo ""

# Copy pg_config binary
echo "Copying pg_config..."
mkdir -p "$PACKAGE_DIR/bin"
cp "$PG_BINDIR/pg_config" "$PACKAGE_DIR/bin/"

# Copy client include headers
echo "Copying client headers..."
mkdir -p "$PACKAGE_DIR/include"
if [ -d "$PG_INCLUDEDIR" ]; then
    cp -r "$PG_INCLUDEDIR"/* "$PACKAGE_DIR/include/" 2>/dev/null || true
fi

# Copy server include headers (most important for extensions)
echo "Copying server headers..."
mkdir -p "$PACKAGE_DIR/include-server"
if [ -d "$PG_INCLUDEDIR_SERVER" ]; then
    cp -r "$PG_INCLUDEDIR_SERVER"/* "$PACKAGE_DIR/include-server/" 2>/dev/null || true
fi

# Copy libpq and related libraries
echo "Copying PostgreSQL libraries..."
mkdir -p "$PACKAGE_DIR/lib"

# Core libraries needed for building
for lib in libpq.so libpq.so.* libpgcommon.a libpgport.a libecpg.so libecpg.so.*; do
    find "$PG_LIBDIR" -name "$lib" -exec cp {} "$PACKAGE_DIR/lib/" \; 2>/dev/null || true
done

# Copy pkg-config files if they exist
if [ -d "$PG_LIBDIR/pkgconfig" ]; then
    mkdir -p "$PACKAGE_DIR/lib/pkgconfig"
    cp "$PG_LIBDIR/pkgconfig"/*.pc "$PACKAGE_DIR/lib/pkgconfig/" 2>/dev/null || true
fi

# Create a config file with the paths
echo "Creating configuration file..."
cat > "$PACKAGE_DIR/pg-config.env" << EOF
# PostgreSQL Development Environment
# Source this file before building: source pg-config.env

export PG_VERSION="$PG_VERSION"
export PG_DEV_ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

export PATH="\$PG_DEV_ROOT/bin:\$PATH"
export PG_INCLUDEDIR="\$PG_DEV_ROOT/include"
export PG_INCLUDEDIR_SERVER="\$PG_DEV_ROOT/include-server"
export PG_LIBDIR="\$PG_DEV_ROOT/lib"

# For CMake FindPostgreSQL
export PostgreSQL_ROOT="\$PG_DEV_ROOT"
export PostgreSQL_INCLUDE_DIR="\$PG_INCLUDEDIR"
export PostgreSQL_LIBRARY="\$PG_LIBDIR/libpq.so"

# For pkg-config
export PKG_CONFIG_PATH="\$PG_LIBDIR/pkgconfig:\$PKG_CONFIG_PATH"

# For runtime
export LD_LIBRARY_PATH="\$PG_LIBDIR:\$LD_LIBRARY_PATH"

echo "PostgreSQL development environment loaded"
echo "  Version: \$PG_VERSION"
echo "  Root: \$PG_DEV_ROOT"
echo ""
echo "pg_config available at: \$(which pg_config)"
EOF

# Create README
cat > "$PACKAGE_DIR/README.md" << 'EOF'
# PostgreSQL Development Package

This package contains PostgreSQL development headers and libraries needed to build PostgreSQL extensions.

## Contents

- `bin/pg_config` - PostgreSQL configuration utility
- `include/` - Client library headers (libpq)
- `include-server/` - Server development headers (for extensions)
- `lib/` - PostgreSQL libraries (libpq, etc.)
- `pg-config.env` - Environment setup script

## Usage on Development Machine

### 1. Extract Package

```bash
tar -xzf pg-dev-package.tar.gz -C /path/to/hartonomous
cd /path/to/hartonomous/pg-dev-package
```

### 2. Set Up Environment

```bash
# Source the environment file
source pg-config.env

# Verify pg_config works
pg_config --version
```

### 3. Build PostgreSQL Extensions

```bash
cd /path/to/Hartonomous-Opus/cpp
mkdir -p build && cd build

# CMake will find PostgreSQL via the environment variables
cmake -DCMAKE_BUILD_TYPE=Release \
      -DHYPERCUBE_ENABLE_PG_EXTENSIONS=ON \
      ..

# Build extensions
make -j$(nproc)

# Extensions will be in: build/bin/*.so
ls -lh bin/*.so
```

### 4. Package Extensions for hart-server

```bash
# Create deployment package
mkdir -p ~/pg-extensions-deploy
cp bin/hypercube.so ~/pg-extensions-deploy/
cp bin/generative.so ~/pg-extensions-deploy/
cp bin/hypercube_ops.so ~/pg-extensions-deploy/
cp bin/embedding_ops.so ~/pg-extensions-deploy/
cp bin/semantic_ops.so ~/pg-extensions-deploy/

# Copy control and SQL files
cp ../sql/*.control ~/pg-extensions-deploy/
cp ../sql/*--*.sql ~/pg-extensions-deploy/

# Create tarball
cd ~/pg-extensions-deploy
tar -czf ../hartonomous-pg-extensions.tar.gz .
```

### 5. Install on hart-server

Transfer `hartonomous-pg-extensions.tar.gz` to hart-server, then:

```bash
# On hart-server
PG_LIBDIR=$(pg_config --pkglibdir)
PG_SHAREDIR=$(pg_config --sharedir)

sudo tar -xzf hartonomous-pg-extensions.tar.gz -C /tmp/
sudo cp /tmp/*.so $PG_LIBDIR/
sudo cp /tmp/*.control $PG_SHAREDIR/extension/
sudo cp /tmp/*--*.sql $PG_SHAREDIR/extension/

# Enable extensions
sudo -u postgres psql hypercube << 'SQL'
CREATE EXTENSION IF NOT EXISTS hypercube;
CREATE EXTENSION IF NOT EXISTS hypercube_ops;
CREATE EXTENSION IF NOT EXISTS embedding_ops;
CREATE EXTENSION IF NOT EXISTS generative;
CREATE EXTENSION IF NOT EXISTS semantic_ops;

-- Test
SELECT hc_map_codepoint(65);
SQL
```

## Notes

- This package is specific to PostgreSQL version from hart-server
- Extensions built with this package will ONLY work on hart-server (or systems with matching PostgreSQL version)
- Runtime on hart-server doesn't need development headers, just the .so files

## Troubleshooting

**CMake can't find PostgreSQL:**
- Ensure you sourced `pg-config.env`
- Check `pg_config --version` works

**Wrong PostgreSQL version:**
- Extensions must be built against the exact PostgreSQL version running on hart-server
- Rebuild from hart-server's development package if versions mismatch

**Missing libraries at runtime:**
- Extensions link against libpq, but hart-server already has it
- No need to deploy libpq.so, only the extension .so files
EOF

# Create tarball
TARBALL="$HOME/pg-dev-package.tar.gz"
echo ""
echo "Creating tarball..."
cd "$HOME"
tar -czf "$TARBALL" pg-dev-package/

# Cleanup
rm -rf "$PACKAGE_DIR"

# Summary
TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
echo ""
echo "============================================"
echo "Package created successfully!"
echo "============================================"
echo ""
echo "Location: $TARBALL"
echo "Size: $TARBALL_SIZE"
echo ""
echo "Next steps:"
echo "  1. Copy to your dev machine:"
echo "     scp ahart@hart-server:~/pg-dev-package.tar.gz ."
echo ""
echo "  2. Extract and use:"
echo "     tar -xzf pg-dev-package.tar.gz"
echo "     cd pg-dev-package"
echo "     source pg-config.env"
echo "     cd /path/to/Hartonomous-Opus/cpp/build"
echo "     cmake -DCMAKE_BUILD_TYPE=Release .."
echo "     make -j\$(nproc)"
echo ""
echo "  3. See pg-dev-package/README.md for full instructions"
echo ""
