#!/bin/bash
# Build PostgreSQL extensions using the packaged PostgreSQL development files
# Run this on your development machine after extracting pg-dev-package.tar.gz

set -e

echo "============================================"
echo "Build PostgreSQL Extensions"
echo "============================================"
echo ""

# Find pg-dev-package
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PG_DEV_PACKAGE=""

# Check common locations
for candidate in \
    "$SCRIPT_DIR/pg-dev-package" \
    "$HOME/pg-dev-package" \
    "/d/Repositories/Hartonomous-Opus/pg-dev-package" \
    "/d/Repositories/Hartonomous-Opus/deployment/pg-dev-package"
do
    if [ -f "$candidate/pg-config.env" ]; then
        PG_DEV_PACKAGE="$candidate"
        break
    fi
done

if [ -z "$PG_DEV_PACKAGE" ]; then
    echo "ERROR: PostgreSQL development package not found"
    echo ""
    echo "Please extract pg-dev-package.tar.gz first:"
    echo "  scp ahart@hart-server:~/pg-dev-package.tar.gz ."
    echo "  tar -xzf pg-dev-package.tar.gz"
    echo ""
    exit 1
fi

echo "Found PostgreSQL dev package: $PG_DEV_PACKAGE"
echo ""

# Source the environment
echo "Loading PostgreSQL environment..."
source "$PG_DEV_PACKAGE/pg-config.env"

# Verify pg_config works
if ! command -v pg_config &> /dev/null; then
    echo "ERROR: pg_config not in PATH after sourcing environment"
    exit 1
fi

PG_VERSION=$(pg_config --version)
echo "Using PostgreSQL: $PG_VERSION"
echo ""

# Find Hartonomous repository root
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_DIR="$REPO_ROOT/cpp"

if [ ! -f "$CPP_DIR/CMakeLists.txt" ]; then
    echo "ERROR: Could not find cpp/CMakeLists.txt"
    echo "Please run this script from the deployment directory"
    exit 1
fi

echo "Repository root: $REPO_ROOT"
echo "C++ source: $CPP_DIR"
echo ""

# Build directory
BUILD_DIR="$CPP_DIR/build-pg-extensions"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Building in: $BUILD_DIR"
echo ""

# Configure with CMake
echo "============================================"
echo "Configuring with CMake..."
echo "============================================"
echo ""

cmake -DCMAKE_BUILD_TYPE=Release \
      -DHYPERCUBE_ENABLE_PG_EXTENSIONS=ON \
      -DHYPERCUBE_ENABLE_TOOLS=OFF \
      -DHYPERCUBE_ENABLE_TESTS=OFF \
      ..

echo ""
echo "============================================"
echo "Building PostgreSQL extensions..."
echo "============================================"
echo ""

# Build only the extension targets
make -j$(nproc) hypercube generative hypercube_ops embedding_ops semantic_ops

echo ""
echo "============================================"
echo "Build complete!"
echo "============================================"
echo ""

# List built extensions
echo "Built extensions:"
ls -lh bin/*.so 2>/dev/null || ls -lh bin/*.dll 2>/dev/null || echo "No extensions found in bin/"
echo ""

# Create deployment package
DEPLOY_DIR="$HOME/hartonomous-pg-extensions"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

echo "Creating deployment package in: $DEPLOY_DIR"
echo ""

# Copy extensions
cp bin/*.so "$DEPLOY_DIR/" 2>/dev/null || cp bin/*.dll "$DEPLOY_DIR/" 2>/dev/null || true

# Copy control files
cp ../sql/*.control "$DEPLOY_DIR/" 2>/dev/null || echo "Warning: No .control files found"

# Copy SQL files
cp ../sql/*--*.sql "$DEPLOY_DIR/" 2>/dev/null || echo "Warning: No SQL files found"

# Create installation script
cat > "$DEPLOY_DIR/install-on-hart-server.sh" << 'INSTALL_SCRIPT'
#!/bin/bash
# Install Hartonomous PostgreSQL extensions on hart-server
# Run this ON hart-server after copying the extensions

set -e

echo "============================================"
echo "Installing Hartonomous PostgreSQL Extensions"
echo "============================================"
echo ""

if ! command -v pg_config &> /dev/null; then
    echo "ERROR: pg_config not found"
    exit 1
fi

PG_VERSION=$(pg_config --version)
PG_LIBDIR=$(pg_config --pkglibdir)
PG_SHAREDIR=$(pg_config --sharedir)
PG_EXTDIR="$PG_SHAREDIR/extension"

echo "PostgreSQL: $PG_VERSION"
echo "Library directory: $PG_LIBDIR"
echo "Extension directory: $PG_EXTDIR"
echo ""

# Check if we need sudo
SUDO=""
if [ ! -w "$PG_LIBDIR" ]; then
    SUDO="sudo"
    echo "Using sudo for installation"
fi

# Install shared libraries
echo "Installing shared libraries..."
$SUDO cp *.so "$PG_LIBDIR/" 2>/dev/null || true
ls -lh "$PG_LIBDIR"/*.so | grep -E "(hypercube|generative|embedding|semantic)" || echo "Warning: No .so files found"
echo ""

# Install control files
echo "Installing control files..."
$SUDO cp *.control "$PG_EXTDIR/" 2>/dev/null || true
ls -lh "$PG_EXTDIR"/*.control | grep -E "(hypercube|generative|embedding|semantic)" || echo "Warning: No .control files found"
echo ""

# Install SQL files
echo "Installing SQL files..."
$SUDO cp *--*.sql "$PG_EXTDIR/" 2>/dev/null || true
ls -lh "$PG_EXTDIR"/*--*.sql | grep -E "(hypercube|generative|embedding|semantic)" || echo "Warning: No SQL files found"
echo ""

echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "To enable extensions in your database:"
echo ""
echo "  sudo -u postgres psql hypercube << SQL"
echo "  CREATE EXTENSION IF NOT EXISTS hypercube;"
echo "  CREATE EXTENSION IF NOT EXISTS hypercube_ops;"
echo "  CREATE EXTENSION IF NOT EXISTS embedding_ops;"
echo "  CREATE EXTENSION IF NOT EXISTS generative;"
echo "  CREATE EXTENSION IF NOT EXISTS semantic_ops;"
echo ""
echo "  -- Test"
echo "  SELECT hc_map_codepoint(65);"
echo "  SQL"
echo ""
INSTALL_SCRIPT

chmod +x "$DEPLOY_DIR/install-on-hart-server.sh"

# Create tarball
cd "$HOME"
TARBALL="hartonomous-pg-extensions.tar.gz"
tar -czf "$TARBALL" hartonomous-pg-extensions/

TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)

echo ""
echo "============================================"
echo "Deployment package created!"
echo "============================================"
echo ""
echo "Location: $HOME/$TARBALL"
echo "Size: $TARBALL_SIZE"
echo ""
echo "Contents:"
ls -lh "$DEPLOY_DIR"
echo ""
echo "Next steps:"
echo ""
echo "  1. Copy to hart-server:"
echo "     scp ~/$TARBALL ahart@hart-server:~/"
echo ""
echo "  2. On hart-server, extract and install:"
echo "     tar -xzf ~/$TARBALL"
echo "     cd ~/hartonomous-pg-extensions"
echo "     ./install-on-hart-server.sh"
echo ""
echo "  3. Enable extensions in PostgreSQL:"
echo "     sudo -u postgres psql hypercube"
echo "     CREATE EXTENSION hypercube;"
echo "     CREATE EXTENSION generative;"
echo "     SELECT hc_map_codepoint(65);"
echo ""
