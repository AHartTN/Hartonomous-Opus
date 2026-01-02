#!/bin/bash
# Build script for hypercube PostgreSQL extension

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CPP_DIR="$PROJECT_ROOT/cpp"
BUILD_DIR="$CPP_DIR/build"

echo "=== Building Hypercube Extension ==="
echo "Project root: $PROJECT_ROOT"
echo "C++ dir: $CPP_DIR"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(nproc)

# Run tests
echo "Running tests..."
ctest --output-on-failure

echo ""
echo "=== Build Complete ==="
echo "Extension built at: $BUILD_DIR/libhypercube.so"
echo ""
echo "To install:"
echo "  sudo make install"
echo ""
echo "Then in PostgreSQL:"
echo "  CREATE EXTENSION hypercube;"
