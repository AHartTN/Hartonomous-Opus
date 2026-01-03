#!/bin/bash
# Hartonomous Hypercube - Build C++ Components (Linux)
# Usage: ./scripts/linux/build.sh [--clean] [--install]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

CLEAN=false
INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean) CLEAN=true; shift ;;
        --install) INSTALL=true; shift ;;
        *) shift ;;
    esac
done

if [ "$CLEAN" = true ]; then
    "$SCRIPT_DIR/clean.sh"
fi

echo "=== Building Hypercube C++ ==="
echo "Build type: $HC_BUILD_TYPE"
echo "Parallel jobs: $HC_PARALLEL_JOBS"

mkdir -p "$HC_BUILD_DIR"
cd "$HC_BUILD_DIR"

# Configure with CMake
echo -e "\nConfiguring..."
cmake .. -DCMAKE_BUILD_TYPE="$HC_BUILD_TYPE"

# Build
echo -e "\nBuilding..."
make -j"$HC_PARALLEL_JOBS"

echo -e "\n=== Build Complete ==="

# Show built artifacts
echo -e "\nExecutables:"
find . -maxdepth 2 -type f -executable -name "*_ingest" -o -name "seed_*" -o -name "test_*" -o -name "extract_*" 2>/dev/null | xargs -n1 basename 2>/dev/null || true

echo -e "\nExtensions:"
ls -1 *.so 2>/dev/null || echo "(none)"

# Install extensions if requested
if [ "$INSTALL" = true ]; then
    echo -e "\n=== Installing PostgreSQL Extensions ==="
    
    if ! command -v pg_config &>/dev/null; then
        echo "pg_config not found. Cannot install extensions."
        exit 1
    fi
    
    PG_LIB_DIR=$(pg_config --pkglibdir)
    PG_SHARE_DIR=$(pg_config --sharedir)
    PG_EXT_DIR="$PG_SHARE_DIR/extension"
    
    echo "Target lib dir: $PG_LIB_DIR"
    echo "Target ext dir: $PG_EXT_DIR"
    
    # Check write permissions
    if [ -w "$PG_LIB_DIR" ] && [ -w "$PG_EXT_DIR" ]; then
        # Copy shared libraries
        echo -e "\nCopying extension libraries..."
        [ -f "hypercube.so" ] && cp hypercube.so "$PG_LIB_DIR/" && echo "  hypercube.so"
        [ -f "semantic_ops.so" ] && cp semantic_ops.so "$PG_LIB_DIR/" && echo "  semantic_ops.so"
        [ -f "hypercube_ops.so" ] && cp hypercube_ops.so "$PG_LIB_DIR/" && echo "  hypercube_ops.so"
        [ -f "libhypercube_c.so" ] && cp libhypercube_c.so "$PG_LIB_DIR/" && echo "  libhypercube_c.so"
        
        # Copy SQL and control files
        echo -e "\nCopying extension metadata..."
        for ext in hypercube semantic_ops hypercube_ops; do
            [ -f "../sql/${ext}--1.0.sql" ] && cp "../sql/${ext}--1.0.sql" "$PG_EXT_DIR/"
            [ -f "../sql/${ext}.control" ] && cp "../sql/${ext}.control" "$PG_EXT_DIR/"
            [ -f "../${ext}.control" ] && cp "../${ext}.control" "$PG_EXT_DIR/"
        done
        echo "  SQL and control files installed"
        
        echo -e "\n=== Extensions Installed ==="
        echo "Run setup-db.sh to load into database"
    else
        echo "Cannot write to PostgreSQL directories"
        echo ""
        echo "Fix with sudo or adjust permissions:"
        echo "  sudo cp hypercube.so $PG_LIB_DIR/"
        echo "  sudo cp semantic_ops.so $PG_LIB_DIR/"
        echo ""
        echo "Or run: sudo make install"
        exit 1
    fi
fi

cd "$HC_PROJECT_ROOT"
echo -e "\nDone"
