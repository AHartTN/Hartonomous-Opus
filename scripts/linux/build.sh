#!/bin/bash
# ============================================================================
# Hartonomous Hypercube - Build C++ Components (Linux)
# ============================================================================
# Optimized build with Intel oneAPI (MKL + OpenMP) if available.
#
# Usage:
#   ./scripts/linux/build.sh                  # Standard build
#   ./scripts/linux/build.sh --clean          # Clean rebuild
#   ./scripts/linux/build.sh --install        # Install PostgreSQL extensions
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

CLEAN=false
INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean|-c) CLEAN=true; shift ;;
        --install|-i) INSTALL=true; shift ;;
        *) shift ;;
    esac
done

if [ "$CLEAN" = true ]; then
    "$SCRIPT_DIR/clean.sh"
fi

echo ""
echo "============================================================"
echo " Hartonomous Hypercube - Build"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Build Type:     $HC_BUILD_TYPE"
echo "  Parallel Jobs:  $HC_PARALLEL_JOBS"
echo "  Project Root:   $HC_PROJECT_ROOT"
echo ""

# ============================================================================
# CHECK INTEL MKL
# ============================================================================

if [ -n "$MKLROOT" ]; then
    echo "Intel MKL:        $MKLROOT"
else
    echo "Intel MKL:        Not found (using system BLAS)"
fi

# ============================================================================
# CMAKE CONFIGURE
# ============================================================================

mkdir -p "$HC_BUILD_DIR"
cd "$HC_BUILD_DIR"

echo ""
echo "Configuring with CMake..."

cmake_args=(
    "-DCMAKE_BUILD_TYPE=$HC_BUILD_TYPE"
)

# Use Ninja if available
if command -v ninja &> /dev/null; then
    cmake_args+=("-G" "Ninja")
fi

cmake "${cmake_args[@]}" ..

# ============================================================================
# BUILD
# ============================================================================

echo ""
echo "Building with $HC_PARALLEL_JOBS parallel jobs..."

START_TIME=$(date +%s)

if [ -f "build.ninja" ]; then
    ninja -j "$HC_PARALLEL_JOBS"
else
    make -j "$HC_PARALLEL_JOBS"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo " BUILD COMPLETE"
echo " Time: ${ELAPSED}s"
echo "============================================================"

echo ""
echo "Executables:"
find . -maxdepth 2 -type f -executable \( -name "ingest*" -o -name "seed_*" -o -name "test_*" -o -name "extract_*" -o -name "hc" \) 2>/dev/null | while read f; do
    echo "  $(basename "$f")"
done

echo ""
echo "Extensions:"
ls -1 *.so 2>/dev/null || echo "  (none)"

# ============================================================================
# INSTALL EXTENSIONS
# ============================================================================

if [ "$INSTALL" = true ]; then
    echo ""
    echo "Installing PostgreSQL Extensions..."
    
    if ! command -v pg_config &> /dev/null; then
        echo "ERROR: pg_config not found. Cannot install extensions."
        exit 1
    fi
    
    PG_LIB_DIR=$(pg_config --pkglibdir)
    PG_SHARE_DIR=$(pg_config --sharedir)
    PG_EXT_DIR="$PG_SHARE_DIR/extension"
    
    echo "  Target: $PG_LIB_DIR"
    
    for ext in hypercube semantic_ops hypercube_ops embedding_ops generative; do
        if [ -f "$ext.so" ]; then
            sudo cp "$ext.so" "$PG_LIB_DIR/"
            echo "  Installed: $ext.so"
        fi
        if [ -f "../sql/$ext--1.0.sql" ]; then
            sudo cp "../sql/$ext--1.0.sql" "$PG_EXT_DIR/"
        fi
        if [ -f "../sql/$ext.control" ]; then
            sudo cp "../sql/$ext.control" "$PG_EXT_DIR/"
        fi
    done
    
    # Core library
    if [ -f "libhypercube_c.so" ]; then
        sudo cp "libhypercube_c.so" "$PG_LIB_DIR/"
        echo "  Installed: libhypercube_c.so"
    fi
fi

echo ""
echo "Next steps:"
echo "  ./scripts/linux/setup-db.sh       # Setup database schema"
echo "  ./scripts/linux/ingest.sh <path>  # Ingest a model"
