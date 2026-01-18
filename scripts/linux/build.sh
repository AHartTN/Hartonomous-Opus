#!/bin/bash
# ============================================================================
# Hartonomous Hypercube - Build C++ Components (Linux)
# ============================================================================
# Builds and installs PostgreSQL extensions.
#
# Usage:
#   ./scripts/linux/build.sh              # Build + install extensions
#   ./scripts/linux/build.sh --clean      # Clean rebuild
#   ./scripts/linux/build.sh --no-install # Build only, skip extension install
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

CLEAN=false
INSTALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean|-c) CLEAN=true; shift ;;
        --no-install) INSTALL=false; shift ;;
        *) shift ;;
    esac
done

if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$HC_BUILD_DIR"
fi

echo ""
echo "============================================================"
echo " Hartonomous Hypercube - Build (Linux)"
echo "============================================================"
echo ""
echo "  Build Type:     $HC_BUILD_TYPE"
echo "  Parallel Jobs:  $HC_PARALLEL_JOBS"
echo "  Project Root:   $HC_PROJECT_ROOT"
echo "  Output Dir:     $HC_BIN_DIR"
echo ""

# ============================================================================
# CMAKE CONFIGURE & BUILD
# ============================================================================

cd "$HC_PROJECT_ROOT/cpp"
mkdir -p build
cd build

echo "Configuring with CMake..."

cmake_args=(
    "-DCMAKE_BUILD_TYPE=$HC_BUILD_TYPE"
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$HC_BIN_DIR"
    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$HC_BIN_DIR"
)

# Use Ninja if available
if command -v ninja &> /dev/null; then
    cmake_args+=("-G" "Ninja")
fi

cmake "${cmake_args[@]}" ..

echo ""
echo "Building with $HC_PARALLEL_JOBS parallel jobs..."

START_TIME=$(date +%s)
cmake --build . --config "$HC_BUILD_TYPE" --parallel "$HC_PARALLEL_JOBS"
END_TIME=$(date +%s)

echo ""
echo "============================================================"
echo " BUILD COMPLETE ($(($END_TIME - $START_TIME))s)"
echo "============================================================"

# Show what was built
echo ""
echo "Executables in $HC_BIN_DIR:"
ls -1 "$HC_BIN_DIR"/*.exe 2>/dev/null || ls -1 "$HC_BIN_DIR"/*[!.so] 2>/dev/null | head -10 || echo "  (none found)"

echo ""
echo "Shared libraries:"
ls -1 "$HC_BIN_DIR"/*.so 2>/dev/null || echo "  (none found)"

# ============================================================================
# INSTALL POSTGRESQL EXTENSIONS
# ============================================================================

if [ "$INSTALL" = true ]; then
    echo ""
    echo "============================================================"
    echo " Installing PostgreSQL Extensions"
    echo "============================================================"

    if ! command -v pg_config &> /dev/null; then
        echo "ERROR: pg_config not found"
        echo "  Install postgresql-server-dev-all or set PATH"
        exit 1
    fi

    PG_LIB_DIR=$(pg_config --pkglibdir)
    PG_SHARE_DIR=$(pg_config --sharedir)
    PG_EXT_DIR="$PG_SHARE_DIR/extension"

    echo "  pkglibdir:  $PG_LIB_DIR"
    echo "  extension:  $PG_EXT_DIR"
    echo ""

    # Use sudo if we don't have write access
    SUDO=""
    if [ ! -w "$PG_LIB_DIR" ]; then
        SUDO="sudo"
        echo "  (Using sudo for installation)"
    fi

    # Extension shared libraries
    EXTENSIONS=(hypercube hypercube_c hypercube_ops embedding_ops embedding_c semantic_ops generative generative_c)

    for ext in "${EXTENSIONS[@]}"; do
        so_file="$HC_BIN_DIR/$ext.so"
        if [ -f "$so_file" ]; then
            $SUDO cp "$so_file" "$PG_LIB_DIR/"
            echo "  [lib] $ext.so"
        fi
    done

    # Control files
    CTRL_FILES=(
        "$HC_PROJECT_ROOT/cpp/hypercube.control"
        "$HC_PROJECT_ROOT/cpp/sql/hypercube_ops.control"
        "$HC_PROJECT_ROOT/cpp/sql/embedding_ops.control"
        "$HC_PROJECT_ROOT/cpp/sql/semantic_ops.control"
        "$HC_PROJECT_ROOT/cpp/sql/generative.control"
    )

    for ctrl in "${CTRL_FILES[@]}"; do
        if [ -f "$ctrl" ]; then
            $SUDO cp "$ctrl" "$PG_EXT_DIR/"
            echo "  [ctrl] $(basename $ctrl)"
        fi
    done

    # SQL files
    SQL_FILES=(
        "$HC_PROJECT_ROOT/cpp/sql/hypercube--1.0.sql"
        "$HC_PROJECT_ROOT/cpp/sql/hypercube_ops--1.0.sql"
        "$HC_PROJECT_ROOT/cpp/sql/embedding_ops--1.0.sql"
        "$HC_PROJECT_ROOT/cpp/sql/semantic_ops--1.0.sql"
        "$HC_PROJECT_ROOT/cpp/sql/generative--1.0.sql"
    )

    for sql in "${SQL_FILES[@]}"; do
        if [ -f "$sql" ]; then
            $SUDO cp "$sql" "$PG_EXT_DIR/"
            echo "  [sql] $(basename $sql)"
        fi
    done

    echo ""
    echo "Extensions installed. Enable with:"
    echo "  CREATE EXTENSION hypercube;"
    echo "  CREATE EXTENSION hypercube_ops;"
fi

echo ""
echo "Next: ./scripts/linux/setup-db.sh"
