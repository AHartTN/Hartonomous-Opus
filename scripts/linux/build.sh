#!/bin/bash
# Hartonomous Hypercube - Build C++ Components (Linux)
# Usage: ./scripts/linux/build.sh [--clean]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

if [ "$1" == "--clean" ]; then
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

cd "$HC_PROJECT_ROOT"

echo -e "\nBuild complete"
echo "Executables in: $HC_BUILD_DIR"
