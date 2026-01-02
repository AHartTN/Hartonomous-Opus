#!/bin/bash
# Hartonomous Hypercube - Clean Build Artifacts (Linux)
# Usage: ./scripts/linux/clean.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=== Cleaning Build Artifacts ==="

# C++ build directory
if [ -d "$HC_BUILD_DIR" ]; then
    echo "Removing: $HC_BUILD_DIR"
    rm -rf "$HC_BUILD_DIR"
fi

# CMake cache files
find "$HC_PROJECT_ROOT/cpp" -name "CMakeCache.txt" -delete 2>/dev/null || true
find "$HC_PROJECT_ROOT/cpp" -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true

# Compiled objects
find "$HC_PROJECT_ROOT" -name "*.o" -delete 2>/dev/null || true
find "$HC_PROJECT_ROOT" -name "*.so" -delete 2>/dev/null || true
find "$HC_PROJECT_ROOT" -name "*.a" -delete 2>/dev/null || true

echo "Clean complete"
