#!/bin/bash

# ============================================================================
# Hartonomous-Opus - Unified Cross-Platform Build Script
# ============================================================================
# Builds C++ components with CMake, automatically detecting platform
# and delegating to platform-specific logic where needed.
#
# Usage:
#   ./scripts/build.sh                    # Build + install extensions
#   ./scripts/build.sh --clean            # Clean rebuild
#   ./scripts/build.sh --no-install       # Build only, skip extension install
#   ./scripts/build.sh --extensions-only  # Build PostgreSQL extensions only
#   ./scripts/build.sh --debug            # Debug build
# ============================================================================

set -e

# Get script directory and load utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/shared/detect-platform.sh"
source "$SCRIPT_DIR/shared/logging.sh"
source "$SCRIPT_DIR/shared/requirements-check.sh"

# Configuration variables
PLATFORM=$(detect_os)
PLATFORM_DIR=$(get_platform_dir)
BUILD_TYPE="Release"
CLEAN=false
INSTALL=true
EXTENSIONS_ONLY=false
PARALLEL_JOBS=$(nproc 2>/dev/null || echo 4)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean|-c) CLEAN=true; shift ;;
        --no-install) INSTALL=false; shift ;;
        --extensions-only) EXTENSIONS_ONLY=true; shift ;;
        --debug|-d) BUILD_TYPE="Debug"; shift ;;
        --release|-r) BUILD_TYPE="Release"; shift ;;
        --jobs|-j) PARALLEL_JOBS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean, -c           Clean build directory before building"
            echo "  --no-install           Build only, skip PostgreSQL extension install"
            echo "  --extensions-only      Build PostgreSQL extensions only"
            echo "  --debug, -d            Debug build (default: Release)"
            echo "  --release, -r          Release build (default: Release)"
            echo "  --jobs, -j N           Number of parallel jobs (default: auto)"
            echo "  --help, -h             Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Check requirements
check_all_requirements || exit 1

# PostgreSQL extensions build function
build_postgresql_extensions() {
    log_subsection "Building PostgreSQL Extensions"

    # Find PostgreSQL development package
    PG_DEV_PACKAGE=""
    for candidate in \
        "$SCRIPT_DIR/../deployment/pg-dev-package" \
        "$HOME/pg-dev-package" \
        "$SCRIPT_DIR/../pg-dev-package"
    do
        if [ -f "$candidate/pg-config.env" ]; then
            PG_DEV_PACKAGE="$candidate"
            break
        fi
    done

    if [ -z "$PG_DEV_PACKAGE" ]; then
        log_error "PostgreSQL development package not found"
        log_info "Run package-pg-dev-from-server.sh on the server first"
        log_info "Or extract pg-dev-package.tar.gz to deployment/pg-dev-package/"
        return 1
    fi

    log_info "Found PostgreSQL dev package: $PG_DEV_PACKAGE"

    # Source PostgreSQL environment
    if ! source "$PG_DEV_PACKAGE/pg-config.env"; then
        log_error "Failed to source PostgreSQL environment"
        return 1
    fi

    # Verify pg_config works
    if ! command -v pg_config &> /dev/null; then
        log_error "pg_config not available after sourcing environment"
        return 1
    fi

    PG_VERSION=$(pg_config --version)
    log_info "Using PostgreSQL: $PG_VERSION"

    # Create extensions build directory
    EXT_BUILD_DIR="$PROJECT_ROOT/cpp/build-pg-extensions"
    log_info "Extensions build directory: $EXT_BUILD_DIR"

    if [ "$CLEAN" = true ]; then
        rm -rf "$EXT_BUILD_DIR"
    fi

    mkdir -p "$EXT_BUILD_DIR"
    cd "$EXT_BUILD_DIR"

    # Configure with CMake for extensions only
    log_info "Configuring extensions build with CMake..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DHYPERCUBE_ENABLE_PG_EXTENSIONS=ON \
          -DHYPERCUBE_ENABLE_TOOLS=OFF \
          -DHYPERCUBE_ENABLE_TESTS=OFF \
          .. || return 1

    # Build database wrapper library first
    log_info "Building database wrapper library..."
    cmake --build . --config Release --parallel "$PARALLEL_JOBS" --target db_wrapper_pg || return 1

    # Build extensions
    log_info "Building PostgreSQL extensions..."
    EXTENSIONS=(hypercube generative hypercube_ops embedding_ops semantic_ops)
    for ext in "${EXTENSIONS[@]}"; do
        log_info "Building extension: $ext"
        cmake --build . --config Release --parallel "$PARALLEL_JOBS" --target "$ext" || return 1
    done

    # Show built extensions
    echo
    echo "Built extensions in $EXT_BUILD_DIR/bin/:"
    ls -lh bin/*.so 2>/dev/null || ls -lh bin/*.dll 2>/dev/null || echo "  (none found)"

    log_success "PostgreSQL extensions built successfully"
}

# Load platform-specific build script if it exists
PLATFORM_BUILD_SCRIPT="$SCRIPT_DIR/platforms/$PLATFORM_DIR/build.sh"
if [ -f "$PLATFORM_BUILD_SCRIPT" ]; then
    log_info "Loading platform-specific build script for $PLATFORM"
    source "$PLATFORM_BUILD_SCRIPT"
fi

# Set build directories
BUILD_DIR="$PROJECT_ROOT/cpp/build"
BIN_DIR="$BUILD_DIR/bin/$BUILD_TYPE"

# Clean if requested
if [ "$CLEAN" = true ]; then
    log_info "Cleaning build directory and dependency cache..."
    rm -rf "$BUILD_DIR"
    rm -rf "$PROJECT_ROOT/external/_deps"
fi

# Handle extensions-only mode
if [ "$EXTENSIONS_ONLY" = true ]; then
    log_section "Hartonomous-Opus PostgreSQL Extensions Build ($PLATFORM)"

    echo "  Platform:       $PLATFORM"
    echo "  Parallel Jobs:  $PARALLEL_JOBS"
    echo "  Project Root:   $PROJECT_ROOT"
    echo

    build_postgresql_extensions
    log_success "PostgreSQL extensions build completed successfully!"
    exit 0
fi

log_section "Hartonomous-Opus Build ($PLATFORM - $BUILD_TYPE)"

echo "  Build Type:     $BUILD_TYPE"
echo "  Platform:       $PLATFORM"
echo "  Parallel Jobs:  $PARALLEL_JOBS"
echo "  Project Root:   $PROJECT_ROOT"
echo "  Build Dir:      $BUILD_DIR"
echo "  Output Dir:     $BIN_DIR"
echo

# Run platform-specific pre-build setup
if command -v pre_build_setup >/dev/null 2>&1; then
    log_info "Running pre-build setup..."
    pre_build_setup
fi

# ============================================================================
# CMAKE CONFIGURE & BUILD
# ============================================================================

cd "$PROJECT_ROOT/cpp"
mkdir -p build
cd build

log_info "Configuring with CMake..."

cmake_args=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$BIN_DIR"
    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$BIN_DIR"
)

# Use Ninja if available (cross-platform)
if command -v ninja &> /dev/null; then
    cmake_args+=("-G" "Ninja")
    log_info "Using Ninja generator"
fi

# Add platform-specific CMake arguments
if command -v get_cmake_args >/dev/null 2>&1; then
    platform_args=$(get_cmake_args)
    if [ -n "$platform_args" ]; then
        cmake_args+=($platform_args)
    fi
fi

log_info "CMake command: cmake ${cmake_args[*]} .."
cmake "${cmake_args[@]}" ..

log_info "Building with $PARALLEL_JOBS parallel jobs..."

START_TIME=$(date +%s)
cmake --build . --config "$BUILD_TYPE" --parallel "$PARALLEL_JOBS"
END_TIME=$(date +%s)

BUILD_TIME=$((END_TIME - START_TIME))
log_success "Build completed in ${BUILD_TIME}s"

# Show what was built
echo
echo "Executables in $BIN_DIR:"
ls -1 "$BIN_DIR"/*.exe 2>/dev/null || ls -1 "$BIN_DIR"/*[!.so,.dll] 2>/dev/null | head -10 || echo "  (none found)"

echo
echo "Libraries:"
ls -1 "$BIN_DIR"/*.so 2>/dev/null || ls -1 "$BIN_DIR"/*.dll 2>/dev/null || echo "  (none found)"

# ============================================================================
# INSTALL POSTGRESQL EXTENSIONS
# ============================================================================

if [ "$INSTALL" = true ]; then
    log_subsection "Installing PostgreSQL Extensions"

    # Run platform-specific extension installation
    if command -v install_extensions >/dev/null 2>&1; then
        install_extensions
    else
        log_warning "No platform-specific extension installer found. Skipping PostgreSQL extension installation."
        log_info "To install extensions manually, run: cmake --install . --config $BUILD_TYPE"
    fi
else
    log_info "Skipping PostgreSQL extension installation (--no-install specified)"
fi

log_success "Build process completed successfully!"
echo
echo "Next steps:"
log_info "Next steps paths verification:"
echo "  Run tests:    ./scripts/test.sh"
echo "  Setup DB:     ./scripts/deploy-database.sh"
echo "  Start orchestrator: ./scripts/deploy-orchestrator.sh"