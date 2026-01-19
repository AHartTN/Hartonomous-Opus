#!/bin/bash

# full-build-test.sh
# Production-grade build and test script for Hartonomous-Opus
# This script executes a complete build, compile, test, and ingestion process

set -e  # Exit immediately on any command failure

# Function to log messages with timestamps
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Variables for paths and configuration
ORIGINAL_DIR=$(pwd)
WORKSPACE_DIR="/d/Repositories/Hartonomous-Opus"
LOG_DIR="$WORKSPACE_DIR/logs"

# Prerequisite checks
check_prerequisites() {
    log "Checking prerequisites..."

    # Special check for cmake to handle Windows installations
    if ! command -v cmake >/dev/null 2>&1; then
        log "CMake not found in PATH, checking common Windows locations..."
        local cmake_found=false
        cmake_paths=(
            "/c/Program Files/CMake/bin/cmake.exe"
            "/c/Program Files (x86)/CMake/bin/cmake.exe"
        )
        for path in "${cmake_paths[@]}"; do
            if [ -x "$path" ]; then
                cmake_dir=$(dirname "$path")
                export PATH="$cmake_dir:$PATH"
                log "Found CMake at $cmake_dir, added to PATH"
                cmake_found=true
                break
            fi
        done
        if [ "$cmake_found" = false ] && ! command -v cmake >/dev/null 2>&1; then
            log "ERROR: CMake not found. Please install CMake from https://cmake.org/download/ or via Visual Studio, and ensure it's in PATH or in one of the standard installation directories."
            exit 1
        fi
    fi

    # Check for required tools
    local required_tools=("ctest")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log "ERROR: Required tool '$tool' not found in PATH"
            exit 1
        fi
    done

    # Check if PostgreSQL is running (using pg_isready if available)
    if command -v pg_isready >/dev/null 2>&1; then
        if ! pg_isready -q >/dev/null 2>&1; then
            log "ERROR: PostgreSQL is not running or not accessible"
            exit 1
        fi
    else
        log "WARNING: pg_isready not found, cannot verify PostgreSQL status"
    fi

    # Check if workspace directory exists
    if [ ! -d "$WORKSPACE_DIR" ]; then
        log "ERROR: Workspace directory '$WORKSPACE_DIR' does not exist"
        exit 1
    fi

    log "Prerequisites check completed successfully"
}

# Cleanup function for signal handling
cleanup() {
    log "Received interrupt signal, cleaning up..."
    cd "$ORIGINAL_DIR" || true
    exit 1
}

# Set up signal traps for graceful cleanup
trap cleanup SIGINT SIGTERM

# Main execution function
main() {
    log "Starting full build and test process for Hartonomous-Opus"

    # Change to workspace directory
    cd "$WORKSPACE_DIR"

    # Source environment setup
    log "Sourcing environment setup"
    if [ -f "scripts/linux/env.sh" ]; then
        source scripts/linux/env.sh
    else
        log "WARNING: env.sh not found, continuing without it"
    fi

    # Perform prerequisite checks
    check_prerequisites

    # Clean and prepare build directory
    log "Cleaning previous build directory"
    rm -rf cpp/build
    mkdir -p cpp/build

    # CMake configure
    log "Running CMake configure"
    if ! cmake -B cpp/build -S cpp > "$LOG_DIR/01_build.txt" 2>&1; then
        log "ERROR: CMake configure failed"
        exit 1
    fi

    # CMake build
    log "Running CMake build"
    if ! cmake --build cpp/build --config Release > "$LOG_DIR/02_compile.txt" 2>&1; then
        log "ERROR: CMake build failed"
        exit 1
    fi

    # Run C++ tests
    log "Running C++ tests with ctest"
    cd cpp/build
    if ! ctest -V > "$LOG_DIR/03_cpp-test-log.txt" 2>&1; then
        log "ERROR: C++ tests failed"
        cd ../..
        exit 1
    fi
    cd ../..

    # Run clean script
    log "Running clean script"
    if ! ./scripts/linux/clean.sh > "$LOG_DIR/04_clean-script-log.txt" 2>&1; then
        log "ERROR: Clean script failed"
        exit 1
    fi

    # Run build script
    log "Running build script"
    if ! ./scripts/linux/build.sh > "$LOG_DIR/05_build-script-log.txt" 2>&1; then
        log "ERROR: Build script failed"
        exit 1
    fi

    # Run database setup script
    log "Running database setup script"
    if ! ./scripts/linux/setup-db.sh > "$LOG_DIR/06_setup-db-log.txt" 2>&1; then
        log "ERROR: Database setup failed"
        exit 1
    fi

    # Run test data ingestion script
    log "Running test data ingestion script"
    if ! ./scripts/linux/ingest-testdata.sh > "$LOG_DIR/07_ingest-testdata-log.txt" 2>&1; then
        log "ERROR: Test data ingestion failed"
        exit 1
    fi

    # Run full hypercube test script
    log "Running full hypercube test script"
    if ! ./scripts/linux/test-hypercube-full.sh > "$LOG_DIR/Test-Hypercube-Full-log.txt" 2>&1; then
        log "ERROR: Full hypercube test failed"
        exit 1
    fi

    # Success completion
    log "All build and test processes completed successfully"
    cd "$ORIGINAL_DIR"
    exit 0
}

# Execute main function
main "$@"