#!/bin/bash

# ============================================================================
# Hartonomous-Opus - Unified Test Runner
# ============================================================================
# Runs all test suites across the project with platform detection
# and unified reporting.
#
# Usage:
#   ./scripts/test.sh                    # Run all tests
#   ./scripts/test.sh --unit             # Unit tests only
#   ./scripts/test.sh --integration      # Integration tests only
#   ./scripts/test.sh --system           # System validation only
#   ./scripts/test.sh --cpp              # C++ tests only
#   ./scripts/test.sh --python           # Python tests only
# ============================================================================

set -e

# Get script directory and load utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/shared/detect-platform.sh"
source "$SCRIPT_DIR/shared/logging.sh"
source "$SCRIPT_DIR/shared/requirements-check.sh"

# Test configuration
PLATFORM=$(detect_os)
PLATFORM_DIR=$(get_platform_dir)
RUN_UNIT=true
RUN_INTEGRATION=true
RUN_SYSTEM=true
RUN_CPP=true
RUN_PYTHON=true
RUN_BENCHMARK=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit) RUN_INTEGRATION=false; RUN_SYSTEM=false; RUN_CPP=false; RUN_PYTHON=false; shift ;;
        --integration) RUN_UNIT=false; RUN_SYSTEM=false; RUN_CPP=false; RUN_PYTHON=false; shift ;;
        --system) RUN_UNIT=false; RUN_INTEGRATION=false; RUN_CPP=false; RUN_PYTHON=false; shift ;;
        --cpp) RUN_UNIT=false; RUN_INTEGRATION=false; RUN_SYSTEM=false; RUN_PYTHON=false; shift ;;
        --python) RUN_UNIT=false; RUN_INTEGRATION=false; RUN_SYSTEM=false; RUN_CPP=false; shift ;;
        --benchmark) RUN_BENCHMARK=true; shift ;;
        --verbose|-v) VERBOSE=true; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --unit         Run unit tests only"
            echo "  --integration  Run integration tests only"
            echo "  --system       Run system validation only"
            echo "  --cpp          Run C++ tests only"
            echo "  --python       Run Python tests only"
            echo "  --benchmark    Include performance benchmarks"
            echo "  --verbose, -v  Verbose output"
            echo "  --help, -h     Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Load platform-specific test script if it exists
PLATFORM_TEST_SCRIPT="$SCRIPT_DIR/test/platforms/$PLATFORM_DIR/test.sh"
if [ -f "$PLATFORM_TEST_SCRIPT" ]; then
    log_info "Loading platform-specific test script for $PLATFORM"
    source "$PLATFORM_TEST_SCRIPT"
fi

log_section "Hartonomous-Opus Test Suite ($PLATFORM)"

echo "  Test Types: $([ "$RUN_UNIT" = true ] && echo "unit ")$([ "$RUN_INTEGRATION" = true ] && echo "integration ")$([ "$RUN_SYSTEM" = true ] && echo "system ")$([ "$RUN_CPP" = true ] && echo "cpp ")$([ "$RUN_PYTHON" = true ] && echo "python ")$([ "$RUN_BENCHMARK" = true ] && echo "benchmark ")"
echo "  Platform:   $PLATFORM"
echo "  Verbose:    $VERBOSE"
echo

# ============================================================================
# UNIT TESTS
# ============================================================================

if [ "$RUN_UNIT" = true ]; then
    log_subsection "Running Unit Tests"

    # C++ Unit Tests
    if [ "$RUN_CPP" = true ] && [ -d "$PROJECT_ROOT/cpp/build" ]; then
        log_info "Running C++ unit tests..."

        cd "$PROJECT_ROOT/cpp/build"

        if [ "$VERBOSE" = true ]; then
            ctest --output-on-failure -V
        else
            ctest --output-on-failure
        fi

        check_result "C++ unit tests"
    fi

    # Python Unit Tests
    if [ "$RUN_PYTHON" = true ]; then
        log_info "Running Python unit tests..."

        # Test gateway components
        if [ -d "$PROJECT_ROOT/Hartonomous-Orchestrator" ]; then
            cd "$PROJECT_ROOT/Hartonomous-Orchestrator"

            if [ -f "test_gateway.py" ]; then
                python3 test_gateway.py
                check_result "Python gateway tests"
            fi

            if [ -f "test_refactor.py" ]; then
                python3 test_refactor.py
                check_result "Python refactor tests"
            fi
        fi
    fi
fi

# ============================================================================
# SYSTEM VALIDATION
# ============================================================================

if [ "$RUN_SYSTEM" = true ]; then
    log_subsection "System Validation"

    # Set up database environment
    export HC_DB_HOST="${HC_DB_HOST:-hart-server}"
    export HC_DB_PORT="${HC_DB_PORT:-5432}"
    export HC_DB_USER="${HC_DB_USER:-postgres}"
    export HC_DB_PASS="${HC_DB_PASS:-postgres}"
    export HC_DB_NAME="${HC_DB_NAME:-hypercube}"
    export PGPASSWORD="$HC_DB_PASS"

    # Test database connectivity
    if PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -c "SELECT 1;" >/dev/null 2>&1; then
        log_info "Database connection: ✓"

        # Test PostGIS
        if PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -c "SELECT PostGIS_Version();" >/dev/null 2>&1; then
            log_info "PostGIS extension: ✓"
        else
            log_warning "PostGIS extension: ✗"
        fi

        # Test database exists and has atoms
        if PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -c "SELECT COUNT(*) FROM atom" >/dev/null 2>&1; then
            ATOM_COUNT=$(PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null)
            if [ "$ATOM_COUNT" -ge 1100000 ]; then
                log_info "Atom seeding: ✓ ($ATOM_COUNT atoms)"
            else
                log_warning "Atom seeding: ⚠ ($ATOM_COUNT atoms, expected ~1.1M)"
            fi
        else
            log_warning "Database setup: ✗ (no atom table found)"
        fi

        # Test extensions
        EXTENSIONS=("hypercube" "hypercube_ops" "embedding_ops" "generative" "semantic_ops")
        for ext in "${EXTENSIONS[@]}"; do
            if PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -c "SELECT 1 FROM pg_extension WHERE extname='$ext'" >/dev/null 2>&1; then
                log_info "Extension $ext: ✓"
            else
                log_warning "Extension $ext: ✗"
            fi
        done
    else
        log_warning "Database connection: ✗"
        log_info "Skipping database tests - ensure PostgreSQL is running and accessible"
    fi

    unset PGPASSWORD

    # Test orchestrator
    if command -v curl &> /dev/null; then
        if curl -s "http://localhost:8700/health" >/dev/null 2>&1; then
            log_info "Orchestrator health: ✓"
        else
            log_warning "Orchestrator health: ✗ (not running or unreachable)"
        fi
    else
        log_info "curl not available - skipping orchestrator health check"
    fi

    # Test build artifacts
    if [ -f "$PROJECT_ROOT/cpp/build/bin/Release/hypercube" ] || [ -f "$PROJECT_ROOT/cpp/build/bin/Release/hypercube.exe" ]; then
        log_info "C++ build: ✓"
    else
        log_warning "C++ build: ✗ (executables not found)"
    fi
fi

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

if [ "$RUN_INTEGRATION" = true ]; then
    log_subsection "Running Integration Tests"

    # Database integration tests (require PostgreSQL)
    if command -v psql &> /dev/null; then
        log_info "Running database integration tests..."

        # Set database environment variables
        export HC_DB_NAME="${HC_DB_NAME:-hypercube_test}"
        export HC_DB_USER="${HC_DB_USER:-postgres}"
        export HC_DB_HOST="${HC_DB_HOST:-hart-server}"
        export HC_DB_PORT="${HC_DB_PORT:-5432}"

        # Run integration tests if configured
        log_info "Database integration tests require manual setup"
        log_info "Set HC_DB_* environment variables and run: ctest -R IntegrationTest"
    else
        log_warning "PostgreSQL client not found, skipping database integration tests"
    fi

    # API integration tests
    if [ -d "$PROJECT_ROOT/Hartonomous-Orchestrator" ]; then
        log_info "Testing orchestrator components..."

        cd "$PROJECT_ROOT/Hartonomous-Orchestrator"

        # Test OpenAI gateway imports
        python3 -c "from openai_gateway.clients.hartonomous_client import get_hartonomous_client; print('Gateway import successful')" 2>/dev/null && \
        log_success "Orchestrator imports successful" || \
        log_warning "Orchestrator import issues detected"
    fi
fi

# ============================================================================
# BENCHMARKS
# ============================================================================

if [ "$RUN_BENCHMARK" = true ]; then
    log_subsection "Running Benchmarks"

    # Run benchmark script if available
    if [ -f "$SCRIPT_DIR/benchmark.sh" ]; then
        log_info "Running performance benchmarks..."
        "$SCRIPT_DIR/benchmark.sh" --quick
        check_result "Performance benchmarks"
    else
        log_warning "Benchmark script not found"
    fi
fi

# ============================================================================
# PLATFORM-SPECIFIC TESTS
# ============================================================================

# Run platform-specific tests if available
if command -v run_platform_tests >/dev/null 2>&1; then
    log_subsection "Running Platform-Specific Tests"
    run_platform_tests
fi

log_success "Test suite completed!"
echo
echo "For detailed C++ test output, run:"
echo "  cd cpp/build && ctest --output-on-failure -V"
echo
echo "For database integration tests, ensure PostgreSQL is running and run:"
echo "  ctest -R IntegrationTest"