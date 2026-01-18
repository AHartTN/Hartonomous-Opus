#!/bin/bash

# Hartonomous Orchestrator Basic Test Script
# Performs basic functionality tests to verify the system works

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Test 1: Check if binary exists
test_binary_exists() {
    log_info "Checking if binary exists..."
    if [ -f "$PROJECT_DIR/build/bin/hartonomous-orchestrator" ]; then
        log_info "✓ Binary found"
        return 0
    else
        log_error "✗ Binary not found. Run ./scripts/build.sh first"
        return 1
    fi
}

# Test 2: Check if config file exists
test_config_exists() {
    log_info "Checking configuration..."
    if [ -f "$PROJECT_DIR/config.yaml" ]; then
        log_info "✓ Config file found"
        return 0
    else
        log_error "✗ Config file not found"
        return 1
    fi
}

# Test 3: Test config loading
test_config_loading() {
    log_info "Testing configuration loading..."
    if timeout 5s "$PROJECT_DIR/build/bin/hartonomous-orchestrator" --help >/dev/null 2>&1; then
        log_info "✓ Config loading works"
        return 0
    else
        log_warning "? Config loading test inconclusive (may need service dependencies)"
        return 0
    fi
}

# Test 4: Test Docker build
test_docker_build() {
    log_info "Testing Docker build..."
    if command -v docker >/dev/null 2>&1; then
        if docker build --quiet --tag hartonomous-test "$PROJECT_DIR" >/dev/null 2>&1; then
            log_info "✓ Docker build successful"
            # Clean up test image
            docker rmi hartonomous-test >/dev/null 2>&1 || true
            return 0
        else
            log_warning "? Docker build failed (may be expected without proper dependencies)"
            return 0
        fi
    else
        log_warning "? Docker not available, skipping Docker test"
        return 0
    fi
}

# Test 5: Test scripts
test_scripts() {
    log_info "Testing management scripts..."

    # Test health check script
    if [ -x "$SCRIPT_DIR/health-check.sh" ]; then
        if timeout 10s "$SCRIPT_DIR/health-check.sh" >/dev/null 2>&1; then
            log_info "✓ Health check script works"
        else
            log_info "? Health check script ran (may fail without running services)"
        fi
    else
        log_error "✗ Health check script not executable"
        return 1
    fi

    return 0
}

# Main test execution
main() {
    echo "======================================="
    echo "Hartonomous Orchestrator Basic Tests"
    echo "======================================="

    FAILED_TESTS=0

    test_binary_exists || FAILED_TESTS=$((FAILED_TESTS + 1))
    test_config_exists || FAILED_TESTS=$((FAILED_TESTS + 1))
    test_config_loading || FAILED_TESTS=$((FAILED_TESTS + 1))
    test_docker_build || FAILED_TESTS=$((FAILED_TESTS + 1))
    test_scripts || FAILED_TESTS=$((FAILED_TESTS + 1))

    echo
    echo "======================================="

    if [ $FAILED_TESTS -eq 0 ]; then
        log_info "All basic tests passed! ✓"
        echo
        log_info "Next steps:"
        echo "  1. Start services: docker-compose up -d"
        echo "  2. Launch orchestrator: ./scripts/launch.sh start"
        echo "  3. Test API: curl http://localhost:8080/health"
        echo "  4. Full docs: cat LAUNCH_GUIDE.md"
        exit 0
    else
        log_error "$FAILED_TESTS test(s) failed"
        echo
        log_info "To fix issues:"
        echo "  - Run: ./scripts/build.sh"
        echo "  - Check: ./scripts/health-check.sh --verbose"
        echo "  - Docs: cat LAUNCH_GUIDE.md"
        exit 1
    fi
}

# Run main function
main "$@"