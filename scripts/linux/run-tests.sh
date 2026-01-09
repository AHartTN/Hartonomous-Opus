#!/bin/bash
# Hartonomous Hypercube - Comprehensive Test Runner (Linux)
# ============================================================================
# Runs all available tests with proper error handling and reporting.
#
# Usage:
#   ./run-tests.sh                    # Run all tests
#   ./run-tests.sh --quick            # Skip slow/long-running tests
#   ./run-tests.sh --verbose          # Show detailed output
#   ./run-tests.sh --skip-db          # Skip database-dependent tests
#   ./run-tests.sh --filter pattern   # Run only tests matching pattern
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Parse arguments
QUICK=false
VERBOSE=false
SKIP_DB=false
FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --skip-db) SKIP_DB=true; shift ;;
        --filter) FILTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      Hartonomous Hypercube - Comprehensive Test Runner      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$QUICK" = true ]; then
    echo -e "${YELLOW}Quick mode: Skipping slow tests${NC}"
fi

if [ "$VERBOSE" = true ]; then
    echo -e "${YELLOW}Verbose mode: Showing detailed output${NC}"
fi

if [ "$SKIP_DB" = true ]; then
    echo -e "${YELLOW}Skipping database-dependent tests${NC}"
fi

if [ -n "$FILTER" ]; then
    echo -e "${YELLOW}Filter: Only running tests matching '$FILTER'${NC}"
fi

echo ""

# Build test command
TEST_CMD="$SCRIPT_DIR/test.sh"

if [ "$QUICK" = true ]; then
    TEST_CMD="$TEST_CMD --quick"
fi

# Run tests
if "$TEST_CMD"; then
    TESTS_PASSED=$?
else
    TESTS_FAILED=$?
fi

# Summary
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed!${NC}"
fi
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

exit $TESTS_FAILED
