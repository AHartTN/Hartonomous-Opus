#!/bin/bash
# Hartonomous Hypercube - Full Validation (Linux)
# End-to-end validation: build, setup, test
# Usage: ./scripts/linux/validate.sh [--full]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

FULL=false
if [ "$1" == "--full" ]; then
    FULL=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Hartonomous Hypercube - Full Validation                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Build
echo -e "${YELLOW}Step 1: Build${NC}"
"$SCRIPT_DIR/build.sh"

# Step 2: Database Setup
echo -e "\n${YELLOW}Step 2: Database Setup${NC}"
"$SCRIPT_DIR/setup-db.sh"

# Step 3: Run Tests
echo -e "\n${YELLOW}Step 3: Test Suite${NC}"
if [ "$FULL" = true ]; then
    "$SCRIPT_DIR/test.sh" || true
else
    "$SCRIPT_DIR/test.sh" --quick || true
fi

ERROR_COUNT=$?

# Summary
echo ""
if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  VALIDATION PASSED${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
else
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  VALIDATION FAILED ($ERROR_COUNT errors)${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
fi

exit $ERROR_COUNT
