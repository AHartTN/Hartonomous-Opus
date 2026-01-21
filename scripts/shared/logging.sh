#!/bin/bash

# Common logging utilities for Hartonomous-Opus scripts
# Provides consistent logging across all build/test/deploy scripts

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

# Print section header
log_section() {
    echo
    echo "=================================================================================="
    echo "$*"
    echo "=================================================================================="
    echo
}

# Print subsection header
log_subsection() {
    echo
    echo "----------------------------------------------------------------------------------"
    echo "$*"
    echo "----------------------------------------------------------------------------------"
    echo
}

# Check command result and log appropriately
check_result() {
    local exit_code=$?
    local operation="$1"

    if [ $exit_code -eq 0 ]; then
        log_success "$operation completed successfully"
        return 0
    else
        log_error "$operation failed with exit code $exit_code"
        return $exit_code
    fi
}

# Export functions
export -f log_info
export -f log_success
export -f log_warning
export -f log_error
export -f log_section
export -f log_subsection
export -f check_result