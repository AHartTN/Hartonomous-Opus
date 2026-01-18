#!/bin/bash
# ============================================================================
# Hartonomous Hypercube - Linux Environment Setup
# ============================================================================
# Source this: source scripts/linux/env.sh
#
# This script:
#   1. Loads config.env settings
#   2. Sets up Intel oneAPI if available
#   3. Configures database connection helpers
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
# LOAD CONFIG.ENV
# ============================================================================

CONFIG_FILE="$PROJECT_ROOT/scripts/config.env"
if [ -f "$CONFIG_FILE" ]; then
    export $(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | xargs)
fi

# ============================================================================
# INTEL oneAPI (if available)
# ============================================================================

INTEL_ONEAPI_PATHS=(
    "/opt/intel/oneapi"
    "$HOME/intel/oneapi"
    "/usr/local/intel/oneapi"
)

for oneapi_path in "${INTEL_ONEAPI_PATHS[@]}"; do
    if [ -d "$oneapi_path" ]; then
        # Source the oneAPI environment if setvars.sh exists
        if [ -f "$oneapi_path/setvars.sh" ]; then
            source "$oneapi_path/setvars.sh" --force > /dev/null 2>&1
        fi
        break
    fi
done

# ============================================================================
# DEFAULTS
# ============================================================================

export HC_DB_HOST="${HC_DB_HOST:-hart-server}"
export HC_DB_PORT="${HC_DB_PORT:-5432}"
export HC_DB_USER="${HC_DB_USER:-postgres}"
export HC_DB_PASS="${HC_DB_PASS:-postgres}"
export HC_DB_NAME="${HC_DB_NAME:-hypercube}"
export HC_BUILD_TYPE="${HC_BUILD_TYPE:-Release}"
export HC_PARALLEL_JOBS="${HC_PARALLEL_JOBS:-$(nproc)}"
export HC_INGEST_THRESHOLD="${HC_INGEST_THRESHOLD:-0.5}"

export HC_PROJECT_ROOT="$PROJECT_ROOT"
export HC_BUILD_DIR="$PROJECT_ROOT/cpp/build"
export HC_BIN_DIR="$PROJECT_ROOT/cpp/build/bin/$HC_BUILD_TYPE"

# Connection info for libpq
export HC_CONNINFO="host=$HC_DB_HOST port=$HC_DB_PORT dbname=$HC_DB_NAME user=$HC_DB_USER password=$HC_DB_PASS"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

hc_psql() {
    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" "$@"
}

hc_psql_admin() {
    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres "$@"
}

export -f hc_psql
export -f hc_psql_admin

# ============================================================================
# BANNER
# ============================================================================

if [ -z "$HC_ENV_LOADED" ]; then
    export HC_ENV_LOADED=1
    echo "Hartonomous environment loaded"
    echo "  Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
    echo "  User:     $HC_DB_USER"
    echo "  Project:  $HC_PROJECT_ROOT"
fi
