#!/bin/bash
# Hartonomous Hypercube - Linux Environment Setup
# Source this: source scripts/linux/env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load config.env if it exists
CONFIG_FILE="$PROJECT_ROOT/scripts/config.env"
if [ -f "$CONFIG_FILE" ]; then
    export $(grep -v '^#' "$CONFIG_FILE" | xargs)
fi

# Defaults (app-specific, not global PG* vars)
export HC_DB_HOST="${HC_DB_HOST:-localhost}"
export HC_DB_PORT="${HC_DB_PORT:-5432}"
export HC_DB_USER="${HC_DB_USER:-hartonomous}"
export HC_DB_PASS="${HC_DB_PASS:-hartonomous}"
export HC_DB_NAME="${HC_DB_NAME:-hypercube}"
export HC_BUILD_TYPE="${HC_BUILD_TYPE:-Release}"
export HC_PARALLEL_JOBS="${HC_PARALLEL_JOBS:-$(nproc)}"

export HC_PROJECT_ROOT="$PROJECT_ROOT"
export HC_BUILD_DIR="$PROJECT_ROOT/cpp/build"

# Connection info for libpq (used by C++ tools)
export HC_CONNINFO="host=$HC_DB_HOST port=$HC_DB_PORT dbname=$HC_DB_NAME user=$HC_DB_USER password=$HC_DB_PASS"

# Helper function for psql with app credentials
hc_psql() {
    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" "$@"
}

# Helper for running psql against postgres db (for db creation)
hc_psql_admin() {
    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres "$@"
}

export -f hc_psql
export -f hc_psql_admin

echo "Hartonomous environment loaded"
echo "  Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo "  Project:  $HC_PROJECT_ROOT"
