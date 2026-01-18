#!/bin/bash
# Hartonomous Hypercube - Database Deployment (Linux)
# ============================================================================
# Deploys schema to PostgreSQL database.
# No admin privileges required - pure SQL deployment.
#
# Usage:
#   ./deploy.sh                    # Deploy to configured database
#   ./deploy.sh --rebuild          # Rebuild consolidated schema first
#   ./deploy.sh --create-db        # Create database if not exists
#   ./deploy.sh --reset            # DESTRUCTIVE: Drop and recreate
# ============================================================================

set -e

# Parse arguments
REBUILD=false
CREATE_DB=false
RESET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --create-db)
            CREATE_DB=true
            shift
            ;;
        --reset)
            RESET=true
            CREATE_DB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SQL_DIR="$HC_PROJECT_ROOT/sql"
DEPLOY_DIR="$SQL_DIR/deploy"
SCHEMA_FILE="$DEPLOY_DIR/full_schema.sql"

echo ""
echo "========================================"
echo "  Hypercube Database Deployment"
echo "========================================"
echo "  Target: $HC_DB_USER@$HC_DB_HOST:$HC_DB_PORT/$HC_DB_NAME"
echo ""

# ========================================================================
# REBUILD SCHEMA
# ========================================================================
if [[ "$REBUILD" == "true" ]] || [[ ! -f "$SCHEMA_FILE" ]]; then
    echo "[1/4] Building consolidated schema..."
    if [[ -f "$DEPLOY_DIR/build-schema.sh" ]]; then
        bash "$DEPLOY_DIR/build-schema.sh"
    else
        echo "  ERROR: build-schema.sh not found"
        exit 1
    fi
else
    echo "[1/4] Using existing schema file"
fi

# ========================================================================
# CONNECTION TEST
# ========================================================================
echo -n "[2/4] Testing connection..."
if ! psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -tAc "SELECT 1" > /dev/null 2>&1; then
    echo " FAILED"
    echo "  Cannot connect to PostgreSQL server"
    exit 1
fi
echo " OK"

# ========================================================================
# DATABASE CREATION/RESET
# ========================================================================
if [[ "$RESET" == "true" ]]; then
    echo "[3/4] Resetting database..."
    psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $HC_DB_NAME" 2>/dev/null || true
    echo "  Dropped $HC_DB_NAME"
fi

if [[ "$CREATE_DB" == "true" ]]; then
    echo -n "[3/4] Creating database..."
    DB_EXISTS=$(psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'" 2>/dev/null)
    if [[ "$DB_EXISTS" != "1" ]]; then
        psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -c "CREATE DATABASE $HC_DB_NAME" > /dev/null 2>&1
        echo " created"
    else
        echo " exists"
    fi
else
    echo -n "[3/4] Database check..."
    DB_EXISTS=$(psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'" 2>/dev/null)
    if [[ "$DB_EXISTS" != "1" ]]; then
        echo " NOT FOUND"
        echo "  Run with --create-db to create the database"
        exit 1
    fi
    echo " OK"
fi

# ========================================================================
# DEPLOY SCHEMA
# ========================================================================
echo -n "[4/4] Deploying schema..."

# Run with ON_ERROR_STOP and quiet mode
if psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" \
    -q -v ON_ERROR_STOP=1 -f "$SCHEMA_FILE" 2>&1 | grep -v "^NOTICE:"; then
    echo " OK"
else
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo " FAILED"
        exit 1
    fi
    echo " OK"
fi

# ========================================================================
# VERIFY
# ========================================================================
echo ""
echo "Verifying deployment..."

STATS=$(psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -tAc "SELECT * FROM db_stats()" 2>/dev/null)
if [[ -n "$STATS" ]]; then
    IFS='|' read -ra S <<< "$STATS"
    echo "  Atoms:        ${S[0]}"
    echo "  Compositions: ${S[1]}"
    echo "  Relations:    ${S[3]}"
else
    echo "  db_stats() not available yet"
fi

echo ""
echo "========================================"
echo "  Deployment Complete"
echo "========================================"
echo ""
