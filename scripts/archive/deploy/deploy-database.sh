#!/bin/bash

# ============================================================================
# Hartonomous-Opus - Database Deployment Script
# ============================================================================
# Sets up PostgreSQL database schema and extensions for Hartonomous-Opus
#
# Usage:
#   ./scripts/deploy-database.sh          # Setup database with defaults
#   ./scripts/deploy-database.sh --clean  # Drop and recreate database
#   ./scripts/deploy-database.sh --test   # Setup test database
# ============================================================================

set -e

# Get script directory and load utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/shared/detect-platform.sh"
source "$SCRIPT_DIR/shared/logging.sh"

# Database configuration
PLATFORM=$(detect_os)
DB_NAME="${HC_DB_NAME:-hypercube}"
DB_USER="${HC_DB_USER:-postgres}"
DB_HOST="${HC_DB_HOST:-hart-server}"
DB_PORT="${HC_DB_PORT:-5432}"
DB_PASS="${HC_DB_PASS:-postgres}"
CLEAN=false
TEST_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean) CLEAN=true; shift ;;
        --test) TEST_MODE=true; DB_NAME="${DB_NAME}_test"; shift ;;
        --db-name) DB_NAME="$2"; shift 2 ;;
        --db-user) DB_USER="$2"; shift 2 ;;
        --db-host) DB_HOST="$2"; shift 2 ;;
        --db-port) DB_PORT="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean       Drop and recreate database"
            echo "  --test        Setup test database (${DB_NAME}_test)"
            echo "  --db-name     Database name (default: hypercube)"
            echo "  --db-user     Database user (default: postgres)"
            echo "  --db-host     Database host (default: hart-server)"
            echo "  --db-port     Database port (default: 5432)"
            echo "  --help, -h    Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Set PGPASSWORD for non-interactive operation
export PGPASSWORD="$DB_PASS"

# Source environment for additional variables
source "$PROJECT_ROOT/scripts/linux/env.sh"

log_section "Database Deployment ($PLATFORM)"

echo "  Database:    $DB_NAME"
echo "  Host:        $DB_HOST:$DB_PORT"
echo "  User:        $DB_USER"
echo "  Clean:       $CLEAN"
echo "  Test Mode:   $TEST_MODE"
echo

# Check PostgreSQL client
if ! command -v psql &> /dev/null; then
    log_error "psql command not found. Please install PostgreSQL client."
    exit 1
fi

# Test database connection
log_info "Testing database connection..."
if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT version();" >/dev/null 2>&1; then
    log_error "Cannot connect to PostgreSQL at $DB_HOST:$DB_PORT"
    log_info "Please check:"
    log_info "  1. PostgreSQL is running"
    log_info "  2. Connection parameters are correct"
    log_info "  3. Firewall allows connections"
    exit 1
fi

log_success "Database connection successful"

# ============================================================================
# DATABASE SETUP
# ============================================================================

# Drop database if clean requested
if [ "$CLEAN" = true ]; then
    log_info "Dropping existing database (if exists)..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS \"$DB_NAME\";" 2>/dev/null || true
fi

# Create database
log_info "Creating database: $DB_NAME"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE \"$DB_NAME\";" 2>/dev/null || {
    log_warning "Database $DB_NAME already exists or creation failed"
}

# Enable PostGIS extension
log_info "Enabling PostGIS extension..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# ============================================================================
# SCHEMA DEPLOYMENT
# ============================================================================

log_subsection "Deploying Database Schema"

SCHEMA_DIR="$PROJECT_ROOT/sql/schema"

if [ ! -d "$SCHEMA_DIR" ]; then
    log_error "Schema directory not found: $SCHEMA_DIR"
    exit 1
fi

log_info "Applying schema files from: $SCHEMA_DIR"

# Apply schema files in order
for schema_file in "$SCHEMA_DIR"/*.sql; do
    if [ -f "$schema_file" ]; then
        log_info "Applying $(basename "$schema_file")"
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$schema_file"
    fi
done

check_result "Schema deployment"

# ============================================================================
# FUNCTION DEPLOYMENT
# ============================================================================

log_subsection "Deploying Functions"

FUNCTIONS_DIR="$PROJECT_ROOT/sql/functions"

if [ -d "$FUNCTIONS_DIR" ]; then
    log_info "Applying function files from: $FUNCTIONS_DIR"

    # Apply function files (recursive)
    find "$FUNCTIONS_DIR" -name "*.sql" | while read -r func_file; do
        log_info "Applying $(basename "$func_file")"
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$func_file"
    done
fi

check_result "Function deployment"

# Verify schema deployment
log_info "Verifying schema deployment..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT 'Tables created:' as status, COUNT(*) as count FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
UNION ALL
SELECT 'Extensions installed:', COUNT(*) FROM pg_extension WHERE extname IN ('postgis', 'hypercube', 'hypercube_ops');
" 2>/dev/null || log_warning "Schema verification queries failed"

# ============================================================================
# ATOM SEEDING
# ============================================================================

log_subsection "Seeding Atoms"

ATOM_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]')
ATOM_COUNT=${ATOM_COUNT:-0}

if [ "$ATOM_COUNT" -ge 1100000 ]; then
    log_info "Atoms already seeded: $ATOM_COUNT"
else
    log_info "Seeding atoms (current count: $ATOM_COUNT, need ~1.1M)..."

    SEEDER="$HC_BIN_DIR/seed_atoms_parallel"

    # Build the standalone seeder if not available
    if [ ! -x "$SEEDER" ]; then
        log_info "Standalone seeder not found, building it..."
        if ! "$SCRIPT_DIR/build.sh" --no-install >/dev/null 2>&1; then
            log_error "Failed to build standalone seeder"
            exit 1
        fi
        if [ ! -x "$SEEDER" ]; then
            log_error "Standalone seeder still not available after build"
            exit 1
        fi
    fi

    log_info "Using standalone seeder: $SEEDER"
    if ! "$SEEDER" -d "$DB_NAME" -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" >/dev/null 2>&1; then
        log_error "Standalone seeder failed"
        exit 1
    fi

    NEW_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]')
    if [ "$NEW_COUNT" -ge 1100000 ]; then
        log_success "Seeded $NEW_COUNT atoms"
    else
        log_error "Only $NEW_COUNT atoms seeded (expected ~1.1M)"
        exit 1
    fi
fi

# ============================================================================
# EXTENSION VERIFICATION
# ============================================================================

log_subsection "Verifying Extensions"

log_info "Checking installed extensions..."

psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name LIKE '%hypercube%'
ORDER BY name;" | head -20

echo

# ============================================================================
# TEST DATA (OPTIONAL)
# ============================================================================

if [ "$TEST_MODE" = true ]; then
    log_subsection "Loading Test Data"

    log_info "Test mode enabled - loading sample data..."

    # Create test schema or load test data here
    # This would depend on having test data files

    log_info "Test data loading not implemented yet"
    log_info "To add test data, create SQL files in sql/test_data/"
fi

# ============================================================================
# VERIFICATION
# ============================================================================

log_subsection "Database Verification"

log_info "Running basic verification queries..."

# Check tables exist
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT schemaname, tablename
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;"

echo

# Check extensions are available
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT * FROM pg_extension WHERE extname LIKE '%hypercube%';"

echo

log_success "Database deployment completed successfully!"

echo
echo "Database: $DB_NAME"
echo "Host:     $DB_HOST:$DB_PORT"
echo
echo "To connect manually:"
echo "  psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
echo
echo "Next steps:"
echo "  1. Ingest data: ./scripts/data/ingest-models.py"
echo "  2. Start orchestrator: ./scripts/deploy-orchestrator.sh"