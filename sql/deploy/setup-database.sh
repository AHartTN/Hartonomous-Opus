#!/bin/bash

# =============================================================================
# HYPERCUBE DATABASE SETUP AND REPAIR SCRIPT
# =============================================================================
# This script handles complete PostgreSQL database initialization for Hartonomous.
# Features:
#   - Database creation (hypercube)
#   - Extension installation (PostGIS + custom extensions)
#   - Schema deployment (tables, functions, indexes, triggers)
#   - Permissions and roles setup
#   - Data initialization and repairs
#   - Comprehensive validation
#   - Rollback capabilities
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_ROOT="$(dirname "$SCRIPT_DIR")"
FULL_SCHEMA_FILE="$SCRIPT_DIR/full_schema.sql"

# Database configuration - override with environment variables
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-hypercube}"
ADMIN_USER="${ADMIN_USER:-postgres}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-postgres}"
APP_USER="${APP_USER:-hartonomous}"
APP_PASSWORD="${APP_PASSWORD:-hartonomous}"

# Extensions to install
EXTENSIONS=(
    "postgis"
    "hypercube"
    "generative"
    "hypercube_ops"
    "embedding_ops"
    "semantic_ops"
)

# Logging
LOG_FILE="/tmp/hypercube-setup-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

die() {
    error "$1"
    exit 1
}

# Check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        die "$1 is not installed or not in PATH"
    fi
}

# Execute SQL command
execute_sql() {
    local db="$1"
    local sql="$2"
    local desc="$3"

    log "Executing: $desc"
    if ! PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$db" -c "$sql"; then
        error "Failed to execute: $desc"
        return 1
    fi
}

# Execute SQL file
execute_sql_file() {
    local db="$1"
    local file="$2"
    local desc="$3"

    log "Executing SQL file: $desc ($file)"
    if ! PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$db" -f "$file"; then
        error "Failed to execute SQL file: $desc"
        return 1
    fi
}

# Check if database exists
database_exists() {
    local db_name="$1"
    PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -lqt | cut -d\| -f1 | grep -qw "$db_name"
}

# Check if extension is installed in database
extension_exists() {
    local db_name="$1"
    local ext_name="$2"
    PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$db_name" -c "SELECT 1 FROM pg_extension WHERE extname = '$ext_name';" | grep -q "1"
}

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

check_prerequisites() {
    log "Checking prerequisites..."

    # Check required commands
    check_command psql
    check_command pg_config

    # Check PostgreSQL version
    local pg_version
    pg_version=$(pg_config --version | awk '{print $2}' | cut -d. -f1)
    if [[ $pg_version -lt 14 ]]; then
        die "PostgreSQL 14+ required, found version $pg_version"
    fi
    log "PostgreSQL version: $pg_version"

    # Check connection to postgres database
    if ! PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d postgres -c "SELECT 1;" &> /dev/null; then
        die "Cannot connect to PostgreSQL as $ADMIN_USER. Check credentials and server."
    fi

    # Check if schema file exists
    if [[ ! -f "$FULL_SCHEMA_FILE" ]]; then
        warn "Schema file not found: $FULL_SCHEMA_FILE"
        warn "Building schema file..."
        if [[ -f "$SCRIPT_DIR/build-schema.sh" ]]; then
            bash "$SCRIPT_DIR/build-schema.sh"
        else
            die "Schema file missing and build script not found"
        fi
    fi

    log "Prerequisites check passed"
}

create_database() {
    log "Creating database '$DB_NAME' if it doesn't exist..."

    if database_exists "$DB_NAME"; then
        warn "Database '$DB_NAME' already exists, skipping creation"
        return 0
    fi

    local sql="CREATE DATABASE \"$DB_NAME\" WITH OWNER = \"$ADMIN_USER\" ENCODING = 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8' TEMPLATE = template0;"
    if ! execute_sql "postgres" "$sql" "Create database $DB_NAME"; then
        die "Failed to create database"
    fi

    log "Database '$DB_NAME' created successfully"
}

install_extensions() {
    log "Installing PostgreSQL extensions..."

    for ext in "${EXTENSIONS[@]}"; do
        if extension_exists "$DB_NAME" "$ext"; then
            log "Extension '$ext' already exists, skipping"
            continue
        fi

        # Special handling for custom extensions - check if they exist in PostgreSQL
        if [[ "$ext" != "postgis" ]]; then
            if ! PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d postgres -c "SELECT 1 FROM pg_available_extensions WHERE name = '$ext';" | grep -q "1"; then
                warn "Extension '$ext' not available in PostgreSQL. Make sure custom extensions are built and installed."
                continue
            fi
        fi

        local sql="CREATE EXTENSION IF NOT EXISTS \"$ext\";"
        if ! execute_sql "$DB_NAME" "$sql" "Install extension $ext"; then
            warn "Failed to install extension '$ext', continuing..."
        else
            log "Extension '$ext' installed"
        fi
    done
}

deploy_schema() {
    log "Deploying database schema..."

    if ! execute_sql_file "$DB_NAME" "$FULL_SCHEMA_FILE" "Deploy full schema"; then
        die "Schema deployment failed"
    fi

    log "Schema deployed successfully"
}

setup_permissions() {
    log "Setting up database permissions and roles..."

    # Create application user if it doesn't exist
    if ! PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d postgres -c "SELECT 1 FROM pg_roles WHERE rolname = '$APP_USER';" | grep -q "1"; then
        local sql="CREATE ROLE \"$APP_USER\" LOGIN PASSWORD '$APP_PASSWORD';"
        execute_sql "postgres" "$sql" "Create application user $APP_USER"
    else
        log "User '$APP_USER' already exists"
    fi

    # Grant permissions
    local sql="
        GRANT CONNECT ON DATABASE \"$DB_NAME\" TO \"$APP_USER\";
        GRANT USAGE ON SCHEMA public TO \"$APP_USER\";
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO \"$APP_USER\";
        GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO \"$APP_USER\";
        GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO \"$APP_USER\";
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO \"$APP_USER\";
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO \"$APP_USER\";
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE ON SEQUENCES TO \"$APP_USER\";
    "
    execute_sql "$DB_NAME" "$sql" "Set up permissions for $APP_USER"

    log "Permissions set up successfully"
}

initialize_data() {
    log "Running data initialization..."

    # Run seed atoms procedure if it exists
    if PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -c "SELECT 1 FROM pg_proc WHERE proname = 'seed_atoms';" | grep -q "1"; then
        execute_sql "$DB_NAME" "SELECT seed_atoms();" "Initialize seed atoms"
    else
        warn "seed_atoms procedure not found, skipping data initialization"
    fi

    log "Data initialization completed"
}

validate_setup() {
    log "Validating database setup..."

    # Check required tables
    local required_tables=("composition" "relation" "projection")
    for table in "${required_tables[@]}"; do
        if ! PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -c "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '$table';" | grep -q "1"; then
            error "Required table '$table' not found"
            return 1
        fi
    done

    # Check required functions
    local required_functions=("db_stats" "atom_knn" "search")
    for func in "${required_functions[@]}"; do
        if ! PGPASSWORD="$ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -c "SELECT 1 FROM pg_proc WHERE proname = '$func';" | grep -q "1"; then
            error "Required function '$func' not found"
            return 1
        fi
    done

    # Test db_stats function
    if ! execute_sql "$DB_NAME" "SELECT db_stats();" "Validate db_stats function"; then
        error "db_stats function test failed"
        return 1
    fi

    # Check extensions
    for ext in "${EXTENSIONS[@]}"; do
        if ! extension_exists "$DB_NAME" "$ext"; then
            warn "Extension '$ext' not installed"
        fi
    done

    log "Database setup validation passed"
}

# =============================================================================
# ROLLBACK FUNCTIONS
# =============================================================================

rollback() {
    warn "Starting rollback procedure..."

    # Confirm rollback
    read -p "Are you sure you want to rollback the database setup? This will DROP the database and all data. (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Rollback cancelled"
        exit 0
    fi

    # Drop database
    if database_exists "$DB_NAME"; then
        local sql="DROP DATABASE \"$DB_NAME\";"
        execute_sql "postgres" "$sql" "Drop database $DB_NAME"
        log "Database '$DB_NAME' dropped"
    fi

    # Drop application user
    local sql="DROP ROLE IF EXISTS \"$APP_USER\";"
    execute_sql "postgres" "$sql" "Drop application user $APP_USER"
    log "Application user '$APP_USER' dropped"

    log "Rollback completed"
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Comprehensive database setup script for Hartonomous PostgreSQL database.

OPTIONS:
    -h, --help              Show this help message
    --rollback              Rollback database setup (drops database and user)
    --skip-validation       Skip final validation checks
    --verbose               Enable verbose output

ENVIRONMENT VARIABLES:
    DB_HOST                 Database host (default: localhost)
    DB_PORT                 Database port (default: 5432)
    DB_NAME                 Database name (default: hypercube)
    ADMIN_USER              Admin username (default: postgres)
    ADMIN_PASSWORD          Admin password (default: postgres)
    APP_USER                Application username (default: hartonomous)
    APP_PASSWORD            Application password (default: hartonomous)

EXAMPLES:
    $0                              # Full setup
    $0 --rollback                  # Rollback everything
    DB_HOST=hart-server $0         # Setup on remote server

EOF
}

main() {
    local rollback_mode=false
    local skip_validation=false
    local verbose=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            --rollback)
                rollback_mode=true
                shift
                ;;
            --skip-validation)
                skip_validation=true
                shift
                ;;
            --verbose)
                verbose=true
                set -x
                shift
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    log "Hartonomous Database Setup Script"
    log "=================================="
    log "Database: $DB_NAME"
    log "Host: $DB_HOST:$DB_PORT"
    log "Admin User: $ADMIN_USER"
    log "App User: $APP_USER"
    log "Log file: $LOG_FILE"
    log ""

    if [[ "$rollback_mode" == true ]]; then
        rollback
        exit 0
    fi

    # Run setup steps
    check_prerequisites
    create_database
    install_extensions
    deploy_schema
    setup_permissions
    initialize_data

    if [[ "$skip_validation" == false ]]; then
        validate_setup
    fi

    log ""
    log "🎉 Database setup completed successfully!"
    log "Database: $DB_NAME"
    log "Connection: postgresql://$APP_USER:***@$DB_HOST:$DB_PORT/$DB_NAME"
    log "Log file: $LOG_FILE"
}

# Run main function
main "$@"