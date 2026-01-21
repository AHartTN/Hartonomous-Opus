#!/bin/bash

# ============================================================================
# Hartonomous-Opus - Unified Deployment Script
# ============================================================================
# Comprehensive deployment script that handles database setup, PostgreSQL
# extension installation, and orchestrator management in a unified way.
#
# Usage:
#   ./scripts/deploy.sh database      # Setup database and extensions
#   ./scripts/deploy.sh orchestrator  # Manage orchestrator service
#   ./scripts/deploy.sh extensions    # Install PostgreSQL extensions only
#   ./scripts/deploy.sh all          # Full deployment (default)
# ============================================================================

set -e

# Get script directory and load utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/shared/detect-platform.sh"
source "$SCRIPT_DIR/shared/logging.sh"

# Configuration variables
PLATFORM=$(detect_os)
PLATFORM_DIR=$(get_platform_dir)
PARALLEL_JOBS=$(nproc 2>/dev/null || echo 4)
COMMAND="${1:-all}"

# Database configuration (with defaults)
export HC_DB_HOST="${HC_DB_HOST:-hart-server}"
export HC_DB_PORT="${HC_DB_PORT:-5432}"
export HC_DB_USER="${HC_DB_USER:-postgres}"
export HC_DB_PASS="${HC_DB_PASS:-postgres}"
export HC_DB_NAME="${HC_DB_NAME:-hypercube}"

# Orchestrator configuration
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8700}"
export ORCHESTRATOR_HOST="${ORCHESTRATOR_HOST:-0.0.0.0}"

log_section "Hartonomous-Opus Deployment ($PLATFORM - $COMMAND)"

echo "  Database:       $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
echo "  Orchestrator:   $ORCHESTRATOR_HOST:$ORCHESTRATOR_PORT"
echo

# ============================================================================
# DATABASE DEPLOYMENT FUNCTIONS
# ============================================================================

setup_database_connection() {
    log_subsection "Testing Database Connection"

    if ! command -v psql &> /dev/null; then
        log_error "psql command not found. Please install PostgreSQL client."
        return 1
    fi

    # Test connection to postgres database (always exists)
    log_info "Testing database connection..."
    if ! PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -c "SELECT 1;" >/dev/null 2>&1; then
        log_error "Cannot connect to PostgreSQL at $HC_DB_HOST:$HC_DB_PORT"
        log_info "Please check:"
        log_info "  1. PostgreSQL is running"
        log_info "  2. Connection parameters are correct"
        log_info "  3. Firewall allows connections"
        return 1
    fi

    log_success "Database connection successful"
}

create_database() {
    log_subsection "Creating Database"

    # Check if database already exists
    if PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$HC_DB_NAME'" | grep -q 1; then
        log_info "Database '$HC_DB_NAME' already exists"
        return 0
    fi

    log_info "Creating database: $HC_DB_NAME"
    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres -c "CREATE DATABASE \"$HC_DB_NAME\";" 2>/dev/null || {
        log_warning "Database creation failed or already exists"
    }
}

enable_extensions() {
    log_subsection "Enabling PostgreSQL Extensions"

    log_info "Enabling PostGIS extension..."
    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS postgis;"

    # Enable Hartonomous extensions if available
    EXTENSIONS=("hypercube" "hypercube_ops" "embedding_ops" "generative" "semantic_ops")
    for ext in "${EXTENSIONS[@]}"; do
        log_info "Enabling extension: $ext"
        PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS $ext;" 2>/dev/null || {
            log_warning "Extension $ext not available or already enabled"
        }
    done
}

deploy_schema() {
    log_subsection "Deploying Database Schema"

    SCHEMA_DIR="$PROJECT_ROOT/sql/schema"
    FUNCTIONS_DIR="$PROJECT_ROOT/sql/functions"

    # Apply schema files
    if [ -d "$SCHEMA_DIR" ]; then
        log_info "Applying schema files from: $SCHEMA_DIR"
        for schema_file in "$SCHEMA_DIR"/*.sql; do
            if [ -f "$schema_file" ]; then
                log_info "Applying $(basename "$schema_file")"
                PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -f "$schema_file"
            fi
        done
    fi

    # Apply function files
    if [ -d "$FUNCTIONS_DIR" ]; then
        log_info "Applying function files from: $FUNCTIONS_DIR"
        find "$FUNCTIONS_DIR" -name "*.sql" | while read -r func_file; do
            log_info "Applying $(basename "$func_file")"
            PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -f "$func_file" 2>/dev/null || log_warning "Function application failed: $(basename "$func_file")"
        done
    fi
}

seed_atoms() {
    log_subsection "Seeding Atoms"

    # Check if atoms already exist
    ATOM_COUNT=$(PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]')
    ATOM_COUNT=${ATOM_COUNT:-0}

    if [ "$ATOM_COUNT" -ge 1100000 ]; then
        log_info "Atoms already seeded: $ATOM_COUNT"
        return 0
    fi

    log_info "Seeding atoms (current count: $ATOM_COUNT, need ~1.1M)..."

    # Find seeder executable
    SEEDER=""
    for candidate in \
        "$PROJECT_ROOT/cpp/build/bin/Release/seed_atoms_parallel" \
        "$PROJECT_ROOT/cpp/build/bin/Debug/seed_atoms_parallel" \
        "$PROJECT_ROOT/cpp/build/seed_atoms_parallel"
    do
        if [ -x "$candidate" ]; then
            SEEDER="$candidate"
            break
        fi
    done

    if [ -z "$SEEDER" ]; then
        log_error "Standalone seeder not found. Run build script first."
        return 1
    fi

    log_info "Using seeder: $SEEDER"
    if ! "$SEEDER" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -p "$HC_DB_PORT" >/tmp/seeder_output.txt 2>&1; then
        log_error "Atom seeding failed"
        cat /tmp/seeder_output.txt >&2
        rm -f /tmp/seeder_output.txt
        return 1
    fi

    NEW_COUNT=$(PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null | tr -d '[:space:]')
    if [ "$NEW_COUNT" -ge 1100000 ]; then
        log_success "Seeded $NEW_COUNT atoms"
    else
        log_error "Only $NEW_COUNT atoms seeded (expected ~1.1M)"
        return 1
    fi

    rm -f /tmp/seeder_output.txt
}

verify_database() {
    log_subsection "Database Verification"

    log_info "Verifying schema deployment..."
    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" -c "
SELECT 'Tables created:' as status, COUNT(*) as count FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
UNION ALL
SELECT 'Extensions installed:', COUNT(*) FROM pg_extension WHERE extname LIKE '%hypercube%';
" 2>/dev/null || log_warning "Schema verification queries failed"

    echo
    echo "Database: $HC_DB_NAME"
    echo "Host:     $HC_DB_HOST:$HC_DB_PORT"
    echo
    echo "To connect manually:"
    echo "  PGPASSWORD=$HC_DB_PASS psql -h $HC_DB_HOST -p $HC_DB_PORT -U $HC_DB_USER -d $HC_DB_NAME"
}

# PostgreSQL extensions build function
build_postgresql_extensions() {
    log_subsection "Building PostgreSQL Extensions"

    # Find PostgreSQL development package
    PG_DEV_PACKAGE=""
    for candidate in \
        "$SCRIPT_DIR/../deployment/pg-dev-package" \
        "$HOME/pg-dev-package" \
        "$SCRIPT_DIR/../pg-dev-package"
    do
        if [ -f "$candidate/pg-config.env" ]; then
            PG_DEV_PACKAGE="$candidate"
            break
        fi
    done

    if [ -z "$PG_DEV_PACKAGE" ]; then
        log_error "PostgreSQL development package not found"
        log_info "Run package-pg-dev-from-server.sh on the server first"
        log_info "Or extract pg-dev-package.tar.gz to deployment/pg-dev-package/"
        return 1
    fi

    log_info "Found PostgreSQL dev package: $PG_DEV_PACKAGE"

    # Source PostgreSQL environment
    if ! source "$PG_DEV_PACKAGE/pg-config.env"; then
        log_error "Failed to source PostgreSQL environment"
        return 1
    fi

    # Verify pg_config works
    if ! command -v pg_config &> /dev/null; then
        log_error "pg_config not available after sourcing environment"
        return 1
    fi

    PG_VERSION=$(pg_config --version)
    log_info "Using PostgreSQL: $PG_VERSION"

    # Create extensions build directory
    EXT_BUILD_DIR="$PROJECT_ROOT/cpp/build-pg-extensions"
    log_info "Extensions build directory: $EXT_BUILD_DIR"

    rm -rf "$EXT_BUILD_DIR"

    mkdir -p "$EXT_BUILD_DIR"
    cd "$EXT_BUILD_DIR"

    # Configure with CMake for extensions only
    log_info "Configuring extensions build with CMake..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DHYPERCUBE_ENABLE_PG_EXTENSIONS=ON \
          -DHYPERCUBE_ENABLE_TOOLS=OFF \
          -DHYPERCUBE_ENABLE_TESTS=OFF \
          .. || return 1

    # Build database wrapper library first
    log_info "Building database wrapper library..."
    cmake --build . --config Release --parallel "$PARALLEL_JOBS" --target db_wrapper_pg || return 1

    # Build extensions
    log_info "Building PostgreSQL extensions..."
    EXTENSIONS=(hypercube generative hypercube_ops embedding_ops semantic_ops)
    for ext in "${EXTENSIONS[@]}"; do
        log_info "Building extension: $ext"
        cmake --build . --config Release --parallel "$PARALLEL_JOBS" --target "$ext" || return 1
    done

    # Show built extensions
    echo
    echo "Built extensions in $EXT_BUILD_DIR/bin/:"
    ls -lh bin/*.so 2>/dev/null || ls -lh bin/*.dll 2>/dev/null || echo "  (none found)"

    log_success "PostgreSQL extensions built successfully"
}

# ============================================================================
# EXTENSION INSTALLATION FUNCTIONS
# ============================================================================

install_postgresql_extensions() {
    log_subsection "Installing PostgreSQL Extensions (Symlinked Mode)"

    if ! command -v pg_config &> /dev/null; then
        log_error "pg_config not found. Cannot install PostgreSQL extensions."
        return 1
    fi

    PG_LIBDIR=$(pg_config --pkglibdir)
    PG_SHAREDIR=$(pg_config --sharedir)
    PG_EXTDIR="$PG_SHAREDIR/extension"
    
    # Persistent storage for artifacts
    OPT_LIB="/opt/libraries/lib"
    OPT_EXT="/opt/libraries/pgext"

    log_info "Installing extensions:"
    echo "  Storage:    /opt/libraries"
    echo "  Postgres:   $PG_LIBDIR"

    # Check if we need sudo
    SUDO=""
    if [ ! -w "$PG_LIBDIR" ] || [ ! -w "/opt/libraries" ]; then
        SUDO="sudo"
        log_info "Using sudo for installation"
    fi
    
    # Ensure /opt structure exists
    $SUDO mkdir -p "$OPT_LIB"
    $SUDO mkdir -p "$OPT_EXT"

    # Find extensions build directory
    EXT_BUILD_DIR=""
    for candidate in \
        "$PROJECT_ROOT/cpp/build-pg-extensions" \
        "$PROJECT_ROOT/cpp/build"
    do
        # Check multiple possible locations for extensions
        for subdir in bin bin/Release lib/Release; do
            if [ -d "$candidate/$subdir" ] && [ -f "$candidate/$subdir/hypercube.so" ]; then
                EXT_BUILD_DIR="$candidate"
                EXT_BIN_DIR="$candidate/$subdir"
                break 2
            fi
        done
    done

    if [ -z "$EXT_BUILD_DIR" ]; then
        log_info "Extensions not found, building them now..."
        if ! build_postgresql_extensions; then
            log_error "Failed to build PostgreSQL extensions"
            return 1
        fi
        EXT_BUILD_DIR="$PROJECT_ROOT/cpp/build-pg-extensions"
        EXT_BIN_DIR="$EXT_BUILD_DIR/bin"
    fi

    log_info "Using extensions from: $EXT_BUILD_DIR"

    # Install shared libraries (Copy to /opt, Symlink to Postgres)
    EXTENSIONS=("hypercube" "generative" "hypercube_ops" "embedding_ops" "semantic_ops")
    BRIDGE_LIBS=("hypercube_c" "embedding_c" "generative_c" "db_wrapper_pg")
    
    for lib in "${EXTENSIONS[@]}" "${BRIDGE_LIBS[@]}"; do
        lib_file="$EXT_BIN_DIR/$lib.so"
        # Check alternative locations if not in bin
        if [ ! -f "$lib_file" ]; then
            lib_file="$EXT_BUILD_DIR/lib/$lib.so"
        fi
        
        if [ -f "$lib_file" ]; then
            # Copy to /opt
            $SUDO cp "$lib_file" "$OPT_LIB/"
            
            # Symlink in Postgres
            $SUDO rm -f "$PG_LIBDIR/$lib.so"
            $SUDO ln -sf "$OPT_LIB/$lib.so" "$PG_LIBDIR/$lib.so"
            
            log_success "Installed library: $lib.so (symlinked)"
        elif [[ " ${EXTENSIONS[*]} " =~ " ${lib} " ]]; then
            log_warning "Extension library not found: $lib.so"
        fi
    done

    # Install control/SQL files (Copy to /opt, Symlink to Postgres)
    deploy_ext_file() {
        local src="$1"
        local filename=$(basename "$src")
        
        $SUDO cp "$src" "$OPT_EXT/"
        $SUDO rm -f "$PG_EXTDIR/$filename"
        $SUDO ln -sf "$OPT_EXT/$filename" "$PG_EXTDIR/$filename"
        log_success "Installed: $filename"
    }

    for ext in "${EXTENSIONS[@]}"; do
        # Control files
        ctrl_file="$PROJECT_ROOT/cpp/$ext.control"
        if [ -f "$ctrl_file" ]; then
            deploy_ext_file "$ctrl_file"
        fi
        
        # SQL files
        sql_file="$PROJECT_ROOT/cpp/sql/$ext--1.0.sql"
        if [ -f "$sql_file" ]; then
            deploy_ext_file "$sql_file"
        fi
    done

    log_success "PostgreSQL extensions installed successfully"
    echo
    echo "To enable extensions in your database:"
    echo "  PGPASSWORD=$HC_DB_PASS psql -h $HC_DB_HOST -p $HC_DB_PORT -U $HC_DB_USER -d $HC_DB_NAME"
    echo "  CREATE EXTENSION hypercube;"
    echo "  CREATE EXTENSION hypercube_ops;"
    echo "  CREATE EXTENSION embedding_ops;"
    echo "  CREATE EXTENSION generative;"
    echo "  CREATE EXTENSION semantic_ops;"
}

# ============================================================================
# ORCHESTRATOR MANAGEMENT FUNCTIONS
# ============================================================================

manage_orchestrator() {
    local action="$1"
    local ORCHESTRATOR_DIR="$PROJECT_ROOT/Hartonomous-Orchestrator"
    local PID_FILE="$PROJECT_ROOT/orchestrator.pid"
    local LOG_FILE="$PROJECT_ROOT/logs/orchestrator.log"

    log_subsection "Orchestrator Management ($action)"

    # Check if orchestrator directory exists
    if [ ! -d "$ORCHESTRATOR_DIR" ]; then
        log_error "Orchestrator directory not found: $ORCHESTRATOR_DIR"
        return 1
    fi

    cd "$ORCHESTRATOR_DIR"

    case "$action" in
        start)
            start_orchestrator "$PID_FILE" "$LOG_FILE"
            ;;
        stop)
            stop_orchestrator "$PID_FILE"
            ;;
        restart)
            stop_orchestrator "$PID_FILE"
            sleep 2
            start_orchestrator "$PID_FILE" "$LOG_FILE"
            ;;
        status)
            check_orchestrator_status "$PID_FILE"
            ;;
        *)
            log_error "Unknown orchestrator action: $action"
            return 1
            ;;
    esac
}

start_orchestrator() {
    local pid_file="$1"
    local log_file="$2"

    log_info "Starting orchestrator..."

    # Check if already running
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log_error "Orchestrator is already running (PID: $pid)"
            return 1
        fi
    fi

    # Check Python availability
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Please install Python 3.9+"
        return 1
    fi

    # Check requirements
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt not found in orchestrator directory"
        return 1
    fi

    # Install/update dependencies
    log_info "Installing Python dependencies..."
    pip3 install -r requirements.txt --quiet

    # Set environment variables
    export ORCHESTRATOR_PORT="$ORCHESTRATOR_PORT"
    export ORCHESTRATOR_HOST="$ORCHESTRATOR_HOST"

    # Create logs directory
    mkdir -p "$(dirname "$log_file")"

    # Start orchestrator in background
    log_info "Starting FastAPI server on $ORCHESTRATOR_HOST:$ORCHESTRATOR_PORT..."

    if [ "$PLATFORM" = "windows" ]; then
        python3 openai_gateway.py > "$log_file" 2>&1 &
        local pid=$!
    else
        nohup python3 openai_gateway.py > "$log_file" 2>&1 &
        local pid=$!
    fi

    # Save PID
    echo $pid > "$pid_file"

    # Wait a moment for startup
    sleep 3

    # Check if process is still running
    if kill -0 "$pid" 2>/dev/null; then
        log_success "Orchestrator started successfully (PID: $pid)"
        echo
        echo "API Endpoint: http://localhost:$ORCHESTRATOR_PORT"
        echo "OpenAI-compatible: http://localhost:$ORCHESTRATOR_PORT/v1/chat/completions"
        echo "Health check: http://localhost:$ORCHESTRATOR_PORT/health"
        echo "Logs: $log_file"
    else
        log_error "Orchestrator failed to start"
        log_info "Check logs: $log_file"
        rm -f "$pid_file"
        return 1
    fi
}

stop_orchestrator() {
    local pid_file="$1"

    log_info "Stopping orchestrator..."

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -TERM "$pid" 2>/dev/null; then
            log_info "Sent SIGTERM to process $pid"

            # Wait for graceful shutdown
            local count=0
            while [ $count -lt 30 ] && kill -0 "$pid" 2>/dev/null; do
                sleep 1
                ((count++))
            done

            if kill -0 "$pid" 2>/dev/null; then
                log_warning "Process still running, sending SIGKILL"
                kill -KILL "$pid" 2>/dev/null || true
                sleep 1
            fi
        else
            log_warning "Could not send signal to process $pid"
        fi

        rm -f "$pid_file"
    else
        log_warning "No PID file found"
    fi

    # Also try to kill by port if available
    if command -v lsof &> /dev/null; then
        local port_pid
        port_pid=$(lsof -ti ":$ORCHESTRATOR_PORT" 2>/dev/null)
        if [ -n "$port_pid" ]; then
            log_info "Killing process using port $ORCHESTRATOR_PORT (PID: $port_pid)"
            kill -TERM "$port_pid" 2>/dev/null || true
        fi
    fi

    log_success "Orchestrator stopped"
}

check_orchestrator_status() {
    local pid_file="$1"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log_success "Orchestrator is running (PID: $pid)"
            return 0
        else
            log_warning "PID file exists but process not running, cleaning up"
            rm -f "$pid_file"
        fi
    fi

    log_info "Orchestrator is not running"
    return 1
}

# ============================================================================
# COMMAND EXECUTION
# ============================================================================

case "$COMMAND" in
    database)
        setup_database_connection || exit 1
        create_database || exit 1
        enable_extensions || exit 1
        deploy_schema || exit 1
        seed_atoms || exit 1
        verify_database || exit 1
        ;;
    orchestrator)
        # Default action is status if no second argument
        local action="${2:-status}"
        manage_orchestrator "$action" || exit 1
        ;;
    extensions)
        install_postgresql_extensions || exit 1
        ;;
    all)
        log_info "Performing full deployment..."

        # Database deployment
        setup_database_connection || exit 1
        create_database || exit 1
        install_postgresql_extensions || exit 1
        enable_extensions || exit 1
        deploy_schema || exit 1
        seed_atoms || exit 1
        verify_database || exit 1

        # Orchestrator deployment
        manage_orchestrator "start" || exit 1
        ;;
    help|--help|-h)
        echo "Usage: $0 <command> [options]"
        echo
        echo "Commands:"
        echo "  database            Setup database, schema, and seed atoms"
        echo "  orchestrator <cmd>  Manage orchestrator service (start|stop|restart|status)"
        echo "  extensions          Install PostgreSQL extensions only"
        echo "  all                 Full deployment (database + orchestrator)"
        echo
        echo "Examples:"
        echo "  $0 database"
        echo "  $0 orchestrator start"
        echo "  $0 orchestrator status"
        echo "  $0 extensions"
        echo "  $0 all"
        echo
        echo "Environment variables:"
        echo "  HC_DB_HOST, HC_DB_PORT, HC_DB_USER, HC_DB_PASS, HC_DB_NAME"
        echo "  ORCHESTRATOR_PORT, ORCHESTRATOR_HOST"
        exit 0
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac

log_success "Deployment operation completed successfully!"