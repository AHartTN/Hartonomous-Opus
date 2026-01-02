#!/bin/bash
# =============================================================================
# Hartonomous Hypercube - Complete Setup & Operations
# =============================================================================
#
# Single entry point for all operations. Idempotent - safe to run multiple times.
#
# SETUP (first time):
#   ./setup.sh init
#
# INGEST CONTENT:
#   ./setup.sh ingest <file_or_directory>
#   ./setup.sh ingest ~/Documents/
#   ./setup.sh ingest model.safetensors
#
# QUERY:
#   ./setup.sh query "hello world"
#   ./setup.sh similar "computer"
#
# STATUS:
#   ./setup.sh status
#
# Environment variables (or set in .env file):
#   PGHOST=localhost
#   PGPORT=5432
#   PGUSER=hartonomous
#   PGPASSWORD=hartonomous
#   PGDATABASE=hypercube

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Defaults
export PGHOST="${PGHOST:-localhost}"
export PGPORT="${PGPORT:-5432}"
export PGUSER="${PGUSER:-hartonomous}"
export PGPASSWORD="${PGPASSWORD:-hartonomous}"
export PGDATABASE="${PGDATABASE:-hypercube}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Helper functions
# =============================================================================

check_postgres() {
    if ! command -v psql &> /dev/null; then
        log_error "PostgreSQL client (psql) not found"
        echo "Install with: sudo apt install postgresql-client"
        exit 1
    fi
}

check_connection() {
    # Try connecting to postgres database first (always exists)
    if PGDATABASE=postgres psql -c "SELECT 1" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

check_db_exists() {
    PGDATABASE=postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$PGDATABASE'" | grep -q 1
}

ensure_database() {
    if check_db_exists; then
        return 0
    fi
    
    log_info "Creating database $PGDATABASE..."
    PGDATABASE=postgres createdb "$PGDATABASE" || {
        log_error "Failed to create database. Check credentials."
        exit 1
    }
    log_ok "Database created"
}

ensure_schema() {
    # Check if atom table exists
    if psql -tAc "SELECT 1 FROM pg_tables WHERE tablename='atom'" | grep -q 1; then
        return 0
    fi
    
    log_info "Applying schema..."
    psql -q -f sql/001_schema.sql
    log_ok "Schema applied"
}

ensure_atoms() {
    local count=$(psql -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null || echo 0)
    if [ "$count" -gt 1000000 ]; then
        return 0
    fi
    
    log_info "Seeding atoms (this takes ~30 seconds)..."
    
    # Build seed tool if needed
    if [ ! -x "cpp/build/seed_atoms_parallel" ]; then
        ensure_build
    fi
    
    # Run parallel seeder
    ./cpp/build/seed_atoms_parallel -d "$PGDATABASE" -U "$PGUSER" -h "$PGHOST" 2>&1 | \
        grep -E "(Complete|Rate|atoms)" || true
    
    log_ok "Atoms seeded"
}

ensure_functions() {
    # Check if key function exists
    if psql -tAc "SELECT 1 FROM pg_proc WHERE proname='hypercube_ingest_text'" | grep -q 1; then
        return 0
    fi
    
    log_info "Applying SQL functions..."
    psql -q -f sql/002_functions.sql 2>/dev/null || true
    psql -q -f sql/003_ingestion.sql 2>/dev/null || true
    psql -q -f sql/006_spatial_queries.sql 2>/dev/null || true
    psql -q -f sql/008_metadata.sql 2>/dev/null || true
    log_ok "Functions applied"
}

ensure_extension() {
    # Check if extension functions exist
    if psql -tAc "SELECT 1 FROM pg_proc WHERE proname='hypercube_blake3'" | grep -q 1; then
        return 0
    fi
    
    log_info "Installing hypercube extension..."
    
    # Install .so file to PostgreSQL lib directory
    local pg_lib=$(pg_config --pkglibdir)
    local ext_file="cpp/build/hypercube.so"
    
    if [ -f "$ext_file" ]; then
        # Try sudo install, fall back to asking user
        if sudo cp "$ext_file" "$pg_lib/hypercube.so" 2>/dev/null; then
            sudo cp cpp/sql/hypercube--1.0.sql "$pg_lib/../share/postgresql/extension/" 2>/dev/null || true
            
            # Create extension control file
            sudo tee "$pg_lib/../share/postgresql/extension/hypercube.control" > /dev/null << 'EOF'
comment = 'Hypercube 4D semantic substrate functions'
default_version = '1.0'
module_pathname = '$libdir/hypercube'
relocatable = false
EOF
            
            psql -c "CREATE EXTENSION IF NOT EXISTS hypercube;" || {
                # Extension installation failed, try loading functions directly
                log_warn "Extension registration failed, loading functions manually"
                psql -c "CREATE OR REPLACE FUNCTION hypercube_blake3(bytea) RETURNS bytea AS '\$libdir/hypercube', 'hypercube_blake3' LANGUAGE C IMMUTABLE STRICT;"
                psql -c "CREATE OR REPLACE FUNCTION hypercube_coords_to_hilbert(bigint, bigint, bigint, bigint) RETURNS TABLE(hilbert_lo bigint, hilbert_hi bigint) AS '\$libdir/hypercube', 'hypercube_coords_to_hilbert' LANGUAGE C IMMUTABLE STRICT;"
            }
            log_ok "Extension installed"
        else
            log_warn "Cannot install extension without sudo"
            echo "Run: sudo cp $ext_file $pg_lib/hypercube.so"
            echo "Then run './setup.sh init' again"
            
            # Try to use pgcrypto as fallback for hashing
            log_info "Attempting pgcrypto fallback..."
            psql -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;" 2>/dev/null || true
            
            # Create wrapper function using SHA256 (not BLAKE3, but functional)
            psql -c "
            CREATE OR REPLACE FUNCTION hypercube_blake3(data bytea) RETURNS bytea AS \$\$
                SELECT digest(data, 'sha256');
            \$\$ LANGUAGE SQL IMMUTABLE STRICT;
            " 2>/dev/null && log_ok "Using SHA256 fallback (pgcrypto)" || log_error "No hashing available"
        fi
    else
        log_error "Extension not built: $ext_file"
        log_info "Run: cd cpp/build && cmake .. && make"
    fi
}

ensure_build() {
    if [ -x "cpp/build/seed_atoms_parallel" ] && [ -x "cpp/build/extract_embeddings" ]; then
        return 0
    fi
    
    log_info "Building C++ tools..."
    mkdir -p cpp/build
    cd cpp/build
    cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null
    make -j$(nproc) 2>&1 | grep -E "(Built|Linking|error)" || true
    cd ../..
    log_ok "Build complete"
}

# =============================================================================
# Commands
# =============================================================================

cmd_init() {
    echo "=============================================="
    echo "  Hartonomous Hypercube Initialization"
    echo "=============================================="
    echo "Database: $PGDATABASE @ $PGHOST:$PGPORT"
    echo ""
    
    check_postgres
    
    # Test connection
    log_info "Testing database connection..."
    if ! check_connection; then
        log_error "Cannot connect to PostgreSQL"
        echo ""
        echo "Make sure PostgreSQL is running and credentials are set:"
        echo "  export PGUSER=your_user"
        echo "  export PGPASSWORD=your_password"
        echo ""
        echo "Or create a .env file:"
        echo "  PGUSER=hartonomous"
        echo "  PGPASSWORD=hartonomous"
        exit 1
    fi
    log_ok "Connected"
    
    ensure_database
    ensure_build
    ensure_schema
    ensure_extension
    ensure_atoms
    ensure_functions
    
    echo ""
    echo "=============================================="
    log_ok "Initialization complete!"
    echo "=============================================="
    echo ""
    cmd_status
}

cmd_status() {
    echo "=== Hypercube Status ==="
    echo "Database: $PGDATABASE @ $PGHOST"
    echo ""
    
    if ! check_connection; then
        log_error "Cannot connect to PostgreSQL"
        return 1
    fi
    
    if ! check_db_exists; then
        log_warn "Database '$PGDATABASE' does not exist"
        echo "Run './setup.sh init' to initialize"
        return 1
    fi
    
    psql -c "
    SELECT 
        'atoms' as entity,
        count(*)::text as count,
        '-' as info
    FROM atom
    UNION ALL
    SELECT 
        'compositions',
        count(*)::text,
        'depth 1-' || max(depth)::text
    FROM relation
    UNION ALL
    SELECT 
        'edges',
        count(*)::text,
        '-'
    FROM relation_edge;
    "
}

cmd_ingest() {
    local target="$1"
    
    if [ -z "$target" ]; then
        log_error "Usage: ./setup.sh ingest <file_or_directory>"
        exit 1
    fi
    
    if [ ! -e "$target" ]; then
        log_error "Not found: $target"
        exit 1
    fi
    
    ensure_functions
    
    if [ -d "$target" ]; then
        # Directory - ingest all text files
        log_info "Ingesting directory: $target"
        
        local count=0
        find "$target" -type f \( -name "*.txt" -o -name "*.md" -o -name "*.json" \) | while read -r file; do
            log_info "  $file"
            ingest_file "$file"
            count=$((count + 1))
        done
        
        # Check for safetensors
        find "$target" -type f -name "*.safetensors" | while read -r file; do
            log_info "  Model: $file"
            ingest_model "$file"
        done
        
        log_ok "Directory ingested"
        
    elif [[ "$target" == *.safetensors ]]; then
        ingest_model "$target"
        
    else
        ingest_file "$target"
    fi
    
    cmd_status
}

ingest_file() {
    local file="$1"
    local batch=""
    local count=0
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^\[.*\]$ ]] && continue  # Skip special tokens
        
        escaped="${line//\'/\'\'}"
        batch="${batch}SELECT hypercube_ingest_text('${escaped}');"
        count=$((count + 1))
        
        if [ $((count % 500)) -eq 0 ]; then
            echo "$batch" | psql -q
            batch=""
        fi
    done < "$file"
    
    [ -n "$batch" ] && echo "$batch" | psql -q
    log_ok "Ingested $count items from $file"
}

ingest_model() {
    local model="$1"
    local vocab_file=""
    local model_dir=$(dirname "$model")
    
    # Find vocab file
    if [ -f "$model_dir/vocab.txt" ]; then
        vocab_file="$model_dir/vocab.txt"
    fi
    
    if [ ! -x "cpp/build/extract_embeddings" ]; then
        ensure_build
    fi
    
    log_info "Ingesting vocab..."
    [ -n "$vocab_file" ] && ingest_file "$vocab_file"
    
    log_info "Extracting semantic edges (threshold 0.85)..."
    ./cpp/build/extract_embeddings \
        --model "$model" \
        ${vocab_file:+--vocab "$vocab_file"} \
        --threshold 0.85 \
        -d "$PGDATABASE" \
        -U "$PGUSER" \
        -h "$PGHOST" 2>&1 | grep -E "(Complete|Edges|Sparsity)"
    
    log_ok "Model ingested"
}

cmd_query() {
    local text="$1"
    [ -z "$text" ] && { log_error "Usage: ./setup.sh query <text>"; exit 1; }
    
    psql -c "SELECT encode(hypercube_ingest_text('$text'), 'hex') as composition_id;"
}

cmd_similar() {
    local text="$1"
    [ -z "$text" ] && { log_error "Usage: ./setup.sh similar <text>"; exit 1; }
    
    psql -c "SELECT * FROM hypercube_find_similar('$text', 10);"
}

cmd_reset() {
    read -p "This will DELETE all data. Are you sure? (yes/no) " confirm
    if [ "$confirm" = "yes" ]; then
        log_warn "Dropping database..."
        dropdb --if-exists "$PGDATABASE"
        log_ok "Database dropped"
        log_info "Run './setup.sh init' to reinitialize"
    else
        log_info "Cancelled"
    fi
}

cmd_help() {
    cat << 'EOF'
Hartonomous Hypercube - Geometric Semantic Substrate

USAGE:
    ./setup.sh <command> [arguments]

COMMANDS:
    init                    Initialize database, build tools, seed atoms
    status                  Show database statistics
    ingest <path>           Ingest file or directory
    query <text>            Get composition ID for text
    similar <text>          Find similar compositions
    reset                   Drop and reset database

EXAMPLES:
    # First time setup
    ./setup.sh init
    
    # Ingest a model
    ./setup.sh ingest ./test-data/embedding_models/models--sentence-transformers--all-MiniLM-L6-v2/
    
    # Ingest documents
    ./setup.sh ingest ~/Documents/notes/
    
    # Query similarity
    ./setup.sh similar "machine learning"

CONFIGURATION:
    Set environment variables or create .env file:
    
    PGHOST=localhost
    PGPORT=5432
    PGUSER=hartonomous
    PGPASSWORD=hartonomous
    PGDATABASE=hypercube

EOF
}

# =============================================================================
# Main
# =============================================================================

case "${1:-help}" in
    init)       cmd_init ;;
    status)     cmd_status ;;
    ingest)     shift; cmd_ingest "$@" ;;
    query)      shift; cmd_query "$@" ;;
    similar)    shift; cmd_similar "$@" ;;
    reset)      cmd_reset ;;
    help|--help|-h) cmd_help ;;
    *)
        log_error "Unknown command: $1"
        echo "Run './setup.sh help' for usage"
        exit 1
        ;;
esac
