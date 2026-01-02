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
    if ! psql -tAc "SELECT 1 FROM pg_tables WHERE tablename='atom'" | grep -q 1; then
        log_info "Applying schema..."
        psql -q -f sql/001_schema.sql
        log_ok "Schema applied"
    fi
    
    # Ensure lossless integer coordinate columns exist
    if ! psql -tAc "SELECT 1 FROM information_schema.columns WHERE table_name='atom' AND column_name='coord_x'" | grep -q 1; then
        log_info "Adding lossless coordinate columns..."
        psql -q -c "ALTER TABLE atom ADD COLUMN IF NOT EXISTS coord_x INTEGER"
        psql -q -c "ALTER TABLE atom ADD COLUMN IF NOT EXISTS coord_y INTEGER"
        psql -q -c "ALTER TABLE atom ADD COLUMN IF NOT EXISTS coord_z INTEGER"
        psql -q -c "ALTER TABLE atom ADD COLUMN IF NOT EXISTS coord_m INTEGER"
        psql -q -c "ALTER TABLE relation ADD COLUMN IF NOT EXISTS coord_x INTEGER"
        psql -q -c "ALTER TABLE relation ADD COLUMN IF NOT EXISTS coord_y INTEGER"
        psql -q -c "ALTER TABLE relation ADD COLUMN IF NOT EXISTS coord_z INTEGER"
        psql -q -c "ALTER TABLE relation ADD COLUMN IF NOT EXISTS coord_m INTEGER"
        log_ok "Lossless columns added"
    fi
}

ensure_atoms() {
    # Check if atoms exist AND have integer coordinates populated
    local count=$(psql -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null || echo 0)
    local with_coords=$(psql -tAc "SELECT COUNT(*) FROM atom WHERE coord_x IS NOT NULL" 2>/dev/null || echo 0)
    
    if [ "$count" -gt 1000000 ] && [ "$with_coords" -gt 1000000 ]; then
        return 0
    fi
    
    if [ "$count" -gt 1000000 ] && [ "$with_coords" -eq 0 ]; then
        log_warn "Atoms exist but missing integer coords - reseeding..."
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
    log_info "Applying SQL functions..."
    
    # Always apply these (idempotent - uses CREATE OR REPLACE)
    psql -q -f sql/002_functions.sql 2>/dev/null || true
    psql -q -f sql/003_ingestion.sql 2>/dev/null || true
    psql -q -f sql/006_spatial_queries.sql 2>/dev/null || true
    psql -q -f sql/008_metadata.sql 2>/dev/null || true
    psql -q -f sql/009_cascading_pair_encoding.sql 2>/dev/null || true
    
    # Apply lossless schema migration if needed
    if ! psql -tAc "SELECT 1 FROM information_schema.columns WHERE table_name='atom' AND column_name='coord_x'" | grep -q 1; then
        log_info "Adding lossless integer coordinate columns..."
        psql -q -f sql/010_lossless_schema.sql 2>/dev/null || true
    fi
    
    log_ok "Functions applied"
}

ensure_extension() {
    # Check if extension functions exist
    if psql -tAc "SELECT 1 FROM pg_proc WHERE proname='hypercube_blake3'" | grep -q 1; then
        return 0
    fi
    
    log_info "Installing hypercube extension..."
    
    local pg_lib=$(pg_config --pkglibdir)
    local pg_share=$(pg_config --sharedir)/extension
    local ext_so="cpp/build/hypercube.so"
    local ext_sql="cpp/sql/hypercube--1.0.sql"
    
    # Check if extension files exist in build
    if [ ! -f "$ext_so" ]; then
        log_error "Extension not built: $ext_so"
        log_info "Building now..."
        ensure_build
    fi
    
    # Check if we can write to extension directories (group permissions set up)
    if [ -w "$pg_lib" ] && [ -w "$pg_share" ]; then
        # We have write access - install without sudo
        cp "$ext_so" "$pg_lib/hypercube.so"
        cp "$ext_sql" "$pg_share/"
        
        cat > "$pg_share/hypercube.control" << 'EOF'
comment = 'Hypercube 4D semantic substrate functions'
default_version = '1.0'
module_pathname = '$libdir/hypercube'
relocatable = false
EOF
        log_ok "Extension files installed"
    elif [ -f "$pg_lib/hypercube.so" ]; then
        # Already installed by someone else
        log_ok "Extension files already present"
    else
        log_error "Cannot write to PostgreSQL directories"
        echo ""
        echo "Fix with these commands (one time setup):"
        echo "  sudo groupadd postgres-extensions"
        echo "  sudo usermod -aG postgres-extensions \$USER"
        echo "  sudo chown -R root:postgres-extensions $pg_lib $pg_share"
        echo "  sudo chmod -R 775 $pg_lib $pg_share"
        echo "  # Then log out and back in for group to take effect"
        echo ""
        return 1
    fi
    
    # Create extension in database
    psql -c "CREATE EXTENSION IF NOT EXISTS hypercube;" || {
        log_error "Failed to create extension"
        return 1
    }
    log_ok "Extension created in database"
}

ensure_build() {
    if [ -x "cpp/build/seed_atoms_parallel" ] && [ -x "cpp/build/cpe_ingest" ]; then
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
        'depth 1-' || COALESCE(max(depth)::text, '0')
    FROM relation
    UNION ALL
    SELECT 
        'edges',
        count(*)::text,
        '-'
    FROM relation_edge;
    "
    
    # Show CPE depth distribution if we have compositions
    local comp_count=$(psql -tAc "SELECT COUNT(*) FROM relation")
    if [ "$comp_count" -gt 0 ]; then
        echo ""
        echo "=== Composition Depth Distribution ==="
        psql -c "SELECT * FROM cpe_stats LIMIT 10;" 2>/dev/null || true
    fi
}

cmd_tree() {
    local text="$1"
    [ -z "$text" ] && { log_error "Usage: ./setup.sh tree <text>"; exit 1; }
    
    local query_id
    query_id=$(psql -tAc "SELECT encode(cpe_ingest_text('$text'), 'hex');")
    
    echo "Composition: $query_id"
    echo "Structure:"
    echo ""
    
    psql -c "SELECT * FROM cpe_show_tree(decode('$query_id', 'hex'), 10);"
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
    
    # Ensure C++ ingester is built
    if [ ! -x "cpp/build/cpe_ingest" ]; then
        ensure_build
    fi
    
    log_info "Ingesting: $target"
    
    # Use C++ CPE ingester (fast)
    ./cpp/build/cpe_ingest \
        -d "$PGDATABASE" \
        -U "$PGUSER" \
        -h "$PGHOST" \
        "$target"
    
    echo ""
    cmd_status
}

ingest_file() {
    local file="$1"
    local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
    local start_time=$(date +%s)
    
    # Skip binary files and very large files
    if file "$file" | grep -q "binary\|executable\|data"; then
        log_warn "Skipping binary file: $file"
        return 0
    fi
    
    if [ "$size" -gt 10485760 ]; then  # 10MB limit
        log_warn "Skipping large file (>10MB): $file"
        return 0
    fi
    
    # Read entire file content
    local content
    content=$(cat "$file" 2>/dev/null) || {
        log_warn "Cannot read: $file"
        return 0
    }
    
    # Skip empty files
    if [ -z "$content" ]; then
        return 0
    fi
    
    # Escape for SQL (double single quotes)
    local escaped="${content//\'/\'\'}"
    
    # Ingest as a single document using CPE
    local result
    result=$(psql -tAc "SELECT encode(cpe_ingest_document('${escaped}'), 'hex');" 2>&1)
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ "$result" =~ ^[a-f0-9]{64}$ ]]; then
        log_ok "Ingested $file (${size} bytes, ${duration}s) → ${result:0:16}..."
    else
        log_warn "Failed to ingest $file: ${result:0:100}"
    fi
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
    
    psql -c "SELECT encode(cpe_ingest_text('$text'), 'hex') as composition_id;"
}

cmd_similar() {
    local text="$1"
    [ -z "$text" ] && { log_error "Usage: ./setup.sh similar <text>"; exit 1; }
    
    # First ingest the query text, then find similar
    local query_id
    query_id=$(psql -tAc "SELECT encode(cpe_ingest_text('$text'), 'hex');")
    
    echo "Query composition: $query_id"
    echo ""
    
    # Find compositions with similar centroids using Hilbert distance (lossless integer coords)
    psql -c "
    WITH query AS (
        SELECT coord_x, coord_y, coord_z, coord_m, hilbert_hi, hilbert_lo
        FROM relation WHERE id = decode('$query_id', 'hex')
    )
    SELECT 
        encode(r.id, 'hex') as id,
        r.depth,
        r.atom_count,
        -- Euclidean distance in 4D using raw integer coords
        sqrt(
            power((int32_to_uint32(r.coord_x) - int32_to_uint32(q.coord_x))::numeric, 2) +
            power((int32_to_uint32(r.coord_y) - int32_to_uint32(q.coord_y))::numeric, 2) +
            power((int32_to_uint32(r.coord_z) - int32_to_uint32(q.coord_z))::numeric, 2) +
            power((int32_to_uint32(r.coord_m) - int32_to_uint32(q.coord_m))::numeric, 2)
        )::numeric(20,0) as distance
    FROM relation r, query q
    WHERE r.id != decode('$query_id', 'hex')
    ORDER BY 
        abs(r.hilbert_hi - q.hilbert_hi),
        abs(r.hilbert_lo - q.hilbert_lo)
    LIMIT 10;
    "
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
    ingest <path>           Ingest file or directory (whole files, not lines)
    query <text>            Get composition ID for text
    similar <text>          Find similar compositions by centroid distance
    tree <text>             Show the Merkle DAG structure of a text
    reset                   Drop and reset database

EXAMPLES:
    # First time setup
    ./setup.sh init
    
    # Ingest a directory of documents
    ./setup.sh ingest ~/Documents/notes/
    
    # Ingest a model
    ./setup.sh ingest ./test-data/embedding_models/
    
    # Query and visualize
    ./setup.sh query "Hello World"
    ./setup.sh tree "Mississippi"
    ./setup.sh similar "machine learning"

HOW IT WORKS:
    Text is ingested using Cascading Pair Encoding (CPE):
    
    1. Characters → Atom IDs (from Unicode seed)
    2. Pairs merged → Binary compositions (a,b) → ab
    3. Cascade → Each pass halves the count until single root
    4. Documents → Split on paragraphs, ingest each, cascade roots
    
    Example: "Hello" (5 chars) becomes:
        H + e → He
        l + l → ll  
        He + ll → Hell
        Hell + o → Hello  (root composition)
    
    Each composition is content-addressed (BLAKE3 hash of children).
    Identical substrings across documents share the same hash.

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
    tree)       shift; cmd_tree "$@" ;;
    reset)      cmd_reset ;;
    help|--help|-h) cmd_help ;;
    *)
        log_error "Unknown command: $1"
        echo "Run './setup.sh help' for usage"
        exit 1
        ;;
esac
