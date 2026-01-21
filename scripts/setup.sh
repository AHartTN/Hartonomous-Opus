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

# Load config from scripts/config.env first (new preferred location)
if [ -f scripts/config.env ]; then
    export $(grep -v '^#' scripts/config.env | xargs)
fi

# Fall back to .env for backwards compatibility
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Support both HC_* (new) and PG* (legacy) variable names
export PGHOST="${HC_DB_HOST:-${PGHOST:-HART-SERVER}}"
export PGPORT="${HC_DB_PORT:-${PGPORT:-5432}}"
export PGUSER="${HC_DB_USER:-${PGUSER:-postgres}}"
export PGPASSWORD="${HC_DB_PASS:-${PGPASSWORD:-postgres}}"
export PGDATABASE="${HC_DB_NAME:-${PGDATABASE:-hypercube}}"

# Also set HC_* for new scripts
export HC_DB_HOST="$PGHOST"
export HC_DB_PORT="$PGPORT"
export HC_DB_USER="$PGUSER"
export HC_DB_PASS="$PGPASSWORD"
export HC_DB_NAME="$PGDATABASE"
export HC_PROJECT_ROOT="$SCRIPT_DIR"
export HC_BUILD_DIR="$SCRIPT_DIR/cpp/build"

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
    # Check if new three-table schema exists (atom, composition, relation tables)
    if psql -tAc "SELECT 1 FROM information_schema.tables WHERE table_name='composition'" | grep -q 1; then
        return 0  # Already using new schema
    fi

    # Drop old tables if they exist (migration to new schema)
    if psql -tAc "SELECT 1 FROM pg_tables WHERE tablename='relation_edge'" | grep -q 1; then
        log_info "Migrating from old schema to new three-table schema..."
        psql -q -c "DROP TABLE IF EXISTS relation_edge CASCADE;"
        psql -q -c "DROP TABLE IF EXISTS relation CASCADE;"
        psql -q -c "DROP TABLE IF EXISTS composition CASCADE;"
        psql -q -c "DROP TABLE IF EXISTS atom CASCADE;"
        psql -q -c "DROP TYPE IF EXISTS atom_category CASCADE;"
        psql -q -c "DROP DOMAIN IF EXISTS blake3_hash CASCADE;"
    fi

    log_info "Applying schema files..."

    # Tables first
    psql -q -f sql/schema/01_tables.sql || {
        log_error "Failed to create tables"
        return 1
    }

    # Indexes second
    psql -q -f sql/schema/02_indexes.sql || {
        log_error "Failed to create indexes"
        return 1
    }

    # Constraints third
    psql -q -f sql/schema/03_constraints.sql 2>/dev/null || true

    # Functions - apply all function SQL files
    for func_dir in sql/functions/*/; do
        [ -d "$func_dir" ] || continue
        for func_file in "$func_dir"*.sql; do
            [ -f "$func_file" ] || continue
            psql -q -f "$func_file" 2>/dev/null || true
        done
    done

    log_ok "Schema applied"
}

ensure_atoms() {
    # Check if atoms exist
    local count=$(psql -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null || echo 0)

    if [ "$count" -gt 1000000 ]; then
        return 0
    fi

    log_info "Seeding atoms (this takes ~30 seconds)..."

    # Build seed tool if needed
    if [ ! -x "cpp/build/seed_atoms_parallel" ]; then
        ensure_build
    fi

    # Run parallel seeder with proper error checking
    if ./cpp/build/seed_atoms_parallel -d "$PGDATABASE" -U "$PGUSER" -h "$PGHOST" >/tmp/seeder_output.txt 2>&1; then
        # Check if seeding actually succeeded by counting atoms
        local new_count=$(psql -tAc "SELECT COUNT(*) FROM atom" 2>/dev/null || echo 0)
        if [ "$new_count" -gt 1000000 ]; then
            log_ok "Atoms seeded successfully ($new_count atoms)"
        else
            log_error "Seeder completed but only $new_count atoms found (expected >1M)"
            return 1
        fi
    else
        log_error "Parallel seeder failed"
        cat /tmp/seeder_output.txt >&2
        rm -f /tmp/seeder_output.txt
        return 1
    fi
    rm -f /tmp/seeder_output.txt
}


ensure_extension() {
    # Check if extension functions exist
    if psql -tAc "SELECT 1 FROM pg_proc WHERE proname='hypercube_blake3'" | grep -q 1; then
        return 0
    fi
    
    log_info "Installing hypercube extension..."
    
    # Install extension files via the unified deploy script
    log_info "Installing extension files..."
    ./deploy.sh extensions || {
        log_error "Failed to install extensions via deploy.sh"
        return 1
    }
    
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
    if ! cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null; then
        log_error "CMake configuration failed"
        cd ../..
        return 1
    fi
    if ! make -j$(nproc) >/tmp/build_output.txt 2>&1; then
        log_error "Build failed"
        cat /tmp/build_output.txt >&2
        rm -f /tmp/build_output.txt
        cd ../..
        return 1
    fi
    cd ../..
    rm -f /tmp/build_output.txt
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
    ensure_schema      # Creates tables, indexes, constraints, functions
    ensure_extension   # Hypercube extension requires PostGIS
    ensure_atoms
    
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

    # Show atom and composition counts
    psql -c "
    SELECT 'atoms' as entity, count(*)::text as count, 'Unicode codepoints' as info FROM atom
    UNION ALL
    SELECT 'compositions' as entity, count(*)::text as count,
           CASE WHEN count(*) > 0 THEN 'depth 1-' || max(depth)::text ELSE 'none' END as info
    FROM composition;
    "

    # Show depth distribution if we have compositions
    local comp_count=$(psql -tAc "SELECT COUNT(*) FROM composition")
    if [ "$comp_count" -gt 0 ]; then
        echo ""
        echo "=== Composition Depth Distribution ==="
        psql -c "SELECT depth, count(*) as count FROM composition GROUP BY depth ORDER BY depth;" 2>/dev/null || true
    fi
}

cmd_tree() {
    local text="$1"
    [ -z "$text" ] && { log_error "Usage: ./setup.sh tree <text>"; exit 1; }

    # Write text to temp file and ingest with C++ tool
    local tmpfile=$(mktemp)
    echo -n "$text" > "$tmpfile"

    log_info "Ingesting text..."
    local root_id
    root_id=$(./cpp/build/cpe_ingest -d "$PGDATABASE" -U "$PGUSER" -h "$PGHOST" "$tmpfile" 2>/dev/null)
    rm -f "$tmpfile"

    if [ -z "$root_id" ]; then
        log_error "Failed to ingest text"
        return 1
    fi

    echo "Composition: $root_id"
    echo "Structure:"
    echo ""

    # Show tree structure using recursive CTE
    psql -c "
    WITH RECURSIVE tree AS (
        SELECT
            CASE WHEN a.id IS NOT NULL THEN a.id ELSE c.id END as id,
            CASE WHEN a.id IS NOT NULL THEN convert_from(a.value, 'UTF8') ELSE c.label END as display_text,
            CASE WHEN a.id IS NOT NULL THEN 0 ELSE c.depth END as depth,
            0 as level,
            ARRAY[]::INTEGER[] as path,
            CASE WHEN a.id IS NOT NULL THEN 'atom' ELSE 'composition' END as type
        FROM (SELECT decode('$root_id', 'hex') as root_id) r
        LEFT JOIN atom a ON a.id = r.root_id
        LEFT JOIN composition c ON c.id = r.root_id
        WHERE a.id IS NOT NULL OR c.id IS NOT NULL
        UNION ALL
        SELECT
            CASE WHEN a.id IS NOT NULL THEN a.id ELSE c.id END as id,
            CASE WHEN a.id IS NOT NULL THEN convert_from(a.value, 'UTF8') ELSE c.label END as display_text,
            CASE WHEN a.id IS NOT NULL THEN 0 ELSE c.depth END as depth,
            t.level + 1,
            t.path || cc.ordinal::INTEGER,
            CASE WHEN a.id IS NOT NULL THEN 'atom' ELSE 'composition' END as type
        FROM tree t
        JOIN composition_child cc ON cc.composition_id = t.id
        LEFT JOIN atom a ON a.id = cc.child_id AND cc.child_type = 'A'
        LEFT JOIN composition c ON c.id = cc.child_id AND cc.child_type = 'C'
        WHERE t.level < 10 AND t.type = 'composition'
    )
    SELECT
        repeat('  ', level) || COALESCE(display_text, '...') as node,
        encode(id, 'hex')::text as id,
        depth
    FROM tree
    ORDER BY path;
    "
}

cmd_ingest() {
    local target="$1"
    local model_name="$2"

    if [ -z "$target" ]; then
        log_error "Usage: ./setup.sh ingest <path> [model_name]"
        exit 1
    fi

    if [ ! -e "$target" ]; then
        log_error "Not found: $target"
        exit 1
    fi

    ensure_build

    # Check if this is a model directory (has .safetensors or tokenizer.json)
    if [ -d "$target" ] && ([ -f "$target/model.safetensors" ] || [ -f "$target/tokenizer.json" ] || find "$target" -name "*.safetensors" 2>/dev/null | grep -q .); then
        # Model ingestion
        if [ -z "$model_name" ]; then
            model_name=$(basename "$target")
        fi
        log_info "Ingesting model: $model_name from $target"
        if ! ./cpp/build/hc.exe ingest -d "$PGDATABASE" -n "$model_name" "$target"; then
            log_error "Model ingestion failed"
            return 1
        fi
    else
        # Text ingestion
        log_info "Ingesting text: $target"
        if ! ./cpp/build/cpe_ingest \
                -d "$PGDATABASE" \
                -U "$PGUSER" \
                -h "$PGHOST" \
                "$target"; then
            log_error "Text ingestion failed"
            return 1
        fi
    fi

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
    if ! ./cpp/build/extract_embeddings \
            --model "$model" \
            ${vocab_file:+--vocab "$vocab_file"} \
            --threshold 0.85 \
            -d "$PGDATABASE" \
            -U "$PGUSER" \
            -h "$PGHOST" >/tmp/extract_output.txt 2>&1; then
        log_error "Semantic edge extraction failed"
        cat /tmp/extract_output.txt >&2
        rm -f /tmp/extract_output.txt
        return 1
    fi
    cat /tmp/extract_output.txt | grep -E "(Complete|Edges|Sparsity)" || true
    rm -f /tmp/extract_output.txt

    log_ok "Model ingested"
}

cmd_query() {
    local text="$1"
    [ -z "$text" ] && { log_error "Usage: ./setup.sh query <text>"; exit 1; }

    # Write text to temp file and ingest with C++ tool
    local tmpfile=$(mktemp)
    echo -n "$text" > "$tmpfile"

    local root_id
    root_id=$(./cpp/build/cpe_ingest -d "$PGDATABASE" -U "$PGUSER" -h "$PGHOST" "$tmpfile" 2>/dev/null)
    rm -f "$tmpfile"

    echo "composition_id"
    echo "----------------------------------------------------------------"
    echo "$root_id"
}

cmd_similar() {
    local text="$1"
    [ -z "$text" ] && { log_error "Usage: ./setup.sh similar <text>"; exit 1; }

    # Write text to temp file and ingest with C++ tool
    local tmpfile=$(mktemp)
    echo -n "$text" > "$tmpfile"

    local query_id
    query_id=$(./cpp/build/cpe_ingest -d "$PGDATABASE" -U "$PGUSER" -h "$PGHOST" "$tmpfile" 2>/dev/null)
    rm -f "$tmpfile"

    if [ -z "$query_id" ]; then
        log_error "Failed to ingest text"
        return 1
    fi

    echo "Query composition: $query_id"
    echo ""

    # Find compositions with similar centroids using 4D spatial distance
    psql -c "
    WITH query AS (
        SELECT COALESCE(a.geom, c.centroid) as centroid
        FROM (SELECT decode('$query_id', 'hex') as id) q
        LEFT JOIN atom a ON a.id = q.id
        LEFT JOIN composition c ON c.id = q.id
    )
    SELECT
        left(encode(c.id, 'hex'), 16) || '...' as id,
        c.depth,
        c.atom_count as atoms,
        centroid_distance(c.centroid, q.centroid)::numeric(12,8) as distance,
        c.label as content
    FROM composition c, query q
    WHERE c.id != decode('$query_id', 'hex')
      AND c.centroid IS NOT NULL
    ORDER BY centroid_distance(c.centroid, q.centroid)
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

cmd_test() {
    log_info "Running Hartonomous test suite..."
    
    # Build tests if needed
    if [ ! -f cpp/build/test_integration ] || [ ! -f cpp/build/test_query_api ]; then
        log_info "Building test executables..."
        cd cpp/build
        make -j4 test_blake3 test_coordinates test_hilbert test_integration test_query_api >/dev/null 2>&1
        cd "$SCRIPT_DIR"
    fi
    
    local total_passed=0
    local total_failed=0
    
    echo
    echo "=== Core Unit Tests ==="
    for test in test_blake3 test_coordinates test_hilbert; do
        if [ -f "cpp/build/$test" ]; then
            if cpp/build/$test >/dev/null 2>&1; then
                log_ok "$test"
                ((total_passed++)) || true
            else
                log_error "$test"
                ((total_failed++)) || true
            fi
        fi
    done
    
    echo
    echo "=== Integration Tests ==="
    if cpp/build/test_integration 2>&1 | grep -q "Failed: 0"; then
        local int_passed=$(cpp/build/test_integration 2>&1 | grep "Passed:" | awk '{print $2}')
        log_ok "test_integration ($int_passed assertions)"
        ((total_passed++)) || true
    else
        log_error "test_integration"
        ((total_failed++)) || true
    fi
    
    echo
    echo "=== Query API Tests ==="
    if cpp/build/test_query_api 2>&1 | grep -q "Failed: 0"; then
        local api_passed=$(cpp/build/test_query_api 2>&1 | grep "Passed:" | awk '{print $2}')
        log_ok "test_query_api ($api_passed assertions)"
        ((total_passed++)) || true
    else
        log_error "test_query_api"
        ((total_failed++)) || true
    fi
    
    echo
    echo "=============================================="
    if [ $total_failed -eq 0 ]; then
        log_ok "All $total_passed test suites passed!"
    else
        log_error "$total_failed test suite(s) failed"
        exit 1
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
    test                    Run the full test suite
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
    
    PGHOST=HART-SERVER
    PGPORT=5432
    PGUSER=postgres
    PGPASSWORD=postgres
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
    test)       cmd_test ;;
    reset)      cmd_reset ;;
    help|--help|-h) cmd_help ;;
    *)
        log_error "Unknown command: $1"
        echo "Run './setup.sh help' for usage"
        exit 1
        ;;
esac
