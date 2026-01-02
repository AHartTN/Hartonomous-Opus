#!/bin/bash
# Content ingestion script for Hartonomous Hypercube
# Ingests text files, vocab files, or stdin as universal compositions
#
# Usage:
#   ./ingest_content.sh [options] [file...]
#
# Options:
#   -d, --dbname NAME     Database name (default: hypercube)
#   -U, --user USER       Database user (default: $PGUSER or hartonomous)
#   -h, --host HOST       Database host (default: localhost)
#   -p, --port PORT       Database port (default: 5432)
#   --vocab               Treat input as vocab file (one token per line)
#   --text                Treat input as plain text (default)
#   --batch-size N        Process N items per batch (default: 500)
#   --skip-special        Skip lines matching [TOKEN] or <TOKEN> patterns
#   --stats               Show statistics after ingestion
#
# Environment:
#   PGPASSWORD            Database password
#
# Examples:
#   # Ingest vocab file
#   ./ingest_content.sh --vocab vocab.txt
#
#   # Ingest multiple files
#   ./ingest_content.sh file1.txt file2.txt file3.txt
#
#   # Ingest from stdin
#   cat document.txt | ./ingest_content.sh -
#
#   # With credentials
#   PGPASSWORD=secret ./ingest_content.sh -U myuser --vocab vocab.txt

set -e

# Defaults
DB_NAME="hypercube"
DB_USER="${PGUSER:-hartonomous}"
DB_HOST="${PGHOST:-localhost}"
DB_PORT="${PGPORT:-5432}"
MODE="text"
BATCH_SIZE=500
SKIP_SPECIAL=false
SHOW_STATS=false
FILES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dbname) DB_NAME="$2"; shift 2 ;;
        -U|--user) DB_USER="$2"; shift 2 ;;
        -h|--host) DB_HOST="$2"; shift 2 ;;
        -p|--port) DB_PORT="$2"; shift 2 ;;
        --vocab) MODE="vocab"; shift ;;
        --text) MODE="text"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --skip-special) SKIP_SPECIAL=true; shift ;;
        --stats) SHOW_STATS=true; shift ;;
        --help)
            head -40 "$0" | tail -35
            exit 0
            ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) FILES+=("$1"); shift ;;
    esac
done

# Build psql connection string
PSQL_CMD="psql -U $DB_USER -h $DB_HOST -p $DB_PORT -d $DB_NAME -q"

# Test connection
if ! $PSQL_CMD -c "SELECT 1" >/dev/null 2>&1; then
    echo "Error: Cannot connect to database $DB_NAME"
    echo "Check PGPASSWORD and connection parameters"
    exit 1
fi

echo "=============================================="
echo "  Hartonomous Content Ingestion"
echo "=============================================="
echo "Database: $DB_NAME (${DB_USER}@${DB_HOST}:${DB_PORT})"
echo "Mode: $MODE"
echo "Batch size: $BATCH_SIZE"
echo ""

# Get counts before
BEFORE_COMPOSITIONS=$($PSQL_CMD -tAc "SELECT count(*) FROM relation")
BEFORE_EDGES=$($PSQL_CMD -tAc "SELECT count(*) FROM relation_edge")

# Ingestion function
ingest_batch() {
    local sql="$1"
    echo "$sql" | $PSQL_CMD
}

# Process input
process_input() {
    local input="$1"
    local count=0
    local batch=""
    local total=0

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines
        [[ -z "$line" ]] && continue

        # Skip special tokens if requested
        if [ "$SKIP_SPECIAL" = true ]; then
            [[ "$line" =~ ^\[.*\]$ ]] && continue
            [[ "$line" =~ ^\<.*\>$ ]] && continue
        fi

        # Escape single quotes for SQL
        escaped="${line//\'/\'\'}"

        # Add to batch
        batch="${batch}SELECT hypercube_ingest_text('${escaped}');"
        count=$((count + 1))
        total=$((total + 1))

        # Execute batch when full
        if [ $((count % BATCH_SIZE)) -eq 0 ]; then
            ingest_batch "$batch" >/dev/null
            echo -ne "\rIngested $total items..."
            batch=""
        fi
    done < "$input"

    # Execute remaining
    if [ -n "$batch" ]; then
        ingest_batch "$batch" >/dev/null
    fi

    echo -e "\rIngested $total items.     "
}

# Process files or stdin
START_TIME=$(date +%s)

if [ ${#FILES[@]} -eq 0 ] || [ "${FILES[0]}" = "-" ]; then
    # Read from stdin
    echo "Reading from stdin..."
    tmp=$(mktemp)
    cat > "$tmp"
    process_input "$tmp"
    rm -f "$tmp"
else
    # Process each file
    for file in "${FILES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "Warning: File not found: $file"
            continue
        fi
        echo "Processing: $file"
        process_input "$file"
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Get counts after
AFTER_COMPOSITIONS=$($PSQL_CMD -tAc "SELECT count(*) FROM relation")
AFTER_EDGES=$($PSQL_CMD -tAc "SELECT count(*) FROM relation_edge")

NEW_COMPOSITIONS=$((AFTER_COMPOSITIONS - BEFORE_COMPOSITIONS))
NEW_EDGES=$((AFTER_EDGES - BEFORE_EDGES))

echo ""
echo "=============================================="
echo "  Ingestion Complete"
echo "=============================================="
echo "Time: ${ELAPSED}s"
echo "New compositions: $NEW_COMPOSITIONS"
echo "New edges: $NEW_EDGES"
echo "Total compositions: $AFTER_COMPOSITIONS"
echo "Total edges: $AFTER_EDGES"

# Show statistics if requested
if [ "$SHOW_STATS" = true ]; then
    echo ""
    echo "=== Composition Statistics ==="
    $PSQL_CMD -c "
    SELECT
        depth,
        count(*) as compositions,
        sum(child_count) as edges,
        avg(atom_count)::numeric(10,1) as avg_atoms
    FROM relation
    GROUP BY depth
    ORDER BY depth;
    "
fi

echo ""
echo "Usage examples:"
echo "  # Find similar content"
echo "  psql -d $DB_NAME -c \"SELECT hypercube_similarity("
echo "    hypercube_ingest_text('hello'),"
echo "    hypercube_ingest_text('hallo'));\""
echo ""
