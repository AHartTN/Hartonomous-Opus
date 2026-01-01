#!/bin/bash
# Complete Model Ingestion Pipeline
#
# Ingests an AI model's vocabulary as compositions and extracts semantic edges
# from the embedding matrix (sparse representation above threshold).
#
# Usage:
#   ./ingest_model.sh [options] <model_directory>
#
# Options:
#   -d, --dbname NAME       Database name (default: hypercube)
#   -U, --user USER         Database user (default: hartonomous)
#   -h, --host HOST         Database host (default: localhost)
#   -t, --threshold FLOAT   Similarity threshold for sparse edges (default: 0.5)
#   --skip-vocab            Skip vocabulary ingestion
#   --skip-embeddings       Skip embedding extraction
#   --rebuild               Rebuild C++ tools before running
#   --help                  Show this help
#
# Environment:
#   PGPASSWORD              Database password
#
# Example:
#   PGPASSWORD=secret ./ingest_model.sh -t 0.6 ./models/all-MiniLM-L6-v2/

set -e

# Defaults
DB_NAME="hypercube"
DB_USER="${PGUSER:-hartonomous}"
DB_HOST="${PGHOST:-localhost}"
THRESHOLD="0.5"
SKIP_VOCAB=false
SKIP_EMBEDDINGS=false
REBUILD=false
MODEL_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dbname) DB_NAME="$2"; shift 2 ;;
        -U|--user) DB_USER="$2"; shift 2 ;;
        -h|--host) DB_HOST="$2"; shift 2 ;;
        -t|--threshold) THRESHOLD="$2"; shift 2 ;;
        --skip-vocab) SKIP_VOCAB=true; shift ;;
        --skip-embeddings) SKIP_EMBEDDINGS=true; shift ;;
        --rebuild) REBUILD=true; shift ;;
        --help)
            head -30 "$0" | tail -26
            exit 0
            ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) MODEL_DIR="$1"; shift ;;
    esac
done

if [ -z "$MODEL_DIR" ]; then
    echo "Error: Model directory required"
    echo "Usage: $0 [options] <model_directory>"
    exit 1
fi

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/cpp/build"

# Find model files
find_model_file() {
    local pattern="$1"
    find "$MODEL_DIR" -name "$pattern" -type f 2>/dev/null | head -1
}

VOCAB_FILE=$(find_model_file "vocab.txt")
TOKENIZER_FILE=$(find_model_file "tokenizer.json")
SAFETENSOR_FILE=$(find_model_file "*.safetensors")
CONFIG_FILE=$(find_model_file "config.json")

# Extract model name from directory
MODEL_NAME=$(basename "$MODEL_DIR")

echo "=============================================="
echo "  Hartonomous Model Ingestion"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Directory: $MODEL_DIR"
echo "Database: $DB_NAME ($DB_USER@$DB_HOST)"
echo "Threshold: $THRESHOLD"
echo ""
echo "Files found:"
echo "  Vocab: ${VOCAB_FILE:-not found}"
echo "  Tokenizer: ${TOKENIZER_FILE:-not found}"
echo "  Safetensor: ${SAFETENSOR_FILE:-not found}"
echo "  Config: ${CONFIG_FILE:-not found}"
echo ""

# Build tools if needed
if [ "$REBUILD" = true ] || [ ! -f "$BUILD_DIR/extract_embeddings" ]; then
    echo "=== Building C++ Tools ==="
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake ..
    make -j$(nproc) extract_embeddings ingest_model
    cd - > /dev/null
    echo ""
fi

# Database connection
export PGPASSWORD="${PGPASSWORD:-}"
PSQL_CMD="psql -U $DB_USER -h $DB_HOST -d $DB_NAME -q"

# Verify database connection
if ! $PSQL_CMD -c "SELECT 1" >/dev/null 2>&1; then
    echo "Error: Cannot connect to database"
    echo "Check PGPASSWORD and connection parameters"
    exit 1
fi

# Get before counts
BEFORE_COMPOSITIONS=$($PSQL_CMD -tAc "SELECT count(*) FROM relation" 2>/dev/null || echo 0)
BEFORE_EDGES=$($PSQL_CMD -tAc "SELECT count(*) FROM relation_edge" 2>/dev/null || echo 0)

START_TIME=$(date +%s)

# Step 1: Ingest vocabulary as compositions
if [ "$SKIP_VOCAB" != true ] && [ -n "$VOCAB_FILE" ]; then
    echo "=== Step 1: Ingesting Vocabulary ==="
    echo "Source: $VOCAB_FILE"

    # Count tokens
    TOKEN_COUNT=$(wc -l < "$VOCAB_FILE")
    echo "Tokens: $TOKEN_COUNT"

    # Ingest using the content ingestion script
    "$SCRIPT_DIR/ingest_content.sh" \
        -d "$DB_NAME" \
        -U "$DB_USER" \
        -h "$DB_HOST" \
        --vocab \
        --skip-special \
        "$VOCAB_FILE"

    echo ""
elif [ "$SKIP_VOCAB" != true ] && [ -n "$TOKENIZER_FILE" ]; then
    echo "=== Step 1: Ingesting Vocabulary from tokenizer.json ==="
    echo "Source: $TOKENIZER_FILE"

    # Use C++ ingest_model tool for tokenizer.json
    "$BUILD_DIR/ingest_model" \
        --tokenizer "$TOKENIZER_FILE" \
        --name "$MODEL_NAME" \
        -d "$DB_NAME" \
        -U "$DB_USER" \
        -h "$DB_HOST" || echo "  (C++ tokenizer ingestion not yet implemented)"

    echo ""
else
    echo "=== Step 1: Skipping Vocabulary (not found or --skip-vocab) ==="
    echo ""
fi

# Step 2: Extract embeddings and create sparse edges
if [ "$SKIP_EMBEDDINGS" != true ] && [ -n "$SAFETENSOR_FILE" ]; then
    echo "=== Step 2: Extracting Semantic Edges ==="
    echo "Source: $SAFETENSOR_FILE"
    echo "Threshold: $THRESHOLD"

    "$BUILD_DIR/extract_embeddings" \
        --model "$SAFETENSOR_FILE" \
        --threshold "$THRESHOLD" \
        -d "$DB_NAME" \
        -U "$DB_USER" \
        -h "$DB_HOST"

    echo ""
else
    echo "=== Step 2: Skipping Embeddings (not found or --skip-embeddings) ==="
    echo ""
fi

# Step 3: Store model metadata
if [ -n "$CONFIG_FILE" ]; then
    echo "=== Step 3: Storing Model Metadata ==="

    # Read config and store as JSON
    CONFIG_JSON=$(cat "$CONFIG_FILE" | tr -d '\n' | sed "s/'/''/g")

    $PSQL_CMD -c "
    INSERT INTO model (id, name, model_type, vocab_size, config, source_path)
    VALUES (
        hypercube_blake3(convert_to('$MODEL_NAME', 'UTF8')),
        '$MODEL_NAME',
        'embedding',
        (SELECT count(*) FROM relation WHERE depth = 2),
        '${CONFIG_JSON}'::jsonb,
        '$MODEL_DIR'
    )
    ON CONFLICT (name, model_type) DO UPDATE SET
        config = EXCLUDED.config,
        source_path = EXCLUDED.source_path;
    " 2>/dev/null || echo "  (metadata table not available)"
    echo ""
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Get after counts
AFTER_COMPOSITIONS=$($PSQL_CMD -tAc "SELECT count(*) FROM relation" 2>/dev/null || echo 0)
AFTER_EDGES=$($PSQL_CMD -tAc "SELECT count(*) FROM relation_edge" 2>/dev/null || echo 0)
SEMANTIC_EDGES=$($PSQL_CMD -tAc "SELECT count(*) FROM semantic_edge WHERE model_name LIKE '%${MODEL_NAME}%'" 2>/dev/null || echo 0)

NEW_COMPOSITIONS=$((AFTER_COMPOSITIONS - BEFORE_COMPOSITIONS))
NEW_EDGES=$((AFTER_EDGES - BEFORE_EDGES))

echo "=============================================="
echo "  Ingestion Complete"
echo "=============================================="
echo "Time: ${ELAPSED}s"
echo ""
echo "Compositions:"
echo "  Before: $BEFORE_COMPOSITIONS"
echo "  After: $AFTER_COMPOSITIONS"
echo "  New: $NEW_COMPOSITIONS"
echo ""
echo "Merkle Edges:"
echo "  Before: $BEFORE_EDGES"
echo "  After: $AFTER_EDGES"
echo "  New: $NEW_EDGES"
echo ""
echo "Semantic Edges (sparse, >= $THRESHOLD): $SEMANTIC_EDGES"
echo ""
echo "Query examples:"
echo "  # Find similar tokens"
echo "  psql -d $DB_NAME -c \"SELECT * FROM hypercube_find_similar('computer', 10);\""
echo ""
echo "  # Check semantic edges"
echo "  psql -d $DB_NAME -c \"SELECT * FROM semantic_edge WHERE weight > 0.8 LIMIT 10;\""
echo ""
