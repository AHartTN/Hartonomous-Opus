#!/bin/bash
# Hartonomous Hypercube - Safetensor Model Ingestion (Linux)
# Usage: ./scripts/linux/ingest-safetensor.sh <model_directory> [-t threshold]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODEL_DIR=""
THRESHOLD="0.1"

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        *)
            MODEL_DIR="$1"
            shift
            ;;
    esac
done

if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <model_directory> [-t threshold]"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Not found: $MODEL_DIR"
    exit 1
fi

echo "=== Safetensor Model Ingestion ==="
echo "Model: $MODEL_DIR"
echo "Threshold: $THRESHOLD"

INGESTER="$HC_BIN_DIR/ingest_safetensor"
if [ ! -x "$INGESTER" ]; then
    echo "Safetensor ingester not found. Run build.sh first."
    exit 1
fi

# libpq uses PGPASSWORD env var for authentication
export PGPASSWORD="$HC_DB_PASS"

"$INGESTER" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -t "$THRESHOLD" "$MODEL_DIR"

unset PGPASSWORD

echo -e "\nModel ingestion complete"