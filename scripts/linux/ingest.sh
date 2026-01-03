#!/bin/bash
# Hartonomous Hypercube - Content Ingestion (Linux)
# Usage: ./scripts/linux/ingest.sh <path> [--binary]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

if [ -z "$1" ]; then
    echo "Usage: $0 <path> [--binary]"
    exit 1
fi

TARGET="$1"
BINARY_FLAG=""
if [ "$2" == "--binary" ]; then
    BINARY_FLAG="--binary"
fi

if [ ! -e "$TARGET" ]; then
    echo "Not found: $TARGET"
    exit 1
fi

echo "=== Hypercube Content Ingestion ==="
echo "Target: $TARGET"

# Prefer vocabulary_ingest (proper grammar-based) over cpe_ingest (binary pairing)
INGESTER=""
if [ -x "$HC_BUILD_DIR/vocabulary_ingest" ]; then
    INGESTER="$HC_BUILD_DIR/vocabulary_ingest"
    echo "Using: vocabulary_ingest (grammar-based)"
elif [ -x "$HC_BUILD_DIR/cpe_ingest" ]; then
    INGESTER="$HC_BUILD_DIR/cpe_ingest"
    echo "Using: cpe_ingest (binary pairing fallback)"
else
    echo "Ingester not found. Run build.sh first."
    exit 1
fi

"$INGESTER" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" -p "$HC_DB_PORT" $BINARY_FLAG "$TARGET"

echo -e "\nIngestion complete"
