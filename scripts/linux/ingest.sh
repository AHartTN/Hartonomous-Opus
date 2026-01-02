#!/bin/bash
# Hartonomous Hypercube - Content Ingestion (Linux)
# Usage: ./scripts/linux/ingest.sh <path>

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

if [ -z "$1" ]; then
    echo "Usage: $0 <path>"
    exit 1
fi

TARGET="$1"
if [ ! -e "$TARGET" ]; then
    echo "Not found: $TARGET"
    exit 1
fi

echo "=== Hypercube Content Ingestion ==="
echo "Target: $TARGET"

INGESTER="$HC_BUILD_DIR/cpe_ingest"
if [ ! -x "$INGESTER" ]; then
    echo "Ingester not found. Run build.sh first."
    exit 1
fi

"$INGESTER" -d "$HC_DB_NAME" -U "$HC_DB_USER" -h "$HC_DB_HOST" "$TARGET"

echo -e "\nIngestion complete"
