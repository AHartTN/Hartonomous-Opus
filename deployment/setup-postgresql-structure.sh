#!/bin/bash

# Detect PostgreSQL version
if ! command -v pg_config &> /dev/null; then
    echo "Error: pg_config not found. Install PostgreSQL development package."
    exit 1
fi

PG_VERSION=$(pg_config --version | awk '{print $2}')
echo "Detected PostgreSQL version: $PG_VERSION"

BASE_DIR="/opt/libraries/postgresql"
VERSION_DIR="$BASE_DIR/$PG_VERSION"

# Create versioned directory structure
DIRS=("bin" "extensions" "lib" "config" "logs" "backup")

# Create base directory
if [ ! -d "$BASE_DIR" ]; then
    sudo mkdir -p "$BASE_DIR"
    echo "Created $BASE_DIR"
else
    echo "$BASE_DIR already exists"
fi

# Create version-specific directories
for dir in "${DIRS[@]}"; do
    full_dir="$VERSION_DIR/$dir"
    if [ ! -d "$full_dir" ]; then
        sudo mkdir -p "$full_dir"
        echo "Created $full_dir"
    else
        echo "$full_dir already exists"
    fi
done

# Set ownership to root:root and permissions to 755 for deployment staging
sudo chown -R root:root "$BASE_DIR"
sudo chmod -R 755 "$BASE_DIR"

echo "Versioned directory structure created and permissions set for deployment staging."