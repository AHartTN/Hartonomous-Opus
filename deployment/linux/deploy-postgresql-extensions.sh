#!/bin/bash
# PostgreSQL Extensions Deployment Script for Linux
# Builds extensions directly on target system and installs with symlinks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAGING_BASE="/opt/libraries/postgresql"
EXTENSIONS=("hypercube" "generative" "hypercube_ops" "embedding_ops" "semantic_ops")
DB_WRAPPER="db_wrapper_pg"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Validation functions
validate_environment() {
    log_info "Validating environment..."

    # Check PostgreSQL
    if ! command -v pg_config &> /dev/null; then
        log_error "pg_config not found. Install PostgreSQL development packages."
        return 1
    fi

    local pg_version=$(pg_config --version | awk '{print $2}')
    local major_version=$(echo "$pg_version" | cut -d. -f1)
    if [[ $major_version -lt 12 ]]; then
        log_error "PostgreSQL $pg_version is too old. Requires 12+"
        return 1
    fi

    log_success "PostgreSQL $pg_version validated"

    # Check build tools
    for tool in cmake make gcc g++; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool not found. Install build tools."
            return 1
        fi
    done
    log_success "Build tools validated"

    # Check repository
    if [[ ! -f "$REPO_ROOT/cpp/CMakeLists.txt" ]]; then
        log_error "Repository not found at $REPO_ROOT"
        return 1
    fi
    log_success "Repository structure validated"
}

build_extensions() {
    log_info "Building PostgreSQL extensions..."

    cd "$REPO_ROOT/cpp"

    # Clean previous build
    local build_dir="build-pg-extensions"
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    cd "$build_dir"

    # Configure
    log_info "Configuring build with CMake..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DHYPERCUBE_ENABLE_PG_EXTENSIONS=ON \
          -DHYPERCUBE_ENABLE_TOOLS=OFF \
          -DHYPERCUBE_ENABLE_TESTS=OFF \
          ..

    # Build wrapper first
    log_info "Building database wrapper..."
    make -j$(nproc) "$DB_WRAPPER"

    # Build extensions
    log_info "Building extensions..."
    make -j$(nproc) "${EXTENSIONS[@]}"

    log_success "Build completed successfully"
}

stage_artifacts() {
    log_info "Staging artifacts to $STAGING_BASE..."

    # Create versioned directory
    local pg_version=$(pg_config --version | awk '{print $2}')
    local staging_dir="$STAGING_BASE/$pg_version"
    sudo mkdir -p "$staging_dir"

    local build_dir="$REPO_ROOT/cpp/build-pg-extensions"

    # Copy built libraries
    if [[ -d "$build_dir/bin" ]]; then
        sudo cp "$build_dir/bin/"*.so "$staging_dir/"
    fi

    # Copy control and SQL files
    sudo cp "$REPO_ROOT/cpp/sql/"*.control "$staging_dir/" 2>/dev/null || true
    sudo cp "$REPO_ROOT/cpp/sql/"*--*.sql "$staging_dir/" 2>/dev/null || true

    # Create metadata
    local metadata="$staging_dir/metadata.json"
    cat << EOF | sudo tee "$metadata" > /dev/null
{
    "version": "$pg_version",
    "built_at": "$(date -Iseconds)",
    "extensions": $(printf '%s\n' "${EXTENSIONS[@]}" | jq -R . | jq -s .),
    "db_wrapper": "$DB_WRAPPER",
    "architecture": "unified_wrapper"
}
EOF

    log_success "Artifacts staged in $staging_dir"
    echo "$staging_dir"
}

install_extensions() {
    local staging_dir="$1"

    log_info "Installing extensions with symlinks..."

    # PostgreSQL directories
    local pg_libdir=$(pg_config --pkglibdir)
    local pg_sharedir=$(pg_config --sharedir)
    local pg_extdir="$pg_sharedir/extension"

    # Create backup
    local backup_dir="$STAGING_BASE/backup/$(date +%Y%m%d_%H%M%S)"
    sudo mkdir -p "$backup_dir"
    log_info "Backups stored in $backup_dir"

    # Backup existing files
    for ext in "${EXTENSIONS[@]}"; do
        sudo cp "$pg_libdir/$ext.so" "$backup_dir/" 2>/dev/null || true
        sudo cp "$pg_extdir/$ext.control" "$backup_dir/" 2>/dev/null || true
        sudo find "$pg_extdir" -name "$ext--*.sql" -exec sudo cp {} "$backup_dir/" \; 2>/dev/null || true
    done
    sudo cp "$pg_libdir/$DB_WRAPPER.so" "$backup_dir/" 2>/dev/null || true

    # Install symlinks
    log_info "Creating symlinks..."

    # Libraries
    for lib in "$staging_dir/"*.so; do
        if [[ -f "$lib" ]]; then
            sudo ln -sf "$lib" "$pg_libdir/$(basename "$lib")"
        fi
    done

    # Control files
    for control in "$staging_dir/"*.control; do
        if [[ -f "$control" ]]; then
            sudo ln -sf "$control" "$pg_extdir/$(basename "$control")"
        fi
    done

    # SQL files
    for sql in "$staging_dir/"*--*.sql; do
        if [[ -f "$sql" ]]; then
            sudo ln -sf "$sql" "$pg_extdir/$(basename "$sql")"
        fi
    done

    log_success "Symlinks created successfully"
}

validate_installation() {
    local staging_dir="$1"

    log_info "Validating installation..."

    local pg_libdir=$(pg_config --pkglibdir)
    local pg_extdir=$(pg_config --sharedir)/extension

    # Check symlinks
    local failed=false

    for lib in "$staging_dir/"*.so; do
        if [[ -f "$lib" ]]; then
            local name=$(basename "$lib")
            if [[ ! -L "$pg_libdir/$name" ]]; then
                log_error "$name symlink missing"
                failed=true
            elif [[ "$(readlink "$pg_libdir/$name")" != "$lib" ]]; then
                log_error "$name symlink points to wrong target"
                failed=true
            fi
        fi
    done

    for control in "$staging_dir/"*.control; do
        if [[ -f "$control" ]]; then
            local name=$(basename "$control")
            if [[ ! -L "$pg_extdir/$name" ]]; then
                log_error "$name symlink missing"
                failed=true
            elif [[ "$(readlink "$pg_extdir/$name")" != "$control" ]]; then
                log_error "$name symlink points to wrong target"
                failed=true
            fi
        fi
    done

    for sql in "$staging_dir/"*--*.sql; do
        if [[ -f "$sql" ]]; then
            local name=$(basename "$sql")
            if [[ ! -L "$pg_extdir/$name" ]]; then
                log_error "$name symlink missing"
                failed=true
            elif [[ "$(readlink "$pg_extdir/$name")" != "$sql" ]]; then
                log_error "$name symlink points to wrong target"
                failed=true
            fi
        fi
    done

    if [[ "$failed" == "true" ]]; then
        return 1
    fi

    log_success "Installation validation passed"
}

rollback_installation() {
    log_warn "Rolling back installation..."

    local pg_libdir=$(pg_config --pkglibdir)
    local pg_extdir=$(pg_config --sharedir)/extension
    local backup_dir="$STAGING_BASE/backup/$(ls -t "$STAGING_BASE/backup" | head -1)"

    if [[ -d "$backup_dir" ]]; then
        log_info "Restoring from $backup_dir"

        # Remove symlinks
        for ext in "${EXTENSIONS[@]}"; do
            sudo rm -f "$pg_libdir/$ext.so"
            sudo rm -f "$pg_extdir/$ext.control"
            sudo find "$pg_extdir" -name "$ext--*.sql" -exec sudo rm -f {} \;
        done
        sudo rm -f "$pg_libdir/$DB_WRAPPER.so"

        # Restore backups
        sudo cp "$backup_dir/"*.so "$pg_libdir/" 2>/dev/null || true
        sudo cp "$backup_dir/"*.control "$pg_extdir/" 2>/dev/null || true
        sudo cp "$backup_dir/"*--*.sql "$pg_extdir/" 2>/dev/null || true

        log_success "Rollback completed"
    else
        log_error "No backup found for rollback"
    fi
}

cleanup_old_backups() {
    log_info "Cleaning up old backups..."

    # Keep only last 5 backups
    local backup_count=$(ls -d "$STAGING_BASE/backup"/* 2>/dev/null | wc -l)
    if [[ $backup_count -gt 5 ]]; then
        ls -dt "$STAGING_BASE/backup"/* | tail -n +6 | xargs sudo rm -rf
        log_success "Old backups cleaned up"
    fi
}

main() {
    log_info "Starting PostgreSQL extensions deployment..."

    # Validate environment
    if ! validate_environment; then
        exit 1
    fi

    # Build extensions
    if ! build_extensions; then
        log_error "Build failed"
        exit 1
    fi

    # Stage artifacts
    local staging_dir
    if ! staging_dir=$(stage_artifacts); then
        log_error "Staging failed"
        exit 1
    fi

    # Set trap for rollback on error
    trap rollback_installation ERR

    # Install extensions
    if ! install_extensions "$staging_dir"; then
        log_error "Installation failed - rolling back"
        exit 1
    fi

    # Validate installation
    if ! validate_installation "$staging_dir"; then
        log_error "Validation failed - rolling back"
        rollback_installation
        exit 1
    fi

    # Remove trap
    trap - ERR

    # Cleanup old backups
    cleanup_old_backups

    log_success "Deployment completed successfully!"
    echo ""
    echo "To enable extensions in PostgreSQL:"
    echo ""
    echo "  sudo -u postgres psql hypercube << SQL"
    echo "  CREATE EXTENSION IF NOT EXISTS hypercube CASCADE;"
    echo "  CREATE EXTENSION IF NOT EXISTS hypercube_ops;"
    echo "  CREATE EXTENSION IF NOT EXISTS embedding_ops;"
    echo "  CREATE EXTENSION IF NOT EXISTS generative;"
    echo "  CREATE EXTENSION IF NOT EXISTS semantic_ops;"
    echo ""
    echo "  -- Test function"
    echo "  SELECT hc_map_codepoint(65);"
    echo "  SQL"
}

# Run with error handling
if main "$@"; then
    log_success "Script completed successfully"
else
    log_error "Script failed"
    exit 1
fi