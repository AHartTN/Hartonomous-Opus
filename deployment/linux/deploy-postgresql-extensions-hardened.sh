#!/bin/bash
# Hardened deployment script for PostgreSQL extensions on Linux
# Integrates with package-pg-dev-from-server.sh and build-extensions-with-pg-dev.sh

set -euo pipefail

# Constants
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAGING_DIR="/opt/libraries/postgresql"
EXTENSIONS=("hypercube" "generative" "hypercube_ops" "embedding_ops" "semantic_ops")
# Database wrapper library - deployed alongside extensions
DB_WRAPPER="db_wrapper_pg"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Hardened PostgreSQL extensions deployment script for Linux.

OPTIONS:
    -d, --dry-run              Show what would be done without making changes
    -v, --verbose              Enable verbose output
    -h, --help                 Show this help message
    --skip-build              Skip the build phase (assumes extensions are already built)
    --staging-dir DIR         Override staging directory (default: $STAGING_DIR)
    --pg-dev-package DIR      Path to PostgreSQL dev package directory

EXAMPLES:
    # Dry run to see what would happen
    $0 --dry-run

    # Deploy with custom staging directory
    $0 --staging-dir /tmp/postgresql-staging

    # Skip build if extensions are already built
    $0 --skip-build

EOF
}

# Parse command line arguments
DRY_RUN=false
VERBOSE=false
SKIP_BUILD=false
STAGING_DIR="/opt/libraries/postgresql"
PG_DEV_PACKAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --staging-dir)
            STAGING_DIR="$2"
            shift 2
            ;;
        --pg-dev-package)
            PG_DEV_PACKAGE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Enable verbose output if requested
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Dry run banner
if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "DRY RUN MODE - No changes will be made"
    echo
fi

# Version compatibility check
check_version_compatibility() {
    log_info "Checking PostgreSQL version compatibility..."

    if ! command -v pg_config &> /dev/null; then
        log_error "pg_config not found. Install PostgreSQL development package."
        return 1
    fi

    local current_version=$(pg_config --version | awk '{print $2}')
    local major_version=$(echo "$current_version" | cut -d. -f1)
    local minor_version=$(echo "$current_version" | cut -d. -f2)

    log_info "Current PostgreSQL version: $current_version"

    # Check if version is supported (PostgreSQL 12+ for modern extensions)
    if [[ $major_version -lt 12 ]]; then
        log_error "PostgreSQL version $current_version is too old. Requires PostgreSQL 12+"
        return 1
    fi

    log_success "PostgreSQL version $current_version is compatible"
    return 0
}

# Find PostgreSQL dev package
find_pg_dev_package() {
    if [[ -n "$PG_DEV_PACKAGE" && -d "$PG_DEV_PACKAGE" ]]; then
        echo "$PG_DEV_PACKAGE"
        return 0
    fi

    # Check common locations
    for candidate in \
        "$SCRIPT_DIR/pg-dev-package" \
        "$REPO_ROOT/pg-dev-package" \
        "$HOME/pg-dev-package"
    do
        if [[ -f "$candidate/pg-config.env" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

# Build extensions
build_extensions() {
    local pg_dev_dir="$1"

    log_info "Building PostgreSQL extensions..."

    # Source PostgreSQL environment
    if [[ ! -f "$pg_dev_dir/pg-config.env" ]]; then
        log_error "PostgreSQL environment file not found: $pg_dev_dir/pg-config.env"
        return 1
    fi

    # Load environment in subshell to avoid affecting parent
    (
        source "$pg_dev_dir/pg-config.env" || exit 1

        if ! command -v pg_config &> /dev/null; then
            log_error "pg_config not available after sourcing environment"
            exit 1
        fi

        local pg_version=$(pg_config --version)
        log_info "Using PostgreSQL dev package version: $pg_version"

        # Change to C++ directory
        cd "$REPO_ROOT/cpp" || exit 1

        # Create build directory
        local build_dir="build-pg-extensions"
        rm -rf "$build_dir"
        mkdir -p "$build_dir"
        cd "$build_dir" || exit 1

        # Configure with CMake
        log_info "Configuring build with CMake..."
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "Would run: cmake -DCMAKE_BUILD_TYPE=Release -DHYPERCUBE_ENABLE_PG_EXTENSIONS=ON -DHYPERCUBE_ENABLE_TOOLS=OFF -DHYPERCUBE_ENABLE_TESTS=OFF .."
        else
            cmake -DCMAKE_BUILD_TYPE=Release \
                  -DHYPERCUBE_ENABLE_PG_EXTENSIONS=ON \
                  -DHYPERCUBE_ENABLE_TOOLS=OFF \
                  -DHYPERCUBE_ENABLE_TESTS=OFF \
                  .. || exit 1
        fi

        # Build database wrapper library first (links to PostgreSQL libraries)
        log_info "Building database wrapper library..."
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "Would run: make -j$(nproc) $DB_WRAPPER"
        else
            make -j$(nproc) "$DB_WRAPPER" || exit 1
        fi

        # Build extensions (depend on wrapper but don't link to PostgreSQL directly)
        log_info "Building extensions..."
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "Would run: make -j$(nproc) ${EXTENSIONS[*]}"
        else
            make -j$(nproc) "${EXTENSIONS[@]}" || exit 1
        fi

        log_success "Extensions built successfully"
    )
}

# Stage artifacts
stage_artifacts() {
    log_info "Staging artifacts in $STAGING_DIR..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Would create directory: $STAGING_DIR"
        echo "Would copy built artifacts to staging area"
    else
        # Create staging directory
        sudo mkdir -p "$STAGING_DIR" || return 1

        # Create version directory
        local version_dir="$STAGING_DIR/$(pg_config --version | awk '{print $2}')"
        sudo mkdir -p "$version_dir" || return 1

        # Copy built extensions
        local build_dir="$REPO_ROOT/cpp/build-pg-extensions"
        if [[ -d "$build_dir/bin" ]]; then
            sudo cp "$build_dir/bin/"*.so "$version_dir/" 2>/dev/null || true
        fi

        # Copy control and SQL files
        if [[ -d "$REPO_ROOT/sql" ]]; then
            sudo cp "$REPO_ROOT/sql/"*.control "$version_dir/" 2>/dev/null || true
            sudo cp "$REPO_ROOT/sql/"*--*.sql "$version_dir/" 2>/dev/null || true
        fi

        # Create metadata file
        local metadata_file="$version_dir/metadata.json"
        cat << EOF | sudo tee "$metadata_file" > /dev/null
{
    "version": "$(pg_config --version | awk '{print $2}')",
    "built_at": "$(date -Iseconds)",
    "extensions": $(printf '%s\n' "${EXTENSIONS[@]}" | jq -R . | jq -s .),
    "db_wrapper": "$DB_WRAPPER",
    "architecture": "unified_wrapper",
    "staging_dir": "$STAGING_DIR"
}
EOF

        log_success "Artifacts staged in $version_dir"
        echo "$version_dir"
    fi
}

# Install extensions
install_extensions() {
    local staging_path="$1"

    log_info "Installing extensions from staging..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Would create symlinks from staging area to PostgreSQL directories"
        echo "Would create backup of existing files in staging area"
        return 0
    fi

    # Get PostgreSQL directories
    local pg_libdir=$(pg_config --pkglibdir)
    local pg_sharedir=$(pg_config --sharedir)
    local pg_extdir="$pg_sharedir/extension"

    # Create backup directory in staging area
    local backup_dir="$STAGING_DIR/backup/$(date +%Y%m%d_%H%M%S)"
    sudo mkdir -p "$backup_dir"

    # Function to rollback on error
    rollback_install() {
        log_warn "Rolling back installation..."
        # Remove symlinks and restore backed up files
        if [[ -d "$backup_dir" ]]; then
            # Remove symlinks
            for ext in "${EXTENSIONS[@]}"; do
                sudo rm -f "$pg_libdir/$ext.so"
                sudo rm -f "$pg_extdir/$ext.control"
                sudo find "$pg_extdir" -name "$ext--*.sql" -exec sudo rm -f {} \;
            done
            sudo rm -f "$pg_libdir/$DB_WRAPPER.so"

            # Restore backed up files
            sudo cp "$backup_dir/"*.so "$pg_libdir/" 2>/dev/null || true
            sudo cp "$backup_dir/"*.control "$pg_extdir/" 2>/dev/null || true
            sudo cp "$backup_dir/"*--*.sql "$pg_extdir/" 2>/dev/null || true
        fi
    }

    # Set trap for rollback
    trap rollback_install ERR

    # Backup existing files
    log_info "Creating backups in $backup_dir..."
    for ext in "${EXTENSIONS[@]}"; do
        sudo cp "$pg_libdir/$ext.so" "$backup_dir/" 2>/dev/null || true
        sudo cp "$pg_extdir/$ext.control" "$backup_dir/" 2>/dev/null || true
        sudo find "$pg_extdir" -name "$ext--*.sql" -exec sudo cp {} "$backup_dir/" \; 2>/dev/null || true
    done
    # Backup database wrapper library
    sudo cp "$pg_libdir/$DB_WRAPPER.so" "$backup_dir/" 2>/dev/null || true

    # Create symlinks for shared libraries
    log_info "Creating symlinks for shared libraries..."
    for lib in "$staging_path/"*.so; do
        if [[ -f "$lib" ]]; then
            sudo ln -sf "$lib" "$pg_libdir/$(basename "$lib")"
        fi
    done

    # Create symlinks for control files
    log_info "Creating symlinks for control files..."
    for control in "$staging_path/"*.control; do
        if [[ -f "$control" ]]; then
            sudo ln -sf "$control" "$pg_extdir/$(basename "$control")"
        fi
    done

    # Create symlinks for SQL files
    log_info "Creating symlinks for SQL files..."
    for sql in "$staging_path/"*--*.sql; do
        if [[ -f "$sql" ]]; then
            sudo ln -sf "$sql" "$pg_extdir/$(basename "$sql")"
        fi
    done

    # Verify installation
    log_info "Verifying installation..."

    # Function to verify symlink
    verify_symlink() {
        local target="$1"
        local link="$2"
        if [[ -L "$link" ]]; then
            local actual_target=$(readlink "$link")
            if [[ "$actual_target" == "$target" ]]; then
                return 0
            else
                log_error "Symlink $link points to $actual_target, expected $target"
                return 1
            fi
        else
            log_error "$link is not a symlink"
            return 1
        fi
    }

    # Verify database wrapper library
    if verify_symlink "$staging_path/$DB_WRAPPER.so" "$pg_libdir/$DB_WRAPPER.so"; then
        log_success "Database wrapper $DB_WRAPPER symlink verified"
    else
        log_error "Database wrapper $DB_WRAPPER symlink verification failed"
        return 1
    fi

    # Verify extensions
    for ext in "${EXTENSIONS[@]}"; do
        if verify_symlink "$staging_path/$ext.so" "$pg_libdir/$ext.so"; then
            log_success "Extension $ext symlink verified"
        else
            log_error "Extension $ext symlink verification failed"
            return 1
        fi

        # Verify control file symlink
        if [[ -f "$staging_path/$ext.control" ]]; then
            if verify_symlink "$staging_path/$ext.control" "$pg_extdir/$ext.control"; then
                log_success "Control file $ext.control symlink verified"
            else
                log_error "Control file $ext.control symlink verification failed"
                return 1
            fi
        fi

        # Verify SQL files symlinks
        for sql in "$staging_path/"$ext--*.sql; do
            if [[ -f "$sql" ]]; then
                if verify_symlink "$sql" "$pg_extdir/$(basename "$sql")"; then
                    log_success "SQL file $(basename "$sql") symlink verified"
                else
                    log_error "SQL file $(basename "$sql") symlink verification failed"
                    return 1
                fi
            fi
        done
    done

    # Clean up backup
    rm -rf "$backup_dir"

    # Remove trap
    trap - ERR

    log_success "All extensions installed successfully"
}

# Main execution
main() {
    log_info "Starting PostgreSQL extensions deployment..."

    # Check version compatibility
    if ! check_version_compatibility; then
        exit 1
    fi

    # Find PostgreSQL dev package
    local pg_dev_dir
    if ! pg_dev_dir=$(find_pg_dev_package); then
        log_error "PostgreSQL development package not found."
        log_error "Run package-pg-dev-from-server.sh on the server first."
        exit 1
    fi

    log_info "Found PostgreSQL dev package: $pg_dev_dir"

    # Build extensions unless skipped
    if [[ "$SKIP_BUILD" != "true" ]]; then
        if ! build_extensions "$pg_dev_dir"; then
            log_error "Failed to build extensions"
            exit 1
        fi
    else
        log_info "Skipping build phase as requested"
    fi

    # Stage artifacts
    local staging_path
    if ! staging_path=$(stage_artifacts); then
        log_error "Failed to stage artifacts"
        exit 1
    fi

    # Install extensions
    if ! install_extensions "$staging_path"; then
        log_error "Failed to install extensions"
        exit 1
    fi

    log_success "PostgreSQL extensions deployment completed successfully!"

    # Print usage instructions
    cat << EOF

To enable the extensions in your database:

    sudo -u postgres psql hypercube << SQL
    CREATE EXTENSION IF NOT EXISTS hypercube;
    CREATE EXTENSION IF NOT EXISTS hypercube_ops;
    CREATE EXTENSION IF NOT EXISTS embedding_ops;
    CREATE EXTENSION IF NOT EXISTS generative;
    CREATE EXTENSION IF NOT EXISTS semantic_ops;

    -- Test
    SELECT hc_map_codepoint(65);
    SQL

EOF
}

# Run main function
main "$@"