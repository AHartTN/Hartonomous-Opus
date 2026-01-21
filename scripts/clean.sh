#!/bin/bash

# ============================================================================
# Hartonomous-Opus - Unified Cleanup Script
# ============================================================================
# Comprehensive cleanup of build artifacts, temporary files, logs, and caches
# across the entire project with safety checks and selective cleanup options.
#
# Usage:
#   ./scripts/clean.sh               # Safe cleanup (preserves important data)
#   ./scripts/clean.sh --all         # Full cleanup (removes everything)
#   ./scripts/clean.sh --build       # Clean build artifacts only
#   ./scripts/clean.sh --logs        # Clean logs only
#   ./scripts/clean.sh --cache       # Clean caches only
# ============================================================================

set -e

# Get script directory and load utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/shared/detect-platform.sh"
source "$SCRIPT_DIR/shared/logging.sh"

# Configuration variables
PLATFORM=$(detect_os)
CLEAN_BUILD=true
CLEAN_LOGS=true
CLEAN_CACHE=true
CLEAN_TEMP=true
CLEAN_DRY_RUN=false
FORCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all|-a)
            CLEAN_BUILD=true
            CLEAN_LOGS=true
            CLEAN_CACHE=true
            CLEAN_TEMP=true
            shift
            ;;
        --build|-b)
            CLEAN_BUILD=true
            CLEAN_LOGS=false
            CLEAN_CACHE=false
            CLEAN_TEMP=false
            shift
            ;;
        --logs|-l)
            CLEAN_BUILD=false
            CLEAN_LOGS=true
            CLEAN_CACHE=false
            CLEAN_TEMP=false
            shift
            ;;
        --cache|-c)
            CLEAN_BUILD=false
            CLEAN_LOGS=false
            CLEAN_CACHE=true
            CLEAN_TEMP=false
            shift
            ;;
        --temp|-t)
            CLEAN_BUILD=false
            CLEAN_LOGS=false
            CLEAN_CACHE=false
            CLEAN_TEMP=true
            shift
            ;;
        --dry-run|-n)
            CLEAN_DRY_RUN=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --all, -a       Clean everything (default)"
            echo "  --build, -b     Clean build artifacts only"
            echo "  --logs, -l      Clean logs only"
            echo "  --cache, -c     Clean caches only"
            echo "  --temp, -t      Clean temporary files only"
            echo "  --dry-run, -n   Show what would be cleaned without doing it"
            echo "  --force, -f     Skip safety confirmations"
            echo "  --help, -h      Show this help"
            echo
            echo "Examples:"
            echo "  $0              # Safe cleanup"
            echo "  $0 --all        # Full cleanup"
            echo "  $0 --build      # Build artifacts only"
            echo "  $0 --logs --cache  # Logs and caches only"
            echo
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_section "Hartonomous-Opus Cleanup ($PLATFORM)"

if [ "$CLEAN_DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No files will be deleted"
    echo
fi

echo "  Clean targets: $([ "$CLEAN_BUILD" = true ] && echo "build ")$([ "$CLEAN_LOGS" = true ] && echo "logs ")$([ "$CLEAN_CACHE" = true ] && echo "cache ")$([ "$CLEAN_TEMP" = true ] && echo "temp ")"
echo "  Dry run:       $CLEAN_DRY_RUN"
echo "  Force:          $FORCE"
echo

# Safety confirmation for full cleanup
if [ "$CLEAN_BUILD" = true ] && [ "$CLEAN_LOGS" = true ] && [ "$CLEAN_CACHE" = true ] && [ "$CLEAN_TEMP" = true ] && [ "$FORCE" = false ]; then
    echo "WARNING: This will remove all build artifacts, logs, caches, and temporary files."
    echo "         This action cannot be undone."
    echo
    read -p "Are you sure you want to proceed? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log_info "Cleanup cancelled by user"
        exit 0
    fi
    echo
fi

TOTAL_CLEANED=0

# Function to safely remove files/directories
safe_remove() {
    local target="$1"
    local description="$2"

    if [ ! -e "$target" ]; then
        return 0
    fi

    if [ "$CLEAN_DRY_RUN" = true ]; then
        if [ -d "$target" ]; then
            local size
            size=$(du -sh "$target" 2>/dev/null | cut -f1)
            echo "  Would remove directory: $target ($size)"
        else
            echo "  Would remove file: $target"
        fi
        return 0
    fi

    if [ -d "$target" ]; then
        local size
        size=$(du -sh "$target" 2>/dev/null | cut -f1)
        log_info "Removing directory: $target ($size)"
        rm -rf "$target"
        ((TOTAL_CLEANED++))
    elif [ -f "$target" ]; then
        log_info "Removing file: $target"
        rm -f "$target"
        ((TOTAL_CLEANED++))
    fi
}

# ============================================================================
# BUILD ARTIFACTS CLEANUP
# ============================================================================

if [ "$CLEAN_BUILD" = true ]; then
    log_subsection "Cleaning Build Artifacts"

    # C++ build directories
    safe_remove "$PROJECT_ROOT/cpp/build" "C++ build directory"
    safe_remove "$PROJECT_ROOT/cpp/build-pg-extensions" "PostgreSQL extensions build directory"

    # Dependency cache
    safe_remove "$PROJECT_ROOT/external/_deps" "CMake dependency cache"

    # Platform-specific build artifacts
    safe_remove "$PROJECT_ROOT/.vs" "Visual Studio cache"
    safe_remove "$PROJECT_ROOT/bin" "Binary output directory"
    safe_remove "$PROJECT_ROOT/lib" "Library output directory"

    # Temporary build files
    find "$PROJECT_ROOT" -name "*.tmp" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.obj" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.o" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pdb" -type f -exec rm -f {} \; 2>/dev/null || true
fi

# ============================================================================
# LOGS CLEANUP
# ============================================================================

if [ "$CLEAN_LOGS" = true ]; then
    log_subsection "Cleaning Logs"

    # Main logs directory
    safe_remove "$PROJECT_ROOT/logs" "Main logs directory"

    # Orchestrator logs
    safe_remove "$PROJECT_ROOT/Hartonomous-Orchestrator/logs" "Orchestrator logs"

    # Benchmark logs
    safe_remove "$PROJECT_ROOT/Hartonomous-Benchmark/results" "Benchmark results"
    safe_remove "$PROJECT_ROOT/Hartonomous-Benchmark/logs" "Benchmark logs"

    # Log files scattered around
    find "$PROJECT_ROOT" -name "*.log" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*_log.txt" -type f -exec rm -f {} \; 2>/dev/null || true

    # PID files
    safe_remove "$PROJECT_ROOT/orchestrator.pid" "Orchestrator PID file"
fi

# ============================================================================
# CACHE CLEANUP
# ============================================================================

if [ "$CLEAN_CACHE" = true ]; then
    log_subsection "Cleaning Caches"

    # Python cache
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pyc" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pyo" -type f -exec rm -f {} \; 2>/dev/null || true

    # Node.js cache (if any)
    safe_remove "$PROJECT_ROOT/node_modules/.cache" "Node.js cache"

    # IntelliSense/VS Code cache
    safe_remove "$PROJECT_ROOT/.vscode-cache" "VS Code cache"

    # CMake cache files
    find "$PROJECT_ROOT" -name "CMakeCache.txt" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true

    # Package manager caches
    safe_remove "$PROJECT_ROOT/.nuget" "NuGet cache"
    safe_remove "$PROJECT_ROOT/packages" "Package cache"
fi

# ============================================================================
# TEMPORARY FILES CLEANUP
# ============================================================================

if [ "$CLEAN_TEMP" = true ]; then
    log_subsection "Cleaning Temporary Files"

    # System temp files in project
    find "$PROJECT_ROOT" -name "*.tmp" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.temp" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*~" -type f -exec rm -f {} \; 2>/dev/null || true

    # Editor backup files
    find "$PROJECT_ROOT" -name "*.bak" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.swp" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.swo" -type f -exec rm -f {} \; 2>/dev/null || true

    # OS-specific temp files
    find "$PROJECT_ROOT" -name ".DS_Store" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "Thumbs.db" -type f -exec rm -f {} \; 2>/dev/null || true
    find "$PROJECT_ROOT" -name "desktop.ini" -type f -exec rm -f {} \; 2>/dev/null || true

    # Test artifacts
    safe_remove "$PROJECT_ROOT/test_output" "Test output directory"
    safe_remove "$PROJECT_ROOT/coverage" "Coverage reports"
    safe_remove "$PROJECT_ROOT/.coverage" "Coverage data"

    # Downloaded files and artifacts
    safe_remove "$PROJECT_ROOT/downloads" "Downloads directory"
    safe_remove "$PROJECT_ROOT/artifacts" "Build artifacts"
fi

# ============================================================================
# SUMMARY
# ============================================================================

log_subsection "Cleanup Summary"

if [ "$CLEAN_DRY_RUN" = true ]; then
    log_info "Dry run completed - no files were actually removed"
else
    if [ $TOTAL_CLEANED -gt 0 ]; then
        log_success "Cleanup completed - removed $TOTAL_CLEANED items"
    else
        log_info "Cleanup completed - no items found to remove"
    fi
fi

echo
echo "Cleanup targets processed:"
echo "  Build artifacts: $([ "$CLEAN_BUILD" = true ] && echo "✓" || echo "✗")"
echo "  Logs:            $([ "$CLEAN_LOGS" = true ] && echo "✓" || echo "✗")"
echo "  Caches:          $([ "$CLEAN_CACHE" = true ] && echo "✓" || echo "✗")"
echo "  Temp files:      $([ "$CLEAN_TEMP" = true ] && echo "✓" || echo "✗")"
echo

log_success "Cleanup operation completed successfully!"