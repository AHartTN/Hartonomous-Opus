#!/bin/bash

# Linux-specific build functions for Hartonomous-Opus
# Called by the main build script when running on Linux

source "$(dirname "${BASH_SOURCE[0]}")/../../shared/logging.sh"

# Pre-build setup for Linux
pre_build_setup() {
    log_info "Linux pre-build setup: checking PostgreSQL development headers"

    if ! command -v pg_config &> /dev/null; then
        log_warning "pg_config not found. PostgreSQL extensions will not be built."
        log_info "Install postgresql-server-dev-all or equivalent package"
    else
        log_success "PostgreSQL development headers found"
    fi
}

# Get platform-specific CMake arguments
get_cmake_args() {
    echo ""
}

# Install PostgreSQL extensions on Linux
install_extensions() {
    if ! command -v pg_config &> /dev/null; then
        log_error "pg_config not found. Cannot install PostgreSQL extensions."
        log_info "Install postgresql-server-dev-all package"
        return 1
    fi

    PG_LIB_DIR=$(pg_config --pkglibdir)
    PG_SHARE_DIR=$(pg_config --sharedir)
    PG_EXT_DIR="$PG_SHARE_DIR/extension"

    log_info "Installing extensions to:"
    echo "  pkglibdir:  $PG_LIB_DIR"
    echo "  extension:  $PG_EXT_DIR"

    # Use sudo if we don't have write access
    SUDO=""
    if [ ! -w "$PG_LIB_DIR" ]; then
        SUDO="sudo"
        log_info "Using sudo for installation (no write access to PostgreSQL directories)"
    fi

    # Extension shared libraries are in lib/ directory
    LIB_DIR="$BUILD_DIR/lib"

    EXTENSIONS=(db_wrapper_pg hypercube hypercube_c hypercube_ops embedding_ops embedding_c semantic_ops generative generative_c)

    for ext in "${EXTENSIONS[@]}"; do
        so_file="$LIB_DIR/$ext.so"
        if [ -f "$so_file" ]; then
            $SUDO cp "$so_file" "$PG_LIB_DIR/"
            log_success "Installed library: $ext.so"
        else
            log_warning "Library not found: $so_file"
        fi
    done

    # PostgreSQL may not load libraries from symlinks or user directories
    # Copy all dependencies directly to PostgreSQL lib directory
    ALL_LIBS=(db_wrapper_pg hypercube hypercube_c hypercube_ops embedding_ops embedding_c semantic_ops generative generative_c)
    for lib in "${ALL_LIBS[@]}"; do
        lib_file="$LIB_DIR/$lib.so"
        if [ -f "$lib_file" ]; then
            $SUDO cp "$lib_file" "$PG_LIB_DIR/"
            log_success "Installed dependency: $lib.so"
        fi
    done

    # Control files
    CTRL_FILES=(
        "$PROJECT_ROOT/cpp/hypercube.control"
        "$PROJECT_ROOT/cpp/sql/hypercube_ops.control"
        "$PROJECT_ROOT/cpp/sql/embedding_ops.control"
        "$PROJECT_ROOT/cpp/sql/semantic_ops.control"
        "$PROJECT_ROOT/cpp/sql/generative.control"
    )

    for ctrl in "${CTRL_FILES[@]}"; do
        if [ -f "$ctrl" ]; then
            $SUDO cp "$ctrl" "$PG_EXT_DIR/"
            log_success "Installed control: $(basename $ctrl)"
        fi
    done

    # SQL files
    SQL_FILES=(
        "$PROJECT_ROOT/cpp/sql/hypercube--1.0.sql"
        "$PROJECT_ROOT/cpp/sql/hypercube_ops--1.0.sql"
        "$PROJECT_ROOT/cpp/sql/embedding_ops--1.0.sql"
        "$PROJECT_ROOT/cpp/sql/semantic_ops--1.0.sql"
        "$PROJECT_ROOT/cpp/sql/generative--1.0.sql"
    )

    for sql in "${SQL_FILES[@]}"; do
        if [ -f "$sql" ]; then
            $SUDO cp "$sql" "$PG_EXT_DIR/"
            log_success "Installed SQL: $(basename $sql)"
        fi
    done

    log_success "PostgreSQL extensions installed successfully"
    echo
    echo "Enable extensions with:"
    echo "  CREATE EXTENSION hypercube;"
    echo "  CREATE EXTENSION hypercube_ops;"
}