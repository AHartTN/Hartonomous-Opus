#!/bin/bash

# Windows-specific build functions for Hartonomous-Opus
# Called by the main build script when running on Windows (MSYS/MinGW/Cygwin)

source "$(dirname "${BASH_SOURCE[0]}")/../../shared/logging.sh"

# Pre-build setup for Windows
pre_build_setup() {
    log_info "Windows pre-build setup: configuring Visual Studio environment"

    # Find Visual Studio
    if command -v vswhere &> /dev/null; then
        VS_PATH=$(vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>/dev/null)
        if [ -n "$VS_PATH" ]; then
            log_info "Found Visual Studio at: $VS_PATH"

            # Try to run VsDevCmd.bat to set up environment
            if [ -f "$VS_PATH/Common7/Tools/VsDevCmd.bat" ]; then
                log_info "Setting up Visual Studio environment..."
                # Note: This would need to be run in a separate shell or the environment inherited
                log_warning "Visual Studio environment setup requires running in Developer Command Prompt"
            fi
        else
            log_warning "Visual Studio not found via vswhere"
        fi
    else
        log_warning "vswhere not found, cannot locate Visual Studio automatically"
    fi

    # Check for Intel MKL
    setup_intel_mkl

    log_success "Windows pre-build setup completed"
}

# Setup Intel MKL if available
setup_intel_mkl() {
    local mkl_found=false

    # Common Intel oneAPI installation paths
    local oneapi_paths=(
        "D:/Intel/oneAPI"
        "C:/Program Files (x86)/Intel/oneAPI"
        "C:/Program Files/Intel/oneAPI"
        "$HOME/intel/oneAPI"
    )

    for oneapi_path in "${oneapi_paths[@]}"; do
        if [ -d "$oneapi_path" ]; then
            local mkl_path="$oneapi_path/mkl/latest"
            if [ -d "$mkl_path" ]; then
                export MKLROOT="$mkl_path"
                local compiler_bin="$oneapi_path/compiler/latest/bin"
                local mkl_bin="$mkl_path/bin"

                if [ -d "$compiler_bin" ] && [ -d "$mkl_bin" ]; then
                    export PATH="$compiler_bin:$mkl_bin:$PATH"
                    log_success "Intel MKL configured from: $mkl_path"
                    mkl_found=true
                    break
                fi
            fi
        fi
    done

    if [ "$mkl_found" = false ]; then
        log_info "Intel MKL not found, building without MKL acceleration"
    fi
}

# Get platform-specific CMake arguments
get_cmake_args() {
    echo ""
}

# Install PostgreSQL extensions on Windows
install_extensions() {
    log_info "Installing PostgreSQL extensions on Windows"

    # Try to find PostgreSQL installation
    local pg_install_dir=""
    local pg_versions=("16" "15" "14" "13" "12")

    for version in "${pg_versions[@]}"; do
        local candidate="C:/Program Files/PostgreSQL/$version"
        if [ -d "$candidate" ]; then
            pg_install_dir="$candidate"
            break
        fi
    done

    if [ -z "$pg_install_dir" ]; then
        log_warning "PostgreSQL installation not found in standard locations"
        log_info "PostgreSQL extensions will be built but not installed automatically"
        log_info "To install manually, copy .dll files to PostgreSQL lib directory and .control/.sql files to share/extension"
        return 1
    fi

    local pg_lib_dir="$pg_install_dir/lib"
    local pg_share_dir="$pg_install_dir/share"
    local pg_ext_dir="$pg_share_dir/extension"

    log_info "Installing extensions to:"
    echo "  lib:         $pg_lib_dir"
    echo "  extension:   $pg_ext_dir"

    # Extension DLLs
    local extensions=(hypercube hypercube_c hypercube_ops embedding_ops embedding_c semantic_ops generative generative_c)

    for ext in "${extensions[@]}"; do
        local dll_file="$BIN_DIR/$ext.dll"
        if [ -f "$dll_file" ]; then
            cp "$dll_file" "$pg_lib_dir/"
            log_success "Installed DLL: $ext.dll"
        else
            log_warning "DLL not found: $dll_file"
        fi
    done

    # Control files
    local ctrl_files=(
        "$PROJECT_ROOT/cpp/hypercube.control"
        "$PROJECT_ROOT/cpp/sql/hypercube_ops.control"
        "$PROJECT_ROOT/cpp/sql/embedding_ops.control"
        "$PROJECT_ROOT/cpp/sql/semantic_ops.control"
        "$PROJECT_ROOT/cpp/sql/generative.control"
    )

    for ctrl in "${ctrl_files[@]}"; do
        if [ -f "$ctrl" ]; then
            cp "$ctrl" "$pg_ext_dir/"
            log_success "Installed control: $(basename $ctrl)"
        fi
    done

    # SQL files
    local sql_files=(
        "$PROJECT_ROOT/cpp/sql/hypercube--1.0.sql"
        "$PROJECT_ROOT/cpp/sql/hypercube_ops--1.0.sql"
        "$PROJECT_ROOT/cpp/sql/embedding_ops--1.0.sql"
        "$PROJECT_ROOT/cpp/sql/semantic_ops--1.0.sql"
        "$PROJECT_ROOT/cpp/sql/generative--1.0.sql"
    )

    for sql in "${sql_files[@]}"; do
        if [ -f "$sql" ]; then
            cp "$sql" "$pg_ext_dir/"
            log_success "Installed SQL: $(basename $sql)"
        fi
    done

    log_success "PostgreSQL extensions installed successfully"
    echo
    echo "Enable extensions with:"
    echo "  CREATE EXTENSION hypercube;"
    echo "  CREATE EXTENSION hypercube_ops;"
}