#!/bin/bash

# Requirement checking utilities for Hartonomous-Opus
# Validates system dependencies before running build/test/deploy operations

source "$(dirname "${BASH_SOURCE[0]}")/logging.sh"

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check CMake version
check_cmake() {
    local min_version="${1:-3.24}"

    if ! command_exists cmake; then
        log_error "CMake not found. Please install CMake $min_version or later."
        return 1
    fi

    local cmake_version
    cmake_version=$(cmake --version | head -n1 | cut -d' ' -f3)

    if ! printf '%s\n%s\n' "$min_version" "$cmake_version" | sort -V -C; then
        log_error "CMake version $cmake_version found, but $min_version or later is required."
        return 1
    fi

    log_success "CMake $cmake_version found"
    return 0
}

# Check compiler
check_compiler() {
    local platform
    platform=$(source "$(dirname "${BASH_SOURCE[0]}")/detect-platform.sh" && detect_os)

    case "$platform" in
        linux|macos)
            if command_exists clang++; then
                log_success "Clang++ found"
                return 0
            elif command_exists g++; then
                log_success "G++ found"
                return 0
            else
                log_error "No C++ compiler found. Please install clang++ or g++."
                return 1
            fi
            ;;
        windows)
            if command_exists cl; then
                log_success "MSVC cl found"
                return 0
            else
                log_error "MSVC compiler not found. Please run from Developer Command Prompt."
                return 1
            fi
            ;;
        *)
            log_error "Unsupported platform for compiler check"
            return 1
            ;;
    esac
}

# Check Python
check_python() {
    local min_version="${1:-3.9}"

    if ! command_exists python3 && ! command_exists python; then
        log_error "Python not found. Please install Python $min_version or later."
        return 1
    fi

    local python_cmd
    if command_exists python3; then
        python_cmd="python3"
    else
        python_cmd="python"
    fi

    local python_version
    python_version=$($python_cmd --version 2>&1 | cut -d' ' -f2)

    if ! printf '%s\n%s\n' "$min_version" "$python_version" | sort -V -C; then
        log_error "Python version $python_version found, but $min_version or later is required."
        return 1
    fi

    log_success "Python $python_version found"
    return 0
}

# Check PostgreSQL (optional)
check_postgresql() {
    if command_exists psql; then
        local pg_version
        pg_version=$(psql --version | head -n1 | awk '{print $3}' | cut -d'.' -f1-2)
        log_success "PostgreSQL client $pg_version found"
        return 0
    else
        log_warning "PostgreSQL client not found. Database operations will be limited."
        return 1
    fi
}

# Check Git
check_git() {
    if command_exists git; then
        local git_version
        git_version=$(git --version | cut -d' ' -f3)
        log_success "Git $git_version found"
        return 0
    else
        log_error "Git not found. Please install Git."
        return 1
    fi
}

# Run all checks
check_all_requirements() {
    local failed=0

    log_section "Checking System Requirements"

    check_cmake || ((failed++))
    check_compiler || ((failed++))
    check_python || ((failed++))
    check_git || ((failed++))
    check_postgresql  # Warning only, not a failure

    echo

    if [ $failed -gt 0 ]; then
        log_error "$failed requirement(s) not met. Please install missing dependencies."
        return 1
    else
        log_success "All core requirements satisfied"
        return 0
    fi
}

# Export functions
export -f command_exists
export -f check_cmake
export -f check_compiler
export -f check_python
export -f check_postgresql
export -f check_git
export -f check_all_requirements