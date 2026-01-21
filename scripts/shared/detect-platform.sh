#!/bin/bash

# Platform detection utilities for Hartonomous-Opus scripts
# Provides consistent platform identification across all build/test/deploy scripts

# Detect operating system
detect_os() {
    local os
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    case "$os" in
        linux*)
            echo "linux"
            ;;
        darwin*)
            echo "macos"
            ;;
        msys*|mingw*|cygwin*)
            echo "windows"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Detect architecture
detect_arch() {
    local arch
    arch=$(uname -m | tr '[:upper:]' '[:lower:]')
    case "$arch" in
        x86_64|amd64)
            echo "x64"
            ;;
        i386|i686)
            echo "x86"
            ;;
        arm64|aarch64)
            echo "arm64"
            ;;
        arm*)
            echo "arm"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Get platform directory name
get_platform_dir() {
    local os arch
    os=$(detect_os)
    arch=$(detect_arch)

    if [ "$os" = "windows" ]; then
        echo "windows"
    elif [ "$os" = "linux" ]; then
        echo "linux"
    elif [ "$os" = "macos" ]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Check if running on Windows (MSYS, MinGW, Cygwin)
is_windows() {
    case "$(uname -s)" in
        MSYS*|MINGW*|CYGWIN*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Check if running on Linux
is_linux() {
    [ "$(uname -s)" = "Linux" ]
}

# Check if running on macOS
is_macos() {
    [ "$(uname -s)" = "Darwin" ]
}

# Get script directory (works with symlinks)
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    while [ -h "$source" ]; do
        local dir
        dir="$(cd -P "$(dirname "$source")" && pwd)"
        source="$(readlink "$source")"
        [[ $source != /* ]] && source="$dir/$source"
    done
    cd -P "$(dirname "$source")" && pwd
}

# Export functions for use in other scripts
export -f detect_os
export -f detect_arch
export -f get_platform_dir
export -f is_windows
export -f is_linux
export -f is_macos
export -f get_script_dir