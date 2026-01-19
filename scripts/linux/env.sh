#!/bin/bash
# ============================================================================
# Hartonomous Hypercube - Linux Environment Setup
# ============================================================================
# Source this: source scripts/linux/env.sh
#
# This script:
#   1. Loads config.env settings
#   2. Sets up Intel oneAPI if available
#   3. Configures database connection helpers
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
# LOAD CONFIG.ENV
# ============================================================================

CONFIG_FILE="$PROJECT_ROOT/scripts/config.env"
if [ -f "$CONFIG_FILE" ]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ $line =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        # Skip lines without =
        [[ $line != *"="* ]] && continue
        export "$line"
    done < "$CONFIG_FILE"
fi

# ============================================================================
# VISUAL STUDIO DEVELOPER ENVIRONMENT (Windows Git Bash only)
# ============================================================================

# Only initialize VS environment on Windows (Git Bash/MSYS/MinGW)
if [[ "$MSYSTEM" ]] || [[ "$(uname -s)" == *"MSYS"* ]] || [[ "$(uname -s)" == *"MINGW"* ]]; then
    if [ -z "$VSCMD_VER" ]; then
        # Find Visual Studio installation
        vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
        if [ ! -f "$vswhere" ]; then
            vswhere="/c/Program Files/Microsoft Visual Studio/Installer/vswhere.exe"
        fi

        vs_path=""
        if [ -f "$vswhere" ]; then
            vs_path=$(cmd /c "$vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath" 2>/dev/null)
        fi

        # Fallback: check common locations
        if [ -z "$vs_path" ]; then
            search_paths=(
                "/d/Microsoft Visual Studio/18/Community"
                "/d/Microsoft Visual Studio/2022/Community"
                "/c/Program Files/Microsoft Visual Studio/2022/Community"
                "/c/Program Files/Microsoft Visual Studio/2022/Professional"
                "/c/Program Files/Microsoft Visual Studio/2022/Enterprise"
                "/c/Program Files (x86)/Microsoft Visual Studio/2019/Community"
                "/c/Program Files (x86)/Microsoft Visual Studio/2019/Professional"
                "/c/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise"
            )
            for path in "${search_paths[@]}"; do
                if [ -f "$path/Common7/Tools/VsDevCmd.bat" ]; then
                    vs_path="$path"
                    break
                fi
            done
        fi

        if [ -n "$vs_path" ]; then
            vs_dev_cmd_path="$vs_path/Common7/Tools/VsDevCmd.bat"
            if [ -f "$vs_dev_cmd_path" ]; then
                # Initialize VS environment by capturing and exporting variables
                if eval "$(cmd /c '"'"$vs_dev_cmd_path"'" -arch=amd64 -no_logo && set' | sed 's/^/export /')"; then
                    echo "Visual Studio environment initialized from: $vs_path"
                else
                    echo "Warning: Failed to initialize Visual Studio environment (C++ builds may fail)" >&2
                fi
            else
                echo "Visual Studio found but VsDevCmd.bat missing: $vs_path" >&2
            fi
        else
            echo "Visual Studio not found (optional for database-only operations)" >&2
        fi
    fi
fi

# ============================================================================
# INTEL oneAPI (if available)
# ============================================================================

INTEL_ONEAPI_PATHS=(
    "/opt/intel/oneapi"
    "$HOME/intel/oneapi"
    "/usr/local/intel/oneapi"
)

for oneapi_path in "${INTEL_ONEAPI_PATHS[@]}"; do
    if [ -d "$oneapi_path" ]; then
        # Source the oneAPI environment if setvars.sh exists
        if [ -f "$oneapi_path/setvars.sh" ]; then
            source "$oneapi_path/setvars.sh" --force > /dev/null 2>&1
        fi
        break
    fi
done

# Set MKL threading for oneAPI
export MKL_NUM_THREADS=8

# ============================================================================
# DEFAULTS
# ============================================================================

export HC_DB_HOST="${HC_DB_HOST:-hart-server}"
export HC_DB_PORT="${HC_DB_PORT:-5432}"
export HC_DB_USER="${HC_DB_USER:-postgres}"
export HC_DB_PASS="${HC_DB_PASS:-postgres}"
export HC_DB_NAME="${HC_DB_NAME:-hypercube}"
export HC_BUILD_TYPE="${HC_BUILD_TYPE:-Release}"
export HC_PARALLEL_JOBS="${HC_PARALLEL_JOBS:-$(nproc)}"
export HC_INGEST_THRESHOLD="${HC_INGEST_THRESHOLD:-0.5}"

export HC_PROJECT_ROOT="$PROJECT_ROOT"
export HC_BUILD_DIR="$PROJECT_ROOT/cpp/build"
export HC_BIN_DIR="$PROJECT_ROOT/cpp/build/bin/$HC_BUILD_TYPE"

# Validate critical paths
if [ ! -d "$HC_PROJECT_ROOT" ]; then
    echo "Error: HC_PROJECT_ROOT ($HC_PROJECT_ROOT) does not exist or is not accessible" >&2
    return 1
fi
if [ ! -d "$HC_BUILD_DIR" ]; then
    echo "Error: HC_BUILD_DIR ($HC_BUILD_DIR) does not exist or is not accessible" >&2
    return 1
fi

# Connection info for libpq
export HC_CONNINFO="host=$HC_DB_HOST port=$HC_DB_PORT dbname=$HC_DB_NAME user=$HC_DB_USER password=$HC_DB_PASS"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

hc_psql() {
    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d "$HC_DB_NAME" "$@"
}

hc_psql_admin() {
    PGPASSWORD="$HC_DB_PASS" psql -h "$HC_DB_HOST" -p "$HC_DB_PORT" -U "$HC_DB_USER" -d postgres "$@"
}

# Note: In Bash, functions must be exported with 'export -f' to be available in subshells.
# This differs from PowerShell, where functions are automatically inherited by child processes.
export -f hc_psql
export -f hc_psql_admin

# ============================================================================
# BANNER
# ============================================================================

if [ -z "$HC_ENV_LOADED" ]; then
    export HC_ENV_LOADED=1
    echo "Hartonomous environment loaded"
    echo "  Database: $HC_DB_NAME @ $HC_DB_HOST:$HC_DB_PORT"
    echo "  User:     $HC_DB_USER"
    echo "  Project:  $HC_PROJECT_ROOT"
fi
