#!/bin/bash

# Hartonomous Orchestrator Build Script
# This script handles building, testing, and packaging the application

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_TESTS="${ENABLE_TESTS:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check for required tools
    local required_tools=("cmake" "make" "g++")
    local missing_tools=()

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again."
        exit 1
    fi

    # Check for dependencies
    log_info "Checking for dependencies..."

    # Check for Boost
    if ! pkg-config --exists boost; then
        log_warning "Boost not found via pkg-config. Make sure Boost is installed."
    fi

    # Check for other dependencies
    local deps=("spdlog" "yaml-cpp" "prometheus-cpp")
    for dep in "${deps[@]}"; do
        if ! pkg-config --exists "$dep"; then
            log_warning "$dep not found. You may need to install it."
        fi
    done

    log_success "Requirements check completed"
}

# Setup build directory
setup_build_dir() {
    log_info "Setting up build directory..."

    if [ -d "$BUILD_DIR" ]; then
        log_info "Build directory exists, cleaning..."
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    log_success "Build directory ready"
}

# Configure with CMake
configure_build() {
    log_info "Configuring build with CMake..."
    log_info "Build type: $BUILD_TYPE"
    log_info "Tests enabled: $ENABLE_TESTS"

    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    )

    if [ "$ENABLE_TESTS" = "true" ]; then
        cmake_args+=("-DBUILD_TESTS=ON")
    else
        cmake_args+=("-DBUILD_TESTS=OFF")
    fi

    # Add any additional CMake arguments from environment
    if [ -n "$CMAKE_EXTRA_ARGS" ]; then
        IFS=' ' read -ra EXTRA_ARGS <<< "$CMAKE_EXTRA_ARGS"
        cmake_args+=("${EXTRA_ARGS[@]}")
    fi

    log_info "CMake command: cmake ${cmake_args[*]} $PROJECT_DIR"
    cmake "${cmake_args[@]}" "$PROJECT_DIR"

    log_success "CMake configuration completed"
}

# Build the project
build_project() {
    log_info "Building project..."

    local make_args=("-j$(nproc)")

    if [ -n "$VERBOSE_BUILD" ]; then
        make_args+=("VERBOSE=1")
    fi

    log_info "Make command: make ${make_args[*]}"
    make "${make_args[@]}"

    log_success "Build completed successfully"
}

# Run tests
run_tests() {
    if [ "$ENABLE_TESTS" != "true" ]; then
        log_info "Tests disabled, skipping..."
        return 0
    fi

    log_info "Running tests..."

    if [ ! -f "bin/hartonomous-tests" ]; then
        log_error "Test binary not found. Build failed?"
        return 1
    fi

    # Run tests
    if ctest --output-on-failure; then
        log_success "All tests passed"
    else
        log_error "Some tests failed"
        return 1
    fi
}

# Install the project
install_project() {
    log_info "Installing project..."

    local install_prefix="${INSTALL_PREFIX:-/usr/local}"

    if [ "$EUID" -eq 0 ]; then
        # Running as root, install system-wide
        make install
    else
        # Running as user, install to local directory
        log_info "Installing to $install_prefix (run with sudo for system-wide installation)"
        make install DESTDIR="$install_prefix"
    fi

    log_success "Installation completed"
}

# Create package
create_package() {
    log_info "Creating distribution package..."

    # Create version info
    local version="1.0.0"
    local package_name="hartonomous-orchestrator-${version}"
    local package_dir="$BUILD_DIR/$package_name"

    mkdir -p "$package_dir"

    # Copy built binaries
    cp -r bin "$package_dir/"
    cp -r lib "$package_dir/" 2>/dev/null || true

    # Copy configuration and documentation
    cp "$PROJECT_DIR/config.yaml" "$package_dir/"
    cp "$PROJECT_DIR/README.md" "$package_dir/"
    cp -r "$PROJECT_DIR/scripts" "$package_dir/"

    # Create archive
    cd "$BUILD_DIR"
    tar -czf "${package_name}.tar.gz" "$package_name"

    log_success "Package created: ${package_name}.tar.gz"
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."

    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        log_success "Build directory cleaned"
    else
        log_info "Build directory not found, nothing to clean"
    fi
}

# Show usage
usage() {
    cat << EOF
Hartonomous Orchestrator Build Script

USAGE:
    $0 [command] [options]

COMMANDS:
    all         Build everything (default)
    configure   Configure build with CMake
    build       Build the project
    test        Run tests
    install     Install the project
    package     Create distribution package
    clean       Clean build artifacts

OPTIONS:
    --build-type TYPE    Build type (Debug, Release, RelWithDebInfo) [default: Release]
    --no-tests           Disable building and running tests
    --verbose            Enable verbose build output
    --install-prefix DIR Install prefix for local installation
    --help, -h           Show this help message

ENVIRONMENT VARIABLES:
    BUILD_TYPE           Build type (same as --build-type)
    ENABLE_TESTS         Enable/disable tests (true/false)
    CMAKE_EXTRA_ARGS     Additional CMake arguments
    INSTALL_PREFIX       Install prefix
    VERBOSE_BUILD        Enable verbose make output (1)

EXAMPLES:
    $0                          # Build everything
    $0 --build-type Debug       # Debug build
    $0 --no-tests               # Build without tests
    $0 clean                    # Clean build
    $0 test                     # Run tests only

EOF
}

# Parse command line arguments
COMMAND="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --no-tests)
            ENABLE_TESTS="false"
            shift
            ;;
        --verbose)
            VERBOSE_BUILD="1"
            shift
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        configure|build|test|install|package|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute command
case $COMMAND in
    all)
        check_requirements
        setup_build_dir
        configure_build
        build_project
        run_tests
        ;;
    configure)
        check_requirements
        setup_build_dir
        configure_build
        ;;
    build)
        build_project
        ;;
    test)
        run_tests
        ;;
    install)
        install_project
        ;;
    package)
        create_package
        ;;
    clean)
        clean_build
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac

log_success "Build script completed successfully"