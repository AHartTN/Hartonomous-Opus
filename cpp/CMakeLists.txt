cmake_minimum_required(VERSION 3.24)

# ============================================================================
# Project definition
# ============================================================================

project(
    hypercube
    VERSION 1.0.0
    LANGUAGES CXX C
)

# ============================================================================
# Module path wiring
# ============================================================================

# All modular build logic lives under cpp/cmake/
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ============================================================================
# FetchContent caching - persist downloads across builds
# ============================================================================

# Store FetchContent downloads in a persistent directory outside the build tree.
# This prevents re-downloading dependencies on every clean build.
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/_deps" CACHE PATH "FetchContent cache directory")
message(STATUS "FetchContent cache: ${FETCHCONTENT_BASE_DIR}")

# ============================================================================
# Global options
# ============================================================================

# Option to enable/disable PostgreSQL extensions explicitly.
# NOTE: FindDependencies.cmake will still auto-detect pg_config/libpq and
#       respect this switch (we'll wire that in when we refactor that file).
option(HYPERCUBE_ENABLE_PG_EXTENSIONS "Build PostgreSQL extensions if possible" ON)

# Option to build tools/CLI that depend on PostgreSQL client libraries.
option(HYPERCUBE_ENABLE_TOOLS "Build CLI tools and ingesters" ON)

# Option to build tests (both gtest-based and standalone tests).
option(HYPERCUBE_ENABLE_TESTS "Build test targets" ON)

# ============================================================================
# Core configuration (compiler, CPU features, dependencies)
# ============================================================================

# 1) Compiler flags, language standards, SIMD/AVX bundles, output layout, RPATH
include(CompilerFlags)

# 2) Third-party dependencies (Eigen, MKL, OpenMP, Threads, PostgreSQL, HNSW, GTest)
include(FindDependencies)

# ============================================================================
# Targets
# ============================================================================

# 3) Core libraries (C++ core + C-bridge shared libraries)
include(TargetsCore)

# 4) PostgreSQL extensions (if enabled and dependencies present)
if(HYPERCUBE_ENABLE_PG_EXTENSIONS AND BUILD_PG_EXTENSION)
    include(TargetsPgExtensions)
else()
    message(STATUS "PostgreSQL extensions are disabled or pg_config not found; skipping extension targets")
endif()

# 5) CLI tools and ingesters (if enabled and PostgreSQL client is available)
if(HYPERCUBE_ENABLE_TOOLS AND PostgreSQL_FOUND)
    include(TargetsTools)
else()
    message(STATUS "CLI tools/ingesters are disabled or PostgreSQL client not found; skipping tool targets")
endif()

# 6) Test targets (gtest + standalone tests)
if(HYPERCUBE_ENABLE_TESTS)
    include(CTest)  # sets BUILD_TESTING, ctest integration
    include(TargetsTests)
else()
    message(STATUS "Tests are disabled; skipping test targets")
endif()

# 7) Installation configuration (extensions, tools, optional library exports)
include(InstallConfig)

# ============================================================================
# Runtime environment guidance
# ============================================================================

message(STATUS "")
message(STATUS "Runtime Environment Setup for AVX Optimization:")
if(WIN32)
    message(STATUS "  Configure these in your Windows user or system environment:")
    message(STATUS "    OMP_PROC_BIND=true")
    message(STATUS "    OMP_PLACES=cores")
    message(STATUS "    MKL_DEBUG_CPU_TYPE=5  # Force AVX2 code path")
else()
    message(STATUS "  Add these to your shell profile (~/.bashrc, ~/.zshrc, etc.):")
    message(STATUS "    export OMP_PROC_BIND=true")
    message(STATUS "    export OMP_PLACES=cores")
    message(STATUS "    export MKL_DEBUG_CPU_TYPE=5  # Force AVX2 code path")
endif()
message(STATUS "  This improves thread scheduling and MKL code-path selection.")
message(STATUS "")

# ============================================================================
# Configuration summary
# ============================================================================

# These variables are populated by CompilerFlags.cmake, CpuFeatures.cmake,
# and FindDependencies.cmake (which we will refactor file-by-file next).

message(STATUS "")
message(STATUS "=== Hypercube Build Configuration ===")

# Build / compiler
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build configurations:    ${CMAKE_CONFIGURATION_TYPES}")
else()
    message(STATUS "Build type:              ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "C++ compiler:            ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler:              ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "C++ standard:            ${CMAKE_CXX_STANDARD}")

# Dependencies
message(STATUS "Eigen3 present:          ${HAS_EIGEN}")
message(STATUS "Intel MKL present:       ${HAS_MKL}")
message(STATUS "OpenMP present:          ${HAS_OPENMP}")
message(STATUS "HNSWLib present:         ${HAS_HNSWLIB}")
message(STATUS "PostgreSQL extensions:   ${BUILD_PG_EXTENSION}")
message(STATUS "PostgreSQL client libs:  ${PostgreSQL_FOUND}")
message(STATUS "Google Test present:     ${GTest_FOUND}")

# SIMD / CPU features (from CpuFeatures.cmake via CompilerFlags.cmake)
message(STATUS "SIMD baseline HAS_AVX:   ${HAS_AVX}")
message(STATUS "HAS_AVX2:                ${HAS_AVX2}")
message(STATUS "HAS_AVX512F:             ${HAS_AVX512F}")
message(STATUS "HAS_FMA3:                ${HAS_FMA3}")
message(STATUS "HAS_AVX_VNNI:            ${HAS_AVX_VNNI}")
message(STATUS "")

message(STATUS "Hypercube configuration complete.")
message(STATUS "")
