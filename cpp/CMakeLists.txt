cmake_minimum_required(VERSION 3.24)
project(hypercube VERSION 1.0.0 LANGUAGES CXX C)

# ============================================================================
# Modular Build System
# ============================================================================

# Include all modular components
include(cmake/FindDependencies.cmake)
include(cmake/CompilerFlags.cmake)
include(cmake/BuildTargets.cmake)
include(cmake/InstallConfig.cmake)

# ============================================================================
# Runtime Environment Optimization for AVX/AVX_VNNI
# ============================================================================

message(STATUS "Runtime Environment Setup for AVX Optimization:")
message(STATUS "  Set these environment variables in your shell profile (.bashrc, .zshrc) or Windows environment variables:")
if(WIN32)
    message(STATUS "  set OMP_PROC_BIND=true")
    message(STATUS "  set OMP_PLACES=cores")
    message(STATUS "  set MKL_DEBUG_CPU_TYPE=5  # Force AVX2 code path")
else()
    message(STATUS "  export OMP_PROC_BIND=true")
    message(STATUS "  export OMP_PLACES=cores")
    message(STATUS "  export MKL_DEBUG_CPU_TYPE=5  # Force AVX2 code path")
endif()
message(STATUS "")
message(STATUS "  This ensures optimal thread scheduling and MKL path selection.")
message(STATUS "")

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== Hypercube Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "PostgreSQL extension: ${BUILD_PG_EXTENSION}")
if(PostgreSQL_FOUND)
    message(STATUS "PostgreSQL tools: ON")
else()
    message(STATUS "PostgreSQL tools: OFF (libpq not found)")
endif()
message(STATUS "Eigen3: ON")
if(HAS_MKL)
    message(STATUS "Intel MKL: ON")
else()
    message(STATUS "Intel MKL: ERROR - MKL required but not found!")
endif()
if(HAS_OPENMP)
    message(STATUS "OpenMP: ON")
else()
    message(STATUS "OpenMP: OFF")
endif()
if(HAS_HNSWLIB)
    message(STATUS "k-NN: Approximate (HNSWLib enabled)")
else()
    message(STATUS "k-NN: ERROR - HNSWLib required but not found!")
endif()
if(GTest_FOUND)
    message(STATUS "Tests: ON (Google Test)")
else()
    message(STATUS "Tests: OFF")
endif()
message(STATUS "")
