# ============================================================================
# FindDependencies.cmake
#
# Comprehensive dependency detection for hypercube.
# - Conservative: prefer imported CMake targets when available
# - Robust: try pkg-config, vcpkg-style locations, common system paths
# - Fallbacks: FetchContent for libraries we can build (HNSW, GTest)
# - Windows-aware: common Program Files and vcpkg locations probed
#
# Exports canonical variables for the rest of the build:
#   HAS_THREADS, HAS_OPENMP, HAS_EIGEN, HAS_MKL, HAS_HNSWLIB,
#   PostgreSQL_FOUND, BUILD_PG_EXTENSION, PostgreSQL_INCLUDE_DIRS,
#   PostgreSQL_LIBRARIES, POSTGRES_LIB, GTest_FOUND, GTEST_INCLUDE_DIRS
#
# This file performs detection only. It does not link targets or set flags.
# ============================================================================

include(FetchContent)
include(CheckCXXCompilerFlag)
include(CheckCXXSymbolExists)
include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------
# Threads (required)
# ----------------------------------------------------------------------------
find_package(Threads REQUIRED)
if(Threads_FOUND)
    set(HAS_THREADS ON)
    message(STATUS "[deps] Threads: found")

    # On non-Windows platforms, check for pthread_create as additional validation
    # Note: Windows uses native threads (via Threads::Threads), not pthreads
    if(NOT WIN32)
        set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} Threads::Threads)
        check_cxx_symbol_exists(pthread_create "pthread.h" COMPILER_SUPPORTS_PTHREAD_CREATE)
        if(NOT COMPILER_SUPPORTS_PTHREAD_CREATE)
            message(WARNING "[deps] pthread_create not found, but Threads package is available")
        endif()
    endif()
else()
    set(HAS_THREADS OFF)
    message(FATAL_ERROR "[deps] Threads: not found (required)")
endif()

# ----------------------------------------------------------------------------
# OpenMP (optional)
# ----------------------------------------------------------------------------
set(HAS_OPENMP OFF)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    set(HAS_OPENMP ON)
    message(STATUS "[deps] OpenMP: found (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "[deps] OpenMP: not found")
endif()

# ----------------------------------------------------------------------------
# Eigen3 detection (multiple strategies)
# ----------------------------------------------------------------------------
set(HAS_EIGEN OFF)
set(Eigen3_INCLUDE_DIRS "")

# 1) Prefer CMake imported target
find_package(Eigen3 QUIET NO_MODULE)
if(TARGET Eigen3::Eigen)
    set(HAS_EIGEN ON)
    message(STATUS "[deps] Eigen3: found (imported target Eigen3::Eigen)")
else()
    # 2) Try pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(EIGEN3_PKG QUIET eigen3)
        if(EIGEN3_PKG_FOUND)
            set(HAS_EIGEN ON)
            set(Eigen3_INCLUDE_DIRS ${EIGEN3_PKG_INCLUDE_DIRS})
            message(STATUS "[deps] Eigen3: found via pkg-config (${Eigen3_INCLUDE_DIRS})")
        endif()
    endif()
endif()

# 3) Manual search common paths
if(NOT HAS_EIGEN)
    find_path(Eigen3_INCLUDE_DIR
        NAMES Eigen/Core
        PATHS
            /usr/include/eigen3
            /usr/local/include/eigen3
            $ENV{CONDA_PREFIX}/include/eigen3
            $ENV{HOME}/.local/include/eigen3
            C:/vcpkg/installed/x64-windows/include/eigen3
            "C:/Program Files/eigen3/include"
        NO_DEFAULT_PATH
    )
    if(Eigen3_INCLUDE_DIR)
        set(HAS_EIGEN ON)
        set(Eigen3_INCLUDE_DIRS ${Eigen3_INCLUDE_DIR})
        message(STATUS "[deps] Eigen3: found at ${Eigen3_INCLUDE_DIRS}")
    endif()
endif()

if(NOT HAS_EIGEN)
    message(WARNING "[deps] Eigen3 not found. Some features will be disabled or slower.")
endif()

# ----------------------------------------------------------------------------
# Intel MKL detection (multi-strategy)
# ----------------------------------------------------------------------------
set(HAS_MKL OFF)
set(MKL_INCLUDE_DIRS "")
set(MKL_LIBRARIES "")

# 1) Prefer CMake config package (vcpkg / oneAPI)
find_package(MKL QUIET)
if(MKL_FOUND)
    set(HAS_MKL ON)
    message(STATUS "[deps] MKL: found via CMake package (MKL::MKL)")
endif()

# 2) Try Intel oneAPI environment variable MKLROOT
if(NOT HAS_MKL)
    if(DEFINED ENV{MKLROOT})
        set(_MKLROOT $ENV{MKLROOT})
        find_path(_MKL_INCLUDE mkl.h PATHS "${_MKLROOT}/include" "${_MKLROOT}/include/mkl" NO_DEFAULT_PATH)
        find_library(_MKL_CORE mkl_core PATHS "${_MKLROOT}/lib/intel64" "${_MKLROOT}/lib" NO_DEFAULT_PATH)
        find_library(_MKL_INTEL mkl_intel_lp64 PATHS "${_MKLROOT}/lib/intel64" "${_MKLROOT}/lib" NO_DEFAULT_PATH)
        find_library(_MKL_SEQUENTIAL mkl_sequential PATHS "${_MKLROOT}/lib/intel64" "${_MKLROOT}/lib" NO_DEFAULT_PATH)
        if(_MKL_INCLUDE AND _MKL_CORE)
            set(HAS_MKL ON)
            set(MKL_INCLUDE_DIRS ${_MKL_INCLUDE})
            set(MKL_LIBRARIES ${_MKL_SEQUENTIAL} ${_MKL_INTEL} ${_MKL_CORE})
            message(STATUS "[deps] MKL: found via MKLROOT (${_MKLROOT})")
        endif()
    endif()
endif()

# 3) Try common system locations (Linux/macOS/Windows)
if(NOT HAS_MKL)
    if(WIN32)
        set(_MKL_SEARCH
            "C:/Program Files/Intel/oneAPI/mkl/latest"
            "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl"
            "D:/Intel/oneAPI/mkl/latest"
        )
    else()
        set(_MKL_SEARCH
            "/opt/intel/oneapi/mkl/latest"
            "/opt/intel/mkl"
            "/usr/local/intel/mkl"
        )
    endif()

    foreach(_p IN LISTS _MKL_SEARCH)
        find_path(_MKL_INCLUDE mkl.h PATHS "${_p}/include" "${_p}/include/mkl" NO_DEFAULT_PATH)
        find_library(_MKL_CORE mkl_core PATHS "${_p}/lib/intel64" "${_p}/lib" NO_DEFAULT_PATH)
        if(_MKL_INCLUDE AND _MKL_CORE)
            set(HAS_MKL ON)
            set(MKL_INCLUDE_DIRS ${_MKL_INCLUDE})
            set(MKL_LIBRARIES ${_MKL_CORE})
            message(STATUS "[deps] MKL: found at ${_p}")
            break()
        endif()
    endforeach()
endif()

# 4) Final status
if(HAS_MKL)
    message(STATUS "[deps] MKL: enabled")
else()
    message(STATUS "[deps] MKL: not found (will fall back to Eigen or custom implementations)")
endif()

# ----------------------------------------------------------------------------
# HNSWLib detection / FetchContent fallback
# ----------------------------------------------------------------------------
set(HAS_HNSWLIB OFF)
set(HNSWLIB_INCLUDE_DIRS "")
set(HNSWLIB_LIBRARIES "")

# Prefer imported target
find_package(hnswlib QUIET)
if(hnswlib_FOUND)
    set(HAS_HNSWLIB ON)
    message(STATUS "[deps] HNSWLib: found via package")
else()
    # Try pkg-config
    if(PkgConfig_FOUND)
        pkg_check_modules(HNSW_PKG QUIET hnswlib)
        if(HNSW_PKG_FOUND)
            set(HAS_HNSWLIB ON)
            set(HNSWLIB_INCLUDE_DIRS ${HNSW_PKG_INCLUDE_DIRS})
            set(HNSWLIB_LIBRARIES ${HNSW_PKG_LIBRARIES})
            message(STATUS "[deps] HNSWLib: found via pkg-config")
        endif()
    endif()
endif()

# If still not found, fetch source and build as part of the tree
if(NOT HAS_HNSWLIB)
    message(STATUS "[deps] HNSWLib: not found system-wide, will fetch from source")
    FetchContent_Declare(
        hnswlib_src
        GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
        GIT_TAG v0.8.0
    )
    # Use FetchContent_MakeAvailable instead of Populate (CMP0169)
    FetchContent_MakeAvailable(hnswlib_src)

    # hnswlib is header-only in many usages; mark as available
    set(HAS_HNSWLIB ON)
    set(HNSWLIB_FOUND ON)
    set(HNSWLIB_INCLUDE_DIRS ${hnswlib_src_SOURCE_DIR})
    message(STATUS "[deps] HNSWLib: fetched into ${hnswlib_src_SOURCE_DIR}")
endif()

# ----------------------------------------------------------------------------
# PostgreSQL detection (pg_config, libpq, postgres.lib)
# ----------------------------------------------------------------------------
set(BUILD_PG_EXTENSION OFF)
set(PostgreSQL_FOUND OFF)
set(PostgreSQL_INCLUDE_DIRS "")
set(PostgreSQL_LIBRARIES "")
set(POSTGRES_LIB "")

# Try to find pg_config first (best for extension development)
if(WIN32)
    # Search common Windows install locations for pg_config
    find_program(PG_CONFIG pg_config
        HINTS
            "$ENV{ProgramFiles}/PostgreSQL/*/bin"
            "C:/Program Files/PostgreSQL/*/bin"
            "C:/Program Files (x86)/PostgreSQL/*/bin"
    )
else()
    find_program(PG_CONFIG pg_config)
endif()

if(PG_CONFIG)
    execute_process(COMMAND ${PG_CONFIG} --includedir-server OUTPUT_VARIABLE PG_INCLUDEDIR_SERVER OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --includedir OUTPUT_VARIABLE PG_INCLUDEDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --pkglibdir OUTPUT_VARIABLE PG_PKGLIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --sharedir OUTPUT_VARIABLE PG_SHAREDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --libdir OUTPUT_VARIABLE PG_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --version OUTPUT_VARIABLE PG_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(BUILD_PG_EXTENSION ON)
    message(STATUS "[deps] pg_config found: ${PG_CONFIG} (PostgreSQL ${PG_VERSION})")
    message(STATUS "[deps] PostgreSQL server include: ${PG_INCLUDEDIR_SERVER}")
    message(STATUS "[deps] PostgreSQL pkglibdir: ${PG_PKGLIBDIR}")
    message(STATUS "[deps] PostgreSQL libdir: ${PG_LIBDIR}")
else()
    message(STATUS "[deps] pg_config not found; PostgreSQL extension builds will be disabled unless overridden")
endif()

# Find libpq (client library) for tools
find_package(PostgreSQL QUIET)
if(PostgreSQL_FOUND)
    set(PostgreSQL_FOUND ON)
    set(PostgreSQL_INCLUDE_DIRS ${PostgreSQL_INCLUDE_DIRS})
    set(PostgreSQL_LIBRARIES ${PostgreSQL_LIBRARIES})
    message(STATUS "[deps] libpq: found (PostgreSQL client libraries available)")
else()
    # Manual search for libpq on Windows if not found
    if(WIN32)
        find_library(LIBPQ_LIB pq libpq
            HINTS
                "$ENV{ProgramFiles}/PostgreSQL/*/lib"
                "C:/Program Files/PostgreSQL/*/lib"
                "C:/Program Files (x86)/PostgreSQL/*/lib"
        )
        if(LIBPQ_LIB)
            get_filename_component(PG_LIB_DIR "${LIBPQ_LIB}" DIRECTORY)
            get_filename_component(PG_ROOT "${PG_LIB_DIR}" DIRECTORY)
            set(PostgreSQL_INCLUDE_DIRS "${PG_ROOT}/include")
            set(PostgreSQL_LIBRARIES "${LIBPQ_LIB}")
            set(PostgreSQL_FOUND ON)
            message(STATUS "[deps] libpq: found manually at ${LIBPQ_LIB}")
        endif()
    endif()
endif()

# On Windows, try to find postgres.lib for linking extensions
if(WIN32 AND BUILD_PG_EXTENSION)
    # First try the libdir from pg_config
    find_library(POSTGRES_LIB postgres HINTS "${PG_LIBDIR}")
    if(NOT POSTGRES_LIB)
        # Try pkglibdir if different
        if(DEFINED PG_PKGLIBDIR AND PG_PKGLIBDIR)
            find_library(POSTGRES_LIB postgres HINTS "${PG_PKGLIBDIR}")
        endif()
    endif()
    if(NOT POSTGRES_LIB)
        # Try common locations
        find_library(POSTGRES_LIB postgres HINTS
            "$ENV{ProgramFiles}/PostgreSQL/*/lib"
            "C:/Program Files/PostgreSQL/*/lib"
            "C:/Program Files (x86)/PostgreSQL/*/lib"
        )
    endif()
    if(POSTGRES_LIB)
        message(STATUS "[deps] postgres.lib found: ${POSTGRES_LIB}")
    else()
        # Try explicit path
        set(EXPLICIT_POSTGRES_LIB "${PG_LIBDIR}/postgres.lib")
        if(EXISTS "${EXPLICIT_POSTGRES_LIB}")
            set(POSTGRES_LIB "${EXPLICIT_POSTGRES_LIB}")
            message(STATUS "[deps] postgres.lib found explicitly: ${POSTGRES_LIB}")
        else()
            message(WARNING "[deps] postgres.lib not found; disabling PostgreSQL extensions")
            set(BUILD_PG_EXTENSION OFF)
        endif()
    endif()
endif()

# ----------------------------------------------------------------------------
# BLAKE3 (fast cryptographic hashing with SIMD)
# ----------------------------------------------------------------------------
set(HAS_BLAKE3 OFF)
set(BLAKE3_INCLUDE_DIRS "")

# Try to find system BLAKE3 first
find_package(blake3 QUIET CONFIG)
if(blake3_FOUND OR TARGET BLAKE3::blake3)
    set(HAS_BLAKE3 ON)
    message(STATUS "[deps] BLAKE3: found via package")
else()
    # Fetch official BLAKE3 from GitHub
    message(STATUS "[deps] BLAKE3: fetching from GitHub with SIMD optimizations")
    FetchContent_Declare(
        blake3_fetch
        GIT_REPOSITORY https://github.com/BLAKE3-team/BLAKE3.git
        GIT_TAG 1.5.4
        SOURCE_SUBDIR c
    )
    # Configure BLAKE3 build options for maximum SIMD
    set(BLAKE3_FETCH_CONTENT ON CACHE BOOL "" FORCE)

    # Suppress deprecation warning from BLAKE3's old cmake_minimum_required
    set(_SAVED_WARN_DEPRECATED ${CMAKE_WARN_DEPRECATED})
    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(blake3_fetch)
    set(CMAKE_WARN_DEPRECATED ${_SAVED_WARN_DEPRECATED} CACHE BOOL "" FORCE)
    unset(_SAVED_WARN_DEPRECATED)

    # BLAKE3's CMake creates a target we can use
    if(TARGET blake3)
        set(HAS_BLAKE3 ON)
        message(STATUS "[deps] BLAKE3: fetched with SIMD support (AVX2/AVX-512)")
    else()
        message(WARNING "[deps] BLAKE3: fetch succeeded but target not created")
    endif()
endif()

# ----------------------------------------------------------------------------
# Google Test detection / FetchContent fallback
# ----------------------------------------------------------------------------
set(GTest_FOUND OFF)
set(GTEST_INCLUDE_DIRS "")

find_package(GTest QUIET)
if(GTest_FOUND)
    set(GTest_FOUND ON)
    message(STATUS "[deps] GoogleTest: found via package")
else()
    message(STATUS "[deps] GoogleTest: not found system-wide; fetching via FetchContent")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    # Use FetchContent_MakeAvailable instead of Populate (CMP0169)
    # This automatically calls add_subdirectory and creates GTest::gtest targets
    FetchContent_MakeAvailable(googletest)

    set(GTest_FOUND ON)
    set(GTEST_INCLUDE_DIRS ${googletest_SOURCE_DIR}/googletest/include)
    message(STATUS "[deps] GoogleTest: fetched into ${googletest_SOURCE_DIR}")
endif()

# ----------------------------------------------------------------------------
# Hints for vcpkg / conan users (informational)
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Dependency hints:")
message(STATUS "  - On Windows, vcpkg is recommended: vcpkg install eigen3 mkl hnswlib libpq gtest")
message(STATUS "  - On Linux: apt install libeigen3-dev libpq-dev libgtest-dev (or use vcpkg/conan)")
message(STATUS "")

# ----------------------------------------------------------------------------
# Final canonical variables (ensure defined)
# ----------------------------------------------------------------------------
if(NOT DEFINED HAS_THREADS)
    set(HAS_THREADS OFF)
endif()
if(NOT DEFINED HAS_OPENMP)
    set(HAS_OPENMP OFF)
endif()
if(NOT DEFINED HAS_EIGEN)
    set(HAS_EIGEN OFF)
endif()
if(NOT DEFINED HAS_MKL)
    set(HAS_MKL OFF)
endif()
if(NOT DEFINED HAS_HNSWLIB)
    set(HAS_HNSWLIB OFF)
endif()
if(NOT DEFINED PostgreSQL_FOUND)
    set(PostgreSQL_FOUND OFF)
endif()
if(NOT DEFINED BUILD_PG_EXTENSION)
    set(BUILD_PG_EXTENSION OFF)
endif()
if(NOT DEFINED GTest_FOUND)
    set(GTest_FOUND OFF)
endif()

# ----------------------------------------------------------------------------
# Print final summary
# ----------------------------------------------------------------------------
message(STATUS "=== Dependency Detection Summary ===")
message(STATUS "Threads:            ${HAS_THREADS}")
message(STATUS "OpenMP:             ${HAS_OPENMP}")
message(STATUS "Eigen3:             ${HAS_EIGEN}")
if(HAS_EIGEN AND TARGET Eigen3::Eigen)
    message(STATUS "  Eigen target:     Eigen3::Eigen")
elseif(HAS_EIGEN AND Eigen3_INCLUDE_DIRS)
    message(STATUS "  Eigen include:    ${Eigen3_INCLUDE_DIRS}")
endif()
message(STATUS "Intel MKL:          ${HAS_MKL}")
message(STATUS "BLAKE3 (SIMD):      ${HAS_BLAKE3}")
message(STATUS "HNSWLib:            ${HAS_HNSWLIB}")
message(STATUS "PostgreSQL ext:     ${BUILD_PG_EXTENSION}")
message(STATUS "PostgreSQL client:  ${PostgreSQL_FOUND}")
message(STATUS "Google Test:        ${GTest_FOUND}")
message(STATUS "")