# ============================================================================
# Dependency Detection Module
# ============================================================================

include(FetchContent)

# ============================================================================
# Threading Support
# ============================================================================

find_package(Threads REQUIRED)

# ============================================================================
# OpenMP (Optional - Compiler Feature)
# ============================================================================

# OpenMP is a compiler feature, not a library - use standard CMake detection
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: Enabled (version ${OpenMP_CXX_VERSION})")
    set(HAS_OPENMP ON)
else()
    message(STATUS "OpenMP: Not found - building without OpenMP support")
    message(STATUS "Note: OpenMP is a compiler feature. Ensure your compiler supports it.")
    set(HAS_OPENMP OFF)
endif()

# ============================================================================
# Eigen3 Linear Algebra Library (REQUIRED)
# ============================================================================

# Try multiple ways to find Eigen3 - REQUIRED dependency
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    # Try pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(EIGEN3 QUIET eigen3)
        if(EIGEN3_FOUND)
            set(Eigen3_FOUND TRUE)
            set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
        endif()
    endif()
endif()

if(NOT Eigen3_FOUND)
    # Try manual path detection
    find_path(Eigen3_INCLUDE_DIR Eigen/Core
              PATHS "/usr/local/include/eigen3"
                    "/usr/include/eigen3"
                    "$ENV{HOME}/.local/include/eigen3"
                    "$ENV{CONDA_PREFIX}/include/eigen3"
                    "C:/vcpkg/installed/x64-windows/include/eigen3"
                    "C:/Program Files/eigen3/include/eigen3"
              PATH_SUFFIXES eigen3)
    if(Eigen3_INCLUDE_DIR)
        set(Eigen3_FOUND TRUE)
        set(Eigen3_INCLUDE_DIRS ${Eigen3_INCLUDE_DIR})
    endif()
endif()

if(Eigen3_FOUND)
    message(STATUS "Eigen3: Found - linear algebra library ready")
    message(STATUS "Eigen3 include dirs: ${Eigen3_INCLUDE_DIRS}")
    set(HAS_EIGEN ON)
else()
    message(FATAL_ERROR "\n=== EIGEN3 REQUIRED BUT NOT FOUND ===\n"
                       "Eigen3 is a required dependency for this project.\n\n"
                       "Ensure Eigen3 is installed and available:\n"
                       "- Check that Eigen3 headers are in your include path\n"
                       "- Set CMAKE_PREFIX_PATH to include Eigen3's installation directory\n"
                       "- Or install via your preferred package manager\n\n"
                       "Common Eigen3 locations:\n"
                       "- Windows: C:/Program Files/eigen3 or via vcpkg\n"
                       "- Linux: /usr/include/eigen3\n"
                       "- macOS: /usr/local/include/eigen3\n"
                       "=======================================\n")
endif()

# ============================================================================
# Intel MKL BLAS Library (REQUIRED FOR PERFORMANCE)
# ============================================================================

# First try CMake's built-in MKL package (from vcpkg intel-mkl)
find_package(MKL CONFIG QUIET)

# If not found, try manual detection for local installations
if(NOT MKL_FOUND)
    # Manual MKL detection for common installation paths
    if(WIN32)
        # Windows MKL paths - include common installation locations
        set(MKL_ROOT_SEARCH_PATHS
            "$ENV{MKLROOT}"
            "D:/Intel/oneAPI/mkl/latest"
            "C:/Intel/oneAPI/mkl/latest"
            "$ENV{ProgramFiles}/Intel/oneAPI/mkl/latest"
            "C:/Program Files/Intel/oneAPI/mkl/latest"
            "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
            "$ENV{ProgramFiles}/Intel/Composer XE/mkl"
            # Direct path to known MKL installations
            "D:/Intel/oneAPI/mkl/2025.3"
            "C:/Intel/oneAPI/mkl/2025.3"
        )
        find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT_SEARCH_PATHS} PATH_SUFFIXES include)
        find_library(MKL_CORE_LIBRARY mkl_core PATHS ${MKL_ROOT_SEARCH_PATHS} PATH_SUFFIXES lib/intel64 lib)
        find_library(MKL_SEQUENTIAL_LIBRARY mkl_sequential PATHS ${MKL_ROOT_SEARCH_PATHS} PATH_SUFFIXES lib/intel64 lib)
        find_library(MKL_INTEL_LIBRARY mkl_intel_lp64 PATHS ${MKL_ROOT_SEARCH_PATHS} PATH_SUFFIXES lib/intel64 lib)
    else()
        # Linux/macOS MKL paths
        set(MKL_ROOT_SEARCH_PATHS
            "$ENV{MKLROOT}"
            "/opt/intel/oneapi/mkl/latest"
            "/opt/intel/composerxe/mkl"
            "/usr/local/intel/mkl"
        )
        find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT_SEARCH_PATHS} PATH_SUFFIXES include)
        find_library(MKL_CORE_LIBRARY mkl_core PATHS ${MKL_ROOT_SEARCH_PATHS} PATH_SUFFIXES lib/intel64 lib)
        find_library(MKL_SEQUENTIAL_LIBRARY mkl_sequential PATHS ${MKL_ROOT_SEARCH_PATHS} PATH_SUFFIXES lib/intel64 lib)
        find_library(MKL_INTEL_LIBRARY mkl_intel_lp64 PATHS ${MKL_ROOT_SEARCH_PATHS} PATH_SUFFIXES lib/intel64 lib)
    endif()

    if(MKL_INCLUDE_DIR AND MKL_CORE_LIBRARY)
        set(MKL_FOUND TRUE)
        set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
        set(MKL_LIBRARIES ${MKL_SEQUENTIAL_LIBRARY} ${MKL_INTEL_LIBRARY} ${MKL_CORE_LIBRARY})
        if(WIN32)
            list(APPEND MKL_LIBRARIES ${MKL_CORE_LIBRARY})
        endif()
    endif()
endif()

if(MKL_FOUND)
    message(STATUS "Intel MKL: Found - using optimized BLAS operations")
    message(STATUS "MKL include dirs: ${MKL_INCLUDE_DIRS}")
    message(STATUS "MKL libraries: ${MKL_LIBRARIES}")
    set(HAS_MKL ON)
else()
    message(FATAL_ERROR "\n=== INTEL MKL REQUIRED BUT NOT FOUND ===\n"
                       "Intel MKL is required for optimal performance in this project.\n\n"
                       "Ensure Intel MKL is installed and available:\n"
                       "- Check that MKL libraries and headers are in your paths\n"
                       "- Set MKLROOT environment variable to MKL installation directory\n"
                       "- Set CMAKE_PREFIX_PATH to include MKL's installation directory\n\n"
                       "Common MKL locations:\n"
                       "- Windows: D:/Intel/oneAPI/mkl/latest, C:/Intel/oneAPI/mkl/latest\n"
                       "- Linux: /opt/intel/oneapi/mkl/latest\n"
                       "- macOS: /opt/intel/oneapi/mkl/latest\n\n"
                       "Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html\n"
                       "=======================================\n")
endif()

# ============================================================================
# PostgreSQL Detection
# ============================================================================

# Find pg_config for extension development
if(WIN32)
    # Try common PostgreSQL installation paths on Windows
    find_program(PG_CONFIG pg_config
        PATHS
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

    message(STATUS "PostgreSQL: ${PG_VERSION}")
    message(STATUS "PostgreSQL server includes: ${PG_INCLUDEDIR_SERVER}")
    message(STATUS "PostgreSQL pkglibdir: ${PG_PKGLIBDIR}")
    set(BUILD_PG_EXTENSION ON)
else()
    message(WARNING "pg_config not found - building without PostgreSQL extension support")
    set(BUILD_PG_EXTENSION OFF)
endif()

# Find libpq for client tools
find_package(PostgreSQL QUIET)
if(PostgreSQL_FOUND)
    message(STATUS "Found PostgreSQL client library: ${PostgreSQL_LIBRARIES}")
else()
    # Try manual libpq detection on Windows
    if(WIN32 AND NOT PostgreSQL_FOUND)
        find_library(LIBPQ_LIB pq libpq
            PATHS
                "$ENV{ProgramFiles}/PostgreSQL/*/lib"
                "C:/Program Files/PostgreSQL/*/lib"
                "C:/Program Files (x86)/PostgreSQL/*/lib"
        )
        if(LIBPQ_LIB)
            get_filename_component(PG_LIB_DIR "${LIBPQ_LIB}" DIRECTORY)
            get_filename_component(PG_ROOT "${PG_LIB_DIR}" DIRECTORY)
            set(PostgreSQL_INCLUDE_DIRS "${PG_ROOT}/include")
            set(PostgreSQL_LIBRARIES "${LIBPQ_LIB}")
            set(PostgreSQL_FOUND TRUE)
            message(STATUS "Found libpq manually: ${LIBPQ_LIB}")
            message(STATUS "Include dirs: ${PostgreSQL_INCLUDE_DIRS}")
        endif()
    endif()
endif()

# Find postgres.lib on Windows for extensions
if(WIN32 AND BUILD_PG_EXTENSION)
    find_library(POSTGRES_LIB postgres HINTS "${PG_PKGLIBDIR}" "${PG_LIBDIR}")
    if(POSTGRES_LIB)
        message(STATUS "Found postgres.lib: ${POSTGRES_LIB}")
    else()
        message(WARNING "postgres.lib not found - extensions may not link correctly")
    endif()
endif()

# ============================================================================
# Google Test Framework (Optional)
# ============================================================================

# ============================================================================
# HNSWLib k-NN Library (REQUIRED)
# ============================================================================

# Try to find HNSWLib package - REQUIRED for k-NN operations
find_package(hnswlib CONFIG QUIET)
if(NOT hnswlib_FOUND)
    # Fallback: Try pkg-config for HNSWLib
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(HNSWLIB QUIET hnswlib)
    endif()

    # Manual detection for common paths (including vcpkg)
    if(NOT HNSWLIB_FOUND)
        find_path(HNSWLIB_INCLUDE_DIR hnswlib.h
                  PATHS "/usr/local/include"
                        "/usr/include"
                        "$ENV{HOME}/.local/include"
                        "$ENV{CONDA_PREFIX}/include"
                        "C:/vcpkg/installed/x64-windows/include"
                        "C:/Program Files/hnswlib/include"
                  PATH_SUFFIXES hnswlib)

        find_library(HNSWLIB_LIBRARY hnswlib
                     PATHS "/usr/local/lib"
                           "/usr/lib"
                           "$ENV{HOME}/.local/lib"
                           "$ENV{CONDA_PREFIX}/lib"
                           "C:/vcpkg/installed/x64-windows/lib"
                           "C:/Program Files/hnswlib/lib")

        if(HNSWLIB_INCLUDE_DIR AND HNSWLIB_LIBRARY)
            set(HNSWLIB_FOUND TRUE)
            set(HNSWLIB_INCLUDE_DIRS ${HNSWLIB_INCLUDE_DIR})
            set(HNSWLIB_LIBRARIES ${HNSWLIB_LIBRARY})
        endif()
    endif()
endif()

if(HNSWLIB_FOUND OR hnswlib_FOUND)
    message(STATUS "HNSWLib: Found - approximate k-NN enabled")
    set(HAS_HNSWLIB ON)
    if(HNSWLIB_INCLUDE_DIRS)
        message(STATUS "HNSWLib include dirs: ${HNSWLIB_INCLUDE_DIRS}")
    endif()
    if(HNSWLIB_LIBRARIES)
        message(STATUS "HNSWLib libraries: ${HNSWLIB_LIBRARIES}")
    endif()
else()
    # Fetch HNSWLib from source if not found system-wide
    message(STATUS "HNSWLib: Not found system-wide, fetching from source...")
    FetchContent_Declare(
        hnswlib_src
        GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
        GIT_TAG v0.8.0
    )
    FetchContent_MakeAvailable(hnswlib_src)
    set(HAS_HNSWLIB ON)
    message(STATUS "HNSWLib: Built from source")
endif()

# ============================================================================
# Google Test Framework (Optional)
# ============================================================================

find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "Found Google Test: ${GTEST_INCLUDE_DIRS}")
else()
    message(STATUS "Google Test not found - skipping test suite")
    message(STATUS "To enable tests: vcpkg install gtest:x64-windows (or equivalent)")
endif()