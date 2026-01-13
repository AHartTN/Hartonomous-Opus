# ============================================================================
# Dependency Detection Module
# ============================================================================

include(FetchContent)

# ============================================================================
# Threading Support
# ============================================================================

find_package(Threads REQUIRED)

# ============================================================================
# OpenMP (Optional)
# ============================================================================

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: Enabled (version ${OpenMP_CXX_VERSION})")
    set(HAS_OPENMP ON)
else()
    message(STATUS "OpenMP: Not found - building without OpenMP support")
    set(HAS_OPENMP OFF)
endif()

# ============================================================================
# Eigen3 Linear Algebra Library
# ============================================================================

find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3: Found via system package")
    set(HAS_EIGEN ON)
else()
    # Fetch Eigen3 if not found system-wide
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
    set(HAS_EIGEN ON)
    message(STATUS "Eigen3: Fetched and built from source")
endif()

# ============================================================================
# Intel MKL (Optional BLAS)
# ============================================================================

find_package(MKL QUIET)
if(MKL_FOUND)
    message(STATUS "Intel MKL: Found - using optimized BLAS operations")
    set(HAS_MKL ON)
else()
    message(STATUS "Intel MKL: Not found - using Eigen3 built-in BLAS")
    set(HAS_MKL OFF)
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

find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "Found Google Test: ${GTEST_INCLUDE_DIRS}")
else()
    message(STATUS "Google Test not found - skipping test suite")
    message(STATUS "To enable tests: vcpkg install gtest:x64-windows (or equivalent)")
endif()