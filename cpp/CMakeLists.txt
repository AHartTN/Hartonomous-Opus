cmake_minimum_required(VERSION 3.24)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(hypercube VERSION 1.0.0 LANGUAGES CXX C)

# ============================================================================
# Enterprise-Grade Build Configuration
# ============================================================================

# Modern C++ standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build optimization
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable comprehensive warnings and treat as errors in Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        add_compile_options(/W4 /WX)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

# RPATH settings - use install path even during build
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/usr/lib/postgresql/18/lib")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Cross-platform threading support
find_package(Threads REQUIRED)

# ============================================================================
# Modern Dependency Management with FetchContent
# ============================================================================

include(FetchContent)

# OpenMP for parallel operations
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: Found (version ${OpenMP_CXX_VERSION})")
else()
    message(FATAL_ERROR "OpenMP is required for parallel operations")
endif()

# HNSWLIB for approximate nearest neighbors
FetchContent_Declare(
    hnswlib
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(hnswlib)

# Intel MKL (optional, preferred for performance)
find_package(MKL QUIET)
if(MKL_FOUND)
    message(STATUS "Intel MKL: Found - using optimized BLAS operations")
    set(HAS_MKL ON)
else()
    message(STATUS "Intel MKL: Not found - using Eigen3 fallback")
    set(HAS_MKL OFF)

    # Eigen3 as fallback
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
endif()

# ============================================================================
# Compiler Optimization and SIMD Detection
# ============================================================================

# Detect SIMD capabilities
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(HAS_AVX ON)
    message(STATUS "SIMD: Native architecture optimizations enabled")
else()
    set(HAS_AVX OFF)
    message(STATUS "SIMD: Native architecture not supported, using generic optimizations")
endif()

# Common optimization flags
if(MSVC)
    set(COMPILER_OPT_FLAGS "/O2 /DNDEBUG /fp:precise")
    set(COMPILER_DEBUG_FLAGS "/Od /DDEBUG /Zi")
    set(COMPILER_WARNING_FLAGS "/W3 /EHsc /permissive- /DNOMINMAX")
    set(COMPILER_AVX_FLAGS "/arch:AVX2")
    add_compile_options(/wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
else()
    set(COMPILER_OPT_FLAGS "-O3 -DNDEBUG -ffast-math")
    set(COMPILER_DEBUG_FLAGS "-g -O0 -DDEBUG")
    set(COMPILER_WARNING_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare")
    set(COMPILER_AVX_FLAGS "-march=native")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(COMPILER_DEBUG_FLAGS "${COMPILER_DEBUG_FLAGS} -fsanitize=address,undefined")
    endif()
endif()

# Apply flags based on build type
set(CMAKE_CXX_FLAGS_RELEASE "${COMPILER_OPT_FLAGS} ${COMPILER_WARNING_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${COMPILER_OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${COMPILER_DEBUG_FLAGS} ${COMPILER_WARNING_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${COMPILER_DEBUG_FLAGS}")

if(HAS_AVX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMPILER_AVX_FLAGS}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${COMPILER_AVX_FLAGS}")
endif()

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "SIMD support: ${HAS_AVX}")

# ============================================================================
# PostgreSQL Configuration
# ============================================================================

find_program(PG_CONFIG pg_config)
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

# ============================================================================
# Core C++ Library (Enterprise-Grade Architecture)
# ============================================================================

add_library(hypercube_core STATIC
    src/core/hilbert.cpp
    src/core/coordinates.cpp
    src/core/blake3_pg.cpp
    src/core/ops.cpp
    src/util/utf8.cpp
    src/core/atom_calculator.cpp
    src/core/laplacian_4d.cpp
    src/core/lanczos.cpp
    src/core/semantic_ordering.cpp
)

target_include_directories(hypercube_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Dependencies
target_link_libraries(hypercube_core
    PUBLIC OpenMP::OpenMP_CXX
    PRIVATE Threads::Threads
)

# HNSWLIB integration
if(TARGET hnswlib)
    target_link_libraries(hypercube_core PRIVATE hnswlib)
    target_compile_definitions(hypercube_core PUBLIC HAS_HNSWLIB=1)
else()
    target_compile_definitions(hypercube_core PUBLIC HAS_HNSWLIB=0)
endif()

# Linear algebra backend
if(HAS_MKL)
    target_link_libraries(hypercube_core PUBLIC MKL::MKL)
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=1 HAS_EIGEN=0)
elseif(HAS_EIGEN)
    if(TARGET Eigen3::Eigen)
        target_link_libraries(hypercube_core PUBLIC Eigen3::Eigen)
    else()
        target_include_directories(hypercube_core PUBLIC ${EIGEN3_INCLUDE_DIR})
    endif()
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=0 HAS_EIGEN=1)
else()
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=0 HAS_EIGEN=0)
endif()

# SIMD capabilities
target_compile_definitions(hypercube_core PUBLIC HAS_AVX=$<BOOL:${HAS_AVX}>)

# Platform-specific definitions
if(WIN32)
    target_compile_definitions(hypercube_core PRIVATE NOMINMAX)
endif()

# ============================================================================
# 2. C API Bridge Library (C++ implementation, C interface)
# Must be SHARED on Windows so that the PostgreSQL C extensions can link to it
# ============================================================================

add_library(hypercube_c SHARED
    src/bridge/hypercube_c.cpp
)

target_include_directories(hypercube_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hypercube_c
    PRIVATE hypercube_core
    PRIVATE Threads::Threads
)

# Export symbols properly on Windows
if(WIN32)
    target_compile_definitions(hypercube_c PRIVATE HYPERCUBE_C_EXPORTS)
endif()

set_target_properties(hypercube_c PROPERTIES
    PREFIX ""
    OUTPUT_NAME "hypercube_c"
)

# ============================================================================
# 3 & 4. PostgreSQL Extensions (pure C - compiled with C compiler)
#
# CRITICAL: These are .c files that include PostgreSQL headers.
# PostgreSQL headers have C++-incompatible code (atomics, type punning).
# By keeping these as pure C, we avoid all MSVC C++ errors.
# ============================================================================

if(BUILD_PG_EXTENSION)
    # Windows-specific defines for PostgreSQL
    if(WIN32)
        set(PG_COMPILE_DEFS
            WIN32_LEAN_AND_MEAN
            _CRT_SECURE_NO_WARNINGS
            _WINSOCK_DEPRECATED_NO_WARNINGS
        )
    else()
        set(PG_COMPILE_DEFS "")
    endif()
    
    # PostgreSQL include directories (proper order for Windows)
    if(WIN32)
        set(PG_INCLUDE_DIRS
            ${PG_INCLUDEDIR_SERVER}/port/win32_msvc
            ${PG_INCLUDEDIR_SERVER}/port/win32
            ${PG_INCLUDEDIR_SERVER}
            ${PG_INCLUDEDIR}
        )
    else()
        set(PG_INCLUDE_DIRS
            ${PG_INCLUDEDIR_SERVER}
            ${PG_INCLUDEDIR}
        )
    endif()
    
    # Find postgres.lib on Windows
    if(WIN32)
        find_library(POSTGRES_LIB postgres HINTS "${PG_PKGLIBDIR}" "${PG_LIBDIR}")
        if(POSTGRES_LIB)
            message(STATUS "Found postgres.lib: ${POSTGRES_LIB}")
        else()
            message(WARNING "postgres.lib not found - extensions may not link correctly")
        endif()
    endif()
    
    # -------------------------------------------------------------------------
    # hypercube PostgreSQL Extension (pure C)
    # -------------------------------------------------------------------------
    
    add_library(hypercube SHARED
        src/pg/hypercube_pg.c
    )
    
    # Force C language for this file
    set_source_files_properties(src/pg/hypercube_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(hypercube PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(hypercube PRIVATE
        ${PG_COMPILE_DEFS}
        HYPERCUBE_VERSION="${PROJECT_VERSION}"
    )

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(hypercube PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()
    
    # Force C linker so we behave like native PG extension DLLs on Windows
    set_target_properties(hypercube PROPERTIES LINKER_LANGUAGE C)

    target_link_libraries(hypercube
        PRIVATE hypercube_c
    )
    
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(hypercube PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(hypercube PROPERTIES
        PREFIX ""
        OUTPUT_NAME "hypercube"
    )
    
    # -------------------------------------------------------------------------
    # semantic_ops PostgreSQL Extension (pure C)
    # -------------------------------------------------------------------------
    
    add_library(semantic_ops SHARED
        src/pg/semantic_ops_pg.c
    )
    
    set_source_files_properties(src/pg/semantic_ops_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(semantic_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(semantic_ops PRIVATE
        ${PG_COMPILE_DEFS}
    )

    set_target_properties(semantic_ops PROPERTIES LINKER_LANGUAGE C)

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(semantic_ops PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()

    target_link_libraries(semantic_ops
        PRIVATE hypercube_c
    )
    
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(semantic_ops PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(semantic_ops PROPERTIES
        PREFIX ""
        OUTPUT_NAME "semantic_ops"
    )
    
    # -------------------------------------------------------------------------
    # hypercube_ops PostgreSQL Extension (pure C - optimized batch operations)
    # Uses hypercube_c for BLAKE3, Hilbert, and coordinate functions
    # -------------------------------------------------------------------------
    
    add_library(hypercube_ops SHARED
        src/pg/hypercube_ops_pg.c
    )
    
    set_source_files_properties(src/pg/hypercube_ops_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(hypercube_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(hypercube_ops PRIVATE
        ${PG_COMPILE_DEFS}
    )

    set_target_properties(hypercube_ops PROPERTIES LINKER_LANGUAGE C)

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(hypercube_ops PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()

    # Link against hypercube_c for C API functions (seed_atoms needs them)
    target_link_libraries(hypercube_ops PRIVATE hypercube_c)
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(hypercube_ops PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(hypercube_ops PROPERTIES
        PREFIX ""
        OUTPUT_NAME "hypercube_ops"
    )
    
    # -------------------------------------------------------------------------
    # embedding_ops PostgreSQL Extension (pure C - SIMD embedding operations)
    # -------------------------------------------------------------------------
    
    # C bridge library for embedding operations (C++ with C API)
    add_library(embedding_c SHARED
        src/bridge/embedding_c.cpp
    )

    target_include_directories(embedding_c PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    target_link_libraries(embedding_c
        PRIVATE hypercube_core
        PRIVATE Threads::Threads
    )
    
    if(WIN32)
        target_compile_definitions(embedding_c PRIVATE EMBEDDING_C_EXPORTS)
    endif()
    
    set_target_properties(embedding_c PROPERTIES
        PREFIX ""
        OUTPUT_NAME "embedding_c"
    )
    
    # Pure C PostgreSQL extension
    add_library(embedding_ops SHARED
        src/pg/embedding_ops_pg.c
    )
    
    set_source_files_properties(src/pg/embedding_ops_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(embedding_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(embedding_ops PRIVATE
        ${PG_COMPILE_DEFS}
    )

    set_target_properties(embedding_ops PROPERTIES LINKER_LANGUAGE C)

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(embedding_ops PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()
    
    target_link_libraries(embedding_ops
        PRIVATE embedding_c
    )
    
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(embedding_ops PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(embedding_ops PROPERTIES
        PREFIX ""
        OUTPUT_NAME "embedding_ops"
    )
    
    # -------------------------------------------------------------------------
    # generative PostgreSQL Extension (generative walk engine)
    # -------------------------------------------------------------------------
    
    # C bridge library for generative engine (C++ with C API)
    add_library(generative_c SHARED
        src/bridge/generative_c.cpp
    )

    target_include_directories(generative_c PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    target_link_libraries(generative_c
        PRIVATE hypercube_core
        PRIVATE Threads::Threads
    )
    
    if(WIN32)
        target_compile_definitions(generative_c PRIVATE GENERATIVE_C_EXPORTS)
    endif()
    
    set_target_properties(generative_c PROPERTIES
        PREFIX ""
        OUTPUT_NAME "generative_c"
    )
    
    # Pure C PostgreSQL extension
    add_library(generative SHARED
        src/pg/generative_pg.c
    )
    
    set_source_files_properties(src/pg/generative_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(generative PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(generative PRIVATE
        ${PG_COMPILE_DEFS}
    )

    set_target_properties(generative PROPERTIES LINKER_LANGUAGE C)

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(generative PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()
    
    target_link_libraries(generative
        PRIVATE generative_c
    )
    
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(generative PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(generative PROPERTIES
        PREFIX ""
        OUTPUT_NAME "generative"
    )
    
    # -------------------------------------------------------------------------
    # Installation
    # -------------------------------------------------------------------------

    # The extension DLLs depend on hypercube_c.dll at runtime (Windows loader).
    # Install it into pkglibdir alongside the modules.
    install(TARGETS hypercube_c
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(TARGETS hypercube
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/hypercube--1.0.sql
        hypercube.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
    
    install(TARGETS semantic_ops
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/semantic_ops--1.0.sql
        sql/semantic_ops.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
    
    install(TARGETS hypercube_ops
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/hypercube_ops--1.0.sql
        sql/hypercube_ops.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
    
    install(TARGETS embedding_c
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(TARGETS embedding_ops
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/embedding_ops--1.0.sql
        sql/embedding_ops.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
    
    install(TARGETS generative_c
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(TARGETS generative
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/generative--1.0.sql
        sql/generative.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
endif()

# ============================================================================
# Tests (C++ tests, no PostgreSQL)
# ============================================================================

enable_testing()

add_executable(test_hilbert tests/test_hilbert.cpp)
target_link_libraries(test_hilbert hypercube_core)
add_test(NAME HilbertTest COMMAND test_hilbert)

add_executable(test_coordinates tests/test_coordinates.cpp)
target_link_libraries(test_coordinates hypercube_core)
add_test(NAME CoordinatesTest COMMAND test_coordinates)

add_executable(test_blake3 tests/test_blake3.cpp)
target_link_libraries(test_blake3 hypercube_core)
add_test(NAME Blake3Test COMMAND test_blake3)

add_executable(test_semantic tests/test_semantic.cpp)
target_link_libraries(test_semantic hypercube_core)
add_test(NAME SemanticTest COMMAND test_semantic)

add_executable(debug_semantic_order tests/debug_semantic_order.cpp)
add_executable(test_vocab_parse tests/test_vocab_parse.cpp)
target_include_directories(test_vocab_parse PRIVATE ${PostgreSQL_INCLUDE_DIRS})
target_link_libraries(debug_semantic_order hypercube_core)
target_link_libraries(test_vocab_parse hypercube_core hypercube_c ${PostgreSQL_LIBRARIES} Threads::Threads)
add_test(NAME DebugSemanticOrder COMMAND debug_semantic_order)



add_executable(test_clustering tests/test_clustering.cpp)
target_link_libraries(test_clustering hypercube_core)
add_test(NAME ClusteringTest COMMAND test_clustering)

add_executable(test_laplacian_4d tests/test_laplacian_4d.cpp)
target_link_libraries(test_laplacian_4d hypercube_core)
add_test(NAME Laplacian4DTest COMMAND test_laplacian_4d)

add_executable(test_eigen_solver_paths test_eigen_solver_paths.cpp)
target_link_libraries(test_eigen_solver_paths hypercube_core)

add_executable(test_cpu_features tests/test_cpu_features.cpp)
target_link_libraries(test_cpu_features hypercube_core)

add_executable(test_int8_quantization tests/test_int8_quantization.cpp)
target_link_libraries(test_int8_quantization hypercube_core)

# ============================================================================
# PostgreSQL Tools (require libpq client library)
# ============================================================================

# Try to find libpq via vcpkg or system
# Prefer system libpq over Windows installations
set(PostgreSQL_ROOT "/usr")
find_package(PostgreSQL QUIET)
if(PostgreSQL_FOUND)
    message(STATUS "Found PostgreSQL client library: ${PostgreSQL_LIBRARIES}")

    # -------------------------------------------------------------------------
    # Modular ingester library (must be defined before tools that link to it)
    # -------------------------------------------------------------------------
    add_library(hypercube_ingest STATIC
        src/db/atom_cache.cpp
        src/db/geometry.cpp
        src/db/insert.cpp
        src/ingest/cpe.cpp
        src/ingest/universal.cpp
        src/ingest/parallel_cpe.cpp
        src/ingest/sequitur.cpp
        src/ingest/pmi_contraction.cpp
        src/ingest/projection_db.cpp
        # Modular DB operations for safetensor ingestion
        src/ingest/tensor_io.cpp
        src/ingest/compositions.cpp
        src/ingest/tensor_hierarchy.cpp
        src/ingest/embedding_relations.cpp
        src/ingest/attention_relations.cpp
        src/ingest/semantic_extraction.cpp
        src/ingest/multimodal_extraction.cpp
    )
    target_include_directories(hypercube_ingest PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PostgreSQL_INCLUDE_DIRS}
        ${MKL_INCLUDE_DIRS}
    )
    if(TARGET hnswlib)
        target_link_libraries(hypercube_ingest PRIVATE hnswlib)
        target_include_directories(hypercube_ingest PRIVATE ${hnswlib_SOURCE_DIR})
        target_compile_definitions(hypercube_ingest PRIVATE HAS_HNSWLIB=1)
    endif()
    target_link_libraries(hypercube_ingest PRIVATE
        hypercube_core
        ${PostgreSQL_LIBRARIES}
        Threads::Threads
    )

    # Fix Windows min/max macro conflicts with HNSWLIB
    if(WIN32)
        target_compile_definitions(hypercube_ingest PRIVATE NOMINMAX)
    endif()

    # -------------------------------------------------------------------------
    # PostgreSQL CLI tools
    # -------------------------------------------------------------------------

    # Modular safetensor ingester (replaces archived monolith)
    add_executable(ingest_safetensor src/tools/ingest_safetensor_modular.cpp)
    target_include_directories(ingest_safetensor PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    if(TARGET hnswlib)
        target_link_libraries(ingest_safetensor PRIVATE hnswlib)
        target_include_directories(ingest_safetensor PRIVATE ${hnswlib_SOURCE_DIR})
        target_compile_definitions(ingest_safetensor PRIVATE HAS_HNSWLIB=1)
    endif()
    target_link_libraries(ingest_safetensor PRIVATE hypercube_core hypercube_ingest ${PostgreSQL_LIBRARIES} Threads::Threads)

    # Fix Windows min/max macro conflicts with HNSWLIB
    if(WIN32)
        target_compile_definitions(ingest_safetensor PRIVATE NOMINMAX)
    endif()

    # ARCHIVED: ingest_safetensor_4d (moved to src/archive/)
    # ARCHIVED: ingest_safetensor_universal (moved to src/archive/)
    # USE: ingest_safetensor (modular version with shared library components)

    add_executable(extract_embeddings src/tools/extract_embeddings.cpp)
    target_include_directories(extract_embeddings PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(extract_embeddings hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)

    # New unified ingester tool
    add_executable(ingest src/ingest/main.cpp)
    target_link_libraries(ingest hypercube_ingest hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)

    # ========================================================================
    # Unified CLI Tool: hypercube_cli
    # ========================================================================
    # Single entry point for all operations: ingest, query, test, stats
    add_executable(hypercube_cli src/cli/main.cpp)
    target_include_directories(hypercube_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PostgreSQL_INCLUDE_DIRS}
    )
    target_link_libraries(hypercube_cli PRIVATE
        hypercube_core
        hypercube_ingest
        ${PostgreSQL_LIBRARIES}
        Threads::Threads
    )

    # Add backend detection defines
    if(HAS_MKL)
        target_compile_definitions(hypercube_cli PRIVATE HAS_MKL=1)
    endif()
    if(HAS_EIGEN)
        target_compile_definitions(hypercube_cli PRIVATE HAS_EIGEN=1)
        target_include_directories(hypercube_cli PRIVATE ${EIGEN3_INCLUDE_DIR})
    endif()
    if(TARGET hnswlib)
        target_link_libraries(hypercube_cli PRIVATE hnswlib)
        target_include_directories(hypercube_cli PRIVATE ${hnswlib_SOURCE_DIR})
        target_compile_definitions(hypercube_cli PRIVATE HAS_HNSWLIB=1)
    endif()

    # Set output name to just "hypercube" (conflicts with extension on some systems)
    set_target_properties(hypercube_cli PROPERTIES OUTPUT_NAME "hc")

    # Model discovery tool (scans HC_MODEL_PATHS for HuggingFace models)
    add_executable(model_discovery src/ingest/model_discovery.cpp)
    target_link_libraries(model_discovery Threads::Threads)

    # Vocabulary extractor (extracts vocab from any model as text)
    add_executable(vocab_extract src/ingest/vocab_extract.cpp)
    target_link_libraries(vocab_extract Threads::Threads)

    # Streaming vocabulary ingester (one token at a time through PMI) - standalone
    add_executable(vocab_ingest src/ingest/vocab_ingest.cpp)
    target_link_libraries(vocab_ingest Threads::Threads)

    # Integration tests with libpq
    add_executable(test_integration tests/test_integration.cpp)
    target_include_directories(test_integration PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(test_integration hypercube_core ${PostgreSQL_LIBRARIES})
    add_test(NAME IntegrationTest COMMAND test_integration)
    set_tests_properties(IntegrationTest PROPERTIES
        ENVIRONMENT "HC_DB_NAME=hypercube;HC_DB_USER=hartonomous;HC_DB_PASS=hartonomous;HC_DB_HOST=localhost;HC_DB_PORT=5432"
    )

    add_executable(test_query_api tests/test_query_api.cpp)
    target_include_directories(test_query_api PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(test_query_api hypercube_core ${PostgreSQL_LIBRARIES})
    add_test(NAME QueryAPITest COMMAND test_query_api)
    set_tests_properties(QueryAPITest PROPERTIES
        ENVIRONMENT "HC_DB_NAME=hypercube;HC_DB_USER=hartonomous;HC_DB_PASS=hartonomous;HC_DB_HOST=localhost;HC_DB_PORT=5432"
    )
else()
    message(STATUS "PostgreSQL client library not found - skipping tool builds")
    message(STATUS "To build tools, install libpq via: vcpkg install libpq:x64-windows")
endif()

# ============================================================================
# Seed Atoms Tool (ALWAYS BUILT - uses extension when available)
# ============================================================================

# Try to find libpq for standalone seeder (even without full PostgreSQL package)
if(NOT PostgreSQL_FOUND)
    # Try manual libpq detection on Windows
    if(WIN32)
        # Look for PostgreSQL installations
        set(PG_SEARCH_PATHS
            "C:/Program Files/PostgreSQL/*/lib"
            "C:/Program Files (x86)/PostgreSQL/*/lib"
            "$ENV{ProgramFiles}/PostgreSQL/*/lib"
            "$ENV{ProgramFiles(x86)}/PostgreSQL/*/lib"
        )

        find_library(LIBPQ_LIB pq libpq PATHS ${PG_SEARCH_PATHS})
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

if(PostgreSQL_FOUND)
    add_executable(seed_atoms_parallel src/tools/seed_atoms_parallel.cpp)
    target_include_directories(seed_atoms_parallel PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(seed_atoms_parallel hypercube_core hypercube_c ${PostgreSQL_LIBRARIES} Threads::Threads)

    # Fix Windows min/max macro conflicts
    if(WIN32)
        target_compile_definitions(seed_atoms_parallel PRIVATE NOMINMAX)
    endif()

    message(STATUS "seed_atoms_parallel will be built")
else()
    message(STATUS "libpq not found - seed_atoms_parallel will not be built (extension seeding will be used)")
endif()

# ============================================================================
# Google Test Suite
# ============================================================================
find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "Found Google Test: ${GTEST_INCLUDE_DIRS}")
    enable_testing()
    
    # Core unit tests (no PostgreSQL required)
    add_executable(hypercube_tests
        tests/gtest/test_main.cpp
        tests/gtest/test_blake3.cpp
        tests/gtest/test_hilbert.cpp
        tests/gtest/test_coordinates.cpp
        tests/gtest/test_laplacian.cpp
        tests/gtest/test_backend.cpp
    )
    target_include_directories(hypercube_tests PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GTEST_INCLUDE_DIRS}
    )
    target_link_libraries(hypercube_tests 
        hypercube_core 
        GTest::gtest 
        GTest::gtest_main
        Threads::Threads
    )
    
    # Add backend defines
    if(HAS_MKL)
        target_compile_definitions(hypercube_tests PRIVATE HAS_MKL=1)
    endif()
    if(HAS_EIGEN)
        target_compile_definitions(hypercube_tests PRIVATE HAS_EIGEN=1)
        target_include_directories(hypercube_tests PRIVATE ${EIGEN3_INCLUDE_DIR})
    endif()
    if(HAS_HNSWLIB)
        target_compile_definitions(hypercube_tests PRIVATE HAS_HNSWLIB=1)
        target_include_directories(hypercube_tests PRIVATE ${HNSWLIB_INCLUDE_DIR})
    endif()
    
    add_test(NAME HypercubeTests COMMAND hypercube_tests)
    # Increase stack size to prevent crashes (MSVC only)
    if(MSVC)
        set_target_properties(hypercube_tests PROPERTIES LINK_FLAGS "/STACK:8388608")
    endif()
    
    # SQL integration tests (requires PostgreSQL)
    if(PostgreSQL_FOUND)
        add_executable(hypercube_sql_tests
            tests/gtest/test_main.cpp
            tests/gtest/test_sql_schema.cpp
            tests/gtest/test_sql_functions.cpp
            tests/gtest/test_sql_query_api.cpp
        )
        target_include_directories(hypercube_sql_tests PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${GTEST_INCLUDE_DIRS}
            ${PostgreSQL_INCLUDE_DIRS}
        )
        target_link_libraries(hypercube_sql_tests 
            hypercube_core 
            GTest::gtest 
            GTest::gtest_main
            ${PostgreSQL_LIBRARIES}
            Threads::Threads
        )
        add_test(NAME HypercubeSQLTests COMMAND hypercube_sql_tests)
        set_tests_properties(HypercubeSQLTests PROPERTIES
            ENVIRONMENT "HC_DB_NAME=hypercube;HC_DB_USER=hartonomous;HC_DB_PASS=hartonomous;HC_DB_HOST=localhost;HC_DB_PORT=5432"
        )
    endif()
else()
    message(STATUS "Google Test not found - skipping test suite")
    message(STATUS "To enable tests: vcpkg install gtest:x64-windows")
endif()

# ============================================================================
# Runtime Environment Optimization for AVX/AVX_VNNI
# ============================================================================

message(STATUS "Runtime Environment Setup for AVX Optimization:")
message(STATUS "  Set these in your ~/.bashrc or runtime environment:")
message(STATUS "  export OMP_PROC_BIND=true")
message(STATUS "  export OMP_PLACES=cores")
message(STATUS "  export MKL_DEBUG_CPU_TYPE=5  # Force AVX2 code path")
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
if(HAS_MKL)
    message(STATUS "Intel MKL: ON")
elseif(HAS_EIGEN)
    message(STATUS "Eigen3: ON")
else()
    message(STATUS "Linear algebra: Jacobi (fallback)")
endif()
if(HAS_HNSWLIB)
    message(STATUS "HNSWLIB: ON")
else()
    message(STATUS "k-NN: Brute-force")
endif()
if(GTest_FOUND)
    message(STATUS "Tests: ON (Google Test)")
else()
    message(STATUS "Tests: OFF")
endif()
message(STATUS "")
