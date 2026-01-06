cmake_minimum_required(VERSION 3.20)
project(hypercube VERSION 1.0.0 LANGUAGES CXX C)

# ============================================================================
# Build Architecture:
#   1. hypercube_core   - Pure C++ library (no PostgreSQL headers)
#   2. hypercube_c      - C API bridge (extern "C" wrappers)
#   3. hypercube        - PostgreSQL extension (pure C, links hypercube_c)
#   4. semantic_ops     - PostgreSQL extension (pure C, links hypercube_c)
#   5. hypercube (CLI)  - Unified command-line tool
#
# This architecture solves the PostgreSQL header incompatibility with C++20
# by ensuring that PostgreSQL headers are only included in pure C files.
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# Intel MKL for Optimal Performance on Intel CPUs
# ============================================================================
# Priority: MKL > Eigen > Custom Jacobi
set(HAS_MKL OFF)
find_package(MKL QUIET CONFIG)
if(MKL_FOUND)
    message(STATUS "Found Intel MKL: ${MKL_ROOT}")
    set(HAS_MKL ON)
else()
    # Try finding MKL via environment variable
    if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT "$ENV{MKLROOT}")
        find_path(MKL_INCLUDE_DIR mkl.h HINTS "${MKL_ROOT}/include")
        find_library(MKL_CORE_LIB mkl_core HINTS "${MKL_ROOT}/lib/intel64")
        if(MKL_INCLUDE_DIR AND MKL_CORE_LIB)
            message(STATUS "Found Intel MKL (via MKLROOT): ${MKL_ROOT}")
            set(HAS_MKL ON)
        endif()
    endif()
endif()

if(NOT HAS_MKL)
    message(STATUS "Intel MKL not found - trying Eigen3")
endif()

# ============================================================================
# Eigen3 for Optimized Linear Algebra (eigensolvers, BLAS)
# ============================================================================
set(HAS_EIGEN OFF)
if(NOT HAS_MKL)
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND)
        message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
        set(HAS_EIGEN ON)
    else()
        # Try vcpkg path
        find_path(EIGEN3_INCLUDE_DIR Eigen/Core
            HINTS
            "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg_installed/x64-windows/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg_installed/x64-linux/include"
            "/usr/include/eigen3"
            "/usr/local/include/eigen3"
        )
        if(EIGEN3_INCLUDE_DIR)
            message(STATUS "Found Eigen3 manually: ${EIGEN3_INCLUDE_DIR}")
            set(HAS_EIGEN ON)
        else()
            message(STATUS "Eigen3 not found - using fallback Jacobi eigensolver")
        endif()
    endif()
else()
    message(STATUS "Skipping Eigen3 (MKL is preferred)")
endif()

# ============================================================================
# Compiler Detection and Flags
# ============================================================================

if(MSVC)
    message(STATUS "Compiler: MSVC ${CMAKE_CXX_COMPILER_VERSION}")
    # Use AVX2 for broader compatibility (AVX-512 can cause issues on some CPUs)
    set(AVX_FLAG "/arch:AVX2")
    message(STATUS "SIMD: AVX2 enabled")
    # Enable optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /W3 /EHsc ${AVX_FLAG} /fp:precise /GL /Oi /Ot")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /DDEBUG /W3 /Zi /EHsc")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG /W3 ${AVX_FLAG} /fp:precise /GL /Oi /Ot")
    set(CMAKE_C_FLAGS_DEBUG "/Od /DDEBUG /W3 /Zi")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    add_compile_options(/wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Compiler: Clang ${CMAKE_CXX_COMPILER_VERSION}")
    # Clang on Windows needs explicit AVX flags
    if(WIN32)
        # Use native for best performance on current CPU
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra -march=native -mtune=native -ffast-math")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
        set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra -march=native -mtune=native -ffast-math")
        set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
        message(STATUS "SIMD: Native architecture (march=native)")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra -march=native")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra -fsanitize=address,undefined")
        set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra -march=native")
        set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
    endif()
    add_compile_options(-Wno-unused-parameter -Wno-sign-compare)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(STATUS "Compiler: GCC ${CMAKE_CXX_COMPILER_VERSION}")
    # Enable native arch (detects AVX2/AVX-512), fast-math, loop unrolling
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG -Wall -Wextra -ffast-math -funroll-loops -ftree-vectorize")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra -fsanitize=address,undefined")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG -Wall -Wextra -ffast-math -funroll-loops -ftree-vectorize")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
    message(STATUS "SIMD: Native architecture (march=native)")
else()
    message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

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
# 1. Core C++ Library (no PostgreSQL dependencies)
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
)

target_include_directories(hypercube_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add HNSWLIB for fast approximate nearest neighbors
set(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/hnswlib")
if(EXISTS "${HNSWLIB_INCLUDE_DIR}/hnswlib/hnswlib.h")
    message(STATUS "HNSWLIB: Found at ${HNSWLIB_INCLUDE_DIR}")
    set(HAS_HNSWLIB TRUE)
else()
    message(STATUS "HNSWLIB: Not found - using brute-force k-NN")
    set(HAS_HNSWLIB FALSE)
endif()

# Add Eigen include directory if available
if(HAS_EIGEN)
    target_include_directories(hypercube_core PUBLIC ${EIGEN3_INCLUDE_DIR})
    target_compile_definitions(hypercube_core PUBLIC HAS_EIGEN=1)
endif()

# Add HNSWLIB include directory if available
if(HAS_HNSWLIB)
    target_include_directories(hypercube_core PUBLIC ${HNSWLIB_INCLUDE_DIR})
    target_compile_definitions(hypercube_core PUBLIC HAS_HNSWLIB=1)
endif()

# Link Intel MKL if available
if(HAS_MKL)
    # MKL linking on Windows with Clang/LLVM
    if(WIN32)
        # Find MKL libraries - use explicit paths since find_library may fail with .lib suffix
        set(MKL_LIB_DIR "${MKL_ROOT}/lib")
        
        set(MKL_INTEL_LP64 "${MKL_LIB_DIR}/mkl_intel_lp64.lib")
        set(MKL_SEQUENTIAL "${MKL_LIB_DIR}/mkl_sequential.lib")
        set(MKL_CORE "${MKL_LIB_DIR}/mkl_core.lib")
        
        if(EXISTS "${MKL_INTEL_LP64}" AND EXISTS "${MKL_SEQUENTIAL}" AND EXISTS "${MKL_CORE}")
            message(STATUS "MKL libraries found:")
            message(STATUS "  mkl_intel_lp64: ${MKL_INTEL_LP64}")
            message(STATUS "  mkl_sequential: ${MKL_SEQUENTIAL}")
            message(STATUS "  mkl_core: ${MKL_CORE}")
            
            target_include_directories(hypercube_core PUBLIC "${MKL_ROOT}/include")
            target_link_libraries(hypercube_core
                PUBLIC "${MKL_INTEL_LP64}" "${MKL_SEQUENTIAL}" "${MKL_CORE}"
            )
            target_compile_definitions(hypercube_core PUBLIC HAS_MKL=1)
        else()
            message(WARNING "MKL libraries not found at ${MKL_LIB_DIR} - disabling MKL")
            set(HAS_MKL OFF)
            target_compile_definitions(hypercube_core PUBLIC HAS_MKL=0)
        endif()
    else()
        # Linux/Mac linking
        target_include_directories(hypercube_core PUBLIC "${MKL_ROOT}/include")
        target_link_libraries(hypercube_core
            PUBLIC mkl_intel_lp64 mkl_sequential mkl_core pthread m dl
        )
        target_compile_definitions(hypercube_core PUBLIC HAS_MKL=1)
    endif()
else()
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=0)
endif()

target_link_libraries(hypercube_core
    PRIVATE Threads::Threads
)

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

add_executable(test_clustering tests/test_clustering.cpp)
target_link_libraries(test_clustering hypercube_core)
add_test(NAME ClusteringTest COMMAND test_clustering)

add_executable(test_laplacian_4d tests/test_laplacian_4d.cpp)
target_link_libraries(test_laplacian_4d hypercube_core)
add_test(NAME Laplacian4DTest COMMAND test_laplacian_4d)

# ============================================================================
# PostgreSQL Tools (require libpq client library)
# ============================================================================

# Try to find libpq via vcpkg or system
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
    )
    target_include_directories(hypercube_ingest PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PostgreSQL_INCLUDE_DIRS}
    )
    target_link_libraries(hypercube_ingest
        hypercube_core
        ${PostgreSQL_LIBRARIES}
        Threads::Threads
    )
    
    # -------------------------------------------------------------------------
    # PostgreSQL CLI tools
    # -------------------------------------------------------------------------
    
    add_executable(seed_atoms_parallel src/tools/seed_atoms_parallel.cpp)
    target_include_directories(seed_atoms_parallel PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(seed_atoms_parallel hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    # Modular safetensor ingester (replaces archived monolith)
    add_executable(ingest_safetensor src/tools/ingest_safetensor_modular.cpp)
    target_include_directories(ingest_safetensor PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    if(HAS_HNSWLIB)
        target_include_directories(ingest_safetensor PRIVATE ${HNSWLIB_INCLUDE_DIR})
        target_compile_definitions(ingest_safetensor PRIVATE HAS_HNSWLIB=1)
    endif()
    target_link_libraries(ingest_safetensor hypercube_core hypercube_ingest ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    # Safetensor ingestion with 4D Laplacian projection (no shape table)
    add_executable(ingest_safetensor_4d src/tools/ingest_safetensor_4d.cpp)
    target_include_directories(ingest_safetensor_4d PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(ingest_safetensor_4d hypercube_core hypercube_ingest ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    # UNIVERSAL safetensor ingestion - ingests ALL tensors from ANY model (vision, language, etc)
    add_executable(ingest_safetensor_universal src/tools/ingest_safetensor_universal.cpp)
    target_include_directories(ingest_safetensor_universal PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    if(HAS_HNSWLIB)
        target_include_directories(ingest_safetensor_universal PRIVATE ${HNSWLIB_INCLUDE_DIR})
        target_compile_definitions(ingest_safetensor_universal PRIVATE HAS_HNSWLIB=1)
    endif()
    target_link_libraries(ingest_safetensor_universal hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    add_executable(extract_embeddings src/tools/extract_embeddings.cpp)
    target_include_directories(extract_embeddings PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(extract_embeddings hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    # DEPRECATED: manifold_project (JL random projection) - moved to archive
    # DEPRECATED: manifold_4d (normalized Laplacian + random power iteration) - moved to archive
    # USE: ingest_safetensor_4d (Laplacian Eigenmaps + Lanczos + Gram-Schmidt) - CANONICAL
    
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
    target_link_libraries(hypercube_cli 
        hypercube_core 
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
    if(HAS_HNSWLIB)
        target_compile_definitions(hypercube_cli PRIVATE HAS_HNSWLIB=1)
        target_include_directories(hypercube_cli PRIVATE ${HNSWLIB_INCLUDE_DIR})
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
