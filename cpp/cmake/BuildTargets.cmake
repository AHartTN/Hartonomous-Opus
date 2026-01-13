# ============================================================================
# Build Targets Module
# ============================================================================

# ============================================================================
# Core C++ Library
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
    src/core/superfibonacci.cpp
    src/core/unicode_categorization.cpp
    src/core/coordinate_utilities.cpp
    src/core/dense_registry.cpp
)

target_include_directories(hypercube_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Core library dependencies
target_link_libraries(hypercube_core
    PRIVATE Threads::Threads
)
if(HAS_OPENMP)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(hypercube_core PUBLIC OpenMP::OpenMP_CXX)
    else()
        # Fallback to manual OpenMP linking
        target_compile_options(hypercube_core PUBLIC ${OpenMP_CXX_FLAGS})
        target_link_libraries(hypercube_core PUBLIC ${OpenMP_CXX_LIBRARIES})
    endif()
endif()

# Linear algebra backend
if(TARGET Eigen3::Eigen)
    target_link_libraries(hypercube_core PUBLIC Eigen3::Eigen)
else()
    target_include_directories(hypercube_core PUBLIC ${EIGEN3_INCLUDE_DIR})
endif()
target_compile_definitions(hypercube_core PUBLIC HAS_EIGEN=1)

# MKL for optimized BLAS operations
if(HAS_MKL)
    if(TARGET MKL::MKL)
        # Use vcpkg-installed MKL with CMake targets
        target_link_libraries(hypercube_core PUBLIC MKL::MKL)
    else()
        # Fallback to manual linking for local installations
        target_include_directories(hypercube_core PUBLIC ${MKL_INCLUDE_DIRS})
        target_link_libraries(hypercube_core PUBLIC ${MKL_LIBRARIES})
    endif()
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=1)
else()
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=0)
endif()

# SIMD capabilities - detailed feature detection
target_compile_definitions(hypercube_core PUBLIC
    HAS_AVX=$<BOOL:${HAS_AVX}>
    HAS_AVX2=$<BOOL:${HAS_AVX2}>
    HAS_AVX512F=$<BOOL:${HAS_AVX512F}>
    HAS_FMA3=$<BOOL:${HAS_FMA3}>
    HAS_AVX_VNNI=$<BOOL:${HAS_AVX_VNNI}>
    HAS_OPENMP=$<BOOL:${HAS_OPENMP}>
)

# AVX512 features are now handled at the compiler flag level
# No need to undefine macros since we don't enable AVX512 flags when CPU doesn't support them
if(NOT HAS_AVX512F)
    target_compile_definitions(hypercube_core PRIVATE
        # Disable AVX512 code paths at compile time
        DISABLE_AVX512
        DISABLE_AVX512F
        DISABLE_AVX512DQ
        DISABLE_AVX512BW
        DISABLE_AVX512VL
        DISABLE_AVX512VNNI
    )
    message(STATUS "AVX-512: All AVX-512 features disabled (not supported by CPU)")
else()
    message(STATUS "AVX-512: CPU supports AVX-512 - features enabled where available")
endif()

# Platform-specific definitions
if(WIN32)
    target_compile_definitions(hypercube_core PRIVATE NOMINMAX)
endif()

# ============================================================================
# C API Bridge Library
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
# PostgreSQL Extensions
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

    # -------------------------------------------------------------------------
    # hypercube PostgreSQL Extension
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

    # Suppress warnings from PostgreSQL headers - PostgreSQL uses Microsoft extensions
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(hypercube PRIVATE
            -Wno-macro-redefined
            -Wno-incompatible-pointer-types
            -Wno-microsoft-include
            -Wno-unused-but-set-variable
            -Wno-pointer-sign
            -Wno-language-extension-token  # Allow __int64, __forceinline, etc.
            -Wno-flexible-array-extensions  # Allow flexible array members
        )
    elseif(MSVC)
        target_compile_options(hypercube PRIVATE /wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
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
    # semantic_ops PostgreSQL Extension
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

    # Suppress warnings from PostgreSQL headers - PostgreSQL uses Microsoft extensions
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(semantic_ops PRIVATE
            -Wno-macro-redefined
            -Wno-incompatible-pointer-types
            -Wno-microsoft-include
            -Wno-unused-but-set-variable
            -Wno-pointer-sign
            -Wno-language-extension-token  # Allow __int64, __forceinline, etc.
            -Wno-flexible-array-extensions  # Allow flexible array members
        )
    elseif(MSVC)
        target_compile_options(semantic_ops PRIVATE /wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
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
    # hypercube_ops PostgreSQL Extension
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

    # Suppress warnings from PostgreSQL headers - PostgreSQL uses Microsoft extensions
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(hypercube_ops PRIVATE
            -Wno-macro-redefined
            -Wno-incompatible-pointer-types
            -Wno-microsoft-include
            -Wno-unused-but-set-variable
            -Wno-pointer-sign
            -Wno-language-extension-token  # Allow __int64, __forceinline, etc.
            -Wno-flexible-array-extensions  # Allow flexible array members
        )
    elseif(MSVC)
        target_compile_options(hypercube_ops PRIVATE /wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
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
    # embedding_ops PostgreSQL Extension
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

    # Suppress warnings from PostgreSQL headers - PostgreSQL uses Microsoft extensions
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(embedding_ops PRIVATE
            -Wno-macro-redefined
            -Wno-incompatible-pointer-types
            -Wno-microsoft-include
            -Wno-unused-but-set-variable
            -Wno-pointer-sign
            -Wno-language-extension-token  # Allow __int64, __forceinline, etc.
            -Wno-flexible-array-extensions  # Allow flexible array members
        )
    elseif(MSVC)
        target_compile_options(embedding_ops PRIVATE /wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
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

    # Suppress warnings from PostgreSQL headers - PostgreSQL uses Microsoft extensions
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(generative PRIVATE
            -Wno-macro-redefined
            -Wno-incompatible-pointer-types
            -Wno-microsoft-include
            -Wno-unused-but-set-variable
            -Wno-pointer-sign
            -Wno-language-extension-token  # Allow __int64, __forceinline, etc.
            -Wno-flexible-array-extensions  # Allow flexible array members
        )
    elseif(MSVC)
        target_compile_options(generative PRIVATE /wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
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

add_executable(test_unicode_seeding tests/test_unicode_seeding.cpp)
target_link_libraries(test_unicode_seeding hypercube_core)
add_test(NAME UnicodeSeedingTest COMMAND test_unicode_seeding)

# ============================================================================
# PostgreSQL Tools (require libpq client library)
# ============================================================================

if(PostgreSQL_FOUND)
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
    # HNSWLib disabled
    target_link_libraries(hypercube_ingest PRIVATE
        hypercube_core
        ${PostgreSQL_LIBRARIES}
        Threads::Threads
    )

    # Link HNSWLib if found
    if(HAS_HNSWLIB)
        if(TARGET hnswlib::hnswlib)
            # Use vcpkg-installed HNSWLib with CMake targets
            target_link_libraries(hypercube_ingest PRIVATE hnswlib::hnswlib)
        elseif(HNSWLIB_FOUND)
            # Fallback to manual linking
            target_include_directories(hypercube_ingest PRIVATE ${HNSWLIB_INCLUDE_DIRS})
            target_link_libraries(hypercube_ingest PRIVATE ${HNSWLIB_LIBRARIES})
        endif()
        target_compile_definitions(hypercube_ingest PRIVATE HAS_HNSWLIB=1)
    else()
        target_compile_definitions(hypercube_ingest PRIVATE HAS_HNSWLIB=0)
    endif()

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

    # Link HNSWLib if found (inherited from hypercube_ingest)
    if(HAS_HNSWLIB)
        if(TARGET hnswlib::hnswlib)
            # Use vcpkg-installed HNSWLib with CMake targets
            target_link_libraries(ingest_safetensor PRIVATE hnswlib::hnswlib)
        elseif(HNSWLIB_FOUND)
            # Fallback to manual linking
            target_include_directories(ingest_safetensor PRIVATE ${HNSWLIB_INCLUDE_DIRS})
            target_link_libraries(ingest_safetensor PRIVATE ${HNSWLIB_LIBRARIES})
        endif()
        target_compile_definitions(ingest_safetensor PRIVATE HAS_HNSWLIB=1)
    else()
        target_compile_definitions(ingest_safetensor PRIVATE HAS_HNSWLIB=0)
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

    # Link HNSWLib if found
    if(HAS_HNSWLIB)
        if(TARGET hnswlib::hnswlib)
            # Use vcpkg-installed HNSWLib with CMake targets
            target_link_libraries(hypercube_cli PRIVATE hnswlib::hnswlib)
        elseif(HNSWLIB_FOUND)
            # Fallback to manual linking
            target_include_directories(hypercube_cli PRIVATE ${HNSWLIB_INCLUDE_DIRS})
            target_link_libraries(hypercube_cli PRIVATE ${HNSWLIB_LIBRARIES})
        endif()
        target_compile_definitions(hypercube_cli PRIVATE HAS_HNSWLIB=1)
    else()
        target_compile_definitions(hypercube_cli PRIVATE HAS_HNSWLIB=0)
    endif()

    # Add backend detection defines
    target_compile_definitions(hypercube_cli PRIVATE HAS_EIGEN=1)
    if(HAS_MKL)
        target_compile_definitions(hypercube_cli PRIVATE HAS_MKL=1)
    else()
        target_compile_definitions(hypercube_cli PRIVATE HAS_MKL=0)
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
    # Build link libraries list
    set(TEST_LINK_LIBRARIES
        hypercube_core
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
    )

    # Add HNSWLib if found
    if(HAS_HNSWLIB)
        if(TARGET hnswlib::hnswlib)
            # Use vcpkg-installed HNSWLib with CMake targets
            list(APPEND TEST_LINK_LIBRARIES hnswlib::hnswlib)
        elseif(HNSWLIB_FOUND)
            # Fallback to manual linking
            target_include_directories(hypercube_tests PRIVATE ${HNSWLIB_INCLUDE_DIRS})
            list(APPEND TEST_LINK_LIBRARIES ${HNSWLIB_LIBRARIES})
        endif()
        target_compile_definitions(hypercube_tests PRIVATE HAS_HNSWLIB=1)
    else()
        target_compile_definitions(hypercube_tests PRIVATE HAS_HNSWLIB=0)
    endif()

    target_link_libraries(hypercube_tests PRIVATE ${TEST_LINK_LIBRARIES})

    # Add backend defines
    target_compile_definitions(hypercube_tests PRIVATE HAS_EIGEN=1)
    if(HAS_MKL)
        target_compile_definitions(hypercube_tests PRIVATE HAS_MKL=1)
    else()
        target_compile_definitions(hypercube_tests PRIVATE HAS_MKL=0)
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