cmake_minimum_required(VERSION 3.20)
project(hypercube VERSION 1.0.0 LANGUAGES CXX C)

# ============================================================================
# Build Architecture:
#   1. hypercube_core   - Pure C++ library (no PostgreSQL headers)
#   2. hypercube_c      - C API bridge (extern "C" wrappers)
#   3. hypercube        - PostgreSQL extension (pure C, links hypercube_c)
#   4. semantic_ops     - PostgreSQL extension (pure C, links hypercube_c)
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

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Cross-platform threading support
find_package(Threads REQUIRED)

# ============================================================================
# Compiler Detection and Flags
# ============================================================================

if(MSVC)
    message(STATUS "Compiler: MSVC ${CMAKE_CXX_COMPILER_VERSION}")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /W3 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /DDEBUG /W3 /Zi /EHsc")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG /W3")
    set(CMAKE_C_FLAGS_DEBUG "/Od /DDEBUG /W3 /Zi")
    add_compile_options(/wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Compiler: Clang ${CMAKE_CXX_COMPILER_VERSION}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
    if(NOT WIN32)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
    endif()
    add_compile_options(-Wno-unused-parameter -Wno-sign-compare)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(STATUS "Compiler: GCC ${CMAKE_CXX_COMPILER_VERSION}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra -fsanitize=address,undefined")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
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
    src/hilbert.cpp
    src/coordinates.cpp
    src/blake3_pg.cpp
)

target_include_directories(hypercube_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hypercube_core
    PRIVATE Threads::Threads
)

# ============================================================================
# 2. C API Bridge Library (C++ implementation, C interface)
# Must be SHARED on Windows so that the PostgreSQL C extensions can link to it
# ============================================================================

add_library(hypercube_c SHARED
    src/hypercube_c.cpp
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

    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(hypercube_ops PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(hypercube_ops PROPERTIES
        PREFIX ""
        OUTPUT_NAME "hypercube_ops"
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

# ============================================================================
# PostgreSQL Tools (require libpq client library)
# ============================================================================

# Try to find libpq via vcpkg or system
find_package(PostgreSQL QUIET)
if(PostgreSQL_FOUND)
    message(STATUS "Found PostgreSQL client library: ${PostgreSQL_LIBRARIES}")
    
    add_executable(seed_atoms_parallel src/seed_atoms_parallel.cpp)
    target_include_directories(seed_atoms_parallel PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(seed_atoms_parallel hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    add_executable(cpe_ingest src/cpe_ingest.cpp)
    target_include_directories(cpe_ingest PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(cpe_ingest hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    add_executable(sequitur_ingest src/sequitur_ingest.cpp)
    target_include_directories(sequitur_ingest PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(sequitur_ingest hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    add_executable(ingest_safetensor src/ingest_safetensor.cpp)
    target_include_directories(ingest_safetensor PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(ingest_safetensor hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    add_executable(extract_embeddings src/extract_embeddings.cpp)
    target_include_directories(extract_embeddings PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(extract_embeddings hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    add_executable(semantic_ingest src/semantic_ingest.cpp)
    target_include_directories(semantic_ingest PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(semantic_ingest hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)

    add_executable(vocabulary_ingest src/vocabulary_ingest.cpp)
    target_include_directories(vocabulary_ingest PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(vocabulary_ingest hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
    
    # Integration tests with libpq
    add_executable(test_integration tests/test_integration.cpp)
    target_include_directories(test_integration PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(test_integration hypercube_core ${PostgreSQL_LIBRARIES})
    add_test(NAME IntegrationTest COMMAND test_integration)
    
    add_executable(test_query_api tests/test_query_api.cpp)
    target_include_directories(test_query_api PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(test_query_api hypercube_core ${PostgreSQL_LIBRARIES})
    add_test(NAME QueryAPITest COMMAND test_query_api)
else()
    message(STATUS "PostgreSQL client library not found - skipping tool builds")
    message(STATUS "To build tools, install libpq via: vcpkg install libpq:x64-windows")
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
message(STATUS "")
