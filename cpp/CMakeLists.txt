cmake_minimum_required(VERSION 3.20)
project(hypercube VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Cross-platform threading support
find_package(Threads REQUIRED)

# Compiler flags - platform specific
if(MSVC)
    # Microsoft Visual C++
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /W4 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /DDEBUG /W4 /Zi /EHsc /RTC1")
    # Disable some noisy warnings
    add_compile_options(/wd4244 /wd4267 /wd4996)
else()
    # GCC/Clang
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
    # Only enable sanitizers on non-Windows (MSVC doesn't support them the same way)
    if(NOT WIN32)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
    endif()
endif()

# Find PostgreSQL
find_package(PostgreSQL)
if(PostgreSQL_FOUND)
    execute_process(
        COMMAND pg_config --pkglibdir
        OUTPUT_VARIABLE POSTGRESQL_PKGLIBDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND pg_config --sharedir
        OUTPUT_VARIABLE POSTGRESQL_SHAREDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND pg_config --includedir-server
        OUTPUT_VARIABLE POSTGRESQL_SERVER_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "PostgreSQL found: ${PostgreSQL_VERSION_STRING}")
    message(STATUS "PostgreSQL pkglibdir: ${POSTGRESQL_PKGLIBDIR}")
    message(STATUS "PostgreSQL server include: ${POSTGRESQL_SERVER_INCLUDE}")
    set(BUILD_PG_EXTENSION ON)
else()
    message(WARNING "PostgreSQL not found - building without extension support")
    set(BUILD_PG_EXTENSION OFF)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core library (no PostgreSQL dependency)
add_library(hypercube_core STATIC
    src/hilbert.cpp
    src/coordinates.cpp
    src/blake3_pg.cpp
)
target_include_directories(hypercube_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# PostgreSQL extension (only if PostgreSQL found)
if(BUILD_PG_EXTENSION)
    add_library(hypercube SHARED
        src/hypercube.cpp
    )
    
    target_include_directories(hypercube PRIVATE
        ${PostgreSQL_INCLUDE_DIRS}
        ${POSTGRESQL_SERVER_INCLUDE}
    )
    
    target_link_libraries(hypercube
        hypercube_core
    )
    
    target_compile_definitions(hypercube PRIVATE
        HYPERCUBE_VERSION="${PROJECT_VERSION}"
    )
    
    # Remove 'lib' prefix for PostgreSQL extensions
    set_target_properties(hypercube PROPERTIES PREFIX "")
    
    # Installation
    install(TARGETS hypercube
        LIBRARY DESTINATION ${POSTGRESQL_PKGLIBDIR}
    )
    
    install(FILES
        sql/hypercube--1.0.sql
        hypercube.control
        DESTINATION ${POSTGRESQL_SHAREDIR}/extension
    )
    
    # Semantic operations extension (high-performance AI/ML queries)
    add_library(semantic_ops SHARED
        src/semantic_ops.cpp
    )
    
    target_include_directories(semantic_ops PRIVATE
        ${PostgreSQL_INCLUDE_DIRS}
        ${POSTGRESQL_SERVER_INCLUDE}
    )
    
    target_link_libraries(semantic_ops
        hypercube_core
    )
    
    set_target_properties(semantic_ops PROPERTIES PREFIX "")
    
    install(TARGETS semantic_ops
        LIBRARY DESTINATION ${POSTGRESQL_PKGLIBDIR}
    )
    
    install(FILES
        sql/semantic_ops--1.0.sql
        sql/semantic_ops.control
        DESTINATION ${POSTGRESQL_SHAREDIR}/extension
    )
endif()

# Tests (always build)
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

# Integration tests (require PostgreSQL)
if(PostgreSQL_FOUND)
    add_executable(test_integration tests/test_integration.cpp)
    target_include_directories(test_integration PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(test_integration hypercube_core ${PostgreSQL_LIBRARIES})
    add_test(NAME IntegrationTest COMMAND test_integration)
    
    add_executable(test_query_api tests/test_query_api.cpp)
    target_include_directories(test_query_api PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(test_query_api hypercube_core ${PostgreSQL_LIBRARIES})
    add_test(NAME QueryAPITest COMMAND test_query_api)
endif()

# PostgreSQL tools (require libpq)
if(PostgreSQL_FOUND)
    # Parallel partitioned atom seeder (12 partitions, fastest)
    add_executable(seed_atoms_parallel src/seed_atoms_parallel.cpp)
    target_include_directories(seed_atoms_parallel PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(seed_atoms_parallel hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)

    # CPE ingester (binary tree merging)
    add_executable(cpe_ingest src/cpe_ingest.cpp)
    target_include_directories(cpe_ingest PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(cpe_ingest hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)

    # Sequitur grammar-based ingester (proper semantic composition)
    add_executable(sequitur_ingest src/sequitur_ingest.cpp)
    target_include_directories(sequitur_ingest PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(sequitur_ingest hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)

    # Safetensor package ingester (vocab, BPE merges, semantic edges)
    add_executable(ingest_safetensor src/ingest_safetensor.cpp)
    target_include_directories(ingest_safetensor PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(ingest_safetensor hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)

    # Embedding extractor (pairwise cosine similarity → sparse edges)
    add_executable(extract_embeddings src/extract_embeddings.cpp)
    target_include_directories(extract_embeddings PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(extract_embeddings hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)
endif()

# Summary
message(STATUS "")
message(STATUS "=== Hypercube Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PostgreSQL extension: ${BUILD_PG_EXTENSION}")
message(STATUS "")
