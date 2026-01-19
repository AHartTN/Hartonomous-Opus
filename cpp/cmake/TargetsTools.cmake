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
    add_test(NAME IntegrationTest COMMAND $<TARGET_FILE:test_integration>)
    set_tests_properties(IntegrationTest PROPERTIES
        ENVIRONMENT "HC_DB_NAME=hypercube;HC_DB_USER=postgres;HC_DB_PASS=postgres;HC_DB_HOST=HART-SERVER;HC_DB_PORT=5432"
    )

    add_executable(test_query_api tests/test_query_api.cpp)
    target_include_directories(test_query_api PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(test_query_api hypercube_core ${PostgreSQL_LIBRARIES})
    add_test(NAME QueryAPITest COMMAND $<TARGET_FILE:test_query_api>)
    set_tests_properties(QueryAPITest PROPERTIES
        ENVIRONMENT "HC_DB_NAME=hypercube;HC_DB_USER=postgres;HC_DB_PASS=postgres;HC_DB_HOST=HART-SERVER;HC_DB_PORT=5432"
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