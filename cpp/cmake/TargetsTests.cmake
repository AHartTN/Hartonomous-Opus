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

add_executable(test_simd_dispatch tests/test_simd_dispatch.cpp)
target_link_libraries(test_simd_dispatch hypercube_core)
add_test(NAME SimdDispatchTest COMMAND test_simd_dispatch)

# ============================================================================
# Google Test Suite
# ============================================================================

if(GTest_FOUND)
    message(STATUS "Found Google Test: ${GTEST_INCLUDE_DIRS}")

    # Core unit tests (no PostgreSQL required)
    # Note: test_simd_dispatch.cpp has its own main() and is built as a standalone exe
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