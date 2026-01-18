# Tests CMakeLists.txt
find_package(GTest REQUIRED)

# Test executable
add_executable(hartonomous-tests
    test_main.cpp
    test_config.cpp
    test_embedding_client.cpp
    test_reranking_client.cpp
    test_generative_client.cpp
    test_qdrant_client.cpp
    test_orchestrator.cpp
    test_http_server.cpp
    test_integration.cpp
)

# Link test libraries
target_link_libraries(hartonomous-tests
    GTest::gtest_main
    ${Boost_LIBRARIES}
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    prometheus-cpp::prometheus-cpp
)

# Include directories for tests
target_include_directories(hartonomous-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${Boost_INCLUDE_DIRS}
)

# Add test
add_test(NAME hartonomous-tests COMMAND hartonomous-tests)