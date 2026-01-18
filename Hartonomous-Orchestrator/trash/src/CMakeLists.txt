# Source CMakeLists.txt

# Define the main executable
add_executable(hartonomous-orchestrator
    main.cpp
    config.cpp
    logging.cpp
    metrics.cpp
    http_client.cpp
    embedding_client.cpp
    reranking_client.cpp
    generative_client.cpp
    qdrant_client.cpp
    orchestrator.cpp
    http_server.cpp
)

# Link libraries
target_link_libraries(hartonomous-orchestrator
    ${Boost_LIBRARIES}
    spdlog::spdlog
    yaml-cpp::yaml-cpp
)

# Link prometheus-cpp if available
if(prometheus-cpp_FOUND)
    target_link_libraries(hartonomous-orchestrator prometheus-cpp::prometheus-cpp)
endif()

# Include directories
target_include_directories(hartonomous-orchestrator PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
)