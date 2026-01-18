# CMakeLists.txt for Hartonomous Orchestrator
cmake_minimum_required(VERSION 3.20)
project(hartonomous-orchestrator VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set compiler options
if(MSVC)
    # MSVC specific options
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    # GCC/Clang specific options
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

# Enable testing
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
endif()

# Find required packages
find_package(Boost REQUIRED COMPONENTS beast json system thread)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

# Optional packages
find_package(prometheus-cpp QUIET)
find_package(GTest QUIET)

# Add subdirectories
add_subdirectory(src)
if(BUILD_TESTS AND GTest_FOUND)
    add_subdirectory(tests)
endif()

# Main executable is defined in src/CMakeLists.txt

# Set up install rules
install(TARGETS hartonomous-orchestrator
    RUNTIME DESTINATION bin
)

install(FILES config.yaml
    DESTINATION config
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Add tests conditionally
if(BUILD_TESTS AND GTest_FOUND)
    add_test(NAME hartonomous-tests COMMAND hartonomous-tests)
endif()

# Add documentation target
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        # Configure Doxyfile
        configure_file(doxygen.config.in doxygen.config @ONLY)
        add_custom_target(docs
            ${DOXYGEN_EXECUTABLE} doxygen.config
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
    endif()
endif()