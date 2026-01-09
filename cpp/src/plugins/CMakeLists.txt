# Plugin Build Configuration
# ==========================
#
# This builds the example geometric plugins as shared libraries
# that can be loaded dynamically by the main hypercube system.

cmake_minimum_required(VERSION 3.16)
project(hypercube_plugins)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Find hypercube core library (assuming it's built)
# This would need to be adjusted based on your build system
# find_library(HYPERCUBE_LIB hypercube_core PATHS ${CMAKE_BINARY_DIR}/../lib)

# Build semantic distance plugin
add_library(semantic_distance_plugin SHARED
    example_geometric_plugin.cpp
)

# Set plugin-specific compile definitions
target_compile_definitions(semantic_distance_plugin PRIVATE
    semantic_distance_plugin_EXPORTS
)

# Link against hypercube core (if available)
# target_link_libraries(semantic_distance_plugin PRIVATE ${HYPERCUBE_LIB})

# Set output properties
set_target_properties(semantic_distance_plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"  # Use .dll on Windows
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
)

# Build cosine distance plugin
add_library(cosine_distance_plugin SHARED
    example_geometric_plugin.cpp
)

target_compile_definitions(cosine_distance_plugin PRIVATE
    cosine_distance_plugin_EXPORTS
)

set_target_properties(cosine_distance_plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
)

# Install plugins
install(TARGETS semantic_distance_plugin cosine_distance_plugin
    LIBRARY DESTINATION lib/hypercube/plugins
)