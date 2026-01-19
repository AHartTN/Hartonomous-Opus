# Math library CMakeLists.txt
# This directory contains mathematical utilities and libraries

# Create interface library for headers
add_library(math INTERFACE)
target_include_directories(math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Link dependencies
target_link_libraries(math INTERFACE Eigen3::Eigen)
if(MKL_FOUND)
    target_link_libraries(math INTERFACE ${MKL_LIBRARIES})
endif()

# Set C++23 standard
target_compile_features(math INTERFACE cxx_std_23)