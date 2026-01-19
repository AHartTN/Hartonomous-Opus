add_executable(hardware_check hardware.cpp)
if(WIN32)
    target_link_libraries(hardware_check msvcrt)
endif()

add_subdirectory(benchmark)
add_subdirectory(timing)
add_subdirectory(results)
add_subdirectory(math)