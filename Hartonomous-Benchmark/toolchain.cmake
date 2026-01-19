# Platform-agnostic toolchain file for cross-platform compatibility

# Detect platform
if(WIN32)
    set(PLATFORM "windows")
elseif(UNIX AND NOT APPLE)
    set(PLATFORM "linux")
elseif(APPLE)
    set(PLATFORM "macos")
else()
    set(PLATFORM "unknown")
endif()

message(STATUS "Detected platform: ${PLATFORM}")

# Compiler settings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /Oy- /fp:fast")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# Threading: Let CMake detect automatically, no forced POSIX libs on Windows
if(WIN32)
    # Ensure no POSIX threading forced
    set(THREADS_PREFER_PTHREAD_FLAG OFF CACHE BOOL "" FORCE)
endif()

# Regex: Let libraries detect automatically
# No forced HAVE_*_REGEX flags

# VCPKG integration if available
if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    include("$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
elseif(EXISTS "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    include("D:/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()