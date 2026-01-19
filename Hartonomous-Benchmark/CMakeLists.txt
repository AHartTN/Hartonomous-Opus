cmake_minimum_required(VERSION 3.10)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
project(BenchmarkSuite VERSION 1.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
    if(WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmsvcrt")
    endif()
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /Oy- /fp:fast")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

include(FetchContent)

FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
  GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(hnswlib)

set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_WERROR OFF)
set(BENCHMARK_ENABLE_LIBPFM OFF)
set(HAVE_STD_REGEX TRUE CACHE BOOL "" FORCE)
set(HAVE_GNU_POSIX_REGEX TRUE CACHE BOOL "" FORCE)
set(HAVE_POSIX_REGEX TRUE CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG main
  PATCH_COMMAND powershell -File ${CMAKE_SOURCE_DIR}/patches/patch.ps1
)
FetchContent_MakeAvailable(benchmark)

set(MKL_ROOT "D:/Intel/oneAPI/mkl/latest")
find_package(MKL)
if(MKL_FOUND)
  add_definitions(-DHAVE_MKL)
  message(STATUS "MKL found")
else()
  message(WARNING "MKL not found. MKL-related benchmarks will be disabled.")
endif()

find_package(Eigen3)
if(NOT Eigen3_FOUND)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(eigen)
  set(Eigen3_FOUND TRUE)
endif()

add_subdirectory(benchmarks)
add_subdirectory(src)