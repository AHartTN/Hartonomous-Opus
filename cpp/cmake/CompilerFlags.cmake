# ============================================================================
# Cross-Platform Compiler Configuration Module
# ============================================================================

include(CheckCXXCompilerFlag)

# ============================================================================
# Basic Compiler Standards and Position Independent Code
# ============================================================================

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Warning and Error Settings
# ============================================================================

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        add_compile_options(/W3 /permissive-)
    else()
        # More permissive warnings for cross-platform compatibility
        # Only treat critical warnings as errors
        add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable)
    endif()
endif()

# ============================================================================
# SIMD and CPU Feature Detection
# ============================================================================

# Try to run a test program to detect CPU features at configure time
try_run(
    CPU_FEATURES_RUN_RESULT
    CPU_FEATURES_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpu_features_test.cpp
    CMAKE_FLAGS -DCMAKE_CXX_STANDARD=23
    RUN_OUTPUT_VARIABLE CPU_FEATURES_OUTPUT
)

# Parse CPU features from test output
if(CPU_FEATURES_COMPILE_RESULT AND CPU_FEATURES_RUN_RESULT EQUAL 0)
    string(REGEX MATCH "AVX2:([01])" AVX2_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX512F:([01])" AVX512F_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX512DQ:([01])" AVX512DQ_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX512BW:([01])" AVX512BW_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX512VL:([01])" AVX512VL_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "FMA3:([01])" FMA3_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX_VNNI:([01])" AVX_VNNI_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "BMI2:([01])" BMI2_MATCH ${CPU_FEATURES_OUTPUT})

    # Set feature flags based on detection
    macro(set_cpu_feature FEATURE_VAR MATCH_VAR)
        if(${MATCH_VAR})
            string(REGEX REPLACE "${FEATURE_VAR}:([01])" "\\1" ${FEATURE_VAR}_VALUE ${${MATCH_VAR}})
            if(${FEATURE_VAR}_VALUE STREQUAL "1")
                set(HAS_${FEATURE_VAR} ON)
                message(STATUS "CPU Feature: ${FEATURE_VAR} detected")
            endif()
        endif()
    endmacro()

    set_cpu_feature(AVX2 AVX2_MATCH)
    set_cpu_feature(AVX512F AVX512F_MATCH)
    set_cpu_feature(AVX512DQ AVX512DQ_MATCH)
    set_cpu_feature(AVX512BW AVX512BW_MATCH)
    set_cpu_feature(AVX512VL AVX512VL_MATCH)
    set_cpu_feature(FMA3 FMA3_MATCH)
    set_cpu_feature(AVX_VNNI AVX_VNNI_MATCH)
    set_cpu_feature(BMI2 BMI2_MATCH)
else()
    # Fallback: Use compiler flag detection when runtime test fails
    message(STATUS "CPU feature runtime detection failed - using compiler-based detection")
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(HAS_AVX ON)
        message(STATUS "SIMD: Using -march=native (fallback detection)")
    else()
        set(HAS_AVX OFF)
        message(STATUS "SIMD: Generic optimizations only")
    endif()
endif()

# Set HAS_AVX for backward compatibility
if(HAS_AVX2 OR HAS_AVX512F)
    set(HAS_AVX ON)
endif()

# ============================================================================
# AVX512 Feature Support Detection
# ============================================================================

# Check compiler support for AVX512 flags
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
check_cxx_compiler_flag("-mavx512bw" COMPILER_SUPPORTS_AVX512BW)
check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)
check_cxx_compiler_flag("-mavx512vnni" COMPILER_SUPPORTS_AVX512VNNI)

# Set feature-specific flags only if both CPU and compiler support them
function(set_avx512_flag FLAG_VAR FLAG_NAME)
    if(HAS_AVX512F AND COMPILER_SUPPORTS_${FLAG_VAR})
        set(${FLAG_NAME} "-m${FLAG_VAR}" PARENT_SCOPE)
        message(STATUS "${FLAG_VAR}: Enabled (CPU + compiler support)")
    else()
        set(${FLAG_NAME} "" PARENT_SCOPE)
        if(NOT HAS_AVX512F)
            message(STATUS "${FLAG_VAR}: Disabled (CPU does not support)")
        elseif(NOT COMPILER_SUPPORTS_${FLAG_VAR})
            message(STATUS "${FLAG_VAR}: Disabled (compiler does not support)")
        endif()
    endif()
endfunction()

set_avx512_flag("AVX512F" AVX512F_FLAG)
set_avx512_flag("AVX512DQ" AVX512DQ_FLAG)
set_avx512_flag("AVX512BW" AVX512BW_FLAG)
set_avx512_flag("AVX512VL" AVX512VL_FLAG)

if(HAS_AVX512F AND HAS_AVX_VNNI AND COMPILER_SUPPORTS_AVX512VNNI)
    set(AVX512VNNI_FLAG "-mavx512vnni")
else()
    set(AVX512VNNI_FLAG "")
endif()

# ============================================================================
# Platform-Specific Compiler Flags
# ============================================================================

if(MSVC)
    # MSVC-specific optimizations and warnings
    set(COMPILER_OPT_FLAGS "/O2 /DNDEBUG /fp:precise")
    set(COMPILER_DEBUG_FLAGS "/Od /DDEBUG /Zi")
    set(COMPILER_WARNING_FLAGS "/W3 /EHsc /permissive- /DNOMINMAX")
    set(COMPILER_AVX_FLAGS "/arch:AVX2")

    # Suppress common warnings
    add_compile_options(/wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
else()
    # GCC/Clang optimizations and warnings
    set(COMPILER_OPT_FLAGS "-O3 -DNDEBUG -ffast-math")
    set(COMPILER_DEBUG_FLAGS "-g -O0 -DDEBUG")
    set(COMPILER_WARNING_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare")

    # Start with AVX2 as baseline, add AVX512 features conditionally
    set(COMPILER_AVX_FLAGS "-mavx2 -mfma")

    # Debug sanitizers
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(COMPILER_DEBUG_FLAGS "${COMPILER_DEBUG_FLAGS} -fsanitize=address,undefined")
    endif()
endif()

# ============================================================================
# Apply Compiler Flags Based on Build Type
# ============================================================================

set(CMAKE_CXX_FLAGS_RELEASE "${COMPILER_OPT_FLAGS} ${COMPILER_WARNING_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${COMPILER_OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${COMPILER_DEBUG_FLAGS} ${COMPILER_WARNING_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${COMPILER_DEBUG_FLAGS}")

# Apply SIMD flags if AVX2 is supported
if(HAS_AVX2)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMPILER_AVX_FLAGS}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${COMPILER_AVX_FLAGS}")

    # Add AVX512 flags only if supported by both CPU and compiler
    foreach(flag AVX512F AVX512DQ AVX512BW AVX512VL AVX_VNNI)
        if(HAS_${flag} AND ${flag}_FLAG)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${${flag}_FLAG}")
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${${flag}_FLAG}")
        endif()
    endforeach()
endif()

# ============================================================================
# RPATH Configuration
# ============================================================================

# RPATH settings - use install path even during build
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

if(WIN32)
    # Windows doesn't use RPATH, but we can set DLL search paths
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
else()
    # Unix-like systems use RPATH
    if(BUILD_PG_EXTENSION)
        set(CMAKE_INSTALL_RPATH "/usr/lib/postgresql/18/lib")
    endif()
endif()

# ============================================================================
# Feature Summary
# ============================================================================

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

if(HAS_AVX2)
    message(STATUS "CPU Features: AVX2 detected")
endif()
if(HAS_AVX512F)
    message(STATUS "CPU Features: AVX-512 Foundation detected")
endif()
if(HAS_AVX512DQ)
    message(STATUS "CPU Features: AVX-512 Doubleword/Quadword detected")
endif()
if(HAS_AVX512BW)
    message(STATUS "CPU Features: AVX-512 Byte/Word detected")
endif()
if(HAS_AVX512VL)
    message(STATUS "CPU Features: AVX-512 Vector Length detected")
endif()
if(HAS_FMA3)
    message(STATUS "CPU Features: FMA3 detected")
endif()
if(HAS_AVX_VNNI)
    message(STATUS "CPU Features: AVX-VNNI detected")
endif()
if(HAS_BMI2)
    message(STATUS "CPU Features: BMI2 detected")
endif()

message(STATUS "SIMD support: ${HAS_AVX}")