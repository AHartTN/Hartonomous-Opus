# ============================================================================
# Compiler configuration and SIMD/CPU feature negotiation
# ============================================================================

include(CheckCXXCompilerFlag)

# ----------------------------------------------------------------------------
# Language standards
# ----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# PIC for all targets (static + shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------------------------
# Build type handling (single- vs multi-config generators)
# ----------------------------------------------------------------------------

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------------------------------------------------
# Baseline warnings (more in Debug, modest in Release)
# ----------------------------------------------------------------------------

if(MSVC)
    add_compile_options(
        $<$<CONFIG:Debug>:/W3>
        $<$<CONFIG:Debug>:/permissive->
    )
else()
    add_compile_options(
        $<$<CONFIG:Debug>:-Wall>
        $<$<CONFIG:Debug>:-Wextra>
        $<$<CONFIG:Debug>:-Wno-unused-parameter>
        $<$<CONFIG:Debug>:-Wno-unused-variable>
    )
endif()

# ============================================================================
# CPU feature runtime detection
# ============================================================================

# We expect a small C++ program at: ${CMAKE_SOURCE_DIR}/cmake/cpu_features_test.cpp
# that prints lines like:
#   AVX2:1
#   AVX512F:0
#   AVX_VNNI:1
#   BMI2:1
set(HYPERCUBE_CPU_FEATURE_TEST "${CMAKE_SOURCE_DIR}/cmake/cpu_features_test.cpp")

# Initialize all feature flags to OFF
set(HAS_AVX        OFF)
set(HAS_AVX2       OFF)
set(HAS_AVX512F    OFF)
set(HAS_AVX512DQ   OFF)
set(HAS_AVX512BW   OFF)
set(HAS_AVX512VL   OFF)
set(HAS_FMA3       OFF)
set(HAS_AVX_VNNI   OFF)  # This is generic AVX-VNNI support (256- or 512-bit)
set(HAS_BMI2       OFF)

set(CPU_FEATURES_COMPILE_RESULT 0)
set(CPU_FEATURES_RUN_RESULT 1)
set(CPU_FEATURES_OUTPUT "")

if(EXISTS "${HYPERCUBE_CPU_FEATURE_TEST}")
    try_run(
        CPU_FEATURES_RUN_RESULT
        CPU_FEATURES_COMPILE_RESULT
        "${CMAKE_BINARY_DIR}"
        "${HYPERCUBE_CPU_FEATURE_TEST}"
        CMAKE_FLAGS -DCMAKE_CXX_STANDARD=23
        RUN_OUTPUT_VARIABLE CPU_FEATURES_OUTPUT
    )
else()
    message(STATUS "CPU feature test source not found: ${HYPERCUBE_CPU_FEATURE_TEST}")
endif()

if(CPU_FEATURES_COMPILE_RESULT AND CPU_FEATURES_RUN_RESULT EQUAL 0)
    # Parse runtime output
    string(REGEX MATCH "AVX2:([01])"      AVX2_MATCH      "${CPU_FEATURES_OUTPUT}")
    string(REGEX MATCH "AVX512F:([01])"   AVX512F_MATCH   "${CPU_FEATURES_OUTPUT}")
    string(REGEX MATCH "AVX512DQ:([01])"  AVX512DQ_MATCH  "${CPU_FEATURES_OUTPUT}")
    string(REGEX MATCH "AVX512BW:([01])"  AVX512BW_MATCH  "${CPU_FEATURES_OUTPUT}")
    string(REGEX MATCH "AVX512VL:([01])"  AVX512VL_MATCH  "${CPU_FEATURES_OUTPUT}")
    string(REGEX MATCH "FMA3:([01])"      FMA3_MATCH      "${CPU_FEATURES_OUTPUT}")
    string(REGEX MATCH "AVX_VNNI:([01])"  AVX_VNNI_MATCH  "${CPU_FEATURES_OUTPUT}")
    string(REGEX MATCH "BMI2:([01])"      BMI2_MATCH      "${CPU_FEATURES_OUTPUT}")

    macro(_hc_set_cpu_feature FEATURE_VAR MATCH_VAR)
        if(${MATCH_VAR})
            string(REGEX REPLACE "${FEATURE_VAR}:([01])" "\\1" _VAL "${${MATCH_VAR}}")
            if(_VAL STREQUAL "1")
                set(HAS_${FEATURE_VAR} ON)
                message(STATUS "CPU Feature: ${FEATURE_VAR} detected")
            endif()
        endif()
    endmacro()

    _hc_set_cpu_feature(AVX2      AVX2_MATCH)
    _hc_set_cpu_feature(AVX512F   AVX512F_MATCH)
    _hc_set_cpu_feature(AVX512DQ  AVX512DQ_MATCH)
    _hc_set_cpu_feature(AVX512BW  AVX512BW_MATCH)
    _hc_set_cpu_feature(AVX512VL  AVX512VL_MATCH)
    _hc_set_cpu_feature(FMA3      FMA3_MATCH)
    _hc_set_cpu_feature(AVX_VNNI  AVX_VNNI_MATCH)
    _hc_set_cpu_feature(BMI2      BMI2_MATCH)
else()
    # Fallback: no runtime info; rely on compiler capability only
    message(STATUS "CPU feature runtime detection failed - using compiler-based fallback")
    if(NOT MSVC)
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            set(HAS_AVX ON)
            message(STATUS "SIMD: assuming -march=native is available (fallback)")
        else()
            message(STATUS "SIMD: generic baseline (no AVX assumptions)")
        endif()
    else()
        message(STATUS "SIMD: MSVC baseline; AVX features will be conservative")
    endif()
endif()

# Backward-compatible HAS_AVX: ON if we have AVX2 or any AVX512
if(HAS_AVX2 OR HAS_AVX512F)
    set(HAS_AVX ON)
endif()

# ============================================================================
# Compiler AVX/AVX512/VNNI flag detection (GCC/Clang only)
# ============================================================================

# These flags are GCC/Clang-specific; MSVC uses /arch:*
if(NOT MSVC)
    # Baseline SIMD flags
    check_cxx_compiler_flag("-mavx2"      COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma"       COMPILER_SUPPORTS_FMA)

    # AVX-512 feature flags
    check_cxx_compiler_flag("-mavx512f"    COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("-mavx512dq"   COMPILER_SUPPORTS_AVX512DQ)
    check_cxx_compiler_flag("-mavx512bw"   COMPILER_SUPPORTS_AVX512BW)
    check_cxx_compiler_flag("-mavx512vl"   COMPILER_SUPPORTS_AVX512VL)

    # VNNI variants:
    #  - AVX-512 VNNI (requires AVX-512F):   -mavx512vnni
    #  - AVX-VNNI (256-bit, no AVX-512 req): -mavxvnni
    check_cxx_compiler_flag("-mavx512vnni" COMPILER_SUPPORTS_AVX512VNNI)
    check_cxx_compiler_flag("-mavxvnni"    COMPILER_SUPPORTS_AVXVNNI)
else()
    set(COMPILER_SUPPORTS_AVX2          OFF)
    set(COMPILER_SUPPORTS_FMA           OFF)
    set(COMPILER_SUPPORTS_AVX512F       OFF)
    set(COMPILER_SUPPORTS_AVX512DQ      OFF)
    set(COMPILER_SUPPORTS_AVX512BW      OFF)
    set(COMPILER_SUPPORTS_AVX512VL      OFF)
    set(COMPILER_SUPPORTS_AVX512VNNI    OFF)
    set(COMPILER_SUPPORTS_AVXVNNI       OFF)
endif()

# ============================================================================
# Flag bundles for targets
# ============================================================================
# We construct these:
#   HYPERCUBE_OPT_FLAGS     : optimization + NDEBUG
#   HYPERCUBE_DEBUG_FLAGS   : debug + sanitizer (where supported)
#   HYPERCUBE_WARNING_FLAGS : cross-platform warnings
#   HYPERCUBE_AVX_FLAGS     : SIMD flags (AVX2 / AVX512 / VNNI)
# Targets will use:
#   $<$<CONFIG:Release>:${HYPERCUBE_FLAGS_RELEASE}>
#   $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_FLAGS_DEBUG}>
# ============================================================================
set(HYPERCUBE_OPT_FLAGS     "")
set(HYPERCUBE_DEBUG_FLAGS   "")
set(HYPERCUBE_WARNING_FLAGS "")
set(HYPERCUBE_AVX_FLAGS     "")

if(MSVC)
    # MSVC: safe baseline
    set(HYPERCUBE_OPT_FLAGS     /O2 /DNDEBUG /fp:precise)
    set(HYPERCUBE_DEBUG_FLAGS   /Od /DDEBUG /Zi)
    set(HYPERCUBE_WARNING_FLAGS /W3 /EHsc /permissive- /DNOMINMAX)

    # AVX2 is /arch:AVX2; AVX-512 is limited and inconsistent, so we stay at AVX2.
    if(HAS_AVX2)
        set(HYPERCUBE_AVX_FLAGS "/arch:AVX2")
    endif()

    # Suppress common warnings globally (you can localize later if desired)
    add_compile_options(/wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
else()
    # GCC / Clang
    set(HYPERCUBE_OPT_FLAGS     -O3 -DNDEBUG -ffast-math)
    set(HYPERCUBE_DEBUG_FLAGS   -g -O0 -DDEBUG)
    set(HYPERCUBE_WARNING_FLAGS -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)

    # SIMD flags: start from AVX2 if CPU & compiler support it
    if(HAS_AVX2 AND COMPILER_SUPPORTS_AVX2)
        set(HYPERCUBE_AVX_FLAGS "-mavx2")

        if(HAS_FMA3 AND COMPILER_SUPPORTS_FMA)
            string(APPEND HYPERCUBE_AVX_FLAGS " -mfma")
        endif()

        # Add AVX-512 flags only if both CPU & compiler support them
        if(HAS_AVX512F AND COMPILER_SUPPORTS_AVX512F)
            string(APPEND HYPERCUBE_AVX_FLAGS " -mavx512f")

            if(HAS_AVX512DQ AND COMPILER_SUPPORTS_AVX512DQ)
                string(APPEND HYPERCUBE_AVX_FLAGS " -mavx512dq")
            endif()
            if(HAS_AVX512BW AND COMPILER_SUPPORTS_AVX512BW)
                string(APPEND HYPERCUBE_AVX_FLAGS " -mavx512bw")
            endif()
            if(HAS_AVX512VL AND COMPILER_SUPPORTS_AVX512VL)
                string(APPEND HYPERCUBE_AVX_FLAGS " -mavx512vl")
            endif()
        endif()

        # AVX-VNNI handling:
        # - If HAS_AVX_VNNI and AVX-512F, prefer AVX-512 VNNI if available
        # - Otherwise, prefer AVX-VNNI (256-bit) flag if supported
        if(HAS_AVX_VNNI)
            if(HAS_AVX512F AND COMPILER_SUPPORTS_AVX512VNNI)
                string(APPEND HYPERCUBE_AVX_FLAGS " -mavx512vnni")
                message(STATUS "SIMD: enabling AVX-512 VNNI (-mavx512vnni)")
            elseif(COMPILER_SUPPORTS_AVXVNNI)
                string(APPEND HYPERCUBE_AVX_FLAGS " -mavxvnni")
                message(STATUS "SIMD: enabling AVX-VNNI (-mavxvnni)")
            else()
                message(STATUS "SIMD: AVX_VNNI detected but compiler lacks -mavx512vnni/-mavxvnni")
            endif()
        endif()
    elseif(HAS_AVX AND COMPILER_SUPPORTS_MARCH_NATIVE)
        # Fallback: runtime detection failed but -march=native is supported
        # Use -march=native to let the compiler auto-detect CPU features
        set(HYPERCUBE_AVX_FLAGS "-march=native")
        message(STATUS "SIMD: using -march=native (fallback mode)")
    endif()

    # Sanitizers for Debug where available (non-MSVC, and not Clang on Windows)
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        list(APPEND HYPERCUBE_DEBUG_FLAGS -fsanitize=address,undefined)
    endif()
endif()

# Final bundles for targets (as CMake lists for proper expansion)
set(HYPERCUBE_FLAGS_RELEASE
    ${HYPERCUBE_OPT_FLAGS} ${HYPERCUBE_WARNING_FLAGS} ${HYPERCUBE_AVX_FLAGS}
)
set(HYPERCUBE_FLAGS_DEBUG
    ${HYPERCUBE_DEBUG_FLAGS} ${HYPERCUBE_WARNING_FLAGS}
)

# ============================================================================
# Output layout and RPATH
# ============================================================================

# Unified build tree layout
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(UNIX AND NOT APPLE)
    # RPATH behavior: don't skip build rpath, don't hardcode install rpath here.
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    # If PG extensions are built, default to relative RPATH unless user overrides.
    if(BUILD_PG_EXTENSION AND NOT CMAKE_INSTALL_RPATH)
        set(CMAKE_INSTALL_RPATH "$ORIGIN")
    endif()
endif()

# ============================================================================
# Feature summary
# ============================================================================

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Available configurations: ${CMAKE_CONFIGURATION_TYPES}")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

if(HAS_AVX2)
    message(STATUS "CPU Features: AVX2 detected")
endif()
if(HAS_AVX512F)
    message(STATUS "CPU Features: AVX-512F detected")
endif()
if(HAS_AVX512DQ)
    message(STATUS "CPU Features: AVX-512DQ detected")
endif()
if(HAS_AVX512BW)
    message(STATUS "CPU Features: AVX-512BW detected")
endif()
if(HAS_AVX512VL)
    message(STATUS "CPU Features: AVX-512VL detected")
endif()
if(HAS_FMA3)
    message(STATUS "CPU Features: FMA3 detected")
endif()
if(HAS_AVX_VNNI)
    message(STATUS "CPU Features: AVX_VNNI detected (may be AVX-512 VNNI or AVX-VNNI)")
endif()
if(HAS_BMI2)
    message(STATUS "CPU Features: BMI2 detected")
endif()

message(STATUS "SIMD baseline HAS_AVX: ${HAS_AVX}")
message(STATUS "HYPERCUBE_AVX_FLAGS:   ${HYPERCUBE_AVX_FLAGS}")
