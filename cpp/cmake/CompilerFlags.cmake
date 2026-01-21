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
# Runtime-Only CPU Feature Detection
# ============================================================================
#
# All CPU feature detection now occurs at runtime during application startup.
# No build-time CPU feature detection or SIMD-specific compiler flags are used.
# This enables universal binary deployment across different CPU architectures.

# ============================================================================
# Runtime CPU Detection - Compile-time flags removed
# ============================================================================
#
# The build system now compiles universal binaries without SIMD-specific flags.
# CPU feature detection and dispatch happens entirely at runtime.
#
# This simplifies the build process and ensures compatibility across different
# CPU architectures while maintaining optimal performance through runtime dispatch.
#

message(STATUS "Runtime dispatch enabled - building universal binary")
message(STATUS "CPU feature detection will occur at application startup")

# ============================================================================
# Universal Binary Compilation
# ============================================================================
#
# Build universal binaries with runtime SIMD dispatch.
# Compile with x86-64 to allow all SIMD intrinsics for runtime dispatch,
# but detect CPU capabilities at startup for safe execution.
#
if(NOT MSVC)
    # For GCC/Clang: use x86-64 to allow SIMD intrinsics while maintaining compatibility
    check_cxx_compiler_flag("-march=x86-64" COMPILER_SUPPORTS_X86_64)
    if(COMPILER_SUPPORTS_X86_64)
        set(HYPERCUBE_SIMD_FLAGS "-march=native")
        message(STATUS "Native binary: Using native CPU for SIMD intrinsic availability")
    else()
        set(HYPERCUBE_SIMD_FLAGS "")
        message(STATUS "Universal binary: Using compiler default")
    endif()
else()
    # For MSVC: allow SIMD intrinsics through compiler defaults
    set(HYPERCUBE_SIMD_FLAGS "")
    message(STATUS "Universal binary: Using MSVC defaults (SIMD intrinsics available)")
endif()

# ============================================================================
# Flag bundles for targets
# ============================================================================
# We construct these:
#   HYPERCUBE_OPT_FLAGS     : optimization + NDEBUG
#   HYPERCUBE_DEBUG_FLAGS   : debug + sanitizer (where supported)
#   HYPERCUBE_WARNING_FLAGS : cross-platform warnings
# Targets will use:
#   $<$<CONFIG:Release>:${HYPERCUBE_FLAGS_RELEASE}>
#   $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_FLAGS_DEBUG}>
# ============================================================================
set(HYPERCUBE_OPT_FLAGS     "")
set(HYPERCUBE_DEBUG_FLAGS   "")
set(HYPERCUBE_WARNING_FLAGS "")

if(MSVC)
    # MSVC: safe baseline for universal binary
    set(HYPERCUBE_OPT_FLAGS     /O2 /DNDEBUG /fp:precise)
    set(HYPERCUBE_DEBUG_FLAGS   /Od /DDEBUG /Zi)
    set(HYPERCUBE_WARNING_FLAGS /W3 /EHsc /permissive- /DNOMINMAX)

    # Suppress common warnings globally (you can localize later if desired)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4996>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4005>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4200>
        $<$<COMPILE_LANGUAGE:C>:/wd4244>
        $<$<COMPILE_LANGUAGE:C>:/wd4267>
        $<$<COMPILE_LANGUAGE:C>:/wd4996>
        $<$<COMPILE_LANGUAGE:C>:/wd4005>
        $<$<COMPILE_LANGUAGE:C>:/wd4200>
    )
else()
    # GCC / Clang - universal binary compilation
    set(HYPERCUBE_OPT_FLAGS     -O3 -DNDEBUG -ffast-math)
    set(HYPERCUBE_DEBUG_FLAGS   -g -O0 -DDEBUG)
    set(HYPERCUBE_WARNING_FLAGS -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)

    # Sanitizers for Debug where available (non-MSVC, and not Clang on Windows)
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        list(APPEND HYPERCUBE_DEBUG_FLAGS -fsanitize=address,undefined)
    endif()
endif()

# Final bundles for targets (as CMake lists for proper expansion)
set(HYPERCUBE_FLAGS_RELEASE
    ${HYPERCUBE_OPT_FLAGS} ${HYPERCUBE_WARNING_FLAGS} ${HYPERCUBE_SIMD_FLAGS}
)
set(HYPERCUBE_FLAGS_DEBUG
    ${HYPERCUBE_DEBUG_FLAGS} ${HYPERCUBE_WARNING_FLAGS} ${HYPERCUBE_SIMD_FLAGS}
)

# ============================================================================
# Output layout and RPATH
# ============================================================================

# Unified build tree layout
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>")

if(UNIX AND NOT APPLE)
    # RPATH behavior: don't skip build rpath, don't hardcode install rpath here.
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    # Use relative RPATH for both build and install.
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
    set(CMAKE_INSTALL_RPATH_USE_ORIGIN ON)
endif()

# ============================================================================
# Build Summary
# ============================================================================

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Available configurations: ${CMAKE_CONFIGURATION_TYPES}")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Runtime dispatch: ENABLED (universal binary)")
message(STATUS "SIMD optimization: Runtime detection at startup")
