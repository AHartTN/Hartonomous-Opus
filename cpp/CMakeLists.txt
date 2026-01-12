cmake_minimum_required(VERSION 3.24)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(hypercube VERSION 1.0.0 LANGUAGES CXX C)

# ============================================================================
# Enterprise-Grade Build Configuration
# ============================================================================

# Modern C++ standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build optimization
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable comprehensive warnings and treat as errors in Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        add_compile_options(/W4 /WX)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

# RPATH settings - use install path even during build
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/usr/lib/postgresql/18/lib")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Cross-platform threading support
find_package(Threads REQUIRED)

# ============================================================================
# Modern Dependency Management with FetchContent
# ============================================================================

include(FetchContent)

# OpenMP for parallel operations
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: Found (version ${OpenMP_CXX_VERSION})")
else()
    message(FATAL_ERROR "OpenMP is required for parallel operations")
endif()

# HNSWLIB disabled - causes AVX512 issues

# Eigen3 (required for linear algebra)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3: Found via system package")
    set(HAS_EIGEN ON)
else()
    # Fetch Eigen3 if not found system-wide
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
    set(HAS_EIGEN ON)
    message(STATUS "Eigen3: Fetched and built from source")
endif()

# Intel MKL (optional, preferred for performance)
find_package(MKL QUIET)
if(MKL_FOUND)
    message(STATUS "Intel MKL: Found - using optimized BLAS operations")
    set(HAS_MKL ON)
else()
    message(STATUS "Intel MKL: Not found - using Eigen3 built-in BLAS")
    set(HAS_MKL OFF)
endif()

# ============================================================================
# Compiler Optimization and SIMD Detection
# ============================================================================

# Detect SIMD capabilities at configure time
include(CheckCXXCompilerFlag)

# Check for native architecture support
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

# Try to run a test program to detect CPU features at configure time
try_run(
    CPU_FEATURES_RUN_RESULT
    CPU_FEATURES_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpu_features_test.cpp
    CMAKE_FLAGS -DCMAKE_CXX_STANDARD=23
    RUN_OUTPUT_VARIABLE CPU_FEATURES_OUTPUT
)

if(CPU_FEATURES_COMPILE_RESULT AND CPU_FEATURES_RUN_RESULT EQUAL 0)
    # Parse the output to get feature flags
    string(REGEX MATCH "AVX2:([01])" AVX2_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX512F:([01])" AVX512F_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX512DQ:([01])" AVX512DQ_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX512BW:([01])" AVX512BW_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX512VL:([01])" AVX512VL_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "FMA3:([01])" FMA3_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "AVX_VNNI:([01])" AVX_VNNI_MATCH ${CPU_FEATURES_OUTPUT})
    string(REGEX MATCH "BMI2:([01])" BMI2_MATCH ${CPU_FEATURES_OUTPUT})

    # Extract the actual values from regex groups
    if(AVX2_MATCH)
        string(REGEX REPLACE "AVX2:([01])" "\\1" AVX2_VALUE ${AVX2_MATCH})
        if(AVX2_VALUE STREQUAL "1")
            set(HAS_AVX2 ON)
            message(STATUS "CPU Feature: AVX2 detected")
        endif()
    endif()

    if(AVX512F_MATCH)
        string(REGEX REPLACE "AVX512F:([01])" "\\1" AVX512F_VALUE ${AVX512F_MATCH})
        if(AVX512F_VALUE STREQUAL "1")
            set(HAS_AVX512F ON)
            message(STATUS "CPU Feature: AVX-512 Foundation detected")
        endif()
    endif()

    if(AVX512DQ_MATCH)
        string(REGEX REPLACE "AVX512DQ:([01])" "\\1" AVX512DQ_VALUE ${AVX512DQ_MATCH})
        if(AVX512DQ_VALUE STREQUAL "1")
            set(HAS_AVX512DQ ON)
            message(STATUS "CPU Feature: AVX-512 Doubleword/Quadword detected")
        endif()
    endif()

    if(AVX512BW_MATCH)
        string(REGEX REPLACE "AVX512BW:([01])" "\\1" AVX512BW_VALUE ${AVX512BW_MATCH})
        if(AVX512BW_VALUE STREQUAL "1")
            set(HAS_AVX512BW ON)
            message(STATUS "CPU Feature: AVX-512 Byte/Word detected")
        endif()
    endif()

    if(AVX512VL_MATCH)
        string(REGEX REPLACE "AVX512VL:([01])" "\\1" AVX512VL_VALUE ${AVX512VL_MATCH})
        if(AVX512VL_VALUE STREQUAL "1")
            set(HAS_AVX512VL ON)
            message(STATUS "CPU Feature: AVX-512 Vector Length detected")
        endif()
    endif()

    if(FMA3_MATCH)
        string(REGEX REPLACE "FMA3:([01])" "\\1" FMA3_VALUE ${FMA3_MATCH})
        if(FMA3_VALUE STREQUAL "1")
            set(HAS_FMA3 ON)
            message(STATUS "CPU Feature: FMA3 detected")
        endif()
    endif()

    if(AVX_VNNI_MATCH)
        string(REGEX REPLACE "AVX_VNNI:([01])" "\\1" AVX_VNNI_VALUE ${AVX_VNNI_MATCH})
        if(AVX_VNNI_VALUE STREQUAL "1")
            set(HAS_AVX_VNNI ON)
            message(STATUS "CPU Feature: AVX-VNNI detected")
        endif()
    endif()

    if(BMI2_MATCH)
        string(REGEX REPLACE "BMI2:([01])" "\\1" BMI2_VALUE ${BMI2_MATCH})
        if(BMI2_VALUE STREQUAL "1")
            set(HAS_BMI2 ON)
            message(STATUS "CPU Feature: BMI2 detected")
        endif()
    endif()
else()
    # Fallback: Use compiler flag detection when runtime test fails
    message(STATUS "CPU feature runtime detection failed - using compiler-based detection")
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(HAS_AVX ON)
        message(STATUS "SIMD: Using -march=native (fallback detection)")
    endif()
    # Fallback to compiler flag detection
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
# AVX512 Warning Suppression and Feature-Specific Compilation
# ============================================================================

# Check if compiler supports AVX512 flags
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
check_cxx_compiler_flag("-mavx512bw" COMPILER_SUPPORTS_AVX512BW)
check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)
check_cxx_compiler_flag("-mavx512vnni" COMPILER_SUPPORTS_AVX512VNNI)

# Set feature-specific flags only if both CPU and compiler support them
if(HAS_AVX512F AND COMPILER_SUPPORTS_AVX512F)
    set(AVX512F_FLAG "-mavx512f")
    message(STATUS "AVX-512F: Enabled (CPU + compiler support)")
else()
    set(AVX512F_FLAG "")
    if(NOT HAS_AVX512F)
        message(STATUS "AVX-512F: Disabled (CPU does not support)")
    elseif(NOT COMPILER_SUPPORTS_AVX512F)
        message(STATUS "AVX-512F: Disabled (compiler does not support)")
    endif()
endif()

# Additional AVX512 features
if(HAS_AVX512F AND COMPILER_SUPPORTS_AVX512DQ)
    set(AVX512DQ_FLAG "-mavx512dq")
endif()

if(HAS_AVX512F AND COMPILER_SUPPORTS_AVX512BW)
    set(AVX512BW_FLAG "-mavx512bw")
endif()

if(HAS_AVX512F AND COMPILER_SUPPORTS_AVX512VL)
    set(AVX512VL_FLAG "-mavx512vl")
endif()

if(HAS_AVX512F AND HAS_AVX_VNNI AND COMPILER_SUPPORTS_AVX512VNNI)
    set(AVX512VNNI_FLAG "-mavx512vnni")
endif()

# Common optimization flags
if(MSVC)
    set(COMPILER_OPT_FLAGS "/O2 /DNDEBUG /fp:precise")
    set(COMPILER_DEBUG_FLAGS "/Od /DDEBUG /Zi")
    set(COMPILER_WARNING_FLAGS "/W3 /EHsc /permissive- /DNOMINMAX")
    set(COMPILER_AVX_FLAGS "/arch:AVX2")
    add_compile_options(/wd4244 /wd4267 /wd4996 /wd4005 /wd4200)
else()
    set(COMPILER_OPT_FLAGS "-O3 -DNDEBUG -ffast-math")
    set(COMPILER_DEBUG_FLAGS "-g -O0 -DDEBUG")
    set(COMPILER_WARNING_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare")
    # Start with AVX2 as baseline, add AVX512 features conditionally
    set(COMPILER_AVX_FLAGS "-mavx2 -mfma")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(COMPILER_DEBUG_FLAGS "${COMPILER_DEBUG_FLAGS} -fsanitize=address,undefined")
    endif()
endif()

# Apply flags based on build type
set(CMAKE_CXX_FLAGS_RELEASE "${COMPILER_OPT_FLAGS} ${COMPILER_WARNING_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${COMPILER_OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${COMPILER_DEBUG_FLAGS} ${COMPILER_WARNING_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${COMPILER_DEBUG_FLAGS}")

# Apply SIMD flags based on detected CPU capabilities
if(HAS_AVX2)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMPILER_AVX_FLAGS}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${COMPILER_AVX_FLAGS}")

    # Add AVX512 flags only if supported by both CPU and compiler
    # This prevents macro redefinition warnings by not enabling AVX512 at all when CPU doesn't support it
    if(HAS_AVX512F AND AVX512F_FLAG)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${AVX512F_FLAG}")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${AVX512F_FLAG}")
    endif()
    if(HAS_AVX512DQ AND AVX512DQ_FLAG)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${AVX512DQ_FLAG}")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${AVX512DQ_FLAG}")
    endif()
    if(HAS_AVX512BW AND AVX512BW_FLAG)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${AVX512BW_FLAG}")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${AVX512BW_FLAG}")
    endif()
    if(HAS_AVX512VL AND AVX512VL_FLAG)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${AVX512VL_FLAG}")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${AVX512VL_FLAG}")
    endif()
    if(HAS_AVX_VNNI AND AVX512VNNI_FLAG)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${AVX512VNNI_FLAG}")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${AVX512VNNI_FLAG}")
    endif()
endif()

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

# ============================================================================
# PostgreSQL Configuration
# ============================================================================

find_program(PG_CONFIG pg_config)
if(PG_CONFIG)
    execute_process(COMMAND ${PG_CONFIG} --includedir-server OUTPUT_VARIABLE PG_INCLUDEDIR_SERVER OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --includedir OUTPUT_VARIABLE PG_INCLUDEDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --pkglibdir OUTPUT_VARIABLE PG_PKGLIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --sharedir OUTPUT_VARIABLE PG_SHAREDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --libdir OUTPUT_VARIABLE PG_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${PG_CONFIG} --version OUTPUT_VARIABLE PG_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    message(STATUS "PostgreSQL: ${PG_VERSION}")
    message(STATUS "PostgreSQL server includes: ${PG_INCLUDEDIR_SERVER}")
    message(STATUS "PostgreSQL pkglibdir: ${PG_PKGLIBDIR}")
    set(BUILD_PG_EXTENSION ON)
else()
    message(WARNING "pg_config not found - building without PostgreSQL extension support")
    set(BUILD_PG_EXTENSION OFF)
endif()

# ============================================================================
# Core C++ Library (Enterprise-Grade Architecture)
# ============================================================================

add_library(hypercube_core STATIC
    src/core/hilbert.cpp
    src/core/coordinates.cpp
    src/core/blake3_pg.cpp
    src/core/ops.cpp
    src/util/utf8.cpp
    src/core/atom_calculator.cpp
    src/core/laplacian_4d.cpp
    src/core/lanczos.cpp
    src/core/semantic_ordering.cpp
    src/core/superfibonacci.cpp
)

target_include_directories(hypercube_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Dependencies
target_link_libraries(hypercube_core
    PUBLIC OpenMP::OpenMP_CXX
    PRIVATE Threads::Threads
)

# HNSWLIB integration - disabled for now to avoid AVX512 issues
# TODO: Fix HNSWLib AVX512 detection to respect CPU capabilities

# Linear algebra backend
if(TARGET Eigen3::Eigen)
    target_link_libraries(hypercube_core PUBLIC Eigen3::Eigen)
else()
    target_include_directories(hypercube_core PUBLIC ${EIGEN3_INCLUDE_DIR})
endif()
target_compile_definitions(hypercube_core PUBLIC HAS_EIGEN=1)

# MKL for optimized BLAS operations
if(HAS_MKL)
    target_link_libraries(hypercube_core PUBLIC MKL::MKL)
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=1)
else()
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=0)
endif()

# SIMD capabilities - detailed feature detection
target_compile_definitions(hypercube_core PUBLIC
    HAS_AVX=$<BOOL:${HAS_AVX}>
    HAS_AVX2=$<BOOL:${HAS_AVX2}>
    HAS_AVX512F=$<BOOL:${HAS_AVX512F}>
    HAS_FMA3=$<BOOL:${HAS_FMA3}>
    HAS_AVX_VNNI=$<BOOL:${HAS_AVX_VNNI}>
)

# AVX512 features are now handled at the compiler flag level
# No need to undefine macros since we don't enable AVX512 flags when CPU doesn't support them
if(NOT HAS_AVX512F)
    target_compile_definitions(hypercube_core PRIVATE
        # Disable AVX512 code paths at compile time
        DISABLE_AVX512
        DISABLE_AVX512F
        DISABLE_AVX512DQ
        DISABLE_AVX512BW
        DISABLE_AVX512VL
        DISABLE_AVX512VNNI
    )
    message(STATUS "AVX-512: All AVX-512 features disabled (not supported by CPU)")
else()
    message(STATUS "AVX-512: CPU supports AVX-512 - features enabled where available")
endif()

# Platform-specific definitions
if(WIN32)
    target_compile_definitions(hypercube_core PRIVATE NOMINMAX)
endif()

# ============================================================================
# 2. C API Bridge Library (C++ implementation, C interface)
# Must be SHARED on Windows so that the PostgreSQL C extensions can link to it
# ============================================================================

add_library(hypercube_c SHARED
    src/bridge/hypercube_c.cpp
)

target_include_directories(hypercube_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hypercube_c
    PRIVATE hypercube_core
    PRIVATE Threads::Threads
)

# Export symbols properly on Windows
if(WIN32)
    target_compile_definitions(hypercube_c PRIVATE HYPERCUBE_C_EXPORTS)
endif()

set_target_properties(hypercube_c PROPERTIES
    PREFIX ""
    OUTPUT_NAME "hypercube_c"
)

# ============================================================================
# 3 & 4. PostgreSQL Extensions (pure C - compiled with C compiler)
#
# CRITICAL: These are .c files that include PostgreSQL headers.
# PostgreSQL headers have C++-incompatible code (atomics, type punning).
# By keeping these as pure C, we avoid all MSVC C++ errors.
# ============================================================================

if(BUILD_PG_EXTENSION)
    # Windows-specific defines for PostgreSQL
    if(WIN32)
        set(PG_COMPILE_DEFS
            WIN32_LEAN_AND_MEAN
            _CRT_SECURE_NO_WARNINGS
            _WINSOCK_DEPRECATED_NO_WARNINGS
        )
    else()
        set(PG_COMPILE_DEFS "")
    endif()
    
    # PostgreSQL include directories (proper order for Windows)
    if(WIN32)
        set(PG_INCLUDE_DIRS
            ${PG_INCLUDEDIR_SERVER}/port/win32_msvc
            ${PG_INCLUDEDIR_SERVER}/port/win32
            ${PG_INCLUDEDIR_SERVER}
            ${PG_INCLUDEDIR}
        )
    else()
        set(PG_INCLUDE_DIRS
            ${PG_INCLUDEDIR_SERVER}
            ${PG_INCLUDEDIR}
        )
    endif()
    
    # Find postgres.lib on Windows
    if(WIN32)
        find_library(POSTGRES_LIB postgres HINTS "${PG_PKGLIBDIR}" "${PG_LIBDIR}")
        if(POSTGRES_LIB)
            message(STATUS "Found postgres.lib: ${POSTGRES_LIB}")
        else()
            message(WARNING "postgres.lib not found - extensions may not link correctly")
        endif()
    endif()
    
    # -------------------------------------------------------------------------
    # hypercube PostgreSQL Extension (pure C)
    # -------------------------------------------------------------------------
    
    add_library(hypercube SHARED
        src/pg/hypercube_pg.c
    )
    
    # Force C language for this file
    set_source_files_properties(src/pg/hypercube_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(hypercube PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(hypercube PRIVATE
        ${PG_COMPILE_DEFS}
        HYPERCUBE_VERSION="${PROJECT_VERSION}"
    )

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(hypercube PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()
    
    # Force C linker so we behave like native PG extension DLLs on Windows
    set_target_properties(hypercube PROPERTIES LINKER_LANGUAGE C)

    target_link_libraries(hypercube
        PRIVATE hypercube_c
    )
    
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(hypercube PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(hypercube PROPERTIES
        PREFIX ""
        OUTPUT_NAME "hypercube"
    )
    
    # -------------------------------------------------------------------------
    # semantic_ops PostgreSQL Extension (pure C)
    # -------------------------------------------------------------------------
    
    add_library(semantic_ops SHARED
        src/pg/semantic_ops_pg.c
    )
    
    set_source_files_properties(src/pg/semantic_ops_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(semantic_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(semantic_ops PRIVATE
        ${PG_COMPILE_DEFS}
    )

    set_target_properties(semantic_ops PROPERTIES LINKER_LANGUAGE C)

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(semantic_ops PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()

    target_link_libraries(semantic_ops
        PRIVATE hypercube_c
    )
    
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(semantic_ops PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(semantic_ops PROPERTIES
        PREFIX ""
        OUTPUT_NAME "semantic_ops"
    )
    
    # -------------------------------------------------------------------------
    # hypercube_ops PostgreSQL Extension (pure C - optimized batch operations)
    # Uses hypercube_c for BLAKE3, Hilbert, and coordinate functions
    # -------------------------------------------------------------------------
    
    add_library(hypercube_ops SHARED
        src/pg/hypercube_ops_pg.c
    )
    
    set_source_files_properties(src/pg/hypercube_ops_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(hypercube_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(hypercube_ops PRIVATE
        ${PG_COMPILE_DEFS}
    )

    set_target_properties(hypercube_ops PROPERTIES LINKER_LANGUAGE C)

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(hypercube_ops PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()

    # Link against hypercube_c for C API functions (seed_atoms needs them)
    target_link_libraries(hypercube_ops PRIVATE hypercube_c)
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(hypercube_ops PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(hypercube_ops PROPERTIES
        PREFIX ""
        OUTPUT_NAME "hypercube_ops"
    )
    
    # -------------------------------------------------------------------------
    # embedding_ops PostgreSQL Extension (pure C - SIMD embedding operations)
    # -------------------------------------------------------------------------
    
    # C bridge library for embedding operations (C++ with C API)
    add_library(embedding_c SHARED
        src/bridge/embedding_c.cpp
    )

    target_include_directories(embedding_c PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    target_link_libraries(embedding_c
        PRIVATE hypercube_core
        PRIVATE Threads::Threads
    )
    
    if(WIN32)
        target_compile_definitions(embedding_c PRIVATE EMBEDDING_C_EXPORTS)
    endif()
    
    set_target_properties(embedding_c PROPERTIES
        PREFIX ""
        OUTPUT_NAME "embedding_c"
    )
    
    # Pure C PostgreSQL extension
    add_library(embedding_ops SHARED
        src/pg/embedding_ops_pg.c
    )
    
    set_source_files_properties(src/pg/embedding_ops_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(embedding_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(embedding_ops PRIVATE
        ${PG_COMPILE_DEFS}
    )

    set_target_properties(embedding_ops PROPERTIES LINKER_LANGUAGE C)

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(embedding_ops PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()
    
    target_link_libraries(embedding_ops
        PRIVATE embedding_c
    )
    
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(embedding_ops PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(embedding_ops PROPERTIES
        PREFIX ""
        OUTPUT_NAME "embedding_ops"
    )
    
    # -------------------------------------------------------------------------
    # generative PostgreSQL Extension (generative walk engine)
    # -------------------------------------------------------------------------
    
    # C bridge library for generative engine (C++ with C API)
    add_library(generative_c SHARED
        src/bridge/generative_c.cpp
    )

    target_include_directories(generative_c PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    target_link_libraries(generative_c
        PRIVATE hypercube_core
        PRIVATE Threads::Threads
    )
    
    if(WIN32)
        target_compile_definitions(generative_c PRIVATE GENERATIVE_C_EXPORTS)
    endif()
    
    set_target_properties(generative_c PROPERTIES
        PREFIX ""
        OUTPUT_NAME "generative_c"
    )
    
    # Pure C PostgreSQL extension
    add_library(generative SHARED
        src/pg/generative_pg.c
    )
    
    set_source_files_properties(src/pg/generative_pg.c PROPERTIES LANGUAGE C)
    
    target_include_directories(generative PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PG_INCLUDE_DIRS}
    )
    
    target_compile_definitions(generative PRIVATE
        ${PG_COMPILE_DEFS}
    )

    set_target_properties(generative PROPERTIES LINKER_LANGUAGE C)

    # Suppress warnings from PostgreSQL headers
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(generative PRIVATE -Wno-macro-redefined -Wno-incompatible-pointer-types -Wno-microsoft-include -Wno-unused-but-set-variable -Wno-pointer-sign)
    endif()
    
    target_link_libraries(generative
        PRIVATE generative_c
    )
    
    if(WIN32 AND POSTGRES_LIB)
        target_link_libraries(generative PRIVATE ${POSTGRES_LIB})
    endif()
    
    set_target_properties(generative PROPERTIES
        PREFIX ""
        OUTPUT_NAME "generative"
    )
    
    # -------------------------------------------------------------------------
    # Installation
    # -------------------------------------------------------------------------

    # The extension DLLs depend on hypercube_c.dll at runtime (Windows loader).
    # Install it into pkglibdir alongside the modules.
    install(TARGETS hypercube_c
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(TARGETS hypercube
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/hypercube--1.0.sql
        hypercube.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
    
    install(TARGETS semantic_ops
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/semantic_ops--1.0.sql
        sql/semantic_ops.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
    
    install(TARGETS hypercube_ops
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/hypercube_ops--1.0.sql
        sql/hypercube_ops.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
    
    install(TARGETS embedding_c
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(TARGETS embedding_ops
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/embedding_ops--1.0.sql
        sql/embedding_ops.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
    
    install(TARGETS generative_c
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(TARGETS generative
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
        RUNTIME DESTINATION ${PG_PKGLIBDIR}
    )
    
    install(FILES
        sql/generative--1.0.sql
        sql/generative.control
        DESTINATION ${PG_SHAREDIR}/extension
    )
endif()

# ============================================================================
# Tests (C++ tests, no PostgreSQL)
# ============================================================================

enable_testing()

add_executable(test_hilbert tests/test_hilbert.cpp)
target_link_libraries(test_hilbert hypercube_core)
add_test(NAME HilbertTest COMMAND test_hilbert)

add_executable(test_coordinates tests/test_coordinates.cpp)
target_link_libraries(test_coordinates hypercube_core)
add_test(NAME CoordinatesTest COMMAND test_coordinates)

add_executable(test_blake3 tests/test_blake3.cpp)
target_link_libraries(test_blake3 hypercube_core)
add_test(NAME Blake3Test COMMAND test_blake3)

add_executable(test_semantic tests/test_semantic.cpp)
target_link_libraries(test_semantic hypercube_core)
add_test(NAME SemanticTest COMMAND test_semantic)

add_executable(debug_semantic_order tests/debug_semantic_order.cpp)
add_executable(test_vocab_parse tests/test_vocab_parse.cpp)
target_include_directories(test_vocab_parse PRIVATE ${PostgreSQL_INCLUDE_DIRS})
target_link_libraries(debug_semantic_order hypercube_core)
target_link_libraries(test_vocab_parse hypercube_core hypercube_c ${PostgreSQL_LIBRARIES} Threads::Threads)
add_test(NAME DebugSemanticOrder COMMAND debug_semantic_order)



add_executable(test_clustering tests/test_clustering.cpp)
target_link_libraries(test_clustering hypercube_core)
add_test(NAME ClusteringTest COMMAND test_clustering)

add_executable(test_laplacian_4d tests/test_laplacian_4d.cpp)
target_link_libraries(test_laplacian_4d hypercube_core)
add_test(NAME Laplacian4DTest COMMAND test_laplacian_4d)

add_executable(test_eigen_solver_paths test_eigen_solver_paths.cpp)
target_link_libraries(test_eigen_solver_paths hypercube_core)

add_executable(test_cpu_features tests/test_cpu_features.cpp)
target_link_libraries(test_cpu_features hypercube_core)

add_executable(test_int8_quantization tests/test_int8_quantization.cpp)
target_link_libraries(test_int8_quantization hypercube_core)

add_executable(test_unicode_seeding tests/test_unicode_seeding.cpp)
target_link_libraries(test_unicode_seeding hypercube_core)
add_test(NAME UnicodeSeedingTest COMMAND test_unicode_seeding)

# ============================================================================
# PostgreSQL Tools (require libpq client library)
# ============================================================================

# Try to find libpq via vcpkg or system
# Prefer system libpq over Windows installations
set(PostgreSQL_ROOT "/usr")
find_package(PostgreSQL QUIET)
if(PostgreSQL_FOUND)
    message(STATUS "Found PostgreSQL client library: ${PostgreSQL_LIBRARIES}")

    # -------------------------------------------------------------------------
    # Modular ingester library (must be defined before tools that link to it)
    # -------------------------------------------------------------------------
    add_library(hypercube_ingest STATIC
        src/db/atom_cache.cpp
        src/db/geometry.cpp
        src/db/insert.cpp
        src/ingest/cpe.cpp
        src/ingest/universal.cpp
        src/ingest/parallel_cpe.cpp
        src/ingest/sequitur.cpp
        src/ingest/pmi_contraction.cpp
        src/ingest/projection_db.cpp
        # Modular DB operations for safetensor ingestion
        src/ingest/tensor_io.cpp
        src/ingest/compositions.cpp
        src/ingest/tensor_hierarchy.cpp
        src/ingest/embedding_relations.cpp
        src/ingest/attention_relations.cpp
        src/ingest/semantic_extraction.cpp
        src/ingest/multimodal_extraction.cpp
    )
    target_include_directories(hypercube_ingest PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PostgreSQL_INCLUDE_DIRS}
        ${MKL_INCLUDE_DIRS}
    )
    # HNSWLib disabled
    target_link_libraries(hypercube_ingest PRIVATE
        hypercube_core
        ${PostgreSQL_LIBRARIES}
        Threads::Threads
    )

    # Fix Windows min/max macro conflicts with HNSWLIB
    if(WIN32)
        target_compile_definitions(hypercube_ingest PRIVATE NOMINMAX)
    endif()

    # -------------------------------------------------------------------------
    # PostgreSQL CLI tools
    # -------------------------------------------------------------------------

    # Modular safetensor ingester (replaces archived monolith)
    add_executable(ingest_safetensor src/tools/ingest_safetensor_modular.cpp)
    target_include_directories(ingest_safetensor PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    # HNSWLib disabled
    target_link_libraries(ingest_safetensor PRIVATE hypercube_core hypercube_ingest ${PostgreSQL_LIBRARIES} Threads::Threads)

    # Fix Windows min/max macro conflicts with HNSWLIB
    if(WIN32)
        target_compile_definitions(ingest_safetensor PRIVATE NOMINMAX)
    endif()

    # ARCHIVED: ingest_safetensor_4d (moved to src/archive/)
    # ARCHIVED: ingest_safetensor_universal (moved to src/archive/)
    # USE: ingest_safetensor (modular version with shared library components)

    add_executable(extract_embeddings src/tools/extract_embeddings.cpp)
    target_include_directories(extract_embeddings PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(extract_embeddings hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)

    # New unified ingester tool
    add_executable(ingest src/ingest/main.cpp)
    target_link_libraries(ingest hypercube_ingest hypercube_core ${PostgreSQL_LIBRARIES} Threads::Threads)

    # ========================================================================
    # Unified CLI Tool: hypercube_cli
    # ========================================================================
    # Single entry point for all operations: ingest, query, test, stats
    add_executable(hypercube_cli src/cli/main.cpp)
    target_include_directories(hypercube_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PostgreSQL_INCLUDE_DIRS}
    )
    target_link_libraries(hypercube_cli PRIVATE
        hypercube_core
        hypercube_ingest
        ${PostgreSQL_LIBRARIES}
        Threads::Threads
    )

    # Add backend detection defines
    target_compile_definitions(hypercube_cli PRIVATE HAS_EIGEN=1)
    if(HAS_MKL)
        target_compile_definitions(hypercube_cli PRIVATE HAS_MKL=1)
    else()
        target_compile_definitions(hypercube_cli PRIVATE HAS_MKL=0)
    endif()
    # HNSWLib disabled

    # Set output name to just "hypercube" (conflicts with extension on some systems)
    set_target_properties(hypercube_cli PROPERTIES OUTPUT_NAME "hc")

    # Model discovery tool (scans HC_MODEL_PATHS for HuggingFace models)
    add_executable(model_discovery src/ingest/model_discovery.cpp)
    target_link_libraries(model_discovery Threads::Threads)

    # Vocabulary extractor (extracts vocab from any model as text)
    add_executable(vocab_extract src/ingest/vocab_extract.cpp)
    target_link_libraries(vocab_extract Threads::Threads)

    # Streaming vocabulary ingester (one token at a time through PMI) - standalone
    add_executable(vocab_ingest src/ingest/vocab_ingest.cpp)
    target_link_libraries(vocab_ingest Threads::Threads)

    # Integration tests with libpq
    add_executable(test_integration tests/test_integration.cpp)
    target_include_directories(test_integration PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(test_integration hypercube_core ${PostgreSQL_LIBRARIES})
    add_test(NAME IntegrationTest COMMAND test_integration)
    set_tests_properties(IntegrationTest PROPERTIES
        ENVIRONMENT "HC_DB_NAME=hypercube;HC_DB_USER=hartonomous;HC_DB_PASS=hartonomous;HC_DB_HOST=localhost;HC_DB_PORT=5432"
    )

    add_executable(test_query_api tests/test_query_api.cpp)
    target_include_directories(test_query_api PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(test_query_api hypercube_core ${PostgreSQL_LIBRARIES})
    add_test(NAME QueryAPITest COMMAND test_query_api)
    set_tests_properties(QueryAPITest PROPERTIES
        ENVIRONMENT "HC_DB_NAME=hypercube;HC_DB_USER=hartonomous;HC_DB_PASS=hartonomous;HC_DB_HOST=localhost;HC_DB_PORT=5432"
    )
else()
    message(STATUS "PostgreSQL client library not found - skipping tool builds")
    message(STATUS "To build tools, install libpq via: vcpkg install libpq:x64-windows")
endif()

# ============================================================================
# Seed Atoms Tool (ALWAYS BUILT - uses extension when available)
# ============================================================================

# Try to find libpq for standalone seeder (even without full PostgreSQL package)
if(NOT PostgreSQL_FOUND)
    # Try manual libpq detection on Windows
    if(WIN32)
        # Look for PostgreSQL installations
        set(PG_SEARCH_PATHS
            "C:/Program Files/PostgreSQL/*/lib"
            "C:/Program Files (x86)/PostgreSQL/*/lib"
            "$ENV{ProgramFiles}/PostgreSQL/*/lib"
            "$ENV{ProgramFiles(x86)}/PostgreSQL/*/lib"
        )

        find_library(LIBPQ_LIB pq libpq PATHS ${PG_SEARCH_PATHS})
        if(LIBPQ_LIB)
            get_filename_component(PG_LIB_DIR "${LIBPQ_LIB}" DIRECTORY)
            get_filename_component(PG_ROOT "${PG_LIB_DIR}" DIRECTORY)
            set(PostgreSQL_INCLUDE_DIRS "${PG_ROOT}/include")
            set(PostgreSQL_LIBRARIES "${LIBPQ_LIB}")
            set(PostgreSQL_FOUND TRUE)
            message(STATUS "Found libpq manually: ${LIBPQ_LIB}")
            message(STATUS "Include dirs: ${PostgreSQL_INCLUDE_DIRS}")
        endif()
    endif()
endif()

if(PostgreSQL_FOUND)
    add_executable(seed_atoms_parallel src/tools/seed_atoms_parallel.cpp)
    target_include_directories(seed_atoms_parallel PRIVATE ${PostgreSQL_INCLUDE_DIRS})
    target_link_libraries(seed_atoms_parallel hypercube_core hypercube_c ${PostgreSQL_LIBRARIES} Threads::Threads)

    # Fix Windows min/max macro conflicts
    if(WIN32)
        target_compile_definitions(seed_atoms_parallel PRIVATE NOMINMAX)
    endif()

    message(STATUS "seed_atoms_parallel will be built")
else()
    message(STATUS "libpq not found - seed_atoms_parallel will not be built (extension seeding will be used)")
endif()

# ============================================================================
# Google Test Suite
# ============================================================================
find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "Found Google Test: ${GTEST_INCLUDE_DIRS}")
    enable_testing()
    
    # Core unit tests (no PostgreSQL required)
    add_executable(hypercube_tests
        tests/gtest/test_main.cpp
        tests/gtest/test_blake3.cpp
        tests/gtest/test_hilbert.cpp
        tests/gtest/test_coordinates.cpp
        tests/gtest/test_laplacian.cpp
        tests/gtest/test_backend.cpp
    )
    target_include_directories(hypercube_tests PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GTEST_INCLUDE_DIRS}
    )
    target_link_libraries(hypercube_tests 
        hypercube_core 
        GTest::gtest 
        GTest::gtest_main
        Threads::Threads
    )
    
    # Add backend defines
    target_compile_definitions(hypercube_tests PRIVATE HAS_EIGEN=1)
    if(HAS_MKL)
        target_compile_definitions(hypercube_tests PRIVATE HAS_MKL=1)
    else()
        target_compile_definitions(hypercube_tests PRIVATE HAS_MKL=0)
    endif()
    # HNSWLib disabled
    
    add_test(NAME HypercubeTests COMMAND hypercube_tests)
    # Increase stack size to prevent crashes (MSVC only)
    if(MSVC)
        set_target_properties(hypercube_tests PROPERTIES LINK_FLAGS "/STACK:8388608")
    endif()
    
    # SQL integration tests (requires PostgreSQL)
    if(PostgreSQL_FOUND)
        add_executable(hypercube_sql_tests
            tests/gtest/test_main.cpp
            tests/gtest/test_sql_schema.cpp
            tests/gtest/test_sql_functions.cpp
            tests/gtest/test_sql_query_api.cpp
        )
        target_include_directories(hypercube_sql_tests PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${GTEST_INCLUDE_DIRS}
            ${PostgreSQL_INCLUDE_DIRS}
        )
        target_link_libraries(hypercube_sql_tests 
            hypercube_core 
            GTest::gtest 
            GTest::gtest_main
            ${PostgreSQL_LIBRARIES}
            Threads::Threads
        )
        add_test(NAME HypercubeSQLTests COMMAND hypercube_sql_tests)
        set_tests_properties(HypercubeSQLTests PROPERTIES
            ENVIRONMENT "HC_DB_NAME=hypercube;HC_DB_USER=hartonomous;HC_DB_PASS=hartonomous;HC_DB_HOST=localhost;HC_DB_PORT=5432"
        )
    endif()
else()
    message(STATUS "Google Test not found - skipping test suite")
    message(STATUS "To enable tests: vcpkg install gtest:x64-windows")
endif()

# ============================================================================
# Runtime Environment Optimization for AVX/AVX_VNNI
# ============================================================================

message(STATUS "Runtime Environment Setup for AVX Optimization:")
message(STATUS "  Set these in your ~/.bashrc or runtime environment:")
message(STATUS "  export OMP_PROC_BIND=true")
message(STATUS "  export OMP_PLACES=cores")
message(STATUS "  export MKL_DEBUG_CPU_TYPE=5  # Force AVX2 code path")
message(STATUS "")
message(STATUS "  This ensures optimal thread scheduling and MKL path selection.")
message(STATUS "")

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== Hypercube Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "PostgreSQL extension: ${BUILD_PG_EXTENSION}")
if(PostgreSQL_FOUND)
    message(STATUS "PostgreSQL tools: ON")
else()
    message(STATUS "PostgreSQL tools: OFF (libpq not found)")
endif()
message(STATUS "Eigen3: ON")
if(HAS_MKL)
    message(STATUS "Intel MKL: ON")
else()
    message(STATUS "Intel MKL: OFF")
endif()
message(STATUS "k-NN: Brute-force (HNSWLib disabled)")
if(GTest_FOUND)
    message(STATUS "Tests: ON (Google Test)")
else()
    message(STATUS "Tests: OFF")
endif()
message(STATUS "")
