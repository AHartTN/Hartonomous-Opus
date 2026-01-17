# SIMD kernel object libraries with per-ISA compilation

# Baseline (no special flags)
add_library(simd_kernels_baseline OBJECT src/simd_kernels_baseline.cpp)
target_include_directories(simd_kernels_baseline PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(simd_kernels_baseline PUBLIC HAS_AVX=$<BOOL:${HAS_AVX}> HAS_AVX2=$<BOOL:${HAS_AVX2}> HAS_AVX512F=$<BOOL:${HAS_AVX512F}> HAS_FMA3=$<BOOL:${HAS_FMA3}> HAS_AVX_VNNI=$<BOOL:${HAS_AVX_VNNI}> HAS_OPENMP=$<BOOL:${HAS_OPENMP}>)

# SSE42
add_library(simd_kernels_sse42 OBJECT src/simd_kernels_sse42.cpp)
target_include_directories(simd_kernels_sse42 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(simd_kernels_sse42 PUBLIC HAS_AVX=$<BOOL:${HAS_AVX}> HAS_AVX2=$<BOOL:${HAS_AVX2}> HAS_AVX512F=$<BOOL:${HAS_AVX512F}> HAS_FMA3=$<BOOL:${HAS_FMA3}> HAS_AVX_VNNI=$<BOOL:${HAS_AVX_VNNI}> HAS_OPENMP=$<BOOL:${HAS_OPENMP}>)
target_compile_options(simd_kernels_sse42 PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/arch:SSE4.2> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-msse4.2>)

# AVX2
add_library(simd_kernels_avx2 OBJECT src/simd_kernels_avx2.cpp)
target_include_directories(simd_kernels_avx2 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(simd_kernels_avx2 PUBLIC HAS_AVX=$<BOOL:${HAS_AVX}> HAS_AVX2=$<BOOL:${HAS_AVX2}> HAS_AVX512F=$<BOOL:${HAS_AVX512F}> HAS_FMA3=$<BOOL:${HAS_FMA3}> HAS_AVX_VNNI=$<BOOL:${HAS_AVX_VNNI}> HAS_OPENMP=$<BOOL:${HAS_OPENMP}>)
target_compile_options(simd_kernels_avx2 PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2 -mfma>)

# AVX2_VNNI
add_library(simd_kernels_avx2_vnni OBJECT src/simd_kernels_avx2_vnni.cpp)
target_include_directories(simd_kernels_avx2_vnni PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(simd_kernels_avx2_vnni PUBLIC HAS_AVX=$<BOOL:${HAS_AVX}> HAS_AVX2=$<BOOL:${HAS_AVX2}> HAS_AVX512F=$<BOOL:${HAS_AVX512F}> HAS_FMA3=$<BOOL:${HAS_FMA3}> HAS_AVX_VNNI=$<BOOL:${HAS_AVX_VNNI}> HAS_OPENMP=$<BOOL:${HAS_OPENMP}>)
target_compile_options(simd_kernels_avx2_vnni PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2 -mfma -mavxvnni>)

# AVX512
add_library(simd_kernels_avx512 OBJECT src/simd_kernels_avx512.cpp)
target_include_directories(simd_kernels_avx512 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(simd_kernels_avx512 PUBLIC HAS_AVX=$<BOOL:${HAS_AVX}> HAS_AVX2=$<BOOL:${HAS_AVX2}> HAS_AVX512F=$<BOOL:${HAS_AVX512F}> HAS_FMA3=$<BOOL:${HAS_FMA3}> HAS_AVX_VNNI=$<BOOL:${HAS_AVX_VNNI}> HAS_OPENMP=$<BOOL:${HAS_OPENMP}>)
target_compile_options(simd_kernels_avx512 PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX512> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx512f -mavx512dq -mavx512bw -mavx512vl>)

# AVX512_VNNI
add_library(simd_kernels_avx512_vnni OBJECT src/simd_kernels_avx512_vnni.cpp)
target_include_directories(simd_kernels_avx512_vnni PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_definitions(simd_kernels_avx512_vnni PUBLIC HAS_AVX=$<BOOL:${HAS_AVX}> HAS_AVX2=$<BOOL:${HAS_AVX2}> HAS_AVX512F=$<BOOL:${HAS_AVX512F}> HAS_FMA3=$<BOOL:${HAS_FMA3}> HAS_AVX_VNNI=$<BOOL:${HAS_AVX_VNNI}> HAS_OPENMP=$<BOOL:${HAS_OPENMP}>)
target_compile_options(simd_kernels_avx512_vnni PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX512> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni>)

# Core C++ library
add_library(hypercube_core STATIC
    src/core/hilbert.cpp
    src/core/coordinates.cpp
    src/core/blake3_pg.cpp
    src/core/ops.cpp
    src/util/utf8.cpp
    src/core/atom_calculator.cpp
    src/core/laplacian_4d.cpp
    src/core/lanczos.cpp
    src/core/linear_algebra.cpp
    src/core/semantic_ordering.cpp
    src/core/superfibonacci.cpp
    src/core/unicode_categorization.cpp
    src/core/coordinate_utilities.cpp
    src/core/dense_registry.cpp
    src/core/cpu_features.cpp
    src/core/isa_class.cpp
    src/core/dispatch.cpp
    $<TARGET_OBJECTS:simd_kernels_baseline>
    $<TARGET_OBJECTS:simd_kernels_sse42>
    $<TARGET_OBJECTS:simd_kernels_avx2>
    $<TARGET_OBJECTS:simd_kernels_avx2_vnni>
    $<TARGET_OBJECTS:simd_kernels_avx512>
    $<TARGET_OBJECTS:simd_kernels_avx512_vnni>
)

target_include_directories(hypercube_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Per-target compiler flags
target_compile_options(hypercube_core
    PRIVATE
        $<$<CONFIG:Release>:${HYPERCUBE_OPT_FLAGS}>
        $<$<CONFIG:Release>:${HYPERCUBE_WARNING_FLAGS}>
        $<$<CONFIG:Release>:${HYPERCUBE_AVX_FLAGS}>
        $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_DEBUG_FLAGS}>
        $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_WARNING_FLAGS}>
)

# Threads
target_link_libraries(hypercube_core PRIVATE Threads::Threads)

# OpenMP
if(HAS_OPENMP)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(hypercube_core PUBLIC OpenMP::OpenMP_CXX)
    else()
        target_compile_options(hypercube_core PUBLIC ${OpenMP_CXX_FLAGS})
        target_link_libraries(hypercube_core PUBLIC ${OpenMP_CXX_LIBRARIES})
    endif()
endif()

# Eigen
if(TARGET Eigen3::Eigen)
    target_link_libraries(hypercube_core PUBLIC Eigen3::Eigen)
else()
    target_include_directories(hypercube_core PUBLIC ${Eigen3_INCLUDE_DIRS})
endif()
target_compile_definitions(hypercube_core PUBLIC HAS_EIGEN=1)

# MKL
if(HAS_MKL)
    if(TARGET MKL::MKL)
        target_link_libraries(hypercube_core PUBLIC MKL::MKL)
    else()
        target_include_directories(hypercube_core PUBLIC ${MKL_INCLUDE_DIRS})
        target_link_libraries(hypercube_core PUBLIC ${MKL_LIBRARIES})
    endif()
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=1)
else()
    target_compile_definitions(hypercube_core PUBLIC HAS_MKL=0)
endif()

# SIMD macros
target_compile_definitions(hypercube_core PUBLIC
    HAS_AVX=$<BOOL:${HAS_AVX}>
    HAS_AVX2=$<BOOL:${HAS_AVX2}>
    HAS_AVX512F=$<BOOL:${HAS_AVX512F}>
    HAS_FMA3=$<BOOL:${HAS_FMA3}>
    HAS_AVX_VNNI=$<BOOL:${HAS_AVX_VNNI}>
    HAS_OPENMP=$<BOOL:${HAS_OPENMP}>
)

if(NOT HAS_AVX512F)
    target_compile_definitions(hypercube_core PRIVATE
        DISABLE_AVX512
        DISABLE_AVX512F
        DISABLE_AVX512DQ
        DISABLE_AVX512BW
        DISABLE_AVX512VL
        DISABLE_AVX512VNNI
    )
endif()

if(WIN32)
    target_compile_definitions(hypercube_core PRIVATE NOMINMAX)
endif()

# C bridge helper
function(hc_add_c_bridge NAME SRC_FILE EXPORT_DEF)
    add_library(${NAME} SHARED "${SRC_FILE}")

    target_include_directories(${NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(${NAME}
        PRIVATE hypercube_core
        PRIVATE Threads::Threads
    )

    target_compile_options(${NAME}
        PRIVATE
            $<$<CONFIG:Release>:${HYPERCUBE_OPT_FLAGS}>
            $<$<CONFIG:Release>:${HYPERCUBE_WARNING_FLAGS}>
            $<$<CONFIG:Release>:${HYPERCUBE_AVX_FLAGS}>
            $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_DEBUG_FLAGS}>
            $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_WARNING_FLAGS}>
    )

    if(WIN32 AND EXPORT_DEF)
        target_compile_definitions(${NAME} PRIVATE ${EXPORT_DEF})
    endif()

    set_target_properties(${NAME} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${NAME}"
    )
endfunction()

hc_add_c_bridge(hypercube_c  src/bridge/hypercube_c.cpp  HYPERCUBE_C_EXPORTS)
hc_add_c_bridge(embedding_c  src/bridge/embedding_c.cpp  EMBEDDING_C_EXPORTS)
hc_add_c_bridge(generative_c src/bridge/generative_c.cpp GENERATIVE_C_EXPORTS)
