# Runtime dispatch system - unified kernel implementations
# All SIMD kernels are now implemented in a single file with runtime dispatch

# Core C++ library
add_library(hypercube_core STATIC
    src/core/hilbert.cpp
    src/core/coordinates.cpp
    src/core/blake3_pg.cpp
    src/core/ops.cpp
    src/util/utf8.cpp
    src/core/atom_calculator.cpp
    src/core/atom_registry.cpp
    src/core/deterministic_grid4d.cpp
    src/core/laplacian_4d.cpp
    src/core/lanczos.cpp
    src/core/linear_algebra.cpp
    src/core/semantic_ordering.cpp
    src/core/superfibonacci.cpp
    src/core/thread_config.cpp
    src/core/unicode_categorization.cpp
    src/core/coordinate_utilities.cpp
    src/core/dense_registry.cpp
    src/core/cpu_features.cpp
    src/core/isa_class.cpp
    src/core/dispatch.cpp
    src/core/runtime_dispatch.cpp
    src/core/kernel_dispatch.cpp
    src/core/function_pointers.cpp
    src/simd_kernels_distance.cpp
    src/simd_kernels_vector.cpp
)

target_include_directories(hypercube_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Per-target compiler flags (native binary with SIMD flags)
target_compile_options(hypercube_core
    PRIVATE
        $<$<CONFIG:Release>:${HYPERCUBE_OPT_FLAGS}>
        $<$<CONFIG:Release>:${HYPERCUBE_WARNING_FLAGS}>
        $<$<CONFIG:Release>:${HYPERCUBE_SIMD_FLAGS}>
        $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_DEBUG_FLAGS}>
        $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_WARNING_FLAGS}>
        $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_SIMD_FLAGS}>
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

# HNSWLib
if(HAS_HNSWLIB)
    target_include_directories(hypercube_core PUBLIC ${HNSWLIB_INCLUDE_DIRS})
    target_compile_definitions(hypercube_core PUBLIC HAS_HNSWLIB=1)
else()
    target_compile_definitions(hypercube_core PUBLIC HAS_HNSWLIB=0)
endif()

# BLAKE3 (with SIMD optimizations)
if(HAS_BLAKE3)
    if(TARGET BLAKE3::blake3)
        target_link_libraries(hypercube_core PRIVATE BLAKE3::blake3)
    elseif(TARGET blake3)
        target_link_libraries(hypercube_core PRIVATE blake3)
    endif()
    target_compile_definitions(hypercube_core PRIVATE USE_OFFICIAL_BLAKE3=1)
endif()

# Runtime dispatch enabled - SIMD flags applied
target_compile_definitions(hypercube_core PUBLIC
    HAS_OPENMP=$<BOOL:${HAS_OPENMP}>
    RUNTIME_DISPATCH_ENABLED=1
    HAS_AVX=1
)

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
            $<$<CONFIG:Release>:${HYPERCUBE_SIMD_FLAGS}>
            $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_DEBUG_FLAGS}>
            $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_WARNING_FLAGS}>
            $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${HYPERCUBE_SIMD_FLAGS}>
    )

    if(WIN32 AND EXPORT_DEF)
        target_compile_definitions(${NAME} PRIVATE ${EXPORT_DEF})
    endif()

    set_target_properties(${NAME} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${NAME}"
        INSTALL_RPATH "$ORIGIN"
    )
endfunction()

hc_add_c_bridge(hypercube_c  src/bridge/hypercube_c.cpp  HYPERCUBE_C_EXPORTS)
hc_add_c_bridge(embedding_c  src/bridge/embedding_c.cpp  EMBEDDING_C_EXPORTS)
hc_add_c_bridge(generative_c src/bridge/generative_c.cpp GENERATIVE_C_EXPORTS)
