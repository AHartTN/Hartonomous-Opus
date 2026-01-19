add_executable(benchmark_suite benchmark_main.cpp
    matrix_benchmark.hpp
    linear_solve_benchmark.hpp
    fft_benchmark.hpp
    rng_benchmark.hpp
    eigen_matrix_multiply_benchmark.hpp
    eigen_lu_solve_benchmark.hpp
    eigen_svd_benchmark.hpp
    eigen_sparse_benchmark.hpp
    hnsw_index_build_benchmark.hpp
    hnsw_search_benchmark.hpp
    hnsw_insertion_benchmark.hpp
    simd_intrinsics_benchmark.hpp
    vnni_dot_product_benchmark.hpp
    avx_vector_arithmetic_benchmark.hpp
    hybrid_mkl_simd_benchmark.hpp
    hybrid_eigen_avx_benchmark.hpp
    memory_bandwidth_benchmark.hpp
    micro_benchmarks.hpp
)
target_link_libraries(benchmark_suite benchmark::benchmark Eigen3::Eigen hnswlib math)
if(MKL_FOUND)
  target_link_libraries(benchmark_suite MKL::MKL)
endif()
target_include_directories(benchmark_suite PRIVATE ${CMAKE_SOURCE_DIR}/src)