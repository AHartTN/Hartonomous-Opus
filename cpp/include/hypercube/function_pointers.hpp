#pragma once

#include <cstddef>

namespace hypercube {

/**
 * Global function pointers for runtime dispatch to optimal SIMD implementations.
 * These pointers are initialized once at startup based on detected CPU capabilities,
 * eliminating runtime selection overhead for subsequent operations.
 */

// Distance functions
extern double (*distance_l2_f32)(const float* a, const float* b, size_t n);
extern double (*distance_ip_f32)(const float* a, const float* b, size_t n);

// Matrix operations
extern void (*gemm_f32)(float alpha, const float* A, size_t m, size_t k,
                       const float* B, size_t n, float beta, float* C);

// Vector operations - double precision
extern double (*dot_product_d)(const double* a, const double* b, size_t n);
extern void (*scale_inplace_d)(double* v, double s, size_t n);
extern void (*subtract_scaled_d)(double* a, const double* b, double s, size_t n);
extern double (*norm_d)(const double* v, size_t n);

// Vector operations - single precision
extern float (*dot_product_f)(const float* a, const float* b, size_t n);

/**
 * Initialize all function pointers for optimal runtime dispatch.
 * This function should be called once at application startup.
 * Uses std::call_once for thread-safe initialization.
 */
void initialize_function_pointers();

} // namespace hypercube