/**
 * Unified Kernel Dispatch Interface
 *
 * Provides a single entry point for all kernel operations with automatic
 * runtime dispatch based on CPU capabilities and safe fallback mechanisms.
 */

#pragma once

#include <cstdint>
#include "runtime_dispatch.h"

namespace hypercube {
namespace kernels {

/**
 * Distance computation functions with runtime dispatch
 */
namespace distance {

double l2(const float* a, const float* b, size_t n);
double ip(const float* a, const float* b, size_t n);

} // namespace distance

/**
 * Matrix operations with runtime dispatch
 */
namespace matrix {

void gemm_f32(float alpha, const float* A, size_t m, size_t k,
              const float* B, size_t n, float beta, float* C);

} // namespace matrix

/**
 * Vector operations with runtime dispatch
 */
namespace vector {

double dot_product(const double* a, const double* b, size_t n);
float dot_product(const float* a, const float* b, size_t n);
void scale_inplace(double* v, double s, size_t n);
void subtract_scaled(double* a, const double* b, double s, size_t n);
double norm(const double* v, size_t n);

} // namespace vector

/**
 * Internal kernel implementations by capability
 * These are called by the dispatch system
 */

namespace impl {

// Distance kernels
double distance_l2_baseline(const float* a, const float* b, size_t n);
double distance_l2_sse42(const float* a, const float* b, size_t n);
double distance_l2_avx2(const float* a, const float* b, size_t n);
double distance_l2_avx512(const float* a, const float* b, size_t n);
double distance_l2_amx(const float* a, const float* b, size_t n);

double distance_ip_baseline(const float* a, const float* b, size_t n);
double distance_ip_sse42(const float* a, const float* b, size_t n);
double distance_ip_avx2(const float* a, const float* b, size_t n);
double distance_ip_avx512(const float* a, const float* b, size_t n);
double distance_ip_amx(const float* a, const float* b, size_t n);

// Matrix kernels
void gemm_f32_baseline(float alpha, const float* A, size_t m, size_t k,
                      const float* B, size_t n, float beta, float* C);
void gemm_f32_sse42(float alpha, const float* A, size_t m, size_t k,
                   const float* B, size_t n, float beta, float* C);
void gemm_f32_avx2(float alpha, const float* A, size_t m, size_t k,
                  const float* B, size_t n, float beta, float* C);
void gemm_f32_avx512(float alpha, const float* A, size_t m, size_t k,
                    const float* B, size_t n, float beta, float* C);
void gemm_f32_amx(float alpha, const float* A, size_t m, size_t k,
                 const float* B, size_t n, float beta, float* C);

// Vector kernels
double dot_product_d_baseline(const double* a, const double* b, size_t n);
double dot_product_d_sse42(const double* a, const double* b, size_t n);
double dot_product_d_avx2(const double* a, const double* b, size_t n);
double dot_product_d_avx512(const double* a, const double* b, size_t n);
double dot_product_d_amx(const double* a, const double* b, size_t n);

float dot_product_f_baseline(const float* a, const float* b, size_t n);
float dot_product_f_sse42(const float* a, const float* b, size_t n);
float dot_product_f_avx2(const float* a, const float* b, size_t n);
float dot_product_f_avx512(const float* a, const float* b, size_t n);
float dot_product_f_amx(const float* a, const float* b, size_t n);

void scale_inplace_d_baseline(double* v, double s, size_t n);
void scale_inplace_d_sse42(double* v, double s, size_t n);
void scale_inplace_d_avx2(double* v, double s, size_t n);
void scale_inplace_d_avx512(double* v, double s, size_t n);
void scale_inplace_d_amx(double* v, double s, size_t n);

void subtract_scaled_d_baseline(double* a, const double* b, double s, size_t n);
void subtract_scaled_d_sse42(double* a, const double* b, double s, size_t n);
void subtract_scaled_d_avx2(double* a, const double* b, double s, size_t n);
void subtract_scaled_d_avx512(double* a, const double* b, double s, size_t n);
void subtract_scaled_d_amx(double* a, const double* b, double s, size_t n);

double norm_d_baseline(const double* v, size_t n);
double norm_d_sse42(const double* v, size_t n);
double norm_d_avx2(const double* v, size_t n);
double norm_d_avx512(const double* v, size_t n);
double norm_d_amx(const double* v, size_t n);

} // namespace impl

} // namespace kernels
} // namespace hypercube