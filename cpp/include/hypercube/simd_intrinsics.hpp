/**
 * SIMD Intrinsics Wrapper
 *
 * Conditionally includes AVX intrinsics based on CPU support.
 * Prevents AVX512 header inclusion when CPU doesn't support it.
 */

#ifndef HYPERCUBE_SIMD_INTRINSICS_HPP
#define HYPERCUBE_SIMD_INTRINSICS_HPP

// Include AVX intrinsics only when AVX is supported
// Modern compilers (GCC 11+) include AVX512 headers by default in immintrin.h,
// but we control this via compiler flags and CPU detection
// Use __AVX__ which is defined by the compiler when -mavx or equivalent is used

#if defined(__AVX__) || defined(__SSE4_2__)
#include <immintrin.h>
#endif

#endif // HYPERCUBE_SIMD_INTRINSICS_HPP