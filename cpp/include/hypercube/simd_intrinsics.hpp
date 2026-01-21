/**
 * SIMD Intrinsics Wrapper for Runtime Dispatch
 *
 * Always includes SIMD intrinsics headers for runtime dispatch.
 * Runtime checks prevent illegal instruction exceptions.
 */

#ifndef HYPERCUBE_SIMD_INTRINSICS_HPP
#define HYPERCUBE_SIMD_INTRINSICS_HPP

// Always include SIMD intrinsics for runtime dispatch
// Runtime CPU feature detection and safety checks prevent illegal instructions
#include <immintrin.h>

#endif // HYPERCUBE_SIMD_INTRINSICS_HPP