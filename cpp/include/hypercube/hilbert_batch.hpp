/**
 * Batch Hilbert Curve Operations with SIMD/BMI2 Optimization
 * 
 * Parallelized batch encoding/decoding of 4D coordinates to 128-bit Hilbert indices.
 * Uses BMI2 instructions (PEXT/PDEP) when available for fast bit manipulation.
 */

#pragma once

#include "hypercube/hilbert.hpp"
#include "hypercube/thread_pool.hpp"
#include <vector>
#include <array>

#if defined(__BMI2__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#define HAS_BMI2 1
#else
#define HAS_BMI2 0
#endif

namespace hypercube {

/**
 * Batch Hilbert operations with parallelization and SIMD
 */
class HilbertBatch {
public:
    /**
     * Encode multiple points to Hilbert indices in parallel
     */
    static std::vector<HilbertIndex> batch_encode(
        const std::vector<Point4D>& points,
        size_t num_threads = 0
    ) {
        if (points.empty()) return {};
        
        const size_t n = points.size();
        std::vector<HilbertIndex> indices(n);
        
        auto& pool = ThreadPool::instance();
        pool.parallel_for(0, n, [&](size_t i) {
            indices[i] = encode_fast(points[i]);
        });
        
        return indices;
    }
    
    /**
     * Encode multiple coordinate arrays to Hilbert indices in parallel
     */
    static std::vector<HilbertIndex> batch_encode(
        const std::vector<std::array<uint32_t, 4>>& coords,
        size_t num_threads = 0
    ) {
        if (coords.empty()) return {};
        
        const size_t n = coords.size();
        std::vector<HilbertIndex> indices(n);
        
        auto& pool = ThreadPool::instance();
        pool.parallel_for(0, n, [&](size_t i) {
            Point4D pt(coords[i][0], coords[i][1], coords[i][2], coords[i][3]);
            indices[i] = encode_fast(pt);
        });
        
        return indices;
    }
    
    /**
     * Decode multiple Hilbert indices to points in parallel
     */
    static std::vector<Point4D> batch_decode(
        const std::vector<HilbertIndex>& indices,
        size_t num_threads = 0
    ) {
        if (indices.empty()) return {};
        
        const size_t n = indices.size();
        std::vector<Point4D> points(n);
        
        auto& pool = ThreadPool::instance();
        pool.parallel_for(0, n, [&](size_t i) {
            points[i] = decode_fast(indices[i]);
        });
        
        return points;
    }
    
    /**
     * Encode and store both point and index (common pattern)
     */
    struct PointWithIndex {
        Point4D point;
        HilbertIndex index;
    };
    
    static std::vector<PointWithIndex> batch_encode_with_points(
        const std::vector<Point4D>& points,
        size_t num_threads = 0
    ) {
        if (points.empty()) return {};
        
        const size_t n = points.size();
        std::vector<PointWithIndex> results(n);
        
        auto& pool = ThreadPool::instance();
        pool.parallel_for(0, n, [&](size_t i) {
            results[i].point = points[i];
            results[i].index = encode_fast(points[i]);
        });
        
        return results;
    }
    
private:
    /**
     * Fast Hilbert encoding with BMI2 when available
     */
    static HilbertIndex encode_fast(const Point4D& point) noexcept {
#if HAS_BMI2
        return encode_bmi2(point);
#else
        return HilbertCurve::coords_to_index(point);
#endif
    }
    
    static Point4D decode_fast(const HilbertIndex& index) noexcept {
#if HAS_BMI2
        return decode_bmi2(index);
#else
        return HilbertCurve::index_to_coords(index);
#endif
    }

#if HAS_BMI2
    /**
     * BMI2-accelerated Hilbert encoding using PDEP for bit interleaving
     */
    static HilbertIndex encode_bmi2(const Point4D& point) noexcept {
        // Still need the Skilling transform, then BMI2 for interleaving
        constexpr uint32_t CENTER_TO_CORNER = 0x80000000U;
        uint32_t X[4] = {
            point.x ^ CENTER_TO_CORNER,
            point.y ^ CENTER_TO_CORNER,
            point.z ^ CENTER_TO_CORNER,
            point.m ^ CENTER_TO_CORNER
        };
        
        // Skilling's axes_to_transpose
        axes_to_transpose_inline(X, 4, 32);
        
        // BMI2 PDEP for fast bit interleaving
        // Interleave 4 32-bit values into 128 bits (4 bits per position)
        // Pattern: bit i of dim d goes to position i*4 + d
        
        // Lower 64 bits: bits 0-15 of each dimension
        // Upper 64 bits: bits 16-31 of each dimension
        
        constexpr uint64_t MASK_0 = 0x1111111111111111ULL;  // Every 4th bit starting at 0
        constexpr uint64_t MASK_1 = 0x2222222222222222ULL;  // Every 4th bit starting at 1
        constexpr uint64_t MASK_2 = 0x4444444444444444ULL;  // Every 4th bit starting at 2
        constexpr uint64_t MASK_3 = 0x8888888888888888ULL;  // Every 4th bit starting at 3
        
        // Extract low and high 16 bits of each coordinate
        uint64_t x_lo = X[0] & 0xFFFF, x_hi = X[0] >> 16;
        uint64_t y_lo = X[1] & 0xFFFF, y_hi = X[1] >> 16;
        uint64_t z_lo = X[2] & 0xFFFF, z_hi = X[2] >> 16;
        uint64_t m_lo = X[3] & 0xFFFF, m_hi = X[3] >> 16;
        
        HilbertIndex result;
        result.lo = _pdep_u64(x_lo, MASK_0) | _pdep_u64(y_lo, MASK_1) |
                    _pdep_u64(z_lo, MASK_2) | _pdep_u64(m_lo, MASK_3);
        result.hi = _pdep_u64(x_hi, MASK_0) | _pdep_u64(y_hi, MASK_1) |
                    _pdep_u64(z_hi, MASK_2) | _pdep_u64(m_hi, MASK_3);
        
        return result;
    }
    
    /**
     * BMI2-accelerated Hilbert decoding using PEXT for bit de-interleaving
     */
    static Point4D decode_bmi2(const HilbertIndex& index) noexcept {
        constexpr uint64_t MASK_0 = 0x1111111111111111ULL;
        constexpr uint64_t MASK_1 = 0x2222222222222222ULL;
        constexpr uint64_t MASK_2 = 0x4444444444444444ULL;
        constexpr uint64_t MASK_3 = 0x8888888888888888ULL;
        
        // De-interleave using PEXT
        uint64_t x_lo = _pext_u64(index.lo, MASK_0);
        uint64_t y_lo = _pext_u64(index.lo, MASK_1);
        uint64_t z_lo = _pext_u64(index.lo, MASK_2);
        uint64_t m_lo = _pext_u64(index.lo, MASK_3);
        
        uint64_t x_hi = _pext_u64(index.hi, MASK_0);
        uint64_t y_hi = _pext_u64(index.hi, MASK_1);
        uint64_t z_hi = _pext_u64(index.hi, MASK_2);
        uint64_t m_hi = _pext_u64(index.hi, MASK_3);
        
        uint32_t X[4] = {
            static_cast<uint32_t>(x_lo | (x_hi << 16)),
            static_cast<uint32_t>(y_lo | (y_hi << 16)),
            static_cast<uint32_t>(z_lo | (z_hi << 16)),
            static_cast<uint32_t>(m_lo | (m_hi << 16))
        };
        
        // Inverse Skilling transform
        transpose_to_axes_inline(X, 4, 32);
        
        // Convert back to center-origin
        constexpr uint32_t CORNER_TO_CENTER = 0x80000000U;
        return Point4D(
            X[0] ^ CORNER_TO_CENTER,
            X[1] ^ CORNER_TO_CENTER,
            X[2] ^ CORNER_TO_CENTER,
            X[3] ^ CORNER_TO_CENTER
        );
    }
    
    // Inline versions of Skilling's transforms to avoid function call overhead
    static void axes_to_transpose_inline(uint32_t* x, uint32_t n, uint32_t bits) noexcept {
        for (uint32_t b = bits - 1; b >= 1; --b) {
            uint32_t Q = 1U << b;
            uint32_t P = Q - 1;
            for (uint32_t i = 0; i < n; ++i) {
                if (x[i] & Q) {
                    x[0] ^= P;
                } else {
                    uint32_t t = (x[0] ^ x[i]) & P;
                    x[0] ^= t;
                    x[i] ^= t;
                }
            }
        }
        for (uint32_t i = 1; i < n; ++i) {
            x[i] ^= x[i - 1];
        }
        uint32_t t = 0;
        for (uint32_t b = bits - 1; b >= 1; --b) {
            uint32_t Q = 1U << b;
            if (x[n - 1] & Q) t ^= (Q - 1);
        }
        for (uint32_t i = 0; i < n; ++i) {
            x[i] ^= t;
        }
    }
    
    static void transpose_to_axes_inline(uint32_t* x, uint32_t n, uint32_t bits) noexcept {
        uint32_t t = x[n - 1] >> 1;
        for (uint32_t i = n - 1; i > 0; --i) {
            x[i] ^= x[i - 1];
        }
        x[0] ^= t;
        for (uint32_t b = 1; b < bits; ++b) {
            uint32_t Q = 1U << b;
            uint32_t P = Q - 1;
            for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
                if (x[i] & Q) {
                    x[0] ^= P;
                } else {
                    t = (x[0] ^ x[i]) & P;
                    x[0] ^= t;
                    x[i] ^= t;
                }
            }
        }
    }
#endif
};

} // namespace hypercube
