#include "hypercube/hilbert.hpp"

#ifdef __BMI2__
#include <immintrin.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <cassert>

namespace hypercube {

/**
 * 4D Hilbert Curve - Coordinate to Index mapping
 * 
 * 32 bits per dimension × 4 dimensions = 128 bits total
 * Output: two 64-bit unsigned integers (lo, hi)
 * 
 * The Hilbert index is DERIVED from coordinates for ordering/indexing.
 * Coordinates are the source of truth.
 * 
 * Algorithm: Skilling's compact Hilbert index
 */

namespace {

} // anonymous namespace

void HilbertCurve::transpose_to_axes(uint32_t* x, uint32_t n, uint32_t bits) noexcept {
    // Skilling's algorithm: convert from transpose (Hilbert) to axes (Cartesian)

    // Gray decode
    uint32_t t = x[n - 1] >> 1;
    for (uint32_t i = n - 1; i > 0; --i) {
        x[i] ^= x[i - 1];
    }
    x[0] ^= t;

    // Undo rotations - use unsigned arithmetic, consistent with axes_to_transpose
    for (uint32_t b = 1; b < bits; ++b) {
        uint64_t Q = 1ULL << b;
        uint64_t P = Q - 1ULL;
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            if (x[i] & static_cast<uint32_t>(Q)) {
                x[0] ^= static_cast<uint32_t>(P);
            } else {
                t = (x[0] ^ x[i]) & static_cast<uint32_t>(P);
                x[0] ^= t;
                x[i] ^= t;
            }
        }
    }
}

void HilbertCurve::axes_to_transpose(uint32_t* x, uint32_t n, uint32_t bits) noexcept {
    // Skilling's algorithm: convert from axes (Cartesian) to transpose (Hilbert)

    // Rotations - iterate from bits-1 down to 1
    for (int b = static_cast<int>(bits) - 1; b >= 1; --b) {
        uint64_t Q = 1ULL << b;
        uint64_t P = Q - 1ULL;
        for (uint32_t i = 0; i < n; ++i) {
            if (x[i] & static_cast<uint32_t>(Q)) {
                x[0] ^= static_cast<uint32_t>(P);
            } else {
                uint32_t t = (x[0] ^ x[i]) & static_cast<uint32_t>(P);
                x[0] ^= t;
                x[i] ^= t;
            }
        }
    }
    
    // Gray encode
    for (uint32_t i = 1; i < n; ++i) {
        x[i] ^= x[i - 1];
    }
    uint32_t t = 0;
    for (int b = static_cast<int>(bits) - 1; b >= 1; --b) {
        uint64_t Q = 1ULL << b;
        if (x[n - 1] & static_cast<uint32_t>(Q)) t ^= static_cast<uint32_t>((Q - 1));
    }
    for (uint32_t i = 0; i < n; ++i) {
        x[i] ^= t;
    }
}

HilbertIndex HilbertCurve::coords_to_index(const Point4D& point) noexcept {
    // COORDINATE CONVENTION: uint32 with CENTER at 2^31 = 0x80000000
    // Hilbert curve works on corner-origin coordinates [0, 2^32-1]
    // XOR with 0x80000000 maps our CENTER to Hilbert corner 0
    // This means compositions near origin (abstract/complex) cluster at Hilbert index 0
    // and atoms on sphere surface cluster at higher Hilbert indices
    constexpr uint32_t CENTER_TO_CORNER = 0x80000000U;
    uint32_t X[4] = {
        point.x ^ CENTER_TO_CORNER,
        point.y ^ CENTER_TO_CORNER,
        point.z ^ CENTER_TO_CORNER,
        point.m ^ CENTER_TO_CORNER
    };
    axes_to_transpose(X, 4, 32);
    
    // Interleave the 4 transposed coordinates into 128-bit index using BMI2
    // Bit-plane interleave: bit i of dimension d goes to position i*4 + d
    HilbertIndex result{0, 0};

#ifdef __BMI2__
    // Use BMI2 _pdep_u64 for hardware-accelerated bit deposition
    // Masks for bit positions: for dimension d, bits at 4*i + d
    const uint64_t mask_low[4] = {
        0x1111111111111111ULL,  // d=0: bits 0,4,8,...,60
        0x2222222222222222ULL,  // d=1: bits 1,5,9,...,61
        0x4444444444444444ULL,  // d=2: bits 2,6,10,...,62
        0x8888888888888888ULL   // d=3: bits 3,7,11,...,63
    };
    const uint64_t mask_high[4] = {
        0x1111111111111111ULL,  // Upper 64: same pattern
        0x2222222222222222ULL,
        0x4444444444444444ULL,
        0x8888888888888888ULL
    };

    for (int d = 0; d < 4; ++d) {
        uint32_t x = X[d];
        // Deposit lower 16 bits into lower 64 positions
        uint64_t low = _pdep_u64(x & 0xFFFF, mask_low[d]);
        // Deposit upper 16 bits into upper 64 positions
        uint64_t high = _pdep_u64((x >> 16) & 0xFFFF, mask_high[d]);
        result.lo |= low;
        result.hi |= high;
    }
#else
    // Fallback: original bit-loop implementation
    for (uint32_t bit = 0; bit < 32; ++bit) {
        uint32_t out_pos = bit * 4;
        uint64_t nibble = ((X[0] >> bit) & 1) |
                          (((X[1] >> bit) & 1) << 1) |
                          (((X[2] >> bit) & 1) << 2) |
                          (((X[3] >> bit) & 1) << 3);

        if (out_pos < 64) {
            result.lo |= (nibble << out_pos);
        } else {
            result.hi |= (nibble << (out_pos - 64));
        }
    }
#endif
    
    return result;
}

Point4D HilbertCurve::index_to_coords(const HilbertIndex& index) noexcept {
    // De-interleave 128-bit index into 4 transposed coordinates
    uint32_t X[4] = {0, 0, 0, 0};
    
    for (uint32_t bit = 0; bit < 32; ++bit) {
        uint32_t in_pos = bit * 4;
        uint64_t nibble;
        
        if (in_pos < 64) {
            nibble = (index.lo >> in_pos) & 0xF;
        } else {
            nibble = (index.hi >> (in_pos - 64)) & 0xF;
        }
        
        X[0] |= ((nibble >> 0) & 1) << bit;
        X[1] |= ((nibble >> 1) & 1) << bit;
        X[2] |= ((nibble >> 2) & 1) << bit;
        X[3] |= ((nibble >> 3) & 1) << bit;
    }
    
    // Transform back to Cartesian (corner-origin)
    transpose_to_axes(X, 4, 32);

    // Convert back from Hilbert corner-origin to our CENTER-origin coords
    // XOR with 0x80000000 maps Hilbert 0 back to our CENTER (0x80000000)
    constexpr uint32_t CORNER_TO_CENTER = 0x80000000U;
    return Point4D(
        X[0] ^ CORNER_TO_CENTER,
        X[1] ^ CORNER_TO_CENTER,
        X[2] ^ CORNER_TO_CENTER,
        X[3] ^ CORNER_TO_CENTER
    );
}

Point4D HilbertCurve::index_to_raw_coords(const HilbertIndex& index) noexcept {
    // De-interleave 128-bit index into 4 transposed coordinates
    uint32_t X[4] = {0, 0, 0, 0};
    
    for (uint32_t bit = 0; bit < 32; ++bit) {
        uint32_t in_pos = bit * 4;
        uint64_t nibble;
        
        if (in_pos < 64) {
            nibble = (index.lo >> in_pos) & 0xF;
        } else {
            nibble = (index.hi >> (in_pos - 64)) & 0xF;
        }
        
        X[0] |= ((nibble >> 0) & 1) << bit;
        X[1] |= ((nibble >> 1) & 1) << bit;
        X[2] |= ((nibble >> 2) & 1) << bit;
        X[3] |= ((nibble >> 3) & 1) << bit;
    }
    
    // Transform back to Cartesian (corner-origin)
    // NO CENTER adjustment - returns raw corner-origin coordinates
    transpose_to_axes(X, 4, 32);

    return Point4D(X[0], X[1], X[2], X[3]);
}

HilbertIndex HilbertCurve::distance(const HilbertIndex& a, const HilbertIndex& b) noexcept {
    return HilbertIndex::abs_distance(a, b);
}

bool HilbertCurve::in_range(const HilbertIndex& center, const HilbertIndex& point,
                            const HilbertIndex& range) noexcept {
    return distance(center, point) <= range;
}

/* ============================================================================
 * SIMD Batch Processing Functions
 * ============================================================================ */

/**
 * Batch coordinate to index conversion using AVX512 SIMD
 * Processes up to 16 coordinates at once for maximum throughput
 */
void HilbertCurve::coords_to_indices_batch_avx512(const Point4D* points,
                                                  size_t count,
                                                  HilbertIndex* indices) noexcept {
#ifdef __AVX512F__
    // AVX512 can process 16 coordinates in parallel (512 bits / 32 bits per coord = 16)
    const size_t batch_size = 16;

    for (size_t i = 0; i < count; i += batch_size) {
        size_t current_batch = std::min(batch_size, count - i);

        // Load coordinates into AVX512 registers
        // We need to load 4 coordinates per batch element (x,y,z,m)
        // Each coordinate is 32 bits, so we can fit 16 coordinates in one __m512i

        if (current_batch == batch_size) {
            // Full AVX512 vectorization
            constexpr uint32_t CENTER_TO_CORNER = 0x80000000U;

            // Load coordinates into batch arrays
            uint32_t x_batch[16], y_batch[16], z_batch[16], m_batch[16];
            for (size_t j = 0; j < 16; ++j) {
                const Point4D& point = points[i + j];
                x_batch[j] = point.x ^ CENTER_TO_CORNER;
                y_batch[j] = point.y ^ CENTER_TO_CORNER;
                z_batch[j] = point.z ^ CENTER_TO_CORNER;
                m_batch[j] = point.m ^ CENTER_TO_CORNER;
            }

            // Perform axes_to_transpose for each coordinate set
            uint32_t X[16][4];
            for (size_t j = 0; j < 16; ++j) {
                X[j][0] = x_batch[j];
                X[j][1] = y_batch[j];
                X[j][2] = z_batch[j];
                X[j][3] = m_batch[j];
                axes_to_transpose(X[j], 4, 32);
            }

            // Vectorized bit interleaving
            __m512i lo0 = _mm512_setzero_si512();
            __m512i lo1 = _mm512_setzero_si512();
            __m512i hi0 = _mm512_setzero_si512();
            __m512i hi1 = _mm512_setzero_si512();

            for (uint32_t bit = 0; bit < 32; ++bit) {
                __m512i bits0 = _mm512_setzero_si512();
                __m512i bits1 = _mm512_setzero_si512();
                __m512i bits2 = _mm512_setzero_si512();
                __m512i bits3 = _mm512_setzero_si512();

                for (size_t j = 0; j < 16; ++j) {
                    reinterpret_cast<uint32_t*>(&bits0)[j] = (X[j][0] >> bit) & 1;
                    reinterpret_cast<uint32_t*>(&bits1)[j] = (X[j][1] >> bit) & 1;
                    reinterpret_cast<uint32_t*>(&bits2)[j] = (X[j][2] >> bit) & 1;
                    reinterpret_cast<uint32_t*>(&bits3)[j] = (X[j][3] >> bit) & 1;
                }

                __m512i nibbles = _mm512_or_si512(bits0,
                    _mm512_or_si512(_mm512_slli_epi32(bits1, 1),
                        _mm512_or_si512(_mm512_slli_epi32(bits2, 2), _mm512_slli_epi32(bits3, 3))));

                int shift = bit * 4;
                if (shift < 64) {
                    __m512i shifted_low = _mm512_slli_epi64(_mm512_cvtepi32_epi64(_mm512_castsi512_si256(nibbles)), shift);
                    lo0 = _mm512_or_si512(lo0, shifted_low);
                    __m512i shifted_high = _mm512_slli_epi64(_mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(nibbles, 1)), shift);
                    lo1 = _mm512_or_si512(lo1, shifted_high);
                } else {
                    int shift_hi = shift - 64;
                    __m512i shifted_low = _mm512_slli_epi64(_mm512_cvtepi32_epi64(_mm512_castsi512_si256(nibbles)), shift_hi);
                    hi0 = _mm512_or_si512(hi0, shifted_low);
                    __m512i shifted_high = _mm512_slli_epi64(_mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(nibbles, 1)), shift_hi);
                    hi1 = _mm512_or_si512(hi1, shifted_high);
                }
            }

            // Store results
            uint64_t lo_array[16];
            uint64_t hi_array[16];
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&lo_array[0]), lo0);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&lo_array[8]), lo1);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&hi_array[0]), hi0);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&hi_array[8]), hi1);

            for (size_t j = 0; j < 16; ++j) {
                indices[i + j].lo = lo_array[j];
                indices[i + j].hi = hi_array[j];
            }
        } else {
            // Fallback to scalar processing for partial batches
            for (size_t j = 0; j < current_batch; ++j) {
                const Point4D& point = points[i + j];
                indices[i + j] = HilbertCurve::coords_to_index(point);
            }
        }
    }
#else
    // Fallback to scalar processing
    for (size_t i = 0; i < count; ++i) {
        indices[i] = HilbertCurve::coords_to_index(points[i]);
    }
#endif
}

/**
 * Batch index to coordinate conversion using AVX512 SIMD
 * Processes up to 16 indices at once
 */
void HilbertCurve::indices_to_coords_batch_avx512(const HilbertIndex* indices,
                                                  size_t count,
                                                  Point4D* points) noexcept {
#ifdef __AVX512F__
    // AVX512 implementation for batch index to coordinate conversion
    const size_t batch_size = 16;

    for (size_t i = 0; i < count; i += batch_size) {
        size_t current_batch = std::min(batch_size, count - i);

        if (current_batch == batch_size) {
            // Full AVX512 vectorization for de-interleaving
            constexpr uint32_t CORNER_TO_CENTER = 0x80000000U;

            // Load indices into arrays
            uint64_t lo_batch[16], hi_batch[16];
            for (size_t j = 0; j < 16; ++j) {
                lo_batch[j] = indices[i + j].lo;
                hi_batch[j] = indices[i + j].hi;
            }

            // De-interleave bits
            __m512i x_coords = _mm512_setzero_si512();
            __m512i y_coords = _mm512_setzero_si512();
            __m512i z_coords = _mm512_setzero_si512();
            __m512i m_coords = _mm512_setzero_si512();

            for (uint32_t bit = 0; bit < 32; ++bit) {
                __m512i nibbles = _mm512_setzero_si512();
                int shift = bit * 4;
                for (size_t j = 0; j < 16; ++j) {
                    uint64_t nibble;
                    if (shift < 64) {
                        nibble = (lo_batch[j] >> shift) & 0xF;
                    } else {
                        nibble = (hi_batch[j] >> (shift - 64)) & 0xF;
                    }
                    reinterpret_cast<uint32_t*>(&nibbles)[j] = static_cast<uint32_t>(nibble);
                }

                // Extract bits for each dimension
                __m512i bit0 = _mm512_and_si512(nibbles, _mm512_set1_epi32(1));
                __m512i bit1 = _mm512_and_si512(_mm512_srli_epi32(nibbles, 1), _mm512_set1_epi32(1));
                __m512i bit2 = _mm512_and_si512(_mm512_srli_epi32(nibbles, 2), _mm512_set1_epi32(1));
                __m512i bit3 = _mm512_and_si512(_mm512_srli_epi32(nibbles, 3), _mm512_set1_epi32(1));

                x_coords = _mm512_or_si512(x_coords, _mm512_slli_epi32(bit0, bit));
                y_coords = _mm512_or_si512(y_coords, _mm512_slli_epi32(bit1, bit));
                z_coords = _mm512_or_si512(z_coords, _mm512_slli_epi32(bit2, bit));
                m_coords = _mm512_or_si512(m_coords, _mm512_slli_epi32(bit3, bit));
            }

            // Perform transpose_to_axes for each coordinate set
            uint32_t X_batch[16][4];
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&X_batch[0][0]), x_coords);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&X_batch[8][0]), _mm512_permutexvar_epi32(_mm512_setr_epi32(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7), x_coords)); // wait, need to store properly

            // Actually, since X_batch is array of arrays, better to store each dimension separately.
            uint32_t x_array[16], y_array[16], z_array[16], m_array[16];
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&x_array[0]), x_coords);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&x_array[8]), _mm512_extracti32x8_epi32(x_coords, 1));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&y_array[0]), y_coords);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&y_array[8]), _mm512_extracti32x8_epi32(y_coords, 1));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&z_array[0]), z_coords);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&z_array[8]), _mm512_extracti32x8_epi32(z_coords, 1));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&m_array[0]), m_coords);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&m_array[8]), _mm512_extracti32x8_epi32(m_coords, 1));

            for (size_t j = 0; j < 16; ++j) {
                uint32_t X[4] = {x_array[j], y_array[j], z_array[j], m_array[j]};
                transpose_to_axes(X, 4, 32);
                points[i + j] = Point4D(
                    X[0] ^ CORNER_TO_CENTER,
                    X[1] ^ CORNER_TO_CENTER,
                    X[2] ^ CORNER_TO_CENTER,
                    X[3] ^ CORNER_TO_CENTER
                );
            }
        } else {
            // Fallback to scalar processing for partial batches
            for (size_t j = 0; j < current_batch; ++j) {
                points[i + j] = index_to_coords(indices[i + j]);
            }
        }
    }
#else
    // Fallback to scalar processing
    for (size_t i = 0; i < count; ++i) {
        points[i] = index_to_coords(indices[i]);
    }
#endif
}

/**
 * Batch coordinate to index conversion using AVX2 SIMD
 * Processes up to 8 coordinates at once
 */
void HilbertCurve::coords_to_indices_batch_avx2(const Point4D* points,
                                                size_t count,
                                                HilbertIndex* indices) noexcept {
#ifdef __AVX2__
    // AVX2 can process 8 coordinates in parallel (256 bits / 32 bits per coord = 8)
    const size_t batch_size = 8;

    for (size_t i = 0; i < count; i += batch_size) {
        size_t current_batch = std::min(batch_size, count - i);

        if (current_batch == batch_size) {
            // Full AVX2 vectorization
            constexpr uint32_t CENTER_TO_CORNER = 0x80000000U;

            // Load coordinates into batch arrays
            uint32_t x_batch[8], y_batch[8], z_batch[8], m_batch[8];
            for (size_t j = 0; j < 8; ++j) {
                const Point4D& point = points[i + j];
                x_batch[j] = point.x ^ CENTER_TO_CORNER;
                y_batch[j] = point.y ^ CENTER_TO_CORNER;
                z_batch[j] = point.z ^ CENTER_TO_CORNER;
                m_batch[j] = point.m ^ CENTER_TO_CORNER;
            }

            // Perform axes_to_transpose for each coordinate set
            uint32_t X[8][4];
            for (size_t j = 0; j < 8; ++j) {
                X[j][0] = x_batch[j];
                X[j][1] = y_batch[j];
                X[j][2] = z_batch[j];
                X[j][3] = m_batch[j];
                axes_to_transpose(X[j], 4, 32);
            }

            // Vectorized bit interleaving
            __m256i lo0 = _mm256_setzero_si256();
            __m256i lo1 = _mm256_setzero_si256();
            __m256i hi0 = _mm256_setzero_si256();
            __m256i hi1 = _mm256_setzero_si256();

            for (uint32_t bit = 0; bit < 32; ++bit) {
                __m256i bits0 = _mm256_setzero_si256();
                __m256i bits1 = _mm256_setzero_si256();
                __m256i bits2 = _mm256_setzero_si256();
                __m256i bits3 = _mm256_setzero_si256();

                for (size_t j = 0; j < 8; ++j) {
                    reinterpret_cast<uint32_t*>(&bits0)[j] = (X[j][0] >> bit) & 1;
                    reinterpret_cast<uint32_t*>(&bits1)[j] = (X[j][1] >> bit) & 1;
                    reinterpret_cast<uint32_t*>(&bits2)[j] = (X[j][2] >> bit) & 1;
                    reinterpret_cast<uint32_t*>(&bits3)[j] = (X[j][3] >> bit) & 1;
                }

                __m256i nibbles = _mm256_or_si256(bits0,
                    _mm256_or_si256(_mm256_slli_epi32(bits1, 1),
                        _mm256_or_si256(_mm256_slli_epi32(bits2, 2), _mm256_slli_epi32(bits3, 3))));

                int shift = bit * 4;
                if (shift < 64) {
                    __m256i shifted = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(nibbles)), shift);
                    lo0 = _mm256_or_si256(lo0, shifted);
                    __m256i shifted_high = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(nibbles, 1)), shift);
                    lo1 = _mm256_or_si256(lo1, shifted_high);
                } else {
                    int shift_hi = shift - 64;
                    __m256i shifted = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(nibbles)), shift_hi);
                    hi0 = _mm256_or_si256(hi0, shifted);
                    __m256i shifted_high = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(nibbles, 1)), shift_hi);
                    hi1 = _mm256_or_si256(hi1, shifted_high);
                }
            }

            // Store results
            uint64_t lo_array[8];
            uint64_t hi_array[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&lo_array[0]), lo0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&lo_array[4]), lo1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&hi_array[0]), hi0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&hi_array[4]), hi1);

            for (size_t j = 0; j < 8; ++j) {
                indices[i + j].lo = lo_array[j];
                indices[i + j].hi = hi_array[j];
            }
        } else {
            // Fallback to scalar processing for partial batches
            for (size_t j = 0; j < current_batch; ++j) {
                const Point4D& point = points[i + j];
                indices[i + j] = coords_to_index(points[i + j]);
            }
        }
    }
#else
    // Fallback to scalar processing
    for (size_t i = 0; i < count; ++i) {
        indices[i] = coords_to_index(points[i]);
    }
#endif
}

// Test function for SIMD correctness
void test_hilbert_simd_correctness() {
    // Test AVX512 batch functions
#ifdef __AVX512F__
    Point4D points_avx512[16];
    for (int i = 0; i < 16; ++i) {
        points_avx512[i] = Point4D(
            static_cast<uint32_t>(i * 4),
            static_cast<uint32_t>(i * 4 + 1),
            static_cast<uint32_t>(i * 4 + 2),
            static_cast<uint32_t>(i * 4 + 3)
        );
    }

    HilbertIndex indices_avx512[16];
    HilbertCurve::coords_to_indices_batch_avx512(points_avx512, 16, indices_avx512);

    Point4D back_avx512[16];
    HilbertCurve::indices_to_coords_batch_avx512(indices_avx512, 16, back_avx512);

    for (int i = 0; i < 16; ++i) {
        assert(back_avx512[i].x == points_avx512[i].x);
        assert(back_avx512[i].y == points_avx512[i].y);
        assert(back_avx512[i].z == points_avx512[i].z);
        assert(back_avx512[i].m == points_avx512[i].m);
    }
#endif

    // Test AVX2 batch function
#ifdef __AVX2__
    Point4D points_avx2[8];
    for (int i = 0; i < 8; ++i) {
        points_avx2[i] = Point4D(
            static_cast<uint32_t>(i * 4),
            static_cast<uint32_t>(i * 4 + 1),
            static_cast<uint32_t>(i * 4 + 2),
            static_cast<uint32_t>(i * 4 + 3)
        );
    }

    HilbertIndex indices_avx2[8];
    HilbertCurve::coords_to_indices_batch_avx2(points_avx2, 8, indices_avx2);

    // Test round-trip using scalar for coords
    for (int i = 0; i < 8; ++i) {
        Point4D back = HilbertCurve::index_to_coords(indices_avx2[i]);
        assert(back.x == points_avx2[i].x);
        assert(back.y == points_avx2[i].y);
        assert(back.z == points_avx2[i].z);
        assert(back.m == points_avx2[i].m);
    }
#endif
}

} // namespace hypercube
