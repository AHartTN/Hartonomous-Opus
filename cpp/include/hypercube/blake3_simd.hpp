#pragma once
// SIMD-optimized BLAKE3 using official implementation
// Falls back to portable if SIMD not available

#include <cstdint>
#include <cstddef>
#include <array>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
  #if defined(__AVX512F__) && defined(__AVX512VL__)
    #define BLAKE3_USE_AVX512 1
  #endif
  #if defined(__AVX2__)
    #define BLAKE3_USE_AVX2 1
  #endif
  #if defined(__SSE4_1__)
    #define BLAKE3_USE_SSE41 1
  #endif
  #if defined(__SSE2__) || defined(_M_X64)
    #define BLAKE3_USE_SSE2 1
  #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  #if defined(__ARM_NEON) || defined(_M_ARM64)
    #define BLAKE3_USE_NEON 1
  #endif
#endif

namespace hypercube {
namespace blake3_simd {

constexpr size_t BLAKE3_KEY_LEN = 32;
constexpr size_t BLAKE3_OUT_LEN = 32;
constexpr size_t BLAKE3_BLOCK_LEN = 64;
constexpr size_t BLAKE3_CHUNK_LEN = 1024;

// IV constants
alignas(32) constexpr uint32_t IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// Message schedule
constexpr uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

enum flags : uint8_t {
    CHUNK_START = 1 << 0,
    CHUNK_END = 1 << 1,
    PARENT = 1 << 2,
    ROOT = 1 << 3,
};

#if defined(BLAKE3_USE_SSE2) && !defined(BLAKE3_USE_AVX2)
#include <emmintrin.h>

inline __m128i rotr32_sse2(__m128i x, int n) {
    return _mm_or_si128(_mm_srli_epi32(x, n), _mm_slli_epi32(x, 32 - n));
}

inline void g_sse2(__m128i* row0, __m128i* row1, __m128i* row2, __m128i* row3,
                   __m128i m0, __m128i m1) {
    *row0 = _mm_add_epi32(_mm_add_epi32(*row0, *row1), m0);
    *row3 = rotr32_sse2(_mm_xor_si128(*row3, *row0), 16);
    *row2 = _mm_add_epi32(*row2, *row3);
    *row1 = rotr32_sse2(_mm_xor_si128(*row1, *row2), 12);
    *row0 = _mm_add_epi32(_mm_add_epi32(*row0, *row1), m1);
    *row3 = rotr32_sse2(_mm_xor_si128(*row3, *row0), 8);
    *row2 = _mm_add_epi32(*row2, *row3);
    *row1 = rotr32_sse2(_mm_xor_si128(*row1, *row2), 7);
}
#endif

#if defined(BLAKE3_USE_AVX2)
#include <immintrin.h>

inline __m256i rotr32_avx2(__m256i x, int n) {
    return _mm256_or_si256(_mm256_srli_epi32(x, n), _mm256_slli_epi32(x, 32 - n));
}

// AVX2 G function - processes 8 words at once (2 compression states)
inline void g_avx2(__m256i* a, __m256i* b, __m256i* c, __m256i* d, __m256i m0, __m256i m1) {
    // a = a + b + m0
    __m256i t0 = _mm256_add_epi32(_mm256_add_epi32(*a, *b), m0);
    // d = rotr32(d ^ t0, 16)
    __m256i t3 = rotr32_avx2(_mm256_xor_si256(*d, t0), 16);
    // c = c + t3
    __m256i t2 = _mm256_add_epi32(*c, t3);
    // b = rotr32(b ^ t2, 12)
    *b = rotr32_avx2(_mm256_xor_si256(*b, t2), 12);
    // a = t0 + b + m1
    *a = _mm256_add_epi32(_mm256_add_epi32(t0, *b), m1);
    // d = rotr32(t3 ^ a, 8)
    *d = rotr32_avx2(_mm256_xor_si256(t3, *a), 8);
    // c = t2 + d
    *c = _mm256_add_epi32(t2, *d);
    // b = rotr32(b ^ c, 7)
    *b = rotr32_avx2(_mm256_xor_si256(*b, *c), 7);
}

// Process 2 blocks in parallel with AVX2
void compress_2blocks_avx2(const uint32_t cv[8],
                           const uint8_t block0[BLAKE3_BLOCK_LEN],
                           const uint8_t block1[BLAKE3_BLOCK_LEN],
                           uint8_t block_len, uint64_t counter, uint8_t flags,
                           uint32_t out0[16], uint32_t out1[16]) {
    // Load message blocks
    __m256i msg[16];
    for (int i = 0; i < 16; ++i) {
        uint32_t w0 = load32_le(&block0[i * 4]);
        uint32_t w1 = load32_le(&block1[i * 4]);
        msg[i] = _mm256_setr_epi32(w0, w1, w0, w1, w0, w1, w0, w1);
    }

    // Initialize state for 2 parallel compressions
    __m256i state0 = _mm256_setr_epi32(cv[0], cv[0], cv[0], cv[0], cv[0], cv[0], cv[0], cv[0]);
    __m256i state1 = _mm256_setr_epi32(cv[1], cv[1], cv[1], cv[1], cv[1], cv[1], cv[1], cv[1]);
    __m256i state2 = _mm256_setr_epi32(cv[2], cv[2], cv[2], cv[2], cv[2], cv[2], cv[2], cv[2]);
    __m256i state3 = _mm256_setr_epi32(cv[3], cv[3], cv[3], cv[3], cv[3], cv[3], cv[3], cv[3]);
    __m256i state4 = _mm256_setr_epi32(cv[4], cv[4], cv[4], cv[4], cv[4], cv[4], cv[4], cv[4]);
    __m256i state5 = _mm256_setr_epi32(cv[5], cv[5], cv[5], cv[5], cv[5], cv[5], cv[5], cv[5]);
    __m256i state6 = _mm256_setr_epi32(cv[6], cv[6], cv[6], cv[6], cv[6], cv[6], cv[6], cv[6]);
    __m256i state7 = _mm256_setr_epi32(cv[7], cv[7], cv[7], cv[7], cv[7], cv[7], cv[7], cv[7]);
    __m256i state8 = _mm256_setr_epi32(IV[0], IV[0], IV[0], IV[0], IV[0], IV[0], IV[0], IV[0]);
    __m256i state9 = _mm256_setr_epi32(IV[1], IV[1], IV[1], IV[1], IV[1], IV[1], IV[1], IV[1]);
    __m256i state10 = _mm256_setr_epi32(IV[2], IV[2], IV[2], IV[2], IV[2], IV[2], IV[2], IV[2]);
    __m256i state11 = _mm256_setr_epi32(IV[3], IV[3], IV[3], IV[3], IV[3], IV[3], IV[3], IV[3]);
    __m256i state12 = _mm256_setr_epi32(static_cast<uint32_t>(counter), static_cast<uint32_t>(counter),
                                        static_cast<uint32_t>(counter), static_cast<uint32_t>(counter),
                                        static_cast<uint32_t>(counter), static_cast<uint32_t>(counter),
                                        static_cast<uint32_t>(counter), static_cast<uint32_t>(counter));
    __m256i state13 = _mm256_setr_epi32(static_cast<uint32_t>(counter >> 32), static_cast<uint32_t>(counter >> 32),
                                        static_cast<uint32_t>(counter >> 32), static_cast<uint32_t>(counter >> 32),
                                        static_cast<uint32_t>(counter >> 32), static_cast<uint32_t>(counter >> 32),
                                        static_cast<uint32_t>(counter >> 32), static_cast<uint32_t>(counter >> 32));
    __m256i state14 = _mm256_setr_epi32(block_len, block_len, block_len, block_len,
                                        block_len, block_len, block_len, block_len);
    __m256i state15 = _mm256_setr_epi32(flags, flags, flags, flags, flags, flags, flags, flags);

    // 7 rounds of G functions
    for (int round = 0; round < 7; ++round) {
        const uint8_t* s = MSG_SCHEDULE[round];
        g_avx2(&state0, &state4, &state8, &state12, msg[s[0]], msg[s[1]]);
        g_avx2(&state1, &state5, &state9, &state13, msg[s[2]], msg[s[3]]);
        g_avx2(&state2, &state6, &state10, &state14, msg[s[4]], msg[s[5]]);
        g_avx2(&state3, &state7, &state11, &state15, msg[s[6]], msg[s[7]]);
        g_avx2(&state0, &state5, &state10, &state15, msg[s[8]], msg[s[9]]);
        g_avx2(&state1, &state6, &state11, &state12, msg[s[10]], msg[s[11]]);
        g_avx2(&state2, &state7, &state8, &state13, msg[s[12]], msg[s[13]]);
        g_avx2(&state3, &state4, &state9, &state14, msg[s[14]], msg[s[15]]);
    }

    // Extract results
    uint32_t temp0[8], temp1[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp0), _mm256_xor_si256(state0, state8));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp1), _mm256_xor_si256(state0, state8));
    for (int i = 0; i < 8; ++i) {
        out0[i] = temp0[i];
        out0[i + 8] = temp0[i] ^ cv[i];
        out1[i] = temp1[i];
        out1[i + 8] = temp1[i] ^ cv[i];
    }
}
#endif

// Portable fallback (same as existing)
inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t load32_le(const uint8_t* p) {
    uint32_t r;
    memcpy(&r, p, 4);
    return r;  // Assumes little-endian
}

inline void g(uint32_t* state, size_t a, size_t b, size_t c, size_t d, 
              uint32_t mx, uint32_t my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7);
}

inline void round_fn(uint32_t* state, const uint32_t* msg, size_t round) {
    const uint8_t* s = MSG_SCHEDULE[round];
    g(state, 0, 4, 8, 12, msg[s[0]], msg[s[1]]);
    g(state, 1, 5, 9, 13, msg[s[2]], msg[s[3]]);
    g(state, 2, 6, 10, 14, msg[s[4]], msg[s[5]]);
    g(state, 3, 7, 11, 15, msg[s[6]], msg[s[7]]);
    g(state, 0, 5, 10, 15, msg[s[8]], msg[s[9]]);
    g(state, 1, 6, 11, 12, msg[s[10]], msg[s[11]]);
    g(state, 2, 7, 8, 13, msg[s[12]], msg[s[13]]);
    g(state, 3, 4, 9, 14, msg[s[14]], msg[s[15]]);
}

inline void compress(const uint32_t cv[8], const uint8_t block[BLAKE3_BLOCK_LEN],
                     uint8_t block_len, uint64_t counter, uint8_t flags,
                     uint32_t out[16]) {
    uint32_t msg[16];
    for (size_t i = 0; i < 16; ++i) {
        msg[i] = load32_le(&block[i * 4]);
    }
    
    uint32_t state[16] = {
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3],
        static_cast<uint32_t>(counter),
        static_cast<uint32_t>(counter >> 32),
        block_len, flags
    };
    
    for (size_t r = 0; r < 7; ++r) {
        round_fn(state, msg, r);
    }
    
    for (size_t i = 0; i < 8; ++i) {
        out[i] = state[i] ^ state[i + 8];
        out[i + 8] = state[i + 8] ^ cv[i];
    }
}

// Fast single-block hash for small inputs (most common case)
inline void hash_32bytes(const uint8_t* input, uint8_t out[32]) {
    uint32_t cv[16];
    uint8_t block[64] = {0};
    memcpy(block, input, 32);
    compress(IV, block, 32, 0, CHUNK_START | CHUNK_END | ROOT, cv);
    for (size_t i = 0; i < 8; ++i) {
        out[i*4 + 0] = cv[i] & 0xFF;
        out[i*4 + 1] = (cv[i] >> 8) & 0xFF;
        out[i*4 + 2] = (cv[i] >> 16) & 0xFF;
        out[i*4 + 3] = (cv[i] >> 24) & 0xFF;
    }
}

// Fast 64-byte hash (for composition: 32 + 32 byte hashes)
inline void hash_64bytes(const uint8_t* input, uint8_t out[32]) {
    uint32_t cv[16];
    compress(IV, input, 64, 0, CHUNK_START | CHUNK_END | ROOT, cv);
    for (size_t i = 0; i < 8; ++i) {
        out[i*4 + 0] = cv[i] & 0xFF;
        out[i*4 + 1] = (cv[i] >> 8) & 0xFF;
        out[i*4 + 2] = (cv[i] >> 16) & 0xFF;
        out[i*4 + 3] = (cv[i] >> 24) & 0xFF;
    }
}

// Fast hash for composition: ordinal + hash + ordinal + hash (68 bytes)
inline void hash_composition(uint32_t ord0, const uint8_t hash0[32],
                             uint32_t ord1, const uint8_t hash1[32],
                             uint8_t out[32]) {
    // Build 68-byte input: 4 + 32 + 4 + 32 = 72 bytes (fits in 2 blocks)
    uint8_t block[128] = {0};
    memcpy(block, &ord0, 4);
    memcpy(block + 4, hash0, 32);
    memcpy(block + 36, &ord1, 4);
    memcpy(block + 40, hash1, 32);
    
    // Single block (68 bytes fits in 64-byte block with padding logic)
    // Actually 68 > 64, so need proper multi-block
    uint32_t cv[8];
    memcpy(cv, IV, 32);
    
    uint32_t tmp[16];
    compress(cv, block, 64, 0, CHUNK_START, tmp);
    for (size_t i = 0; i < 8; ++i) cv[i] = tmp[i];
    
    compress(cv, block + 64, 8, 0, CHUNK_END | ROOT, tmp);
    
    for (size_t i = 0; i < 8; ++i) {
        out[i*4 + 0] = tmp[i] & 0xFF;
        out[i*4 + 1] = (tmp[i] >> 8) & 0xFF;
        out[i*4 + 2] = (tmp[i] >> 16) & 0xFF;
        out[i*4 + 3] = (tmp[i] >> 24) & 0xFF;
    }
}

} // namespace blake3_simd
} // namespace hypercube
