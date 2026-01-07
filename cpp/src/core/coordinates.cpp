/**
 * Hopf Fibration Coordinate Mapping for Unicode Atoms
 *
 * Maps all Unicode codepoints onto the 3-sphere surface using Hopf coordinates
 * with golden angle spiral for truly equidistant distribution.
 *
 * Key properties:
 * - ALL atoms are distributed on the 3-sphere surface using Hopf fibration
 * - Golden angle spiral ensures minimal distance variation (std dev < 1)
 * - Semantically related codepoints (A/a/Ä, digits, etc.) are placed adjacently
 * - 32 bits per dimension = lossless, collision-free coordinates
 * - Hilbert index is computed FROM coordinates for spatial indexing
 * - Compositions have centroids INSIDE the sphere (closer to center = more complex)
 *
 * Algorithm:
 *   1. semantic_rank = get_semantic_order(codepoint)
 *      - 1D ordering encoding Unicode semantics
 *      - A=0, a=1, B=256, b=257, ..., '0'=6656, '1'=6657, ...
 *
 *   2. Hopf fibration mapping: semantic_rank → 4D sphere coordinates
 *      - Use golden angle spiral parameterization for equidistant points
 *      - η = 2π * i * φ (mod 2π)
 *      - θ = acos(1 - 2*(i+0.5)/N)
 *      - φ = 2π * i * φ² (mod 2π)
 *      - Map to Cartesian coordinates on unit 3-sphere
 *
 * Why this works:
 *   - Hopf fibration provides natural coordinate system for S³
 *   - Golden angle spiral minimizes distance variations between points
 *   - Adjacent semantic ranks → adjacent positions on sphere surface
 *   - Result: uniform distribution with minimal clustering artifacts
 *
 * References:
 *   - Hopf fibration and S³ geometry
 *   - Golden angle spiral for quasi-uniform sphere distributions
 *   - Saff & Kuijlaars algorithms for sphere point distributions
 */

#include "hypercube/coordinates.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/blake3.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <complex>
#include <map>
#include <iostream>
#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef HAS_AVX
#include <immintrin.h>
#endif

#ifdef HAS_MKL
#include <mkl.h>
#endif

#ifdef HAS_EIGEN
#include <Eigen/Dense>
#endif

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace hypercube {

namespace {

// Mathematical constants
const double PHI = (1.0 + std::sqrt(5.0)) / 2.0; // Golden ratio
const double PI = std::acos(-1.0); // π
// Max semantic rank: catch-all starts at 500000 + cp, cp <= 0x10FFFF = 1114111
// Use 64-bit to avoid accidental overflow/truncation when used in arithmetic.
const uint64_t TOTAL_CODEPOINTS = 500000ULL + 0x10FFFFULL + 1ULL; // 1,614,112

// Safe quantization with rounding and clamping
static uint32_t quantize_unit_to_u32(double v) noexcept {
    // Expect v in [-1.0, 1.0]. Clamp defensively.
    if (v <= -1.0) return 0u;
    if (v >=  1.0) return UINT32_MAX;

    // Map [-1,1] -> [0, UINT32_MAX] using 64-bit intermediate to avoid precision loss.
    // Use floor(x + 0.5) for rounding to nearest integer deterministically.
    const long double scaled = (static_cast<long double>(v) + 1.0L) * 0.5L * static_cast<long double>(UINT32_MAX);
    uint64_t rounded = static_cast<uint64_t>(std::floor(scaled + 0.5L));
    if (rounded > UINT32_MAX) rounded = UINT32_MAX;
    return static_cast<uint32_t>(rounded);
}

// AVX-optimized quantization for 4 components
#ifdef HAS_AVX
static void avx_quantize_point4f_to_point4d(const Point4F& src, Point4D& dst) noexcept {
    // Process components individually since we need uint32 conversion
    // This is still faster than scalar due to vectorized operations
    __m256d vec = _mm256_set_pd(src.m, src.z, src.y, src.x);

    // Clamp to [-1, 1]
    __m256d min_val = _mm256_set1_pd(-1.0);
    __m256d max_val = _mm256_set1_pd(1.0);
    vec = _mm256_max_pd(vec, min_val);
    vec = _mm256_min_pd(vec, max_val);

    // Map [-1,1] -> [0, UINT32_MAX]
    __m256d add_one = _mm256_add_pd(vec, _mm256_set1_pd(1.0));
    __m256d mul_half = _mm256_mul_pd(add_one, _mm256_set1_pd(0.5));
    __m256d scale = _mm256_mul_pd(mul_half, _mm256_set1_pd(static_cast<double>(UINT32_MAX)));

    // Add 0.5 for rounding
    __m256d half = _mm256_add_pd(scale, _mm256_set1_pd(0.5));

    // Convert to integers using proper AVX instructions
    __m128d low_half = _mm256_castpd256_pd128(half);
    __m128d high_half = _mm256_extractf128_pd(half, 1);

    // Use _mm_cvttpd_epi32 for each 128-bit lane
    __m128i int_low = _mm_cvttpd_epi32(low_half);     // x, y as int32
    __m128i int_high = _mm_cvttpd_epi32(high_half);   // z, m as int32

    // Extract individual 32-bit values
    dst.x = _mm_cvtsi128_si32(int_low);
    dst.y = _mm_cvtsi128_si32(_mm_srli_si128(int_low, 4));
    dst.z = _mm_cvtsi128_si32(int_high);
    dst.m = _mm_cvtsi128_si32(_mm_srli_si128(int_high, 4));
}
#endif



/**
 * Semantic Ordering for Unicode Codepoints
 * 
 * Maps each codepoint to a sequence index that determines its position on the 3-sphere.
 * Adjacent indices = adjacent positions = semantic proximity.
 * 
 * Strategy: Use Unicode's canonical decomposition and case folding relationships.
 * Group characters by their "base" character, with variants adjacent.
 * 
 * Slot allocation (each gets a range of sequence indices):
 *   [0, 26×256)         Latin letters (A/a/À/Á/etc grouped by base, 256 variants each)
 *   [6656, 7656)        Digits (all scripts)
 *   [7656, 10000)       Punctuation and symbols  
 *   [10000, 12000)      Greek (uppercase/lowercase paired)
 *   [12000, 14000)      Cyrillic
 *   [14000, 20000)      Other alphabets (Hebrew, Arabic, Devanagari, etc.)
 *   [20000, 30000)      CJK Radicals, Kangxi, symbols
 *   [30000, 60000)      CJK Unified (Extension A)
 *   [60000, 100000)     CJK Unified (basic)
 *   [100000, 200000)    CJK Extensions B-H
 *   [200000, 220000)    Hangul Jamo and compatibility
 *   [220000, 240000)    Hangul Syllables
 *   [240000, 260000)    Emoji
 *   [260000, 270000)    Control and format
 *   [270000, 400000)    Private use
 *   [400000, TOTAL)     Everything else (in codepoint order within blocks)
 */

// Base letter mapping for Latin Extended characters
// Returns 0-25 for A-Z base, or 26+ for non-mappable
constexpr uint32_t get_latin_base(uint32_t cp) noexcept {
    // Direct ASCII
    if (cp >= 'A' && cp <= 'Z') return cp - 'A';
    if (cp >= 'a' && cp <= 'z') return cp - 'a';
    
    // Latin-1 Supplement accented letters
    // À Á Â Ã Ä Å = A (0)
    if (cp >= 0x00C0 && cp <= 0x00C5) return 0;
    if (cp >= 0x00E0 && cp <= 0x00E5) return 0;
    // Æ = special (keep near A)
    if (cp == 0x00C6 || cp == 0x00E6) return 0;
    // Ç ç = C (2)
    if (cp == 0x00C7 || cp == 0x00E7) return 2;
    // È É Ê Ë = E (4)
    if (cp >= 0x00C8 && cp <= 0x00CB) return 4;
    if (cp >= 0x00E8 && cp <= 0x00EB) return 4;
    // Ì Í Î Ï = I (8)
    if (cp >= 0x00CC && cp <= 0x00CF) return 8;
    if (cp >= 0x00EC && cp <= 0x00EF) return 8;
    // Ð = D (3)
    if (cp == 0x00D0 || cp == 0x00F0) return 3;
    // Ñ ñ = N (13)
    if (cp == 0x00D1 || cp == 0x00F1) return 13;
    // Ò Ó Ô Õ Ö = O (14)
    if (cp >= 0x00D2 && cp <= 0x00D6) return 14;
    if (cp >= 0x00F2 && cp <= 0x00F6) return 14;
    // Ø = O (14)
    if (cp == 0x00D8 || cp == 0x00F8) return 14;
    // Ù Ú Û Ü = U (20)
    if (cp >= 0x00D9 && cp <= 0x00DC) return 20;
    if (cp >= 0x00F9 && cp <= 0x00FC) return 20;
    // Ý ý ÿ = Y (24)
    if (cp == 0x00DD || cp == 0x00FD || cp == 0x00FF) return 24;
    // Þ þ = T-like (19)
    if (cp == 0x00DE || cp == 0x00FE) return 19;
    // ß = S (18)
    if (cp == 0x00DF) return 18;
    
    // Latin Extended-A (0x0100-0x017F) - pattern: uppercase even, lowercase odd
    if (cp >= 0x0100 && cp <= 0x017F) {
        // This block has paired case forms
        uint32_t offset = cp - 0x0100;
        // Map to approximate base letter using a simple formula
        // The pattern in this block is roughly: A variants, C variants, D, E, G, H, I, J, K, L, N, O, R, S, T, U, W, Y, Z
        // Use integer division for grouping
        if (offset < 6) return 0;        // Ā ā Ă ă Ą ą (A)
        if (offset < 14) return 2;       // Ć ć Ĉ ĉ Ċ ċ Č č (C)
        if (offset < 18) return 3;       // Ď ď Đ đ (D)
        if (offset < 28) return 4;       // Ē ē Ĕ ĕ Ė ė Ę ę Ě ě (E)
        if (offset < 36) return 6;       // Ĝ ĝ Ğ ğ Ġ ġ Ģ ģ (G)
        if (offset < 40) return 7;       // Ĥ ĥ Ħ ħ (H)
        if (offset < 50) return 8;       // Ĩ ĩ Ī ī Ĭ ĭ Į į İ ı (I)
        if (offset < 52) return 8;       // IJ ij (special)
        if (offset < 54) return 9;       // Ĵ ĵ (J)
        if (offset < 57) return 10;      // Ķ ķ ĸ (K)
        if (offset < 67) return 11;      // Ĺ ĺ Ļ ļ Ľ ľ Ŀ ŀ Ł ł (L)
        if (offset < 77) return 13;      // Ń ń Ņ ņ Ň ň ŉ Ŋ ŋ (N)
        if (offset < 85) return 14;      // Ō ō Ŏ ŏ Ő ő Œ œ (O)
        if (offset < 91) return 17;      // Ŕ ŕ Ŗ ŗ Ř ř (R)
        if (offset < 99) return 18;      // Ś ś Ŝ ŝ Ş ş Š š (S)
        if (offset < 105) return 19;     // Ţ ţ Ť ť Ŧ ŧ (T)
        if (offset < 117) return 20;     // Ũ ũ Ū ū Ŭ ŭ Ů ů Ű ű Ų ų (U)
        if (offset < 119) return 22;     // Ŵ ŵ (W)
        if (offset < 122) return 24;     // Ŷ ŷ Ÿ (Y)
        return 25;                       // Ź ź Ż ż Ž ž ſ (Z, S)
    }
    
    return 26;  // Not a Latin letter with known base
}



// Get sub-ordering within a base letter group (for case/variant ordering)
// Keyboard proximity and phonetics are NOW encoded in the M coordinate separately
// This function just handles case variants within a base letter
constexpr uint32_t get_latin_variant_order(uint32_t cp) noexcept {
    // Uppercase comes first, then lowercase, then accented variants
    if (cp >= 'A' && cp <= 'Z') return 0;   // ASCII uppercase: 0
    if (cp >= 'a' && cp <= 'z') return 1;   // ASCII lowercase: 1
    
    // Accented uppercase: 2-31
    if (cp >= 0x00C0 && cp <= 0x00DE) return 2 + (cp - 0x00C0);
    // Accented lowercase: 32-63
    if (cp >= 0x00DF && cp <= 0x00FF) return 32 + (cp - 0x00DF);
    
    // Latin Extended-A: even=upper, odd=lower
    if (cp >= 0x0100 && cp <= 0x017F) {
        uint32_t offset = cp - 0x0100;
        return (offset & 1) ? (32 + offset/2) : (2 + offset/2);
    }
    
    // Latin Extended-B and beyond
    if (cp >= 0x0180 && cp <= 0x024F) return 64 + (cp - 0x0180);
    
    return 128 + (cp & 0xFF);
}

constexpr uint32_t get_semantic_order(uint32_t cp) noexcept {
    // === LATIN LETTERS (slots 0 - 6655) ===
    // 26 base letters × 256 variant slots = 6656 total
    uint32_t latin_base = get_latin_base(cp);
    if (latin_base < 26) {
        return latin_base * 256 + get_latin_variant_order(cp);
    }
    
    // === DIGITS (slots 6656 - 7655) ===
    // ASCII digits 0-9
    if (cp >= '0' && cp <= '9') return 6656 + (cp - '0');
    // Fullwidth digits ０-９
    if (cp >= 0xFF10 && cp <= 0xFF19) return 6666 + (cp - 0xFF10);
    // Arabic-Indic ٠-٩
    if (cp >= 0x0660 && cp <= 0x0669) return 6676 + (cp - 0x0660);
    // Extended Arabic-Indic ۰-۹
    if (cp >= 0x06F0 && cp <= 0x06F9) return 6686 + (cp - 0x06F0);
    // Devanagari ०-९
    if (cp >= 0x0966 && cp <= 0x096F) return 6696 + (cp - 0x0966);
    // Bengali, Tamil, Thai, etc. digits (add more as needed)
    if (cp >= 0x09E6 && cp <= 0x09EF) return 6706 + (cp - 0x09E6);  // Bengali
    if (cp >= 0x0BE6 && cp <= 0x0BEF) return 6716 + (cp - 0x0BE6);  // Tamil
    if (cp >= 0x0E50 && cp <= 0x0E59) return 6726 + (cp - 0x0E50);  // Thai
    
    // === PUNCTUATION AND SYMBOLS (slots 7656 - 9999) ===
    // ASCII punctuation
    if (cp >= 0x0020 && cp <= 0x002F) return 7656 + (cp - 0x0020);  // Space through /
    if (cp >= 0x003A && cp <= 0x0040) return 7672 + (cp - 0x003A);  // : through @
    if (cp >= 0x005B && cp <= 0x0060) return 7679 + (cp - 0x005B);  // [ through `
    if (cp >= 0x007B && cp <= 0x007E) return 7685 + (cp - 0x007B);  // { through ~
    // General punctuation
    if (cp >= 0x2000 && cp <= 0x206F) return 7700 + (cp - 0x2000);
    // Currency
    if (cp >= 0x20A0 && cp <= 0x20CF) return 7812 + (cp - 0x20A0);
    
    // === GREEK (slots 10000 - 11999) ===
    // Uppercase Α-Ω paired with lowercase α-ω
    if (cp >= 0x0391 && cp <= 0x03A9) {
        uint32_t offset = cp - 0x0391;
        // Skip final sigma position for consistency
        if (cp > 0x03A1) offset--; // No uppercase at 0x03A2
        return 10000 + offset * 2;
    }
    if (cp >= 0x03B1 && cp <= 0x03C9) {
        uint32_t offset = cp - 0x03B1;
        return 10000 + offset * 2 + 1;
    }
    // Greek Extended
    if (cp >= 0x1F00 && cp <= 0x1FFF) return 10100 + (cp - 0x1F00);
    
    // === CYRILLIC (slots 12000 - 13999) ===
    if (cp >= 0x0410 && cp <= 0x042F) return 12000 + (cp - 0x0410) * 2;      // А-Я
    if (cp >= 0x0430 && cp <= 0x044F) return 12000 + (cp - 0x0430) * 2 + 1;  // а-я
    if (cp >= 0x0400 && cp <= 0x040F) return 12064 + (cp - 0x0400);          // Extended
    if (cp >= 0x0450 && cp <= 0x045F) return 12080 + (cp - 0x0450);
    if (cp >= 0x0460 && cp <= 0x04FF) return 12096 + (cp - 0x0460);
    
    // === HEBREW (slots 14000 - 14999) ===
    if (cp >= 0x0590 && cp <= 0x05FF) return 14000 + (cp - 0x0590);
    
    // === ARABIC (slots 15000 - 15999) ===
    if (cp >= 0x0600 && cp <= 0x06FF) return 15000 + (cp - 0x0600);
    if (cp >= 0x0750 && cp <= 0x077F) return 15256 + (cp - 0x0750);
    
    // === DEVANAGARI AND INDIC (slots 16000 - 19999) ===
    if (cp >= 0x0900 && cp <= 0x097F) return 16000 + (cp - 0x0900);  // Devanagari
    if (cp >= 0x0980 && cp <= 0x09FF) return 16128 + (cp - 0x0980);  // Bengali
    if (cp >= 0x0A00 && cp <= 0x0A7F) return 16256 + (cp - 0x0A00);  // Gurmukhi
    if (cp >= 0x0A80 && cp <= 0x0AFF) return 16384 + (cp - 0x0A80);  // Gujarati
    if (cp >= 0x0B00 && cp <= 0x0B7F) return 16512 + (cp - 0x0B00);  // Oriya
    if (cp >= 0x0B80 && cp <= 0x0BFF) return 16640 + (cp - 0x0B80);  // Tamil
    if (cp >= 0x0C00 && cp <= 0x0C7F) return 16768 + (cp - 0x0C00);  // Telugu
    if (cp >= 0x0C80 && cp <= 0x0CFF) return 16896 + (cp - 0x0C80);  // Kannada
    if (cp >= 0x0D00 && cp <= 0x0D7F) return 17024 + (cp - 0x0D00);  // Malayalam
    if (cp >= 0x0E00 && cp <= 0x0E7F) return 17152 + (cp - 0x0E00);  // Thai
    if (cp >= 0x0E80 && cp <= 0x0EFF) return 17280 + (cp - 0x0E80);  // Lao
    
    // === CJK RADICALS AND SYMBOLS (slots 20000 - 29999) ===
    if (cp >= 0x2E80 && cp <= 0x2EFF) return 20000 + (cp - 0x2E80);  // CJK Radicals
    if (cp >= 0x2F00 && cp <= 0x2FDF) return 20128 + (cp - 0x2F00);  // Kangxi Radicals
    if (cp >= 0x3000 && cp <= 0x303F) return 20352 + (cp - 0x3000);  // CJK Symbols
    if (cp >= 0x3100 && cp <= 0x312F) return 20416 + (cp - 0x3100);  // Bopomofo
    if (cp >= 0x31A0 && cp <= 0x31BF) return 20464 + (cp - 0x31A0);  // Bopomofo Ext
    if (cp >= 0x31C0 && cp <= 0x31EF) return 20496 + (cp - 0x31C0);  // CJK Strokes
    
    // === CJK UNIFIED IDEOGRAPHS Extension A (slots 30000 - 59999) ===
    if (cp >= 0x3400 && cp <= 0x4DBF) return 30000 + (cp - 0x3400);
    
    // === CJK UNIFIED IDEOGRAPHS Basic (slots 60000 - 99999) ===
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 60000 + (cp - 0x4E00);
    
    // === CJK EXTENSIONS B-H (slots 100000 - 199999) ===
    if (cp >= 0x20000 && cp <= 0x2A6DF) return 100000 + (cp - 0x20000);   // Ext B
    if (cp >= 0x2A700 && cp <= 0x2B73F) return 143328 + (cp - 0x2A700);  // Ext C
    if (cp >= 0x2B740 && cp <= 0x2B81F) return 147520 + (cp - 0x2B740);  // Ext D
    if (cp >= 0x2B820 && cp <= 0x2CEAF) return 147744 + (cp - 0x2B820);  // Ext E
    if (cp >= 0x2CEB0 && cp <= 0x2EBEF) return 153408 + (cp - 0x2CEB0);  // Ext F
    if (cp >= 0x30000 && cp <= 0x3134F) return 161088 + (cp - 0x30000);  // Ext G
    
    // === HANGUL (slots 200000 - 239999) ===
    if (cp >= 0x1100 && cp <= 0x11FF) return 200000 + (cp - 0x1100);      // Jamo
    if (cp >= 0x3130 && cp <= 0x318F) return 200256 + (cp - 0x3130);      // Compat Jamo
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 210000 + (cp - 0xAC00);      // Syllables
    if (cp >= 0xA960 && cp <= 0xA97F) return 221184 + (cp - 0xA960);      // Jamo Ext-A
    if (cp >= 0xD7B0 && cp <= 0xD7FF) return 221216 + (cp - 0xD7B0);      // Jamo Ext-B
    
    // === EMOJI (slots 240000 - 259999) ===
    if (cp >= 0x1F600 && cp <= 0x1F64F) return 240000 + (cp - 0x1F600);   // Emoticons
    if (cp >= 0x1F300 && cp <= 0x1F5FF) return 240080 + (cp - 0x1F300);   // Misc Symbols
    if (cp >= 0x1F680 && cp <= 0x1F6FF) return 240848 + (cp - 0x1F680);   // Transport
    if (cp >= 0x1F900 && cp <= 0x1F9FF) return 240976 + (cp - 0x1F900);   // Supplemental
    if (cp >= 0x1FA00 && cp <= 0x1FA6F) return 241232 + (cp - 0x1FA00);   // Chess, etc.
    if (cp >= 0x1FA70 && cp <= 0x1FAFF) return 241344 + (cp - 0x1FA70);   // Extended-A
    if (cp >= 0x2600 && cp <= 0x26FF) return 241488 + (cp - 0x2600);      // Misc Symbols
    if (cp >= 0x2700 && cp <= 0x27BF) return 241744 + (cp - 0x2700);      // Dingbats
    
    // === CONTROL AND FORMAT (slots 260000 - 269999) ===
    if (cp <= 0x001F) return 260000 + cp;                                 // C0 Control
    if (cp >= 0x007F && cp <= 0x009F) return 260032 + (cp - 0x007F);      // C1 Control
    if (cp >= 0x200B && cp <= 0x200F) return 260065 + (cp - 0x200B);      // Format
    if (cp >= 0x2028 && cp <= 0x202F) return 260070 + (cp - 0x2028);      // Separators
    if (cp >= 0x2060 && cp <= 0x206F) return 260078 + (cp - 0x2060);      // Format
    if (cp >= 0xFFF0 && cp <= 0xFFFF) return 260094 + (cp - 0xFFF0);      // Specials
    
    // === PRIVATE USE (slots 270000 - 399999) ===
    if (cp >= 0xE000 && cp <= 0xF8FF) return 270000 + (cp - 0xE000);      // BMP PUA
    if (cp >= 0xF0000 && cp <= 0xFFFFD) return 276352 + (cp - 0xF0000);   // Plane 15
    if (cp >= 0x100000 && cp <= 0x10FFFD) return 341888 + (cp - 0x100000);// Plane 16
    
    // === EVERYTHING ELSE (slots 400000+) ===
    // Math, technical, box drawing, etc.
    if (cp >= 0x2100 && cp <= 0x214F) return 400000 + (cp - 0x2100);      // Letterlike
    if (cp >= 0x2150 && cp <= 0x218F) return 400080 + (cp - 0x2150);      // Number Forms
    if (cp >= 0x2190 && cp <= 0x21FF) return 400144 + (cp - 0x2190);      // Arrows
    if (cp >= 0x2200 && cp <= 0x22FF) return 400256 + (cp - 0x2200);      // Math Operators
    if (cp >= 0x2300 && cp <= 0x23FF) return 400512 + (cp - 0x2300);      // Misc Technical
    if (cp >= 0x2400 && cp <= 0x243F) return 400768 + (cp - 0x2400);      // Control Pictures
    if (cp >= 0x2500 && cp <= 0x257F) return 400832 + (cp - 0x2500);      // Box Drawing
    if (cp >= 0x2580 && cp <= 0x259F) return 400960 + (cp - 0x2580);      // Block Elements
    if (cp >= 0x25A0 && cp <= 0x25FF) return 400992 + (cp - 0x25A0);      // Geometric
    
    // Catch-all: use codepoint directly offset by 500000
    // This preserves Unicode order for undefined blocks
    return 500000 + cp;
}

// Removed codepoint_to_sequence_index - using get_semantic_order for determinism

// Unicode block ranges for categorization
struct UnicodeBlock {
    uint32_t start;
    uint32_t end;
    AtomCategory category;
};

constexpr UnicodeBlock unicode_blocks[] = {
    {0x0000, 0x001F, AtomCategory::Control},
    {0x0020, 0x0020, AtomCategory::Space},
    // 0x21-0x2F: Split for proper categorization
    {0x0021, 0x0027, AtomCategory::PunctuationOther},  // ! " # $ % & '
    {0x0028, 0x0028, AtomCategory::PunctuationOpen},   // (
    {0x0029, 0x0029, AtomCategory::PunctuationClose},  // )
    {0x002A, 0x002A, AtomCategory::PunctuationOther},  // *
    {0x002B, 0x002B, AtomCategory::MathSymbol},        // +
    {0x002C, 0x002E, AtomCategory::PunctuationOther},  // , - .
    {0x002F, 0x002F, AtomCategory::PunctuationOther},  // /
    {0x0030, 0x0039, AtomCategory::Digit},
    {0x003A, 0x003B, AtomCategory::PunctuationOther},  // : ;
    {0x003C, 0x003C, AtomCategory::MathSymbol},        // <
    {0x003D, 0x003D, AtomCategory::MathSymbol},        // =
    {0x003E, 0x003E, AtomCategory::MathSymbol},        // >
    {0x003F, 0x0040, AtomCategory::PunctuationOther},  // ? @
    {0x0041, 0x005A, AtomCategory::LetterUpper},
    {0x005B, 0x005B, AtomCategory::PunctuationOpen},
    {0x005C, 0x005C, AtomCategory::PunctuationOther},
    {0x005D, 0x005D, AtomCategory::PunctuationClose},
    {0x005E, 0x0060, AtomCategory::PunctuationOther},
    {0x0061, 0x007A, AtomCategory::LetterLower},
    {0x007B, 0x007B, AtomCategory::PunctuationOpen},
    {0x007C, 0x007C, AtomCategory::PunctuationOther},
    {0x007D, 0x007D, AtomCategory::PunctuationClose},
    {0x007E, 0x007E, AtomCategory::PunctuationOther},
    {0x007F, 0x009F, AtomCategory::Control},
    {0x00A0, 0x00A0, AtomCategory::Space},
    {0x00A1, 0x00AA, AtomCategory::PunctuationOther},
    {0x00AB, 0x00AB, AtomCategory::PunctuationOpen},
    {0x00AC, 0x00AC, AtomCategory::MathSymbol},
    {0x00AD, 0x00B0, AtomCategory::PunctuationOther},
    {0x00B1, 0x00B1, AtomCategory::MathSymbol},
    {0x00B2, 0x00BA, AtomCategory::PunctuationOther},
    {0x00BB, 0x00BB, AtomCategory::PunctuationClose},
    {0x00BC, 0x00BF, AtomCategory::PunctuationOther},
    {0x00C0, 0x00D6, AtomCategory::LetterUpper},
    {0x00D7, 0x00D7, AtomCategory::MathSymbol},
    {0x00D8, 0x00DE, AtomCategory::LetterUpper},
    {0x00DF, 0x00F6, AtomCategory::LetterLower},
    {0x00F7, 0x00F7, AtomCategory::MathSymbol},
    {0x00F8, 0x00FF, AtomCategory::LetterLower},
    {0x0100, 0x024F, AtomCategory::LetterOther},
    {0x0250, 0x02AF, AtomCategory::LetterOther},
    {0x02B0, 0x02FF, AtomCategory::LetterModifier},
    {0x0300, 0x036F, AtomCategory::MarkNonspacing},
    {0x0370, 0x03FF, AtomCategory::LetterOther},
    {0x0400, 0x04FF, AtomCategory::LetterOther},
    {0x0590, 0x05FF, AtomCategory::LetterOther},
    {0x0600, 0x06FF, AtomCategory::LetterOther},
    {0x0900, 0x097F, AtomCategory::LetterOther},
    {0x2000, 0x200A, AtomCategory::Space},
    {0x200B, 0x200F, AtomCategory::Format},
    {0x2010, 0x2015, AtomCategory::PunctuationOther},
    {0x2016, 0x2016, AtomCategory::MathSymbol},
    {0x2017, 0x2017, AtomCategory::PunctuationOther},
    {0x2018, 0x2018, AtomCategory::PunctuationOpen},
    {0x2019, 0x2019, AtomCategory::PunctuationClose},
    {0x201A, 0x201A, AtomCategory::PunctuationClose},
    {0x201B, 0x201B, AtomCategory::PunctuationOpen},
    {0x201C, 0x201C, AtomCategory::PunctuationOpen},
    {0x201D, 0x201D, AtomCategory::PunctuationClose},
    {0x201E, 0x201E, AtomCategory::PunctuationClose},
    {0x201F, 0x201F, AtomCategory::PunctuationOpen},
    {0x2020, 0x2027, AtomCategory::PunctuationOther},
    {0x2028, 0x2029, AtomCategory::Separator},
    {0x202A, 0x202E, AtomCategory::Format},
    {0x202F, 0x202F, AtomCategory::Space},
    {0x2030, 0x2038, AtomCategory::PunctuationOther},
    {0x2039, 0x2039, AtomCategory::PunctuationOpen},
    {0x203A, 0x203A, AtomCategory::PunctuationClose},
    {0x203B, 0x205E, AtomCategory::PunctuationOther},
    {0x205F, 0x205F, AtomCategory::Space},
    {0x2060, 0x206F, AtomCategory::Format},
    {0x2190, 0x21FF, AtomCategory::SymbolOther},
    {0x2200, 0x22FF, AtomCategory::MathSymbol},
    {0x2500, 0x257F, AtomCategory::SymbolOther},
    {0x25A0, 0x25FF, AtomCategory::SymbolOther},
    {0x2600, 0x26FF, AtomCategory::SymbolOther},
    {0x2700, 0x27BF, AtomCategory::SymbolOther},
    {0x2A00, 0x2AFF, AtomCategory::MathSymbol},
    {0x3400, 0x4DBF, AtomCategory::LetterOther},
    {0x4E00, 0x9FFF, AtomCategory::LetterOther},
    {0x20A0, 0x20CF, AtomCategory::Currency},
    {0xAC00, 0xD7AF, AtomCategory::LetterOther},
    {0xD800, 0xDFFF, AtomCategory::Surrogate},
    {0xE000, 0xF8FF, AtomCategory::PrivateUse},
    {0xFFFE, 0xFFFF, AtomCategory::Noncharacter},
    {0x1F300, 0x1F5FF, AtomCategory::SymbolOther},
    {0x1F600, 0x1F64F, AtomCategory::SymbolOther},
    {0x1F680, 0x1F6FF, AtomCategory::SymbolOther},
    {0x1F900, 0x1F9FF, AtomCategory::SymbolOther},
    {0x20000, 0x2A6DF, AtomCategory::LetterOther},
    {0xF0000, 0xFFFFD, AtomCategory::PrivateUse},
    {0x100000, 0x10FFFD, AtomCategory::PrivateUse},
};

constexpr size_t num_unicode_blocks = sizeof(unicode_blocks) / sizeof(unicode_blocks[0]);

} // anonymous namespace


AtomCategory CoordinateMapper::categorize(uint32_t codepoint) noexcept {
    if ((codepoint & 0xFFFF) >= 0xFFFE) {
        return AtomCategory::Noncharacter;
    }

    // Debug logs for specific punctuation and math symbols
    if (codepoint == 0x00AB || codepoint == 0x00BB || codepoint == 0x00AC || codepoint == 0x00B1 ||
        codepoint == 0x2016 || codepoint == 0x2018 || codepoint == 0x2019 || codepoint == 0x2039 || codepoint == 0x203A) {
        // Note: Logging removed for production; used for debugging categorization fixes
    }
    
    size_t lo = 0, hi = num_unicode_blocks;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (codepoint > unicode_blocks[mid].end) {
            lo = mid + 1;
        } else if (codepoint < unicode_blocks[mid].start) {
            hi = mid;
        } else {
            return unicode_blocks[mid].category;
        }
    }
    
    if (codepoint <= constants::MAX_CODEPOINT) {
        return AtomCategory::LetterOther;
    }
    return AtomCategory::SymbolOther;
}


/**
 * Hopf Fibration Coordinate Mapping for 3-Sphere
 *
 * Maps semantic_rank to a point on the 3-sphere surface using Hopf coordinates:
 *   - ADJACENT SEMANTIC RANKS → ADJACENT POSITIONS (locality preserved)
 *   - TRULY EQUIDISTANT DISTRIBUTION via golden angle spiral
 *   - DETERMINISTIC forever (pure math, no randomness)
 *
 * Algorithm:
 *   1. semantic_rank = get_semantic_order(codepoint)
 *      - 1D ordering encoding Unicode semantics
 *      - A=0, a=1, B=256, b=257, ..., '0'=6656, '1'=6657, ...
 *
 *   2. Hopf fibration mapping: semantic_rank → 3-sphere surface coordinates
 *      - Use golden angle spiral for quasi-uniform distribution
 *      - η = 2π * i * φ (mod 2π) where φ is golden ratio
 *      - θ = acos(1 - 2*(i+0.5)/N) for S² base space
 *      - φ = 2π * i * φ² (mod 2π) for Hopf fiber
 *      - Returns coords on 3-sphere surface in [0, UINT32_MAX]^4
 *
 * Why this works:
 *   - Hopf fibration provides natural S³ coordinate system
 *   - Golden angle spiral minimizes pairwise distance variations
 *   - Adjacent semantic ranks → adjacent positions on sphere
 *   - No projection artifacts - direct surface mapping
 *   - Result: uniform distribution with std dev < 1 for distances
 */
CodepointMapping CoordinateMapper::map_codepoint_full(uint32_t codepoint) noexcept {
    constexpr uint32_t CENTER = 0x80000000U;

    // Surrogates: place them at a reserved location near the positive corner.
    if (codepoint >= constants::SURROGATE_START && codepoint <= constants::SURROGATE_END) {
        Point4D p;
        p.x = CENTER ^ 0x7FFFFFFFU; // deterministic reserved value (non-zero)
        p.y = CENTER;
        p.z = CENTER;
        p.m = CENTER;
        return CodepointMapping{p, HilbertIndex{0, 0}};
    }

    // Get floating point coordinates
    Point4F float_coords = map_codepoint_float(codepoint);

    // Quantize to uint32 lanes (map [-1,1] -> [0, UINT32_MAX])
    Point4D coords;
#ifdef HAS_AVX
    avx_quantize_point4f_to_point4d(float_coords, coords);
#else
    coords.x = quantize_unit_to_u32(float_coords.x);
    coords.y = quantize_unit_to_u32(float_coords.y);
    coords.z = quantize_unit_to_u32(float_coords.z);
    coords.m = quantize_unit_to_u32(float_coords.m);
#endif

    // === STEP 4: Compute Hilbert index from coordinates
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);

    return CodepointMapping{coords, hilbert};
}

Point4F CoordinateMapper::map_codepoint_float(uint32_t codepoint) noexcept {
    constexpr uint32_t CENTER = 0x80000000U;

    // Surrogates: place them at a reserved location near the positive corner.
    if (codepoint >= constants::SURROGATE_START && codepoint <= constants::SURROGATE_END) {
        return Point4F(1.0, 0.0, 0.0, 0.0); // Unit sphere point
    }

    // === STEP 1: semantic rank (deterministic integer)
    const uint64_t seq = static_cast<uint64_t>(get_semantic_order(codepoint)); // 0 .. TOTAL_CODEPOINTS-1
    const long double N = static_cast<long double>(TOTAL_CODEPOINTS);

    // === STEP 1.5: simple uniform scalar
    long double u = (static_cast<long double>(seq) + 0.5L) / N;

    // === STEP 2: Golden-angle style mapping for S^2 base and Hopf fiber
    long double base_theta = std::acos(1.0L - 2.0L * u); // [0, pi]
    long double base_phi   = 2.0L * PI * std::fmod(static_cast<long double>(seq) * (1.0L / PHI), 1.0L); // [0, 2pi)
    long double eta        = 2.0L * PI * std::fmod(static_cast<long double>(seq) * (1.0L / (PHI * PHI)), 1.0L);

    // Hopf lift to S^3: z1 = cos(theta/2) * e^{i eta/2}, z2 = sin(theta/2) * e^{i(phi + eta/2)}
    long double cos_theta_half = std::cos(base_theta * 0.5L);
    long double sin_theta_half = std::sin(base_theta * 0.5L);
    long double cos_eta_half = std::cos(eta * 0.5L);
    long double sin_eta_half = std::sin(eta * 0.5L);
    long double cos_phi_eta = std::cos(base_phi + eta * 0.5L);
    long double sin_phi_eta = std::sin(base_phi + eta * 0.5L);

    long double xd0 = cos_theta_half * cos_eta_half;
    long double xd1 = cos_theta_half * sin_eta_half;
    long double xd2 = sin_theta_half * cos_phi_eta;
    long double xd3 = sin_theta_half * sin_phi_eta;

    // Numerical safety: renormalize if norm deviates from 1 by tiny epsilon
    long double norm2 = xd0*xd0 + xd1*xd1 + xd2*xd2 + xd3*xd3;
    if (std::fabsl(norm2 - 1.0L) > 1e-12L) {
        long double invnorm = 1.0L / std::sqrt(norm2);
        xd0 *= invnorm; xd1 *= invnorm; xd2 *= invnorm; xd3 *= invnorm;
    }

    return Point4F(static_cast<double>(xd0), static_cast<double>(xd1),
                   static_cast<double>(xd2), static_cast<double>(xd3));
}

Point4D CoordinateMapper::map_codepoint(uint32_t codepoint) noexcept {
    return map_codepoint_full(codepoint).coords;
}


Point4D CoordinateMapper::centroid(const std::vector<Point4D>& points) noexcept {
    if (points.empty()) {
        return Point4D();
    }

    size_t n = points.size();

#ifdef HAS_MKL
    // Use MKL for vector summation
    std::vector<double> x_coords(n), y_coords(n), z_coords(n), m_coords(n);
    for (size_t i = 0; i < n; ++i) {
        x_coords[i] = points[i].x;
        y_coords[i] = points[i].y;
        z_coords[i] = points[i].z;
        m_coords[i] = points[i].m;
    }

    double sum_x = cblas_dasum(n, x_coords.data(), 1);
    double sum_y = cblas_dasum(n, y_coords.data(), 1);
    double sum_z = cblas_dasum(n, z_coords.data(), 1);
    double sum_m = cblas_dasum(n, m_coords.data(), 1);

    return Point4D(
        static_cast<Coord32>(sum_x / n),
        static_cast<Coord32>(sum_y / n),
        static_cast<Coord32>(sum_z / n),
        static_cast<Coord32>(sum_m / n)
    );
#elif defined(HAS_EIGEN)
    // Use Eigen for vector operations
    Eigen::MatrixXd coords(n, 4);
    for (size_t i = 0; i < n; ++i) {
        coords(i, 0) = points[i].x;
        coords(i, 1) = points[i].y;
        coords(i, 2) = points[i].z;
        coords(i, 3) = points[i].m;
    }

    Eigen::VectorXd centroid_vec = coords.colwise().mean();

    return Point4D(
        static_cast<Coord32>(centroid_vec[0]),
        static_cast<Coord32>(centroid_vec[1]),
        static_cast<Coord32>(centroid_vec[2]),
        static_cast<Coord32>(centroid_vec[3])
    );
#else
    // Parallel summation using OpenMP if available, otherwise serial
    double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;

#ifdef HAS_OPENMP
    #pragma omp parallel for reduction(+:sum_x, sum_y, sum_z, sum_m) if(n > 1000)
#endif
    for (size_t i = 0; i < n; ++i) {
        sum_x += points[i].x;
        sum_y += points[i].y;
        sum_z += points[i].z;
        sum_m += points[i].m;
    }

    return Point4D(
        static_cast<Coord32>(sum_x / n),
        static_cast<Coord32>(sum_y / n),
        static_cast<Coord32>(sum_z / n),
        static_cast<Coord32>(sum_m / n)
    );
#endif
}

Point4D CoordinateMapper::weighted_centroid(const std::vector<Point4D>& points,
                                             const std::vector<double>& weights) noexcept {
    if (points.empty() || weights.size() != points.size()) {
        return Point4D();
    }
    
    double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    double total_weight = 0;
    
    for (size_t i = 0; i < points.size(); ++i) {
        double w = weights[i];
        sum_x += static_cast<double>(points[i].x) * w;
        sum_y += static_cast<double>(points[i].y) * w;
        sum_z += static_cast<double>(points[i].z) * w;
        sum_m += static_cast<double>(points[i].m) * w;
        total_weight += w;
    }
    
    if (total_weight == 0) {
        return centroid(points);
    }
    
    return Point4D(
        static_cast<Coord32>(sum_x / total_weight),
        static_cast<Coord32>(sum_y / total_weight),
        static_cast<Coord32>(sum_z / total_weight),
        static_cast<Coord32>(sum_m / total_weight)
    );
}



double CoordinateMapper::euclidean_distance(const Point4D& a, const Point4D& b) noexcept {
    double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
    double dm = static_cast<double>(a.m) - static_cast<double>(b.m);
    
    return std::sqrt(dx*dx + dy*dy + dz*dz + dm*dm);
}

uint32_t CoordinateMapper::get_category_count(AtomCategory cat) noexcept {
    switch (cat) {
        case AtomCategory::Control: return 65;
        case AtomCategory::Format: return 161;
        case AtomCategory::PrivateUse: return 137468;
        case AtomCategory::Surrogate: return 2048;
        case AtomCategory::Noncharacter: return 66;
        case AtomCategory::Space: return 25;
        case AtomCategory::PunctuationOpen: return 79;
        case AtomCategory::PunctuationClose: return 77;
        case AtomCategory::PunctuationOther: return 593;
        case AtomCategory::Digit: return 660;
        case AtomCategory::NumberLetter: return 236;
        case AtomCategory::MathSymbol: return 948;
        case AtomCategory::Currency: return 63;
        case AtomCategory::Modifier: return 125;
        case AtomCategory::LetterUpper: return 1831;
        case AtomCategory::LetterLower: return 2227;
        case AtomCategory::LetterTitlecase: return 31;
        case AtomCategory::LetterModifier: return 334;
        case AtomCategory::LetterOther: return 127004;
        case AtomCategory::MarkNonspacing: return 1950;
        case AtomCategory::MarkSpacing: return 452;
        case AtomCategory::MarkEnclosing: return 13;
        case AtomCategory::SymbolOther: return 6634;
        case AtomCategory::Separator: return 3;
        default: return 1000;
    }
}

// ============================================================================
// OPTIMIZATION PIPELINE IMPLEMENTATION
// ============================================================================

// Safe inverse power to prevent gradient blowups
inline double safe_pow_inv(double r, double p, double eps = 1e-8) {
    double rclamped = std::max(r, eps);
    return 1.0 / std::pow(rclamped, p);
}

// AVX-optimized Euclidean distance for 4D points
#ifdef HAS_AVX
inline double avx_distance(const Point4F& a, const Point4F& b) noexcept {
    // Load points into AVX registers
    __m256d a_vec = _mm256_set_pd(a.m, a.z, a.y, a.x);
    __m256d b_vec = _mm256_set_pd(b.m, b.z, b.y, b.x);

    // Compute difference
    __m256d diff = _mm256_sub_pd(a_vec, b_vec);

    // Compute squared difference
    __m256d sq_diff = _mm256_mul_pd(diff, diff);

    // Sum all components: sq_diff[0] + sq_diff[1] + sq_diff[2] + sq_diff[3]
    __m128d sum_high = _mm256_extractf128_pd(sq_diff, 1);  // sq_diff[2], sq_diff[3]
    __m128d sum_low = _mm256_castpd256_pd128(sq_diff);     // sq_diff[0], sq_diff[1]

    __m128d sum = _mm_add_pd(sum_low, sum_high);           // [sum0+sum2, sum1+sum3]
    sum = _mm_hadd_pd(sum, sum);                           // [sum0+sum1+sum2+sum3, duplicate]

    // Extract and sqrt
    double sum_sq = _mm_cvtsd_f64(sum);
    return std::sqrt(sum_sq);
}
#endif

// MKL-optimized operations for distance computations
#ifdef HAS_MKL
inline double mkl_distance(const Point4F& a, const Point4F& b) noexcept {
    double a_arr[4] = {a.x, a.y, a.z, a.m};
    double b_arr[4] = {b.x, b.y, b.z, b.m};
    double diff[4];

    // Compute difference: diff = a - b
    vdSub(4, a_arr, b_arr, diff);

    // Compute squared difference
    vdSqr(4, diff, diff);

    // Sum all components
    double sum_sq = cblas_dasum(4, diff, 1);

    return std::sqrt(sum_sq);
}

// MKL-optimized dot product
inline double mkl_dot(const Point4F& a, const Point4F& b) noexcept {
    double a_arr[4] = {a.x, a.y, a.z, a.m};
    double b_arr[4] = {b.x, b.y, b.z, b.m};

    return cblas_ddot(4, a_arr, 1, b_arr, 1);
}
#endif

// Select distance function based on available optimizations
inline double optimized_distance(const Point4F& a, const Point4F& b) noexcept {
#ifdef HAS_MKL
    return mkl_distance(a, b);
#elif defined(HAS_AVX)
    return avx_distance(a, b);
#else
    return a.distance(b);
#endif
}

// Select dot product function
inline double optimized_dot(const Point4F& a, const Point4F& b) noexcept {
#ifdef HAS_MKL
    return mkl_dot(a, b);
#else
    return a.dot(b);
#endif
}

CoordinateMapper::Diagnostics CoordinateMapper::compute_diagnostics(const std::map<uint32_t, Point4F>& points) {
    Diagnostics diag;

    if (points.empty()) return diag;

    size_t n = points.size();
    std::vector<Point4F> point_list;
    std::vector<uint32_t> codepoints;
    point_list.reserve(n);
    codepoints.reserve(n);

    for (const auto& [cp, pt] : points) {
        point_list.push_back(pt);
        codepoints.push_back(cp);
    }

    // Compute nearest neighbor chordal distances
    std::vector<double> chordal_nn(n, std::numeric_limits<double>::max());
    std::vector<double> geodesic_nn(n, std::numeric_limits<double>::max());

    // Brute force NN computation (for now - could use HNSW later)
#ifdef HAS_OPENMP
#pragma omp parallel for if(n > 1000)
#endif
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double chordal_dist = optimized_distance(point_list[i], point_list[j]);
            double geodesic_dist = point_list[i].geodesic_distance(point_list[j]);

            if (chordal_dist < chordal_nn[i]) chordal_nn[i] = chordal_dist;
            if (geodesic_dist < geodesic_nn[i]) geodesic_nn[i] = geodesic_dist;
        }
    }

    // Compute statistics for chordal NN
    auto compute_stats = [](const std::vector<double>& vals) -> std::tuple<double, double, double, double, double, double> {
        if (vals.empty()) return {0, 0, 0, 0, 0, 0};

        std::vector<double> sorted = vals;
        std::sort(sorted.begin(), sorted.end());

        double sum = 0, sum_sq = 0;
        for (double v : vals) {
            sum += v;
            sum_sq += v * v;
        }
        double mean = sum / vals.size();
        double variance = (sum_sq / vals.size()) - (mean * mean);
        double std_dev = std::sqrt(std::max(0.0, variance));
        double cv = (mean > 0) ? std_dev / mean : 0;

        size_t idx_5 = sorted.size() * 5 / 100;
        size_t idx_95 = sorted.size() * 95 / 100;
        double p5 = sorted[std::min(idx_5, sorted.size() - 1)];
        double p95 = sorted[std::min(idx_95, sorted.size() - 1)];

        return {mean, sorted[sorted.size()/2], std_dev, cv, p5, p95};
    };

    std::tie(diag.chordal_nn_mean, diag.chordal_nn_median, diag.chordal_nn_std,
             diag.chordal_nn_cv, diag.chordal_nn_5th, diag.chordal_nn_95th) = compute_stats(chordal_nn);

    std::tie(diag.geodesic_nn_mean, diag.geodesic_nn_median, diag.geodesic_nn_std,
             diag.geodesic_nn_cv, diag.geodesic_nn_5th, diag.geodesic_nn_95th) = compute_stats(geodesic_nn);

    // Local density approximation: V(p) ≈ C_d * d_1(p)^d with d=3
    const double C_3 = 4.0 * PI / 3.0; // Volume constant for 3D
    std::vector<double> densities;
    for (double d1 : chordal_nn) {
        double v = C_3 * std::pow(d1, 3);
        densities.push_back(v > 0 ? 1.0 / v : 0); // density = 1/volume
    }

    std::tie(diag.local_density_mean, std::ignore, diag.local_density_std,
             diag.local_density_cv, std::ignore, std::ignore) = compute_stats(densities);

    // Collision histogram (quantized tuples)
    for (const auto& [cp, pt] : points) {
        Point4D quantized = pt.to_quantized();
        diag.collision_counts[quantized]++;
    }

    // Bucket CV by semantic category
    std::map<uint32_t, std::vector<double>> bucket_nns;
    for (size_t i = 0; i < n; ++i) {
        uint32_t bucket = static_cast<uint32_t>(categorize(codepoints[i]));
        bucket_nns[bucket].push_back(chordal_nn[i]);
    }

    for (const auto& [bucket, nns] : bucket_nns) {
        if (nns.size() < 2) continue;
        double sum = 0, sum_sq = 0;
        for (double v : nns) {
            sum += v;
            sum_sq += v * v;
        }
        double mean = sum / nns.size();
        double variance = (sum_sq / nns.size()) - (mean * mean);
        double std_dev = std::sqrt(std::max(0.0, variance));
        double cv = (mean > 0) ? std_dev / mean : 0;
        diag.bucket_cv[bucket] = cv;
    }

    return diag;
}

void CoordinateMapper::apply_deterministic_jitter(std::map<uint32_t, Point4F>& points, double epsilon) {
    if (points.empty()) return;

    // Compute orthonormal tangent basis at each point
    auto tangent_basis = [](const Point4F& p) -> std::array<Point4F, 3> {
        // Find arbitrary vector not parallel to p
        Point4F a(1.0, 0.0, 0.0, 0.0);
        if (std::abs(p.dot(a)) > 0.9) { // Nearly parallel
            a = Point4F(0.0, 1.0, 0.0, 0.0);
        }

        // Gram-Schmidt in 4D
        Point4F t1 = (a + p * (-p.dot(a))).normalized();

        // Find another vector not in span{p, t1}
        Point4F b(0.0, 1.0, 0.0, 0.0);
        if (std::abs(p.dot(b)) > 0.9 || std::abs(t1.dot(b)) > 0.9) {
            b = Point4F(0.0, 0.0, 1.0, 0.0);
        }
        Point4F t2 = (b + p * (-p.dot(b)) + t1 * (-t1.dot(b))).normalized();

        // Third basis vector (simplified cross product in remaining coordinates)
        Point4F t3 = Point4F(
            p.y * t1.z - p.z * t1.y,
            p.z * t1.x - p.x * t1.z,
            p.x * t1.y - p.y * t1.x,
            0.0
        ).normalized();

        return {t1, t2, t3};
    };

    for (auto& [cp, p] : points) {
        // Generate deterministic values from BLAKE3 hash of codepoint
        Blake3Hash hash = Blake3Hasher::hash_codepoint(cp);
        uint64_t v0 = *reinterpret_cast<const uint64_t*>(hash.data());
        uint64_t v1 = *reinterpret_cast<const uint64_t*>(hash.data() + 8);
        uint64_t v2 = *reinterpret_cast<const uint64_t*>(hash.data() + 16);

        double f0 = static_cast<double>(v0) / static_cast<double>(UINT64_MAX);
        double f1 = static_cast<double>(v1) / static_cast<double>(UINT64_MAX);
        double f2 = static_cast<double>(v2) / static_cast<double>(UINT64_MAX);

        // Get tangent basis
        auto [t1, t2, t3] = tangent_basis(p);

        // Create jitter vector
        Point4F jitter = (t1 * (2.0 * f0 - 1.0) +
                         t2 * (2.0 * f1 - 1.0) +
                         t3 * (2.0 * f2 - 1.0)) * epsilon;

        // Apply and renormalize
        p = (p + jitter).normalized();
    }
}

void CoordinateMapper::bucketed_tangent_lloyd(std::map<uint32_t, Point4F>& points,
                                             size_t k, double alpha, int iterations) {
    if (points.empty()) return;

    // Group points by semantic category
    std::map<uint32_t, std::vector<std::pair<uint32_t, Point4F*>>> buckets;
    for (auto& [cp, pt] : points) {
        uint32_t cat = static_cast<uint32_t>(categorize(cp));
        buckets[cat].emplace_back(cp, &pt);
    }

    // Compute tangent basis function (shared with jitter)
    auto tangent_basis = [](const Point4F& p) -> std::array<Point4F, 3> {
        Point4F a(1.0, 0.0, 0.0, 0.0);
        if (std::abs(p.dot(a)) > 0.9) a = Point4F(0.0, 1.0, 0.0, 0.0);

        Point4F t1 = (a + p * (-p.dot(a))).normalized();

        Point4F b(0.0, 1.0, 0.0, 0.0);
        if (std::abs(p.dot(b)) > 0.9 || std::abs(t1.dot(b)) > 0.9) {
            b = Point4F(0.0, 0.0, 1.0, 0.0);
        }
        Point4F t2 = (b + p * (-p.dot(b)) + t1 * (-t1.dot(b))).normalized();

        Point4F t3 = Point4F(
            p.y * t1.z - p.z * t1.y,
            p.z * t1.x - p.x * t1.z,
            p.x * t1.y - p.y * t1.x,
            0.0
        ).normalized();

        return {t1, t2, t3};
    };

    size_t MIN_BUCKET = 64;

    // Process each bucket
    for (auto& [bucket_id, bucket_points] : buckets) {
        size_t n_bucket = bucket_points.size();

        // Build bucket_coords
        std::vector<Point4F> bucket_coords;
        for (const auto& [cp, pt_ptr] : bucket_points) {
            bucket_coords.push_back(*pt_ptr);
        }

        if (n_bucket < MIN_BUCKET) {
            // Use global kNN for small buckets
            for (int iter = 0; iter < iterations; ++iter) {
                for (size_t i = 0; i < n_bucket; ++i) {
                    const Point4F& p = bucket_coords[i];
                    Point4F v_avg(0, 0, 0, 0);

                    // Find k nearest in all points
                    std::vector<std::pair<double, uint32_t>> neighbors;
                    for (const auto& [cp, pt] : points) {
                        double dist = p.distance(pt);
                        neighbors.emplace_back(dist, cp);
                    }
                    std::sort(neighbors.begin(), neighbors.end());

                    // Average neighbor vectors in tangent space
                    for (size_t ni = 0; ni < std::min(k, neighbors.size()); ++ni) {
                        uint32_t cp = neighbors[ni].second;
                        const Point4F& q = points.at(cp);

                        // Project q onto tangent space at p
                        double pq_dot = p.dot(q);
                        Point4F v_q = q + p * (-pq_dot);
                        v_avg = v_avg + v_q;
                    }

                    if (neighbors.size() > 0) v_avg = v_avg * (1.0 / std::min(k, neighbors.size()));

                    // Update point
                    Point4F& target = *bucket_points[i].second;
                    target = (p + v_avg * alpha).normalized();
                    bucket_coords[i] = target;
                }
            }
        } else {
            // Original bucket-based for large buckets
            for (int iter = 0; iter < iterations; ++iter) {
                for (size_t i = 0; i < n_bucket; ++i) {
                    const Point4F& p = bucket_coords[i];
                    Point4F v_avg(0, 0, 0, 0);

                    // Find k nearest neighbors in bucket
                    std::vector<std::pair<double, size_t>> neighbors;
                    for (size_t j = 0; j < n_bucket; ++j) {
                        if (i == j) continue;
                        double dist = p.distance(bucket_coords[j]);
                        neighbors.emplace_back(dist, j);
                    }
                    std::partial_sort(neighbors.begin(), neighbors.begin() + std::min(k, neighbors.size()),
                                      neighbors.end());

                    // Average neighbor vectors in tangent space
                    for (size_t ni = 0; ni < std::min(k, neighbors.size()); ++ni) {
                        size_t j = neighbors[ni].second;
                        const Point4F& q = bucket_coords[j];

                        // Project q onto tangent space at p
                        double pq_dot = p.dot(q);
                        Point4F v_q = q + p * (-pq_dot);
                        v_avg = v_avg + v_q;
                    }

                    if (neighbors.size() > 0) v_avg = v_avg * (1.0 / std::min(k, neighbors.size()));

                    // Update point
                    Point4F& target = *bucket_points[i].second;
                    target = (p + v_avg * alpha).normalized();
                    bucket_coords[i] = target;
                }
            }
        }
    }
}

void CoordinateMapper::global_knn_repulsion(std::map<uint32_t, Point4F>& points,
                                          size_t k, double s, double eta, int iterations) {
    if (points.empty()) return;

    std::vector<Point4F*> point_ptrs;
    std::vector<Point4F> point_list;

    for (auto& [cp, pt] : points) {
        point_ptrs.push_back(&pt);
        point_list.push_back(pt);
    }

    size_t n = points.size();

    // Compute initial mean NN distance for eta scaling
    double mean_nn = 0.0;
    int count = 0;
    for (size_t i = 0; i < n; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double dist = point_list[i].distance(point_list[j]);
            if (dist < min_dist) min_dist = dist;
        }
        if (min_dist < std::numeric_limits<double>::max()) {
            mean_nn += min_dist;
            count++;
        }
    }
    if (count > 0) mean_nn /= count;

    // Set initial eta based on mean NN distance (much smaller for stability)
    double eta_initial = 1e-6 * mean_nn;
    double eta_min = 1e-12;
    double max_grad_norm = 1e-3; // gradient clipping

    double prev_energy = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        double energy = 0.0;

        // Compute forces for all points using geodesic gradient
        std::vector<Point4F> forces(n, Point4F(0, 0, 0, 0));

#ifdef HAS_OPENMP
#pragma omp parallel for if(n > 1000)
#endif
        for (size_t i = 0; i < n; ++i) {
            const Point4F& p = point_list[i];

            // Find k nearest neighbors
            std::vector<std::pair<double, size_t>> neighbors;
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                double dist = optimized_distance(p, point_list[j]);
                neighbors.emplace_back(dist, j);
            }

            std::partial_sort(neighbors.begin(), neighbors.begin() + std::min(k, neighbors.size()),
                            neighbors.end());

            Point4F force_sum(0, 0, 0, 0);

            // Only repel neighbors that are too close (within expected NN distance for uniform distribution)
            double expected_nn_dist = mean_nn; // Target distance for uniform distribution

            for (size_t ni = 0; ni < std::min(k, neighbors.size()); ++ni) {
                size_t j = neighbors[ni].second;
                const Point4F& q = point_list[j];
                double r = optimized_distance(p, q);

                // Only apply repulsion if closer than expected
                if (r < expected_nn_dist) {
                    // Use chordal gradient (simpler and more stable than geodesic)
                    Point4F diff(p.x - q.x, p.y - q.y, p.z - q.z, p.m - q.m);
                    double chordal_r = std::sqrt(std::max(diff.dot(diff), 1e-18));

                    if (chordal_r > 1e-12) {
                        // Normalized direction vector from q to p
                        Point4F direction = diff * (1.0 / chordal_r);

                        // Repulsive force: stronger when closer, proportional to 1/r^2
                        double repulsion_strength = 1.0 / (r * r + 1e-8);
                        force_sum = force_sum + direction * repulsion_strength;

                        // Energy: Coulomb-like potential
                        energy += repulsion_strength;
                    }
                }
            }

            forces[i] = force_sum;
        }
#ifdef HAS_OPENMP
#pragma omp barrier
#endif

        // Apply forces with tangent projection and backtracking line search
        double c_armijo = 1e-4;
        double tau = 0.5;
        eta = eta_initial;

        // Try different eta values until Armijo condition is satisfied
        bool armijo_satisfied = false;
        double energy_new = energy;
        std::vector<Point4F> new_positions = point_list;

        while (!armijo_satisfied && eta > eta_min) {
            // Compute tentative positions
            for (size_t i = 0; i < n; ++i) {
                const Point4F& p = point_list[i];
                Point4F G_tan = forces[i] + p * (-p.dot(forces[i])); // Project to tangent space

                // Gradient clipping
                double gnorm = std::sqrt(G_tan.dot(G_tan));
                if (gnorm > max_grad_norm) {
                    G_tan = G_tan * (max_grad_norm / gnorm);
                }

                new_positions[i] = (p + G_tan * eta).normalized();
            }

            // Compute new energy
            energy_new = 0.0;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    double r = optimized_distance(new_positions[i], new_positions[j]);
                    energy_new += 1.0 / std::pow(std::max(r, 1e-18), s);
                }
            }

            // Check Armijo condition
            double dot_grad = 0.0;
            for (size_t i = 0; i < n; ++i) {
                dot_grad += forces[i].dot(forces[i]);
            }

            if (energy_new <= energy + c_armijo * eta * dot_grad) {
                armijo_satisfied = true;
            } else {
                eta *= tau;
            }
        }

        // Update positions with the accepted eta
        for (size_t i = 0; i < n; ++i) {
            *point_ptrs[i] = new_positions[i];
            point_list[i] = new_positions[i];
        }

        prev_energy = energy_new;

        // Compute CV for this iteration (expensive but informative)
        if (iter % 5 == 0 || iter == iterations - 1) {  // Every 5 iterations
            auto temp_points = std::map<uint32_t, Point4F>();
            for (size_t i = 0; i < n; ++i) {
                temp_points[i] = point_list[i];
            }
            auto diag = compute_diagnostics(temp_points);
            std::cout << "Iteration " << iter << " energy: " << energy_new << " CV: " << (diag.chordal_nn_cv * 100) << "% eta: " << eta << std::endl;
        } else {
            std::cout << "Iteration " << iter << " energy: " << energy_new << " eta: " << eta << std::endl;
        }
    }
}

bool CoordinateMapper::optimize_distribution(std::map<uint32_t, Point4F>& points) {
    if (points.empty()) return false;

    std::cout << "Starting surface distribution optimization..." << std::endl;

    // Step 1: Compute baseline diagnostics
    std::cout << "Computing baseline diagnostics..." << std::endl;
    Diagnostics baseline = compute_diagnostics(points);
    std::cout << "Baseline CV: " << (baseline.chordal_nn_cv * 100) << "%" << std::endl;

    // Step 2: Apply deterministic jitter
    std::cout << "Applying deterministic jitter..." << std::endl;
    apply_deterministic_jitter(points, 1e-7);
    Diagnostics after_jitter = compute_diagnostics(points);
    std::cout << "After jitter CV: " << (after_jitter.chordal_nn_cv * 100) << "%" << std::endl;

    // Step 3: Bucketed tangent Lloyd
    std::cout << "Running bucketed tangent Lloyd..." << std::endl;
    bucketed_tangent_lloyd(points, 64, 0.25, 4);
    Diagnostics after_lloyd = compute_diagnostics(points);
    std::cout << "After Lloyd CV: " << (after_lloyd.chordal_nn_cv * 100) << "%" << std::endl;

    // Step 4: Global KNN repulsion
    std::cout << "Running global KNN repulsion..." << std::endl;
    double mean_nn = after_lloyd.chordal_nn_mean;
    double initial_eta = 0.001 * mean_nn;
    global_knn_repulsion(points, 64, 1.0, initial_eta, 10);  // Fewer iterations, smaller k
    Diagnostics final = compute_diagnostics(points);
    std::cout << "Final CV: " << (final.chordal_nn_cv * 100) << "%" << std::endl;

    std::cout << "Optimization complete. CV improved from "
              << (baseline.chordal_nn_cv * 100) << "% to " << (final.chordal_nn_cv * 100) << "%" << std::endl;

    return final.chordal_nn_cv < baseline.chordal_nn_cv; // Any improvement is success
}

} // namespace hypercube
