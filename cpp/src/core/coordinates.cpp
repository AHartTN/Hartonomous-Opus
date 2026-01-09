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
 * - 32 bits per dimension = lossless, collision-free coordinates (with jitter for rare collisions)
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
// Max semantic rank: accommodates all possible semantic ranks
// Use 64-bit to avoid accidental overflow/truncation when used in arithmetic.
const uint64_t TOTAL_CODEPOINTS = 10000000ULL; // 10,000,000

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
 *   [6656, 6656+80)     Digits (consecutive within scripts, scripts grouped)
 *   [6736, 7000)        Greek
 *   [7000, 8000)        Cyrillic
 *   [8000, 9000)        Other alphabets
 *   [9000, 10000)       Punctuation
 *   [10000, 20000)      Symbols
 *   [20000, 30000)      CJK
 *   [900000, ...)       Other/Private  
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
// Relation-based semantic ordering functions

uint32_t get_script_id(uint32_t cp) {
    if (cp <= 0x024F) return 0; // Latin and extended
    if (cp >= 0x0370 && cp <= 0x03FF) return 1; // Greek
    if (cp >= 0x0400 && cp <= 0x04FF) return 2; // Cyrillic
    if (cp >= 0x0590 && cp <= 0x05FF) return 3; // Hebrew
    if (cp >= 0x0600 && cp <= 0x06FF) return 4; // Arabic
    if (cp >= 0x0900 && cp <= 0x097F) return 5; // Devanagari
    if (cp >= 0x0980 && cp <= 0x09FF) return 6; // Bengali
    if (cp >= 0x0A00 && cp <= 0x0A7F) return 7; // Gurmukhi
    if (cp >= 0x0A80 && cp <= 0x0AFF) return 8; // Gujarati
    if (cp >= 0x0B00 && cp <= 0x0B7F) return 9; // Oriya
    if (cp >= 0x0B80 && cp <= 0x0BFF) return 10; // Tamil
    if (cp >= 0x0C00 && cp <= 0x0C7F) return 11; // Telugu
    if (cp >= 0x0C80 && cp <= 0x0CFF) return 12; // Kannada
    if (cp >= 0x0D00 && cp <= 0x0D7F) return 13; // Malayalam
    if (cp >= 0x0E00 && cp <= 0x0E7F) return 14; // Thai
    if (cp >= 0x0E80 && cp <= 0x0EFF) return 15; // Lao
    if (cp >= 0x2E80 && cp <= 0x2EFF) return 16; // CJK radicals
    if (cp >= 0x2F00 && cp <= 0x2FDF) return 17; // Kangxi
    if (cp >= 0x3000 && cp <= 0x303F) return 18; // CJK symbols
    if (cp >= 0x3100 && cp <= 0x312F) return 19; // Bopomofo
    if (cp >= 0x3400 && cp <= 0x4DBF) return 20; // CJK ext A
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 21; // CJK basic
    if (cp >= 0x20000 && cp <= 0x2A6DF) return 22; // CJK ext B
    if (cp >= 0x2A700 && cp <= 0x2B73F) return 23; // CJK ext C
    if (cp >= 0x2B740 && cp <= 0x2B81F) return 24; // CJK ext D
    if (cp >= 0x2B820 && cp <= 0x2CEAF) return 25; // CJK ext E
    if (cp >= 0x2CEB0 && cp <= 0x2EBEF) return 26; // CJK ext F
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 27; // Hangul
    if (cp >= 0x1F600 && cp <= 0x1F64F) return 28; // Emoji
    return 100; // other
}

uint32_t case_fold(uint32_t cp) {
    // ASCII
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    // Greek
    if (cp >= 0x0391 && cp <= 0x03A9) return cp + 32;
    // Cyrillic (basic)
    if (cp >= 0x0410 && cp <= 0x042F) return cp + 32;
    // Add more as needed
    return cp;
}



// Get the base letter (0-25 for A-Z) for Latin characters, accounting for accents
uint32_t get_latin_base(uint32_t cp) noexcept {
    // ASCII A-Z
    if (cp >= 'A' && cp <= 'Z') return cp - 'A';
    // ASCII a-z (fold to uppercase base)
    if (cp >= 'a' && cp <= 'z') return cp - 'a';

    // Latin Extended-A: pairs of (uppercase, lowercase)
    if (cp >= 0x00C0 && cp <= 0x00D6) {  // À-Ö (skip ÷)
        if (cp == 0x00D7) return 26; // × is not a letter
        uint32_t offset = cp - 0x00C0;
        return offset;  // A + offset
    }
    if (cp >= 0x00D8 && cp <= 0x00DE) {  // Ø-Þ
        uint32_t offset = cp - 0x00D8;
        return 14 + offset;  // O + offset (after N)
    }
    if (cp >= 0x00DF && cp <= 0x00F6) {  // ß-ö
        if (cp == 0x00DF) return 18; // ß -> S (approximate)
        if (cp == 0x00F7) return 26; // ÷ not a letter
        uint32_t offset = cp - 0x00DF;
        return offset;  // a + offset, but we return base as uppercase equivalent
    }
    if (cp >= 0x00F8 && cp <= 0x00FF) {  // ø-ÿ
        uint32_t offset = cp - 0x00F8;
        return 14 + offset;  // o + offset (after n)
    }

    // Latin Extended-B (simplified mapping)
    if (cp >= 0x0100 && cp <= 0x017F) {
        // Each pair: uppercase, lowercase
        uint32_t pair_idx = (cp - 0x0100) / 2;
        return pair_idx % 26;  // Cycle through A-Z
    }
    if (cp >= 0x0180 && cp <= 0x024F) {
        // More extended Latin
        return (cp - 0x0180) % 26;
    }

    // Not a Latin letter
    return 26;
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

// Relation-based semantic ordering system
// Groups characters by linguistic relationships rather than blocks
uint64_t get_semantic_order(uint32_t cp) noexcept {
    // Categories by linguistic similarity (base offsets) - adjusted for consecutive ordering
    const uint64_t LATIN_BASE = 0ULL;
    const uint64_t DIGIT_BASE = 6656ULL;
    const uint64_t GREEK_BASE = 7000ULL;
    const uint64_t CYRILLIC_BASE = 8000ULL;
    const uint64_t OTHER_ALPHA_BASE = 9000ULL;
    const uint64_t PUNCTUATION_BASE = 10000ULL;
    const uint64_t SYMBOL_BASE = 20000ULL;
    const uint64_t CJK_BASE = 30000ULL;
    const uint64_t OTHER_BASE = 900000ULL;

    // === LATIN LETTERS ===
    // Group by base letter, then case, then accent type
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
        (cp >= 0x00C0 && cp <= 0x024F)) {
        uint32_t base = get_latin_base(cp);
        if (base < 26) {
            uint32_t variant = get_latin_variant_order(cp);
            return LATIN_BASE + base * 256ULL + variant;
        }
    }

    // === GREEK LETTERS ===
    // Lowercase first, then uppercase paired, then extended
    if (cp >= 0x0370 && cp <= 0x03FF) {
        if (cp >= 0x03B1 && cp <= 0x03C9) {
            uint32_t offset = cp - 0x03B1;
            return GREEK_BASE + offset;
        }
        if (cp >= 0x0391 && cp <= 0x03A9) {
            uint32_t offset = cp - 0x0391;
            if (cp > 0x03A1) offset--; // Skip final sigma
            return GREEK_BASE + 25ULL + offset;
        }
        return GREEK_BASE + 1000ULL + (cp - 0x0370);
    }
    if (cp >= 0x1F00 && cp <= 0x1FFF) {
        return GREEK_BASE + 2000ULL + (cp - 0x1F00);
    }

    // === CYRILLIC ===
    if (cp >= 0x0400 && cp <= 0x052F) {
        if (cp >= 0x0410 && cp <= 0x042F) return CYRILLIC_BASE + (cp - 0x0410) * 2ULL;
        if (cp >= 0x0430 && cp <= 0x044F) return CYRILLIC_BASE + (cp - 0x0430) * 2ULL + 1ULL;
        return CYRILLIC_BASE + 1000ULL + (cp - 0x0400);
    }

    // === OTHER ALPHABETS ===
    // Group similar scripts together
    if (cp >= 0x0590 && cp <= 0x05FF) return OTHER_ALPHA_BASE + 0ULL + (cp - 0x0590);      // Hebrew
    if (cp >= 0x0600 && cp <= 0x077F) return OTHER_ALPHA_BASE + 1000ULL + (cp - 0x0600);    // Arabic
    if (cp >= 0x0900 && cp <= 0x0DFF) return OTHER_ALPHA_BASE + 2000ULL + (cp - 0x0900);    // Indic
    if (cp >= 0x0E00 && cp <= 0x0EFF) return OTHER_ALPHA_BASE + 3000ULL + (cp - 0x0E00);    // Thai/Lao
    if (cp >= 0x1100 && cp <= 0x11FF) return OTHER_ALPHA_BASE + 4000ULL + (cp - 0x1100);    // Hangul Jamo
    if (cp >= 0xAC00 && cp <= 0xD7AF) return OTHER_ALPHA_BASE + 5000ULL + (cp - 0xAC00);    // Hangul Syllables

    // === DIGITS ===
    // All digit representations close together
    if ((cp >= '0' && cp <= '9') || (cp >= 0x0660 && cp <= 0x0669) ||
        (cp >= 0x06F0 && cp <= 0x06F9) || (cp >= 0x0966 && cp <= 0x096F) ||
        (cp >= 0x09E6 && cp <= 0x09EF) || (cp >= 0x0BE6 && cp <= 0x0BEF) ||
        (cp >= 0x0E50 && cp <= 0x0E59) || (cp >= 0xFF10 && cp <= 0xFF19)) {
        // Extract digit value 0-9
        uint32_t digit_val = 0;
        uint32_t script_offset = 0;
        if (cp >= '0' && cp <= '9') {
            digit_val = cp - '0';
            script_offset = 0;
        } else if (cp >= 0xFF10 && cp <= 0xFF19) {
            digit_val = cp - 0xFF10;
            script_offset = 1;
        } else if (cp >= 0x0660 && cp <= 0x0669) {
            digit_val = cp - 0x0660;
            script_offset = 2;
        } else if (cp >= 0x06F0 && cp <= 0x06F9) {
            digit_val = cp - 0x06F0;
            script_offset = 3;
        } else if (cp >= 0x0966 && cp <= 0x096F) {
            digit_val = cp - 0x0966;
            script_offset = 4;
        } else if (cp >= 0x09E6 && cp <= 0x09EF) {
            digit_val = cp - 0x09E6;
            script_offset = 5;
        } else if (cp >= 0x0BE6 && cp <= 0x0BEF) {
            digit_val = cp - 0x0BE6;
            script_offset = 6;
        } else if (cp >= 0x0E50 && cp <= 0x0E59) {
            digit_val = cp - 0x0E50;
            script_offset = 7;
        }
        return DIGIT_BASE + script_offset * 10ULL + digit_val; // Consecutive within script, scripts grouped
    }

    // === PUNCTUATION ===
    // Group by function
    if ((cp >= 0x0020 && cp <= 0x002F) || (cp >= 0x003A && cp <= 0x0040) ||
        (cp >= 0x005B && cp <= 0x0060) || (cp >= 0x007B && cp <= 0x007E)) {
        return PUNCTUATION_BASE + 0ULL + cp; // ASCII punctuation
    }
    if (cp >= 0x2000 && cp <= 0x206F) {
        return PUNCTUATION_BASE + 1000ULL + (cp - 0x2000); // General punctuation
    }
    if (cp >= 0x3000 && cp <= 0x303F) {
        return PUNCTUATION_BASE + 2000ULL + (cp - 0x3000); // CJK punctuation
    }

    // === SYMBOLS ===
    if (cp >= 0x20A0 && cp <= 0x20CF) return SYMBOL_BASE + 0ULL + (cp - 0x20A0);      // Currency
    if (cp >= 0x2100 && cp <= 0x214F) return SYMBOL_BASE + 1000ULL + (cp - 0x2100);   // Letterlike
    if (cp >= 0x2190 && cp <= 0x21FF) return SYMBOL_BASE + 2000ULL + (cp - 0x2190);   // Arrows
    if (cp >= 0x2200 && cp <= 0x22FF) return SYMBOL_BASE + 3000ULL + (cp - 0x2200);   // Math
    if (cp >= 0x2500 && cp <= 0x257F) return SYMBOL_BASE + 4000ULL + (cp - 0x2500);   // Box drawing
    if (cp >= 0x25A0 && cp <= 0x25FF) return SYMBOL_BASE + 5000ULL + (cp - 0x25A0);   // Geometric
    if (cp >= 0x2600 && cp <= 0x26FF) return SYMBOL_BASE + 6000ULL + (cp - 0x2600);   // Misc symbols
    if (cp >= 0x2700 && cp <= 0x27BF) return SYMBOL_BASE + 7000ULL + (cp - 0x2700);   // Dingbats

    // === CJK ===
    if (cp >= 0x2E80 && cp <= 0x2EFF) return CJK_BASE + 0ULL + (cp - 0x2E80);         // Radicals
    if (cp >= 0x2F00 && cp <= 0x2FDF) return CJK_BASE + 1000ULL + (cp - 0x2F00);      // Kangxi
    if (cp >= 0x3100 && cp <= 0x312F) return CJK_BASE + 2000ULL + (cp - 0x3100);      // Bopomofo
    if (cp >= 0x3400 && cp <= 0x4DBF) return CJK_BASE + 3000ULL + (cp - 0x3400);      // Ext A
    if (cp >= 0x4E00 && cp <= 0x9FFF) return CJK_BASE + 30000ULL + (cp - 0x4E00);     // Basic
    if (cp >= 0x20000 && cp <= 0x2A6DF) return CJK_BASE + 100000ULL + (cp - 0x20000); // Ext B
    if (cp >= 0x2A700 && cp <= 0x2B73F) return CJK_BASE + 200000ULL + (cp - 0x2A700); // Ext C
    // More CJK extensions...

    // === EMOJI ===
    if (cp >= 0x1F300 && cp <= 0x1F9FF) {
        return SYMBOL_BASE + 10000ULL + (cp - 0x1F300);
    }

    // === CONTROL/FORMAT ===
    if (cp <= 0x001F || (cp >= 0x007F && cp <= 0x009F)) {
        return OTHER_BASE + 0ULL + cp;
    }
    if (cp >= 0x200B && cp <= 0x206F) {
        return OTHER_BASE + 1000ULL + (cp - 0x200B);
    }

    // === PRIVATE USE ===
    if (cp >= 0xE000 && cp <= 0xF8FF) return OTHER_BASE + 2000ULL + (cp - 0xE000);
    if (cp >= 0xF0000 && cp <= 0x10FFFD) return OTHER_BASE + 10000ULL + (cp - 0xF0000);

    // === CATCH-ALL ===
    return OTHER_BASE + 50000ULL + cp;
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
        Point4F float_coords(1.0, 0.0, 0.0, 0.0); // Unit sphere point
        Point4D p;
        p.x = CENTER ^ 0x7FFFFFFFU; // deterministic reserved value (non-zero)
        p.y = CENTER;
        p.z = CENTER;
        p.m = CENTER;
        return CodepointMapping{float_coords, p, HilbertIndex{0, 0}};
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

    // Collision resolution: if this quantized point is already used by another codepoint,
    // apply tiny deterministic jitter to find an unused slot
    //
    // ⚠️ This proves Hilbert indices are NOT unique identifiers!
    // Different codepoints can end up with identical Hilbert indices after collision resolution.
    struct Point4DHash {
        size_t operator()(const Point4D& p) const noexcept {
            return std::hash<uint32_t>()(p.x) ^ std::hash<uint32_t>()(p.y) ^
                   std::hash<uint32_t>()(p.z) ^ std::hash<uint32_t>()(p.m);
        }
    };
    static std::unordered_map<Point4D, uint32_t, Point4DHash> collision_table;
    static std::mutex collision_mutex;

    {
        std::lock_guard<std::mutex> lock(collision_mutex);
        auto it = collision_table.find(coords);
        if (it != collision_table.end() && it->second != codepoint) {
            // Collision detected - apply deterministic jitter
            Blake3Hash hash = Blake3Hasher::hash_codepoint(codepoint);
            uint64_t v0 = *reinterpret_cast<const uint64_t*>(hash.data());

            // Jitter large enough to change quantization (quantum = 2.0/2^32 ≈ 4.66e-10)
            // Use 1e-8 = ~21 quanta, ensuring collision resolution without geometry distortion
            double eps = 1e-8;
            double jitter[4] = {
                (static_cast<double>((v0 >> 0) & 0xFF) / 255.0 - 0.5) * eps,
                (static_cast<double>((v0 >> 8) & 0xFF) / 255.0 - 0.5) * eps,
                (static_cast<double>((v0 >> 16) & 0xFF) / 255.0 - 0.5) * eps,
                (static_cast<double>((v0 >> 24) & 0xFF) / 255.0 - 0.5) * eps
            };

            // Apply jitter and requantize
            Point4F jittered = float_coords + Point4F(jitter[0], jitter[1], jitter[2], jitter[3]);
            jittered = jittered.normalized();  // Keep on sphere
            float_coords = jittered;  // Update canonical coords

#ifdef HAS_AVX
            avx_quantize_point4f_to_point4d(jittered, coords);
#else
            coords.x = quantize_unit_to_u32(jittered.x);
            coords.y = quantize_unit_to_u32(jittered.y);
            coords.z = quantize_unit_to_u32(jittered.z);
            coords.m = quantize_unit_to_u32(jittered.m);
#endif
        }

        // Record this mapping
        collision_table[coords] = codepoint;
    }

    // === STEP 4: Compute Hilbert index from coordinates
    // WARNING: Hilbert index is for SPATIAL INDEXING only, NOT unique identification
    // Multiple different Point4D coordinates can produce the same HilbertIndex due to quantization
    // Use Blake3Hash for unique identification and primary keys
    HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);

    return CodepointMapping{float_coords, coords, hilbert};
}

Point4F CoordinateMapper::map_codepoint_float(uint32_t codepoint) noexcept {
    // Surrogates: place them at a reserved location
    if (codepoint >= constants::SURROGATE_START && codepoint <= constants::SURROGATE_END) {
        return Point4F(1.0, 0.0, 0.0, 0.0); // Unit sphere point
    }

    // ========================================================================
    // OPTIMIZED SUPER-FIBONACCI SPIRAL FOR S³ (3-sphere/hypersphere)
    // ========================================================================
    // Use double precision instead of long double for performance
    // Precompute constants and use SIMD-friendly operations

    // Use semantic order to position semantically related codepoints adjacently
    const uint64_t i = get_semantic_order(codepoint);
    const double N = static_cast<double>(TOTAL_CODEPOINTS);

    // Precomputed irrational constants (sufficient precision for our use case)
    const double PHI_INV = 0.707106781186547524400844362104849039284835937688474036588; // 1/√2
    const double PSI_INV = 0.652703644666139308692278852717862990236130525703414;     // 1/ψ

    // Normalized position and angular base (use double throughout)
    const double s = (static_cast<double>(i) + 0.5) / N;
    const double ab = 2.0 * PI * (static_cast<double>(i) + 0.5);

    // Spiral angles - compute simultaneously for better ILP
    const double theta = ab * PHI_INV;
    const double phi = ab * PSI_INV;

    // Radii ensuring unit sphere (use fast sqrt approximations if available)
    const double r = std::sqrt(s);
    const double R = std::sqrt(1.0 - s);

    // Compute quaternion coordinates
    // Use separate sin/cos calls (sincos not available on all platforms)
    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);
    const double sin_phi = std::sin(phi);
    const double cos_phi = std::cos(phi);

    const double q0 = r * sin_theta;
    const double q1 = r * cos_theta;
    const double q2 = R * sin_phi;
    const double q3 = R * cos_phi;

    // Skip expensive normalization check for performance
    // The mathematical construction guarantees unit norm within numerical precision
    return Point4F(q0, q1, q2, q3);
}

Point4D CoordinateMapper::map_codepoint(uint32_t codepoint) noexcept {
    return map_codepoint_full(codepoint).coords;
}


Point4D CoordinateMapper::centroid(const std::vector<Point4D>& points) noexcept {
    if (points.empty()) {
        return Point4D();
    }

    const size_t n = points.size();

    // Use SIMD-friendly accumulation for better performance
    double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;

#ifdef HAS_AVX
    // AVX-optimized accumulation for large arrays
    if (n >= 8) {  // Only worthwhile for larger arrays
        __m256d sum_vec = _mm256_setzero_pd();  // [sum_x, sum_y, sum_z, sum_m]

        // Process 2 points at a time (8 doubles)
        for (size_t i = 0; i < n; ++i) {
            Point4F p(points[i]);  // Convert to float coordinates [-1,1]

            __m256d point_vec = _mm256_set_pd(p.m, p.z, p.y, p.x);
            sum_vec = _mm256_add_pd(sum_vec, point_vec);
        }

        // Extract sums (AVX has no horizontal sum, so we do it manually)
        double sums[4];
        _mm256_storeu_pd(sums, sum_vec);
        sum_x = sums[3]; sum_y = sums[2]; sum_z = sums[1]; sum_m = sums[0];
    } else {
#endif
        // Scalar fallback for small arrays
#ifdef HAS_OPENMP
        #pragma omp parallel for reduction(+:sum_x, sum_y, sum_z, sum_m) if(n > 1000)
#endif
        for (size_t i = 0; i < n; ++i) {
            Point4F p(points[i]);  // Convert uint32 to float [-1,1]
            sum_x += p.x;
            sum_y += p.y;
            sum_z += p.z;
            sum_m += p.m;
        }
#ifdef HAS_AVX
    }
#endif

    // Compute average (reciprocal multiplication is faster than division)
    const double inv_n = 1.0 / static_cast<double>(n);
    Point4F centroid_float(sum_x * inv_n, sum_y * inv_n, sum_z * inv_n, sum_m * inv_n);

    // Normalize to S³ surface (unless it's the origin, which stays interior)
    // Use fast inverse sqrt approximation for better performance
    double norm_sq = centroid_float.dot(centroid_float);
    if (norm_sq > 1e-12) {  // Avoid division by zero
        centroid_float = centroid_float * (1.0 / std::sqrt(norm_sq));
    }

    // Quantize to uint32 for indexing
    return centroid_float.to_quantized();
}

Point4F CoordinateMapper::centroid_float(const std::vector<Point4F>& points) noexcept {
    if (points.empty()) {
        return Point4F();
    }

    size_t n = points.size();

    // Compute centroid in float space
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

    // Average in float space
    Point4F centroid_float(sum_x / n, sum_y / n, sum_z / n, sum_m / n);

    // Normalize to S³ surface (unless it's the origin, which stays interior)
    double norm_sq = centroid_float.dot(centroid_float);
    if (norm_sq > 1e-12) {  // Avoid division by zero
        centroid_float = centroid_float * (1.0 / std::sqrt(norm_sq));
    }

    return centroid_float;
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
        Point4F p(points[i]);  // Convert to float
        sum_x += p.x * w;
        sum_y += p.y * w;
        sum_z += p.z * w;
        sum_m += p.m * w;
        total_weight += w;
    }

    if (total_weight == 0) {
        return centroid(points);
    }

    // Weighted average in float space
    Point4F centroid_float(sum_x / total_weight, sum_y / total_weight,
                          sum_z / total_weight, sum_m / total_weight);

    // Normalize to S³ surface
    double norm_sq = centroid_float.dot(centroid_float);
    if (norm_sq > 1e-12) {
        centroid_float = centroid_float * (1.0 / std::sqrt(norm_sq));
    }

    // Quantize
    return centroid_float.to_quantized();
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

// Select distance function based on available optimizations
inline double optimized_distance(const Point4F& a, const Point4F& b) noexcept {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    double dm = a.m - b.m;
    return std::sqrt(dx*dx + dy*dy + dz*dz + dm*dm);
}

// Select dot product function
inline double optimized_dot(const Point4F& a, const Point4F& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.m * b.m;
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
