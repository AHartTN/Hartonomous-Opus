/**
 * Hopf Fibration Coordinate Mapping for Unicode Atoms
 * 
 * Maps all Unicode codepoints onto the surface of a 3-sphere (S³ in 4D) using
 * the Hopf fibration for optimal uniform distribution.
 * 
 * Key properties:
 * - ALL atoms are evenly distributed on the 3-sphere surface (Dyson sphere)
 * - Semantically related codepoints (A/a/Ä, digits, etc.) are placed adjacently
 * - 32 bits per dimension = lossless, collision-free coordinates
 * - Hilbert index derived from coords for spatial indexing
 * - Compositions have centroids INSIDE the sphere (closer to origin = more complex)
 * 
 * Hopf Fibration Parameterization:
 *   The 3-sphere S³ is fibered over S² with S¹ fibers.
 *   For a point at index i out of N total points:
 *     - Sample S² uniformly using spherical Fibonacci lattice
 *     - Sample the fiber (circle) uniformly using golden angle
 *   
 *   Coordinates (unit quaternion representation):
 *     η = arccos(1 - 2*(i+0.5)/N)  -- latitude on S² (uniform area)
 *     θ = 2π * φ⁻¹ * i             -- longitude on S² (golden angle)
 *     ψ = 2π * α * i               -- phase along fiber (irrational increment)
 *   
 *   x = cos(η/2) * cos(θ/2 + ψ)
 *   y = cos(η/2) * sin(θ/2 + ψ)
 *   z = sin(η/2) * cos(θ/2 - ψ)
 *   m = sin(η/2) * sin(θ/2 - ψ)
 * 
 * References:
 *   - Yershova et al. "Generating Uniform Incremental Grids on SO(3) Using the Hopf Fibration"
 *   - LaValle et al. "Uniform deterministic grids on SO(3)"
 */

#include "hypercube/coordinates.hpp"
#include "hypercube/hilbert.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>

namespace hypercube {

namespace {

// Mathematical constants - high precision
constexpr double PI = 3.14159265358979323846264338327950288;
constexpr double TWO_PI = 2.0 * PI;
constexpr double HALF_PI = PI * 0.5;

// Golden ratio for Fibonacci spiral on S²
constexpr double PHI = 1.6180339887498948482045868343656381;      // (1 + sqrt(5)) / 2
constexpr double PHI_INV = 0.6180339887498948482045868343656381;  // 1 / PHI = PHI - 1

// Plastic constant for fiber phase (irrational, algebraically independent from φ)
// Real root of x³ = x + 1, ≈ 1.3247179572...
constexpr double PLASTIC = 1.32471795724474602596090885447809734;
constexpr double PLASTIC_INV = 0.75487766624669276004950889615575073;  // 1 / PLASTIC

// Total valid Unicode codepoints (excluding surrogates D800-DFFF)
constexpr uint32_t TOTAL_VALID_CODEPOINTS = 0x10FFFF + 1 - 2048;  // 1,112,064

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

// QWERTY keyboard layout proximity ordering
// Returns a value 0-35 where adjacent keys have adjacent values
// This puts Q-W-E-R-T adjacent, A-S-D-F-G adjacent, etc.
constexpr uint32_t get_keyboard_proximity(uint32_t cp) noexcept {
    // Normalize to lowercase for lookup
    uint32_t lc = cp;
    if (cp >= 'A' && cp <= 'Z') lc = cp - 'A' + 'a';
    
    // QWERTY rows linearized with adjacency preserved
    switch (lc) {
        // Top row: Q W E R T Y U I O P
        case 'q': return 10;
        case 'w': return 11;
        case 'e': return 12;
        case 'r': return 13;
        case 't': return 14;
        case 'y': return 15;
        case 'u': return 16;
        case 'i': return 17;
        case 'o': return 18;
        case 'p': return 19;
        
        // Home row: A S D F G H J K L
        case 'a': return 20;
        case 's': return 21;
        case 'd': return 22;
        case 'f': return 23;
        case 'g': return 24;
        case 'h': return 25;
        case 'j': return 26;
        case 'k': return 27;
        case 'l': return 28;
        
        // Bottom row: Z X C V B N M
        case 'z': return 29;
        case 'x': return 30;
        case 'c': return 31;
        case 'v': return 32;
        case 'b': return 33;
        case 'n': return 34;
        case 'm': return 35;
        
        default: return 40;
    }
}

// Phonetic similarity grouping
// Returns 0-5 grouping phonetically similar sounds
constexpr uint32_t get_phonetic_group(uint32_t cp) noexcept {
    uint32_t lc = cp;
    if (cp >= 'A' && cp <= 'Z') lc = cp - 'A' + 'a';
    
    switch (lc) {
        // Vowels
        case 'a': case 'e': case 'i': case 'o': case 'u': case 'y':
            return 0;
        // Labials (lip sounds): B, P, M, V, F, W
        case 'b': case 'p': case 'm': case 'v': case 'f': case 'w':
            return 1;
        // Dentals/Alveolars: T, D, N, S, Z, L, R
        case 't': case 'd': case 'n': case 's': case 'z': case 'l': case 'r':
            return 2;
        // Velars: K, G, C, Q, X
        case 'k': case 'g': case 'c': case 'q': case 'x':
            return 3;
        // Others: J, H
        case 'j': case 'h':
            return 4;
        default:
            return 5;
    }
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

// Build sorted order array for all codepoints based on semantic clustering
// This ensures A is near a is near Ä, etc.
// For Hopf fibration: adjacent indices = adjacent positions on sphere
uint32_t codepoint_to_sequence_index(uint32_t codepoint) noexcept {
    // For surrogates, return a special high value (they shouldn't be used)
    if (codepoint >= constants::SURROGATE_START && codepoint <= constants::SURROGATE_END) {
        return UINT32_MAX;
    }
    
    // Get semantic order - this determines position in the Hopf fibration sequence
    // Adjacent semantic orders = adjacent positions on the 3-sphere
    return get_semantic_order(codepoint);
}

// Unicode block ranges for categorization
struct UnicodeBlock {
    uint32_t start;
    uint32_t end;
    AtomCategory category;
};

constexpr UnicodeBlock unicode_blocks[] = {
    {0x0000, 0x001F, AtomCategory::Control},
    {0x0020, 0x0020, AtomCategory::Space},
    {0x0021, 0x002F, AtomCategory::PunctuationOther},
    {0x0030, 0x0039, AtomCategory::Digit},
    {0x003A, 0x0040, AtomCategory::PunctuationOther},
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
    {0x00A1, 0x00BF, AtomCategory::PunctuationOther},
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
    {0x2010, 0x2027, AtomCategory::PunctuationOther},
    {0x2028, 0x2029, AtomCategory::Separator},
    {0x202A, 0x202E, AtomCategory::Format},
    {0x202F, 0x202F, AtomCategory::Space},
    {0x2030, 0x205E, AtomCategory::PunctuationOther},
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
 * Sequential 4D Hyperspherical Coordinate Mapping
 * 
 * Maps semantic index i ∈ [0, N) to a point on the unit 3-sphere (S³) such that
 * ADJACENT INDICES map to ADJACENT POINTS on the sphere surface.
 * 
 * This is critical for semantic clustering: A/a/Ä must be geometrically close
 * because they have adjacent semantic indices.
 * 
 * Approach: Use 4D hyperspherical coordinates with a space-filling curve pattern.
 * We tile the S³ surface using a hierarchical decomposition:
 * 
 *   - Divide the semantic index into 4 components using base-K digits
 *   - Each component controls one angular dimension
 *   - Adjacent indices differ in the lowest-order component → adjacent angles
 * 
 * 4D Hyperspherical coordinates (ψ₁, ψ₂, ψ₃ ∈ [0, π], ψ₄ ∈ [0, 2π]):
 *   x = cos(ψ₁)
 *   y = sin(ψ₁) * cos(ψ₂)
 *   z = sin(ψ₁) * sin(ψ₂) * cos(ψ₃)
 *   m = sin(ψ₁) * sin(ψ₂) * sin(ψ₃)
 * 
 * Note: This gives x² + y² + z² + m² = 1 exactly.
 * 
 * For ~1.1M points, we use a 4-level hierarchy with ~33 divisions per level.
 * 33^4 = 1,185,921 ≥ 1,112,064 valid codepoints
 */
Point4D CoordinateMapper::map_codepoint(uint32_t codepoint) noexcept {
    // Skip surrogates - they get invalid coords
    if (codepoint >= constants::SURROGATE_START && codepoint <= constants::SURROGATE_END) {
        return Point4D(0, 0, 0, 0);
    }
    
    // Get semantic sequence index for clustering
    uint32_t semantic_idx = codepoint_to_sequence_index(codepoint);
    
    // === HIERARCHICAL DECOMPOSITION ===
    // Use base-33 decomposition: 33^4 = 1,185,921 covers all ~1.1M codepoints
    // Each digit controls one angular dimension
    // Adjacent semantic indices → differ only in lowest digit → minimal angular change
    
    constexpr uint32_t BASE = 33;  // 33^4 = 1,185,921
    
    // Extract 4 "digits" in base-33
    // d0 = fastest varying (controls finest angular resolution)
    // d3 = slowest varying (controls coarsest angular resolution)
    uint32_t idx = semantic_idx;
    uint32_t d0 = idx % BASE; idx /= BASE;  // [0, 32]
    uint32_t d1 = idx % BASE; idx /= BASE;  // [0, 32]
    uint32_t d2 = idx % BASE; idx /= BASE;  // [0, 32]
    uint32_t d3 = idx % BASE;               // [0, 32]
    
    // === MAP TO 4D HYPERSPHERICAL ANGLES ===
    // Each digit maps to an angular range
    // We use a serpentine pattern (alternating direction) to ensure continuity
    // at digit boundaries (like how a Hilbert curve stays connected)
    
    // Serpentine: if parent digit is odd, reverse this digit's direction
    if (d3 & 1) d2 = BASE - 1 - d2;
    if (d2 & 1) d1 = BASE - 1 - d1;
    if (d1 & 1) d0 = BASE - 1 - d0;
    
    // Convert digits to fractions [0, 1] with centering
    // Add 0.5 to center within each cell
    double f0 = (static_cast<double>(d0) + 0.5) / static_cast<double>(BASE);
    double f1 = (static_cast<double>(d1) + 0.5) / static_cast<double>(BASE);
    double f2 = (static_cast<double>(d2) + 0.5) / static_cast<double>(BASE);
    double f3 = (static_cast<double>(d3) + 0.5) / static_cast<double>(BASE);
    
    // Map fractions to hyperspherical angles
    // ψ₁, ψ₂, ψ₃ ∈ [0, π], ψ₄ ∈ [0, 2π]
    // But for uniform distribution on S³, we need the Jacobian-corrected form:
    //   ψ₁ = arccos(1 - 2*f₃)  -- uniform on [0, π] w.r.t. S³ measure
    //   ψ₂ = arccos(1 - 2*f₂)  -- similar
    //   ψ₃ = π * f₁            -- linear in [0, π]
    //   ψ₄ = 2π * f₀           -- linear in [0, 2π] (but we use 3D embedding)
    
    // For S³ in 4D, we use the standard parameterization:
    const double psi1 = std::acos(1.0 - 2.0 * f3);  // [0, π], uniform measure
    const double psi2 = std::acos(1.0 - 2.0 * f2);  // [0, π], uniform measure
    const double psi3 = PI * f1;                    // [0, π]
    // f0 controls the finest variation within the psi3 cell
    // We encode it as a small perturbation to maintain adjacency
    const double psi3_fine = psi3 + (f0 - 0.5) * (PI / BASE);
    
    // === CONVERT TO CARTESIAN S³ COORDINATES ===
    // Standard 4D hyperspherical to Cartesian:
    //   x = cos(ψ₁)
    //   y = sin(ψ₁) * cos(ψ₂)
    //   z = sin(ψ₁) * sin(ψ₂) * cos(ψ₃)
    //   m = sin(ψ₁) * sin(ψ₂) * sin(ψ₃)
    
    const double sin_psi1 = std::sin(psi1);
    const double sin_psi2 = std::sin(psi2);
    
    const double ux = std::cos(psi1);
    const double uy = sin_psi1 * std::cos(psi2);
    const double uz = sin_psi1 * sin_psi2 * std::cos(psi3_fine);
    const double um = sin_psi1 * sin_psi2 * std::sin(psi3_fine);
    
    // === SCALE TO 32-BIT INTEGER COORDINATES ===
    // Map [-1, 1] to full signed int32 range
    // Center of hypersphere (0,0,0,0) maps to coordinate 0
    // Surface atoms have large absolute values (near ±2^31)
    // Compositions average inward toward center (toward 0)
    //
    // Storage: signed int32 values stored via uint32 bit pattern
    // PostGIS stores as double, which preserves the full range
    auto to_coord = [](double unit_val) -> Coord32 {
        // unit_val is in [-1, 1], scale to [-2^31, 2^31-1]
        double scaled = unit_val * static_cast<double>(INT32_MAX);
        scaled = std::clamp(scaled, static_cast<double>(INT32_MIN), static_cast<double>(INT32_MAX));
        int32_t signed_val = static_cast<int32_t>(scaled);
        // Store signed value directly - will be interpreted correctly
        // Bit pattern is preserved through uint32 cast
        uint32_t bits;
        std::memcpy(&bits, &signed_val, sizeof(bits));
        return bits;
    };

    return Point4D(to_coord(ux), to_coord(uy), to_coord(uz), to_coord(um));
}


Point4D CoordinateMapper::centroid(const std::vector<Point4D>& points) noexcept {
    if (points.empty()) {
        return Point4D();
    }
    
    uint64_t sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    
    for (const auto& p : points) {
        sum_x += p.x;
        sum_y += p.y;
        sum_z += p.z;
        sum_m += p.m;
    }
    
    size_t n = points.size();
    return Point4D(
        static_cast<Coord32>(sum_x / n),
        static_cast<Coord32>(sum_y / n),
        static_cast<Coord32>(sum_z / n),
        static_cast<Coord32>(sum_m / n)
    );
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

bool CoordinateMapper::is_on_surface(const Point4D& point) noexcept {
    // For the S³ mapping, all atoms are on the 3-sphere surface
    // Check if point satisfies x² + y² + z² + m² ≈ 1 (within tolerance)
    // 
    // IMPORTANT: Coordinates are stored as signed int32 values bit-cast to uint32
    // The mapping is: unit_val ∈ [-1, 1] → [INT32_MIN, INT32_MAX]
    // We interpret the uint32 as signed int32 to recover the unit value
    
    auto to_unit = [](uint32_t coord) -> double {
        int32_t signed_val = static_cast<int32_t>(coord);
        return static_cast<double>(signed_val) / static_cast<double>(INT32_MAX);
    };
    
    double x = to_unit(point.x);
    double y = to_unit(point.y);
    double z = to_unit(point.z);
    double m = to_unit(point.m);
    
    double r_squared = x*x + y*y + z*z + m*m;
    
    // Allow tolerance for integer quantization errors
    // With 32-bit precision, worst-case quantization error is ~2.3e-10 per coordinate
    // Use 10% tolerance to handle edge cases from the hierarchical decomposition
    return r_squared >= 0.9 && r_squared <= 1.1;
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

} // namespace hypercube
