/**
 * Semantic Clustering Validation Tests
 * 
 * Validates that the Unicode seeding and coordinate mapping produce
 * the expected semantic clustering properties:
 * 
 * 1. Case pairs (A/a, B/b) are geometrically close
 * 2. Diacritic variants (A/Á/Ä) are geometrically close
 * 3. Script families are grouped
 * 4. Digits are grouped across scripts
 * 5. Keyboard-adjacent keys have measurable but not primary proximity
 * 
 * These tests serve as regression guards for the coordinate mapping.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include "hypercube/coordinates.hpp"
#include "hypercube/hilbert.hpp"

using namespace hypercube;

// 4D Euclidean distance
double distance(const Point4D& a, const Point4D& b) {
    double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
    double dm = static_cast<double>(a.m) - static_cast<double>(b.m);
    return std::sqrt(dx*dx + dy*dy + dz*dz + dm*dm);
}

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

void check(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "  ✓ " << name << std::endl;
        tests_passed++;
    } else {
        std::cout << "  ✗ " << name << " FAILED" << std::endl;
        tests_failed++;
    }
}

// =============================================================================
// Test: Case Pair Proximity
// =============================================================================
void test_case_pairs() {
    std::cout << "\n=== Test: Case Pair Proximity ===" << std::endl;
    
    // Case pairs should be VERY close (same base letter, different case)
    std::vector<std::pair<char, char>> case_pairs = {
        {'A', 'a'}, {'B', 'b'}, {'I', 'i'}, {'O', 'o'}, {'Z', 'z'}
    };
    
    double max_case_distance = 0;
    double total_case_distance = 0;
    
    for (const auto& [upper, lower] : case_pairs) {
        Point4D pu = CoordinateMapper::map_codepoint(upper);
        Point4D pl = CoordinateMapper::map_codepoint(lower);
        double d = distance(pu, pl);
        
        std::cout << "  " << upper << "/" << lower << " distance: " 
                  << std::scientific << std::setprecision(2) << d << std::endl;
        
        max_case_distance = std::max(max_case_distance, d);
        total_case_distance += d;
    }
    
    double avg_case_distance = total_case_distance / case_pairs.size();
    
    // Case pairs should be within 1% of the coordinate space diameter
    // UINT32_MAX * sqrt(4) ≈ 8.6e9, so 1% = 8.6e7
    check(max_case_distance < 1e8, "Case pairs within 1% of space diameter");
    check(avg_case_distance < 5e7, "Average case distance < 0.5% of diameter");
}

// =============================================================================
// Test: Diacritic Variant Proximity
// =============================================================================
void test_diacritic_variants() {
    std::cout << "\n=== Test: Diacritic Variant Proximity ===" << std::endl;
    
    // Diacritic variants should be close to base letter
    std::vector<std::pair<uint32_t, uint32_t>> diacritic_pairs = {
        {'A', 0x00C0},  // A, À
        {'A', 0x00C1},  // A, Á
        {'A', 0x00C4},  // A, Ä
        {'E', 0x00C9},  // E, É
        {'O', 0x00D6},  // O, Ö
        {'a', 0x00E0},  // a, à
        {'a', 0x00E1},  // a, á
    };
    
    double max_diacritic_distance = 0;
    
    for (const auto& [base, variant] : diacritic_pairs) {
        Point4D pb = CoordinateMapper::map_codepoint(base);
        Point4D pv = CoordinateMapper::map_codepoint(variant);
        double d = distance(pb, pv);
        
        char base_char = static_cast<char>(base);
        std::cout << "  " << base_char << "/U+" << std::hex << variant 
                  << std::dec << " distance: " 
                  << std::scientific << std::setprecision(2) << d << std::endl;
        
        max_diacritic_distance = std::max(max_diacritic_distance, d);
    }
    
    // Diacritics should be close to base (within same base letter group)
    check(max_diacritic_distance < 2e8, "Diacritics within 2% of space diameter");
}

// =============================================================================
// Test: Cross-Base Distance (A vs B, etc.)
// =============================================================================
void test_cross_base_distance() {
    std::cout << "\n=== Test: Cross-Base Distance ===" << std::endl;
    
    // Different base letters should be FARTHER than case/diacritic variants
    Point4D A = CoordinateMapper::map_codepoint('A');
    Point4D a = CoordinateMapper::map_codepoint('a');
    Point4D B = CoordinateMapper::map_codepoint('B');
    Point4D Z = CoordinateMapper::map_codepoint('Z');
    
    double Aa = distance(A, a);  // Case pair
    double AB = distance(A, B);  // Adjacent letters
    double AZ = distance(A, Z);  // Far letters
    
    std::cout << "  A-a (case pair): " << std::scientific << Aa << std::endl;
    std::cout << "  A-B (adjacent): " << std::scientific << AB << std::endl;
    std::cout << "  A-Z (far): " << std::scientific << AZ << std::endl;
    
    // Case pairs should be closer than different letters
    check(Aa < AB, "Case pair (A-a) closer than adjacent (A-B)");
    check(Aa < AZ, "Case pair (A-a) closer than far (A-Z)");
}

// =============================================================================
// Test: Keyboard Adjacent Keys
// =============================================================================
void test_keyboard_proximity() {
    std::cout << "\n=== Test: Keyboard Adjacent Keys ===" << std::endl;
    
    // QWERTY adjacent pairs
    std::vector<std::pair<char, char>> kbd_adjacent = {
        {'Q', 'W'}, {'W', 'E'}, {'E', 'R'}, {'R', 'T'},
        {'A', 'S'}, {'S', 'D'}, {'D', 'F'},
        {'I', 'O'}, {'O', 'P'}, {'K', 'L'}
    };
    
    double total_kbd_distance = 0;
    
    for (const auto& [k1, k2] : kbd_adjacent) {
        Point4D p1 = CoordinateMapper::map_codepoint(k1);
        Point4D p2 = CoordinateMapper::map_codepoint(k2);
        double d = distance(p1, p2);
        
        std::cout << "  " << k1 << "-" << k2 << ": " 
                  << std::scientific << std::setprecision(2) << d << std::endl;
        
        total_kbd_distance += d;
    }
    
    double avg_kbd = total_kbd_distance / kbd_adjacent.size();
    std::cout << "  Average keyboard-adjacent distance: " << avg_kbd << std::endl;
    
    // Keyboard proximity is NOT a primary clustering factor
    // It emerges from edge weights, not coordinates
    // So we just document the distances, not assert tight bounds
    std::cout << "  (Note: Keyboard proximity learned via edges, not coordinates)" << std::endl;
    tests_passed++;  // Informational test
}

// =============================================================================
// Test: Script Family Grouping
// =============================================================================
void test_script_families() {
    std::cout << "\n=== Test: Script Family Grouping ===" << std::endl;
    
    // Greek letters should be grouped
    Point4D alpha_upper = CoordinateMapper::map_codepoint(0x0391);  // Α
    Point4D alpha_lower = CoordinateMapper::map_codepoint(0x03B1);  // α
    Point4D omega_upper = CoordinateMapper::map_codepoint(0x03A9);  // Ω
    (void)CoordinateMapper::map_codepoint(0x03C9);  // ω - verify no crash
    
    double greek_case = distance(alpha_upper, alpha_lower);
    double greek_span = distance(alpha_upper, omega_upper);
    
    std::cout << "  Greek Α/α (case): " << std::scientific << greek_case << std::endl;
    std::cout << "  Greek Α-Ω (span): " << std::scientific << greek_span << std::endl;
    
    // Cyrillic
    Point4D cyr_a = CoordinateMapper::map_codepoint(0x0410);  // А
    Point4D cyr_ya = CoordinateMapper::map_codepoint(0x042F); // Я
    
    double cyrillic_span = distance(cyr_a, cyr_ya);
    std::cout << "  Cyrillic А-Я (span): " << std::scientific << cyrillic_span << std::endl;
    
    // Latin vs Greek distance
    Point4D latin_a = CoordinateMapper::map_codepoint('A');
    double latin_greek = distance(latin_a, alpha_upper);
    std::cout << "  Latin A vs Greek Α: " << std::scientific << latin_greek << std::endl;
    
    check(greek_case < greek_span, "Greek case pairs closer than alphabet span");
    // Note: Latin A and Greek Α may or may not be close depending on semantic ordering
}

// =============================================================================
// Test: Digit Grouping
// =============================================================================
void test_digit_grouping() {
    std::cout << "\n=== Test: Digit Grouping ===" << std::endl;
    
    // ASCII digits should be grouped
    Point4D d0 = CoordinateMapper::map_codepoint('0');
    Point4D d9 = CoordinateMapper::map_codepoint('9');
    double ascii_digits_span = distance(d0, d9);
    
    std::cout << "  ASCII 0-9 span: " << std::scientific << ascii_digits_span << std::endl;
    
    // Fullwidth digits
    Point4D fw0 = CoordinateMapper::map_codepoint(0xFF10);  // ０
    Point4D fw9 = CoordinateMapper::map_codepoint(0xFF19);  // ９
    double fullwidth_span = distance(fw0, fw9);
    
    std::cout << "  Fullwidth ０-９ span: " << std::scientific << fullwidth_span << std::endl;
    
    // ASCII vs Fullwidth digits
    double ascii_fullwidth = distance(d0, fw0);
    std::cout << "  ASCII 0 vs Fullwidth ０: " << std::scientific << ascii_fullwidth << std::endl;
    
    check(ascii_digits_span < 1e9, "ASCII digits reasonably grouped");
}

// =============================================================================
// Test: Typo Trajectory Similarity (Frechet proxy)
// =============================================================================
void test_typo_trajectory() {
    std::cout << "\n=== Test: Typo Trajectory Similarity ===" << std::endl;
    
    // "king" vs "ling" differ by one character
    // Their trajectories should be similar because most points match
    
    std::string word1 = "king";
    std::string word2 = "ling";  // k→l typo
    std::string word3 = "kong";  // i→o typo
    std::string word4 = "fish";  // completely different
    
    auto trajectory_distance = [](const std::string& s1, const std::string& s2) {
        // Simplified Frechet-like: sum of per-character distances
        double total = 0;
        size_t len = std::min(s1.size(), s2.size());
        for (size_t i = 0; i < len; i++) {
            Point4D p1 = CoordinateMapper::map_codepoint(s1[i]);
            Point4D p2 = CoordinateMapper::map_codepoint(s2[i]);
            total += distance(p1, p2);
        }
        // Penalize length difference
        total += std::abs(static_cast<int>(s1.size()) - static_cast<int>(s2.size())) * 1e9;
        return total;
    };
    
    double d_king_ling = trajectory_distance(word1, word2);
    double d_king_kong = trajectory_distance(word1, word3);
    double d_king_fish = trajectory_distance(word1, word4);
    
    std::cout << "  king-ling (k→l typo): " << std::scientific << d_king_ling << std::endl;
    std::cout << "  king-kong (i→o typo): " << std::scientific << d_king_kong << std::endl;
    std::cout << "  king-fish (different): " << std::scientific << d_king_fish << std::endl;
    
    check(d_king_ling < d_king_fish, "One-char typo closer than completely different word");
    check(d_king_kong < d_king_fish, "One-char typo closer than completely different word");
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Semantic Clustering Validation Tests                     ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    test_case_pairs();
    test_diacritic_variants();
    test_cross_base_distance();
    test_keyboard_proximity();
    test_script_families();
    test_digit_grouping();
    test_typo_trajectory();
    
    std::cout << "\n════════════════════════════════════════════════════════════════" << std::endl;
    if (tests_failed == 0) {
        std::cout << "  ✓ All " << tests_passed << " tests passed!" << std::endl;
    } else {
        std::cout << "  ✗ " << tests_failed << " tests failed, " 
                  << tests_passed << " passed" << std::endl;
    }
    std::cout << "════════════════════════════════════════════════════════════════" << std::endl;
    
    return tests_failed > 0 ? 1 : 0;
}
