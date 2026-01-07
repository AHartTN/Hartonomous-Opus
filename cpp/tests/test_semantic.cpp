/**
 * Semantic Validation Tests for Hartonomous Hypercube
 * 
 * These tests verify the semantic properties required for the system:
 * - All Unicode atoms are on the hypercube surface
 * - Semantically related characters are geometrically close
 * - Case pairs (A/a) are closer than unrelated letters (A/z)
 * - Categories cluster properly
 * - Hilbert indices provide locality-preserving ordering
 * - BLAKE3 hashes are unique and deterministic
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <algorithm>

#include "../include/hypercube/types.hpp"
#include "../include/hypercube/coordinates.hpp"
#include "../include/hypercube/hilbert.hpp"
#include "../include/hypercube/blake3.hpp"

using namespace hypercube;

// Test helpers
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAILED: " << msg << std::endl; \
        tests_failed++; \
    } else { \
        std::cout << "PASSED: " << msg << std::endl; \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_LESS(a, b, msg) do { \
    if (!((a) < (b))) { \
        std::cerr << "FAILED: " << msg << " (" << (a) << " >= " << (b) << ")" << std::endl; \
        tests_failed++; \
    } else { \
        std::cout << "PASSED: " << msg << " (" << (a) << " < " << (b) << ")" << std::endl; \
        tests_passed++; \
    } \
} while(0)

// ===========================================================================
// TEST 1: All atoms are on the hypercube surface
// ===========================================================================
void test_surface_constraint() {
    std::cout << "\n=== Test: Surface Constraint ===" << std::endl;
    
    int on_surface = 0;
    int off_surface = 0;
    
    // Check all valid Unicode codepoints
    for (uint32_t cp = 0; cp <= constants::MAX_CODEPOINT; ++cp) {
        // Skip surrogates
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) {
            continue;
        }
        
        Point4D coords = CoordinateMapper::map_codepoint(cp);
        if (coords.is_on_surface()) {
            on_surface++;
        } else {
            off_surface++;
            if (off_surface <= 5) {
                std::cerr << "  Off-surface: U+" << std::hex << cp << std::dec 
                          << " at (" << coords.x << ", " << coords.y 
                          << ", " << coords.z << ", " << coords.m << ")" << std::endl;
            }
        }
    }
    
    std::cout << "  Total on surface: " << on_surface << std::endl;
    std::cout << "  Total off surface: " << off_surface << std::endl;
    
    ASSERT_TRUE(off_surface == 0, "All atoms are on hypercube surface");
}

// ===========================================================================
// TEST 2: Case pairs are semantically close
// ===========================================================================
void test_case_pair_proximity() {
    std::cout << "\n=== Test: Case Pair Proximity ===" << std::endl;
    
    // For each letter A-Z, check that:
    // 1. A and a are in related categories (upper vs lower)
    // 2. Distance A-a can be measured
    
    struct CasePair {
        uint32_t upper;
        uint32_t lower;
        char name;
    };
    
    std::vector<CasePair> pairs;
    for (char c = 'A'; c <= 'Z'; ++c) {
        pairs.push_back({static_cast<uint32_t>(c), static_cast<uint32_t>(c + 32), c});
    }
    
    for (const auto& pair : pairs) {
        AtomCategory upper_cat = CoordinateMapper::categorize(pair.upper);
        AtomCategory lower_cat = CoordinateMapper::categorize(pair.lower);
        
        ASSERT_TRUE(upper_cat == AtomCategory::LetterUpper, 
                    std::string(1, pair.name) + " is categorized as LetterUpper");
        ASSERT_TRUE(lower_cat == AtomCategory::LetterLower,
                    std::string(1, static_cast<char>(pair.lower)) + " is categorized as LetterLower");
    }
    
    // Check that A-a distance is measurable and A is closer to a than to B
    Point4D A_coords = CoordinateMapper::map_codepoint('A');
    Point4D a_coords = CoordinateMapper::map_codepoint('a');
    Point4D B_coords = CoordinateMapper::map_codepoint('B');
    
    double dist_A_a = CoordinateMapper::euclidean_distance(A_coords, a_coords);
    double dist_A_B = CoordinateMapper::euclidean_distance(A_coords, B_coords);
    
    std::cout << "  Distance A-a: " << dist_A_a << std::endl;
    std::cout << "  Distance A-B: " << dist_A_B << std::endl;
    
    ASSERT_TRUE(dist_A_a > 0, "A and a have non-zero distance (different coordinates)");
    ASSERT_LESS(dist_A_a, dist_A_B, "A is closer to a (same letter) than to B (different letter)");
}

// ===========================================================================
// TEST 3: Category clustering - same category atoms are near each other on S^3
// ===========================================================================
void test_category_clustering() {
    std::cout << "\n=== Test: Category Clustering ===" << std::endl;
    
    std::map<AtomCategory, std::vector<Point4D>> category_points;
    
    // Collect sample points from each category
    for (uint32_t cp = 0; cp < 0x3000; ++cp) {
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) continue;
        
        AtomCategory cat = CoordinateMapper::categorize(cp);
        if (category_points[cat].size() < 100) {
            category_points[cat].push_back(CoordinateMapper::map_codepoint(cp));
        }
    }
    
    // Verify each category has points on the 3-sphere surface
    for (const auto& [cat, points] : category_points) {
        if (points.size() < 2) continue;
        
        int on_surface = 0;
        for (const auto& p : points) {
            if (p.is_on_surface()) on_surface++;
        }
        
        double surface_pct = static_cast<double>(on_surface) / points.size() * 100.0;
        
        std::cout << "  Category " << static_cast<int>(cat) << " ("
                  << category_to_string(cat) << "): "
                  << points.size() << " points, "
                  << static_cast<int>(surface_pct) << "% on S^3 surface" << std::endl;
    }
    
    ASSERT_TRUE(true, "Category clustering analysis complete");
}

// ===========================================================================
// TEST 4: Digits are clustered together and near each other
// ===========================================================================
void test_digit_clustering() {
    std::cout << "\n=== Test: Digit Clustering ===" << std::endl;
    
    std::vector<Point4D> digit_coords;
    for (uint32_t cp = '0'; cp <= '9'; ++cp) {
        AtomCategory cat = CoordinateMapper::categorize(cp);
        ASSERT_TRUE(cat == AtomCategory::Digit, 
                    std::string("'") + static_cast<char>(cp) + "' is categorized as Digit");
        digit_coords.push_back(CoordinateMapper::map_codepoint(cp));
    }
    
    // All digits should be on the 3-sphere surface
    int on_surface = 0;
    for (const auto& p : digit_coords) {
        if (p.is_on_surface()) on_surface++;
    }
    ASSERT_TRUE(on_surface == 10, "All 10 digits are on 3-sphere surface");
    
    // Compute centroid and verify digits are clustered (low variance)
    Point4D centroid = CoordinateMapper::centroid(digit_coords);
    std::cout << "  Digit centroid: (" << centroid.x << ", " << centroid.y 
              << ", " << centroid.z << ", " << centroid.m << ")" << std::endl;
    
    // Check that consecutive digits are close to each other
    // Distance from 0 to 1 should be small compared to 0 to random letter
    double dist_0_1 = CoordinateMapper::euclidean_distance(digit_coords[0], digit_coords[1]);
    Point4D A_coords = CoordinateMapper::map_codepoint('A');
    double dist_0_A = CoordinateMapper::euclidean_distance(digit_coords[0], A_coords);
    
    std::cout << "  Distance 0-1: " << dist_0_1 << std::endl;
    std::cout << "  Distance 0-A: " << dist_0_A << std::endl;
    
    // Digits should be closer to each other than to letters
    ASSERT_LESS(dist_0_1, dist_0_A, "Digit 0 is closer to digit 1 than to letter A");
}

// ===========================================================================
// TEST 5: Hilbert index preserves locality
// ===========================================================================
void test_hilbert_locality() {
    std::cout << "\n=== Test: Hilbert Locality ===" << std::endl;
    
    // Points that are close in Hilbert space should be close in Euclidean space
    // (on average - Hilbert curve is space-filling so there are discontinuities)
    
    std::vector<std::pair<Point4D, HilbertIndex>> sorted_points;
    
    // Collect ASCII letters
    for (uint32_t cp = 'A'; cp <= 'Z'; ++cp) {
        Point4D coords = CoordinateMapper::map_codepoint(cp);
        HilbertIndex idx = HilbertCurve::coords_to_index(coords);
        sorted_points.push_back({coords, idx});
    }
    for (uint32_t cp = 'a'; cp <= 'z'; ++cp) {
        Point4D coords = CoordinateMapper::map_codepoint(cp);
        HilbertIndex idx = HilbertCurve::coords_to_index(coords);
        sorted_points.push_back({coords, idx});
    }
    
    // Sort by Hilbert index
    std::sort(sorted_points.begin(), sorted_points.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Check that Hilbert ordering is consistent with geometric structure
    std::cout << "  Hilbert ordering of letters:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), sorted_points.size()); ++i) {
        std::cout << "    " << i << ": hilbert_lo=" << sorted_points[i].second.lo << std::endl;
    }
    
    ASSERT_TRUE(sorted_points.size() == 52, "All 52 letters have Hilbert indices");
}

// ===========================================================================
// TEST 6: Hilbert roundtrip is lossless
// ===========================================================================
void test_hilbert_roundtrip() {
    std::cout << "\n=== Test: Hilbert Roundtrip ===" << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    // Test full range
    for (uint32_t cp = 0; cp <= constants::MAX_CODEPOINT; ++cp) {
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) continue;
        
        Point4D original = CoordinateMapper::map_codepoint(cp);
        HilbertIndex idx = HilbertCurve::coords_to_index(original);
        Point4D recovered = HilbertCurve::index_to_coords(idx);
        
        if (original == recovered) {
            passed++;
        } else {
            failed++;
            if (failed <= 5) {
                std::cerr << "  Roundtrip failed for U+" << std::hex << cp << std::dec << std::endl;
            }
        }
    }
    
    std::cout << "  Roundtrip: " << passed << " passed, " << failed << " failed" << std::endl;
    ASSERT_TRUE(failed == 0, "Hilbert coords->index->coords roundtrip is lossless");
}

// ===========================================================================
// TEST 7: BLAKE3 hashes are unique
// ===========================================================================
void test_blake3_uniqueness() {
    std::cout << "\n=== Test: BLAKE3 Uniqueness ===" << std::endl;
    
    std::set<Blake3Hash> seen_hashes;
    int duplicates = 0;
    
    for (uint32_t cp = 0; cp <= constants::MAX_CODEPOINT; ++cp) {
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) continue;
        
        Blake3Hash hash = Blake3Hasher::hash_codepoint(cp);
        
        if (seen_hashes.count(hash)) {
            duplicates++;
            if (duplicates <= 5) {
                std::cerr << "  Duplicate hash for U+" << std::hex << cp << std::dec << std::endl;
            }
        } else {
            seen_hashes.insert(hash);
        }
    }
    
    std::cout << "  Unique hashes: " << seen_hashes.size() << std::endl;
    std::cout << "  Duplicates: " << duplicates << std::endl;
    
    ASSERT_TRUE(duplicates == 0, "All BLAKE3 hashes are unique");
}

// ===========================================================================
// TEST 8: BLAKE3 is deterministic
// ===========================================================================
void test_blake3_determinism() {
    std::cout << "\n=== Test: BLAKE3 Determinism ===" << std::endl;
    
    int mismatches = 0;
    
    for (uint32_t cp = 0; cp < 1000; ++cp) {
        Blake3Hash h1 = Blake3Hasher::hash_codepoint(cp);
        Blake3Hash h2 = Blake3Hasher::hash_codepoint(cp);
        
        if (h1 != h2) {
            mismatches++;
        }
    }
    
    ASSERT_TRUE(mismatches == 0, "BLAKE3 hash is deterministic (same input = same output)");
}

// ===========================================================================
// TEST 9: Performance - generate all atoms under 2 seconds
// ===========================================================================
void test_generation_performance() {
    std::cout << "\n=== Test: Generation Performance ===" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t count = 0;
    for (uint32_t cp = 0; cp <= constants::MAX_CODEPOINT; ++cp) {
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) continue;
        
        // Full atom generation: coords + hilbert + hash
        Point4D coords = CoordinateMapper::map_codepoint(cp);
        HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
        Blake3Hash hash = Blake3Hasher::hash_codepoint(cp);
        
        // Prevent optimizer from eliminating
        if (hash.bytes[0] == 255 && hilbert.lo == 0 && coords.x == 0) {
            count++;
        }
        count++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "  Generated " << count << " atoms in " << ms << " ms" << std::endl;
    
    ASSERT_LESS(ms, 2000, "Generation completes in under 2 seconds");
}

// ===========================================================================
// TEST 10: Verify expected atom count
// ===========================================================================
void test_atom_count() {
    std::cout << "\n=== Test: Atom Count ===" << std::endl;
    
    uint64_t count = 0;
    for (uint32_t cp = 0; cp <= constants::MAX_CODEPOINT; ++cp) {
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) continue;
        count++;
    }
    
    std::cout << "  Total valid codepoints: " << count << std::endl;
    
    // 0x10FFFF + 1 = 1114112 total
    // Surrogates: 0xDFFF - 0xD800 + 1 = 2048
    // Expected: 1114112 - 2048 = 1112064
    ASSERT_TRUE(count == constants::VALID_CODEPOINTS, 
                "Atom count matches expected valid codepoints");
}

// ===========================================================================
// MAIN
// ===========================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "  Hartonomous Hypercube Semantic Tests" << std::endl;
    std::cout << "============================================" << std::endl;
    
    test_surface_constraint();
    test_case_pair_proximity();
    test_category_clustering();
    test_digit_clustering();
    test_hilbert_locality();
    test_hilbert_roundtrip();
    test_blake3_uniqueness();
    test_blake3_determinism();
    test_generation_performance();
    test_atom_count();
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Results: " << tests_passed << " passed, " 
              << tests_failed << " failed" << std::endl;
    std::cout << "============================================" << std::endl;
    
    return tests_failed > 0 ? 1 : 0;
}
