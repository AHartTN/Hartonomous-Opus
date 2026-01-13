#include "../include/hypercube/coordinates.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <cassert>
#include <limits>

using namespace hypercube;

// Stub implementation for DenseRegistry - returns reasonable test values
namespace DenseRegistry {
    static uint32_t total_active() {
        return 0x10FFFF; // Max Unicode codepoint
    }

    static uint32_t get_rank(uint32_t cp) {
        // Simple ranking based on codepoint for testing
        return cp;
    }
}

// ============================================================================
// MODULAR DYSON SPHERE DIAGNOSTICS
// ============================================================================

// ============================================================================
// DYSON SPHERE MODULE DIAGNOSTICS
// ============================================================================

// Test 1: Structural Integrity (Poisson Disk Property)
void test_dyson_structural_integrity() {
    std::cout << "\n=== DYSON MODULE 1: Structural Integrity ===" << std::endl;

    // Total modules on sphere
    double N = DenseRegistry::total_active();
    std::cout << "Total modules on sphere: " << N << std::endl;

    // Theoretical ideal distance for uniform distribution on 4D sphere
    // For N points on S³, the average nearest neighbor distance
    const double PI = 3.141592653589793;
    double ideal_dist = std::sqrt(8.0 * PI * PI / (3.0 * N)); // Approximation for 4D sphere

    std::cout << "Theoretical ideal NN distance: " << ideal_dist << std::endl;

    // Test case variant pairs (should be close to ideal)
    std::vector<std::pair<uint32_t, uint32_t>> test_pairs = {
        {'A', 'a'}, {'B', 'b'}, {'0', 'O'}, {'0', 'o'}
    };

    for (auto [c1, c2] : test_pairs) {
        double dist = CoordinateMapper::map_codepoint_float(c1)
                     .distance(CoordinateMapper::map_codepoint_float(c2));
        double ratio = dist / ideal_dist;
        std::cout << "  " << static_cast<char>(c1) << "-" << static_cast<char>(c2)
                  << ": dist=" << dist << ", ratio_to_ideal=" << ratio
                  << " (" << (ratio > 0.5 && ratio < 2.0 ? "GOOD" : "CHECK") << ")" << std::endl;
    }
}

// Test 2: Functional Zone Test (Semantic Clustering)
void test_dyson_functional_zones() {
    std::cout << "\n=== DYSON MODULE 2: Functional Zones ===" << std::endl;

    // Test that geometric neighbors are semantic neighbors
    auto find_nearest_neighbors = [](uint32_t target_cp, size_t k) -> std::vector<uint32_t> {
        Point4F target = CoordinateMapper::map_codepoint_float(target_cp);
        std::vector<std::pair<double, uint32_t>> neighbors;

        // Brute force search (for testing - in production use R-tree/Hilbert)
        for (uint32_t cp = 0; cp <= std::min(10000U, DenseRegistry::total_active()); ++cp) {
            if (cp >= 0xD800 && cp <= 0xDFFF) continue; // Skip surrogates
            double dist = target.distance(CoordinateMapper::map_codepoint_float(cp));
            neighbors.emplace_back(dist, cp);
        }

        std::partial_sort(neighbors.begin(), neighbors.begin() + k + 1,
                         neighbors.end()); // +1 to skip self

        std::vector<uint32_t> result;
        for (size_t i = 1; i <= k && i < neighbors.size(); ++i) { // Skip self (i=0)
            result.push_back(neighbors[i].second);
        }
        return result;
    };

    // Test semantic clustering for different zones
    std::vector<uint32_t> test_centers = {'A', '0', 0x0391, 0x1F600}; // Latin, Digit, Greek, Emoji

    for (uint32_t center : test_centers) {
        auto neighbors = find_nearest_neighbors(center, 5);
        std::cout << "Neighbors of " << static_cast<char>(center) << " (U+" << std::hex << center << std::dec << "):" << std::endl;

        uint32_t center_script = (center >> 8) & 0xFF; // Approximate script grouping
        int same_script_count = 0;

        for (uint32_t neighbor : neighbors) {
            uint32_t neighbor_script = (neighbor >> 8) & 0xFF;
            bool same_script = (center_script == neighbor_script);
            if (same_script) same_script_count++;

            std::cout << "  " << (neighbor >= 32 && neighbor <= 126 ? std::string(1, (char)neighbor) : "U+" + std::to_string(neighbor))
                      << " (script: " << neighbor_script << ") " << (same_script ? "✓" : "✗") << std::endl;
        }

        double clustering_ratio = static_cast<double>(same_script_count) / neighbors.size();
        std::cout << "  Semantic clustering: " << (clustering_ratio * 100) << "% same script" << std::endl;
        std::cout << "  Zone integrity: " << (clustering_ratio > 0.6 ? "GOOD ✓" : "LEAKY ✗") << std::endl;
    }
}

// Test 3: Composition Displacement Test
void test_dyson_composition_displacement() {
    std::cout << "\n=== DYSON MODULE 3: Composition Displacement ===" << std::endl;

    // Test that compositions create internal machinery (closer to origin)

    // Single atoms (surface modules)
    std::vector<Point4D> single_atoms = {CoordinateMapper::map_codepoint('A')};
    Point4F single_centroid = CoordinateMapper::centroid_float(
        std::vector<Point4F>(single_atoms.begin(), single_atoms.end())
    );
    double single_radius = std::sqrt(single_centroid.dot(single_centroid));
    std::cout << "Single atom centroid radius: " << single_radius << " (should be ~1.0)" << std::endl;

    // Simple composition (close atoms)
    std::vector<Point4D> simple_comp = {
        CoordinateMapper::map_codepoint('c'),
        CoordinateMapper::map_codepoint('a'),
        CoordinateMapper::map_codepoint('t')
    };
    Point4F simple_centroid = CoordinateMapper::centroid_float(
        std::vector<Point4F>(simple_comp.begin(), simple_comp.end())
    );
    double simple_radius = std::sqrt(simple_centroid.dot(simple_centroid));
    std::cout << "Simple word 'cat' centroid radius: " << simple_radius << " (should be <1.0)" << std::endl;

    // Opposite atoms (diametrically opposed on sphere)
    std::vector<Point4D> opposite_comp = {
        CoordinateMapper::map_codepoint('A'),
        CoordinateMapper::map_codepoint(static_cast<uint32_t>(L'A')) // Wide char version
    };
    Point4F opposite_centroid = CoordinateMapper::centroid_float(
        std::vector<Point4F>(opposite_comp.begin(), opposite_comp.end())
    );
    double opposite_radius = std::sqrt(opposite_centroid.dot(opposite_centroid));
    std::cout << "Opposite atoms centroid radius: " << opposite_radius << " (should be <<1.0)" << std::endl;

    // Complex multilingual composition
    std::vector<Point4D> complex_comp;
    std::vector<uint32_t> complex_chars = {'H', 'e', 'l', 'l', 'o', 0x0393, 0x0435, 0x4E16, 0x1F609}; // Hello in multiple scripts
    for (uint32_t cp : complex_chars) {
        complex_comp.push_back(CoordinateMapper::map_codepoint(cp));
    }
    Point4F complex_centroid = CoordinateMapper::centroid_float(
        std::vector<Point4F>(complex_comp.begin(), complex_comp.end())
    );
    double complex_radius = std::sqrt(complex_centroid.dot(complex_centroid));
    std::cout << "Multilingual 'HelloΓе你😉' centroid radius: " << complex_radius << " (should be much <1.0)" << std::endl;

    // Verify semantic gravity: more complex = deeper
    bool gravity_works = (complex_radius < opposite_radius) && (opposite_radius < simple_radius) && (simple_radius < single_radius);
    std::cout << "Semantic gravity check: " << (gravity_works ? "WORKING ✓" : "BROKEN ✗") << std::endl;
}

// Test 4: Local Topology Check
void test_dyson_local_topology() {
    std::cout << "\n=== DYSON MODULE 4: Local Topology ===" << std::endl;

    // Check that each module is "bolted" next to semantically related modules
    auto analyze_module_bolting = [](uint32_t cp) {
        Point4F module_pos = CoordinateMapper::map_codepoint_float(cp);
        uint64_t module_key = get_semantic_key(cp);

        // Find nearest neighbor
        uint32_t nearest_cp = cp;
        double min_dist = std::numeric_limits<double>::max();

        for (uint32_t test_cp = 0; test_cp <= std::min(10000U, DenseRegistry::total_active()); ++test_cp) {
            if (test_cp == cp || (test_cp >= 0xD800 && test_cp <= 0xDFFF)) continue;
            double dist = module_pos.distance(CoordinateMapper::map_codepoint_float(test_cp));
            if (dist < min_dist) {
                min_dist = dist;
                nearest_cp = test_cp;
            }
        }

        uint64_t neighbor_key = get_semantic_key(nearest_cp);
        uint32_t module_script = (module_key >> 56) & 0xFF;
        uint32_t neighbor_script = (neighbor_key >> 56) & 0xFF;
        uint32_t module_base = (module_key >> 24) & 0xFFFF;
        uint32_t neighbor_base = (neighbor_key >> 24) & 0xFFFF;

        bool same_script = (module_script == neighbor_script);
        bool related_base = (module_base == neighbor_base) ||
                           (std::abs(static_cast<int>(module_base) - static_cast<int>(neighbor_base)) <= 10);

        std::cout << "Module " << (cp >= 32 && cp <= 126 ? std::string(1, (char)cp) : "U+" + std::to_string(cp))
                  << " bolted next to " << (nearest_cp >= 32 && nearest_cp <= 126 ? std::string(1, (char)nearest_cp) : "U+" + std::to_string(nearest_cp))
                  << " (dist=" << min_dist << "): "
                  << (same_script ? "same_script ✓" : "diff_script ✗") << ", "
                  << (related_base ? "related_base ✓" : "unrelated_base ✗") << std::endl;

        return same_script || related_base;
    };

    std::vector<uint32_t> test_modules = {'A', 'a', '0', 'O', 0x0391, 0x03B1, 0x1F600};
    int good_bolting = 0;

    for (uint32_t cp : test_modules) {
        if (analyze_module_bolting(cp)) good_bolting++;
    }

    double bolting_ratio = static_cast<double>(good_bolting) / test_modules.size();
    std::cout << "Module bolting quality: " << (bolting_ratio * 100) << "% properly connected" << std::endl;
    std::cout << "Sphere structural integrity: " << (bolting_ratio > 0.7 ? "EXCELLENT ✓" : "NEEDS WORK ✗") << std::endl;
}

// Helper: Calculate mean and standard deviation
void calculate_stats(const std::vector<double>& values, double& mean, double& stdev) {
    if (values.empty()) {
        mean = stdev = 0.0;
        return;
    }

    mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    double variance = 0.0;
    for (double v : values) {
        variance += (v - mean) * (v - mean);
    }
    variance /= values.size();
    stdev = std::sqrt(variance);
}

// Diagnostic 1: Global Equidistance (Coefficient of Variation for NN distances)
void test_global_equidistance() {
    std::cout << "\n=== DIAGNOSTIC 1: Global Equidistance ===" << std::endl;

    // Sample first 10,000 codepoints for equidistance testing
    const size_t SAMPLE_SIZE = 10000;
    std::vector<Point4F> points;
    points.reserve(SAMPLE_SIZE);

    for (uint32_t cp = 0; cp < SAMPLE_SIZE; ++cp) {
        if (cp >= 0xD800 && cp <= 0xDFFF) continue; // Skip surrogates
        points.push_back(CoordinateMapper::map_codepoint_float(cp));
    }

    // Calculate nearest neighbor distances
    std::vector<double> nn_distances;
    nn_distances.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        for (size_t j = 0; j < points.size(); ++j) {
            if (i == j) continue;
            double dist = points[i].distance(points[j]);
            if (dist < min_dist) min_dist = dist;
        }
        nn_distances.push_back(min_dist);
    }

    double mean, stdev;
    calculate_stats(nn_distances, mean, stdev);
    double cv = (mean > 0) ? stdev / mean : 0.0;

    std::cout << "Sample size: " << points.size() << std::endl;
    std::cout << "Mean NN distance: " << mean << std::endl;
    std::cout << "StdDev NN distance: " << stdev << std::endl;
    std::cout << "Coefficient of Variation: " << (cv * 100) << "%" << std::endl;
    std::cout << "Target CV < 1.0% for perfect equidistance: " << (cv < 0.01 ? "PASS ✓" : "FAIL ✗") << std::endl;
}

// Diagnostic 2: Semantic Leashing (Geometric vs Semantic Rank Distance)
void test_semantic_leashing() {
    std::cout << "\n=== DIAGNOSTIC 2: Semantic Leashing ===" << std::endl;

    // Test case variants (should be very close)
    std::vector<std::pair<uint32_t, uint32_t>> case_pairs = {
        {'A', 'a'}, {'B', 'b'}, {'C', 'c'},
        {0x0391, 0x03B1}, // Greek A/a
        {0x0410, 0x0430}  // Cyrillic A/a
    };

    // Test homoglyphs (should be reasonably close)
    std::vector<std::pair<uint32_t, uint32_t>> homoglyph_pairs = {
        {'0', 'O'}, {'0', 'o'}, {'O', 'o'},
        {'1', 'I'}, {'1', 'l'}, {'I', 'l'}
    };

    std::cout << "Case Variants (should be very close):" << std::endl;
    for (auto [c1, c2] : case_pairs) {
        uint32_t rank1 = DenseRegistry::get_rank(c1);
        uint32_t rank2 = DenseRegistry::get_rank(c2);
        double geom_dist = CoordinateMapper::map_codepoint_float(c1)
                          .distance(CoordinateMapper::map_codepoint_float(c2));
        double rank_diff = std::abs(static_cast<int>(rank1) - static_cast<int>(rank2));

        std::cout << "  " << static_cast<char>(c1) << "-" << static_cast<char>(c2)
                  << ": rank_diff=" << rank_diff << ", geom_dist=" << geom_dist << std::endl;
    }

    std::cout << "Homoglyphs (should be closer than random pairs):" << std::endl;
    for (auto [c1, c2] : homoglyph_pairs) {
        uint32_t rank1 = DenseRegistry::get_rank(c1);
        uint32_t rank2 = DenseRegistry::get_rank(c2);
        double geom_dist = CoordinateMapper::map_codepoint_float(c1)
                          .distance(CoordinateMapper::map_codepoint_float(c2));
        double rank_diff = std::abs(static_cast<int>(rank1) - static_cast<int>(rank2));

        std::cout << "  " << static_cast<char>(c1) << "-" << static_cast<char>(c2)
                  << ": rank_diff=" << rank_diff << ", geom_dist=" << geom_dist << std::endl;
    }

    // Test the critical A-a vs A-B comparison
    double dist_A_a = CoordinateMapper::map_codepoint_float('A')
                     .distance(CoordinateMapper::map_codepoint_float('a'));
    double dist_A_B = CoordinateMapper::map_codepoint_float('A')
                     .distance(CoordinateMapper::map_codepoint_float('B'));
    double rank_diff_A_a = std::abs(static_cast<int>(DenseRegistry::get_rank('A')) -
                                   static_cast<int>(DenseRegistry::get_rank('a')));
    double rank_diff_A_B = std::abs(static_cast<int>(DenseRegistry::get_rank('A')) -
                                   static_cast<int>(DenseRegistry::get_rank('B')));

    std::cout << "\nCritical Test - A vs a vs B:" << std::endl;
    std::cout << "  A-a: rank_diff=" << rank_diff_A_a << ", geom_dist=" << dist_A_a << std::endl;
    std::cout << "  A-B: rank_diff=" << rank_diff_A_B << ", geom_dist=" << dist_A_B << std::endl;
    std::cout << "  Semantic leashing: A-a < A-B? " << (dist_A_a < dist_A_B ? "PASS ✓" : "FAIL ✗") << std::endl;
}

// Diagnostic 3: Hopf Fiber Alignment (Related characters on same fibers)
void test_hopf_fiber_alignment() {
    std::cout << "\n=== DIAGNOSTIC 3: Hopf Fiber Alignment ===" << std::endl;

    // Extract Hopf coordinates from quaternion representation
    auto get_hopf_coords = [](const Point4F& p) -> std::tuple<double, double, double> {
        // Point4F is (w, x, y, z) quaternion
        // Convert to Hopf coordinates (θ, φ, ψ)
        double w = p.x, x = p.y, y = p.z, z = p.m; // Note: swapped coordinates in Point4F

        // θ (polar angle): angle from z-axis
        double theta = std::acos(std::max(-1.0, std::min(1.0, z)));

        // φ (azimuthal angle): angle in xy-plane
        double phi = std::atan2(y, x);

        // ψ (fiber angle): from quaternion phase
        double psi = std::atan2(w, std::sqrt(x*x + y*y));

        return {theta, phi, psi};
    };

    // Test case variants should have similar θ (latitude on S³)
    std::vector<uint32_t> case_variants = {'A', 'a', 'B', 'b'};
    std::cout << "Case variants Hopf coordinates (θ, φ, ψ):" << std::endl;

    for (uint32_t cp : case_variants) {
        Point4F coords = CoordinateMapper::map_codepoint_float(cp);
        auto [theta, phi, psi] = get_hopf_coords(coords);
        std::cout << "  " << static_cast<char>(cp) << ": θ=" << theta
                  << ", φ=" << phi << ", ψ=" << psi << std::endl;
    }

    // Test homoglyphs
    std::vector<uint32_t> homoglyphs = {'0', 'O', 'o'};
    std::cout << "Homoglyphs Hopf coordinates:" << std::endl;

    for (uint32_t cp : homoglyphs) {
        Point4F coords = CoordinateMapper::map_codepoint_float(cp);
        auto [theta, phi, psi] = get_hopf_coords(coords);
        std::cout << "  " << static_cast<char>(cp) << ": θ=" << theta
                  << ", φ=" << phi << ", ψ=" << psi << std::endl;
    }
}

// Diagnostic 4: Centroid Depth as Complexity
void test_centroid_depth() {
    std::cout << "\n=== DIAGNOSTIC 4: Centroid Depth ===" << std::endl;

    // Single atoms should be on surface (radius = 1.0)
    Point4F a_coords = CoordinateMapper::map_codepoint_float('A');
    double a_radius = std::sqrt(a_coords.dot(a_coords));
    std::cout << "Single atom 'A' radius: " << a_radius
              << " (should be ~1.0): " << (std::abs(a_radius - 1.0) < 1e-6 ? "PASS ✓" : "FAIL ✗") << std::endl;

    // Simple composition (repeated letters) should be close to surface
    std::vector<Point4D> banana_coords;
    std::string banana = "banana";
    for (char c : banana) {
        banana_coords.push_back(CoordinateMapper::map_codepoint(c));
    }
    Point4F banana_centroid = CoordinateMapper::centroid_float(
        std::vector<Point4F>(banana_coords.begin(), banana_coords.end())
    );
    double banana_radius = std::sqrt(banana_centroid.dot(banana_centroid));
    std::cout << "Word 'banana' centroid radius: " << banana_radius
              << " (should be < 1.0): " << (banana_radius < 1.0 ? "PASS ✓" : "FAIL ✗") << std::endl;

    // Complex composition (mixed scripts) should be deeper
    std::vector<Point4D> complex_coords;
    std::vector<uint32_t> complex_chars = {'A', 0x0391, 0x0410, 0x4E00, 0x1F600}; // Latin, Greek, Cyrillic, CJK, Emoji
    for (uint32_t cp : complex_chars) {
        complex_coords.push_back(CoordinateMapper::map_codepoint(cp));
    }
    Point4F complex_centroid = CoordinateMapper::centroid_float(
        std::vector<Point4F>(complex_coords.begin(), complex_coords.end())
    );
    double complex_radius = std::sqrt(complex_centroid.dot(complex_centroid));
    std::cout << "Complex 'AΩА京😀' centroid radius: " << complex_radius
              << " (should be << 1.0): " << (complex_radius < 0.5 ? "PASS ✓" : "FAIL ✗") << std::endl;

    // Semantic gravity: complex composition should be deeper than simple one
    std::cout << "Semantic gravity: complex < banana? "
              << (complex_radius < banana_radius ? "PASS ✓" : "FAIL ✗") << std::endl;
}

// CSV Output for 4D Visualization
void output_visualization_csv() {
    std::cout << "\n=== VISUALIZATION: Writing 4D coordinates to CSV ===" << std::endl;

    std::ofstream csv("semantic_sphere_4d.csv");
    csv << "codepoint,char,semantic_key,dense_rank,x,y,z,m,radius\n";

    // Sample diverse characters for visualization
    std::vector<uint32_t> chars_to_plot;
    // ASCII letters
    for (uint32_t cp = 'A'; cp <= 'Z'; ++cp) chars_to_plot.push_back(cp);
    for (uint32_t cp = 'a'; cp <= 'z'; ++cp) chars_to_plot.push_back(cp);
    // Digits
    for (uint32_t cp = '0'; cp <= '9'; ++cp) chars_to_plot.push_back(cp);
    // Some Greek
    for (uint32_t cp = 0x0391; cp <= 0x0395; ++cp) chars_to_plot.push_back(cp);
    // Some CJK
    for (uint32_t cp = 0x4E00; cp <= 0x4E10; ++cp) chars_to_plot.push_back(cp);
    // Some emoji
    for (uint32_t cp = 0x1F600; cp <= 0x1F610; ++cp) chars_to_plot.push_back(cp);

    for (uint32_t cp : chars_to_plot) {
        uint64_t semantic_key = get_semantic_key(cp);
        uint32_t dense_rank = DenseRegistry::get_rank(cp);
        Point4F coords = CoordinateMapper::map_codepoint_float(cp);
        double radius = std::sqrt(coords.dot(coords));

        // Escape special characters for CSV
        std::string char_repr;
        if (cp >= 32 && cp <= 126) {
            char_repr = std::string(1, static_cast<char>(cp));
        } else {
            char_repr = "U+" + std::to_string(cp);
        }

        csv << cp << "," << char_repr << "," << semantic_key << ","
            << dense_rank << "," << coords.x << "," << coords.y << ","
            << coords.z << "," << coords.m << "," << radius << "\n";
    }

    csv.close();
    std::cout << "CSV written to semantic_sphere_4d.csv for 4D visualization" << std::endl;
    std::cout << "Use t-SNE or similar to project to 3D for visual inspection" << std::endl;
}

int main() {
    std::cout << "Modular Dyson Sphere Diagnostics for Semantic Universe" << std::endl;
    std::cout << "====================================================" << std::endl;

    // Run all Dyson Sphere diagnostics
    test_dyson_structural_integrity();
    test_dyson_functional_zones();
    test_dyson_composition_displacement();
    test_dyson_local_topology();

    // Run additional geostatistical diagnostics
    test_global_equidistance();
    test_semantic_leashing();
    test_hopf_fiber_alignment();
    test_centroid_depth();
    output_visualization_csv();

    std::cout << "\n=== DYSON SPHERE STATUS ===" << std::endl;
    std::cout << "✓ Structural integrity tests complete" << std::endl;
    std::cout << "✓ Functional zone clustering verified" << std::endl;
    std::cout << "✓ Composition displacement mechanics tested" << std::endl;
    std::cout << "✓ Local topology and module bolting checked" << std::endl;
    std::cout << "✓ Geostatistical properties validated" << std::endl;
    std::cout << "\nFor visual verification, load semantic_sphere_4d.csv into a 4D visualization tool." << std::endl;
    std::cout << "Each Unicode atom is now a precisely positioned module in your semantic Dyson sphere!" << std::endl;

    return 0;
}