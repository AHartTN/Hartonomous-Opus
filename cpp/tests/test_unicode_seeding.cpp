#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <set>
#include <limits>
#include <thread>
#include <mutex>
#include <atomic>
#ifdef _WIN32
#include <cstdlib>
#endif
#include "../include/hypercube/coordinates.hpp"
#include "../include/hypercube/semantic_ordering.hpp"

using namespace hypercube;

namespace {

// Statistical helper functions
double calculate_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double calculate_std_dev(const std::vector<double>& values, double mean) {
    if (values.size() <= 1) return 0.0;
    double sum_sq = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / (values.size() - 1));
}

double calculate_coefficient_of_variation(const std::vector<double>& values) {
    double mean = calculate_mean(values);
    if (mean == 0.0) return 0.0;
    double std_dev = calculate_std_dev(values, mean);
    return std_dev / std::abs(mean);
}

std::tuple<double, double, double> calculate_percentiles(const std::vector<double>& values,
                                                        double p5 = 5.0, double p95 = 95.0) {
    if (values.empty()) return {0.0, 0.0, 0.0};

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    auto get_percentile = [&](double p) -> double {
        if (p <= 0.0) return sorted.front();
        if (p >= 100.0) return sorted.back();

        double rank = (p / 100.0) * (sorted.size() - 1);
        size_t lower = static_cast<size_t>(rank);
        size_t upper = std::min(lower + 1, sorted.size() - 1);
        double weight = rank - lower;

        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
    };

    return {get_percentile(p5), get_percentile(50.0), get_percentile(p95)};
}

// Test result structure
struct TestResult {
    std::string test_name;
    bool passed;
    std::string message;
    double metric_value;
    double threshold;

    TestResult(const std::string& name, bool pass, const std::string& msg = "",
               double value = 0.0, double thresh = 0.0)
        : test_name(name), passed(pass), message(msg), metric_value(value), threshold(thresh) {}
};

class UnicodeSeedingTester {
private:
    std::vector<TestResult> results_;
    std::mutex results_mutex_;

    void add_result(const TestResult& result) {
        std::lock_guard<std::mutex> lock(results_mutex_);
        results_.push_back(result);
    }

public:
    // Test semantic proximity between related characters
    void test_semantic_proximity() {
        std::cout << "Testing semantic proximity..." << std::endl;

        // Test case-variant proximity (A should be close to a)
        std::vector<std::pair<uint32_t, uint32_t>> case_pairs = {
            {'A', 'a'}, {'B', 'b'}, {'Z', 'z'}, {'0', 'O'}, {'1', 'I'}, {'1', 'l'}
        };

        double total_distance = 0.0;
        int count = 0;

        for (auto [upper, lower] : case_pairs) {
            Point4F p1 = CoordinateMapper::map_codepoint_float(upper);
            Point4F p2 = CoordinateMapper::map_codepoint_float(lower);
            double dist = p1.distance(p2);
            total_distance += dist;
            count++;

            // Individual check: case variants should be very close (< 0.01 on unit sphere)
            bool close = dist < 0.01;
            add_result(TestResult(
                "Case proximity: " + std::string(1, char(upper)) + "/" + std::string(1, char(lower)),
                close,
                close ? "" : "Distance too large: " + std::to_string(dist),
                dist, 0.01
            ));
        }

        double avg_case_distance = total_distance / count;
        bool case_clustering_good = avg_case_distance < 0.005;
        add_result(TestResult(
            "Average case-variant proximity",
            case_clustering_good,
            case_clustering_good ? "" : "Average distance too large: " + std::to_string(avg_case_distance),
            avg_case_distance, 0.005
        ));

        // Test digit clustering
        std::vector<Point4F> digits;
        for (uint32_t d = '0'; d <= '9'; ++d) {
            digits.push_back(CoordinateMapper::map_codepoint_float(d));
        }

        // Calculate centroid of digits
        Point4F digit_centroid = CoordinateMapper::centroid_float(digits);

        // Check that all digits are close to their centroid
        double max_digit_distance = 0.0;
        for (const auto& digit : digits) {
            double dist = digit.distance(digit_centroid);
            max_digit_distance = std::max(max_digit_distance, dist);
        }

        bool digits_clustered = max_digit_distance < 0.1; // Within 10% of sphere radius
        add_result(TestResult(
            "Digit clustering",
            digits_clustered,
            digits_clustered ? "" : "Max digit-centroid distance too large: " + std::to_string(max_digit_distance),
            max_digit_distance, 0.1
        ));

        // Test letter group clustering (A-Z should be closer to each other than to digits)
        std::vector<Point4F> letters;
        for (uint32_t c = 'A'; c <= 'Z'; ++c) {
            letters.push_back(CoordinateMapper::map_codepoint_float(c));
        }

        Point4F letter_centroid = CoordinateMapper::centroid_float(letters);
        double letter_digit_separation = letter_centroid.distance(digit_centroid);

        // Letters and digits should be reasonably separated but not extremely far
        bool reasonable_separation = letter_digit_separation > 0.05 && letter_digit_separation < 0.8;
        add_result(TestResult(
            "Letter-digit separation",
            reasonable_separation,
            reasonable_separation ? "" : "Unexpected separation: " + std::to_string(letter_digit_separation),
            letter_digit_separation, 0.5
        ));

        std::cout << "  Semantic proximity tests completed" << std::endl;
    }

    // Test geometric properties of the mapping
    void test_geometric_properties() {
        std::cout << "Testing geometric properties..." << std::endl;

        const int SAMPLE_SIZE = 5000;
        std::vector<Point4F> points;

        // Sample across different Unicode ranges for comprehensive testing
        std::vector<std::pair<uint32_t, uint32_t>> ranges = {
            {'A', 'Z'}, {'a', 'z'}, {'0', '9'}, {0x4E00, 0x4E20}, {0x1F600, 0x1F620}
        };

        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<uint32_t> dist;

        for (auto [start, end] : ranges) {
            for (uint32_t cp = start; cp <= end && points.size() < SAMPLE_SIZE; ++cp) {
                if (cp < 0xD800 || cp > 0xDFFF) { // Skip surrogates
                    points.push_back(CoordinateMapper::map_codepoint_float(cp));
                }
            }
        }

        // Fill remaining with random valid codepoints
        while (points.size() < SAMPLE_SIZE) {
            uint32_t cp = dist(gen) % 0x10FFFF;
            if ((cp < 0xD800 || cp > 0xDFFF) && cp <= 0x10FFFF) {
                points.push_back(CoordinateMapper::map_codepoint_float(cp));
            }
        }

        // Test 1: All points are on the 3-sphere surface (norm = 1)
        int on_surface = 0;
        double max_norm_error = 0.0;

        for (const auto& p : points) {
            double norm = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z + p.m*p.m);
            double error = std::abs(norm - 1.0);
            max_norm_error = std::max(max_norm_error, error);
            if (error < 1e-10) on_surface++;
        }

        bool all_on_surface = on_surface == static_cast<int>(points.size());
        add_result(TestResult(
            "Surface constraint (all points on S³)",
            all_on_surface,
            all_on_surface ? "" : std::to_string(points.size() - on_surface) + " points off surface, max error: " + std::to_string(max_norm_error),
            max_norm_error, 1e-10
        ));

        // Test 2: Enhanced uniformity metrics with CV ≤30% target
        std::vector<double> nn_distances;

        // Brute force NN computation for accuracy
        for (size_t i = 0; i < points.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            for (size_t j = 0; j < points.size(); ++j) {
                if (i != j) {
                    double dist = points[i].distance(points[j]);
                    if (dist < min_dist) min_dist = dist;
                }
            }
            nn_distances.push_back(min_dist);
        }

        double cv = calculate_coefficient_of_variation(nn_distances);
        auto [p5, median, p95] = calculate_percentiles(nn_distances);

        // PRIMARY TARGET: CV should be ≤30% for perfect surface distribution
        bool perfect_distribution = cv <= 0.30;
        add_result(TestResult(
            "Perfect surface distribution (CV ≤30%)",
            perfect_distribution,
            perfect_distribution ? "✓ Target achieved: " + std::to_string(cv * 100) + "%" :
                                 "✗ CV too high: " + std::to_string(cv * 100) + "% (target: ≤30%)",
            cv, 0.30
        ));

        // Additional uniformity validation
        bool good_uniformity = cv < 0.35; // Slightly relaxed for secondary check
        add_result(TestResult(
            "Distribution uniformity (NN distance CV)",
            good_uniformity,
            good_uniformity ? "" : "CV too high: " + std::to_string(cv * 100) + "%",
            cv, 0.35
        ));

        // Test 3: No clustering artifacts - check for pathological distributions
        double range_ratio = p95 / p5;
        bool no_artifacts = range_ratio < 10.0; // 95th percentile shouldn't be 10x the 5th
        add_result(TestResult(
            "No clustering artifacts (distance range ratio)",
            no_artifacts,
            no_artifacts ? "" : "Range ratio too large: " + std::to_string(range_ratio),
            range_ratio, 10.0
        ));

        std::cout << "  Geometric properties: CV=" << (cv * 100) << "%, Range=" << range_ratio << "x" << std::endl;
    }

    // Test distribution quality across the sphere
    void test_distribution_quality() {
        std::cout << "Testing distribution quality..." << std::endl;

        const int SAMPLE_SIZE = 10000;
        std::vector<Point4F> points;

        // Sample evenly across semantic ranks for better distribution testing
        uint64_t total_ranks = SemanticOrdering::total_codepoints();
        std::vector<uint32_t> sampled_cps;

        for (uint64_t i = 0; i < SAMPLE_SIZE; ++i) {
            uint64_t rank = i * (total_ranks / SAMPLE_SIZE);
            uint32_t cp = SemanticOrdering::get_codepoint(static_cast<uint32_t>(rank));
            if (cp != 0) {
                sampled_cps.push_back(cp);
                points.push_back(CoordinateMapper::map_codepoint_float(cp));
            }
        }

        // Test 1: Even spread across S³ - check coordinate distribution
        std::vector<double> x_coords, y_coords, z_coords, m_coords;
        for (const auto& p : points) {
            x_coords.push_back(p.x);
            y_coords.push_back(p.y);
            z_coords.push_back(p.z);
            m_coords.push_back(p.m);
        }

        // Check that coordinates are reasonably uniformly distributed in [-1,1]
        auto check_coordinate_distribution = [&](const std::vector<double>& coords, const std::string& name) {
            double cv = calculate_coefficient_of_variation(coords);
            auto [p5, median, p95] = calculate_percentiles(coords);

            // For uniform distribution on sphere, coordinates should have reasonable spread
            bool good_spread = (p95 - p5) > 1.0 && cv < 1.0; // Range > 1.0, CV < 100%
            add_result(TestResult(
                name + " coordinate distribution",
                good_spread,
                good_spread ? "" : "Poor spread: range=" + std::to_string(p95 - p5) + ", CV=" + std::to_string(cv),
                cv, 1.0
            ));

            return good_spread;
        };

        check_coordinate_distribution(x_coords, "X");
        check_coordinate_distribution(y_coords, "Y");
        check_coordinate_distribution(z_coords, "Z");
        check_coordinate_distribution(m_coords, "M");

        // Test 2: Spherical cap uniformity - check density in different regions
        // Divide sphere into octants and check point distribution
        std::vector<int> octant_counts(8, 0);

        for (const auto& p : points) {
            int octant = 0;
            if (p.x >= 0) octant |= 1;
            if (p.y >= 0) octant |= 2;
            if (p.z >= 0) octant |= 4;
            octant_counts[octant]++;
        }

        double expected_per_octant = SAMPLE_SIZE / 8.0;
        double max_deviation = 0.0;

        for (int count : octant_counts) {
            double deviation = std::abs(count - expected_per_octant) / expected_per_octant;
            max_deviation = std::max(max_deviation, deviation);
        }

        bool even_spread = max_deviation < 0.5; // Max 50% deviation from expected
        add_result(TestResult(
            "Even spread across S³ octants",
            even_spread,
            even_spread ? "" : "Max octant deviation: " + std::to_string(max_deviation * 100) + "%",
            max_deviation, 0.5
        ));

        // Test 3: No degenerate mappings - all points should be distinct
        std::set<std::tuple<double, double, double, double>> unique_points;
        for (const auto& p : points) {
            unique_points.insert({p.x, p.y, p.z, p.m});
        }

        bool all_unique = unique_points.size() == points.size();
        add_result(TestResult(
            "No degenerate mappings (all points unique)",
            all_unique,
            all_unique ? "" : std::to_string(points.size() - unique_points.size()) + " duplicate points",
            static_cast<double>(unique_points.size()), static_cast<double>(points.size())
        ));

        std::cout << "  Distribution quality tests completed" << std::endl;
    }

    // Test robustness and edge cases
    void test_robustness() {
        std::cout << "Testing robustness..." << std::endl;

        // Test 1: Surrogate handling
        for (uint32_t cp = 0xD800; cp <= 0xDFFF; ++cp) {
            CodepointMapping mapping = CoordinateMapper::map_codepoint_full(cp);

            // All surrogates should map to the same reserved location
            uint32_t expected_x = 0x80000000U ^ 0x7FFFFFFFU;
            bool correct_reserved = (mapping.coords.x == expected_x) &&
                                   (mapping.coords.y == 0x80000000U) &&
                                   (mapping.coords.z == 0x80000000U) &&
                                   (mapping.coords.m == 0x80000000U) &&
                                   (mapping.hilbert.lo == 0 && mapping.hilbert.hi == 0);

            add_result(TestResult(
                "Surrogate handling: U+" + std::to_string(cp),
                correct_reserved,
                correct_reserved ? "" : "Incorrect surrogate mapping"
            ));

            // Float coordinates should also be consistent
            Point4F float_coords = CoordinateMapper::map_codepoint_float(cp);
            bool float_consistent = std::abs(float_coords.x - 1.0) < 1e-10 &&
                                   std::abs(float_coords.y) < 1e-10 &&
                                   std::abs(float_coords.z) < 1e-10 &&
                                   std::abs(float_coords.m) < 1e-10;

            add_result(TestResult(
                "Surrogate float consistency: U+" + std::to_string(cp),
                float_consistent,
                float_consistent ? "" : "Inconsistent float coordinates for surrogate"
            ));
        }

        // Test 2: Edge cases - maximum codepoint
        uint32_t max_cp = 0x10FFFF;
        Point4F max_coords = CoordinateMapper::map_codepoint_float(max_cp);
        double max_norm = std::sqrt(max_coords.x*max_coords.x + max_coords.y*max_coords.y +
                                   max_coords.z*max_coords.z + max_coords.m*max_coords.m);
        bool max_on_surface = std::abs(max_norm - 1.0) < 1e-10;

        add_result(TestResult(
            "Maximum codepoint handling",
            max_on_surface,
            max_on_surface ? "" : "Max codepoint not on surface, norm=" + std::to_string(max_norm),
            max_norm, 1.0
        ));

        // Test 3: Invalid codepoints (above maximum)
        uint32_t invalid_cp = 0x110000;
        Point4F invalid_coords = CoordinateMapper::map_codepoint_float(invalid_cp);
        double invalid_norm = std::sqrt(invalid_coords.x*invalid_coords.x + invalid_coords.y*invalid_coords.y +
                                       invalid_coords.z*invalid_coords.z + invalid_coords.m*invalid_coords.m);
        bool invalid_handled = std::abs(invalid_norm - 1.0) < 1e-10; // Should still produce valid point

        add_result(TestResult(
            "Invalid codepoint handling",
            invalid_handled,
            invalid_handled ? "" : "Invalid codepoint produced invalid coordinates",
            invalid_norm, 1.0
        ));

        // Test 4: Thread safety
        const int num_threads = 4;
        const int calls_per_thread = 1000;
        std::vector<std::thread> threads;
        std::atomic<bool> thread_safety_ok{true};

        auto worker = [&]() {
            try {
                for (int i = 0; i < calls_per_thread; ++i) {
                    uint32_t cp = 'A' + (i % 26);
                    Point4F coords = CoordinateMapper::map_codepoint_float(cp);
                    double norm = std::sqrt(coords.x*coords.x + coords.y*coords.y +
                                           coords.z*coords.z + coords.m*coords.m);
                    if (std::abs(norm - 1.0) > 1e-6) {
                        thread_safety_ok = false;
                    }
                }
            } catch (...) {
                thread_safety_ok = false;
            }
        };

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker);
        }

        for (auto& th : threads) {
            th.join();
        }

        add_result(TestResult(
            "Thread safety",
            thread_safety_ok.load(),
            thread_safety_ok.load() ? "" : "Thread safety violation detected"
        ));

        // Test 5: Determinism
        uint32_t test_cp = 0x4E00; // Chinese character
        Point4F coords1 = CoordinateMapper::map_codepoint_float(test_cp);
        Point4F coords2 = CoordinateMapper::map_codepoint_float(test_cp);

        bool deterministic = coords1.distance(coords2) < 1e-15;
        add_result(TestResult(
            "Determinism",
            deterministic,
            deterministic ? "" : "Non-deterministic results for same codepoint"
        ));

        std::cout << "  Robustness tests completed" << std::endl;
    }

    // Comprehensive optimization pipeline validation
    void test_optimization_pipeline() {
        std::cout << "Testing comprehensive optimization pipeline..." << std::endl;

        // Create test mapping for all valid codepoints (sample for performance)
        const size_t TEST_SIZE = 50000; // Large enough sample for statistical significance
        std::map<uint32_t, Point4F> test_points;

        // Sample across semantic ranks for comprehensive testing
        uint32_t total_codepoints = SemanticOrdering::total_codepoints();
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            uint32_t rank = static_cast<uint32_t>((i * total_codepoints) / TEST_SIZE);
            uint32_t cp = SemanticOrdering::get_codepoint(rank);
            if (cp != 0 && cp < 0xD800) { // Skip surrogates for initial test
                test_points[cp] = CoordinateMapper::map_codepoint_float(cp);
            }
        }

        std::cout << "  Created test mapping with " << test_points.size() << " codepoints" << std::endl;

        // Run the complete optimization pipeline
        bool optimization_success = CoordinateMapper::optimize_distribution(test_points);

        // Validate final results
        auto final_diag = CoordinateMapper::compute_diagnostics(test_points);

        // PRIMARY SUCCESS CRITERIA: CV ≤30%
        bool target_achieved = final_diag.chordal_nn_cv <= 0.30;
        add_result(TestResult(
            "Optimization pipeline success (CV ≤30%)",
            target_achieved && optimization_success,
            target_achieved ? "✓ Perfect distribution achieved: CV = " + std::to_string(final_diag.chordal_nn_cv * 100) + "%" :
                            "✗ Target not met: CV = " + std::to_string(final_diag.chordal_nn_cv * 100) + "% (target: ≤30%)",
            final_diag.chordal_nn_cv, 0.30
        ));

        // Validate semantic proximity preservation
        validate_semantic_proximity_after_optimization(test_points);

        // Validate surface constraint maintenance
        validate_surface_constraint_after_optimization(test_points);

        std::cout << "  Optimization pipeline validation completed" << std::endl;
    }

    void validate_semantic_proximity_after_optimization(const std::map<uint32_t, Point4F>& points) {
        // Test that case variants remain close after optimization
        std::vector<std::pair<uint32_t, uint32_t>> case_pairs = {
            {'A', 'a'}, {'B', 'b'}, {'0', 'O'}, {'1', 'I'}, {'1', 'l'}
        };

        double total_distance = 0.0;
        int count = 0;

        for (auto [upper, lower] : case_pairs) {
            auto it1 = points.find(upper);
            auto it2 = points.find(lower);
            if (it1 != points.end() && it2 != points.end()) {
                double dist = it1->second.distance(it2->second);
                total_distance += dist;
                count++;

                // Case variants should remain very close (< 0.1 on unit sphere)
                bool proximity_preserved = dist < 0.1;
                add_result(TestResult(
                    "Semantic proximity preserved after optimization: " + std::string(1, char(upper)) + "/" + std::string(1, char(lower)),
                    proximity_preserved,
                    proximity_preserved ? "" : "Distance too large after optimization: " + std::to_string(dist),
                    dist, 0.1
                ));
            }
        }

        if (count > 0) {
            double avg_distance = total_distance / count;
            bool semantic_clustering_good = avg_distance < 0.05;
            add_result(TestResult(
                "Average semantic proximity after optimization",
                semantic_clustering_good,
                semantic_clustering_good ? "" : "Average distance too large: " + std::to_string(avg_distance),
                avg_distance, 0.05
            ));
        }
    }

    void validate_surface_constraint_after_optimization(const std::map<uint32_t, Point4F>& points) {
        // Verify all points remain on the 3-sphere surface after optimization
        int on_surface = 0;
        double max_error = 0.0;

        for (const auto& [cp, pt] : points) {
            double norm = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z + pt.m*pt.m);
            double error = std::abs(norm - 1.0);
            max_error = std::max(max_error, error);
            if (error < 1e-6) on_surface++; // Tighter tolerance after optimization
        }

        bool all_on_surface = on_surface == static_cast<int>(points.size());
        add_result(TestResult(
            "Surface constraint maintained after optimization",
            all_on_surface,
            all_on_surface ? "" : std::to_string(points.size() - on_surface) + " points off surface, max error: " + std::to_string(max_error),
            max_error, 1e-6
        ));
    }

    // Run all tests and report results
    bool run_all_tests() {
        std::cout << "=== Unicode Character Seeding Comprehensive Tests ===" << std::endl;
        std::cout << "Testing semantic proximity, geometric properties, distribution quality, and robustness..." << std::endl;
        std::cout << std::endl;

        test_semantic_proximity();
        test_geometric_properties();
        test_distribution_quality();
        test_robustness();
        test_optimization_pipeline();

        // Report results
        std::cout << std::endl;
        std::cout << "=== Test Results Summary ===" << std::endl;

        int passed = 0;
        int total = 0;

        for (const auto& result : results_) {
            total++;
            if (result.passed) {
                passed++;
                std::cout << "✓ PASS: " << result.test_name << std::endl;
            } else {
                std::cout << "✗ FAIL: " << result.test_name;
                if (!result.message.empty()) {
                    std::cout << " - " << result.message;
                }
                if (result.threshold > 0.0) {
                    std::cout << " (value: " << result.metric_value << ", threshold: " << result.threshold << ")";
                }
                std::cout << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << "Overall: " << passed << "/" << total << " tests passed ("
                  << (total > 0 ? (passed * 100 / total) : 0) << "%)" << std::endl;

        return passed == total;
    }
};

} // anonymous namespace

void test_unicode_seeding_comprehensive() {
    UnicodeSeedingTester tester;
    bool all_passed = tester.run_all_tests();

    if (!all_passed) {
        std::cerr << "Some Unicode seeding tests failed!" << std::endl;
        assert(false && "Unicode seeding tests failed");
    }

    std::cout << "All Unicode seeding tests passed!" << std::endl;
}

int main() {
#ifdef _WIN32
    // Disable abort dialog on Windows to prevent message box when assert fails
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif

    test_unicode_seeding_comprehensive();
    return 0;
}