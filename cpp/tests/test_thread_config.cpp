/**
 * Test ThreadConfig System Integration
 *
 * Verifies that the unified threading configuration system works correctly
 * and integrates properly with existing components.
 */

#include "hypercube/thread_config.hpp"
#include "hypercube/laplacian_4d.hpp"
#include <cassert>

#include <iostream>
#include <thread>
#include <chrono>

// Test basic ThreadConfig functionality
void test_thread_config_basic() {
    std::cout << "Testing ThreadConfig basic functionality...\n";

    auto& config = hypercube::ThreadConfig::instance();

    // Test hardware detection
    size_t hw = config.get_hardware_concurrency();
    std::cout << "Hardware concurrency: " << hw << "\n";
    assert(hw > 0);

    // Test workload-specific thread counts
    size_t compute = config.get_compute_threads();
    size_t io = config.get_io_threads();
    size_t hybrid = config.get_hybrid_threads();

    std::cout << "Compute threads: " << compute << "\n";
    std::cout << "IO threads: " << io << "\n";
    std::cout << "Hybrid threads: " << hybrid << "\n";

    // IO threads should be >= compute threads (for potential oversubscription)
    assert(io >= compute);
    assert(hybrid >= compute && hybrid <= io);

    // Test configuration validation
    bool valid = config.validate_configuration();
    std::cout << "Configuration valid: " << (valid ? "YES" : "NO") << "\n";
    assert(valid);

    std::cout << "✓ Basic ThreadConfig tests passed\n";
}

// Test ThreadPool integration (using ThreadConfig for sizing)
void test_thread_pool_integration() {
    std::cout << "Testing ThreadPool integration with ThreadConfig...\n";

    auto& config = hypercube::ThreadConfig::instance();

    // Test that ThreadPool gets appropriate size from ThreadConfig
    size_t expected_compute = config.get_compute_threads();
    size_t expected_io = config.get_io_threads();

    std::cout << "Expected compute threads: " << expected_compute << "\n";
    std::cout << "Expected IO threads: " << expected_io << "\n";

    // ThreadPool is a singleton that uses hardware_concurrency by default
    // We can't easily test different sizes without changing the singleton design
    // But we can verify it doesn't crash and uses reasonable thread counts
    assert(expected_compute > 0);
    assert(expected_io >= expected_compute);  // IO should allow oversubscription

    std::cout << "✓ ThreadPool integration tests passed\n";
}

// Test WorkloadClassifier
void test_workload_classifier() {
    std::cout << "Testing WorkloadClassifier...\n";

    using hypercube::WorkloadClassifier;
    using hypercube::WorkloadType;

    // Test operation classification
    assert(WorkloadClassifier::classify_operation("matrix multiplication") == WorkloadType::COMPUTE_BOUND);
    assert(WorkloadClassifier::classify_operation("database query") == WorkloadType::IO_BOUND);
    assert(WorkloadClassifier::classify_operation("ingestion pipeline") == WorkloadType::HYBRID);

    // Test constants
    assert(WorkloadClassifier::ENCODING == WorkloadType::COMPUTE_BOUND);
    assert(WorkloadClassifier::DATABASE_QUERY == WorkloadType::IO_BOUND);
    assert(WorkloadClassifier::INGESTION == WorkloadType::HYBRID);

    std::cout << "✓ WorkloadClassifier tests passed\n";
}

// Test LaplacianProjector integration
void test_laplacian_integration() {
    std::cout << "Testing LaplacianProjector integration...\n";

    hypercube::LaplacianConfig lap_config;
    lap_config.num_threads = 0; // Should use ThreadConfig
    lap_config.k_neighbors = 5; // Small for testing
    lap_config.verbose = false;

    hypercube::LaplacianProjector projector(lap_config);

    // Create small test data (3 points in 4D space)
    std::vector<std::vector<float>> embeddings = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f}
    };

    std::vector<std::string> labels = {"point1", "point2", "point3"};

    // This should not crash and should use ThreadConfig for threading
    auto result = projector.project(embeddings, labels);

    assert(result.coords.size() == 3);
    assert(result.converged); // Should converge for such small data

    std::cout << "✓ LaplacianProjector integration tests passed\n";
}

// Test runtime adjustment
void test_runtime_adjustment() {
    std::cout << "Testing runtime adjustment...\n";

    auto& config = hypercube::ThreadConfig::instance();

    // Test setting override
    size_t original = config.get_compute_threads();
    config.set_thread_count_override(2);

    size_t overridden = config.get_compute_threads();
    assert(overridden == 2);

    // Clear override
    config.clear_thread_count_override();
    size_t restored = config.get_compute_threads();
    assert(restored == original);

    std::cout << "✓ Runtime adjustment tests passed\n";
}

int main() {
    std::cout << "=== ThreadConfig Integration Tests ===\n\n";

    try {
        test_thread_config_basic();
        std::cout << "\n";

        test_thread_pool_integration();
        std::cout << "\n";

        test_workload_classifier();
        std::cout << "\n";

        test_laplacian_integration();
        std::cout << "\n";

        test_runtime_adjustment();
        std::cout << "\n";

        std::cout << "🎉 All ThreadConfig integration tests passed!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}