#include "hypercube/laplacian_4d.hpp"
#include <iostream>
#include <random>

using namespace hypercube;

int main() {
    std::cout << "=== Testing Laplacian Projection with Diagnostics ===\n\n";

    // Create small test dataset: 100 points in 32D
    const size_t n = 100;
    const size_t dim = 32;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> embeddings(n);
    std::vector<std::string> labels(n);

    for (size_t i = 0; i < n; ++i) {
        embeddings[i].resize(dim);
        for (size_t j = 0; j < dim; ++j) {
            embeddings[i][j] = dist(rng);
        }
        // Normalize
        float norm = 0.0f;
        for (float v : embeddings[i]) norm += v * v;
        norm = std::sqrt(norm);
        for (float& v : embeddings[i]) v /= norm;

        labels[i] = "token_" + std::to_string(i);
    }

    // Configure projector
    LaplacianConfig config;
    config.k_neighbors = 5;
    config.similarity_threshold = 0.0f;
    config.num_threads = 4;

    LaplacianProjector projector(config);

    // Run projection - this should show our diagnostic output
    auto result = projector.project(embeddings, labels);

    std::cout << "\n=== Test Complete ===\n";
    std::cout << "Projected " << result.coords.size() << " points\n";
    std::cout << "Edge count: " << result.edge_count << "\n";
    std::cout << "Eigenvalues: ";
    for (double ev : result.eigenvalues) {
        std::cout << ev << " ";
    }
    std::cout << "\n";

    // Check if coordinates are all zero (the bug)
    bool all_zero = true;
    for (const auto& coord : result.coords) {
        for (int d = 0; d < 4; ++d) {
            if (coord[d] != 0) {
                all_zero = false;
                break;
            }
        }
        if (!all_zero) break;
    }

    if (all_zero) {
        std::cerr << "\n[FAIL] All coordinates are zero! The Lanczos bug is still present.\n";
        return 1;
    } else {
        std::cout << "\n[SUCCESS] Coordinates are non-zero!\n";
        return 0;
    }
}
