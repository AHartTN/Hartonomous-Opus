#include "hypercube/laplacian_4d.hpp"
#include <iostream>
#include <vector>

using namespace hypercube;

int main() {
    // Create a simple test case
    std::vector<std::vector<float>> embeddings = {
        {1.0f, 0.0f, 0.0f, 0.0f}, // Point 0
        {0.0f, 1.0f, 0.0f, 0.0f}, // Point 1
        {0.0f, 0.0f, 1.0f, 0.0f}, // Point 2
        {0.0f, 0.0f, 0.0f, 1.0f}, // Point 3
        {0.5f, 0.5f, 0.0f, 0.0f}  // Point 4 (connects 0 and 1)
    };

    std::vector<std::string> labels = {"p0", "p1", "p2", "p3", "p4"};

    LaplacianConfig config;
    config.k_neighbors = 3;
    config.num_threads = 1;
    config.verbose = true;

    LaplacianProjector projector(config);

    try {
        ProjectionResult result = projector.project(embeddings, labels);

        std::cout << "SUCCESS! Eigenvalues: ";
        for (double ev : result.eigenvalues) {
            std::cout << ev << " ";
        }
        std::cout << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}