// =============================================================================
// Laplacian Eigenmap Tests
// =============================================================================

#include <gtest/gtest.h>
#include "hypercube/laplacian_4d.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace hypercube;

class LaplacianTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed RNG for reproducible tests
        rng.seed(42);
    }
    
    void TearDown() override {}
    
    std::mt19937 rng;
    
    // Generate random unit vectors
    std::vector<std::vector<float>> random_vectors(size_t n, size_t dim) {
        std::vector<std::vector<float>> vecs(n);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < n; ++i) {
            vecs[i].resize(dim);
            float norm = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                vecs[i][j] = dist(rng);
                norm += vecs[i][j] * vecs[i][j];
            }
            norm = std::sqrt(norm);
            for (size_t j = 0; j < dim; ++j) {
                vecs[i][j] /= norm;
            }
        }
        return vecs;
    }
    
    // Generate clustered vectors
    std::vector<std::vector<float>> clustered_vectors(size_t n, size_t dim, int num_clusters) {
        std::vector<std::vector<float>> vecs(n);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::uniform_int_distribution<int> cluster_dist(0, num_clusters - 1);
        
        // Create cluster centers
        std::vector<std::vector<float>> centers(num_clusters);
        for (int c = 0; c < num_clusters; ++c) {
            centers[c].resize(dim);
            for (size_t j = 0; j < dim; ++j) {
                centers[c][j] = static_cast<float>(c) * 2.0f + (j == 0 ? 1.0f : 0.0f);
            }
        }
        
        // Assign points to clusters with noise
        for (size_t i = 0; i < n; ++i) {
            int c = cluster_dist(rng);
            vecs[i].resize(dim);
            for (size_t j = 0; j < dim; ++j) {
                vecs[i][j] = centers[c][j] + dist(rng);
            }
        }
        
        return vecs;
    }
};

// Test projection produces valid 4D coordinates
TEST_F(LaplacianTest, ProjectionOutput) {
    auto data = random_vectors(50, 32);
    std::vector<std::string> labels(50);
    for (size_t i = 0; i < 50; ++i) {
        labels[i] = "test_" + std::to_string(i);
    }
    
    LaplacianConfig config;
    config.k_neighbors = 5;
    config.num_threads = 1;
    
    LaplacianProjector projector(config);
    auto result = projector.project(data, labels);
    
    // Should have same number of points
    EXPECT_EQ(result.coords.size(), 50);
    
    // Each point should have 4 coordinates
    for (const auto& coord : result.coords) {
        EXPECT_EQ(coord.size(), 4);
    }
}

// Test projection preserves relative distances (approximately)
TEST_F(LaplacianTest, LocalityPreservation) {
    auto data = clustered_vectors(30, 16, 3);  // 3 clusters
    std::vector<std::string> labels(30);
    for (size_t i = 0; i < 30; ++i) {
        labels[i] = "cluster_" + std::to_string(i);
    }
    
    LaplacianConfig config;
    config.k_neighbors = 5;
    config.num_threads = 1;
    
    LaplacianProjector projector(config);
    auto result = projector.project(data, labels);
    
    // Points from same cluster should be closer in projection
    // (This is a weak test - Laplacian Eigenmaps should preserve neighborhood structure)
    EXPECT_EQ(result.coords.size(), 30);
}

// Test determinism with same seed
TEST_F(LaplacianTest, Deterministic) {
    auto data = random_vectors(20, 16);
    std::vector<std::string> labels(20);
    for (size_t i = 0; i < 20; ++i) {
        labels[i] = "point_" + std::to_string(i);
    }
    
    LaplacianConfig config;
    config.k_neighbors = 3;
    config.num_threads = 1;
    
    LaplacianProjector projector1(config);
    LaplacianProjector projector2(config);
    
    auto result1 = projector1.project(data, labels);
    auto result2 = projector2.project(data, labels);
    
    // Results should be identical (or very close)
    for (size_t i = 0; i < 20; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            float diff = std::abs(static_cast<float>(result1.coords[i][j]) - 
                                 static_cast<float>(result2.coords[i][j]));
            EXPECT_LT(diff, 1.0f);  // Should be close (uint32 coords)
        }
    }
}

// Test with small dataset
TEST_F(LaplacianTest, SmallDataset) {
    // Minimum viable dataset
    std::vector<std::vector<float>> data = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f, 0.0f},
    };
    std::vector<std::string> labels = {"a", "b", "c", "d", "e"};
    
    LaplacianConfig config;
    config.k_neighbors = 2;
    config.num_threads = 1;
    
    LaplacianProjector projector(config);
    auto result = projector.project(data, labels);
    
    EXPECT_EQ(result.coords.size(), 5);
}

// Test spherical projection constraint
TEST_F(LaplacianTest, SphericalProjection) {
    auto data = random_vectors(30, 16);
    std::vector<std::string> labels(30);
    for (size_t i = 0; i < 30; ++i) {
        labels[i] = "test_" + std::to_string(i);
    }
    
    LaplacianConfig config;
    config.k_neighbors = 5;
    config.project_to_sphere = true;
    config.num_threads = 1;
    
    LaplacianProjector projector(config);
    auto result = projector.project(data, labels);
    
    // Sphere is centered at (2^31, 2^31, 2^31, 2^31), not origin
    constexpr double CENTER = 2147483648.0;  // 2^31
    
    // All points should lie on a sphere (approximately same distance from CENTER)
    std::vector<double> distances;
    for (const auto& coord : result.coords) {
        double dx = static_cast<double>(coord[0]) - CENTER;
        double dy = static_cast<double>(coord[1]) - CENTER;
        double dz = static_cast<double>(coord[2]) - CENTER;
        double dm = static_cast<double>(coord[3]) - CENTER;
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz + dm*dm);
        distances.push_back(dist);
    }
    
    // Check variance of distances is low
    double mean = 0.0;
    for (double d : distances) mean += d;
    mean /= distances.size();
    
    double variance = 0.0;
    for (double d : distances) {
        variance += (d - mean) * (d - mean);
    }
    variance /= distances.size();
    
    // Coefficient of variation should be small for spherical projection
    double cv = std::sqrt(variance) / mean;
    EXPECT_LT(cv, 0.01);  // Less than 1% variation - should be nearly exact
}
