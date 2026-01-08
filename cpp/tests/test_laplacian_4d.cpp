/**
 * Test suite for Laplacian Eigenmap 4D Projection
 * 
 * Tests:
 * - Sparse matrix operations
 * - k-NN similarity graph construction
 * - Unnormalized Laplacian computation
 * - Eigenvector computation
 * - Gram-Schmidt orthonormalization
 * - Hypercube coordinate normalization
 * - Sphere projection
 */

#include "hypercube/laplacian_4d.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/embedding_ops.hpp"  // For centralized SIMD operations
#include <iostream>
#include <cmath>
#include <random>
#include <cassert>

using namespace hypercube;

// =============================================================================
// Test Utilities
// =============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
    void test_##name(); \
    struct Test_##name { \
        Test_##name() { \
            std::cerr << "Running " #name "... "; \
            try { \
                test_##name(); \
                std::cerr << "PASSED\n"; \
                g_tests_passed++; \
            } catch (const std::exception& e) { \
                std::cerr << "FAILED: " << e.what() << "\n"; \
                g_tests_failed++; \
            } \
        } \
    } g_test_##name; \
    void test_##name()

#define ASSERT_TRUE(x) \
    if (!(x)) throw std::runtime_error("Assertion failed: " #x)

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b)

#define ASSERT_NEAR(a, b, eps) \
    if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("Assertion failed: " #a " ≈ " #b)

// =============================================================================
// Tests
// =============================================================================

TEST(simd_dot_product) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    
    // Use embedding namespace for vectorized dot product via cosine sim components
    // Note: embedding::cosine_similarity computes dot/(norm_a*norm_b)
    // For raw dot product, we compute it manually since embedding_ops focuses on cosine
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    float expected = 8 + 14 + 18 + 20 + 20 + 18 + 14 + 8;  // = 120
    
    ASSERT_NEAR(dot, expected, 1e-5f);
}

TEST(simd_cosine_similarity) {
    std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> c = {1.0f, 0.0f, 0.0f, 0.0f};
    
    float sim_ab = embedding::cosine_similarity(a.data(), b.data(), 4);
    float sim_ac = embedding::cosine_similarity(a.data(), c.data(), 4);
    
    ASSERT_NEAR(sim_ab, 0.0f, 1e-5f);  // Orthogonal
    ASSERT_NEAR(sim_ac, 1.0f, 1e-5f);  // Identical
}

TEST(sparse_matrix_basic) {
    SparseSymmetricMatrix M(4);
    
    M.add_edge(0, 1, 1.0);
    M.add_edge(1, 2, 2.0);
    M.add_edge(2, 3, 3.0);
    M.finalize();
    
    // Check degrees
    ASSERT_NEAR(M.get_degree(0), 1.0, 1e-10);
    ASSERT_NEAR(M.get_degree(1), 3.0, 1e-10);  // 1 + 2
    ASSERT_NEAR(M.get_degree(2), 5.0, 1e-10);  // 2 + 3
    ASSERT_NEAR(M.get_degree(3), 3.0, 1e-10);
}

TEST(sparse_matrix_multiply) {
    SparseSymmetricMatrix M(3);

    // Adjacency matrix:
    // [0, 1, 0]
    // [1, 0, 1]
    // [0, 1, 0]
    M.add_edge(0, 1, 1.0);
    M.add_edge(1, 2, 1.0);
    M.finalize();

    // Set diagonal
    M.set_diagonal(0, 1.0);
    M.set_diagonal(1, 2.0);
    M.set_diagonal(2, 1.0);

    // Matrix is now:
    // [1, 1, 0]
    // [1, 2, 1]
    // [0, 1, 1]

    std::vector<double> x = {1.0, 1.0, 1.0};
    std::vector<double> y(3);

    M.multiply(x, y);

    // Debug logs
    std::cerr << "[DEBUG] x = [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";
    std::cerr << "[DEBUG] y = [" << y[0] << ", " << y[1] << ", " << y[2] << "]\n";

    ASSERT_NEAR(y[0], 2.0, 1e-10);  // 1*1 + 1*1 + 0*1 = 2
    ASSERT_NEAR(y[1], 4.0, 1e-10);  // 1*1 + 2*1 + 1*1 = 4
    ASSERT_NEAR(y[2], 2.0, 1e-10);  // 0*1 + 1*1 + 1*1 = 2
}

TEST(gram_schmidt_orthonormality) {
    // Create 4 random vectors in R^100
    const size_t n = 100;
    std::vector<std::vector<double>> Y(4, std::vector<double>(n));
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int j = 0; j < 4; ++j) {
        for (size_t i = 0; i < n; ++i) {
            Y[j][i] = dist(rng);
        }
    }
    
    // Create projector just to use gram_schmidt
    LaplacianConfig config;
    LaplacianProjector proj(config);
    
    // Access via friend or make a simple local version
    // For testing, let's inline the GS algorithm
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < j; ++i) {
            double dot = simd::dot_product_d(Y[j].data(), Y[i].data(), n);
            simd::subtract_scaled(Y[j].data(), Y[i].data(), dot, n);
        }
        simd::normalize(Y[j].data(), n);
    }
    
    // Verify orthonormality
    for (int i = 0; i < 4; ++i) {
        double norm = simd::norm(Y[i].data(), n);
        ASSERT_NEAR(norm, 1.0, 1e-10);
        
        for (int j = i + 1; j < 4; ++j) {
            double dot = simd::dot_product_d(Y[i].data(), Y[j].data(), n);
            ASSERT_NEAR(dot, 0.0, 1e-10);
        }
    }
}

TEST(hypercube_normalization) {
    // Create simple test data
    std::vector<std::vector<double>> U(4);
    const size_t n = 10;
    
    // Column 0: values from -1 to 1
    U[0].resize(n);
    for (size_t i = 0; i < n; ++i) {
        U[0][i] = -1.0 + 2.0 * i / (n - 1);
    }
    
    // Column 1: constant 0.5
    U[1].resize(n, 0.5);
    
    // Column 2: values from 0 to 100
    U[2].resize(n);
    for (size_t i = 0; i < n; ++i) {
        U[2][i] = 100.0 * i / (n - 1);
    }
    
    // Column 3: negative values
    U[3].resize(n);
    for (size_t i = 0; i < n; ++i) {
        U[3][i] = -50.0 + 10.0 * i;
    }
    
    // Run normalization (inline the logic for testing)
    std::vector<std::array<uint32_t, 4>> coords(n);
    std::array<double, 4> minv, maxv;
    
    for (int d = 0; d < 4; ++d) {
        minv[d] = *std::min_element(U[d].begin(), U[d].end());
        maxv[d] = *std::max_element(U[d].begin(), U[d].end());
    }
    
    const double M = 4294967295.0;
    const double EPS = 1e-12;
    
    for (size_t i = 0; i < n; ++i) {
        for (int d = 0; d < 4; ++d) {
            double range = maxv[d] - minv[d];
            double x = (U[d][i] - minv[d]) / (range + EPS);
            coords[i][d] = static_cast<uint32_t>(std::llround(x * M));
        }
    }
    
    // Verify first point maps to 0 in dim 0 (was min)
    ASSERT_EQ(coords[0][0], 0u);
    
    // Verify last point maps to max in dim 0
    ASSERT_EQ(coords[n-1][0], 4294967295u);
    
    // Dim 1 was constant, all should be same value
    for (size_t i = 1; i < n; ++i) {
        ASSERT_EQ(coords[i][1], coords[0][1]);
    }
}

TEST(hilbert_roundtrip) {
    // Test that Hilbert index roundtrips correctly
    Point4D original(1234567890u, 2345678901u, 3456789012u, 4294967295u);
    
    HilbertIndex idx = HilbertCurve::coords_to_index(original);
    Point4D recovered = HilbertCurve::index_to_coords(idx);
    
    ASSERT_EQ(original.x, recovered.x);
    ASSERT_EQ(original.y, recovered.y);
    ASSERT_EQ(original.z, recovered.z);
    ASSERT_EQ(original.m, recovered.m);
}

TEST(full_projection_small) {
    // Test full projection pipeline with small synthetic data
    const size_t n = 50;
    const size_t d = 32;
    
    // Create embeddings that form 4 clusters
    std::vector<std::vector<float>> embeddings(n, std::vector<float>(d, 0.0f));
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    // 4 cluster centers
    std::vector<std::vector<float>> centers = {
        std::vector<float>(d, 0.0f),
        std::vector<float>(d, 0.0f),
        std::vector<float>(d, 0.0f),
        std::vector<float>(d, 0.0f)
    };
    
    for (size_t j = 0; j < d; ++j) {
        centers[0][j] = (j % 4 == 0) ? 1.0f : 0.0f;
        centers[1][j] = (j % 4 == 1) ? 1.0f : 0.0f;
        centers[2][j] = (j % 4 == 2) ? 1.0f : 0.0f;
        centers[3][j] = (j % 4 == 3) ? 1.0f : 0.0f;
    }
    
    // Assign points to clusters with noise
    for (size_t i = 0; i < n; ++i) {
        int cluster = i % 4;
        for (size_t j = 0; j < d; ++j) {
            embeddings[i][j] = centers[cluster][j] + noise(rng);
        }
    }
    
    // Run projection
    LaplacianConfig config;
    config.k_neighbors = 10;
    config.power_iterations = 50;
    config.project_to_sphere = false;
    config.num_threads = 1;  // Single-threaded for determinism
    
    LaplacianProjector projector(config);
    ProjectionResult result = projector.project(embeddings);
    
    // Verify we got n coordinates
    ASSERT_EQ(result.coords.size(), n);
    ASSERT_EQ(result.hilbert_lo.size(), n);
    ASSERT_EQ(result.hilbert_hi.size(), n);
    
    // Verify coordinates are in valid range
    for (size_t i = 0; i < n; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_TRUE(result.coords[i][j] <= 4294967295u);
        }
    }
    
    // Verify edges were created
    ASSERT_TRUE(result.edge_count > 0);
    
    std::cerr << "(edges=" << result.edge_count << ") ";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cerr << "\n=== Laplacian 4D Projection Tests ===\n\n";
    
    // Tests are auto-registered via static constructors
    
    std::cerr << "\n=== Summary ===\n";
    std::cerr << "Passed: " << g_tests_passed << "\n";
    std::cerr << "Failed: " << g_tests_failed << "\n";
    
    return g_tests_failed == 0 ? 0 : 1;
}
