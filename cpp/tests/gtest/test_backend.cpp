// =============================================================================
// Backend Detection Tests
// =============================================================================

#include <gtest/gtest.h>
#include "hypercube/backend.hpp"

using namespace hypercube;

class BackendTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test CPU detection runs without crashing
TEST_F(BackendTest, CPUDetection) {
    auto info = Backend::detect();
    
    // Should have a vendor string
    EXPECT_FALSE(info.cpu.vendor.empty());
    
    // Should have a brand string (or at least not crash)
    // Brand can be empty on some VMs
    
    // Should detect at least SSE2 on any modern CPU
    EXPECT_TRUE(info.cpu.has_sse2);
}

// Test SIMD level detection
TEST_F(BackendTest, SIMDDetection) {
    auto info = Backend::detect();
    
    // SIMD level should be valid
    EXPECT_GE(static_cast<int>(info.simd_level), 0);
    EXPECT_LE(static_cast<int>(info.simd_level), 5);
    
    // Width should match level
    int width = simd_width(info.simd_level);
    EXPECT_GE(width, 1);
    EXPECT_LE(width, 16);
}

// Test singleton caching
TEST_F(BackendTest, SingletonCaching) {
    const auto& info1 = Backend::info();
    const auto& info2 = Backend::info();
    
    // Should return same cached instance
    EXPECT_EQ(&info1, &info2);
}

// Test convenience accessors
TEST_F(BackendTest, ConvenienceAccessors) {
    auto level = Backend::simd_level();
    auto eigen = Backend::eigensolver();
    auto knn = Backend::knn();
    
    // Just verify they don't crash
    EXPECT_GE(static_cast<int>(level), 0);
    EXPECT_GE(static_cast<int>(eigen), 0);
    EXPECT_GE(static_cast<int>(knn), 0);
}

// Test summary generation
TEST_F(BackendTest, SummaryGeneration) {
    auto info = Backend::detect();
    std::string summary = info.summary();
    
    // Summary should contain key information
    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("SIMD"), std::string::npos);
    EXPECT_NE(summary.find("Eigensolver"), std::string::npos);
}

// Test backend selection logic
TEST_F(BackendTest, BackendSelection) {
    auto info = Backend::detect();
    
    // If MKL is available, it should be selected
    if (info.has_mkl) {
        EXPECT_EQ(info.eigensolver, EigensolverBackend::MKL);
    }
    // If Eigen is available (and not MKL), it should be selected
    else if (info.has_eigen) {
        EXPECT_EQ(info.eigensolver, EigensolverBackend::Eigen);
    }
    // Otherwise Jacobi
    else {
        EXPECT_EQ(info.eigensolver, EigensolverBackend::Jacobi);
    }
    
    // Similar for k-NN
    if (info.has_faiss) {
        EXPECT_EQ(info.knn, KNNBackend::FAISS);
    } else if (info.has_hnswlib) {
        EXPECT_EQ(info.knn, KNNBackend::HNSWLIB);
    } else {
        EXPECT_EQ(info.knn, KNNBackend::BruteForce);
    }
}

// Test AVX detection on modern Intel CPUs
TEST_F(BackendTest, AVXDetectionIntel) {
    auto info = Backend::detect();
    
    // If this is an Intel CPU, check AVX features
    if (info.cpu.vendor == "GenuineIntel") {
        // Modern Intel should have at least AVX2
        // (This test may fail on very old CPUs)
        if (info.cpu.family >= 6 && info.cpu.model >= 60) {
            EXPECT_TRUE(info.cpu.has_avx2) << "Modern Intel CPU should have AVX2";
        }
    }
}

// Test enum name functions
TEST_F(BackendTest, EnumNames) {
    EXPECT_STREQ(simd_level_name(SIMDLevel::Scalar), "Scalar");
    EXPECT_STREQ(simd_level_name(SIMDLevel::AVX512), "AVX-512");
    
    EXPECT_STREQ(eigensolver_name(EigensolverBackend::Jacobi), "Jacobi (custom)");
    EXPECT_STREQ(eigensolver_name(EigensolverBackend::MKL), "Intel MKL DSYEVR");
    
    EXPECT_STREQ(knn_backend_name(KNNBackend::BruteForce), "Brute-force O(n²)");
    EXPECT_STREQ(knn_backend_name(KNNBackend::HNSWLIB), "HNSWLIB O(n log n)");
}

// Test SIMD width function
TEST_F(BackendTest, SIMDWidth) {
    EXPECT_EQ(simd_width(SIMDLevel::Scalar), 1);
    EXPECT_EQ(simd_width(SIMDLevel::SSE2), 4);
    EXPECT_EQ(simd_width(SIMDLevel::SSE4_2), 4);
    EXPECT_EQ(simd_width(SIMDLevel::AVX), 8);
    EXPECT_EQ(simd_width(SIMDLevel::AVX2), 8);
    EXPECT_EQ(simd_width(SIMDLevel::AVX512), 16);
}
