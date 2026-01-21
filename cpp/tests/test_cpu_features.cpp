#include "hypercube/cpu_features.hpp"
#include <gtest/gtest.h>
#include <iostream>

// Legacy standalone test
int main_standalone() {
    std::cout << "=== CPU Feature Detection Test ===\n\n";

    // Display CPU information
    std::cout << hypercube::cpu_features::get_cpu_info() << "\n\n";

    // Test individual features
    std::cout << "Individual Feature Checks:\n";
    std::cout << "AVX2: " << (hypercube::cpu_features::has_avx2() ? "YES" : "NO") << "\n";
    std::cout << "AVX512F: " << (hypercube::cpu_features::has_avx512f() ? "YES" : "NO") << "\n";
    std::cout << "FMA3: " << (hypercube::cpu_features::has_fma3() ? "YES" : "NO") << "\n";
    std::cout << "AVX_VNNI: " << (hypercube::cpu_features::has_avx_vnni() ? "YES" : "NO") << "\n\n";

    // Test feature mask
    uint32_t features = hypercube::cpu_features::get_supported_features();
    std::cout << "Feature Mask: 0x" << std::hex << features << std::dec << "\n";

    // Check if AVX_VNNI is detected
    if (features & static_cast<uint32_t>(hypercube::cpu_features::Feature::AVX_VNNI)) {
        std::cout << "✓ AVX_VNNI detected - ready for Int8 quantization acceleration!\n";
    } else {
        std::cout << "✗ AVX_VNNI not detected - will use AVX2 fallback\n";
    }

    return 0;
}

// Google Test cases
class CpuFeaturesTest : public ::testing::Test {
protected:
    hypercube::cpu_features::RuntimeCpuFeatures features;

    void SetUp() override {
        features = hypercube::cpu_features::detect_runtime_cpu_features();
    }
};

TEST_F(CpuFeaturesTest, DetectRuntimeCpuFeatures) {
    // Basic validation that detection doesn't crash
    EXPECT_FALSE(features.vendor.empty());

    // At minimum, we should have SSE2 on any modern x86-64 CPU
    EXPECT_TRUE(features.sse2);
    EXPECT_TRUE(features.sse);
}

TEST_F(CpuFeaturesTest, SafetyFlagsValidation) {
    // Safety flags should only be set if both CPU feature and OS support exist
    if (features.avx) {
        EXPECT_EQ(features.safe_avx_execution, features.os_avx_support);
    }

    if (features.avx512f) {
        EXPECT_EQ(features.safe_avx512_execution, features.os_avx512_support);
    }

    if (features.amx_tile) {
        EXPECT_EQ(features.safe_amx_execution, features.os_amx_support);
    }
}

TEST_F(CpuFeaturesTest, CapabilityBasedDispatch) {
    using namespace hypercube::cpu_features;

    // Test distance_l2 kernel selection
    auto pref_l2 = select_kernel_implementation(features, "distance_l2");
    EXPECT_NE(pref_l2.primary, KernelCapability::BASELINE); // Should select something better than baseline if possible

    // Test matrix_multiply kernel selection
    auto pref_gemm = select_kernel_implementation(features, "matrix_multiply");
    EXPECT_NE(pref_gemm.primary, KernelCapability::BASELINE);

    // Test unknown kernel type defaults
    auto pref_unknown = select_kernel_implementation(features, "unknown_kernel");
    EXPECT_TRUE(pref_unknown.primary == KernelCapability::AVX512F ||
               pref_unknown.primary == KernelCapability::AVX2 ||
               pref_unknown.primary == KernelCapability::SSE42 ||
               pref_unknown.primary == KernelCapability::BASELINE);
}

TEST_F(CpuFeaturesTest, FallbackHierarchy) {
    using namespace hypercube::cpu_features;

    auto pref = select_kernel_implementation(features, "distance_l2");

    // Emergency should always be BASELINE
    EXPECT_EQ(pref.emergency, KernelCapability::BASELINE);

    // Primary and fallback should be different unless both are BASELINE
    if (pref.primary != KernelCapability::BASELINE) {
        EXPECT_NE(pref.primary, pref.fallback);
    }
}

TEST_F(CpuFeaturesTest, CpuFeatureException) {
    using namespace hypercube::cpu_features;

    // Test exception construction
    CpuFeatureException ex("Test feature");
    EXPECT_STREQ(ex.what(), "CPU feature not available: Test feature");
}

TEST_F(CpuFeaturesTest, LegacyCompatibility) {
    // Test that runtime CPU features can be detected
    auto features = hypercube::cpu_features::detect_runtime_cpu_features();
    EXPECT_FALSE(features.vendor.empty());
}

// Test with mocked CPU features (limited capabilities)
TEST(CpuFeaturesMockTest, BaselineOnlySystem) {
    hypercube::cpu_features::RuntimeCpuFeatures mock_features{};
    mock_features.vendor = "MockCPU";
    mock_features.sse = true;
    mock_features.sse2 = true;
    // No AVX, AVX512, AMX

    auto pref = hypercube::cpu_features::select_kernel_implementation(mock_features, "distance_l2");

    // Should fall back to SSE42 or BASELINE
    EXPECT_TRUE(pref.primary == hypercube::cpu_features::KernelCapability::SSE42 ||
               pref.primary == hypercube::cpu_features::KernelCapability::BASELINE);
    EXPECT_EQ(pref.fallback, hypercube::cpu_features::KernelCapability::BASELINE);
}

int main(int argc, char** argv) {
    // Run standalone test if no arguments (for manual testing)
    if (argc == 1) {
        return main_standalone();
    }

    // Run Google Tests
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}