// =============================================================================
// Blake3 Hash Tests
// =============================================================================

#include <gtest/gtest.h>
#include "hypercube/blake3.hpp"
#include <vector>
#include <cstring>

using namespace hypercube;

class Blake3Test : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test empty input
TEST_F(Blake3Test, EmptyInput) {
    std::vector<uint8_t> empty;
    auto hash = Blake3Hasher::hash(std::span<const uint8_t>(empty));
    
    // Blake3 of empty input is well-defined
    EXPECT_EQ(hash.bytes.size(), 32);
    
    // Not all zeros
    bool all_zero = true;
    for (auto b : hash.bytes) {
        if (b != 0) { all_zero = false; break; }
    }
    EXPECT_FALSE(all_zero);
}

// Test determinism - same input gives same output
TEST_F(Blake3Test, Deterministic) {
    std::vector<uint8_t> data = {0x48, 0x65, 0x6c, 0x6c, 0x6f};  // "Hello"
    
    auto hash1 = Blake3Hasher::hash(std::span<const uint8_t>(data));
    auto hash2 = Blake3Hasher::hash(std::span<const uint8_t>(data));
    
    EXPECT_EQ(hash1.bytes, hash2.bytes);
}

// Test different inputs give different outputs
TEST_F(Blake3Test, DifferentInputsDifferentHashes) {
    std::vector<uint8_t> data1 = {0x48, 0x65, 0x6c, 0x6c, 0x6f};  // "Hello"
    std::vector<uint8_t> data2 = {0x57, 0x6f, 0x72, 0x6c, 0x64};  // "World"
    
    auto hash1 = Blake3Hasher::hash(std::span<const uint8_t>(data1));
    auto hash2 = Blake3Hasher::hash(std::span<const uint8_t>(data2));
    
    EXPECT_NE(hash1.bytes, hash2.bytes);
}

// Test avalanche effect - small change = totally different hash
TEST_F(Blake3Test, AvalancheEffect) {
    std::vector<uint8_t> data1 = {0x00, 0x00, 0x00, 0x00};
    std::vector<uint8_t> data2 = {0x00, 0x00, 0x00, 0x01};  // One bit different
    
    auto hash1 = Blake3Hasher::hash(std::span<const uint8_t>(data1));
    auto hash2 = Blake3Hasher::hash(std::span<const uint8_t>(data2));
    
    // Count differing bytes
    int diff_count = 0;
    for (size_t i = 0; i < 32; ++i) {
        if (hash1.bytes[i] != hash2.bytes[i]) diff_count++;
    }
    
    // Good hash should change most bytes
    EXPECT_GT(diff_count, 16);  // At least half should differ
}

// Test hex conversion
TEST_F(Blake3Test, HexConversion) {
    std::vector<uint8_t> data = {0x41};  // "A"
    auto hash = Blake3Hasher::hash(std::span<const uint8_t>(data));
    
    std::string hex = hash.to_hex();
    EXPECT_EQ(hex.length(), 64);  // 32 bytes = 64 hex chars
    
    // All characters should be valid hex
    for (char c : hex) {
        bool is_hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        EXPECT_TRUE(is_hex);
    }
}

// Test large input
TEST_F(Blake3Test, LargeInput) {
    std::vector<uint8_t> large(1024 * 1024);  // 1MB
    for (size_t i = 0; i < large.size(); ++i) {
        large[i] = static_cast<uint8_t>(i & 0xFF);
    }
    
    auto hash = Blake3Hasher::hash(std::span<const uint8_t>(large));
    EXPECT_EQ(hash.bytes.size(), 32);
}
