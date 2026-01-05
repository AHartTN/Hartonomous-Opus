// =============================================================================
// Coordinate Mapper Tests
// =============================================================================

#include <gtest/gtest.h>
#include "hypercube/coordinates.hpp"
#include <cmath>
#include <vector>
#include <set>

using namespace hypercube;

class CoordinatesTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test ASCII characters map to valid coordinates
TEST_F(CoordinatesTest, ASCIIMapping) {
    for (uint32_t cp = 'A'; cp <= 'Z'; ++cp) {
        Point4D pt = CoordinateMapper::map_codepoint(cp);
        
        // Coordinates should be valid (non-zero for most)
        // At least one coordinate should be non-zero
        bool has_nonzero = (pt.x != 0 || pt.y != 0 || pt.z != 0 || pt.m != 0);
        EXPECT_TRUE(has_nonzero) << "Codepoint " << cp << " should have non-zero coordinates";
    }
}

// Test uniqueness - different codepoints give different coordinates
TEST_F(CoordinatesTest, Uniqueness) {
    std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> seen;
    
    // Test ASCII range
    for (uint32_t cp = 0; cp < 128; ++cp) {
        Point4D pt = CoordinateMapper::map_codepoint(cp);
        auto key = std::make_tuple(pt.x, pt.y, pt.z, pt.m);
        
        EXPECT_TRUE(seen.find(key) == seen.end()) 
            << "Duplicate coordinates for codepoint " << cp;
        seen.insert(key);
    }
}

// Test centroid calculation
TEST_F(CoordinatesTest, CentroidCalculation) {
    std::vector<Point4D> points = {
        Point4D(100, 200, 300, 400),
        Point4D(200, 300, 400, 500),
        Point4D(300, 400, 500, 600),
    };
    
    Point4D centroid = CoordinateMapper::centroid(points);
    
    // Centroid should be average
    EXPECT_EQ(centroid.x, 200u);
    EXPECT_EQ(centroid.y, 300u);
    EXPECT_EQ(centroid.z, 400u);
    EXPECT_EQ(centroid.m, 500u);
}

// Test weighted centroid
TEST_F(CoordinatesTest, WeightedCentroid) {
    std::vector<Point4D> points = {
        Point4D(0, 0, 0, 0),
        Point4D(100, 100, 100, 100),
    };
    std::vector<double> weights = {1.0, 3.0};  // 25%/75% weighting
    
    Point4D wc = CoordinateMapper::weighted_centroid(points, weights);
    
    // Should be closer to second point
    EXPECT_EQ(wc.x, 75u);  // (0*1 + 100*3) / 4 = 75
}

// Test category detection
TEST_F(CoordinatesTest, CategoryDetection) {
    // Letters (using actual enum values from types.hpp)
    EXPECT_EQ(CoordinateMapper::categorize('A'), AtomCategory::LetterUpper);
    EXPECT_EQ(CoordinateMapper::categorize('a'), AtomCategory::LetterLower);
    
    // Digits
    EXPECT_EQ(CoordinateMapper::categorize('0'), AtomCategory::Digit);
    
    // Punctuation - '.' is PunctuationOther
    EXPECT_EQ(CoordinateMapper::categorize('.'), AtomCategory::PunctuationOther);
    
    // Space
    EXPECT_EQ(CoordinateMapper::categorize(' '), AtomCategory::Space);
    
    // Control characters
    EXPECT_EQ(CoordinateMapper::categorize('\n'), AtomCategory::Control);
}

// Test determinism - same codepoint always gives same coordinates
TEST_F(CoordinatesTest, Deterministic) {
    Point4D pt1 = CoordinateMapper::map_codepoint(0x0041);  // 'A'
    Point4D pt2 = CoordinateMapper::map_codepoint(0x0041);
    
    EXPECT_EQ(pt1.x, pt2.x);
    EXPECT_EQ(pt1.y, pt2.y);
    EXPECT_EQ(pt1.z, pt2.z);
    EXPECT_EQ(pt1.m, pt2.m);
}
