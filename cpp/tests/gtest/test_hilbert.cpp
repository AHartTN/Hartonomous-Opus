// =============================================================================
// Hilbert Curve Tests
// =============================================================================

#include <gtest/gtest.h>
#include "hypercube/hilbert.hpp"
#include <vector>
#include <set>
#include <algorithm>

using namespace hypercube;

class HilbertTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test origin produces a valid index and round-trips correctly
TEST_F(HilbertTest, OriginRoundTrip) {
    Point4D origin = {0, 0, 0, 0};
    HilbertIndex idx = HilbertCurve::coords_to_index(origin);
    Point4D recovered = HilbertCurve::index_to_coords(idx);
    
    // Round-trip should be exact
    EXPECT_EQ(recovered.x, origin.x);
    EXPECT_EQ(recovered.y, origin.y);
    EXPECT_EQ(recovered.z, origin.z);
    EXPECT_EQ(recovered.m, origin.m);
}

// Test round-trip: coords -> index -> coords
TEST_F(HilbertTest, RoundTrip) {
    std::vector<Point4D> test_points = {
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {1, 1, 1, 1},
        {100, 200, 300, 400},
        {65535, 65535, 65535, 65535},
    };
    
    for (const auto& pt : test_points) {
        HilbertIndex idx = HilbertCurve::coords_to_index(pt);
        Point4D recovered = HilbertCurve::index_to_coords(idx);
        
        EXPECT_EQ(recovered.x, pt.x) << "x mismatch for point (" << pt.x << "," << pt.y << "," << pt.z << "," << pt.m << ")";
        EXPECT_EQ(recovered.y, pt.y) << "y mismatch";
        EXPECT_EQ(recovered.z, pt.z) << "z mismatch";
        EXPECT_EQ(recovered.m, pt.m) << "m mismatch";
    }
}

// Test locality - nearby points should have similar indices
TEST_F(HilbertTest, LocalityPreservation) {
    Point4D p1 = {100, 100, 100, 100};
    Point4D p2 = {101, 100, 100, 100};  // Adjacent
    Point4D p3 = {200, 200, 200, 200};  // Far away
    
    HilbertIndex idx1 = HilbertCurve::coords_to_index(p1);
    HilbertIndex idx2 = HilbertCurve::coords_to_index(p2);
    HilbertIndex idx3 = HilbertCurve::coords_to_index(p3);
    
    // Adjacent points should have closer indices than distant points
    uint64_t diff_12 = (idx1.lo > idx2.lo) ? (idx1.lo - idx2.lo) : (idx2.lo - idx1.lo);
    uint64_t diff_13 = (idx1.lo > idx3.lo) ? (idx1.lo - idx3.lo) : (idx3.lo - idx1.lo);
    
    EXPECT_LT(diff_12, diff_13);
}

// Test uniqueness - different points give different indices
TEST_F(HilbertTest, Uniqueness) {
    std::set<std::pair<uint64_t, uint64_t>> seen;
    
    // Test a grid of points
    for (uint32_t x = 0; x < 16; ++x) {
        for (uint32_t y = 0; y < 16; ++y) {
            for (uint32_t z = 0; z < 4; ++z) {
                for (uint32_t m = 0; m < 4; ++m) {
                    Point4D pt = {x, y, z, m};
                    HilbertIndex idx = HilbertCurve::coords_to_index(pt);
                    
                    auto key = std::make_pair(idx.lo, idx.hi);
                    EXPECT_TRUE(seen.find(key) == seen.end()) 
                        << "Duplicate index for point (" << x << "," << y << "," << z << "," << m << ")";
                    seen.insert(key);
                }
            }
        }
    }
}

// Test monotonicity along one axis
TEST_F(HilbertTest, AxisCoverage) {
    // Hilbert curve is NOT monotonic along axes, but covers all indices
    std::vector<HilbertIndex> indices;
    
    for (uint32_t i = 0; i < 256; ++i) {
        Point4D pt = {i, 0, 0, 0};
        indices.push_back(HilbertCurve::coords_to_index(pt));
    }
    
    // All indices should be unique
    std::set<uint64_t> unique_lo;
    for (const auto& idx : indices) {
        unique_lo.insert(idx.lo);
    }
    EXPECT_EQ(unique_lo.size(), 256u);
}

// Test max coordinates
TEST_F(HilbertTest, MaxCoordinates) {
    Point4D max_pt = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    
    // Should not crash
    HilbertIndex idx = HilbertCurve::coords_to_index(max_pt);
    Point4D recovered = HilbertCurve::index_to_coords(idx);
    
    EXPECT_EQ(recovered.x, max_pt.x);
    EXPECT_EQ(recovered.y, max_pt.y);
    EXPECT_EQ(recovered.z, max_pt.z);
    EXPECT_EQ(recovered.m, max_pt.m);
}
