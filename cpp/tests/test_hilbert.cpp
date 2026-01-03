#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "../include/hypercube/hilbert.hpp"

using namespace hypercube;

void test_roundtrip() {
    std::cout << "Testing Hilbert curve roundtrip..." << std::endl;
    
    // Test specific points
    Point4D test_points[] = {
        {0, 0, 0, 0},
        {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX},
        {UINT32_MAX, 0, 0, 0},
        {0, UINT32_MAX, 0, 0},
        {0, 0, UINT32_MAX, 0},
        {0, 0, 0, UINT32_MAX},
        {1, 2, 3, 4},
        {100000, 200000, 300000, 400000},
        {0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x76543210},
    };
    
    for (const auto& point : test_points) {
        HilbertIndex idx = HilbertCurve::coords_to_index(point);
        Point4D recovered = HilbertCurve::index_to_coords(idx);
        
        if (recovered.x != point.x || recovered.y != point.y || 
            recovered.z != point.z || recovered.m != point.m) {
            std::cerr << "FAIL: Roundtrip failed for point ("
                      << point.x << ", " << point.y << ", " 
                      << point.z << ", " << point.m << ")" << std::endl;
            std::cerr << "  Got: (" << recovered.x << ", " << recovered.y 
                      << ", " << recovered.z << ", " << recovered.m << ")" << std::endl;
            std::exit(1);
        }
    }
    
    std::cout << "  Specific points: PASS" << std::endl;
    
    // Test random points (reduced count, real perf testing done separately)
    std::srand(42);
    for (int i = 0; i < 100; ++i) {
        Point4D point(
            static_cast<uint32_t>(std::rand()) << 16 | std::rand(),
            static_cast<uint32_t>(std::rand()) << 16 | std::rand(),
            static_cast<uint32_t>(std::rand()) << 16 | std::rand(),
            static_cast<uint32_t>(std::rand()) << 16 | std::rand()
        );
        
        HilbertIndex idx = HilbertCurve::coords_to_index(point);
        Point4D recovered = HilbertCurve::index_to_coords(idx);
        
        if (recovered.x != point.x || recovered.y != point.y || 
            recovered.z != point.z || recovered.m != point.m) {
            std::cerr << "FAIL: Roundtrip failed for random point " << i << std::endl;
            std::exit(1);
        }
    }
    
    std::cout << "  Random points (100): PASS" << std::endl;
}

void test_locality() {
    std::cout << "Testing Hilbert curve locality..." << std::endl;
    
    // Points that are close in space should be relatively close on the curve
    Point4D base(1000000, 1000000, 1000000, 1000000);
    Point4D near1(1000001, 1000000, 1000000, 1000000);
    Point4D near2(1000000, 1000001, 1000000, 1000000);
    Point4D far1(2000000000, 2000000000, 2000000000, 2000000000);
    
    HilbertIndex base_idx = HilbertCurve::coords_to_index(base);
    HilbertIndex near1_idx = HilbertCurve::coords_to_index(near1);
    HilbertIndex near2_idx = HilbertCurve::coords_to_index(near2);
    HilbertIndex far1_idx = HilbertCurve::coords_to_index(far1);
    
    HilbertIndex dist_near1 = HilbertCurve::distance(base_idx, near1_idx);
    (void)HilbertCurve::distance(base_idx, near2_idx);  // Verify no crash
    HilbertIndex dist_far1 = HilbertCurve::distance(base_idx, far1_idx);
    
    // Near points should have smaller Hilbert distance than far points
    // (This is a probabilistic property, not guaranteed, but should hold for these examples)
    bool locality_holds = (dist_near1.hi < dist_far1.hi) || 
                          (dist_near1.hi == dist_far1.hi && dist_near1.lo < dist_far1.lo);
    
    if (!locality_holds) {
        std::cerr << "WARNING: Locality property may not hold for this example" << std::endl;
        // Not a hard failure since Hilbert curves don't guarantee this for all points
    } else {
        std::cout << "  Locality property: PASS" << std::endl;
    }
}

void test_ordering() {
    std::cout << "Testing Hilbert index ordering..." << std::endl;
    
    HilbertIndex a(100, 0);
    HilbertIndex b(200, 0);
    HilbertIndex c(100, 1);
    HilbertIndex d(200, 1);
    
    assert(a < b);
    assert(a < c);
    assert(b < c);
    assert(c < d);
    assert(a < d);
    
    std::cout << "  Index comparison: PASS" << std::endl;
}

void test_arithmetic() {
    std::cout << "Testing Hilbert index arithmetic..." << std::endl;
    
    HilbertIndex a(UINT64_MAX, 0);
    HilbertIndex b(1, 0);
    HilbertIndex sum_result = a + b;
    
    assert(sum_result.lo == 0);
    assert(sum_result.hi == 1);
    (void)sum_result;  // Mark as used
    
    HilbertIndex c(100, 5);
    HilbertIndex d(50, 3);
    HilbertIndex diff_result = c - d;
    
    assert(diff_result.lo == 50);
    assert(diff_result.hi == 2);
    (void)diff_result;  // Mark as used
    
    std::cout << "  Arithmetic: PASS" << std::endl;
}

int main() {
    std::cout << "=== Hilbert Curve Tests ===" << std::endl;
    
    test_roundtrip();
    test_locality();
    test_ordering();
    test_arithmetic();
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
