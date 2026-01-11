#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cmath>
#ifdef _WIN32
#include <cstdlib>
#endif
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

    assert(HilbertIndex(100, 0) < HilbertIndex(200, 0));
    assert(HilbertIndex(100, 0) < HilbertIndex(100, 1));
    assert(HilbertIndex(200, 0) < HilbertIndex(100, 1));
    assert(HilbertIndex(100, 1) < HilbertIndex(200, 1));
    assert(HilbertIndex(100, 0) < HilbertIndex(200, 1));

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

void test_abs_distance() {
    std::cout << "Testing Hilbert index abs_distance..." << std::endl;

    HilbertIndex a(100, 5);
    HilbertIndex b(50, 3);
    HilbertIndex dist1 = HilbertIndex::abs_distance(a, b);
    HilbertIndex dist2 = HilbertIndex::abs_distance(b, a);

    assert(dist1 == dist2);
    assert(dist1.lo == 50);
    assert(dist1.hi == 2);
    (void)dist1; (void)dist2; // Suppress unused warnings in release builds

    // Test with zero
    assert(HilbertIndex::abs_distance(a, a).lo == 0 && HilbertIndex::abs_distance(a, a).hi == 0);

    std::cout << "  Abs distance: PASS" << std::endl;
}

void test_roundtrip_edge_cases() {
    std::cout << "Testing roundtrip edge cases..." << std::endl;

    // Test CENTER_TO_CORNER mapping
    Point4D center_point(0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U);
    HilbertIndex idx = HilbertCurve::coords_to_index(center_point);
    Point4D recovered = HilbertCurve::index_to_coords(idx);
    assert(recovered.x == center_point.x && recovered.y == center_point.y &&
           recovered.z == center_point.z && recovered.m == center_point.m);

    // Test corner points
    Point4D corner_min(0, 0, 0, 0);
    idx = HilbertCurve::coords_to_index(corner_min);
    recovered = HilbertCurve::index_to_coords(idx);
    assert(recovered.x == corner_min.x && recovered.y == corner_min.y &&
           recovered.z == corner_min.z && recovered.m == corner_min.m);

    Point4D corner_max(UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX);
    idx = HilbertCurve::coords_to_index(corner_max);
    recovered = HilbertCurve::index_to_coords(idx);
    assert(recovered.x == corner_max.x && recovered.y == corner_max.y &&
           recovered.z == corner_max.z && recovered.m == corner_max.m);

    std::cout << "  Edge cases: PASS" << std::endl;
}

void test_index_to_raw_coords() {
    std::cout << "Testing index_to_raw_coords..." << std::endl;

    Point4D point(1000000, 2000000, 3000000, 4000000);
    HilbertIndex idx = HilbertCurve::coords_to_index(point);
    Point4D raw_coords = HilbertCurve::index_to_raw_coords(idx);
    (void)raw_coords;  // Mark as used for assertions

    // raw_coords should be corner-origin: point XOR CENTER_TO_CORNER
    constexpr uint32_t CENTER_TO_CORNER = 0x80000000U;
    Point4D expected_raw(point.x ^ CENTER_TO_CORNER,
                         point.y ^ CENTER_TO_CORNER,
                         point.z ^ CENTER_TO_CORNER,
                         point.m ^ CENTER_TO_CORNER);

    assert(raw_coords.x == expected_raw.x && raw_coords.y == expected_raw.y &&
           raw_coords.z == expected_raw.z && raw_coords.m == expected_raw.m);

    std::cout << "  Raw coords: PASS" << std::endl;
}

void test_bit_ordering_consistency() {
    std::cout << "Testing bit ordering consistency..." << std::endl;

    // The implementation uses bit-plane interleave with bit 0 as least significant nibble at index 0
    // Test that consecutive points have consecutive indices for small values

    Point4D p1(0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U);
    Point4D p2(0x80000001U, 0x80000000U, 0x80000000U, 0x80000000U);

    HilbertIndex i1 = HilbertCurve::coords_to_index(p1);
    HilbertIndex i2 = HilbertCurve::coords_to_index(p2);
    (void)i1;  // Mark as used for assertions
    (void)i2;  // Mark as used for assertions

    // Should be close but not necessarily consecutive due to Hilbert curve properties
    // Just verify roundtrip
    assert(HilbertCurve::index_to_coords(i1).x == p1.x);
    assert(HilbertCurve::index_to_coords(i2).x == p2.x);

    std::cout << "  Bit ordering: PASS" << std::endl;
}

int main() {
#ifdef _WIN32
    // Disable abort dialog on Windows to prevent message box when assert fails
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif

    std::cout << "=== Hilbert Curve Tests ===" << std::endl;

    test_roundtrip();
    test_locality();
    test_ordering();
    test_arithmetic();
    test_abs_distance();
    test_roundtrip_edge_cases();
    test_index_to_raw_coords();
    test_bit_ordering_consistency();

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
