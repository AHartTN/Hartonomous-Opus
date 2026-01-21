#include <iostream>
#include <iomanip>
#include <cstdint>
#include "cpp/include/hypercube/types.hpp"
#include "cpp/include/hypercube/coordinates.hpp"

using namespace hypercube;

int main() {
    std::cout << "Testing 32-bit coordinate system..." << std::endl;

    // Test coordinate ranges
    std::cout << "UINT32_MAX = " << UINT32_MAX << std::endl;
    std::cout << "COORD_ORIGIN = " << constants::COORD_ORIGIN << std::endl;
    std::cout << "COORD_RADIUS = " << constants::COORD_RADIUS << std::endl;
    std::cout << "SURFACE_MAX = " << constants::SURFACE_MAX << std::endl;
    std::cout << "BITS_PER_DIM = " << constants::BITS_PER_DIM << std::endl;

    // Test Point4D size
    Point4D p(0x7FFFFFFFU, 0x7FFFFFFFU, 0x7FFFFFFFU, 0x7FFFFFFFU); // Near origin
    std::cout << "Point4D size: " << sizeof(p) << " bytes" << std::endl;
    std::cout << "Point4D values: (" << p.x << ", " << p.y << ", " << p.z << ", " << p.m << ")" << std::endl;

    // Test conversion to float
    Point4F pf(p);
    std::cout << "Point4F values: (" << std::fixed << std::setprecision(6)
              << pf.x << ", " << pf.y << ", " << pf.z << ", " << pf.m << ")" << std::endl;

    // Test quantization back
    Point4D p_back = pf.to_quantized();
    std::cout << "Quantized back: (" << p_back.x << ", " << p_back.y << ", " << p_back.z << ", " << p_back.m << ")" << std::endl;

    // Test coordinate mapping
    Point4D coord_a = CoordinateMapper::map_codepoint('A');
    Point4D coord_z = CoordinateMapper::map_codepoint('Z');
    Point4D coord_0 = CoordinateMapper::map_codepoint('0');

    std::cout << "Coordinate for 'A': (" << coord_a.x << ", " << coord_a.y << ", " << coord_a.z << ", " << coord_a.m << ")" << std::endl;
    std::cout << "Coordinate for 'Z': (" << coord_z.x << ", " << coord_z.y << ", " << coord_z.z << ", " << coord_z.m << ")" << std::endl;
    std::cout << "Coordinate for '0': (" << coord_0.x << ", " << coord_0.y << ", " << coord_0.z << ", " << coord_0.m << ")" << std::endl;

    // Check ranges
    bool in_range = (coord_a.x <= UINT32_MAX && coord_a.y <= UINT32_MAX &&
                     coord_a.z <= UINT32_MAX && coord_a.m <= UINT32_MAX);
    std::cout << "Coordinates in 32-bit range: " << (in_range ? "YES" : "NO") << std::endl;

    // Check if center is at UINT32_MAX/2
    uint32_t center = UINT32_MAX / 2;
    bool centered = (coord_a.x >= center - 1000000000 && coord_a.x <= center + 1000000000); // Allow some tolerance
    std::cout << "Coordinates appear centered: " << (centered ? "YES" : "NO") << std::endl;

    std::cout << "Test completed." << std::endl;
    return 0;
}