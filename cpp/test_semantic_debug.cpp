#include <iostream>
#include <iomanip>
#include "hypercube/coordinates.hpp"

int main() {
    using namespace hypercube;
    
    CoordinateMapper mapper;
    
    std::cout << "Checking semantic ordering:\n";
    std::cout << std::hex << std::uppercase;
    
    for (char c : {'A', 'a', 'B', 'b', 'C'}) {
        auto mapping = mapper.map_codepoint_full(static_cast<uint32_t>(c));
        std::cout << "'" << c << "' (U+" << std::setw(4) << std::setfill('0') << static_cast<uint32_t>(c) << "): ";
        std::cout << "x=" << mapping.coords.x << " ";
        std::cout << "y=" << mapping.coords.y << " ";
        std::cout << "z=" << mapping.coords.z << " ";
        std::cout << "m=" << mapping.coords.m << "\n";
    }
    
    return 0;
}
