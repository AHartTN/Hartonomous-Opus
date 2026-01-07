#include <iostream>
#include <iomanip>
#include "include/hypercube/hilbert.hpp"
#include "include/hypercube/coordinates.hpp"

int main() {
    hypercube::Point4D p(2147483648u, 2147483648u, 2147483648u, 2147483648u);
    hypercube::HilbertIndex idx = hypercube::HilbertCurve::coords_to_index(p);
    std::cout << "Hilbert index lo: " << idx.lo << " hi: " << idx.hi << std::endl;
    std::cout << "lo hex: " << std::hex << idx.lo << " hi hex: " << idx.hi << std::dec << std::endl;

    // Test max values
    hypercube::Point4D max_p(4294967295u, 4294967295u, 4294967295u, 4294967295u);
    hypercube::HilbertIndex max_idx = hypercube::HilbertCurve::coords_to_index(max_p);
    std::cout << "Max point - lo: " << max_idx.lo << " hi: " << max_idx.hi << std::endl;

    return 0;
}