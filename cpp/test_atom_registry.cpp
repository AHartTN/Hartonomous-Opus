#include "hypercube/atom_registry.hpp"
#include "hypercube/coordinate_utilities.hpp"
#include "hypercube/coordinates.hpp"
#include <iostream>
#include <unordered_set>
#include <string>
#include <algorithm>

int main() {
    std::cout << "Building AtomRegistry..." << std::endl;
    auto reg = hypercube::AtomRegistry::build();

    std::cout << "Total atoms: " << reg.atoms.size() << std::endl;

    // Check for unique quantized coordinates (first 128 ASCII like the test)
    std::unordered_set<std::string> coord_set;
    std::vector<uint32_t> duplicates;

    for (uint32_t cp = 0; cp < 128; ++cp) {
        // Find the atom for this codepoint
        auto it = std::find_if(reg.atoms.begin(), reg.atoms.end(),
                              [cp](const hypercube::AtomEntry& a) { return a.codepoint == cp; });
        if (it == reg.atoms.end()) continue;

        // Create a string key from quantized coordinates
        auto quantized = it->base.to_quantized();
        std::string key = std::to_string(quantized.x) + "," +
                         std::to_string(quantized.y) + "," +
                         std::to_string(quantized.z) + "," +
                         std::to_string(quantized.m);

        if (coord_set.count(key)) {
            duplicates.push_back(cp);
        } else {
            coord_set.insert(key);
        }
    }

    std::cout << "Unique coordinates: " << coord_set.size() << std::endl;
    std::cout << "Duplicates found: " << duplicates.size() << std::endl;

    if (!duplicates.empty()) {
        std::cout << "Duplicate codepoints: ";
        for (size_t i = 0; i < std::min(size_t(10), duplicates.size()); ++i) {
            std::cout << duplicates[i] << " ";
        }
        std::cout << (duplicates.size() > 10 ? "..." : "") << std::endl;
    }

    return duplicates.empty() ? 0 : 1;
}