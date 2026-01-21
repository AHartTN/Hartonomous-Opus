#include "hypercube/semantic_ordering.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>

int main() {
    std::unordered_map<uint32_t, std::vector<uint32_t>> rank_to_cps;

    // Check first 128 ASCII for duplicate ranks
    for (uint32_t cp = 0; cp < 128; ++cp) {
        uint32_t rank = hypercube::SemanticOrdering::get_rank(cp);
        rank_to_cps[rank].push_back(cp);
    }

    int duplicates = 0;
    for (const auto& [rank, cps] : rank_to_cps) {
        if (cps.size() > 1) {
            duplicates++;
            std::cout << "Rank " << rank << " used by " << cps.size() << " codepoints: ";
            for (size_t i = 0; i < cps.size(); ++i) {
                std::cout << cps[i];
                if (i < cps.size() - 1) std::cout << ",";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Total ranks with duplicates: " << duplicates << std::endl;
    return duplicates > 0 ? 1 : 0;
}