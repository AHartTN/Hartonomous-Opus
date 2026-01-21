#include <iostream>
#include <iomanip>
#include "hypercube/coordinates.hpp"
#include "hypercube/semantic_ordering.hpp"
#include "hypercube/types.hpp"

using namespace hypercube;

void print_atom(const char* label, uint32_t cp) {
    uint64_t key = SemanticOrdering::get_semantic_key(cp);
    uint32_t rank = SemanticOrdering::get_rank(cp);
    Point4D p_int = CoordinateMapper::map_codepoint(cp);
    Point4F p_float = CoordinateMapper::map_codepoint_float(cp);

    std::cout << "Char: " << label << " (U+" << std::hex << cp << std::dec << ")\n";
    std::cout << "  Semantic Key: 0x" << std::hex << key << std::dec << "\n";
    std::cout << "  Dense Rank:   " << rank << "\n";
    std::cout << "  Coords (Int): (" << p_int.x << ", " << p_int.y << ", " << p_int.z << ", " << p_int.m << ")\n";
    std::cout << "  Coords (Flt): (" << p_float.x << ", " << p_float.y << ", " << p_float.z << ", " << p_float.m << ")\n";
    
    double norm = std::sqrt(p_float.x*p_float.x + p_float.y*p_float.y + p_float.z*p_float.z + p_float.m*p_float.m);
    std::cout << "  Radius:       " << norm << "\n";
    std::cout << "----------------------------------------\n";
}

int main() {
    std::cout << "=== Hypercube Projection Diagnostic ===\n\n";

    // Initialization happens lazily
    std::cout << "Total Codepoints: " << SemanticOrdering::total_codepoints() << "\n\n";

    print_atom("A", 'A');
    print_atom("a", 'a');
    print_atom("B", 'B');
    print_atom("b", 'b');
    print_atom("0", '0');
    print_atom("1", '1');
    
    // Check adjacency
    uint32_t rA = SemanticOrdering::get_rank('A');
    uint32_t ra = SemanticOrdering::get_rank('a');
    std::cout << "Rank Delta (A -> a): " << (int)ra - (int)rA << " (Should be small)\n";

    return 0;
}
