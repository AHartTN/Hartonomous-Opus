#include "hypercube/cpu_features.h"
#include <iostream>

int main() {
    std::cout << "Testing new CPU feature detection...\n";

    auto features = hypercube::cpu_features::detect_cpu_features();

    std::cout << "Vendor: " << features.vendor << "\n";
    std::cout << "AVX: " << (features.avx ? "YES" : "NO") << "\n";
    std::cout << "AVX2: " << (features.avx2 ? "YES" : "NO") << "\n";
    std::cout << "AVX512F: " << (features.avx512f ? "YES" : "NO") << "\n";
    std::cout << "AVX_VNNI: " << (features.avx_vnni ? "YES" : "NO") << "\n";
    std::cout << "AMX_TILE: " << (features.amx_tile ? "YES" : "NO") << "\n";

    return 0;
}