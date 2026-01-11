#include "hypercube/cpu_features.hpp"
#include <iostream>

int main() {
    std::cout << "=== CPU Feature Detection Test ===\n\n";

    // Display CPU information
    std::cout << hypercube::cpu_features::get_cpu_info() << "\n\n";

    // Test individual features
    std::cout << "Individual Feature Checks:\n";
    std::cout << "AVX2: " << (hypercube::cpu_features::has_avx2() ? "YES" : "NO") << "\n";
    std::cout << "AVX512F: " << (hypercube::cpu_features::has_avx512f() ? "YES" : "NO") << "\n";
    std::cout << "FMA3: " << (hypercube::cpu_features::has_fma3() ? "YES" : "NO") << "\n";
    std::cout << "AVX_VNNI: " << (hypercube::cpu_features::has_avx_vnni() ? "YES" : "NO") << "\n\n";

    // Test feature mask
    uint32_t features = hypercube::cpu_features::get_supported_features();
    std::cout << "Feature Mask: 0x" << std::hex << features << std::dec << "\n";

    // Check if AVX_VNNI is detected
    if (features & static_cast<uint32_t>(hypercube::cpu_features::Feature::AVX_VNNI)) {
        std::cout << "✓ AVX_VNNI detected - ready for Int8 quantization acceleration!\n";
    } else {
        std::cout << "✗ AVX_VNNI not detected - will use AVX2 fallback\n";
    }

    return 0;
}