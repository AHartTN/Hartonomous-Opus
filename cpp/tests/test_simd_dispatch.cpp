// =============================================================================
// SIMD Dispatch System Test
// =============================================================================

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

#include "hypercube/cpu_features.h"
#include "hypercube/isa_class.h"
#include "hypercube/dispatch.h"
#include "hypercube/function_pointers.hpp"

using namespace hypercube;

namespace {

// Test data generator
std::vector<double> generate_random_vector_d(size_t n, double min_val = -1.0, double max_val = 1.0) {
    std::vector<double> v(n);
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (auto& val : v) {
        val = dist(gen);
    }
    return v;
}

std::vector<float> generate_random_vector_f(size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<float> v(n);
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& val : v) {
        val = dist(gen);
    }
    return v;
}

// Reference implementations
double ref_dot_product_d(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float ref_dot_product_f(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

} // anonymous namespace

int main() {
    std::cout << "=== SIMD Dispatch System Test ===\n\n";

    // Test CPU feature detection
    std::cout << "1. Testing CPU feature detection...\n";
    auto features = cpu_features::detect_cpu_features();

    std::cout << "   Vendor: " << features.vendor << "\n";
    std::cout << "   SSE4.2: " << (features.sse4_2 ? "yes" : "no") << "\n";
    std::cout << "   AVX:    " << (features.avx ? "yes" : "no") << "\n";
    std::cout << "   AVX2:   " << (features.avx2 ? "yes" : "no") << "\n";
    std::cout << "   FMA:    " << (features.fma ? "yes" : "no") << "\n";
    std::cout << "   AVX512F:" << (features.avx512f ? "yes" : "no") << "\n";
    std::cout << "   AVX-VNNI:" << (features.avx_vnni ? "yes" : "no") << "\n";
    std::cout << "   AVX512-VNNI:" << (features.avx512_vnni ? "yes" : "no") << "\n";
    std::cout << "\n";

    // Initialize function pointers for zero-overhead dispatch
    initialize_function_pointers();

    // Test ISA class detection
    std::cout << "2. Testing ISA class selection...\n";
    IsaClass isa = classify_isa(features);
    std::cout << "   Selected ISA: " << isa_class_name(isa) << "\n\n";

    // Test global function pointers
    std::cout << "3. Testing global function pointers...\n";

    // Generate test data
    constexpr size_t N = 1024;
    auto a_d = generate_random_vector_d(N);
    auto b_d = generate_random_vector_d(N);
    auto a_f = generate_random_vector_f(N);
    auto b_f = generate_random_vector_f(N);

    // Test double dot product
    double expected_d = ref_dot_product_d(a_d.data(), b_d.data(), N);
    double actual_d = dot_product_d(a_d.data(), b_d.data(), N);
    double error_d = std::abs(actual_d - expected_d);
    bool pass_d = error_d < 1e-10;
    std::cout << "   Double dot product: " << (pass_d ? "PASS" : "FAIL")
              << " (expected=" << expected_d << ", actual=" << actual_d
              << ", error=" << error_d << ")\n";

    // Test float dot product
    float expected_f = ref_dot_product_f(a_f.data(), b_f.data(), N);
    float actual_f = dot_product_f(a_f.data(), b_f.data(), N);
    float error_f = std::abs(actual_f - expected_f);
    bool pass_f = error_f < 1e-4f;
    std::cout << "   Float dot product:  " << (pass_f ? "PASS" : "FAIL")
              << " (expected=" << expected_f << ", actual=" << actual_f
              << ", error=" << error_f << ")\n";

    // Test scale in-place
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
    scale_inplace_d(v.data(), 2.0, v.size());
    bool pass_scale = (v[0] == 2.0 && v[1] == 4.0 && v[2] == 6.0 && v[3] == 8.0);
    std::cout << "   Scale in-place:     " << (pass_scale ? "PASS" : "FAIL") << "\n";

    // Test norm
    std::vector<double> unit = {3.0, 4.0}; // ||[3,4]|| = 5
    double norm_val = norm_d(unit.data(), unit.size());
    bool pass_norm = std::abs(norm_val - 5.0) < 1e-10;
    std::cout << "   Norm:               " << (pass_norm ? "PASS" : "FAIL")
              << " (expected=5.0, actual=" << norm_val << ")\n";

    std::cout << "\n=== Test Complete ===\n";

    int failures = (!pass_d) + (!pass_f) + (!pass_scale) + (!pass_norm);
    return failures;
}
