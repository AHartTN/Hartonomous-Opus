#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>
#include "../include/hypercube/blake3.hpp"

using namespace hypercube;

void test_basic_hashing() {
    std::cout << "Testing basic hashing..." << std::endl;
    
    // Empty string
    Blake3Hash empty_hash = Blake3Hasher::hash("");
    assert(!empty_hash.is_zero());
    
    // Known test vector (BLAKE3 official test)
    // Hash of empty input should be:
    // af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262
    std::string empty_hex = empty_hash.to_hex();
    assert(empty_hex == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");
    std::cout << "  Empty hash: PASS" << std::endl;
    
    // "hello" should hash to a specific value
    Blake3Hash hello_hash = Blake3Hasher::hash("hello");
    std::cout << "  hello hash: " << hello_hash.to_hex() << std::endl;
    assert(!hello_hash.is_zero());
    
    // Different inputs should produce different hashes
    Blake3Hash world_hash = Blake3Hasher::hash("world");
    assert(hello_hash != world_hash);
    (void)world_hash;  // Mark as used for release builds
    std::cout << "  Different inputs: PASS" << std::endl;
    
    // Same input should produce same hash
    Blake3Hash hello2_hash = Blake3Hasher::hash("hello");
    assert(hello_hash == hello2_hash);
    (void)hello2_hash;  // Mark as used for release builds
    std::cout << "  Deterministic: PASS" << std::endl;
}

void test_codepoint_hashing() {
    std::cout << "Testing codepoint hashing..." << std::endl;
    
    // ASCII characters
    Blake3Hash a_hash = Blake3Hasher::hash_codepoint('A');
    Blake3Hash b_hash = Blake3Hasher::hash_codepoint('B');
    assert(a_hash != b_hash);
    (void)a_hash; (void)b_hash;  // Mark as used
    
    // UTF-8 encoding verification
    auto utf8 = Blake3Hasher::encode_utf8('A');
    assert(utf8.size() == 1);
    assert(utf8[0] == 'A');
    
    // 2-byte UTF-8 (e.g., ñ = U+00F1)
    utf8 = Blake3Hasher::encode_utf8(0x00F1);
    assert(utf8.size() == 2);
    assert(utf8[0] == 0xC3);
    assert(utf8[1] == 0xB1);
    
    // 3-byte UTF-8 (e.g., 中 = U+4E2D)
    utf8 = Blake3Hasher::encode_utf8(0x4E2D);
    assert(utf8.size() == 3);
    
    // 4-byte UTF-8 (e.g., 😀 = U+1F600)
    utf8 = Blake3Hasher::encode_utf8(0x1F600);
    assert(utf8.size() == 4);
    
    std::cout << "  Codepoint hashing: PASS" << std::endl;
}

void test_merkle_hashing() {
    std::cout << "Testing Merkle DAG hashing..." << std::endl;
    
    Blake3Hash h1 = Blake3Hasher::hash("child1");
    Blake3Hash h2 = Blake3Hasher::hash("child2");
    Blake3Hash h3 = Blake3Hasher::hash("child3");
    
    std::vector<Blake3Hash> children = {h1, h2, h3};
    Blake3Hash parent = Blake3Hasher::hash_children(children);
    
    // Different order should produce different hash (for ordered version)
    std::vector<Blake3Hash> children_reordered = {h2, h1, h3};
    Blake3Hash parent_reordered = Blake3Hasher::hash_children(children_reordered);
    assert(parent != parent_reordered);
    (void)parent; (void)parent_reordered;  // Mark as used
    
    // Ordered hashing includes position
    Blake3Hash ordered1 = Blake3Hasher::hash_children_ordered(children);
    Blake3Hash ordered2 = Blake3Hasher::hash_children_ordered(children);
    assert(ordered1 == ordered2);
    
    Blake3Hash ordered_reordered = Blake3Hasher::hash_children_ordered(children_reordered);
    assert(ordered1 != ordered_reordered);
    (void)ordered1; (void)ordered2; (void)ordered_reordered;  // Mark as used
    
    std::cout << "  Merkle hashing: PASS" << std::endl;
}

void test_incremental_hashing() {
    std::cout << "Testing incremental hashing..." << std::endl;
    
    // Full hash
    Blake3Hash full_hash = Blake3Hasher::hash("hello world");
    
    // Incremental hash
    Blake3Hasher::Incremental inc;
    inc.update("hello");
    inc.update(" ");
    inc.update("world");
    Blake3Hash inc_hash = inc.finalize();
    
    assert(full_hash == inc_hash);
    (void)inc_hash;  // Mark as used
    std::cout << "  Incremental hashing: PASS" << std::endl;
    
    // Reset and reuse
    inc.reset();
    inc.update("different data");
    Blake3Hash reset_hash = inc.finalize();
    assert(reset_hash != full_hash);
    (void)full_hash; (void)reset_hash;  // Mark as used
    std::cout << "  Reset: PASS" << std::endl;
}

void test_hex_conversion() {
    std::cout << "Testing hex conversion..." << std::endl;

    Blake3Hash original = Blake3Hasher::hash("test");
    std::string hex = original.to_hex();

    assert(hex.length() == 64);

    Blake3Hash recovered = Blake3Hash::from_hex(hex);
    assert(original == recovered);
    (void)recovered;  // Mark as used

    std::cout << "  Hex conversion: PASS" << std::endl;
}

void test_keyed_hashing() {
    std::cout << "Testing keyed hashing..." << std::endl;
    
    uint8_t key[32] = {0};
    for (int i = 0; i < 32; ++i) key[i] = static_cast<uint8_t>(i);
    
    Blake3Hash keyed = Blake3Hasher::keyed_hash(
        std::span<const uint8_t>(key, 32),
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>("message"), 7)
    );
    
    Blake3Hash unkeyed = Blake3Hasher::hash("message");
    assert(keyed != unkeyed);
    (void)keyed; (void)unkeyed;  // Mark as used
    
    std::cout << "  Keyed hashing: PASS" << std::endl;
}

void test_key_derivation() {
    std::cout << "Testing key derivation..." << std::endl;
    
    uint8_t key_material[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    
    Blake3Hash derived1 = Blake3Hasher::derive_key("context1", 
        std::span<const uint8_t>(key_material, 16));
    Blake3Hash derived2 = Blake3Hasher::derive_key("context2",
        std::span<const uint8_t>(key_material, 16));
    
    // Different contexts should produce different keys
    assert(derived1 != derived2);
    (void)derived2;  // Mark as used
    
    // Same context and key material should produce same key
    Blake3Hash derived1_again = Blake3Hasher::derive_key("context1",
        std::span<const uint8_t>(key_material, 16));
    assert(derived1 == derived1_again);
    (void)derived1; (void)derived1_again;  // Mark as used
    
    std::cout << "  Key derivation: PASS" << std::endl;
}

int main() {
    std::cout << "=== BLAKE3 Tests ===" << std::endl;
    
    test_basic_hashing();
    test_codepoint_hashing();
    test_merkle_hashing();
    test_incremental_hashing();
    test_hex_conversion();
    test_keyed_hashing();
    test_key_derivation();
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
