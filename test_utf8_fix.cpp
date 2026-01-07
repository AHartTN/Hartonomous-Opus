#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "cpp/include/hypercube/util/utf8.hpp"

void test_valid_utf8() {
    std::cout << "Testing valid UTF-8..." << std::endl;

    // ASCII
    std::string ascii = "Hello";
    auto codepoints = hypercube::util::decode_utf8(ascii);
    assert(codepoints.size() == 5);
    assert(codepoints[0] == 'H');
    assert(codepoints[4] == 'o');

    // Multi-byte UTF-8
    std::string multibyte = "中"; // U+4E2D
    codepoints = hypercube::util::decode_utf8(multibyte);
    assert(codepoints.size() == 1);
    assert(codepoints[0] == 0x4E2D);

    // Emoji
    std::string emoji = "😀"; // U+1F600
    codepoints = hypercube::util::decode_utf8(emoji);
    assert(codepoints.size() == 1);
    assert(codepoints[0] == 0x1F600);

    std::cout << "  Valid UTF-8 tests passed" << std::endl;
}

void test_bom_handling() {
    std::cout << "Testing BOM handling..." << std::endl;

    // BOM + content
    std::string with_bom = "\xEF\xBB\xBFHello";
    auto codepoints = hypercube::util::decode_utf8(with_bom);
    assert(codepoints.size() == 5); // BOM should be stripped
    assert(codepoints[0] == 'H');

    // Just BOM
    std::string just_bom = "\xEF\xBB\xBF";
    codepoints = hypercube::util::decode_utf8(just_bom);
    assert(codepoints.empty()); // BOM stripped, nothing left

    std::cout << "  BOM handling tests passed" << std::endl;
}

void test_invalid_utf8() {
    std::cout << "Testing invalid UTF-8 handling..." << std::endl;

    // Continuation byte without start
    std::string invalid = "Hello\x80World";
    auto codepoints = hypercube::util::decode_utf8(invalid);
    // Should replace invalid byte with U+FFFD and continue
    assert(codepoints.size() == 11); // 5 + 1 (replacement) + 5

    // Overlong encoding
    std::string overlong = "\xC0\x80"; // Overlong null
    codepoints = hypercube::util::decode_utf8(overlong);
    assert(codepoints.size() == 1);
    assert(codepoints[0] == 0xFFFD); // Should be replacement char

    // Surrogate codepoint
    std::string surrogate = "\xED\xA0\x80"; // U+D800
    codepoints = hypercube::util::decode_utf8(surrogate);
    assert(codepoints.size() == 1);
    assert(codepoints[0] == 0xFFFD); // Should be replacement char

    std::cout << "  Invalid UTF-8 handling tests passed" << std::endl;
}

void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;

    // Empty string
    std::string empty = "";
    auto codepoints = hypercube::util::decode_utf8(empty);
    assert(codepoints.empty());

    // Maximum valid codepoint
    std::string max_cp = "\xF4\x8F\xBF\xBF"; // U+10FFFF
    codepoints = hypercube::util::decode_utf8(max_cp);
    assert(codepoints.size() == 1);
    assert(codepoints[0] == 0x10FFFF);

    // Over maximum
    std::string over_max = "\xF4\x90\x80\x80"; // U+110000 (invalid)
    codepoints = hypercube::util::decode_utf8(over_max);
    assert(codepoints.size() == 1);
    assert(codepoints[0] == 0xFFFD);

    std::cout << "  Edge case tests passed" << std::endl;
}

void test_roundtrip() {
    std::cout << "Testing roundtrip encoding/decoding..." << std::endl;

    std::vector<uint32_t> test_codepoints = {'H', 'e', 0x4E2D, 0x1F600, 0x10FFFF};

    for (uint32_t cp : test_codepoints) {
        std::string encoded = hypercube::util::encode_utf8(cp);
        auto decoded = hypercube::util::decode_utf8(encoded);
        assert(decoded.size() == 1);
        assert(decoded[0] == cp);
    }

    std::cout << "  Roundtrip tests passed" << std::endl;
}

int main() {
    std::cout << "=== UTF-8 Fix Verification Test ===" << std::endl;

    try {
        test_valid_utf8();
        test_bom_handling();
        test_invalid_utf8();
        test_edge_cases();
        test_roundtrip();

        std::cout << "\nAll UTF-8 fix tests PASSED!" << std::endl;
        std::cout << "The fixes properly handle:" << std::endl;
        std::cout << "  - Valid UTF-8 sequences" << std::endl;
        std::cout << "  - BOM detection and stripping" << std::endl;
        std::cout << "  - Invalid UTF-8 with proper error recovery" << std::endl;
        std::cout << "  - Overlong and surrogate detection" << std::endl;
        std::cout << "  - Codepoint range validation" << std::endl;
        std::cout << "  - Roundtrip encoding/decoding" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}