#include "hypercube/util/utf8.hpp"
#include <iostream>

namespace hypercube::util {

std::vector<uint32_t> decode_utf8(const std::string& data) {
    std::vector<uint32_t> codepoints;
    codepoints.reserve(data.size());

    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* end = p + data.size();

    // Check for BOM and skip it
    if (data.size() >= 3 && p[0] == 0xEF && p[1] == 0xBB && p[2] == 0xBF) {
        std::cout << "UTF-8 decode: BOM detected and skipped" << std::endl;
        p += 3;
    }

    while (p < end) {
        uint32_t cp;

        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0 && p + 1 < end) {
            uint8_t b1 = *p++;
            uint8_t b2 = *p++;
            if ((b2 & 0xC0) != 0x80) {
                // Invalid continuation byte
                std::cerr << "UTF-8 decode: Invalid continuation byte 0x" << std::hex << (int)b2 << " after 0x" << (int)b1 << std::dec << std::endl;
                cp = 0xFFFD;
                p--; // Rewind to retry b2 as start byte
            } else {
                cp = (b1 & 0x1F) << 6;
                cp |= (b2 & 0x3F);
                if (cp < 0x80) {
                    // Overlong encoding
                    std::cerr << "UTF-8 decode: Overlong 2-byte encoding for cp=" << cp << std::endl;
                    cp = 0xFFFD;
                }
            }
        } else if ((*p & 0xF0) == 0xE0 && p + 2 < end) {
            uint8_t b1 = *p++;
            uint8_t b2 = *p++;
            uint8_t b3 = *p++;
            if ((b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) {
                // Invalid continuation bytes
                cp = 0xFFFD;
                p -= 2; // Rewind
            } else {
                cp = (b1 & 0x0F) << 12;
                cp |= (b2 & 0x3F) << 6;
                cp |= (b3 & 0x3F);
                if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) {
                    // Overlong or surrogate
                    std::cerr << "UTF-8 decode: Overlong or surrogate 3-byte encoding for cp=" << cp << std::endl;
                    cp = 0xFFFD;
                }
            }
        } else if ((*p & 0xF8) == 0xF0 && p + 3 < end) {
            uint8_t b1 = *p++;
            uint8_t b2 = *p++;
            uint8_t b3 = *p++;
            uint8_t b4 = *p++;
            if ((b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80 || (b4 & 0xC0) != 0x80) {
                // Invalid continuation bytes
                cp = 0xFFFD;
                p -= 3; // Rewind
            } else {
                cp = (b1 & 0x07) << 18;
                cp |= (b2 & 0x3F) << 12;
                cp |= (b3 & 0x3F) << 6;
                cp |= (b4 & 0x3F);
                if (cp < 0x10000 || cp > 0x10FFFF) {
                    // Overlong or out of range
                    std::cerr << "UTF-8 decode: Overlong or out-of-range 4-byte encoding for cp=" << cp << std::endl;
                    cp = 0xFFFD;
                }
            }
        } else {
            // Invalid start byte or insufficient bytes
            std::cerr << "UTF-8 decode: Invalid start byte 0x" << std::hex << (int)*p << " or insufficient bytes" << std::dec << std::endl;
            cp = 0xFFFD;
            ++p;
        }

        codepoints.push_back(cp);
    }

    return codepoints;
}

std::unordered_set<uint32_t> extract_unique_codepoints(const std::string& content) {
    std::unordered_set<uint32_t> unique;
    std::vector<uint32_t> codepoints = decode_utf8(content);
    for (uint32_t cp : codepoints) {
        unique.insert(cp);
    }
    return unique;
}

std::string encode_utf8(uint32_t cp) {
    std::string result;
    if (cp < 0x80) {
        result.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        result.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        result.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        result.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return result;
}

} // namespace hypercube::util
