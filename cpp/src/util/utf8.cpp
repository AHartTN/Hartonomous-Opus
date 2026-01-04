#include "hypercube/util/utf8.hpp"

namespace hypercube::util {

std::vector<uint32_t> decode_utf8(const std::string& data) {
    std::vector<uint32_t> codepoints;
    codepoints.reserve(data.size());
    
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    const uint8_t* end = p + data.size();
    
    while (p < end) {
        uint32_t cp;
        
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0 && p + 1 < end) {
            cp = (*p++ & 0x1F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF0) == 0xE0 && p + 2 < end) {
            cp = (*p++ & 0x0F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF8) == 0xF0 && p + 3 < end) {
            cp = (*p++ & 0x07) << 18;
            cp |= (*p++ & 0x3F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else {
            cp = 0xFFFD;  // Replacement character
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
