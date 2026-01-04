#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>

namespace hypercube::util {

// Decode UTF-8 bytes to Unicode codepoints
std::vector<uint32_t> decode_utf8(const std::string& data);

// Extract unique codepoints from UTF-8 content
std::unordered_set<uint32_t> extract_unique_codepoints(const std::string& content);

// Encode codepoint to UTF-8 bytes
std::string encode_utf8(uint32_t codepoint);

} // namespace hypercube::util
