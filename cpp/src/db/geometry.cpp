#include "hypercube/db/geometry.hpp"
#include <cstring>

namespace hypercube::db {

namespace {
    constexpr char hex_chars[] = "0123456789abcdef";
    
    // Bit-preserving cast: int32 to uint32
    // NO OFFSET - the coordinate system has CENTER at 2^31 in uint32 space
    // Storing int32 as uint32 just reinterprets the bit pattern
    inline uint32_t int32_to_uint32(int32_t v) {
        return static_cast<uint32_t>(v);
    }
}

std::string build_linestringzm_ewkb(const std::vector<std::array<int32_t, 4>>& points) {
    std::string ewkb;
    
    auto append_double = [&](double val) {
        uint64_t bits;
        std::memcpy(&bits, &val, 8);
        for (int i = 0; i < 8; i++) {
            uint8_t byte = (bits >> (i * 8)) & 0xFF;
            ewkb += hex_chars[byte >> 4];
            ewkb += hex_chars[byte & 0x0F];
        }
    };
    
    auto append_uint32 = [&](uint32_t val) {
        for (int i = 0; i < 4; i++) {
            uint8_t byte = (val >> (i * 8)) & 0xFF;
            ewkb += hex_chars[byte >> 4];
            ewkb += hex_chars[byte & 0x0F];
        }
    };
    
    ewkb += "01";           // Little-endian
    ewkb += "020000c0";     // LINESTRINGZM type
    append_uint32(static_cast<uint32_t>(points.size()));
    
    for (const auto& p : points) {
        append_double(static_cast<double>(int32_to_uint32(p[0])));
        append_double(static_cast<double>(int32_to_uint32(p[1])));
        append_double(static_cast<double>(int32_to_uint32(p[2])));
        append_double(static_cast<double>(int32_to_uint32(p[3])));
    }
    
    return ewkb;
}

std::string build_pointzm_ewkb(int32_t x, int32_t y, int32_t z, int32_t m) {
    std::string ewkb;
    
    auto append_double = [&](double val) {
        uint64_t bits;
        std::memcpy(&bits, &val, 8);
        for (int i = 0; i < 8; i++) {
            uint8_t byte = (bits >> (i * 8)) & 0xFF;
            ewkb += hex_chars[byte >> 4];
            ewkb += hex_chars[byte & 0x0F];
        }
    };
    
    ewkb += "01";           // Little-endian
    ewkb += "010000c0";     // POINTZM type
    
    append_double(static_cast<double>(int32_to_uint32(x)));
    append_double(static_cast<double>(int32_to_uint32(y)));
    append_double(static_cast<double>(int32_to_uint32(z)));
    append_double(static_cast<double>(int32_to_uint32(m)));
    
    return ewkb;
}

} // namespace hypercube::db
