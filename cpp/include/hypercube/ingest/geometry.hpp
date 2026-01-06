// =============================================================================
// geometry.hpp - EWKB Geometry Builders for PostGIS
// =============================================================================
// Functions to convert embeddings and coordinates to PostGIS geometry types.
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "hypercube/types.hpp"

namespace hypercube {
namespace ingest {

// =============================================================================
// Double to Little-Endian Hex
// =============================================================================

inline void write_double_hex(std::string& out, double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    char hex[17];
    for (int i = 0; i < 8; ++i) {
        snprintf(hex + i * 2, 3, "%02x", static_cast<unsigned>(bits & 0xFF));
        bits >>= 8;
    }
    out += hex;
}

// =============================================================================
// LINESTRINGZM from Float Array (Embeddings)
// =============================================================================
// Stores EVERY value - embeddings carry meaning in all dimensions
// X = dimension index, Y = value, Z = 0, M = 0

inline std::string floats_to_linestring_ewkb(const float* data, size_t count) {
    size_t num_points = count;
    if (num_points < 2) num_points = 2;  // LineString needs at least 2 points
    
    std::string ewkb;
    ewkb.reserve(26 + num_points * 64);
    
    // EWKB Header: little-endian, LINESTRINGZM with SRID, SRID=0
    ewkb += "01";           // Little-endian
    ewkb += "020000e0";     // LINESTRINGZM (3002) + SRID flag (0x20000000)
    ewkb += "00000000";     // SRID = 0
    
    // Number of points
    uint32_t n = static_cast<uint32_t>(num_points);
    char buf[9];
    snprintf(buf, sizeof(buf), "%02x%02x%02x%02x",
             n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF, (n >> 24) & 0xFF);
    ewkb += buf;
    
    // Write points
    for (size_t i = 0; i < count; ++i) {
        write_double_hex(ewkb, static_cast<double>(i));           // X = dimension index
        write_double_hex(ewkb, static_cast<double>(data[i]));     // Y = value
        write_double_hex(ewkb, 0.0);                              // Z = unused
        write_double_hex(ewkb, 0.0);                              // M = unused
    }
    
    // Pad if needed
    for (size_t i = count; i < num_points; ++i) {
        write_double_hex(ewkb, static_cast<double>(i));
        write_double_hex(ewkb, 0.0);
        write_double_hex(ewkb, 0.0);
        write_double_hex(ewkb, 0.0);
    }
    
    return ewkb;
}

// =============================================================================
// LINESTRINGZM from Point4D Vector (Composition Path)
// =============================================================================

inline std::string build_composition_linestringzm_ewkb(const std::vector<Point4D>& points) {
    if (points.size() < 2) return "";
    
    std::string ewkb;
    ewkb.reserve(26 + points.size() * 64);
    
    ewkb += "01";           // Little-endian
    ewkb += "020000e0";     // LINESTRINGZM with SRID flag
    ewkb += "00000000";     // SRID = 0
    
    uint32_t n = static_cast<uint32_t>(points.size());
    char buf[9];
    snprintf(buf, sizeof(buf), "%02x%02x%02x%02x",
             n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF, (n >> 24) & 0xFF);
    ewkb += buf;
    
    for (const auto& pt : points) {
        write_double_hex(ewkb, static_cast<double>(pt.x));
        write_double_hex(ewkb, static_cast<double>(pt.y));
        write_double_hex(ewkb, static_cast<double>(pt.z));
        write_double_hex(ewkb, static_cast<double>(pt.m));
    }
    
    return ewkb;
}

// =============================================================================
// POINTZM from Point4D (Composition Centroid)
// =============================================================================

inline std::string build_composition_pointzm_ewkb(const Point4D& pt) {
    std::string ewkb;
    ewkb.reserve(74);
    
    ewkb += "01";           // Little-endian
    ewkb += "010000e0";     // POINTZM with SRID flag
    ewkb += "00000000";     // SRID = 0
    
    write_double_hex(ewkb, static_cast<double>(pt.x));
    write_double_hex(ewkb, static_cast<double>(pt.y));
    write_double_hex(ewkb, static_cast<double>(pt.z));
    write_double_hex(ewkb, static_cast<double>(pt.m));
    
    return ewkb;
}

} // namespace ingest
} // namespace hypercube
