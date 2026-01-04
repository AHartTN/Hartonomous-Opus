#pragma once

#include "hypercube/types.hpp"
#include <vector>
#include <array>
#include <string>
#include <cstdint>

namespace hypercube::db {

// Build EWKB hex string for LINESTRINGZM geometry
// Points are stored as raw uint32 values (corner-origin) converted to double
std::string build_linestringzm_ewkb(const std::vector<std::array<int32_t, 4>>& points);

// Build EWKB hex string for POINTZM geometry
std::string build_pointzm_ewkb(int32_t x, int32_t y, int32_t z, int32_t m);

} // namespace hypercube::db
