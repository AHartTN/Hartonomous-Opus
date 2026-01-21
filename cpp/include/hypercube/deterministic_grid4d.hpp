#pragma once

#include "hypercube/types.hpp"
#include "hypercube/atom_registry.hpp"
#include <vector>
#include <cstdint>
#include <array>

namespace hypercube {

/**
 * 4D grid cell coordinates (integer indices)
 */
struct CellCoord4D {
    int x, y, z, m;

    constexpr CellCoord4D() noexcept : x(0), y(0), z(0), m(0) {}
    constexpr CellCoord4D(int x_, int y_, int z_, int m_) noexcept
        : x(x_), y(y_), z(z_), m(m_) {}

    constexpr bool operator==(const CellCoord4D& other) const noexcept {
        return x == other.x && y == other.y && z == other.z && m == other.m;
    }

    constexpr bool operator<(const CellCoord4D& other) const noexcept {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        if (z != other.z) return z < other.z;
        return m < other.m;
    }
};

/**
 * Deterministic 4D spatial grid for accelerating neighbor searches.
 * Maps atoms to grid cells using explicit coordinate mapping with no floating-point state.
 */
class DeterministicGrid4D {
public:
    static constexpr int GRID_RESOLUTION = 16;
    static constexpr size_t TOTAL_CELLS = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION;

    DeterministicGrid4D() = default;

    /**
     * Build the grid deterministically from atom registry.
     * Maps each atom to a grid cell using floor((coord + 1) * GRID_RESOLUTION / 2).
     */
    void build(const AtomRegistry& reg);

    /**
     * Get all neighboring atoms for the given atom index.
     * Returns atoms in neighboring cells (including the atom's own cell) in lexicographic order.
     * The order is deterministic and fixed for a given grid resolution.
     */
    std::vector<size_t> enumerate_neighbors(size_t atom_idx) const;

    /**
     * Get the grid cell for a given atom index.
     */
    CellCoord4D get_cell_for_atom(size_t atom_idx) const;

    /**
     * Check if the grid has been built.
     */
    bool is_built() const { return !atom_cells_.empty(); }

    /**
     * Get total number of atoms in the grid.
     */
    size_t atom_count() const { return atom_cells_.size(); }

private:
    // Grid: vector of vectors, grid[cell_index] = list of atom indices in that cell
    std::vector<std::vector<size_t>> grid_;

    // For each atom, its cell coordinates (for quick lookup)
    std::vector<CellCoord4D> atom_cells_;

    // Convert 4D cell coordinates to linear index
    static constexpr size_t cell_index(int x, int y, int z, int m) noexcept {
        return ((size_t(x) * GRID_RESOLUTION + y) * GRID_RESOLUTION + z) * GRID_RESOLUTION + m;
    }

    // Convert linear index to 4D cell coordinates
    static constexpr CellCoord4D cell_coords(size_t idx) noexcept {
        int m = idx % GRID_RESOLUTION;
        idx /= GRID_RESOLUTION;
        int z = idx % GRID_RESOLUTION;
        idx /= GRID_RESOLUTION;
        int y = idx % GRID_RESOLUTION;
        idx /= GRID_RESOLUTION;
        int x = idx % GRID_RESOLUTION;
        return CellCoord4D(x, y, z, m);
    }

    // Map floating-point coordinate [-1,1] to grid cell [0, GRID_RESOLUTION-1]
    static constexpr int coord_to_cell(double coord) noexcept {
        double scaled = (coord + 1.0) * (GRID_RESOLUTION / 2.0);
        int cell = static_cast<int>(std::floor(scaled));
        // Clamp to valid range due to floating-point precision
        if (cell < 0) return 0;
        if (cell >= GRID_RESOLUTION) return GRID_RESOLUTION - 1;
        return cell;
    }

    // Check if cell coordinates are valid
    static constexpr bool is_valid_cell(int x, int y, int z, int m) noexcept {
        return x >= 0 && x < GRID_RESOLUTION &&
               y >= 0 && y < GRID_RESOLUTION &&
               z >= 0 && z < GRID_RESOLUTION &&
               m >= 0 && m < GRID_RESOLUTION;
    }
};

} // namespace hypercube