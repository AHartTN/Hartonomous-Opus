#include "hypercube/deterministic_grid4d.hpp"
#include <algorithm>
#include <cmath>

namespace hypercube {

void DeterministicGrid4D::build(const AtomRegistry& reg) {
    grid_.clear();
    grid_.resize(TOTAL_CELLS);
    atom_cells_.clear();
    atom_cells_.reserve(reg.atoms.size());

    for (size_t i = 0; i < reg.atoms.size(); ++i) {
        const auto& entry = reg.atoms[i];
        // Use jittered positions if available, otherwise base positions
        const Point4F& pos = (entry.jittered.x != 0.0 || entry.jittered.y != 0.0 ||
                             entry.jittered.z != 0.0 || entry.jittered.m != 0.0)
                           ? entry.jittered : entry.base;

        int cx = coord_to_cell(pos.x);
        int cy = coord_to_cell(pos.y);
        int cz = coord_to_cell(pos.z);
        int cm = coord_to_cell(pos.m);

        CellCoord4D cell(cx, cy, cz, cm);
        atom_cells_.push_back(cell);

        size_t cell_idx = cell_index(cx, cy, cz, cm);
        grid_[cell_idx].push_back(i);
    }
}

std::vector<size_t> DeterministicGrid4D::enumerate_neighbors(size_t atom_idx) const {
    if (atom_idx >= atom_cells_.size()) {
        return {};
    }

    const CellCoord4D& center = atom_cells_[atom_idx];
    std::vector<size_t> neighbors;

    // Enumerate all 3x3x3x3 = 81 neighboring cells in lexicographic order
    // Offsets from -1 to +1 in each dimension
    for (int dx = -1; dx <= 1; ++dx) {
        int nx = center.x + dx;
        if (nx < 0 || nx >= GRID_RESOLUTION) continue;

        for (int dy = -1; dy <= 1; ++dy) {
            int ny = center.y + dy;
            if (ny < 0 || ny >= GRID_RESOLUTION) continue;

            for (int dz = -1; dz <= 1; ++dz) {
                int nz = center.z + dz;
                if (nz < 0 || nz >= GRID_RESOLUTION) continue;

                for (int dm = -1; dm <= 1; ++dm) {
                    int nm = center.m + dm;
                    if (nm < 0 || nm >= GRID_RESOLUTION) continue;

                    // Add all atoms in this neighboring cell
                    size_t cell_idx = cell_index(nx, ny, nz, nm);
                    const auto& cell_atoms = grid_[cell_idx];
                    neighbors.insert(neighbors.end(), cell_atoms.begin(), cell_atoms.end());
                }
            }
        }
    }

    // Sort neighbors by atom index for deterministic order
    std::sort(neighbors.begin(), neighbors.end());

    // Remove duplicates (atoms might appear in multiple cells if on boundaries, but unlikely)
    auto last = std::unique(neighbors.begin(), neighbors.end());
    neighbors.erase(last, neighbors.end());

    return neighbors;
}

CellCoord4D DeterministicGrid4D::get_cell_for_atom(size_t atom_idx) const {
    if (atom_idx >= atom_cells_.size()) {
        return CellCoord4D();
    }
    return atom_cells_[atom_idx];
}

} // namespace hypercube