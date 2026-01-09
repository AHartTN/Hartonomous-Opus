/**
 * Manhattan Distance Plugin
 * ==========================
 *
 * Geometric operation plugin that computes Manhattan (L1) distance
 * in 4D space. Useful for certain semantic similarity metrics where
 * axis-aligned movement is more meaningful than diagonal movement.
 */

#include "hypercube/plugin.hpp"
#include "hypercube/types.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace hypercube {
namespace plugins {

class ManhattanDistancePlugin : public GeometricOperationPlugin {
public:
    std::string get_name() const override {
        return "manhattan_distance";
    }

    std::string get_version() const override {
        return "1.0.0";
    }

    std::string get_description() const override {
        return "Compute Manhattan (L1) distance in 4D space - sum of absolute differences";
    }

    bool initialize(const PluginContext& context) override {
        context_ = context;

        if (context_.log_info) {
            context_.log_info("[MANHATTAN] Plugin initialized");
        }

        return true;
    }

    void shutdown() override {
        if (context_.log_info) {
            context_.log_info("[MANHATTAN] Plugin shutdown");
        }
    }

    double compute_distance(const Point4D& a, const Point4D& b) const override {
        // L1 distance: sum of absolute differences
        double dx = std::abs(a.x - b.x);
        double dy = std::abs(a.y - b.y);
        double dz = std::abs(a.z - b.z);
        double dw = std::abs(a.m - b.m);

        return dx + dy + dz + dw;
    }

    double compute_similarity(const Blake3Hash& composition_a,
                             const Blake3Hash& composition_b) const override {
        // Manhattan distance on hash values (for demonstration)
        int total_difference = 0;
        for (size_t i = 0; i < 32; ++i) {
            total_difference += std::abs(
                static_cast<int>(composition_a.bytes[i]) -
                static_cast<int>(composition_b.bytes[i])
            );
        }

        // Normalize to [0, 1] (max difference is 32*255)
        double normalized = 1.0 - (static_cast<double>(total_difference) / (32.0 * 255.0));
        return std::max(0.0, normalized);
    }

    std::vector<std::pair<Blake3Hash, double>>
    find_neighbors(const Point4D& query_point, size_t k) const override {
        if (context_.log_warning) {
            context_.log_warning("[MANHATTAN] find_neighbors not yet implemented - requires DB access");
        }

        return {};
    }

private:
    PluginContext context_;
};

} // namespace plugins
} // namespace hypercube

REGISTER_PLUGIN(hypercube::plugins::ManhattanDistancePlugin)
