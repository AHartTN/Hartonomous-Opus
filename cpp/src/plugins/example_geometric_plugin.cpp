/**
 * Example Geometric Operation Plugin
 * ==================================
 *
 * Demonstrates the plugin architecture with a custom distance metric
 * that emphasizes semantic clustering over raw Euclidean distance.
 */

#include "hypercube/plugin.hpp"
#include "hypercube/coordinates.hpp"
#include <cmath>
#include <algorithm>

namespace hypercube {
namespace plugins {

/**
 * Semantic Distance Plugin
 * ========================
 *
 * Custom distance metric that considers:
 * 1. Euclidean distance in 4D space
 * 2. Semantic clustering (points near semantic axes get bonus)
 * 3. Category-based weighting (some relationships are more important)
 */
class SemanticDistancePlugin : public GeometricOperationPlugin {
public:
    std::string get_name() const override { return "semantic_distance"; }
    std::string get_version() const override { return "1.0.0"; }
    std::string get_description() const override {
        return "Semantic-aware distance metric that considers geometric proximity and category relationships";
    }

    bool initialize(const PluginContext& context) override {
        // Load semantic category weights from config
        auto category_weights = context.config.find("semantic_category_weights");
        if (category_weights != context.config.end()) {
            // Parse comma-separated weights for semantic categories
            // Implementation would parse and store category weights
        }

        context.log_info("[SemanticDistance] Initialized semantic distance plugin");
        return true;
    }

    void shutdown() override {
        // Cleanup resources if needed
    }

    double compute_distance(const Point4D& a, const Point4D& b) const override {
        // Convert to floating point for calculations
        Point4F af(a), bf(b);

        // Base Euclidean distance
        double euclidean_dist = af.distance(bf);

        // Semantic clustering bonus: points near coordinate axes get closer
        double axis_bonus = calculate_axis_bonus(af, bf);

        // Apply semantic weighting
        double weighted_distance = euclidean_dist * (1.0 - axis_bonus * 0.3);

        return std::max(0.0, weighted_distance); // Ensure non-negative
    }

    double compute_similarity(const Blake3Hash& composition_a,
                            const Blake3Hash& composition_b) const override {
        // For similarity, we need to look up coordinates from the database
        // This is a placeholder - real implementation would query the database
        // or use cached geometric data

        // Placeholder similarity calculation
        // In real implementation, this would:
        // 1. Query coordinates for both compositions
        // 2. Compute semantic distance
        // 3. Convert to similarity score (1.0 / (1.0 + distance))

        return 0.5; // Placeholder
    }

    std::vector<std::pair<Blake3Hash, double>>
        find_neighbors(const Point4D& query_point, size_t k) const override {
        // This would perform a k-NN search using the semantic distance metric
        // Real implementation would use spatial indexing (Hilbert curve) with custom distance

        // Placeholder implementation
        std::vector<std::pair<Blake3Hash, double>> neighbors;

        // In real implementation:
        // 1. Use Hilbert range query to get candidate points
        // 2. Score candidates with semantic distance
        // 3. Return top-k results

        return neighbors;
    }

private:
    /**
     * Calculate bonus factor for points near semantic axes
     * Points near coordinate axes (representing pure semantic directions)
     * get a distance reduction bonus
     */
    double calculate_axis_bonus(const Point4F& a, const Point4F& b) const {
        // Calculate how close each point is to coordinate axes
        auto axis_proximity = [](const Point4F& p) -> double {
            double max_coord = std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z), std::abs(p.m)});
            double axis_alignment = 0.0;

            // Check alignment with each axis
            if (std::abs(p.x) == max_coord) axis_alignment += std::abs(p.x);
            if (std::abs(p.y) == max_coord) axis_alignment += std::abs(p.y);
            if (std::abs(p.z) == max_coord) axis_alignment += std::abs(p.z);
            if (std::abs(p.m) == max_coord) axis_alignment += std::abs(p.m);

            return axis_alignment;
        };

        double a_axis = axis_proximity(a);
        double b_axis = axis_proximity(b);

        // Bonus is average of both points' axis alignment
        return (a_axis + b_axis) * 0.5;
    }
};

/**
 * Cosine Distance Plugin
 * ======================
 *
 * Alternative distance metric using cosine similarity
 * Better for high-dimensional semantic spaces
 */
class CosineDistancePlugin : public GeometricOperationPlugin {
public:
    std::string get_name() const override { return "cosine_distance"; }
    std::string get_version() const override { return "1.0.0"; }
    std::string get_description() const override {
        return "Cosine distance metric for angular semantic relationships";
    }

    bool initialize(const PluginContext& context) override {
        context.log_info("[CosineDistance] Initialized cosine distance plugin");
        return true;
    }

    void shutdown() override {}

    double compute_distance(const Point4D& a, const Point4D& b) const override {
        Point4F af(a), bf(b);

        // Cosine distance = 1 - cosine_similarity
        double dot_product = af.dot(bf);
        double norm_a = af.dot(af); // Should be ~1.0 for normalized points
        double norm_b = bf.dot(bf); // Should be ~1.0 for normalized points

        double cosine_sim = dot_product / std::sqrt(norm_a * norm_b);
        double cosine_dist = 1.0 - cosine_sim;

        return std::max(0.0, std::min(2.0, cosine_dist)); // Clamp to [0, 2]
    }

    double compute_similarity(const Blake3Hash& composition_a,
                            const Blake3Hash& composition_b) const override {
        // Cosine similarity is direct for composition vectors
        return 1.0 - compute_distance_for_compositions(composition_a, composition_b);
    }

    std::vector<std::pair<Blake3Hash, double>>
        find_neighbors(const Point4D& query_point, size_t k) const override {
        // Use cosine distance for neighbor finding
        std::vector<std::pair<Blake3Hash, double>> neighbors;
        // Implementation would use vector similarity search
        return neighbors;
    }

private:
    double compute_distance_for_compositions(const Blake3Hash& a, const Blake3Hash& b) const {
        // Placeholder - real implementation would query composition vectors
        return 0.5;
    }
};

} // namespace plugins
} // namespace hypercube

// Plugin Factory Functions
PLUGIN_EXPORT std::unique_ptr<hypercube::Plugin> create_semantic_distance_plugin() {
    return std::make_unique<hypercube::plugins::SemanticDistancePlugin>();
}

PLUGIN_EXPORT std::unique_ptr<hypercube::Plugin> create_cosine_distance_plugin() {
    return std::make_unique<hypercube::plugins::CosineDistancePlugin>();
}