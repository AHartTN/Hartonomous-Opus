/**
 * Cosine Similarity Plugin
 * =========================
 *
 * Geometric operation plugin that computes cosine similarity in 4D space.
 * Treats 4D points as vectors and computes their angular similarity.
 * Particularly useful for semantic similarity where magnitude is less
 * important than direction.
 */

#include "hypercube/plugin.hpp"
#include "hypercube/types.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace hypercube {
namespace plugins {

class CosineSimilarityPlugin : public GeometricOperationPlugin {
public:
    std::string get_name() const override {
        return "cosine_similarity";
    }

    std::string get_version() const override {
        return "1.0.0";
    }

    std::string get_description() const override {
        return "Compute cosine similarity (angular distance) in 4D space";
    }

    bool initialize(const PluginContext& context) override {
        context_ = context;

        if (context_.log_info) {
            context_.log_info("[COSINE] Plugin initialized");
        }

        return true;
    }

    void shutdown() override {
        if (context_.log_info) {
            context_.log_info("[COSINE] Plugin shutdown");
        }
    }

    double compute_distance(const Point4D& a, const Point4D& b) const override {
        // Cosine distance = 1 - cosine similarity
        // cosine_sim = dot(a,b) / (||a|| * ||b||)

        double dot = a.x * b.x + a.y * b.y + a.z * b.z + a.m * b.m;

        double norm_a = std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z + a.m*a.m);
        double norm_b = std::sqrt(b.x*b.x + b.y*b.y + b.z*b.z + b.m*b.m);

        if (norm_a < 1e-10 || norm_b < 1e-10) {
            // Handle zero vectors
            return 1.0;  // Maximum distance
        }

        double cos_sim = dot / (norm_a * norm_b);

        // Clamp to [-1, 1] to handle numerical errors
        cos_sim = std::max(-1.0, std::min(1.0, cos_sim));

        // Convert to distance: 0 = same direction, 1 = opposite, 0.5 = orthogonal
        return (1.0 - cos_sim) / 2.0;
    }

    double compute_similarity(const Blake3Hash& composition_a,
                             const Blake3Hash& composition_b) const override {
        // Treat hash bytes as vector for cosine similarity
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;

        for (size_t i = 0; i < 32; ++i) {
            double va = static_cast<double>(composition_a.bytes[i]);
            double vb = static_cast<double>(composition_b.bytes[i]);

            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        if (norm_a < 1e-10 || norm_b < 1e-10) {
            return 0.0;
        }

        double cos_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));

        // Convert to [0, 1] similarity score
        return (cos_sim + 1.0) / 2.0;
    }

    std::vector<std::pair<Blake3Hash, double>>
    find_neighbors(const Point4D& query_point, size_t k) const override {
        if (context_.log_warning) {
            context_.log_warning("[COSINE] find_neighbors not yet implemented - requires DB access");
        }

        return {};
    }

private:
    PluginContext context_;
};

} // namespace plugins
} // namespace hypercube

REGISTER_PLUGIN(hypercube::plugins::CosineSimilarityPlugin)
