/**
 * Euclidean Distance Plugin
 * =========================
 *
 * Example geometric operation plugin that computes standard Euclidean
 * distance in 4D space. Demonstrates the plugin architecture.
 */

#include "hypercube/plugin.hpp"
#include "hypercube/types.hpp"
#include <cmath>
#include <string>
#include <vector>
#include <libpq-fe.h>

namespace hypercube {
namespace plugins {

class EuclideanDistancePlugin : public GeometricOperationPlugin {
public:
    // Plugin metadata
    std::string get_name() const override {
        return "euclidean_distance";
    }

    std::string get_version() const override {
        return "1.0.0";
    }

    std::string get_description() const override {
        return "Compute standard Euclidean distance in 4D space";
    }

    std::vector<std::string> get_dependencies() const override {
        return {};  // No dependencies
    }

    // Lifecycle methods
    bool initialize(const PluginContext& context) override {
        context_ = context;

        if (context_.log_info) {
            context_.log_info("[EUCLIDEAN] Plugin initialized");
        }

        return true;
    }

    void shutdown() override {
        if (context_.log_info) {
            context_.log_info("[EUCLIDEAN] Plugin shutdown");
        }
    }

    // Geometric operations
    double compute_distance(const Point4D& a, const Point4D& b) const override {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dz = a.z - b.z;
        double dw = a.m - b.m;  // m is the 4th dimension in Point4D

        return std::sqrt(dx*dx + dy*dy + dz*dz + dw*dw);
    }

    double compute_similarity(const Blake3Hash& composition_a,
                             const Blake3Hash& composition_b) const override {
        // Query database to get 4D coordinates for both compositions
        if (!context_.db_connection) {
            // Fall back to hash-based similarity if no DB connection
            int matching_bytes = 0;
            for (size_t i = 0; i < 32; ++i) {
                if (composition_a.bytes[i] == composition_b.bytes[i]) {
                    matching_bytes++;
                }
            }
            return static_cast<double>(matching_bytes) / 32.0;
        }

        PGconn* conn = static_cast<PGconn*>(context_.db_connection);

        // Query for both centroids
        std::string query =
            "SELECT "
            "  ST_X(c1.centroid) as x1, ST_Y(c1.centroid) as y1, "
            "  ST_Z(c1.centroid) as z1, ST_M(c1.centroid) as m1, "
            "  ST_X(c2.centroid) as x2, ST_Y(c2.centroid) as y2, "
            "  ST_Z(c2.centroid) as z2, ST_M(c2.centroid) as m2 "
            "FROM composition c1, composition c2 "
            "WHERE c1.id = $1 AND c2.id = $2";

        std::string hash_a = "\\x" + composition_a.to_hex();
        std::string hash_b = "\\x" + composition_b.to_hex();

        const char* params[2] = {hash_a.c_str(), hash_b.c_str()};

        PGresult* res = PQexecParams(conn, query.c_str(), 2, nullptr,
                                     params, nullptr, nullptr, 0);

        if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
            PQclear(res);
            return 0.0;  // No data found
        }

        // Extract coordinates
        Point4D p1, p2;
        p1.x = std::stod(PQgetvalue(res, 0, 0));
        p1.y = std::stod(PQgetvalue(res, 0, 1));
        p1.z = std::stod(PQgetvalue(res, 0, 2));
        p1.m = std::stod(PQgetvalue(res, 0, 3));

        p2.x = std::stod(PQgetvalue(res, 0, 4));
        p2.y = std::stod(PQgetvalue(res, 0, 5));
        p2.z = std::stod(PQgetvalue(res, 0, 6));
        p2.m = std::stod(PQgetvalue(res, 0, 7));

        PQclear(res);

        // Compute distance
        double dist = compute_distance(p1, p2);

        // Convert distance to similarity (inverse relationship)
        // Use exponential decay: similarity = exp(-distance)
        return std::exp(-dist);
    }

    std::vector<std::pair<Blake3Hash, double>>
    find_neighbors(const Point4D& query_point, size_t k) const override {
        if (!context_.db_connection) {
            if (context_.log_warning) {
                context_.log_warning("[EUCLIDEAN] find_neighbors requires database connection");
            }
            return {};
        }

        PGconn* conn = static_cast<PGconn*>(context_.db_connection);

        // Build query point geometry
        // ST_MakePoint(x, y, z, m) creates a 4D point
        std::string query =
            "SELECT id, "
            "  ST_Distance(centroid, ST_SetSRID(ST_MakePoint($1, $2, $3, $4), 0)) as dist "
            "FROM composition "
            "WHERE centroid IS NOT NULL "
            "ORDER BY dist ASC "
            "LIMIT $5";

        std::string x_str = std::to_string(query_point.x);
        std::string y_str = std::to_string(query_point.y);
        std::string z_str = std::to_string(query_point.z);
        std::string m_str = std::to_string(query_point.m);
        std::string k_str = std::to_string(k);

        const char* params[5] = {
            x_str.c_str(),
            y_str.c_str(),
            z_str.c_str(),
            m_str.c_str(),
            k_str.c_str()
        };

        PGresult* res = PQexecParams(conn, query.c_str(), 5, nullptr,
                                     params, nullptr, nullptr, 0);

        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            if (context_.log_warning) {
                std::string error = PQerrorMessage(conn);
                context_.log_warning("[EUCLIDEAN] Query failed: " + error);
            }
            PQclear(res);
            return {};
        }

        std::vector<std::pair<Blake3Hash, double>> neighbors;
        int n = PQntuples(res);
        neighbors.reserve(n);

        for (int i = 0; i < n; ++i) {
            // Parse hash (format: \xHEXSTRING)
            const char* hash_hex = PQgetvalue(res, i, 0);
            Blake3Hash hash = Blake3Hash::from_hex(hash_hex + 2);  // Skip \x

            double dist = std::stod(PQgetvalue(res, i, 1));

            neighbors.emplace_back(hash, dist);
        }

        PQclear(res);
        return neighbors;
    }

private:
    PluginContext context_;
};

} // namespace plugins
} // namespace hypercube

// Plugin factory function
REGISTER_PLUGIN(hypercube::plugins::EuclideanDistancePlugin)
