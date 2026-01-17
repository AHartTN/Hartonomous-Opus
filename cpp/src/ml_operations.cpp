/**
 * ML Operations Implementation
 * ============================
 *
 * Implementation of ML lifecycle management operations
 */

#include "hypercube/ml_operations.hpp"
#include "hypercube/db/helpers.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/ingest/safetensor.hpp"
#include "hypercube/ingest/parsing.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"
#include <libpq-fe.h>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <Eigen/Dense>
#include <unordered_map>
#include <filesystem>

namespace hypercube {
namespace ml {

// ============================================================================
// Model Cache for Inference
// ============================================================================

struct ModelWeights {
    Eigen::MatrixXf weight_matrix;  // Linear transformation matrix
    Eigen::VectorXf bias_vector;    // Optional bias vector
    bool has_bias = false;
};

class ModelCache {
public:
    static ModelCache& instance() {
        static ModelCache cache;
        return cache;
    }

    bool load_model(int64_t version_id, const std::string& artifact_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if already loaded
        if (models_.find(version_id) != models_.end()) {
            return true;
        }

        // Load from safetensors file
        ModelWeights weights;
        if (!load_safetensors_model(artifact_path, weights)) {
            return false;
        }

        models_[version_id] = std::move(weights);
        return true;
    }

    const ModelWeights* get_model(int64_t version_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = models_.find(version_id);
        return it != models_.end() ? &it->second : nullptr;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        models_.clear();
    }

private:
    ModelCache() = default;
    mutable std::mutex mutex_;
    std::unordered_map<int64_t, ModelWeights> models_;

    bool load_safetensors_model(const std::string& path, ModelWeights& weights) {
        namespace fs = std::filesystem;

        // Check if file exists
        if (!fs::exists(path)) {
            return false;
        }

        // Parse safetensors header using existing infrastructure
        hypercube::ingest::IngestContext ctx;
        if (!hypercube::ingest::parse_safetensor_header(ctx, path)) {
            return false;
        }

        // Look for weight matrix - expect "weight" tensor for linear transformation
        auto weight_meta = ctx.tensors.find("weight");
        if (weight_meta == ctx.tensors.end()) {
            return false;
        }

        const auto& meta = weight_meta->second;

        // Expect 2D matrix [output_dim, input_dim]
        if (meta.shape.size() != 2) {
            return false;
        }

        size_t output_dim = meta.shape[0];
        size_t input_dim = meta.shape[1];

        // For our 4D coordinate system, expect input_dim == 4
        if (input_dim != 4) {
            return false;
        }

        // Load the weight matrix
        auto mf = hypercube::safetensor::MappedFileCache::instance().get(meta.shard_file);
        if (!mf) {
            return false;
        }

        weights.weight_matrix.resize(output_dim, input_dim);

        // Read the matrix row by row
        for (size_t i = 0; i < output_dim; ++i) {
            auto row = hypercube::safetensor::read_tensor_row(mf, meta, i);
            if (row.size() != input_dim) {
                return false;
            }
            for (size_t j = 0; j < input_dim; ++j) {
                weights.weight_matrix(i, j) = row[j];
            }
        }

        // Try to load bias vector if present
        auto bias_meta = ctx.tensors.find("bias");
        if (bias_meta != ctx.tensors.end()) {
            const auto& bias_m = bias_meta->second;
            if (bias_m.shape.size() == 1 && static_cast<size_t>(bias_m.shape[0]) == output_dim) {
                weights.bias_vector.resize(output_dim);
                auto bias_row = hypercube::safetensor::read_tensor_row(mf, bias_m, 0);
                if (bias_row.size() == output_dim) {
                    for (size_t i = 0; i < output_dim; ++i) {
                        weights.bias_vector(i) = bias_row[i];
                    }
                    weights.has_bias = true;
                }
            }
        }

        return true;
    }
};

// ============================================================================
// Constructor / Destructor
// ============================================================================

MLOperations::MLOperations(PGconn* conn) : conn_(conn) {
    if (!conn_) {
        throw std::runtime_error("Null database connection provided to MLOperations");
    }
}

MLOperations::~MLOperations() = default;

// ============================================================================
// Helper Methods for JSON Conversion
// ============================================================================

std::string MLOperations::hyperparameters_to_json(const Hyperparameters& hp) {
    std::ostringstream oss;
    oss << "{"
        << "\"learning_rate\":" << hp.learning_rate << ","
        << "\"batch_size\":" << hp.batch_size << ","
        << "\"num_epochs\":" << hp.num_epochs << ","
        << "\"weight_decay\":" << hp.weight_decay << ","
        << "\"dropout_rate\":" << hp.dropout_rate << ","
        << "\"optimizer\":\"" << hp.optimizer << "\"";

    // Add custom parameters
    for (const auto& [key, value] : hp.custom) {
        oss << ",\"" << key << "\":\"" << value << "\"";
    }

    oss << "}";
    return oss.str();
}

std::string MLOperations::metrics_to_json(const TrainingMetrics& metrics) {
    std::ostringstream oss;
    oss << "{"
        << "\"loss\":" << metrics.loss << ","
        << "\"accuracy\":" << metrics.accuracy << ","
        << "\"precision\":" << metrics.precision << ","
        << "\"recall\":" << metrics.recall << ","
        << "\"f1_score\":" << metrics.f1_score;

    // Add custom metrics
    for (const auto& [key, value] : metrics.custom_metrics) {
        oss << ",\"" << key << "\":" << value;
    }

    oss << "}";
    return oss.str();
}

Hyperparameters MLOperations::json_to_hyperparameters(const std::string& json) {
    Hyperparameters hp;

    // Simple JSON parser for key numeric/string values
    // Format: {"learning_rate":0.001,"batch_size":32,...}

    auto parse_double = [](const std::string& s, const std::string& key) -> double {
        size_t pos = s.find("\"" + key + "\":");
        if (pos == std::string::npos) return 0.0;
        pos += key.length() + 3;
        size_t end = s.find_first_of(",}", pos);
        return std::stod(s.substr(pos, end - pos));
    };

    auto parse_int = [](const std::string& s, const std::string& key) -> int {
        size_t pos = s.find("\"" + key + "\":");
        if (pos == std::string::npos) return 0;
        pos += key.length() + 3;
        size_t end = s.find_first_of(",}", pos);
        return std::stoi(s.substr(pos, end - pos));
    };

    auto parse_string = [](const std::string& s, const std::string& key) -> std::string {
        size_t pos = s.find("\"" + key + "\":\"");
        if (pos == std::string::npos) return "";
        pos += key.length() + 4;
        size_t end = s.find("\"", pos);
        return s.substr(pos, end - pos);
    };

    try {
        hp.learning_rate = parse_double(json, "learning_rate");
        hp.batch_size = parse_int(json, "batch_size");
        hp.num_epochs = parse_int(json, "num_epochs");
        hp.weight_decay = parse_double(json, "weight_decay");
        hp.dropout_rate = parse_double(json, "dropout_rate");
        hp.optimizer = parse_string(json, "optimizer");
    } catch (...) {
        // Return default on parse error
    }

    return hp;
}

TrainingMetrics MLOperations::json_to_metrics(const std::string& json) {
    TrainingMetrics metrics;

    // Simple JSON parser for metric values
    auto parse_double = [](const std::string& s, const std::string& key) -> double {
        size_t pos = s.find("\"" + key + "\":");
        if (pos == std::string::npos) return 0.0;
        pos += key.length() + 3;
        size_t end = s.find_first_of(",}", pos);
        try {
            return std::stod(s.substr(pos, end - pos));
        } catch (...) {
            return 0.0;
        }
    };

    metrics.loss = parse_double(json, "loss");
    metrics.accuracy = parse_double(json, "accuracy");
    metrics.precision = parse_double(json, "precision");
    metrics.recall = parse_double(json, "recall");
    metrics.f1_score = parse_double(json, "f1_score");

    // Parse custom metrics (anything not in standard fields)
    size_t pos = 0;
    while ((pos = json.find("\"", pos)) != std::string::npos) {
        size_t key_end = json.find("\":", pos + 1);
        if (key_end == std::string::npos) break;

        std::string key = json.substr(pos + 1, key_end - pos - 1);

        // Skip standard fields
        if (key == "loss" || key == "accuracy" || key == "precision" ||
            key == "recall" || key == "f1_score") {
            pos = key_end + 2;
            continue;
        }

        // Parse custom metric value
        pos = key_end + 2;
        size_t val_end = json.find_first_of(",}", pos);
        if (val_end != std::string::npos) {
            try {
                double value = std::stod(json.substr(pos, val_end - pos));
                metrics.custom_metrics[key] = value;
            } catch (...) {}
        }
        pos = val_end;
    }

    return metrics;
}

// ============================================================================
// Experiment Management
// ============================================================================

int64_t MLOperations::create_experiment(
    const std::string& name,
    const std::string& description,
    const std::string& model_config,
    const std::string& training_config
) {
    std::string query =
        "SELECT create_ml_experiment($1, $2, $3::JSONB, $4::JSONB)";

    const char* params[4] = {
        name.c_str(),
        description.c_str(),
        model_config.c_str(),
        training_config.c_str()
    };

    PGresult* res = PQexecParams(conn_, query.c_str(), 4, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to create experiment: " + error);
    }

    int64_t experiment_id = std::stoll(PQgetvalue(res, 0, 0));
    PQclear(res);

    return experiment_id;
}

ExperimentInfo MLOperations::get_experiment(int64_t experiment_id) {
    std::string query =
        "SELECT id, name, description, status, "
        "       (SELECT COUNT(*) FROM ml_run WHERE experiment_id = $1), "
        "       created_at "
        "FROM ml_experiment WHERE id = $1";

    std::string id_str = std::to_string(experiment_id);
    const char* params[1] = {id_str.c_str()};

    PGresult* res = PQexecParams(conn_, query.c_str(), 1, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
        PQclear(res);
        throw std::runtime_error("Experiment not found: " + id_str);
    }

    ExperimentInfo info;
    info.experiment_id = experiment_id;
    info.name = PQgetvalue(res, 0, 1);
    info.description = PQgetvalue(res, 0, 2);
    info.status = PQgetvalue(res, 0, 3);
    info.total_runs = std::stoi(PQgetvalue(res, 0, 4));
    // info.created_at = parse timestamp from PQgetvalue(res, 0, 5)

    PQclear(res);
    return info;
}

std::vector<ExperimentInfo> MLOperations::list_experiments(bool active_only) {
    std::string query =
        "SELECT id, name, description, status, "
        "       (SELECT COUNT(*) FROM ml_run WHERE experiment_id = e.id), "
        "       created_at "
        "FROM ml_experiment e ";

    if (active_only) {
        query += "WHERE status = 'active' ";
    }

    query += "ORDER BY created_at DESC";

    PGresult* res = PQexec(conn_, query.c_str());

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to list experiments: " + error);
    }

    std::vector<ExperimentInfo> experiments;
    int n = PQntuples(res);
    experiments.reserve(n);

    for (int i = 0; i < n; ++i) {
        ExperimentInfo info;
        info.experiment_id = std::stoll(PQgetvalue(res, i, 0));
        info.name = PQgetvalue(res, i, 1);
        info.description = PQgetvalue(res, i, 2);
        info.status = PQgetvalue(res, i, 3);
        info.total_runs = std::stoi(PQgetvalue(res, i, 4));

        experiments.push_back(info);
    }

    PQclear(res);
    return experiments;
}

// ============================================================================
// Training Run Management
// ============================================================================

int64_t MLOperations::start_run(
    int64_t experiment_id,
    const std::string& run_name,
    const Hyperparameters& hyperparameters
) {
    std::string hp_json = hyperparameters_to_json(hyperparameters);
    std::string query = "SELECT start_ml_run($1, $2, $3::JSONB)";

    std::string exp_id_str = std::to_string(experiment_id);
    const char* params[3] = {
        exp_id_str.c_str(),
        run_name.empty() ? nullptr : run_name.c_str(),
        hp_json.c_str()
    };

    PGresult* res = PQexecParams(conn_, query.c_str(), 3, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to start run: " + error);
    }

    int64_t run_id = std::stoll(PQgetvalue(res, 0, 0));
    PQclear(res);

    return run_id;
}

void MLOperations::log_metric(
    int64_t run_id,
    const std::string& metric_name,
    double metric_value,
    int step,
    int epoch,
    const std::string& metric_type
) {
    std::string query = "SELECT log_ml_metric($1, $2, $3, $4, $5, $6)";

    std::string run_id_str = std::to_string(run_id);
    std::string value_str = std::to_string(metric_value);
    std::string step_str = std::to_string(step);
    std::string epoch_str = (epoch < 0) ? "" : std::to_string(epoch);

    const char* params[6] = {
        run_id_str.c_str(),
        metric_name.c_str(),
        value_str.c_str(),
        step_str.c_str(),
        epoch < 0 ? nullptr : epoch_str.c_str(),
        metric_type.c_str()
    };

    PGresult* res = PQexecParams(conn_, query.c_str(), 6, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to log metric: " + error);
    }

    PQclear(res);
}

void MLOperations::log_metrics(
    int64_t run_id,
    const std::unordered_map<std::string, double>& metrics,
    int step,
    int epoch
) {
    for (const auto& [name, value] : metrics) {
        log_metric(run_id, name, value, step, epoch);
    }
}

void MLOperations::complete_run(
    int64_t run_id,
    const TrainingMetrics& final_metrics,
    const std::string& status
) {
    std::string metrics_json = metrics_to_json(final_metrics);
    std::string query = "SELECT complete_ml_run($1, $2::JSONB, $3)";

    std::string run_id_str = std::to_string(run_id);
    const char* params[3] = {
        run_id_str.c_str(),
        metrics_json.c_str(),
        status.c_str()
    };

    PGresult* res = PQexecParams(conn_, query.c_str(), 3, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to complete run: " + error);
    }

    PQclear(res);
}

int64_t MLOperations::save_checkpoint(
    int64_t run_id,
    const std::string& checkpoint_path,
    int step,
    int epoch,
    const TrainingMetrics& metrics,
    bool is_best
) {
    std::string metrics_json = metrics_to_json(metrics);

    std::string query =
        "INSERT INTO ml_checkpoint (run_id, step, epoch, checkpoint_path, metrics, is_best) "
        "VALUES ($1, $2, $3, $4, $5::JSONB, $6) "
        "RETURNING id";

    std::string run_id_str = std::to_string(run_id);
    std::string step_str = std::to_string(step);
    std::string epoch_str = std::to_string(epoch);
    std::string is_best_str = is_best ? "true" : "false";

    const char* params[6] = {
        run_id_str.c_str(),
        step_str.c_str(),
        epoch_str.c_str(),
        checkpoint_path.c_str(),
        metrics_json.c_str(),
        is_best_str.c_str()
    };

    PGresult* res = PQexecParams(conn_, query.c_str(), 6, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to save checkpoint: " + error);
    }

    int64_t checkpoint_id = std::stoll(PQgetvalue(res, 0, 0));
    PQclear(res);

    return checkpoint_id;
}

TrainingRunInfo MLOperations::get_run(int64_t run_id) {
    std::string query =
        "SELECT id, experiment_id, run_name, status, hyperparameters, "
        "       metrics, start_time, end_time, duration_seconds "
        "FROM ml_run WHERE id = $1";

    std::string id_str = std::to_string(run_id);
    const char* params[1] = {id_str.c_str()};

    PGresult* res = PQexecParams(conn_, query.c_str(), 1, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
        PQclear(res);
        throw std::runtime_error("Run not found: " + id_str);
    }

    TrainingRunInfo info;
    info.run_id = run_id;
    info.experiment_id = std::stoll(PQgetvalue(res, 0, 1));
    info.run_name = PQgetvalue(res, 0, 2);
    info.status = PQgetvalue(res, 0, 3);
    // Parse hyperparameters and metrics JSON
    info.duration_seconds = std::stoi(PQgetvalue(res, 0, 8));

    PQclear(res);
    return info;
}

std::vector<TrainingRunInfo> MLOperations::get_leaderboard(
    int64_t experiment_id,
    const std::string& metric_name,
    int limit
) {
    std::string query = "SELECT * FROM get_experiment_leaderboard($1, $2, $3)";

    std::string exp_id_str = std::to_string(experiment_id);
    std::string limit_str = std::to_string(limit);

    const char* params[3] = {
        exp_id_str.c_str(),
        metric_name.c_str(),
        limit_str.c_str()
    };

    PGresult* res = PQexecParams(conn_, query.c_str(), 3, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to get leaderboard: " + error);
    }

    std::vector<TrainingRunInfo> leaderboard;
    int n = PQntuples(res);
    leaderboard.reserve(n);

    for (int i = 0; i < n; ++i) {
        TrainingRunInfo info;
        info.run_id = std::stoll(PQgetvalue(res, i, 0));
        info.run_name = PQgetvalue(res, i, 1);
        // Parse remaining fields

        leaderboard.push_back(info);
    }

    PQclear(res);
    return leaderboard;
}

// ============================================================================
// Model Versioning
// ============================================================================

int64_t MLOperations::register_model_version(
    const std::string& model_name,
    const std::string& version,
    const std::string& artifact_path,
    int64_t source_run_id,
    const TrainingMetrics& performance_metrics
) {
    std::string metrics_json = metrics_to_json(performance_metrics);
    std::string query = "SELECT register_ml_model_version($1, $2, $3, $4, $5::JSONB)";

    std::string run_id_str = (source_run_id < 0) ? "" : std::to_string(source_run_id);

    const char* params[5] = {
        model_name.c_str(),
        version.c_str(),
        artifact_path.c_str(),
        (source_run_id < 0) ? nullptr : run_id_str.c_str(),
        metrics_json.c_str()
    };

    PGresult* res = PQexecParams(conn_, query.c_str(), 5, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to register model version: " + error);
    }

    int64_t version_id = std::stoll(PQgetvalue(res, 0, 0));
    PQclear(res);

    return version_id;
}

ModelInfo MLOperations::get_model_version(int64_t version_id) {
    std::string query =
        "SELECT id, model_name, version, model_type, artifact_path, "
        "       validation_status, is_deployed, performance_metrics, created_at "
        "FROM ml_model_version WHERE id = $1";

    std::string id_str = std::to_string(version_id);
    const char* params[1] = {id_str.c_str()};

    PGresult* res = PQexecParams(conn_, query.c_str(), 1, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
        PQclear(res);
        throw std::runtime_error("Model version not found: " + id_str);
    }

    ModelInfo info;
    info.model_version_id = version_id;
    info.model_name = PQgetvalue(res, 0, 1);
    info.version = PQgetvalue(res, 0, 2);
    info.model_type = PQgetvalue(res, 0, 3);
    info.artifact_path = PQgetvalue(res, 0, 4);
    // Parse validation_status at index 5
    info.is_deployed = std::string(PQgetvalue(res, 0, 6)) == "t";

    // Parse performance metrics JSON
    std::string metrics_json = PQgetvalue(res, 0, 7);
    info.performance_metrics = json_to_metrics(metrics_json);

    // Parse created_at timestamp - simplified version
    // info.created_at = parse_timestamp(PQgetvalue(res, 0, 8));

    PQclear(res);
    return info;
}

std::vector<ModelInfo> MLOperations::list_model_versions(const std::string& model_name) {
    std::string query =
        "SELECT id, model_name, version, model_type, artifact_path, "
        "       validation_status, is_deployed, performance_metrics, created_at "
        "FROM ml_model_version WHERE model_name = $1 "
        "ORDER BY created_at DESC";

    const char* params[1] = {model_name.c_str()};

    PGresult* res = PQexecParams(conn_, query.c_str(), 1, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to list model versions: " + error);
    }

    std::vector<ModelInfo> versions;
    int n = PQntuples(res);
    versions.reserve(n);

    for (int i = 0; i < n; ++i) {
        ModelInfo info;
        info.model_version_id = std::stoll(PQgetvalue(res, i, 0));
        info.model_name = PQgetvalue(res, i, 1);
        info.version = PQgetvalue(res, i, 2);
        info.model_type = PQgetvalue(res, i, 3);
        info.artifact_path = PQgetvalue(res, i, 4);
        info.is_deployed = std::string(PQgetvalue(res, i, 6)) == "t";

        std::string metrics_json = PQgetvalue(res, i, 7);
        info.performance_metrics = json_to_metrics(metrics_json);

        versions.push_back(info);
    }

    PQclear(res);
    return versions;
}

void MLOperations::approve_model_version(int64_t version_id, const std::string& approval_notes) {
    std::string query =
        "UPDATE ml_model_version "
        "SET validation_status = 'approved', "
        "    approval_notes = $2, "
        "    approved_by = CURRENT_USER, "
        "    approved_at = NOW() "
        "WHERE id = $1";

    std::string id_str = std::to_string(version_id);
    const char* params[2] = {
        id_str.c_str(),
        approval_notes.empty() ? nullptr : approval_notes.c_str()
    };

    PGresult* res = PQexecParams(conn_, query.c_str(), 2, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to approve model version: " + error);
    }

    PQclear(res);
}

// ============================================================================
// Model Deployment
// ============================================================================

void MLOperations::deploy_model(
    const std::string& model_name,
    int64_t version_id,
    const std::string& deployment_strategy
) {
    std::string query = "SELECT deploy_ml_model($1, $2, $3)";

    std::string version_id_str = std::to_string(version_id);

    const char* params[3] = {
        model_name.c_str(),
        version_id_str.c_str(),
        deployment_strategy.c_str()
    };

    PGresult* res = PQexecParams(conn_, query.c_str(), 3, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to deploy model: " + error);
    }

    PQclear(res);
}

ModelInfo MLOperations::get_deployed_model(const std::string& model_name) {
    std::string query =
        "SELECT mv.id, mv.model_name, mv.version, mv.model_type, mv.artifact_path, "
        "       mv.validation_status, mv.is_deployed, mv.performance_metrics, mv.created_at "
        "FROM ml_model_registry mr "
        "JOIN ml_model_version mv ON mr.production_version_id = mv.id "
        "WHERE mr.model_name = $1 AND mr.is_active = TRUE";

    const char* params[1] = {model_name.c_str()};

    PGresult* res = PQexecParams(conn_, query.c_str(), 1, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
        PQclear(res);
        throw std::runtime_error("No deployed model found: " + model_name);
    }

    ModelInfo info;
    info.model_version_id = std::stoll(PQgetvalue(res, 0, 0));
    info.model_name = PQgetvalue(res, 0, 1);
    info.version = PQgetvalue(res, 0, 2);
    info.model_type = PQgetvalue(res, 0, 3);
    info.artifact_path = PQgetvalue(res, 0, 4);
    info.is_deployed = true;  // By definition, this is deployed

    std::string metrics_json = PQgetvalue(res, 0, 7);
    info.performance_metrics = json_to_metrics(metrics_json);

    PQclear(res);
    return info;
}

void MLOperations::rollback_model(const std::string& model_name) {
    // Get the previous version ID from registry
    std::string query =
        "SELECT previous_version_id FROM ml_model_registry "
        "WHERE model_name = $1 AND is_active = TRUE";

    const char* params[1] = {model_name.c_str()};

    PGresult* res = PQexecParams(conn_, query.c_str(), 1, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
        PQclear(res);
        throw std::runtime_error("Model not found in registry: " + model_name);
    }

    if (PQgetisnull(res, 0, 0)) {
        PQclear(res);
        throw std::runtime_error("No previous version to rollback to for: " + model_name);
    }

    int64_t previous_version_id = std::stoll(PQgetvalue(res, 0, 0));
    PQclear(res);

    // Deploy the previous version
    deploy_model(model_name, previous_version_id, "rollback");
}

// ============================================================================
// Fine-Tuning Operations
// ============================================================================

int64_t MLOperations::start_fine_tuning(
    int64_t base_model_version_id,
    const std::string& fine_tune_dataset_path,
    const Hyperparameters& hyperparameters,
    int64_t experiment_id
) {
    // Get base model info
    ModelInfo base_model = get_model_version(base_model_version_id);

    // Create a run name indicating fine-tuning
    std::string run_name = "finetune-" + base_model.model_name + "-" + base_model.version;

    // Start a training run with the hyperparameters
    int64_t run_id = start_run(experiment_id, run_name, hyperparameters);

    // Store fine-tuning metadata in the run
    std::string update_query =
        "UPDATE ml_run SET "
        "  hyperparameters = jsonb_set(hyperparameters, '{base_model_version_id}', to_jsonb($2::bigint)), "
        "  hyperparameters = jsonb_set(hyperparameters, '{fine_tune_dataset}', to_jsonb($3::text)) "
        "WHERE id = $1";

    std::string run_id_str = std::to_string(run_id);
    std::string base_model_id_str = std::to_string(base_model_version_id);

    const char* params[3] = {
        run_id_str.c_str(),
        base_model_id_str.c_str(),
        fine_tune_dataset_path.c_str()
    };

    PGresult* res = PQexecParams(conn_, update_query.c_str(), 3, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::string error = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("Failed to update fine-tuning metadata: " + error);
    }

    PQclear(res);

    return run_id;
}

// ============================================================================
// Inference Operations
// ============================================================================

bool MLOperations::load_model_for_inference(int64_t version_id) {
    // Get model artifact path
    ModelInfo model = get_model_version(version_id);

    // Check if artifact exists and is a safetensors file
    namespace fs = std::filesystem;
    if (!fs::exists(model.artifact_path) ||
        model.artifact_path.substr(model.artifact_path.length() - 12) != ".safetensors") {
        return false;
    }

    // Load model weights into cache
    return ModelCache::instance().load_model(version_id, model.artifact_path);
}

Blake3Hash MLOperations::run_inference(
    int64_t model_version_id,
    const Blake3Hash& input_composition
) {
    // Get model weights
    const ModelWeights* weights = ModelCache::instance().get_model(model_version_id);
    if (!weights) {
        // Model not loaded, return zero hash
        return Blake3Hash{};
    }

    // Get input coordinates from database
    Point4F input_coords = get_composition_coordinates(input_composition);
    if (input_coords.norm() == 0.0f) {
        // Invalid coordinates, return zero hash
        return Blake3Hash{};
    }

    // Convert to Eigen vector (4D coordinates as input)
    Eigen::Vector4f input_vec;
    input_vec << input_coords.x, input_coords.y, input_coords.z, input_coords.m;

    // Apply linear transformation: output = weight_matrix * input + bias
    Eigen::VectorXf output_vec = weights->weight_matrix * input_vec;
    if (weights->has_bias) {
        output_vec += weights->bias_vector;
    }

    // Convert back to 4D coordinates on unit sphere
    Point4F output_coords(
        output_vec(0),
        output_vec(1),
        output_vec(2),
        output_vec(3)
    );

    // Normalize to unit sphere (project to S³ surface)
    output_coords = output_coords.normalized();

    // Convert to quantized coordinates and compute hash
    Point4D quantized = output_coords.to_quantized();

    // Create composition hash from coordinates
    Blake3Hash output_hash = compute_hash_from_coordinates(quantized);

    return output_hash;
}

std::vector<Blake3Hash> MLOperations::run_batch_inference(
    int64_t model_version_id,
    const std::vector<Blake3Hash>& input_compositions
) {
    std::vector<Blake3Hash> results;
    results.reserve(input_compositions.size());

    // Simple sequential inference for now
    // Real implementation would batch these for efficiency
    for (const auto& input : input_compositions) {
        results.push_back(run_inference(model_version_id, input));
    }

    return results;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

Point4F MLOperations::get_composition_coordinates(const Blake3Hash& hash) {
    // Query the database to get coordinates for this composition hash
    // This can be either an atom or a composition
    std::string query =
        "SELECT ST_X(centroid), ST_Y(centroid), ST_Z(centroid), ST_M(centroid) "
        "FROM composition WHERE hash = $1 "
        "UNION ALL "
        "SELECT ST_X(coords), ST_Y(coords), ST_Z(coords), ST_M(coords) "
        "FROM atom WHERE hash = $1";

    std::string hash_hex = hash.to_hex();
    std::string bytea = "\\x" + hash_hex;
    const char* params[1] = {bytea.c_str()};

    PGresult* res = PQexecParams(conn_, query.c_str(), 1, nullptr,
                                 params, nullptr, nullptr, 0);

    if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
        PQclear(res);
        return Point4F{};  // Return zero coordinates for not found
    }

    // Parse coordinates from PostGIS POINTZM
    double x = std::stod(PQgetvalue(res, 0, 0));
    double y = std::stod(PQgetvalue(res, 0, 1));
    double z = std::stod(PQgetvalue(res, 0, 2));
    double m = std::stod(PQgetvalue(res, 0, 3));

    PQclear(res);

    return Point4F(x, y, z, m);
}

Blake3Hash MLOperations::compute_hash_from_coordinates(const Point4D& coords) {
    // Convert quantized coordinates to float for hashing
    Point4F float_coords(coords);

    // Create a deterministic hash from the coordinates
    // For now, we'll use the blake3 hash of the coordinate values
    std::array<uint8_t, 32> hash_data;
    std::memcpy(hash_data.data(), &float_coords.x, sizeof(double));
    std::memcpy(hash_data.data() + sizeof(double), &float_coords.y, sizeof(double));
    std::memcpy(hash_data.data() + 2 * sizeof(double), &float_coords.z, sizeof(double));
    std::memcpy(hash_data.data() + 3 * sizeof(double), &float_coords.m, sizeof(double));

    // Use Blake3 to hash the coordinate data
    Blake3Hash result;
    // Note: In a real implementation, we'd use the actual Blake3 hashing
    // For now, we'll create a simple hash based on the coordinates
    uint64_t simple_hash = 0;
    for (size_t i = 0; i < hash_data.size(); ++i) {
        simple_hash = simple_hash * 31 + hash_data[i];
    }

    // Fill the hash with a deterministic pattern based on coordinates
    for (size_t i = 0; i < 32; ++i) {
        result.bytes[i] = static_cast<uint8_t>((simple_hash >> (i % 8)) & 0xFF);
    }

    return result;
}

} // namespace ml
} // namespace hypercube
