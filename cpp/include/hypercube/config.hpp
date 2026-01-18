#pragma once

#include <string>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include "logging.hpp"

namespace hypercube {

class Config {
public:
    static Config& getInstance() {
        static Config instance;
        return instance;
    }

    // Load configuration from environment variables and optional config file
    bool load(const std::string& config_file = "") {
        std::lock_guard<std::mutex> lock(mutex_);

        // Load from environment variables first
        load_from_env();

        // Load from config file if specified
        if (!config_file.empty() && std::filesystem::exists(config_file)) {
            load_from_file(config_file);
        }

        // Validate configuration
        return validate();
    }

    // Get configuration value with default
    template<typename T>
    T get(const std::string& key, T default_value = T{}) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = values_.find(key);
        if (it == values_.end()) {
            return default_value;
        }

        try {
            if constexpr (std::is_same_v<T, int>) {
                return std::stoi(it->second);
            } else if constexpr (std::is_same_v<T, double>) {
                return std::stod(it->second);
            } else if constexpr (std::is_same_v<T, bool>) {
                std::string val = it->second;
                std::transform(val.begin(), val.end(), val.begin(), ::tolower);
                return val == "true" || val == "1" || val == "yes" || val == "on";
            } else {
                return it->second;
            }
        } catch (const std::exception&) {
            LOG_WARN("Failed to parse config value for key '", key, "', using default");
            return default_value;
        }
    }

    // Set configuration value
    void set(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        values_[key] = value;
    }

    // Print current configuration (for debugging)
    void print() const {
        std::lock_guard<std::mutex> lock(mutex_);

        LOG_INFO("Current configuration:");
        for (const auto& [key, value] : values_) {
            LOG_INFO("  ", key, " = ", value);
        }
    }

private:
    Config() = default;
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    void load_from_env() {
        // Database configuration
        set_if_env("db.host", "HC_DB_HOST", "HART-SERVER");
        set_if_env("db.port", "HC_DB_PORT", "5432");
        set_if_env("db.user", "HC_DB_USER", "postgres");
        set_if_env("db.password", "HC_DB_PASS", "");
        set_if_env("db.name", "HC_DB_NAME", "hypercube");

        // Logging configuration
        set_if_env("log.level", "HC_LOG_LEVEL", "info");
        set_if_env("log.file", "HC_LOG_FILE", "");

        // Performance configuration
        set_if_env("perf.max_threads", "HC_MAX_THREADS", "0");  // 0 = auto-detect
        set_if_env("perf.memory_limit_mb", "HC_MEMORY_LIMIT_MB", "32768");  // 32GB default for modern systems

        // Build/optimization flags
        set_if_env("build.has_mkl", "HC_HAS_MKL", "true");
        set_if_env("build.has_openmp", "HC_HAS_OPENMP", "true");
        set_if_env("build.has_avx", "HC_HAS_AVX", "true");
    }

    void set_if_env(const std::string& key, const std::string& env_var, const std::string& default_value) {
        const char* env_value = std::getenv(env_var.c_str());
        if (env_value && *env_value) {
            values_[key] = env_value;
        } else {
            values_[key] = default_value;
        }
    }

    void load_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            LOG_WARN("Could not open config file: ", filename);
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;

            // Parse key=value
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = line.substr(0, equals_pos);
                std::string value = line.substr(equals_pos + 1);

                // Trim whitespace
                key.erase(key.begin(), std::find_if(key.begin(), key.end(), [](int ch) { return !std::isspace(ch); }));
                key.erase(std::find_if(key.rbegin(), key.rend(), [](int ch) { return !std::isspace(ch); }).base(), key.end());

                value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](int ch) { return !std::isspace(ch); }));
                value.erase(std::find_if(value.rbegin(), value.rend(), [](int ch) { return !std::isspace(ch); }).base(), value.end());

                if (!key.empty()) {
                    values_[key] = value;
                }
            }
        }

        LOG_INFO("Loaded configuration from file: ", filename);
    }

    bool validate() const {
        bool valid = true;

        // Validate database configuration
        if (get<std::string>("db.host").empty()) {
            LOG_ERROR("Database host not configured");
            valid = false;
        }

        if (get<int>("db.port") <= 0 || get<int>("db.port") > 65535) {
            LOG_ERROR("Invalid database port: ", get<int>("db.port"));
            valid = false;
        }

        if (get<std::string>("db.user").empty()) {
            LOG_ERROR("Database user not configured");
            valid = false;
        }

        // Validate log level
        std::string log_level = get<std::string>("log.level");
        if (log_level != "debug" && log_level != "info" && log_level != "warn" &&
            log_level != "error" && log_level != "fatal") {
            LOG_WARN("Unknown log level '", log_level, "', defaulting to 'info'");
            const_cast<Config*>(this)->set("log.level", "info");
        }

        return valid;
    }

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> values_;
};

// Initialize configuration on startup
inline bool init_config(const std::string& config_file = "config.env") {
    Config& config = Config::getInstance();

    // Set log level from environment (before loading config)
    const char* log_level_env = std::getenv("HC_LOG_LEVEL");
    if (log_level_env) {
        if (strcmp(log_level_env, "debug") == 0) set_log_level(LogLevel::DEBUG);
        else if (strcmp(log_level_env, "info") == 0) set_log_level(LogLevel::INFO);
        else if (strcmp(log_level_env, "warn") == 0) set_log_level(LogLevel::WARN);
        else if (strcmp(log_level_env, "error") == 0) set_log_level(LogLevel::ERROR);
        else if (strcmp(log_level_env, "fatal") == 0) set_log_level(LogLevel::FATAL);
    }

    if (!config.load(config_file)) {
        LOG_ERROR("Failed to load configuration");
        return false;
    }

    // Set log output file if specified
    std::string log_file = config.get<std::string>("log.file");
    if (!log_file.empty()) {
        static std::ofstream log_stream(log_file, std::ios::app);
        if (log_stream.is_open()) {
            set_log_output(log_stream);
        } else {
            LOG_ERROR("Could not open log file: ", log_file);
        }
    }

    LOG_INFO("Configuration loaded successfully");
    return true;
}

} // namespace hypercube