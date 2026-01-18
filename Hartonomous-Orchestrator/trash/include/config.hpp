#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <vector>
#include <cstdint>

struct ServiceConfig {
    std::string host;
    uint16_t port;
    std::string api_key;
    bool enabled;
};

struct ServerConfig {
    std::string host;
    uint16_t port;
    std::string ssl_cert_file;
    std::string ssl_key_file;
    bool use_ssl;
};

struct Config {
    ServerConfig server;
    ServiceConfig embedding_service;
    ServiceConfig reranking_service;
    ServiceConfig generative_service;
    ServiceConfig qdrant_service;
    std::string log_level;
    std::string log_file;
    std::string config_file;
    uint32_t request_timeout_ms;
    uint32_t max_concurrent_requests;
};

// Load configuration from YAML file
Config load_config(const std::string& config_file = "config.yaml");

#endif // CONFIG_HPP