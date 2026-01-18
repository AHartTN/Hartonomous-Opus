#include "config.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>

Config load_config(const std::string& config_file) {
    Config config;
    
    try {
        // Set default values
        config.server.host = "0.0.0.0";
        config.server.port = 8080;
        config.server.use_ssl = false;
        config.server.ssl_cert_file = "";
        config.server.ssl_key_file = "";
        
        config.embedding_service.host = "localhost";
        config.embedding_service.port = 8711;
        config.embedding_service.enabled = true;
        
        config.reranking_service.host = "localhost";
        config.reranking_service.port = 8712;
        config.reranking_service.enabled = true;
        
        config.generative_service.host = "localhost";
        config.generative_service.port = 8710;
        config.generative_service.enabled = true;
        
        config.qdrant_service.host = "localhost";
        config.qdrant_service.port = 6333;
        config.qdrant_service.enabled = true;
        
        config.log_level = "INFO";
        config.log_file = "orchestrator.log";
        config.config_file = config_file;
        config.request_timeout_ms = 30000;
        config.max_concurrent_requests = 1000;
        
        // Load from YAML file if it exists
        if (!config_file.empty()) {
            YAML::Node yaml = YAML::LoadFile(config_file);
            
            if (yaml["server"]) {
                const auto& server = yaml["server"];
                if (server["host"]) config.server.host = server["host"].as<std::string>();
                if (server["port"]) config.server.port = server["port"].as<uint16_t>();
                if (server["use_ssl"]) config.server.use_ssl = server["use_ssl"].as<bool>();
                if (server["ssl_cert_file"]) config.server.ssl_cert_file = server["ssl_cert_file"].as<std::string>();
                if (server["ssl_key_file"]) config.server.ssl_key_file = server["ssl_key_file"].as<std::string>();
            }
            
            if (yaml["embedding_service"]) {
                const auto& emb = yaml["embedding_service"];
                if (emb["host"]) config.embedding_service.host = emb["host"].as<std::string>();
                if (emb["port"]) config.embedding_service.port = emb["port"].as<uint16_t>();
                if (emb["api_key"]) config.embedding_service.api_key = emb["api_key"].as<std::string>();
                if (emb["enabled"]) config.embedding_service.enabled = emb["enabled"].as<bool>();
            }
            
            if (yaml["reranking_service"]) {
                const auto& rerank = yaml["reranking_service"];
                if (rerank["host"]) config.reranking_service.host = rerank["host"].as<std::string>();
                if (rerank["port"]) config.reranking_service.port = rerank["port"].as<uint16_t>();
                if (rerank["api_key"]) config.reranking_service.api_key = rerank["api_key"].as<std::string>();
                if (rerank["enabled"]) config.reranking_service.enabled = rerank["enabled"].as<bool>();
            }
            
            if (yaml["generative_service"]) {
                const auto& gen = yaml["generative_service"];
                if (gen["host"]) config.generative_service.host = gen["host"].as<std::string>();
                if (gen["port"]) config.generative_service.port = gen["port"].as<uint16_t>();
                if (gen["api_key"]) config.generative_service.api_key = gen["api_key"].as<std::string>();
                if (gen["enabled"]) config.generative_service.enabled = gen["enabled"].as<bool>();
            }
            
            if (yaml["qdrant_service"]) {
                const auto& qdrant = yaml["qdrant_service"];
                if (qdrant["host"]) config.qdrant_service.host = qdrant["host"].as<std::string>();
                if (qdrant["port"]) config.qdrant_service.port = qdrant["port"].as<uint16_t>();
                if (qdrant["api_key"]) config.qdrant_service.api_key = qdrant["api_key"].as<std::string>();
                if (qdrant["enabled"]) config.qdrant_service.enabled = qdrant["enabled"].as<bool>();
            }
            
            if (yaml["logging"]) {
                const auto& log = yaml["logging"];
                if (log["level"]) config.log_level = log["level"].as<std::string>();
                if (log["file"]) config.log_file = log["file"].as<std::string>();
            }
            
            if (yaml["request_timeout_ms"]) config.request_timeout_ms = yaml["request_timeout_ms"].as<uint32_t>();
            if (yaml["max_concurrent_requests"]) config.max_concurrent_requests = yaml["max_concurrent_requests"].as<uint32_t>();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading configuration: " << e.what() << std::endl;
        // Use default configuration
    }
    
    return config;
}