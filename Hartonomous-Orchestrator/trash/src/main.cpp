#include "config.hpp"
#include "logging.hpp"
#include "metrics.hpp"
#include "http_client.hpp"
#include "embedding_client.hpp"
#include "reranking_client.hpp"
#include "generative_client.hpp"
#include "qdrant_client.hpp"
#include "orchestrator.hpp"
#include "http_server.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    try {
        LOG_INFO("Starting Hartonomous Orchestrator...");
        
        // Initialize configuration
        auto config = Config::load_config("config.yaml");
        LOG_INFO("Configuration loaded successfully");
        
        // Initialize logging
        Logging::initialize_logging(config.logging);
        LOG_INFO("Logging system initialized");
        
        // Initialize metrics
        Metrics::initialize_metrics(config.metrics);
        LOG_INFO("Metrics system initialized");
        
        // Create HTTP client
        auto http_client = std::make_shared<HttpClient>();
        LOG_INFO("HTTP client created successfully");
        
        // Create service clients
        auto embedding_client = std::make_shared<EmbeddingClient>(http_client);
        auto reranking_client = std::make_shared<RerankingClient>(http_client);
        auto generative_client = std::make_shared<GenerativeClient>(http_client);
        auto qdrant_client = std::make_shared<QdrantClient>(http_client);
        LOG_INFO("Service clients created successfully");
        
        // Create orchestrator
        auto orchestrator = std::make_shared<Orchestrator>(
            http_client,
            embedding_client,
            reranking_client,
            generative_client,
            qdrant_client
        );
        LOG_INFO("Orchestrator created successfully");
        
        // Create HTTP server
        auto server = std::make_shared<HttpServer>(orchestrator, config.server.host, config.server.port);
        LOG_INFO("HTTP server created successfully");
        
        // Start the server
        if (server->start()) {
            LOG_INFO("Hartonomous Orchestrator started successfully on " + config.server.host + ":" + std::to_string(config.server.port));
            
            // Keep the server running
            std::cout << "Press Ctrl+C to stop the server" << std::endl;
            std::cin.get();
            
            // Stop the server
            server->stop();
            LOG_INFO("Server stopped gracefully");
        } else {
            LOG_ERROR("Failed to start HTTP server");
            return 1;
        }
        
        LOG_INFO("Hartonomous Orchestrator stopped successfully");
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal error in main: " + std::string(e.what()));
        return 1;
    } catch (...) {
        LOG_ERROR("Unknown fatal error in main");
        return 1;
    }
}