#include "embedding_client.hpp"
#include "logging.hpp"
#include <boost/json.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

EmbeddingClient::EmbeddingClient(std::shared_ptr<HttpClient> http_client) 
    : http_client_(http_client), 
      host_("localhost"), 
      port_(8711),
      is_healthy_(true) {
    // Initialize with default configuration
    configure_from_config();
}

std::vector<float> EmbeddingClient::generate_embeddings(const std::string& text, const std::string& model) {
    try {
        LOG_DEBUG("Generating embeddings for text: " + text.substr(0, 50) + "...");
        
        // Build the request body
        std::vector<std::string> texts = {text};
        std::string request_body = build_embedding_request(texts, model);
        
        // Send the request
        std::string response;
        bool success = send_embedding_request(request_body, response);
        
        if (!success) {
            throw std::runtime_error("Failed to generate embeddings");
        }
        
        // Parse the response
        auto result = parse_embedding_response(response);
        
        LOG_DEBUG("Embeddings generated successfully");
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to generate embeddings: " + std::string(e.what()));
        throw;
    }
}

std::vector<std::vector<float>> EmbeddingClient::generate_embeddings_batch(
    const std::vector<std::string>& texts, 
    const std::string& model) {
    
    try {
        LOG_DEBUG("Generating batch embeddings for " + std::to_string(texts.size()) + " texts");
        
        // Build the request body
        std::string request_body = build_embedding_request(texts, model);
        
        // Send the request
        std::string response;
        bool success = send_embedding_request(request_body, response);
        
        if (!success) {
            throw std::runtime_error("Failed to generate batch embeddings");
        }
        
        // Parse the response
        auto result = parse_batch_response(response);
        
        LOG_DEBUG("Batch embeddings generated successfully");
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to generate batch embeddings: " + std::string(e.what()));
        throw;
    }
}

bool EmbeddingClient::is_healthy() const {
    return is_healthy_;
}

std::string EmbeddingClient::build_embedding_request(
    const std::vector<std::string>& texts, 
    const std::string& model) {
    
    // Create a JSON request body for embedding service
    boost::json::object root;
    
    // Add input texts
    boost::json::array input_array;
    for (const auto& text : texts) {
        input_array.push_back(text);
    }
    root["input"] = input_array;
    
    // Add model
    root["model"] = model;
    
    // Add encoding format
    root["encoding_format"] = "float";
    
    // Add dimensions if needed
    // root["dimensions"] = 1536;
    
    return boost::json::serialize(root);
}

std::vector<float> EmbeddingClient::parse_embedding_response(const std::string& response) {
    try {
        // Parse the JSON response
        boost::json::value val = boost::json::parse(response);
        
        // Extract the data array
        boost::json::array data = val.as_object()["data"].as_array();
        
        if (data.empty()) {
            throw std::runtime_error("Empty embedding response");
        }
        
        // Get the first embedding
        boost::json::array embedding = data[0].as_object()["embedding"].as_array();
        std::vector<float> result;
        result.reserve(embedding.size());
        
        for (const auto& item : embedding) {
            if (item.is_number()) {
                result.push_back(static_cast<float>(item.as_number().to_double())));
            }
        }
        
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse embedding response: " + std::string(e.what()));
        throw std::runtime_error("Invalid embedding response");
    }
}

std::vector<std::vector<float>> EmbeddingClient::parse_batch_response(const std::string& response) {
    try {
        // Parse the JSON response
        boost::json::value val = boost::json::parse(response);
        
        // Extract the data array
        boost::json::array data = val.as_object()["data"].as_array();
        
        std::vector<std::vector<float>> results;
        results.reserve(data.size());
        
        for (const auto& item : data) {
            boost::json::array embedding = item.as_object()["embedding"].as_array();
            std::vector<float> embedding_vector;
            embedding_vector.reserve(embedding.size());
            
            for (const auto& embedding_item : embedding) {
                if (embedding_item.is_number()) {
                    embedding_vector.push_back(static_cast<float>(embedding_item.as_number().to_double())));
                }
            }
            
            results.push_back(embedding_vector);
        }
        
        return results;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse batch response: " + std::string(e.what()));
        throw std::runtime_error("Invalid batch response");
    }
}

bool EmbeddingClient::send_embedding_request(
    const std::string& request_body,
    std::string& response) {
    
    try {
        // Send the request to the embedding service
        LOG_DEBUG("Sending embedding request to " + host_ + ":" + std::to_string(port_));
        
        // In a real implementation, this would use the HTTP client
        // For now, we'll simulate the call
        bool success = http_client_->send_request(
            "POST", 
            "/embeddings", 
            request_body, 
            response, 
            host_, 
            port_
        );
        
        if (!success) {
            LOG_ERROR("Failed to send embedding request");
            return false;
        }
        
        LOG_DEBUG("Embedding request completed successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to send embedding request: " + std::string(e.what()));
        return false;
    }
}

bool EmbeddingClient::retry_request(
    const std::string& request_body,
    std::string& response,
    int max_retries) {
    
    try {
        // Implement retry logic with exponential backoff
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            if (send_embedding_request(request_body, response)) {
                return true;
            }
            
            // Exponential backoff
            if (attempt < max_retries - 1) {
                LOG_DEBUG("Retrying request (attempt " + std::to_string(attempt + 1) + ")");
                std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (attempt + 1)));
            }
        }
        
        LOG_ERROR("All retry attempts failed");
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Retry request failed: " + std::string(e.what()));
        return false;
    }
}

void EmbeddingClient::configure_from_config() {
    // Configuration would be loaded from YAML
    // This is a placeholder for actual configuration loading
    LOG_DEBUG("Embedding client configured");
}