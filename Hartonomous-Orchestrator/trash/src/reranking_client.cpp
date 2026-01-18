#include "reranking_client.hpp"
#include "logging.hpp"
#include <boost/json.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

RerankingClient::RerankingClient(std::shared_ptr<HttpClient> http_client) 
    : http_client_(http_client), 
      host_("localhost"), 
      port_(8711),
      is_healthy_(true) {
    // Initialize with default configuration
    configure_from_config();
}

std::vector<std::pair<int, float>> RerankingClient::rerank_documents(
    const std::string& query,
    const std::vector<std::string>& documents,
    const std::string& model) {
    
    try {
        LOG_DEBUG("Reranking " + std::to_string(documents.size()) + " documents for query: " + query.substr(0, 50) + "...");
        
        // Build the request body
        std::string request_body = build_reranking_request(query, documents);
        
        // Send the request
        std::string response;
        bool success = send_reranking_request(request_body, response);
        
        if (!success) {
            throw std::runtime_error("Failed to rerank documents");
        }
        
        // Parse the response
        auto result = parse_reranking_response(response);
        
        LOG_DEBUG("Documents reranked successfully");
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to rerank documents: " + std::string(e.what()));
        throw;
    }
}

bool RerankingClient::is_healthy() const {
    return is_healthy_;
}

std::string RerankingClient::build_reranking_request(
    const std::string& query,
    const std::vector<std::string>& documents) {
    
    // Create a JSON request body for reranking service
    boost::json::object root;
    
    // Add query
    root["query"] = query;
    
    // Add documents
    boost::json::array documents_array;
    for (const auto& doc : documents) {
        documents_array.push_back(doc);
    }
    root["documents"] = documents_array;
    
    // Add model
    root["model"] = model;
    
    return boost::json::serialize(root);
}

std::vector<std::pair<int, float>> RerankingClient::parse_reranking_response(const std::string& response) {
    try {
        // Parse the JSON response
        boost::json::value val = boost::json::parse(response);
        
        // Extract the results array
        boost::json::array results = val.as_object()["results"].as_array();
        
        std::vector<std::pair<int, float>> ranked_results;
        ranked_results.reserve(results.size());
        
        for (const auto& item : results) {
            int index = item.as_object()["index"].as_int64();
            float relevance_score = static_cast<float>(item.as_object()["relevance_score"].as_number().to_double());
            ranked_results.push_back({index, relevance_score});
        }
        
        // Sort by relevance score (descending)
        std::sort(ranked_results.begin(), ranked_results.end(),
                   [](const auto& a, const auto& b) {
                       return a.second > b.second;
                   });
        
        return ranked_results;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse reranking response: " + std::string(e.what()));
        throw std::runtime_error("Invalid reranking response");
    }
}

bool RerankingClient::send_reranking_request(
    const std::string& request_body,
    std::string& response) {
    
    try {
        // Send the request to the reranking service
        LOG_DEBUG("Sending reranking request to " + host_ + ":" + std::to_string(port_));
        
        // In a real implementation, this would use the HTTP client
        // For now, we'll simulate the call
        bool success = http_client_->send_request(
            "POST", 
            "/rerank", 
            request_body, 
            response, 
            host_, 
            port_
        );
        
        if (!success) {
            LOG_ERROR("Failed to send reranking request");
            return false;
        }
        
        LOG_DEBUG("Reranking request completed successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to send reranking request: " + std::string(e.what()));
        return false;
    }
}

bool RerankingClient::retry_request(
    const std::string& request_body,
    std::string& response,
    int max_retries) {
    
    try {
        // Implement retry logic with exponential backoff
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            if (send_reranking_request(request_body, response)) {
                return true;
            }
            
            // Exponential backoff
            if (attempt < max_retries - 1) {
                LOG_DEBUG("Retrying reranking request (attempt " + std::to_string(attempt + 1) + ")");
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

void RerankingClient::configure_from_config() {
    // Configuration would be loaded from YAML
    // This is a placeholder for actual configuration loading
    LOG_DEBUG("Reranking client configured");
}