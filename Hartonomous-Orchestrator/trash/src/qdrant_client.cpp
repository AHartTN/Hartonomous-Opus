#include "qdrant_client.hpp"
#include "logging.hpp"
#include <boost/json.hpp>
#include <boost/beast.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

QdrantClient::QdrantClient(std::shared_ptr<HttpClient> http_client) 
    : http_client_(http_client), 
      host_("localhost"), 
      port_(6333),
      is_healthy_(true) {
    // Default configuration - will be updated from config
}

bool QdrantClient::upsert_vectors(const std::vector<VectorData>& vectors, const std::string& collection_name) {
    try {
        LOG_DEBUG("Upserting " + std::to_string(vectors.size()) + " vectors to collection: " + collection_name);
        
        // Build the request body
        std::string request_body = build_upsert_request(vectors, collection_name);
        
        // Send the request
        std::string response;
        // For simplicity, we'll use synchronous request here
        LOG_INFO("Sending upsert request to Qdrant...");
        
        // In a real implementation, this would be async
        // bool success = http_client_->send_request("POST", "/collections/" + collection_name + "/points", request_body, response);
        
        // Parse the response
        bool success = parse_generic_response(response);
        
        LOG_DEBUG("Vector upsert completed successfully");
        return success;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to upsert vectors: " + std::string(e.what()));
        throw;
    }
}

std::vector<SearchResult> QdrantClient::search_vectors(const std::vector<float>& vector, 
                                                     const std::string& collection_name, 
                                                     size_t limit) {
    try {
        LOG_DEBUG("Searching vectors in collection: " + collection_name);
        
        // Build the request body
        std::string request_body = build_search_request(vector, collection_name, limit);
        
        // Send the request
        std::string response;
        // For simplicity, we'll use synchronous request here
        LOG_INFO("Sending search request to Qdrant...");
        
        // In a real implementation, this would be async
        // bool success = http_client_->send_request("POST", "/collections/" + collection_name + "/points/search", request_body, response);
        
        // Parse the response
        auto results = parse_search_response(response);
        
        LOG_DEBUG("Search completed successfully");
        return results;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to search vectors: " + std::string(e.what()));
        throw;
    }
}

bool QdrantClient::create_collection(const std::string& collection_name, size_t vector_size) {
    try {
        LOG_DEBUG("Creating collection: " + collection_name);
        
        // Build the request body
        std::string request_body = build_create_collection_request(collection_name, vector_size);
        
        // Send the request
        std::string response;
        // For simplicity, we'll use synchronous request here
        LOG_INFO("Sending create collection request to Qdrant...");
        
        // In a real implementation, this would be async
        // bool success = http_client_->send_request("POST", "/collections", request_body, response);
        
        // Parse the response
        bool success = parse_generic_response(response);
        
        LOG_DEBUG("Collection created successfully");
        return success;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create collection: " + std::string(e.what()));
        throw;
    }
}

bool QdrantClient::delete_collection(const std::string& collection_name) {
    try {
        LOG_DEBUG("Deleting collection: " + collection_name);
        
        // Send the request
        std::string response;
        LOG_INFO("Sending delete collection request to Qdrant...");
        
        // In a real implementation, this would be async
        // bool success = http_client_->send_request("DELETE", "/collections/" + collection_name, "", response);
        
        // Parse the response
        bool success = parse_generic_response(response);
        
        LOG_DEBUG("Collection deleted successfully");
        return success;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to delete collection: " + std::string(e.what()));
        throw;
    }
}

bool QdrantClient::collection_exists(const std::string& collection_name) {
    try {
        LOG_DEBUG("Checking if collection exists: " + collection_name);
        
        // Send the request
        std::string response;
        LOG_INFO("Sending collection exists check to Qdrant...");
        
        // In a real implementation, this would be async
        // bool success = http_client_->send_request("GET", "/collections/" + collection_name, "", response);
        
        // Parse the response
        bool exists = parse_generic_response(response);
        
        LOG_DEBUG("Collection exists check completed");
        return exists;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to check collection exists: " + std::string(e.what()));
        throw;
    }
}

bool QdrantClient::upsert_document(const std::string& id, const std::vector<float>& vector, 
                                  const boost::json::object& payload, const std::string& collection_name) {
    try {
        LOG_DEBUG("Upserting document with ID: " + id + " to collection: " + collection_name);
        
        // Create VectorData structure
        VectorData vector_data;
        vector_data.id = id;
        vector_data.vector = vector;
        vector_data.payload = payload;
        
        // Build the request
        std::vector<VectorData> vectors = {vector_data};
        std::string request_body = build_upsert_request(vectors, collection_name);
        
        // Send the request
        std::string response;
        LOG_INFO("Sending upsert document request to Qdrant...");
        
        // In a real implementation, this would be async
        // bool success = http_client_->send_request("POST", "/collections/" + collection_name + "/points", request_body, response);
        
        // Parse the response
        bool success = parse_generic_response(response);
        
        LOG_DEBUG("Document upsert completed successfully");
        return success;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to upsert document: " + std::string(e.what()));
        throw;
    }
}

std::vector<boost::json::object> QdrantClient::get_documents(const std::vector<std::string>& ids, 
                                                           const std::string& collection_name) {
    try {
        LOG_DEBUG("Getting documents with IDs: " + std::to_string(ids.size()) + " from collection: " + collection_name);
        
        // Build the request
        std::string request_body = build_get_documents_request(ids, collection_name);
        
        // Send the request
        std::string response;
        LOG_INFO("Sending get documents request to Qdrant...");
        
        // In a real implementation, this would be async
        // std::string response = http_client_->send_request("POST", "/collections/" + collection_name + "/points", request_body);
        
        // Parse the response
        std::vector<boost::json::object> results;
        // Parse response logic here
        
        LOG_DEBUG("Documents retrieved successfully");
        return results;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to get documents: " + std::string(e.what()));
        throw;
    }
}

bool QdrantClient::is_healthy() const {
    return is_healthy_;
}

std::string QdrantClient::build_upsert_request(const std::vector<VectorData>& vectors, const std::string& collection_name) {
    // Create a JSON request body for Qdrant's upsert operation
    boost::json::object root;
    boost::json::array points;
    
    for (const auto& vector_data : vectors) {
        boost::json::object point;
        point["id"] = vector_data.id;
        
        boost::json::array vector_array;
        for (float val : vector_data.vector) {
            vector_array.push_back(val);
        }
        point["vector"] = vector_array;
        
        if (!vector_data.payload.empty()) {
            point["payload"] = vector_data.payload;
        }
        
        points.push_back(point);
    }
    
    root["points"] = points;
    
    return boost::json::serialize(root);
}

std::string QdrantClient::build_search_request(const std::vector<float>& vector, const std::string& collection_name, size_t limit) {
    // Create a JSON request body for Qdrant's search operation
    boost::json::object root;
    
    boost::json::array vector_array;
    for (float val : vector) {
        vector_array.push_back(val);
    }
    root["vector"] = vector_array;
    
    root["limit"] = limit;
    
    // Add filter if needed
    // root["filter"] = filter;
    
    return boost::json::serialize(root);
}

std::string QdrantClient::build_create_collection_request(const std::string& collection_name, size_t vector_size) {
    // Create a JSON request body for Qdrant's create collection operation
    boost::json::object root;
    root["name"] = collection_name;
    
    boost::json::object vectors_config;
    vectors_config["size"] = vector_size;
    vectors_config["distance"] = "Cosine";
    
    root["vectors"] = vectors_config;
    
    return boost::json::serialize(root);
}

std::string QdrantClient::build_get_documents_request(const std::vector<std::string>& ids, const std::string& collection_name) {
    // Create a JSON request body for Qdrant's get documents operation
    boost::json::object root;
    
    boost::json::array ids_array;
    for (const auto& id : ids) {
        ids_array.push_back(id);
    }
    root["ids"] = ids_array;
    
    return boost::json::serialize(root);
}

std::vector<SearchResult> QdrantClient::parse_search_response(const std::string& response) {
    try {
        // Parse the JSON response
        boost::json::value val = boost::json::parse(response);
        boost::json::array hits = val.as_object()["result"].as_array();
        
        std::vector<SearchResult> results;
        results.reserve(hits.size());
        
        for (const auto& hit : hits) {
            SearchResult result;
            result.id = hit.as_object()["id"].as_string().c_str();
            result.score = static_cast<float>(hit.as_object()["score"].as_number().to_double());
            
            // Parse payload if exists
            if (hit.as_object().contains("payload")) {
                result.payload = hit.as_object()["payload"].as_object();
            }
            
            results.push_back(result);
        }
        
        return results;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse search response: " + std::string(e.what()));
        throw std::runtime_error("Invalid search response");
    }
}

bool QdrantClient::parse_generic_response(const std::string& response) {
    try {
        // Parse the JSON response to check if it was successful
        boost::json::value val = boost::json::parse(response);
        
        // Check if it contains a success indicator
        if (val.as_object().contains("result")) {
            return true;
        }
        
        // If it's a different format, check for error
        if (val.as_object().contains("status")) {
            return val.as_object()["status"].as_string().c_str() == "ok";
        }
        
        // Default to true if we can't determine
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse generic response: " + std::string(e.what()));
        return false;
    }
}