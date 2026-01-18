#ifndef QDRANT_CLIENT_HPP
#define QDRANT_CLIENT_HPP

#include <string>
#include <vector>
#include <memory>
#include <boost/json.hpp>
#include "http_client.hpp"
#include "logging.hpp"

// Structure for Qdrant vector data
struct VectorData {
    std::string id;
    std::vector<float> vector;
    boost::json::object payload;
};

// Structure for search result
struct SearchResult {
    std::string id;
    float score;
    boost::json::object payload;
};

class QdrantClient {
public:
    explicit QdrantClient(std::shared_ptr<HttpClient> http_client);
    virtual ~QdrantClient() = default;
    
    // Vector operations
    bool upsert_vectors(const std::vector<VectorData>& vectors, const std::string& collection_name);
    std::vector<SearchResult> search_vectors(const std::vector<float>& vector, 
                                           const std::string& collection_name, 
                                           size_t limit = 10);
    
    // Collection operations
    bool create_collection(const std::string& collection_name, size_t vector_size);
    bool delete_collection(const std::string& collection_name);
    bool collection_exists(const std::string& collection_name);
    
    // Document operations
    bool upsert_document(const std::string& id, const std::vector<float>& vector, 
                         const boost::json::object& payload, const std::string& collection_name);
    std::vector<boost::json::object> get_documents(const std::vector<std::string>& ids, 
                                                 const std::string& collection_name);
    
    // Health check
    bool is_healthy() const;
    
private:
    std::shared_ptr<HttpClient> http_client_;
    std::string host_;
    uint16_t port_;
    bool is_healthy_;
    
    // Internal helper methods
    std::string build_upsert_request(const std::vector<VectorData>& vectors, const std::string& collection_name);
    std::string build_search_request(const std::vector<float>& vector, const std::string& collection_name, size_t limit);
    std::string build_create_collection_request(const std::string& collection_name, size_t vector_size);
    std::string build_get_documents_request(const std::vector<std::string>& ids, const std::string& collection_name);
    std::vector<SearchResult> parse_search_response(const std::string& response);
    bool parse_generic_response(const std::string& response);
};

#endif // QDRANT_CLIENT_HPP