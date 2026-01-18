#ifndef EMBEDDING_CLIENT_HPP
#define EMBEDDING_CLIENT_HPP

#include <string>
#include <vector>
#include <memory>
#include <boost/json.hpp>
#include "http_client.hpp"
#include "logging.hpp"

class EmbeddingClient {
public:
    explicit EmbeddingClient(std::shared_ptr<HttpClient> http_client);
    virtual ~EmbeddingClient() = default;
    
    // Generate single embedding
    std::vector<float> generate_embeddings(const std::string& text, const std::string& model = "text-embedding-3");
    
    // Generate batch embeddings
    std::vector<std::vector<float>> generate_embeddings_batch(
        const std::vector<std::string>& texts, 
        const std::string& model = "text-3");
    
    // Health check
    bool is_healthy() const;
    
private:
    std::shared_ptr<HttpClient> http_client_;
    std::string host_;
    uint16_t port_;
    bool is_healthy_;
    
    // Internal helper methods
    std::string build_embedding_request(
        const std::vector<std::string>& texts, 
        const std::string& model);
    
    std::vector<float> parse_embedding_response(const std::string& response);
    std::vector<std::vector<float>> parse_batch_response(const std::string& response);
    
    // Service communication
    bool send_embedding_request(
        const std::string& request_body,
        std::string& response);
    
    // Retry logic
    bool retry_request(
        const std::string& request_body,
        std::string& response,
        int max_retries = 3);
    
    // Configuration
    void configure_from_config();
};

#endif // EMBEDDING_CLIENT_HPP