#ifndef RERANKING_CLIENT_HPP
#define RERANKING_CLIENT_HPP

#include <string>
#include <vector>
#include <memory>
#include <boost/json.hpp>
#include "http_client.hpp"
#include "logging.hpp"

class RerankingClient {
public:
    explicit RerankingClient(std::shared_ptr<HttpClient> http_client);
    virtual ~RerankingClient() = default;
    
    // Rerank documents
    std::vector<std::pair<int, float>> rerank_documents(
        const std::string& query,
        const std::vector<std::string>& documents,
        const std::string& model = "rerank-english");
    
    // Health check
    bool is_healthy() const;
    
private:
    std::shared_ptr<HttpClient> http_client_;
    std::string host_;
    uint16_t port_;
    bool is_healthy_;
    
    // Internal helper methods
    std::string build_reranking_request(
        const std::string& query,
        const std::vector<std::string>& documents);
    
    std::vector<std::pair<int, float>> parse_reranking_response(const std::string& response);
    
    // Service communication
    bool send_reranking_request(
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

#endif // RERANKING_CLIENT_HPP