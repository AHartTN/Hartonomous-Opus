#ifndef HTTP_SERVER_HPP
#define HTTP_SERVER_HPP

#include <string>
#include <memory>
#include <boost/beast.hpp>
#include <boost/json.hpp>
#include "orchestrator.hpp"
#include "logging.hpp"
#include "metrics.hpp"

// Forward declarations for HTTP request/response types
namespace boost::beast {
    class http::request;
    class http::response;
}

class HttpServer {
public:
    HttpServer(
        std::shared_ptr<Orchestrator> orchestrator,
        const std::string& host = "localhost",
        uint16_t port = 8080);
    
    virtual ~HttpServer() = default;
    
    // Start the HTTP server
    bool start();
    
    // Stop the HTTP server
    void stop();
    
    // Check if server is running
    bool is_running() const;
    
    // Health check endpoint
    bool health_check();
    
    // Metrics endpoint
    std::string get_metrics();
    
private:
    std::shared_ptr<Orchestrator> orchestrator_;
    std::string host_;
    uint16_t port_;
    bool is_running_;
    
    // HTTP server implementation details
    void setup_routes();
    void handle_request(const boost::beast::http::request<boost::beast::http::string_body>& req,
                       boost::beast::http::response<boost::beast::http::string_body>& res);
    
    // Route handlers
    void handle_ingest_request(const boost::beast::http::request<boost::beast::http::string_body>& req,
                               boost::beast::http::response<boost::beast::http::string_body>& res);
    
    void handle_query_request(const boost::beast::http::request<boost::beast::http::string_body>& req,
                             boost::beast::http::response<boost::beast::http::string_body>& res);
    
    void handle_batch_query_request(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                    boost::beast::http::response<boost::beast::http::string_body>& res);
    
    void handle_health_check(const boost::beast::http::request<boost::beast::http::string_body>& req,
                           boost::beast::http::response<boost::beast::http::string_body>& res);
    
    void handle_metrics_endpoint(const boost::beast::http::request<boost::beast::http::string_body>& req,
                               boost::beast::http::response<boost::beast::http::string_body>& res);
    
    void handle_openai_proxy(const boost::beast::http::request<boost::beast::http::string_body>& req,
                             boost::beast::http::response<boost::beast::http::string_body>& res);
    
    // Utility methods
    std::string build_json_response(const boost::json::object& data, int status_code = 200);
    std::string build_error_response(const std::string& error_message, int status_code = 500);
    bool validate_request(const boost::beast::http::request<boost::beast::http::string_body>& req);
    
    // Middleware components
    void apply_authentication_middleware(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                   boost::beast::http::response<boost::beast::http::string_body>& res);
    
    void apply_rate_limiting_middleware(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                     boost::beast::http::response<boost::beast::http::string_body>& res);
    
    void apply_logging_middleware(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                boost::beast::http::string_body>& res);
    
    void apply_tracing_middleware(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                  boost::beast::http::response<boost::beast::http::string_body>& res);
};

#endif // HTTP_SERVER_HPP