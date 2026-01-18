#ifndef HTTP_CLIENT_HPP
#define HTTP_CLIENT_HPP

#include <string>
#include <memory>
#include <boost/beast.hpp>
#include <boost/json.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>

// Forward declarations
namespace boost::beast {
    class http::request;
    class http::response;
}

class HttpClient {
public:
    explicit HttpClient();
    virtual ~HttpClient() = default;
    
    // HTTP operations
    bool send_request(
        const std::string& method,
        const std::string& target,
        const std::string& body,
        std::string& response,
        const std::string& host = "",
        uint16_t port = 80);
    
    bool send_request(
        const std::string& method,
        const std::string& target,
        const boost::json::object& body,
        std::string& response,
        const std::string& host = "",
        uint16_t port = 80);
    
    // Async operations (placeholder)
    void send_async_request(
        const std::string& method,
        const std::string& target,
        const std::string& body,
        const std::string& host = "",
        uint16_t port = 80);
    
    // Connection pooling
    bool is_connected() const;
    void close_connection();
    
    // Configuration
    void set_timeout(int seconds);
    void set_max_retries(int max_retries);
    void set_retry_backoff(double backoff_factor);
    
private:
    std::string host_;
    uint16_t port_;
    int timeout_;
    int max_retries_;
    double retry_backoff_;
    
    // Internal helper methods
    bool connect();
    bool reconnect();
    bool handle_response(const boost::beast::http::response<boost::beast::http::string_body>& res);
    std::string build_request_string(
        const std::string& method,
        const std::string& target,
        const std::string& body,
        const std::string& host);
    
    // SSL support
    bool use_ssl_;
    std::shared_ptr<boost::asio::ssl::context> ssl_context_;
};

#endif // HTTP_CLIENT_HPP