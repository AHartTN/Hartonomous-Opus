#ifndef GENERATIVE_CLIENT_HPP
#define GENERATIVE_CLIENT_HPP

#include <string>
#include <vector>
#include <memory>
#include <boost/json.hpp>
#include "http_client.hpp"
#include "logging.hpp"

// Structure for chat completion message
struct ChatMessage {
    std::string role;  // "system", "user", "assistant"
    std::string content;
    std::string name;  // Optional
};

// Structure for chat completion response
struct ChatCompletion {
    std::string id;
    std::string model;
    std::vector<ChatMessage> choices;
    boost::json::object usage;
};

// Structure for completion response
struct Completion {
    std::string id;
    std::string model;
    std::string text;
    boost::json::object usage;
};

class GenerativeClient {
public:
    explicit GenerativeClient(std::shared_ptr<HttpClient> http_client);
    virtual ~GenerativeClient() = default;
    
    // Chat completions
    ChatCompletion create_chat_completion(
        const std::vector<ChatMessage>& messages,
        const std::string& model = "gpt-3.5-turbo",
        double temperature = 0.7,
        int max_tokens = 1000);
    
    // Text completions
    Completion create_completion(
        const std::string& prompt,
        const std::string& model = "text-davinci-003",
        double temperature = 0.7,
        int max_tokens = 1000);
    
    // Streaming completions (if supported)
    void create_chat_completion_stream(
        const std::vector<ChatMessage>& messages,
        const std::string& model = "gpt-3.5-turbo",
        double temperature = 0.7);
    
    // Health check
    bool is_healthy() const;
    
private:
    std::shared_ptr<HttpClient> http_client_;
    std::string host_;
    uint16_t port_;
    bool is_healthy_;
    
    // Internal helper methods
    std::string build_chat_completion_request(
        const std::vector<ChatMessage>& messages,
        const std::string& model,
        double temperature,
        int max_tokens);
    
    std::string build_completion_request(
        const std::string& prompt,
        const std::string& model,
        double temperature,
        int max_tokens);
    
    ChatCompletion parse_chat_completion_response(const std::string& response);
    
    Completion parse_completion_response(const std::string& response);
};

#endif // GENERATIVE_CLIENT_HPP