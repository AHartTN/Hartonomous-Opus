#include "generative_client.hpp"
#include "logging.hpp"
#include <boost/json.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

GenerativeClient::GenerativeClient(std::shared_ptr<HttpClient> http_client) 
    : http_client_(http_client), 
      host_("localhost"), 
      port_(8711),
      is_healthy_(true) {
    // Default configuration - will be updated from config
}

ChatCompletion GenerativeClient::create_chat_completion(
    const std::vector<ChatMessage>& messages,
    const std::string& model,
    double temperature,
    int max_tokens) {
    
    try {
        LOG_DEBUG("Creating chat completion with " + std::to_string(messages.size()) + " messages");
        
        // Build the request body
        std::string request_body = build_chat_completion_request(messages, model, temperature, max_tokens);
        
        // Send the request
        std::string response;
        LOG_INFO("Sending chat completion request...");
        
        // In a real implementation, this would be async
        // bool success = http_client_->send_request("POST", "/chat/completions", request_body, response);
        
        // Parse the response
        auto result = parse_chat_completion_response(response);
        
        LOG_DEBUG("Chat completion created successfully");
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create chat completion: " + std::string(e.what()));
        throw;
    }
}

Completion GenerativeClient::create_completion(
    const std::string& prompt,
    const std::string& model,
    double temperature,
    int max_tokens) {
    
    try {
        LOG_DEBUG("Creating text completion for prompt: " + prompt.substr(0, 50) + "...");
        
        // Build the request body
        std::string request_body = build_completion_request(prompt, model, temperature, max_tokens);
        
        // Send the request
        std::string response;
        LOG_INFO("Sending text completion request...");
        
        // In a real implementation, this would be async
        // bool success = http_client_->send_request("POST", "/completions", request_body, response);
        
        // Parse the response
        auto result = parse_completion_response(response);
        
        LOG_DEBUG("Text completion created successfully");
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create text completion: " + std::string(e.what()));
        throw;
    }
}

void GenerativeClient::create_chat_completion_stream(
    const std::vector<ChatMessage>& messages,
    const std::string& model,
    double temperature) {
    
    try {
        LOG_DEBUG("Creating streaming chat completion...");
        
        // Build the request body
        std::string request_body = build_chat_completion_request(messages, model, temperature, 1000);
        
        // In a real implementation, this would handle streaming responses
        // This would be an async operation that streams responses
        LOG_INFO("Sending streaming chat completion request...");
        
        LOG_DEBUG("Streaming chat completion created successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create streaming chat completion: " + std::string(e.what()));
        throw;
    }
}

bool GenerativeClient::is_healthy() const {
    return is_healthy_;
}

std::string GenerativeClient::build_chat_completion_request(
    const std::vector<ChatMessage>& messages,
    const std::string& model,
    double temperature,
    int max_tokens) {
    
    // Create a JSON request body for chat completions
    boost::json::object root;
    
    boost::json::array messages_array;
    for (const auto& msg : messages) {
        boost::json::object message;
        message["role"] = msg.role;
        message["content"] = msg.content;
        if (!msg.name.empty()) {
            message["name"] = msg.name;
        }
        messages_array.push_back(message);
    }
    root["messages"] = messages_array;
    
    root["model"] = model;
    root["temperature"] = temperature;
    root["max_tokens"] = max_tokens;
    
    return boost::json::serialize(root);
}

std::string GenerativeClient::build_completion_request(
    const std::string& prompt,
    const std::string& model,
    double temperature,
    int max_tokens) {
    
    // Create a JSON request body for text completions
    boost::json::object root;
    
    root["prompt"] = prompt;
    root["model"] = model;
    root["temperature"] = temperature;
    root["max_tokens"] = max_tokens;
    
    return boost::json::serialize(root);
}

ChatCompletion GenerativeClient::parse_chat_completion_response(const std::string& response) {
    try {
        // Parse the JSON response
        boost::json::value val = boost::json::parse(response);
        
        ChatCompletion completion;
        completion.id = val.as_object()["id"].as_string().c_str();
        completion.model = val.as_object()["model"].as_string().c_str();
        
        // Parse choices
        boost::json::array choices = val.as_object()["choices"].as_array();
        std::vector<ChatMessage> chat_choices;
        chat_choices.reserve(choices.size());
        
        for (const auto& choice : choices) {
            ChatMessage message;
            message.role = choice.as_object()["message"].as_object()["role"].as_string().c_str();
            message.content = choice.as_object()["message"].as_object()["content"].as_string().c_str();
            chat_choices.push_back(message);
        }
        completion.choices = chat_choices;
        
        // Parse usage
        if (val.as_object().contains("usage")) {
            completion.usage = val.as_object()["usage"].as_object();
        }
        
        return completion;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse chat completion response: " + std::string(e.what()));
        throw std::runtime_error("Invalid chat completion response");
    }
}

Completion GenerativeClient::parse_completion_response(const std::string& response) {
    try {
        // Parse the JSON response
        boost::json::value val = boost::json::parse(response);
        
        Completion completion;
        completion.id = val.as_object()["id"].as_string().c_str();
        completion.model = val.as_object()["model"].as_string().c_str();
        completion.text = val.as_object()["choices"].as_array()[0].as_object()["text"].as_string().c_str();
        
        // Parse usage
        if (val.as_object().contains("usage")) {
            completion.usage = val.as_object()["usage"].as_object();
        }
        
        return completion;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse completion response: " + std::string(e.what()));
        throw std::runtime_error("Invalid completion response");
    }
}