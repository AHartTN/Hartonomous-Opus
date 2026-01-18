#include "http_server.hpp"
#include "logging.hpp"
#include "metrics.hpp"
#include <boost/beast.hpp>
#include <boost/json.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <memory>

HttpServer::HttpServer(
    std::shared_ptr<Orchestrator> orchestrator,
    const std::string& host,
    uint16_t port)
    : orchestrator_(orchestrator),
      host_(host),
      port_(port),
      is_running_(false) {
    // Initialize HTTP server with orchestrator
    setup_routes();
}

bool HttpServer::start() {
    try {
        LOG_INFO("Starting HTTP server on " + host_ + ":" + std::to_string(port_));
        
        is_running_ = true;
        
        // In a real implementation, this would start the actual HTTP server
        // using boost::beast::http::listener or similar
        
        LOG_INFO("HTTP server started successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to start HTTP server: " + std::string(e.what()));
        return false;
    }
}

void HttpServer::stop() {
    try {
        LOG_INFO("Stopping HTTP server...");
        is_running_ = false;
        LOG_INFO("HTTP server stopped successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to stop HTTP server: " + std::string(e.what()));
    }
}

bool HttpServer::is_running() const {
    return is_running_;
}

bool HttpServer::health_check() {
    try {
        // Perform health check on all services
        bool orchestrator_healthy = orchestrator_->is_healthy();
        LOG_DEBUG("Health check completed. Orchestrator healthy: " + std::to_string(orchestrator_healthy));
        return orchestrator_healthy;
    } catch (const std::exception& e) {
        LOG_ERROR("Health check failed: " + std::string(e.what()));
        return false;
    }
}

std::string HttpServer::get_metrics() {
    try {
        // Get metrics from the metrics system
        std::string metrics = Metrics::export_metrics();
        LOG_DEBUG("Metrics exported successfully");
        return metrics;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to export metrics: " + std::string(e.what()));
        return "";
    }
}

void HttpServer::setup_routes() {
    // Route setup would be implemented here
    // This would define the endpoints for:
    // - /ingest (POST)
    // - /query (POST)
    // - /batch-query (POST)
    // - /health (GET)
    // - /metrics (GET)
    // - /openai-proxy (GET/POST)
    LOG_DEBUG("HTTP routes setup completed");
}

void HttpServer::handle_request(const boost::beast::http::request<boost::beast::http::string_body>& req,
                           boost::beast::http::response<boost::beast::http::string_body>& res) {
    try {
        LOG_DEBUG("Handling HTTP request for path: " + std::string(req.target()));
        
        // Apply middleware
        apply_logging_middleware(req, res);
        apply_authentication_middleware(req, res);
        apply_rate_limiting_middleware(req, res);
        apply_tracing_middleware(req, res);
        
        // Route the request
        std::string target = std::string(req.target());
        if (target == "/health") {
            handle_health_check(req, res);
        } else if (target == "/metrics") {
            handle_metrics_endpoint(req, res);
        } else if (target == "/ingest") {
            handle_ingest_request(req, res);
        } else if (target == "/query") {
            handle_query_request(req, res);
        } else if (target == "/batch-query") {
            handle_batch_query_request(req, res);
        } else if (target.find("/openai") != std::string::npos) {
            handle_openai_proxy(req, res);
        } else {
            res.result(boost::beast::http::status::not_found);
            res.set(boost::beast::http::field::content_type, "application/json");
            res.body() = build_error_response("Endpoint not found", 404);
        }
        
        LOG_DEBUG("HTTP request handled successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to handle HTTP request: " + std::string(e.what()));
        res.result(boost::beast::http::status::internal_server_error);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_error_response("Internal server error", 500);
    }
}

void HttpServer::handle_ingest_request(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                   boost::beast::http::response<boost::beast::http::string_body>& res) {
    try {
        LOG_DEBUG("Processing ingest request");
        
        // Parse request body
        boost::json::value val = boost::json::parse(req.body());
        
        // Extract document data
        std::string document_id = val.as_object()["id"].as_string().c_str();
        std::string content = val.as_object()["content"].as_string().c_str();
        boost::json::object metadata = val.as_object()["metadata"].as_object();
        
        // Ingest document
        bool success = orchestrator_->ingest_document(document_id, content, metadata);
        
        if (success) {
            res.result(boost::beast::http::status::ok);
            res.set(boost::beast::http::field::content_type, "application/json");
            res.body() = build_json_response(boost::json::object{{"status", "success"}, {"id", document_id}});
        } else {
            res.result(boost::beast::http::status::bad_request);
            res.set(boost::beast::http::field::content_type, "application/json");
            res.body() = build_error_response("Failed to ingest document", 400);
        }
        
        LOG_DEBUG("Ingest request completed");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to handle ingest request: " + std::string(e.what()));
        res.result(boost::beast::http::status::bad_request);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_error_response("Invalid request body", 400);
    }
}

void HttpServer::handle_query_request(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                       boost::beast::http::response<boost::beast::http::string_body>& res) {
    try {
        LOG_DEBUG("Processing query request");
        
        // Parse request body
        boost::json::value val = boost::json::parse(req.body());
        
        // Extract query data
        std::string query = val.as_object()["query"].as_string().c_str();
        std::string collection = val.as_object()["collection"].as_string().c_str();
        
        // Process query
        auto result = orchestrator_->query(query, collection);
        
        // Build response
        boost::json::object response_data;
        response_data["query_id"] = result.query_id;
        response_data["answer"] = result.answer;
        response_data["documents"] = boost::json::array{};
        
        for (const auto& id : result.relevant_document_ids) {
            response_data["documents"].as_array().push_back(id);
        }
        
        res.result(boost::beast::http::status::ok);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_json_response(response_data);
        
        LOG_DEBUG("Query request completed");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to handle query request: " + std::string(e.what()));
        res.result(boost::beast::http::status::bad_request);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_error_response("Invalid request body", 400);
    }
}

void HttpServer::handle_batch_query_request(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                            boost::beast::http::response<boost::beast::http::string_body>& res) {
    try {
        LOG_DEBUG("Processing batch query request");
        
        // Parse request body
        boost::json::value val = boost::json::parse(req.body());
        
        // Extract queries
        boost::json::array queries = val.as_object()["queries"].as_array();
        std::vector<std::string> query_list;
        query_list.reserve(queries.size());
        
        for (const auto& q : queries) {
            query_list.push_back(q.as_string().c_str());
        }
        
        // Process batch query
        auto results = orchestrator_->batch_query(query_list);
        
        // Build response
        boost::json::array response_array;
        response_array.reserve(results.size());
        
        for (const auto& result : results) {
            boost::json::object response_data;
            response_data["query_id"] = result.query_id;
            response_data["answer"] = result.answer;
            response_data["documents"] = boost::json::array{};
            
            for (const auto& id : result.relevant_document_ids) {
                response_data["documents"].as_array().push_back(id);
            }
            
            response_array.push_back(response_data);
        }
        
        res.result(boost::beast::http::status::ok);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_json_response(response_array);
        
        LOG_DEBUG("Batch query request completed");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to handle batch query request: " + std::string(e.what()));
        res.result(boost::beast::http::status::bad_request);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_error_response("Invalid request body", 400);
    }
}

void HttpServer::handle_health_check(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                     boost::beast::http::response<boost::beast::http::string_body>& res) {
    try {
        LOG_DEBUG("Processing health check request");
        
        bool healthy = health_check();
        
        boost::json::object response_data;
        response_data["status"] = healthy ? "healthy" : "unhealthy";
        response_data["timestamp"] = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        res.result(boost::beast::http::status::ok);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_json_response(response_data);
        
        LOG_DEBUG("Health check completed");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to handle health check request: " + std::string(e.what()));
        res.result(boost::beast::http::status::internal_server_error);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_error_response("Health check failed", 500);
    }
}

void HttpServer::handle_metrics_endpoint(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                       boost::beast::http::response<boost::beast::http::string_body>& res) {
    try {
        LOG_DEBUG("Processing metrics request");
        
        std::string metrics = get_metrics();
        
        res.result(boost::beast::http::status::ok);
        res.set(boost::beast::http::field::content_type, "text/plain");
        res.body() = metrics;
        
        LOG_DEBUG("Metrics request completed");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to handle metrics request: " + std::string(e.what()));
        res.result(boost::beast::http::status::internal_server_error);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_error_response("Failed to get metrics", 500);
    }
}

void HttpServer::handle_openai_proxy(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                     boost::beast::http::response<boost::beast::http::string_body>& res) {
    try {
        LOG_DEBUG("Processing OpenAI proxy request");
        
        // This would proxy requests to OpenAI-compatible endpoints
        // For now, we'll just return a placeholder response
        
        boost::json::object response_data;
        response_data["proxy"] = "OpenAI-compatible endpoint";
        response_data["status"] = "active";
        
        res.result(boost::beast::http::status::ok);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_json_response(response_data);
        
        LOG_DEBUG("OpenAI proxy request completed");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to handle OpenAI proxy request: " + std::string(e.what()));
        res.result(boost::beast::http::status::bad_request);
        res.set(boost::beast::http::field::content_type, "application/json");
        res.body() = build_error_response("Invalid proxy request", 400);
    }
}

std::string HttpServer::build_json_response(const boost::json::object& data, int status_code) {
    boost::json::value val = data;
    std::string response = boost::json::serialize(val);
    
    // Add status code if needed
    return response;
}

std::string HttpServer::build_error_response(const std::string& error_message, int status_code) {
    boost::json::object error_data;
    error_data["error"] = error_message;
    error_data["code"] = status_code;
    
    return boost::json::serialize(error_data);
}

bool HttpServer::validate_request(const boost::beast::http::request<boost::beast::http::string_body>& req) {
    // Basic validation of request
    return true;
}

void HttpServer::apply_authentication_middleware(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                              boost::beast::http::response<boost::beast::http::string_body>& res) {
    // Authentication middleware would validate API keys
    // This is a placeholder implementation
    LOG_DEBUG("Authentication middleware applied");
}

void HttpServer::apply_rate_limiting_middleware(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                                    boost::beast::http::response<boost::beast::http::string_body>& res) {
    // Rate limiting middleware would check request limits
    // This is a placeholder implementation
    LOG_DEBUG("Rate limiting middleware applied");
}

void HttpServer::apply_logging_middleware(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                                        boost::beast::http::response<boost::beast::http::string_body>& res) {
    // Logging middleware would log request details
    LOG_DEBUG("Request logged for path: " + std::string(req.target()));
}

void HttpServer::apply_tracing_middleware(const boost::beast::http::request<boost::beast::http::string_body>& req,
                                                          boost::beast::http::response<boost::beast::http::string_body>& res) {
    // Tracing middleware would add correlation IDs
    // This is a placeholder implementation
    LOG_DEBUG("Tracing middleware applied");
}