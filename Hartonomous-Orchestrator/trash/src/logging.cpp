#include "logging.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sink.h>
#include <memory>
#include <mutex>
#include <iostream>

// Logger implementation
class Logger::Impl {
public:
    std::shared_ptr<spdlog::logger> logger;
    
    Impl() {
        // Create console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);
        
        // Create file sink
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("orchestrator.log", true);
        file_sink->set_level(spdlog::level::trace);
        
        // Create logger with both sinks
        logger = std::make_shared<spdlog::logger>("orchestrator", spdlog::sinks_init_list{console_sink, file_sink});
        logger->set_level(spdlog::level::info);
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        logger->flush_on(spdlog::level::info);
    }
    
    void trace(const std::string& message) {
        logger->trace(message);
    }
    
    void debug(const std::string& message) {
        logger->debug(message);
    }
    
    void info(const std::string& message) {
        logger->info(message);
    }
    
    void warning(const std::string& message) {
        logger->warn(message);
    }
    
    void error(const std::string& message) {
        logger->error(message);
    }
    
    void critical(const std::string& message) {
        logger->critical(message);
    }
    
    void set_level(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: logger->set_level(spdlog::level::trace); break;
            case LogLevel::DEBUG: logger->set_level(spdlog::level::debug); break;
            case LogLevel::INFO: logger->set_level(spdlog::level::info); break;
            case LogLevel::WARNING: logger->set_level(spdlog::level::warn); break;
            case LogLevel::ERROR: logger->set_level(spdlog::level::err); break;
            case LogLevel::CRITICAL: logger->set_level(spdlog::level::critical); break;
        }
    }
    
    void set_output_file(const std::string& filename) {
        // This is a simplified version - in practice you'd want to reconfigure the file sink
        // For now we'll just log to the console and default file
    }
};

// Logger singleton implementation
std::shared_ptr<Logger> Logger::getInstance() {
    static std::shared_ptr<Logger> instance = nullptr;
    static std::mutex mutex;
    
    if (!instance) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!instance) {
            instance = std::make_shared<Logger>();
        }
    }
    
    return instance;
}

Logger::Logger() : pImpl(std::make_unique<Impl>()) {}

Logger::~Logger() = default;

void Logger::trace(const std::string& message) {
    pImpl->trace(message);
}

void Logger::debug(const std::string& message) {
    pImpl->debug(message);
}

void Logger::info(const std::string& message) {
    pImpl->info(message);
}

void Logger::warning(const std::string& message) {
    pImpl->warning(message);
}

void Logger::error(const std::string& message) {
    pImpl->error(message);
}

void Logger::critical(const std::string& message) {
    pImpl->critical(message);
}

void Logger::set_level(LogLevel level) {
    pImpl->set_level(level);
}

void Logger::set_output_file(const std::string& filename) {
    pImpl->set_output_file(filename);
}