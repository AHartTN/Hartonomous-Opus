#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <mutex>

namespace hypercube {

enum class LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3,
    LOG_FATAL = 4
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
    }

    void setOutput(std::ostream& stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        output_ = &stream;
    }

    template<typename... Args>
    void log(LogLevel level, const char* file, int line, const char* func, Args&&... args) {
        if (level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms.count();

        // Level string
        const char* level_str = "UNKN";
        switch (level) {
            case LogLevel::LOG_DEBUG: level_str = "DEBG"; break;
            case LogLevel::LOG_INFO:  level_str = "INFO"; break;
            case LogLevel::LOG_WARN:  level_str = "WARN"; break;
            case LogLevel::LOG_ERROR: level_str = "EROR"; break;
            case LogLevel::LOG_FATAL: level_str = "FATL"; break;
        }

        // Extract filename from path
        const char* filename = strrchr(file, '/');
        if (!filename) filename = strrchr(file, '\\');
        filename = filename ? filename + 1 : file;

        // Format message
        std::stringstream msg;
        msg << "[" << ss.str() << "] " << level_str << " "
            << filename << ":" << line << " " << func << "() - ";
        format_message(msg, std::forward<Args>(args)...);

        *output_ << msg.str() << std::endl;

        if (level == LogLevel::LOG_FATAL) {
            *output_ << std::flush;
            std::abort();
        }
    }

private:
    Logger() : level_(LogLevel::LOG_INFO), output_(&std::cout) {}
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void format_message(std::stringstream&) {}

    template<typename T, typename... Args>
    void format_message(std::stringstream& ss, T&& value, Args&&... args) {
        ss << value;
        format_message(ss, std::forward<Args>(args)...);
    }

    LogLevel level_;
    std::ostream* output_;
    std::mutex mutex_;
};

// Convenience macros
#define LOG_DEBUG(...) hypercube::Logger::getInstance().log(hypercube::LogLevel::LOG_DEBUG, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_INFO(...)  hypercube::Logger::getInstance().log(hypercube::LogLevel::LOG_INFO,  __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_WARN(...)  hypercube::Logger::getInstance().log(hypercube::LogLevel::LOG_WARN,  __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_ERROR(...) hypercube::Logger::getInstance().log(hypercube::LogLevel::LOG_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_FATAL(...) hypercube::Logger::getInstance().log(hypercube::LogLevel::LOG_FATAL, __FILE__, __LINE__, __func__, __VA_ARGS__)

// Set log level convenience functions
inline void set_log_level(LogLevel level) {
    Logger::getInstance().setLevel(level);
}

inline void set_log_output(std::ostream& stream) {
    Logger::getInstance().setOutput(stream);
}

} // namespace hypercube