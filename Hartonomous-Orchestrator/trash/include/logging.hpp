#ifndef LOGGING_HPP
#define LOGGING_HPP

#include <string>
#include <memory>

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
};

class Logger {
public:
    static std::shared_ptr<Logger> getInstance();
    
    void trace(const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void critical(const std::string& message);
    
    void set_level(LogLevel level);
    void set_output_file(const std::string& filename);
    
private:
    Logger();
    ~Logger();
    
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Convenience macros
#define LOG_TRACE(msg) Logger::getInstance()->trace(msg)
#define LOG_DEBUG(msg) Logger::getInstance()->debug(msg)
#define LOG_INFO(msg) Logger::getInstance()->info(msg)
#define LOG_WARNING(msg) Logger::getInstance()->warning(msg)
#define LOG_ERROR(msg) Logger::getInstance()->error(msg)
#define LOG_CRITICAL(msg) Logger::getInstance()->critical(msg)

#endif // LOGGING_HPP