#ifndef METRICS_HPP
#define METRICS_HPP

#include <string>
#include <map>
#include <memory>
#include <chrono>

// Forward declarations
class Histogram;
class Counter;
class Gauge;

class Metrics {
public:
    static Metrics& getInstance();
    
    // Counter operations
    void increment_counter(const std::string& name, const std::map<std::string, std::string>& labels = {});
    void increment_counter(const std::string& name, const std::map<std::string, std::string>& labels, int64_t value);
    
    // Gauge operations
    void set_gauge(const std::string& name, double value, const std::map<std::string, std::string>& labels = {});
    void increment_gauge(const std::string& name, double value, const std::map<std::string, std::string>& labels = {});
    void decrement_gauge(const std::string& name, double value, const std::map<std::string, std::string>& labels = {});
    
    // Histogram operations
    void record_histogram(const std::string& name, double value, const std::map<std::string, std::string>& labels = {});
    
    // Timing operations
    class Timer {
    public:
        Timer(const std::string& name, const std::map<std::string, std::string>& labels = {});
        ~Timer();
        
        void stop();
        
    private:
        std::string name_;
        std::map<std::string, std::string> labels_;
        std::chrono::high_resolution_clock::time_point start_;
        bool stopped_;
    };
    
    // Export metrics in Prometheus format
    std::string export_prometheus();
    
private:
    Metrics();
    ~Metrics();
    
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Convenience macros for timers
#define METRICS_TIMER(name) Metrics::Timer timer_##name(#name)

#endif // METRICS_HPP