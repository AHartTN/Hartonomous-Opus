#include "metrics.hpp"
#include <map>
#include <string>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <algorithm>

// Forward declarations for implementation
class Metrics::Impl {
public:
    std::map<std::string, int64_t> counters;
    std::map<std::string, double> gauges;
    std::map<std::string, std::vector<double>> histograms;
    std::mutex metrics_mutex;
    
    void increment_counter(const std::string& name, const std::map<std::string, std::string>& labels = {}, int64_t value = 1);
    void set_gauge(const std::string& name, double value, const std::map<std::string, std::string>& labels = {});
    void record_histogram(const std::string& name, double value, const std::map<std::string, std::string>& labels = {});
    std::string export_prometheus();
};

Metrics::Metrics() : pImpl(std::make_unique<Impl>()) {}

Metrics::~Metrics() = default;

Metrics& Metrics::getInstance() {
    static Metrics instance;
    return instance;
}

void Metrics::increment_counter(const std::string& name, const std::map<std::string, std::string>& labels, int64_t value) {
    pImpl->increment_counter(name, labels, value);
}

void Metrics::increment_counter(const std::string& name, const std::map<std::string, std::string>& labels, int64_t value) {
    pImpl->increment_counter(name, labels, value);
}

void Metrics::Metrics::Impl::increment_counter(const std::string& name, const std::map<std::string, std::string>& labels, int64_t value) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    counters[name] += value;
}

void Metrics::set_gauge(const std::string& name, double value, const std::map<std::string, std::string>& labels) {
    pImpl->set_gauge(name, value, labels);
}

void Metrics::Metrics::Impl::set_gauge(const std::string& name, double value, const std::map<std::string, std::string>& labels) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    gauges[name] = value;
}

void Metrics::increment_gauge(const std::string& name, double value, const std::map<std::string, std::string>& labels) {
    pImpl->set_gauge(name, gauges[name] + value, labels);
}

void Metrics::decrement_gauge(const std::string& name, double value, const std::map<std::string, std::string>& labels) {
    pImpl->set_gauge(name, gauges[name] - value, labels);
}

void Metrics::record_histogram(const std::string& name, double value, const std::map<std::string, std::string>& labels) {
    pImpl->record_histogram(name, value, labels);
}

void Metrics::Metrics::Impl::record_histogram(const std::string& name, double value, const std::map<std::string, std::string>& labels) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    histograms[name].push_back(value);
}

Metrics::Timer::Timer(const std::string& name, const std::map<std::string, std::string>& labels)
    : name_(name), labels_(labels), start_(std::chrono::high_resolution_clock::now()), stopped_(false) {}

Metrics::Timer::~Timer() {
    if (!stopped_) {
        stop();
    }
}

void Metrics::Timer::stop() {
    if (!stopped_) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        Metrics::getInstance().record_histogram(name_, duration, labels_);
        stopped_ = true;
    }
}

std::string Metrics::export_prometheus() {
    return pImpl->export_prometheus();
}

std::string Metrics::Metrics::Impl::export_prometheus() {
    std::ostringstream oss;
    
    // Export counters
    for (const auto& pair : counters) {
        oss << "# TYPE " << pair.first << "_count counter\n";
        oss << pair.first << "_count " << pair.second << "\n\n";
    }
    
    // Export gauges
    for (const auto& pair : gauges) {
        oss << "# TYPE " << pair.first << "_gauge gauge\n";
        oss << pair.first << "_gauge " << std::fixed << std::setprecision(6) << pair.second << "\n\n";
    }
    
    // Export histograms
    for (const auto& pair : histograms) {
        oss << "# TYPE " << pair.first << "_histogram summary\n";
        oss << pair.first << "_histogram{quantile=\"0.5\"} " << (pair.second.empty() ? 0 : pair.second[pair.second.size()/2]) << "\n";
        oss << pair.first << "_histogram{quantile=\"0.9\"} " << (pair.second.empty() ? 0 : pair.second[pair.second.size()*9/10]) << "\n";
        oss << pair.first << "_histogram{quantile=\"0.99\"} " << (pair.second.empty() ? 0 : pair.second[pair.second.size()*99/100]) << "\n";
        oss << pair.first << "_histogram_sum " << (pair.second.empty() ? 0 : std::accumulate(pair.second.begin(), pair.second.end(), 0.0))) << "\n";
        oss << pair.first << "_histogram_count " << pair.second.size() << "\n\n";
    }
    
    return oss.str();
}