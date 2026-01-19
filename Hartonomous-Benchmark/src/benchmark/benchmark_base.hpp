#pragma once

#include <string>
#include <memory>
#include <chrono>

struct BenchmarkConfig {
    size_t iterations = 1000;
    size_t data_size = 10000;
    bool use_gpu = false;
    std::string data_type = "float";
};

struct BenchmarkResult {
    std::string name;
    std::chrono::nanoseconds duration;
    double throughput = 0.0; // operations per second
    size_t memory_used = 0; // bytes
    size_t operations = 0; // number of operations performed
    bool success = true;
    std::string error_message;
};

class BenchmarkBase {
public:
    virtual ~BenchmarkBase() = default;
    virtual void setup(const BenchmarkConfig& config) = 0;
    virtual void run() = 0;
    virtual BenchmarkResult get_result() const = 0;
    virtual std::string get_name() const = 0;
    virtual void calculate_throughput(std::chrono::nanoseconds duration) = 0;
};

template<typename DataType, typename OperationType>
class TemplatedBenchmark : public BenchmarkBase {
public:
    BenchmarkConfig config_;
    BenchmarkResult result_;
    std::chrono::steady_clock::time_point start_time_;

public:
    void setup(const BenchmarkConfig& config) override {
        config_ = config;
        result_.name = get_name();
        result_.success = true;
    }

    void run() override {
        start_time_ = std::chrono::steady_clock::now();
        try {
            execute_operation();
        } catch (const std::exception& e) {
            result_.success = false;
            result_.error_message = e.what();
        }
        auto end_time = std::chrono::steady_clock::now();
        result_.duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_);
        calculate_throughput(result_.duration);
    }

    BenchmarkResult get_result() const override {
        return result_;
    }

    virtual void execute_operation() = 0;
    virtual void calculate_throughput(std::chrono::nanoseconds duration) override {
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
        result_.throughput = static_cast<double>(result_.operations) / seconds;
    }
};