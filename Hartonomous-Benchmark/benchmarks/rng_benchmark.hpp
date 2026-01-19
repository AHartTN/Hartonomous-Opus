#pragma once

#include <vector>
#include <random>
#include <mkl_vsl.h>
#include "../src/benchmark/benchmark_base.hpp"

// Example usage:
// RNGBenchmark<double> bench;
// BenchmarkConfig config{1000, 1000000, false, "double"};
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class RNGBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<T> random_data_;
    VSLStreamStatePtr stream_;
    size_t size_;

public:
    std::string get_name() const override {
        return "MKL_VSL_RNG_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        // Generate uniform random numbers using MKL VSL
        int status;
        if constexpr (std::is_same_v<T, double>) {
            status = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_, size_,
                                 random_data_.data(), 0.0, 1.0);
        } else if constexpr (std::is_same_v<T, float>) {
            status = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_, size_,
                                 random_data_.data(), 0.0f, 1.0f);
        } else {
            throw std::runtime_error("Unsupported data type for MKL VSL RNG");
        }

        if (status != VSL_STATUS_OK) {
            throw std::runtime_error("VSL RNG failed with status: " + std::to_string(status));
        }

        // Calculate throughput (operations per second: size generations)
        size_t operations = size_;
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = operations / seconds;
        this->result_.memory_used = random_data_.size() * sizeof(T);
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        size_ = config.data_size;

        random_data_.resize(size_);

        // Create VSL stream
        int status = vslNewStream(&stream_, VSL_BRNG_MT19937, 777);
        if (status != VSL_STATUS_OK) {
            throw std::runtime_error("Failed to create VSL stream");
        }
    }

    ~RNGBenchmark() {
        if (stream_) {
            vslDeleteStream(&stream_);
        }
    }
};