#pragma once

#include <vector>
#include <random>
#include <memory>
#include "../src/benchmark/benchmark_base.hpp"
#include <hnswlib/hnswlib.h>

// Example usage:
// HNSWIndexBuildBenchmark bench;
// BenchmarkConfig config{1000, 10000, false, "float"}; // iterations, data_size (num vectors), etc.
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class HNSWIndexBuildBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<std::vector<T>> data_;
    std::unique_ptr<hnswlib::HierarchicalNSW<T>> index_;
    std::unique_ptr<hnswlib::SpaceInterface<T>> space_;
    size_t dim_;
    size_t num_elements_;

public:
    std::string get_name() const override {
        return "HNSW_Index_Build_" + std::to_string(num_elements_) + "x" + std::to_string(dim_);
    }

    void execute_operation() override {
        // Build the HNSW index
        space_ = std::make_unique<hnswlib::L2Space>(dim_);
        index_ = std::make_unique<hnswlib::HierarchicalNSW<T>>(space_.get(), num_elements_, 16, 200, true);

        // Add all data points
        for (size_t i = 0; i < num_elements_; ++i) {
            index_->addPoint(data_[i].data(), i);
        }

        // Calculate memory used (approximate)
        size_t vector_memory = num_elements_ * dim_ * sizeof(T);
        // HNSW has additional overhead, but for simplicity, estimate based on vectors
        this->result_.memory_used = vector_memory * 2; // Rough estimate

        // Throughput: operations per second, here number of vectors processed
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = num_elements_ / seconds;
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        num_elements_ = config.data_size;
        dim_ = 128; // Default dimension, can be parameterized later

        // Generate random data
        data_.resize(num_elements_, std::vector<T>(dim_));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-1.0, 1.0);
        for (auto& vec : data_) {
            for (auto& val : vec) {
                val = dis(gen);
            }
        }
    }
};