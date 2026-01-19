#pragma once

#include <vector>
#include <random>
#include <memory>
#include "../src/benchmark/benchmark_base.hpp"
#include <hnswlib/hnswlib.h>

// Example usage:
// HNSWInsertionBenchmark bench;
// BenchmarkConfig config{1000, 1000, false, "float"}; // iterations, data_size (num insertions), etc.
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class HNSWInsertionBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<std::vector<T>> initial_data_;
    std::vector<std::vector<T>> insertion_data_;
    std::unique_ptr<hnswlib::HierarchicalNSW<T>> index_;
    std::unique_ptr<hnswlib::SpaceInterface<T>> space_;
    size_t dim_;
    size_t initial_elements_;
    size_t num_insertions_;

public:
    std::string get_name() const override {
        return "HNSW_Insertion_" + std::to_string(num_insertions_) + "x" + std::to_string(dim_);
    }

    void execute_operation() override {
        // Insert new vectors into the index
        for (size_t i = 0; i < num_insertions_; ++i) {
            index_->addPoint(insertion_data_[i].data(), initial_elements_ + i);
        }

        // Throughput: insertions per second
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = num_insertions_ / seconds;

        // Memory used: approximate for all data
        size_t total_memory = (initial_elements_ + num_insertions_) * dim_ * sizeof(T) * 2;
        this->result_.memory_used = total_memory;
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        initial_elements_ = 10000; // Start with some data
        dim_ = 128;
        num_insertions_ = config.data_size;

        // Generate initial data and build index
        initial_data_.resize(initial_elements_, std::vector<T>(dim_));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-1.0, 1.0);
        for (auto& vec : initial_data_) {
            for (auto& val : vec) {
                val = dis(gen);
            }
        }

        space_ = std::make_unique<hnswlib::L2Space>(dim_);
        index_ = std::make_unique<hnswlib::HierarchicalNSW<T>>(space_.get(), initial_elements_ + num_insertions_, 16, 200, true);

        // Add initial data
        for (size_t i = 0; i < initial_elements_; ++i) {
            index_->addPoint(initial_data_[i].data(), i);
        }

        // Generate insertion data
        insertion_data_.resize(num_insertions_, std::vector<T>(dim_));
        for (auto& vec : insertion_data_) {
            for (auto& val : vec) {
                val = dis(gen);
            }
        }
    }
};