#pragma once

#include <vector>
#include <random>
#include <memory>
#include "../src/benchmark/benchmark_base.hpp"
#include <hnswlib/hnswlib.h>

// Example usage:
// HNSWSearchBenchmark bench;
// BenchmarkConfig config{1000, 10000, false, "float"}; // iterations, data_size (num queries), etc.
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class HNSWSearchBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<std::vector<T>> data_;
    std::unique_ptr<hnswlib::HierarchicalNSW<T>> index_;
    std::unique_ptr<hnswlib::SpaceInterface<T>> space_;
    std::vector<std::vector<T>> queries_;
    size_t dim_;
    size_t num_elements_;
    size_t num_queries_;
    size_t k_; // number of nearest neighbors

public:
    std::string get_name() const override {
        return "HNSW_Search_" + std::to_string(num_queries_) + "x" + std::to_string(dim_) + "_k" + std::to_string(k_);
    }

    void execute_operation() override {
        // Perform k-NN search for each query
        for (const auto& query : queries_) {
            auto result = index_->searchKnn(query.data(), k_);
            // Process result if needed, but for benchmark, just perform the search
        }

        // Throughput: queries per second
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = num_queries_ / seconds;

        // Memory used: approximate for index and queries
        size_t index_memory = num_elements_ * dim_ * sizeof(T) * 2;
        size_t query_memory = num_queries_ * dim_ * sizeof(T);
        this->result_.memory_used = index_memory + query_memory;
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        num_elements_ = 10000; // Fixed for now, can parameterize
        dim_ = 128;
        num_queries_ = config.data_size;
        k_ = 10; // Default k

        // Generate data and build index
        data_.resize(num_elements_, std::vector<T>(dim_));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-1.0, 1.0);
        for (auto& vec : data_) {
            for (auto& val : vec) {
                val = dis(gen);
            }
        }

        space_ = std::make_unique<hnswlib::L2Space>(dim_);
        index_ = std::make_unique<hnswlib::HierarchicalNSW<T>>(space_.get(), num_elements_, 16, 200, true);

        // Add data
        for (size_t i = 0; i < num_elements_; ++i) {
            index_->addPoint(data_[i].data(), i);
        }

        // Generate queries
        queries_.resize(num_queries_, std::vector<T>(dim_));
        for (auto& vec : queries_) {
            for (auto& val : vec) {
                val = dis(gen);
            }
        }
    }
};