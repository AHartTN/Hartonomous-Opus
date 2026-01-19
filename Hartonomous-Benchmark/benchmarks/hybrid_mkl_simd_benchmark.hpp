#pragma once

#include <vector>
#include <random>
#include "../src/benchmark/benchmark_base.hpp"
#include "../src/math/vector_math.hpp"

// Hybrid MKL + SIMD benchmark for vector operations
template<typename T>
class HybridMKL_SIMD_Benchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<T> vector_a_;
    std::vector<T> vector_b_;
    std::vector<T> result_add_;
    std::vector<T> result_multiply_;
    T dot_result_;
    size_t size_;

public:
    std::string get_name() const override {
        return "Hybrid_MKL_SIMD_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        // Test hybrid add operation
        hartonomous::VectorMath<T>::add_hybrid(vector_a_, vector_b_, result_add_);

        // Test hybrid multiply operation
        hartonomous::VectorMath<T>::multiply_simd(vector_a_, vector_b_, result_multiply_);

        // Test hybrid dot product
        dot_result_ = hartonomous::VectorMath<T>::dot_product_hybrid(vector_a_, vector_b_);

        // Calculate throughput (operations per second)
        // For add: size operations, multiply: size operations, dot: 2*size operations
        size_t total_operations = size_ * 3 + 2 * size_; // add + multiply + dot
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = total_operations / seconds;

        // Memory usage
        this->result_.memory_used = (vector_a_.size() + vector_b_.size() +
                                   result_add_.size() + result_multiply_.size()) * sizeof(T);
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        size_ = config.data_size;

        vector_a_.resize(size_);
        vector_b_.resize(size_);
        result_add_.resize(size_);
        result_multiply_.resize(size_);

        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        } else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(-100, 100);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        }
    }
};