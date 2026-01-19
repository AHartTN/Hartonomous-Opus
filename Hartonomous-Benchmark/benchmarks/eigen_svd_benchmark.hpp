#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include "../src/benchmark/benchmark_base.hpp"

// Example usage:
// EigenSVDBenchmark<double> bench;
// BenchmarkConfig config{1000, 256, false, "double"};
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class EigenSVDBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_a_;
    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd_;
    size_t size_;

public:
    std::string get_name() const override {
        return "Eigen_SVD_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        // Perform SVD decomposition
        svd_.compute(matrix_a_, Eigen::ComputeFullU | Eigen::ComputeFullV);

        this->result_.memory_used = matrix_a_.size() * sizeof(T) + svd_.singularValues().size() * sizeof(T) +
                                   svd_.matrixU().size() * sizeof(T) + svd_.matrixV().size() * sizeof(T);
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        size_ = config.data_size;
        this->result_.operations = 4ULL * size_ * size_ * size_;

        matrix_a_.resize(size_, size_);

        // Initialize matrix with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_a_(i, j) = dis(gen);
                }
            }
        } else if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_a_(i, j) = dis(gen);
                }
            }
        }
    }
};