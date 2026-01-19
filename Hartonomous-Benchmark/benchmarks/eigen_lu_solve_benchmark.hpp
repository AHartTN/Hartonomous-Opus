#pragma once

#include <Eigen/Dense>
#include "../src/benchmark/benchmark_base.hpp"

// Example usage:
// EigenLUSolveBenchmark<double> bench;
// BenchmarkConfig config{1000, 512, false, "double"};
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class EigenLUSolveBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_a_;
    Eigen::Vector<T, Eigen::Dynamic> vector_b_;
    Eigen::Vector<T, Eigen::Dynamic> vector_x_;
    size_t size_;

public:
    std::string get_name() const override {
        return "Eigen_LU_Solve_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        // Perform LU decomposition and solve AX = B
        Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> lu(matrix_a_);
        vector_x_ = lu.solve(vector_b_);

        this->result_.memory_used = (matrix_a_.size() + vector_b_.size() + vector_x_.size()) * sizeof(T);
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        size_ = config.data_size;
        this->result_.operations = size_ * size_ * size_;

        matrix_a_.resize(size_, size_);
        vector_b_.resize(size_);
        vector_x_.resize(size_);

        // Initialize matrix and vector with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_a_(i, j) = dis(gen);
                }
                vector_b_(i) = dis(gen);
            }
        } else if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_a_(i, j) = dis(gen);
                }
                vector_b_(i) = dis(gen);
            }
        }

        // Make matrix diagonally dominant for numerical stability
        for (Eigen::Index i = 0; i < size_; ++i) {
            matrix_a_(i, i) += static_cast<T>(size_);
        }
    }
};