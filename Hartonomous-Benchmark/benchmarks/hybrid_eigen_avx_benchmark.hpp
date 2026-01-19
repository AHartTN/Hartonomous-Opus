#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>
#include <immintrin.h>
#include "../src/benchmark/benchmark_base.hpp"

// Hybrid Eigen + AVX benchmark for matrix operations
template<typename T>
class HybridEigen_AVX_Benchmark : public TemplatedBenchmark<T, std::string> {
private:
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    Matrix matrix_a_;
    Matrix matrix_b_;
    Matrix matrix_c_;
    Vector vector_x_;
    Vector vector_y_;
    size_t size_;

public:
    std::string get_name() const override {
        return "Hybrid_Eigen_AVX_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        // Enable AVX in Eigen (if available)
        Eigen::setNbThreads(1); // Single thread for consistent benchmarking

        // Matrix multiplication with AVX
        matrix_c_ = matrix_a_ * matrix_b_;

        // Vector operations with AVX
        vector_y_ = matrix_a_ * vector_x_;

        // Additional AVX-accelerated operations
        // Element-wise operations using AVX
        if constexpr (std::is_same_v<T, float>) {
            for (Eigen::Index i = 0; i < vector_y_.size(); i += 8) {
                if (i + 8 <= vector_y_.size()) {
                    __m256 va = _mm256_loadu_ps(vector_y_.data() + i);
                    __m256 vb = _mm256_set1_ps(2.0f);
                    __m256 vr = _mm256_mul_ps(va, vb);
                    _mm256_storeu_ps(vector_y_.data() + i, vr);
                }
            }
        } else if constexpr (std::is_same_v<T, double>) {
            for (Eigen::Index i = 0; i < vector_y_.size(); i += 4) {
                if (i + 4 <= vector_y_.size()) {
                    __m256d va = _mm256_loadu_pd(vector_y_.data() + i);
                    __m256d vb = _mm256_set1_pd(2.0);
                    __m256d vr = _mm256_mul_pd(va, vb);
                    _mm256_storeu_pd(vector_y_.data() + i, vr);
                }
            }
        }

        // Calculate throughput
        // Matrix multiply: 2 * size^3 operations, vector multiply: 2 * size^2 operations
        size_t matrix_ops = 2ULL * size_ * size_ * size_;
        size_t vector_ops = 2ULL * size_ * size_;
        size_t total_operations = matrix_ops + vector_ops;

        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = total_operations / seconds;

        // Memory usage
        this->result_.memory_used = (matrix_a_.size() + matrix_b_.size() +
                                   matrix_c_.size() + vector_x_.size() + vector_y_.size()) * sizeof(T);
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        size_ = config.data_size;

        matrix_a_.resize(size_, size_);
        matrix_b_.resize(size_, size_);
        matrix_c_.resize(size_, size_);
        vector_x_.resize(size_);
        vector_y_.resize(size_);

        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_a_(i, j) = dis(gen);
                    matrix_b_(i, j) = dis(gen);
                }
                vector_x_(i) = dis(gen);
            }
        } else if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_a_(i, j) = dis(gen);
                    matrix_b_(i, j) = dis(gen);
                }
                vector_x_(i) = dis(gen);
            }
        } else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(-10, 10);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_a_(i, j) = dis(gen);
                    matrix_b_(i, j) = dis(gen);
                }
                vector_x_(i) = dis(gen);
            }
        }
    }
};