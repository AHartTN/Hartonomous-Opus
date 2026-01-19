#pragma once

#include <vector>
#include <random>
#ifdef HAVE_MKL
#include <mkl_cblas.h>
#endif
#include "../src/benchmark/benchmark_base.hpp"
#include "../src/timing/timer.hpp"

// Example usage:
// MatrixMultiplyBenchmark<double> bench;
// BenchmarkConfig config{1000, 1024, false, "double"};
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class MatrixMultiplyBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<T> matrix_a_;
    std::vector<T> matrix_b_;
    std::vector<T> matrix_c_;
    size_t size_;

public:
    std::string get_name() const override {
        return "MKL_DGEMM_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        // Use MKL DGEMM for double precision matrix multiplication
        // C = alpha * A * B + beta * C, with alpha=1, beta=0
        // Matrices are stored in column-major order
        if constexpr (std::is_same_v<T, double>) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                       size_, size_, size_, 1.0,
                       matrix_a_.data(), size_,
                       matrix_b_.data(), size_,
                       0.0, matrix_c_.data(), size_);
        } else if constexpr (std::is_same_v<T, float>) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                       size_, size_, size_, 1.0f,
                       matrix_a_.data(), size_,
                       matrix_b_.data(), size_,
                       0.0f, matrix_c_.data(), size_);
        } else {
            throw std::runtime_error("Unsupported data type for MKL DGEMM");
        }

        this->result_.memory_used = (matrix_a_.size() + matrix_b_.size() + matrix_c_.size()) * sizeof(T);
    }



    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        size_ = config.data_size;
        this->result_.name = get_name();
        this->result_.success = true;
        this->result_.operations = 2ULL * size_ * size_ * size_;

        matrix_a_.resize(size_ * size_);
        matrix_b_.resize(size_ * size_);
        matrix_c_.resize(size_ * size_);

        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (auto& val : matrix_a_) val = dis(gen);
            for (auto& val : matrix_b_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (auto& val : matrix_a_) val = dis(gen);
            for (auto& val : matrix_b_) val = dis(gen);
        }
    }
};