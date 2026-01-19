#pragma once

#include <vector>
#include <random>
#ifdef HAVE_MKL
#include <mkl_lapacke.h>
#endif
#include "../src/benchmark/benchmark_base.hpp"

// Example usage:
// LinearSolveBenchmark<double> bench;
// BenchmarkConfig config{1000, 512, false, "double"};
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class LinearSolveBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<T> matrix_a_;
    std::vector<T> vector_b_;
    std::vector<T> vector_x_;
    std::vector<lapack_int> ipiv_;
    size_t size_;

public:
    std::string get_name() const override {
        return "MKL_DGESV_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
#ifdef HAVE_MKL
        // Use MKL DGESV to solve AX = B
        // A is overwritten with LU factorization, B with solution X
        if constexpr (std::is_same_v<T, double>) {
            int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, size_, 1,
                                    matrix_a_.data(), size_,
                                    ipiv_.data(), vector_b_.data(), size_);
            if (info != 0) {
                throw std::runtime_error("DGESV failed with info: " + std::to_string(info));
            }
            vector_x_ = vector_b_; // Solution is in b
        } else if constexpr (std::is_same_v<T, float>) {
            int info = LAPACKE_sgesv(LAPACK_COL_MAJOR, size_, 1,
                                    matrix_a_.data(), size_,
                                    ipiv_.data(), vector_b_.data(), size_);
            if (info != 0) {
                throw std::runtime_error("SGESV failed with info: " + std::to_string(info));
            }
            vector_x_ = vector_b_;
        } else {
            throw std::runtime_error("Unsupported data type for MKL DGESV");
        }
#else
        throw std::runtime_error("MKL not available");
#endif

        this->result_.memory_used = (matrix_a_.size() + vector_b_.size() + vector_x_.size()) * sizeof(T) + ipiv_.size() * sizeof(lapack_int);
    }



    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        size_ = config.data_size;
        this->result_.name = get_name();
        this->result_.success = true;
        this->result_.operations = size_ * size_ * size_;

        matrix_a_.resize(size_ * size_);
        vector_b_.resize(size_);
        vector_x_.resize(size_);
        ipiv_.resize(size_);

        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (auto& val : matrix_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (auto& val : matrix_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        }

        // Make matrix diagonally dominant for numerical stability
        for (size_t i = 0; i < size_; ++i) {
            matrix_a_[i * size_ + i] += size_;
        }
    }
};