#pragma once

#include <vector>
#include <complex>
#include <random>
#include <mkl_dfti.h>
#include "../src/benchmark/benchmark_base.hpp"

// Example usage:
// FFTBenchmark<std::complex<double>> bench;
// BenchmarkConfig config{1000, 1024, false, "complex<double>"};
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class FFTBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<T> input_;
    std::vector<T> output_;
    DFTI_DESCRIPTOR_HANDLE handle_;
    size_t size_;

public:
    std::string get_name() const override {
        return "MKL_DFTI_FFT_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        // Perform forward FFT using MKL DFTI
        MKL_LONG status = DftiComputeForward(handle_, input_.data(), output_.data());
        if (status != DFTI_NO_ERROR) {
            throw std::runtime_error("DFTI forward FFT failed");
        }

        this->result_.memory_used = (input_.size() + output_.size()) * sizeof(T);
    }



    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        size_ = config.data_size;

        // Calculate operations: 5 * size * log2(size) for FFT
        size_t log_size = 0;
        size_t temp = size_;
        while (temp >>= 1) ++log_size;
        this->result_.operations = 5ULL * size_ * log_size;

        input_.resize(size_);
        output_.resize(size_);

        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (auto& val : input_) {
                val = std::complex<double>(dis(gen), dis(gen));
            }
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (auto& val : input_) {
                val = std::complex<float>(dis(gen), dis(gen));
            }
        } else {
            throw std::runtime_error("Unsupported data type for MKL DFTI FFT");
        }

        // Create DFTI descriptor
        MKL_LONG status = DftiCreateDescriptor(&handle_, DFTI_DOUBLE, DFTI_COMPLEX, 1, size_);
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            status = DftiCreateDescriptor(&handle_, DFTI_SINGLE, DFTI_COMPLEX, 1, size_);
        }
        if (status != DFTI_NO_ERROR) {
            throw std::runtime_error("Failed to create DFTI descriptor");
        }

        // Set placement to out-of-place
        status = DftiSetValue(handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (status != DFTI_NO_ERROR) {
            DftiFreeDescriptor(&handle_);
            throw std::runtime_error("Failed to set DFTI placement");
        }

        // Commit descriptor
        status = DftiCommitDescriptor(handle_);
        if (status != DFTI_NO_ERROR) {
            DftiFreeDescriptor(&handle_);
            throw std::runtime_error("Failed to commit DFTI descriptor");
        }
    }

    ~FFTBenchmark() {
        if (handle_) {
            DftiFreeDescriptor(&handle_);
        }
    }
};