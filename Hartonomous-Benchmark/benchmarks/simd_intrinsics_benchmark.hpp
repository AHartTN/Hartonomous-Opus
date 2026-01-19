#pragma once

#include <vector>
#include <random>
#include <immintrin.h>
#include <limits>
#include "../src/benchmark/benchmark_base.hpp"
#include "../src/hardware.hpp"

// SIMD Intrinsics Benchmark
// Tests various SIMD operations using AVX/AVX2/AVX512 intrinsics
// Includes vector addition, multiplication, and fused multiply-add operations

template<typename T>
class SimdIntrinsicsBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<T> input_a_;
    std::vector<T> input_b_;
    std::vector<T> output_;
    size_t vector_size_;
    HardwareDetector hw_detector_;
    CpuFeatures cpu_features_;

#ifdef __AVX512F__
    // AVX-512 version for float/double
    void execute_avx512_float() {
        const size_t num_vectors = vector_size_ / 16; // AVX-512 processes 16 floats at once

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 16;
            __m512 va = _mm512_loadu_ps(&input_a_[offset]);
            __m512 vb = _mm512_loadu_ps(&input_b_[offset]);
            __m512 result = _mm512_fmadd_ps(va, vb, _mm512_set1_ps(1.0f));
            _mm512_storeu_ps(&output_[offset], result);
        }
    }

    void execute_avx512_double() {
        const size_t num_vectors = vector_size_ / 8; // AVX-512 processes 8 doubles at once

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 8;
            __m512d va = _mm512_loadu_pd(&input_a_[offset]);
            __m512d vb = _mm512_loadu_pd(&input_b_[offset]);
            __m512d result = _mm512_fmadd_pd(va, vb, _mm512_set1_pd(1.0));
            _mm512_storeu_pd(&output_[offset], result);
        }
    }
#endif

    // AVX2 version for float/double
    void execute_avx2_float() {
        const size_t num_vectors = vector_size_ / 8; // AVX2 processes 8 floats at once

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 8;
            __m256 va = _mm256_loadu_ps(&input_a_[offset]);
            __m256 vb = _mm256_loadu_ps(&input_b_[offset]);
            __m256 result = _mm256_fmadd_ps(va, vb, _mm256_set1_ps(1.0f));
            _mm256_storeu_ps(&output_[offset], result);
        }
    }

    void execute_avx2_double() {
        const size_t num_vectors = vector_size_ / 4; // AVX2 processes 4 doubles at once

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 4;
            __m256d va = _mm256_loadu_pd(&input_a_[offset]);
            __m256d vb = _mm256_loadu_pd(&input_b_[offset]);
            __m256d result = _mm256_fmadd_pd(va, vb, _mm256_set1_pd(1.0));
            _mm256_storeu_pd(&output_[offset], result);
        }
    }

    // AVX version (fallback)
    void execute_avx_float() {
        const size_t num_vectors = vector_size_ / 8; // AVX processes 8 floats at once

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 8;
            __m256 va = _mm256_loadu_ps(&input_a_[offset]);
            __m256 vb = _mm256_loadu_ps(&input_b_[offset]);
            __m256 vmul = _mm256_mul_ps(va, vb);
            __m256 vadd = _mm256_add_ps(vmul, _mm256_set1_ps(1.0f));
            _mm256_storeu_ps(&output_[offset], vadd);
        }
    }

    void execute_avx_double() {
        const size_t num_vectors = vector_size_ / 4; // AVX processes 4 doubles at once

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 4;
            __m256d va = _mm256_loadu_pd(&input_a_[offset]);
            __m256d vb = _mm256_loadu_pd(&input_b_[offset]);
            __m256d vmul = _mm256_mul_pd(va, vb);
            __m256d vadd = _mm256_add_pd(vmul, _mm256_set1_pd(1.0));
            _mm256_storeu_pd(&output_[offset], vadd);
        }
    }

    // Scalar fallback
    void execute_scalar() {
        for (size_t i = 0; i < vector_size_; ++i) {
            output_[i] = input_a_[i] * input_b_[i] + 1.0;
        }
    }

public:
    std::string get_name() const override {
        std::string type_name = (std::is_same_v<T, float>) ? "float" : "double";
        std::string isa_name;
        if (cpu_features_.avx512f) isa_name = "AVX512";
        else if (cpu_features_.avx2) isa_name = "AVX2";
        else if (cpu_features_.avx) isa_name = "AVX";
        else isa_name = "SCALAR";

        return "SIMD_INTRINSICS_" + isa_name + "_" + type_name + "_" + std::to_string(vector_size_);
    }

    void execute_operation() override {
        cpu_features_ = hw_detector_.detect_cpu_features();

        // Choose the best available SIMD instruction set
        if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
            if (cpu_features_.avx512f) {
                execute_avx512_float();
            } else
#endif
            if (cpu_features_.avx2) {
                execute_avx2_float();
            } else if (cpu_features_.avx) {
                execute_avx_float();
            } else {
                execute_scalar();
            }
        } else if constexpr (std::is_same_v<T, double>) {
#ifdef __AVX512F__
            if (cpu_features_.avx512f) {
                execute_avx512_double();
            } else
#endif
            if (cpu_features_.avx2) {
                execute_avx2_double();
            } else if (cpu_features_.avx) {
                execute_avx_double();
            } else {
                execute_scalar();
            }
        }

        // Calculate throughput (FLOPS: 2 operations per element - mul + add)
        size_t operations = 2ULL * vector_size_;
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = operations / seconds;
        this->result_.memory_used = (input_a_.size() + input_b_.size() + output_.size()) * sizeof(T);
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        vector_size_ = config.data_size;

        // Ensure vector size is aligned for SIMD operations
        size_t alignment;
        if (cpu_features_.avx512f) {
            alignment = (std::is_same_v<T, float>) ? 16 : 8;
        } else if (cpu_features_.avx2 || cpu_features_.avx) {
            alignment = (std::is_same_v<T, float>) ? 8 : 4;
        } else {
            alignment = 1;
        }

        vector_size_ = (vector_size_ / alignment) * alignment;
        if (vector_size_ == 0) vector_size_ = alignment;

        input_a_.resize(vector_size_);
        input_b_.resize(vector_size_);
        output_.resize(vector_size_);

        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (auto& val : input_a_) val = dis(gen);
            for (auto& val : input_b_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (auto& val : input_a_) val = dis(gen);
            for (auto& val : input_b_) val = dis(gen);
        }
    }
};