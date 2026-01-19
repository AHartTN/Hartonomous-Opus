#pragma once

#include <vector>
#include <random>
#include <immintrin.h>
#include <limits>
#include <cstdint>
#include "../src/benchmark/benchmark_base.hpp"
#include "../src/hardware.hpp"

// VNNI Dot Product Benchmark
// Tests VNNI instructions for dot product accumulation
// VNNI provides specialized instructions for efficient dot products in neural networks

template<typename T>
class VnniDotProductBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<T> vector_a_;
    std::vector<T> vector_b_;
    std::vector<T> results_;
    size_t vector_size_;
    HardwareDetector hw_detector_;
    CpuFeatures cpu_features_;

#ifdef __AVX512F__
    // AVX-512 VNNI version for int8 dot products
    void execute_avx512_vnni_int8() {
        const size_t num_elements = vector_size_ / 64; // Process 64 elements at a time (4x 16-element vectors)

        for (size_t i = 0; i < num_elements; ++i) {
            size_t offset = i * 64;

            // Load 4 pairs of 16 int8 values each
            __m512i va1 = _mm512_loadu_si512(&vector_a_[offset]);
            __m512i vb1 = _mm512_loadu_si512(&vector_b_[offset]);
            __m512i va2 = _mm512_loadu_si512(&vector_a_[offset + 16]);
            __m512i vb2 = _mm512_loadu_si512(&vector_b_[offset + 16]);
            __m512i va3 = _mm512_loadu_si512(&vector_a_[offset + 32]);
            __m512i vb3 = _mm512_loadu_si512(&vector_b_[offset + 32]);
            __m512i va4 = _mm512_loadu_si512(&vector_a_[offset + 48]);
            __m512i vb4 = _mm512_loadu_si512(&vector_b_[offset + 48]);

            // Perform VNNI dot product accumulation
            __m512i result1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), va1, vb1);
            __m512i result2 = _mm512_dpbusd_epi32(result1, va2, vb2);
            __m512i result3 = _mm512_dpbusd_epi32(result2, va3, vb3);
            __m512i result4 = _mm512_dpbusd_epi32(result3, va4, vb4);

            // Store accumulated result (single int32 value)
            _mm512_storeu_si512(&results_[i * 16], result4);
        }
    }

    // AVX-512 VNNI version for int16 dot products (if supported)
    void execute_avx512_vnni_int16() {
        const size_t num_elements = vector_size_ / 32; // Process 32 elements at a time

        for (size_t i = 0; i < num_elements; ++i) {
            size_t offset = i * 32;

            __m512i va = _mm512_loadu_si512(&vector_a_[offset]);
            __m512i vb = _mm512_loadu_si512(&vector_b_[offset]);

            // Use VNNI for 16-bit dot product
            __m512i result = _mm512_dpbusd_epi32(_mm512_setzero_si512(), va, vb);
            _mm512_storeu_si512(&results_[i * 16], result);
        }
    }
#endif

    // AVX2 fallback for int8 dot products - use scalar since AVX2 doesn't support int8 mul
    void execute_avx2_int8() {
        execute_scalar();
    }

    // Scalar fallback
    void execute_scalar() {
        T accumulator = 0;
        for (size_t i = 0; i < vector_size_; ++i) {
            accumulator += static_cast<T>(vector_a_[i]) * static_cast<T>(vector_b_[i]);
        }
        results_[0] = accumulator;
    }

    // Float/double dot product using AVX
    void execute_avx_float_dot_product() {
        const size_t num_vectors = vector_size_ / 8; // AVX processes 8 floats at once

        __m256 accumulator = _mm256_setzero_ps();
        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 8;
            __m256 va = _mm256_loadu_ps(&vector_a_[offset]);
            __m256 vb = _mm256_loadu_ps(&vector_b_[offset]);
            __m256 product = _mm256_mul_ps(va, vb);
            accumulator = _mm256_add_ps(accumulator, product);
        }

        // Horizontal sum
        __m256 hsum = _mm256_hadd_ps(accumulator, accumulator);
        hsum = _mm256_hadd_ps(hsum, hsum);
        float result[8];
        _mm256_storeu_ps(result, hsum);
        results_[0] = result[0] + result[4];
    }

    void execute_avx_double_dot_product() {
        const size_t num_vectors = vector_size_ / 4; // AVX processes 4 doubles at once

        __m256d accumulator = _mm256_setzero_pd();
        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 4;
            __m256d va = _mm256_loadu_pd(&vector_a_[offset]);
            __m256d vb = _mm256_loadu_pd(&vector_b_[offset]);
            __m256d product = _mm256_mul_pd(va, vb);
            accumulator = _mm256_add_pd(accumulator, product);
        }

        // Horizontal sum
        __m256d hsum = _mm256_hadd_pd(accumulator, accumulator);
        double result[4];
        _mm256_storeu_pd(result, hsum);
        results_[0] = result[0] + result[2];
    }

public:
    std::string get_name() const override {
        std::string type_name;
        if constexpr (std::is_same_v<T, int8_t>) type_name = "int8";
        else if constexpr (std::is_same_v<T, int16_t>) type_name = "int16";
        else if constexpr (std::is_same_v<T, float>) type_name = "float";
        else if constexpr (std::is_same_v<T, double>) type_name = "double";

        std::string isa_name;
        if (cpu_features_.vnni && cpu_features_.avx512f) isa_name = "VNNI_AVX512";
        else if (cpu_features_.avx512f) isa_name = "AVX512";
        else if (cpu_features_.avx2) isa_name = "AVX2";
        else if (cpu_features_.avx) isa_name = "AVX";
        else isa_name = "SCALAR";

        return "VNNI_DOT_PRODUCT_" + isa_name + "_" + type_name + "_" + std::to_string(vector_size_);
    }

    void execute_operation() override {
        cpu_features_ = hw_detector_.detect_cpu_features();

        // Choose the best available instruction set for dot products
        if constexpr (std::is_same_v<T, int8_t>) {
#ifdef __AVX512F__
            if (cpu_features_.vnni && cpu_features_.avx512f) {
                execute_avx512_vnni_int8();
            } else if (cpu_features_.avx512f) {
                execute_avx512_vnni_int8(); // Fallback to regular AVX512
            } else
#endif
            if (cpu_features_.avx2) {
                execute_avx2_int8();
            } else {
                execute_scalar();
            }
        } else if constexpr (std::is_same_v<T, int16_t>) {
#ifdef __AVX512F__
            if (cpu_features_.vnni && cpu_features_.avx512f) {
                execute_avx512_vnni_int16();
            } else
#endif
            {
                execute_scalar();
            }
        } else if constexpr (std::is_same_v<T, float>) {
            if (cpu_features_.avx) {
                execute_avx_float_dot_product();
            } else {
                execute_scalar();
            }
        } else if constexpr (std::is_same_v<T, double>) {
            if (cpu_features_.avx) {
                execute_avx_double_dot_product();
            } else {
                execute_scalar();
            }
        }

        // Calculate throughput (operations per second)
        size_t operations = vector_size_; // One multiply-accumulate per element
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = operations / seconds;
        this->result_.memory_used = (vector_a_.size() + vector_b_.size() + results_.size()) * sizeof(T);
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        vector_size_ = config.data_size;

        // Align vector size for SIMD operations
        size_t alignment = 1;
        if (cpu_features_.avx512f) {
            alignment = 64; // AVX-512 processes 64 bytes at a time for int8
        } else if (cpu_features_.avx2 || cpu_features_.avx) {
            alignment = 32; // AVX2/AVX processes 32 bytes at a time
        }

        vector_size_ = (vector_size_ / alignment) * alignment;
        if (vector_size_ == 0) vector_size_ = alignment;

        vector_a_.resize(vector_size_);
        vector_b_.resize(vector_size_);
        results_.resize(vector_size_ / alignment * 16); // Store accumulated results

        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_same_v<T, int8_t>) {
            std::uniform_int_distribution<int16_t> dis(-127, 127);
            for (auto& val : vector_a_) val = static_cast<int8_t>(dis(gen));
            for (auto& val : vector_b_) val = static_cast<int8_t>(dis(gen));
        } else if constexpr (std::is_same_v<T, int16_t>) {
            std::uniform_int_distribution<int16_t> dis(-32767, 32767);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        }
    }
};