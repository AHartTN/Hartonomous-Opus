#pragma once

#include <vector>
#include <random>
#include <immintrin.h>
#include <limits>
#include "../src/benchmark/benchmark_base.hpp"
#include "../src/hardware.hpp"

// AVX Vector Arithmetic Benchmark
// Tests basic vector arithmetic operations using AVX/AVX2/AVX512 instructions
// Covers addition, subtraction, multiplication, division, and compound operations

template<typename T>
class AvxVectorArithmeticBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    std::vector<T> input_a_;
    std::vector<T> input_b_;
    std::vector<T> input_c_;
    std::vector<T> output_add_;
    std::vector<T> output_sub_;
    std::vector<T> output_mul_;
    std::vector<T> output_div_;
    std::vector<T> output_compound_;
    size_t vector_size_;
    HardwareDetector hw_detector_;
    CpuFeatures cpu_features_;

    // AVX-512 versions
    void execute_avx512_float_arithmetic() {
        const size_t num_vectors = vector_size_ / 16;

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 16;
            __m512 va = _mm512_loadu_ps(&input_a_[offset]);
            __m512 vb = _mm512_loadu_ps(&input_b_[offset]);
            __m512 vc = _mm512_loadu_ps(&input_c_[offset]);

            // Basic arithmetic operations
            __m512 vadd = _mm512_add_ps(va, vb);
            __m512 vsub = _mm512_sub_ps(va, vb);
            __m512 vmul = _mm512_mul_ps(va, vb);
            __m512 vdiv = _mm512_div_ps(va, _mm512_add_ps(vb, _mm512_set1_ps(1.0f))); // Avoid division by zero

            // Compound operation: (a + b) * c - a / (b + 1)
            __m512 vcompound = _mm512_sub_ps(_mm512_mul_ps(vadd, vc), vdiv);

            _mm512_storeu_ps(&output_add_[offset], vadd);
            _mm512_storeu_ps(&output_sub_[offset], vsub);
            _mm512_storeu_ps(&output_mul_[offset], vmul);
            _mm512_storeu_ps(&output_div_[offset], vdiv);
            _mm512_storeu_ps(&output_compound_[offset], vcompound);
        }
    }

    void execute_avx512_double_arithmetic() {
        const size_t num_vectors = vector_size_ / 8;

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 8;
            __m512d va = _mm512_loadu_pd(&input_a_[offset]);
            __m512d vb = _mm512_loadu_pd(&input_b_[offset]);
            __m512d vc = _mm512_loadu_pd(&input_c_[offset]);

            __m512d vadd = _mm512_add_pd(va, vb);
            __m512d vsub = _mm512_sub_pd(va, vb);
            __m512d vmul = _mm512_mul_pd(va, vb);
            __m512d vdiv = _mm512_div_pd(va, _mm512_add_pd(vb, _mm512_set1_pd(1.0)));

            __m512d vcompound = _mm512_sub_pd(_mm512_mul_pd(vadd, vc), vdiv);

            _mm512_storeu_pd(&output_add_[offset], vadd);
            _mm512_storeu_pd(&output_sub_[offset], vsub);
            _mm512_storeu_pd(&output_mul_[offset], vmul);
            _mm512_storeu_pd(&output_div_[offset], vdiv);
            _mm512_storeu_pd(&output_compound_[offset], vcompound);
        }
    }

    // AVX2 versions
    void execute_avx2_float_arithmetic() {
        const size_t num_vectors = vector_size_ / 8;

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 8;
            __m256 va = _mm256_loadu_ps(&input_a_[offset]);
            __m256 vb = _mm256_loadu_ps(&input_b_[offset]);
            __m256 vc = _mm256_loadu_ps(&input_c_[offset]);

            __m256 vadd = _mm256_add_ps(va, vb);
            __m256 vsub = _mm256_sub_ps(va, vb);
            __m256 vmul = _mm256_mul_ps(va, vb);
            __m256 vdiv = _mm256_div_ps(va, _mm256_add_ps(vb, _mm256_set1_ps(1.0f)));

            __m256 vcompound = _mm256_sub_ps(_mm256_mul_ps(vadd, vc), vdiv);

            _mm256_storeu_ps(&output_add_[offset], vadd);
            _mm256_storeu_ps(&output_sub_[offset], vsub);
            _mm256_storeu_ps(&output_mul_[offset], vmul);
            _mm256_storeu_ps(&output_div_[offset], vdiv);
            _mm256_storeu_ps(&output_compound_[offset], vcompound);
        }
    }

    void execute_avx2_double_arithmetic() {
        const size_t num_vectors = vector_size_ / 4;

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 4;
            __m256d va = _mm256_loadu_pd(&input_a_[offset]);
            __m256d vb = _mm256_loadu_pd(&input_b_[offset]);
            __m256d vc = _mm256_loadu_pd(&input_c_[offset]);

            __m256d vadd = _mm256_add_pd(va, vb);
            __m256d vsub = _mm256_sub_pd(va, vb);
            __m256d vmul = _mm256_mul_pd(va, vb);
            __m256d vdiv = _mm256_div_pd(va, _mm256_add_pd(vb, _mm256_set1_pd(1.0)));

            __m256d vcompound = _mm256_sub_pd(_mm256_mul_pd(vadd, vc), vdiv);

            _mm256_storeu_pd(&output_add_[offset], vadd);
            _mm256_storeu_pd(&output_sub_[offset], vsub);
            _mm256_storeu_pd(&output_mul_[offset], vmul);
            _mm256_storeu_pd(&output_div_[offset], vdiv);
            _mm256_storeu_pd(&output_compound_[offset], vcompound);
        }
    }

    // AVX versions (256-bit float/double)
    void execute_avx_float_arithmetic() {
        const size_t num_vectors = vector_size_ / 8;

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 8;
            __m256 va = _mm256_loadu_ps(&input_a_[offset]);
            __m256 vb = _mm256_loadu_ps(&input_b_[offset]);
            __m256 vc = _mm256_loadu_ps(&input_c_[offset]);

            __m256 vadd = _mm256_add_ps(va, vb);
            __m256 vsub = _mm256_sub_ps(va, vb);
            __m256 vmul = _mm256_mul_ps(va, vb);
            __m256 vdiv = _mm256_div_ps(va, _mm256_add_ps(vb, _mm256_set1_ps(1.0f)));

            __m256 vcompound = _mm256_sub_ps(_mm256_mul_ps(vadd, vc), vdiv);

            _mm256_storeu_ps(&output_add_[offset], vadd);
            _mm256_storeu_ps(&output_sub_[offset], vsub);
            _mm256_storeu_ps(&output_mul_[offset], vmul);
            _mm256_storeu_ps(&output_div_[offset], vdiv);
            _mm256_storeu_ps(&output_compound_[offset], vcompound);
        }
    }

    void execute_avx_double_arithmetic() {
        const size_t num_vectors = vector_size_ / 4;

        for (size_t i = 0; i < num_vectors; ++i) {
            size_t offset = i * 4;
            __m256d va = _mm256_loadu_pd(&input_a_[offset]);
            __m256d vb = _mm256_loadu_pd(&input_b_[offset]);
            __m256d vc = _mm256_loadu_pd(&input_c_[offset]);

            __m256d vadd = _mm256_add_pd(va, vb);
            __m256d vsub = _mm256_sub_pd(va, vb);
            __m256d vmul = _mm256_mul_pd(va, vb);
            __m256d vdiv = _mm256_div_pd(va, _mm256_add_pd(vb, _mm256_set1_pd(1.0)));

            __m256d vcompound = _mm256_sub_pd(_mm256_mul_pd(vadd, vc), vdiv);

            _mm256_storeu_pd(&output_add_[offset], vadd);
            _mm256_storeu_pd(&output_sub_[offset], vsub);
            _mm256_storeu_pd(&output_mul_[offset], vmul);
            _mm256_storeu_pd(&output_div_[offset], vdiv);
            _mm256_storeu_pd(&output_compound_[offset], vcompound);
        }
    }

    // Scalar fallback
    void execute_scalar_arithmetic() {
        for (size_t i = 0; i < vector_size_; ++i) {
            T safe_b = input_b_[i] + 1.0; // Avoid division by zero
            output_add_[i] = input_a_[i] + input_b_[i];
            output_sub_[i] = input_a_[i] - input_b_[i];
            output_mul_[i] = input_a_[i] * input_b_[i];
            output_div_[i] = input_a_[i] / safe_b;
            output_compound_[i] = (input_a_[i] + input_b_[i]) * input_c_[i] - input_a_[i] / safe_b;
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

        return "AVX_VECTOR_ARITHMETIC_" + isa_name + "_" + type_name + "_" + std::to_string(vector_size_);
    }

    void execute_operation() override {
        cpu_features_ = hw_detector_.detect_cpu_features();

        // Choose the best available SIMD instruction set
        if constexpr (std::is_same_v<T, float>) {
            // if (cpu_features_.avx512f) {
            //     execute_avx512_float_arithmetic();
            // } else
            if (cpu_features_.avx2) {
                execute_avx2_float_arithmetic();
            } else if (cpu_features_.avx) {
                execute_avx_float_arithmetic();
            } else {
                execute_scalar_arithmetic();
            }
        } else if constexpr (std::is_same_v<T, double>) {
            // if (cpu_features_.avx512f) {
            //     execute_avx512_double_arithmetic();
            // } else
            if (cpu_features_.avx2) {
                execute_avx2_double_arithmetic();
            } else if (cpu_features_.avx) {
                execute_avx_double_arithmetic();
            } else {
                execute_scalar_arithmetic();
            }
        }

        // Calculate throughput (FLOPS: 5 operations per element - add, sub, mul, div, compound)
        size_t operations = 5ULL * vector_size_;
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = operations / seconds;
        this->result_.memory_used = (input_a_.size() + input_b_.size() + input_c_.size() +
                                    output_add_.size() + output_sub_.size() + output_mul_.size() +
                                    output_div_.size() + output_compound_.size()) * sizeof(T);
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
        input_c_.resize(vector_size_);
        output_add_.resize(vector_size_);
        output_sub_.resize(vector_size_);
        output_mul_.resize(vector_size_);
        output_div_.resize(vector_size_);
        output_compound_.resize(vector_size_);

        std::random_device rd;
        std::mt19937 gen(rd());
        if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-10.0, 10.0);
            for (auto& val : input_a_) val = dis(gen);
            for (auto& val : input_b_) val = dis(gen);
            for (auto& val : input_c_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
            for (auto& val : input_a_) val = dis(gen);
            for (auto& val : input_b_) val = dis(gen);
            for (auto& val : input_c_) val = dis(gen);
        }
    }
};