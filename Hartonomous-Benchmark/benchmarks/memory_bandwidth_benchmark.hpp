#pragma once

#include <vector>
#include <random>
#include <cstring>
#include <immintrin.h>
#include "../src/benchmark/benchmark_base.hpp"
#include "../src/math/vector_math.hpp"

// Memory bandwidth benchmark
template<typename T>
class MemoryBandwidthBenchmark : public TemplatedBenchmark<T, std::string> {
protected:
    std::vector<T> test_data_;
    std::vector<T> result_data_;
    size_t data_size_;
    size_t iterations_;
    size_t bytes_processed_ = 0;

    enum class TestType {
        READ_SEQUENTIAL,
        WRITE_SEQUENTIAL,
        READ_RANDOM,
        WRITE_RANDOM,
        STREAM_COPY,
        STREAM_SCALE
    };

    TestType test_type_;

public:
    MemoryBandwidthBenchmark(TestType type = TestType::READ_SEQUENTIAL)
        : test_type_(type), iterations_(100) {}

    std::string get_name() const override {
        std::string type_str;
        switch (test_type_) {
            case TestType::READ_SEQUENTIAL: type_str = "Read_Sequential"; break;
            case TestType::WRITE_SEQUENTIAL: type_str = "Write_Sequential"; break;
            case TestType::READ_RANDOM: type_str = "Read_Random"; break;
            case TestType::WRITE_RANDOM: type_str = "Write_Random"; break;
            case TestType::STREAM_COPY: type_str = "Stream_Copy"; break;
            case TestType::STREAM_SCALE: type_str = "Stream_Scale"; break;
        }
        return "Memory_Bandwidth_" + type_str + "_" + std::string(typeid(T).name()) + "_" + std::to_string(data_size_);
    }

    void execute_operation() override {
        switch (test_type_) {
            case TestType::READ_SEQUENTIAL:
                benchmark_read_sequential();
                break;
            case TestType::WRITE_SEQUENTIAL:
                benchmark_write_sequential();
                break;
            case TestType::READ_RANDOM:
                benchmark_read_random();
                break;
            case TestType::WRITE_RANDOM:
                benchmark_write_random();
                break;
            case TestType::STREAM_COPY:
                benchmark_stream_copy();
                break;
            case TestType::STREAM_SCALE:
                benchmark_stream_scale();
                break;
        }
    }

    void calculate_throughput(std::chrono::nanoseconds duration) override {
        this->result_.operations = bytes_processed_;
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
        this->result_.throughput = static_cast<double>(bytes_processed_) / seconds; // bytes/s
        this->result_.memory_used = bytes_processed_;
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        data_size_ = config.data_size;
        iterations_ = config.iterations > 0 ? config.iterations : 100;

        test_data_.resize(data_size_);
        result_data_.resize(data_size_);

        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (auto& val : test_data_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (auto& val : test_data_) val = dis(gen);
        } else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(-1000, 1000);
            for (auto& val : test_data_) val = dis(gen);
        }

        // Initialize result data
        std::fill(result_data_.begin(), result_data_.end(), T{0});
    }

private:
    void benchmark_read_sequential() {
        volatile T sum = 0;
        for (size_t iter = 0; iter < iterations_; ++iter) {
            for (const auto& val : test_data_) {
                sum += val;
            }
        }
        // Prevent optimization
        volatile T dummy = sum;

        bytes_processed_ = data_size_ * sizeof(T) * iterations_;
    }

    void benchmark_write_sequential() {
        for (size_t iter = 0; iter < iterations_; ++iter) {
            for (auto& val : result_data_) {
                val = static_cast<T>(iter);
            }
        }

        bytes_processed_ = data_size_ * sizeof(T) * iterations_;
    }

    void benchmark_read_random() {
        std::vector<size_t> indices(data_size_);
        for (size_t i = 0; i < data_size_; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        volatile T sum = 0;
        for (size_t iter = 0; iter < iterations_; ++iter) {
            for (size_t idx : indices) {
                sum += test_data_[idx];
            }
        }
        volatile T dummy = sum;

        bytes_processed_ = data_size_ * sizeof(T) * iterations_;
    }

    void benchmark_write_random() {
        std::vector<size_t> indices(data_size_);
        for (size_t i = 0; i < data_size_; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        for (size_t iter = 0; iter < iterations_; ++iter) {
            for (size_t idx : indices) {
                result_data_[idx] = static_cast<T>(iter + idx);
            }
        }

        bytes_processed_ = data_size_ * sizeof(T) * iterations_;
    }

    void benchmark_stream_copy() {
        // STREAM copy benchmark: result[i] = test_data[i]
        for (size_t iter = 0; iter < iterations_; ++iter) {
            if constexpr (std::is_same_v<T, float>) {
                size_t i = 0;
                for (; i + 8 <= data_size_; i += 8) {
                    __m256 va = _mm256_loadu_ps(&test_data_[i]);
                    _mm256_storeu_ps(&result_data_[i], va);
                }
                for (; i < data_size_; ++i) {
                    result_data_[i] = test_data_[i];
                }
            } else if constexpr (std::is_same_v<T, double>) {
                size_t i = 0;
                for (; i + 4 <= data_size_; i += 4) {
                    __m256d va = _mm256_loadu_pd(&test_data_[i]);
                    _mm256_storeu_pd(&result_data_[i], va);
                }
                for (; i < data_size_; ++i) {
                    result_data_[i] = test_data_[i];
                }
            } else {
                for (size_t i = 0; i < data_size_; ++i) {
                    result_data_[i] = test_data_[i];
                }
            }
        }

        bytes_processed_ = data_size_ * sizeof(T) * iterations_ * 2; // read + write
    }

    void benchmark_stream_scale() {
        // STREAM scale benchmark: result[i] = scalar * test_data[i]
        T scalar = static_cast<T>(3.0);
        for (size_t iter = 0; iter < iterations_; ++iter) {
            if constexpr (std::is_same_v<T, float>) {
                __m256 s = _mm256_set1_ps(scalar);
                size_t i = 0;
                for (; i + 8 <= data_size_; i += 8) {
                    __m256 va = _mm256_loadu_ps(&test_data_[i]);
                    __m256 vr = _mm256_mul_ps(va, s);
                    _mm256_storeu_ps(&result_data_[i], vr);
                }
                for (; i < data_size_; ++i) {
                    result_data_[i] = scalar * test_data_[i];
                }
            } else if constexpr (std::is_same_v<T, double>) {
                __m256d s = _mm256_set1_pd(scalar);
                size_t i = 0;
                for (; i + 4 <= data_size_; i += 4) {
                    __m256d va = _mm256_loadu_pd(&test_data_[i]);
                    __m256d vr = _mm256_mul_pd(va, s);
                    _mm256_storeu_pd(&result_data_[i], vr);
                }
                for (; i < data_size_; ++i) {
                    result_data_[i] = scalar * test_data_[i];
                }
            } else {
                for (size_t i = 0; i < data_size_; ++i) {
                    result_data_[i] = scalar * test_data_[i];
                }
            }
        }

        bytes_processed_ = data_size_ * sizeof(T) * iterations_ * 2; // read + write
    }
};

// Convenience classes for different test types
template<typename T>
class MemoryReadSequentialBenchmark : public MemoryBandwidthBenchmark<T> {
public:
    MemoryReadSequentialBenchmark() : MemoryBandwidthBenchmark<T>(
        MemoryBandwidthBenchmark<T>::TestType::READ_SEQUENTIAL) {}
};

template<typename T>
class MemoryWriteSequentialBenchmark : public MemoryBandwidthBenchmark<T> {
public:
    MemoryWriteSequentialBenchmark() : MemoryBandwidthBenchmark<T>(
        MemoryBandwidthBenchmark<T>::TestType::WRITE_SEQUENTIAL) {}
};

template<typename T>
class MemoryReadRandomBenchmark : public MemoryBandwidthBenchmark<T> {
public:
    MemoryReadRandomBenchmark() : MemoryBandwidthBenchmark<T>(
        MemoryBandwidthBenchmark<T>::TestType::READ_RANDOM) {}
};

template<typename T>
class MemoryWriteRandomBenchmark : public MemoryBandwidthBenchmark<T> {
public:
    MemoryWriteRandomBenchmark() : MemoryBandwidthBenchmark<T>(
        MemoryBandwidthBenchmark<T>::TestType::WRITE_RANDOM) {}
};

template<typename T>
class MemoryStreamCopyBenchmark : public MemoryBandwidthBenchmark<T> {
public:
    MemoryStreamCopyBenchmark() : MemoryBandwidthBenchmark<T>(
        MemoryBandwidthBenchmark<T>::TestType::STREAM_COPY) {}
};

template<typename T>
class MemoryStreamScaleBenchmark : public MemoryBandwidthBenchmark<T> {
public:
    MemoryStreamScaleBenchmark() : MemoryBandwidthBenchmark<T>(
        MemoryBandwidthBenchmark<T>::TestType::STREAM_SCALE) {}
};