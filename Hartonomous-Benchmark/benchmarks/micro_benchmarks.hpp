#pragma once

#include <vector>
#include <random>
#include <immintrin.h>
#include <mkl.h>
#include <Eigen/Dense>
#include "../src/benchmark/benchmark_base.hpp"
#include "../src/math/vector_math.hpp"

// Micro-benchmark for vector addition
template<typename T>
class VectorAdditionMicroBenchmark : public TemplatedBenchmark<T, std::string> {
protected:
    std::vector<T> vector_a_;
    std::vector<T> vector_b_;
    std::vector<T> result_vector_;
    size_t size_;
    enum class Implementation { SCALAR, SIMD, MKL, HYBRID };
    Implementation impl_;

public:
    VectorAdditionMicroBenchmark(Implementation impl = Implementation::SCALAR)
        : impl_(impl) {}

    std::string get_name() const override {
        std::string impl_str;
        switch (impl_) {
            case Implementation::SCALAR: impl_str = "Scalar"; break;
            case Implementation::SIMD: impl_str = "SIMD"; break;
            case Implementation::MKL: impl_str = "MKL"; break;
            case Implementation::HYBRID: impl_str = "Hybrid"; break;
        }
        return "Vector_Addition_" + impl_str + "_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        switch (impl_) {
            case Implementation::SCALAR:
                hartonomous::VectorMath<T>::add(vector_a_, vector_b_, result_vector_);
                break;
            case Implementation::SIMD:
                hartonomous::VectorMath<T>::add_simd(vector_a_, vector_b_, result_vector_);
                break;
            case Implementation::MKL:
                hartonomous::VectorMath<T>::add_mkl(vector_a_, vector_b_, result_vector_);
                break;
            case Implementation::HYBRID:
                hartonomous::VectorMath<T>::add_hybrid(vector_a_, vector_b_, result_vector_);
                break;
        }

        this->result_.memory_used = (vector_a_.size() + vector_b_.size() + result_vector_.size()) * sizeof(T);
    }



    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        size_ = config.data_size;
        this->result_.name = get_name();
        this->result_.success = true;
        this->result_.operations = size_;

        vector_a_.resize(size_);
        vector_b_.resize(size_);
        result_vector_.resize(size_);

        initialize_data();
    }

private:
    void initialize_data() {
        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        } else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(-100, 100);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        }
    }
};

// Micro-benchmark for dot product
template<typename T>
class DotProductMicroBenchmark : public TemplatedBenchmark<T, std::string> {
protected:
    std::vector<T> vector_a_;
    std::vector<T> vector_b_;
    T dot_result_;
    size_t size_;
    enum class Implementation { SCALAR, SIMD, MKL, HYBRID };
    Implementation impl_;

public:
    DotProductMicroBenchmark(Implementation impl = Implementation::SCALAR)
        : impl_(impl) {}

    std::string get_name() const override {
        std::string impl_str;
        switch (impl_) {
            case Implementation::SCALAR: impl_str = "Scalar"; break;
            case Implementation::SIMD: impl_str = "SIMD"; break;
            case Implementation::MKL: impl_str = "MKL"; break;
            case Implementation::HYBRID: impl_str = "Hybrid"; break;
        }
        return "Dot_Product_" + impl_str + "_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        switch (impl_) {
            case Implementation::SCALAR:
                dot_result_ = hartonomous::VectorMath<T>::dot_product(vector_a_, vector_b_);
                break;
            case Implementation::SIMD:
                dot_result_ = hartonomous::VectorMath<T>::dot_product_simd(vector_a_, vector_b_);
                break;
            case Implementation::MKL:
                dot_result_ = hartonomous::VectorMath<T>::dot_product_mkl(vector_a_, vector_b_);
                break;
            case Implementation::HYBRID:
                dot_result_ = hartonomous::VectorMath<T>::dot_product_hybrid(vector_a_, vector_b_);
                break;
        }

        this->result_.memory_used = (vector_a_.size() + vector_b_.size()) * sizeof(T) + sizeof(T);
    }



    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        size_ = config.data_size;
        this->result_.name = get_name();
        this->result_.success = true;
        this->result_.operations = 2 * size_;

        vector_a_.resize(size_);
        vector_b_.resize(size_);

        initialize_data();
    }

private:
    void initialize_data() {
        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        } else if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        } else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(-100, 100);
            for (auto& val : vector_a_) val = dis(gen);
            for (auto& val : vector_b_) val = dis(gen);
        }
    }
};

// Micro-benchmark for matrix-vector multiplication
template<typename T>
class MatrixVectorMicroBenchmark : public TemplatedBenchmark<T, std::string> {
protected:
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_;
    Eigen::Matrix<T, Eigen::Dynamic, 1> vector_;
    Eigen::Matrix<T, Eigen::Dynamic, 1> output_;
    size_t size_;
    enum class Implementation { EIGEN, EIGEN_AVX };
    Implementation impl_;

public:
    MatrixVectorMicroBenchmark(Implementation impl = Implementation::EIGEN)
        : impl_(impl) {}

    std::string get_name() const override {
        std::string impl_str = (impl_ == Implementation::EIGEN) ? "Eigen" : "Eigen_AVX";
        return "Matrix_Vector_" + impl_str + "_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        if (impl_ == Implementation::EIGEN_AVX) {
            // Force AVX usage through Eigen
            Eigen::setNbThreads(1);
        }

        output_ = matrix_ * vector_;

        this->result_.memory_used = (matrix_.size() + vector_.size() + output_.size()) * sizeof(T);
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        size_ = config.data_size;
        this->result_.name = get_name();
        this->result_.success = true;
        this->result_.operations = 2 * size_ * size_;

        matrix_.resize(size_, size_);
        vector_.resize(size_);
        output_.resize(size_);

        initialize_data();
    }

private:
    void initialize_data() {
        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_(i, j) = dis(gen);
                }
                vector_(i) = dis(gen);
            }
        } else if constexpr (std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(-1.0, 1.0);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_(i, j) = dis(gen);
                }
                vector_(i) = dis(gen);
            }
        } else if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dis(-10, 10);
            for (Eigen::Index i = 0; i < size_; ++i) {
                for (Eigen::Index j = 0; j < size_; ++j) {
                    matrix_(i, j) = dis(gen);
                }
                vector_(i) = dis(gen);
            }
        }
    }
};

// Convenience classes for different implementations
template<typename T> class VectorAdditionScalarBenchmark : public VectorAdditionMicroBenchmark<T> {
public: VectorAdditionScalarBenchmark() : VectorAdditionMicroBenchmark<T>(
    VectorAdditionMicroBenchmark<T>::Implementation::SCALAR) {} };
template<typename T> class VectorAdditionSIMDBenchmark : public VectorAdditionMicroBenchmark<T> {
public: VectorAdditionSIMDBenchmark() : VectorAdditionMicroBenchmark<T>(
    VectorAdditionMicroBenchmark<T>::Implementation::SIMD) {} };
template<typename T> class VectorAdditionMKLBenchmark : public VectorAdditionMicroBenchmark<T> {
public: VectorAdditionMKLBenchmark() : VectorAdditionMicroBenchmark<T>(
    VectorAdditionMicroBenchmark<T>::Implementation::MKL) {} };
template<typename T> class VectorAdditionHybridBenchmark : public VectorAdditionMicroBenchmark<T> {
public: VectorAdditionHybridBenchmark() : VectorAdditionMicroBenchmark<T>(
    VectorAdditionMicroBenchmark<T>::Implementation::HYBRID) {} };

template<typename T> class DotProductScalarBenchmark : public DotProductMicroBenchmark<T> {
public: DotProductScalarBenchmark() : DotProductMicroBenchmark<T>(
    DotProductMicroBenchmark<T>::Implementation::SCALAR) {} };
template<typename T> class DotProductSIMDBenchmark : public DotProductMicroBenchmark<T> {
public: DotProductSIMDBenchmark() : DotProductMicroBenchmark<T>(
    DotProductMicroBenchmark<T>::Implementation::SIMD) {} };
template<typename T> class DotProductMKLBenchmark : public DotProductMicroBenchmark<T> {
public: DotProductMKLBenchmark() : DotProductMicroBenchmark<T>(
    DotProductMicroBenchmark<T>::Implementation::MKL) {} };
template<typename T> class DotProductHybridBenchmark : public DotProductMicroBenchmark<T> {
public: DotProductHybridBenchmark() : DotProductMicroBenchmark<T>(
    DotProductMicroBenchmark<T>::Implementation::HYBRID) {} };

template<typename T> class MatrixVectorEigenBenchmark : public MatrixVectorMicroBenchmark<T> {
public: MatrixVectorEigenBenchmark() : MatrixVectorMicroBenchmark<T>(
    MatrixVectorMicroBenchmark<T>::Implementation::EIGEN) {} };
template<typename T> class MatrixVectorEigenAVXBenchmark : public MatrixVectorMicroBenchmark<T> {
public: MatrixVectorEigenAVXBenchmark() : MatrixVectorMicroBenchmark<T>(
    MatrixVectorMicroBenchmark<T>::Implementation::EIGEN_AVX) {} };
