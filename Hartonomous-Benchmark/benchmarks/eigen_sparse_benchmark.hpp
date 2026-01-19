#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "../src/benchmark/benchmark_base.hpp"

// Example usage:
// EigenSparseBenchmark<double> bench;
// BenchmarkConfig config{1000, 1024, false, "double"};
// bench.setup(config);
// bench.run();
// auto result = bench.get_result();

template<typename T>
class EigenSparseBenchmark : public TemplatedBenchmark<T, std::string> {
private:
    Eigen::SparseMatrix<T> sparse_matrix_;
    Eigen::Vector<T, Eigen::Dynamic> vector_b_;
    Eigen::Vector<T, Eigen::Dynamic> vector_x_;
    size_t size_;
    double density_ = 0.01; // 1% density for sparse matrix

public:
    std::string get_name() const override {
        return "Eigen_Sparse_MatrixVector_" + std::string(typeid(T).name()) + "_" + std::to_string(size_);
    }

    void execute_operation() override {
        // Perform sparse matrix-vector multiplication
        vector_x_ = sparse_matrix_ * vector_b_;

        // Calculate throughput (operations: roughly 2 * nnz for sparse mat-vec)
        size_t nnz = sparse_matrix_.nonZeros();
        size_t operations = 2ULL * nnz;
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            this->result_.duration).count();
        this->result_.throughput = operations / seconds;
        this->result_.memory_used = sparse_matrix_.nonZeros() * sizeof(T) * 3 + // sparse storage (value, row, col)
                                   (vector_b_.size() + vector_x_.size()) * sizeof(T);
    }

    void setup(const BenchmarkConfig& config) override {
        this->config_ = config;
        this->result_.name = get_name();
        this->result_.success = true;
        size_ = config.data_size;

        vector_b_.resize(size_);
        vector_x_.resize(size_);

        // Create sparse matrix with given density
        std::vector<Eigen::Triplet<T>> triplets;
        triplets.reserve(static_cast<size_t>(size_ * size_ * density_));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis_val(-1.0, 1.0);
        std::uniform_real_distribution<double> dis_prob(0.0, 1.0);

        for (Eigen::Index i = 0; i < size_; ++i) {
            for (Eigen::Index j = 0; j < size_; ++j) {
                if (dis_prob(gen) < density_) {
                    triplets.emplace_back(i, j, dis_val(gen));
                }
            }
            vector_b_(i) = dis_val(gen);
        }

        sparse_matrix_.resize(size_, size_);
        sparse_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        sparse_matrix_.makeCompressed();
    }
};