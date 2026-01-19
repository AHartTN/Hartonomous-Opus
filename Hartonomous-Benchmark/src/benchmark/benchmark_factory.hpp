#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <type_traits>
#include "benchmark_base.hpp"

// Abstract factory for creating benchmarks
class BenchmarkAbstractFactory {
public:
    virtual ~BenchmarkAbstractFactory() = default;
    virtual std::unique_ptr<BenchmarkBase> create_benchmark(const std::string& type) = 0;
    virtual std::vector<std::string> get_supported_types() const = 0;
};

// Templated concrete factory
template<typename DataType, typename OperationType>
class TemplatedBenchmarkFactory : public BenchmarkAbstractFactory {
private:
    using BenchmarkPtr = std::unique_ptr<BenchmarkBase>;
    using FactoryFunction = std::function<BenchmarkPtr()>;

    std::unordered_map<std::string, FactoryFunction> factories_;

public:
    template<typename BenchmarkClass>
    void register_benchmark(const std::string& name) {
        static_assert(std::is_base_of_v<BenchmarkBase, BenchmarkClass>,
                     "BenchmarkClass must inherit from BenchmarkBase");

        factories_[name] = []() -> BenchmarkPtr {
            return std::make_unique<BenchmarkClass>();
        };
    }

    std::unique_ptr<BenchmarkBase> create_benchmark(const std::string& type) override {
        auto it = factories_.find(type);
        if (it != factories_.end()) {
            return it->second();
        }
        return nullptr;
    }

    std::vector<std::string> get_supported_types() const override {
        std::vector<std::string> types;
        for (const auto& pair : factories_) {
            types.push_back(pair.first);
        }
        return types;
    }
};

// Factory registry for managing multiple factories
class BenchmarkFactoryRegistry {
private:
    std::unordered_map<std::string, std::unique_ptr<BenchmarkAbstractFactory>> factories_;

public:
    template<typename DataType, typename OperationType>
    void register_factory(const std::string& category) {
        factories_[category] = std::make_unique<TemplatedBenchmarkFactory<DataType, OperationType>>();
    }

    BenchmarkAbstractFactory* get_factory(const std::string& category) {
        auto it = factories_.find(category);
        return it != factories_.end() ? it->second.get() : nullptr;
    }

    std::vector<std::string> get_categories() const {
        std::vector<std::string> categories;
        for (const auto& pair : factories_) {
            categories.push_back(pair.first);
        }
        return categories;
    }

    std::unique_ptr<BenchmarkBase> create_benchmark(
        const std::string& category, const std::string& type) {

        auto factory = get_factory(category);
        if (factory) {
            return factory->create_benchmark(type);
        }
        return nullptr;
    }
};

// Helper macros for easy registration
#define REGISTER_BENCHMARK_IN_FACTORY(factory, BenchmarkClass, name) \
    factory.template register_benchmark<BenchmarkClass>(name)

#define REGISTER_VECTOR_BENCHMARK(factory, BenchmarkClass) \
    REGISTER_BENCHMARK_IN_FACTORY(factory, BenchmarkClass<float>, #BenchmarkClass "_float"); \
    REGISTER_BENCHMARK_IN_FACTORY(factory, BenchmarkClass<double>, #BenchmarkClass "_double"); \
    REGISTER_BENCHMARK_IN_FACTORY(factory, BenchmarkClass<int32_t>, #BenchmarkClass "_int32"); \
    REGISTER_BENCHMARK_IN_FACTORY(factory, BenchmarkClass<int64_t>, #BenchmarkClass "_int64")