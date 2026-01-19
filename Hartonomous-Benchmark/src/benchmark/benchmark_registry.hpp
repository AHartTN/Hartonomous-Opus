#pragma once

#include <memory>
#include <unordered_map>
#include <functional>
#include <string>
#include <variant>
#include <typeindex>
#include "benchmark_base.hpp"

// Type-erased benchmark base
using BenchmarkPtr = std::unique_ptr<BenchmarkBase>;
using BenchmarkFactoryFunc = std::function<BenchmarkPtr()>;

// Multi-type benchmark registry
class BenchmarkRegistry {
private:
    std::unordered_map<std::string, BenchmarkFactoryFunc> factories_;

public:
    template<typename BenchmarkClass>
    void register_benchmark(const std::string& name) {
        static_assert(std::is_base_of_v<BenchmarkBase, BenchmarkClass>,
                     "BenchmarkClass must inherit from BenchmarkBase");

        factories_[name] = []() -> BenchmarkPtr {
            return std::make_unique<BenchmarkClass>();
        };
    }

    template<typename DataType, typename OperationType, typename BenchmarkClass>
    void register_typed_benchmark(const std::string& name) {
        static_assert(std::is_base_of_v<BenchmarkBase, BenchmarkClass>,
                     "BenchmarkClass must inherit from BenchmarkBase");

        factories_[name] = []() -> BenchmarkPtr {
            return std::make_unique<BenchmarkClass>();
        };
    }

    BenchmarkPtr create_benchmark(const std::string& name) {
        auto it = factories_.find(name);
        if (it != factories_.end()) {
            return it->second();
        }
        return nullptr;
    }

    std::vector<std::string> get_registered_benchmarks() const {
        std::vector<std::string> names;
        for (const auto& pair : factories_) {
            names.push_back(pair.first);
        }
        return names;
    }

    bool has_benchmark(const std::string& name) const {
        return factories_.find(name) != factories_.end();
    }
};

// Legacy templated registry for backward compatibility
using BenchmarkFactory = std::function<std::unique_ptr<BenchmarkBase>()>;

class TemplatedBenchmarkRegistry {
private:
    std::unordered_map<std::string, BenchmarkFactory> factories_;

public:
    void register_benchmark(const std::string& name, BenchmarkFactory factory) {
        factories_[name] = std::move(factory);
    }

    std::unique_ptr<BenchmarkBase> create_benchmark(const std::string& name) {
        auto it = factories_.find(name);
        if (it != factories_.end()) {
            return it->second();
        }
        return nullptr;
    }

    std::vector<std::string> get_registered_benchmarks() const {
        std::vector<std::string> names;
        for (const auto& pair : factories_) {
            names.push_back(pair.first);
        }
        return names;
    }
};