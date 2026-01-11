#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <typeindex>
#include <mutex>

namespace hypercube {

/**
 * Enterprise-grade factory pattern implementation
 * Provides type-safe, thread-safe object creation with dependency injection
 */

template<typename BaseType, typename... Args>
class Factory {
public:
    using CreatorFunction = std::function<std::unique_ptr<BaseType>(Args...)>;

    static Factory& instance() {
        static Factory instance;
        return instance;
    }

    template<typename DerivedType>
    void register_type(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        creators_[name] = [](Args... args) -> std::unique_ptr<BaseType> {
            return std::make_unique<DerivedType>(std::forward<Args>(args)...);
        };
    }

    void register_creator(const std::string& name, CreatorFunction creator) {
        std::lock_guard<std::mutex> lock(mutex_);
        creators_[name] = std::move(creator);
    }

    std::unique_ptr<BaseType> create(const std::string& name, Args... args) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = creators_.find(name);
        if (it == creators_.end()) {
            throw InvalidArgumentError("Unknown factory type: " + name);
        }
        return it->second(std::forward<Args>(args)...);
    }

    std::vector<std::string> get_registered_types() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> types;
        types.reserve(creators_.size());
        for (const auto& pair : creators_) {
            types.push_back(pair.first);
        }
        return types;
    }

    bool is_registered(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return creators_.find(name) != creators_.end();
    }

private:
    Factory() = default;
    Factory(const Factory&) = delete;
    Factory& operator=(const Factory&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, CreatorFunction> creators_;
};

// Convenience macros for factory registration
#define REGISTER_FACTORY_TYPE(BaseType, DerivedType, name, ...) \
    namespace { \
        struct FactoryRegistrar##DerivedType { \
            FactoryRegistrar##DerivedType() { \
                hypercube::Factory<BaseType, ##__VA_ARGS__>::instance() \
                    .register_type<DerivedType>(name); \
            } \
        } registrar##DerivedType; \
    }

// Type-erased factory for polymorphic types
class PolymorphicFactory {
public:
    static PolymorphicFactory& instance() {
        static PolymorphicFactory instance;
        return instance;
    }

    template<typename BaseType, typename DerivedType>
    void register_type(const std::string& category, const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = std::make_pair(category, name);
        creators_[key] = []() -> std::shared_ptr<void> {
            return std::static_pointer_cast<void>(std::make_shared<DerivedType>());
        };
    }

    template<typename T>
    std::shared_ptr<T> create(const std::string& category, const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = std::make_pair(category, name);
        auto it = creators_.find(key);
        if (it == creators_.end()) {
            throw InvalidArgumentError("Unknown polymorphic type: " + category + "::" + name);
        }
        auto base_ptr = it->second();
        return std::static_pointer_cast<T>(base_ptr);
    }

private:
    PolymorphicFactory() = default;

    mutable std::mutex mutex_;
    std::map<std::pair<std::string, std::string>,
             std::function<std::shared_ptr<void>()>> creators_;
};

// Configuration-driven factory
class ConfigurableFactory {
public:
    static ConfigurableFactory& instance() {
        static ConfigurableFactory instance;
        return instance;
    }

    template<typename T>
    void register_configurable_type(const std::string& type_name,
                                   std::function<std::unique_ptr<T>(const std::unordered_map<std::string, std::string>&)> creator) {
        std::lock_guard<std::mutex> lock(mutex_);
        configurable_creators_[type_name] = [creator](const auto& config) {
            return creator(config);
        };
    }

    template<typename T>
    std::unique_ptr<T> create_from_config(const std::string& type_name,
                                        const std::unordered_map<std::string, std::string>& config) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = configurable_creators_.find(type_name);
        if (it == configurable_creators_.end()) {
            throw InvalidArgumentError("Unknown configurable type: " + type_name);
        }
        return std::static_pointer_cast<T>(it->second(config));
    }

private:
    ConfigurableFactory() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::string,
                      std::function<std::shared_ptr<void>(const std::unordered_map<std::string, std::string>&)>> configurable_creators_;
};

} // namespace hypercube</content>
</xai:function_call">The file cpp/include/hypercube/factory.hpp was created successfully.