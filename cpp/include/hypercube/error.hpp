#pragma once

#include <stdexcept>
#include <string>
#include <system_error>
#include <memory>

namespace hypercube {

/**
 * Enterprise-grade error handling system
 * Provides structured error reporting with context and recovery suggestions
 */

enum class ErrorCode {
    // General errors
    SUCCESS = 0,
    INVALID_ARGUMENT = 1,
    OUT_OF_MEMORY = 2,
    NOT_IMPLEMENTED = 3,

    // Database errors
    CONNECTION_FAILED = 100,
    QUERY_FAILED = 101,
    TRANSACTION_FAILED = 102,

    // Mathematical errors
    NUMERICAL_ERROR = 200,
    CONVERGENCE_FAILED = 201,
    SINGULAR_MATRIX = 202,

    // I/O errors
    FILE_NOT_FOUND = 300,
    PERMISSION_DENIED = 301,
    DISK_FULL = 302,

    // Network errors
    NETWORK_TIMEOUT = 400,
    CONNECTION_LOST = 401,

    // Internal errors
    INTERNAL_ERROR = 500,
    ASSERTION_FAILED = 501
};

class HypercubeException : public std::runtime_error {
public:
    explicit HypercubeException(ErrorCode code, const std::string& message,
                               const std::string& context = "",
                               const std::string& suggestion = "")
        : std::runtime_error(format_message(code, message, context, suggestion))
        , code_(code)
        , context_(context)
        , suggestion_(suggestion) {}

    ErrorCode code() const noexcept { return code_; }
    const std::string& context() const noexcept { return context_; }
    const std::string& suggestion() const noexcept { return suggestion_; }

private:
    static std::string format_message(ErrorCode code, const std::string& message,
                                    const std::string& context, const std::string& suggestion) {
        std::string result = "Hypercube error [" + std::to_string(static_cast<int>(code)) + "]: " + message;
        if (!context.empty()) {
            result += "\nContext: " + context;
        }
        if (!suggestion.empty()) {
            result += "\nSuggestion: " + suggestion;
        }
        return result;
    }

    ErrorCode code_;
    std::string context_;
    std::string suggestion_;
};

// Convenience exception types
class InvalidArgumentError : public HypercubeException {
public:
    explicit InvalidArgumentError(const std::string& message,
                                 const std::string& context = "",
                                 const std::string& suggestion = "")
        : HypercubeException(ErrorCode::INVALID_ARGUMENT, message, context, suggestion) {}
};

class DatabaseError : public HypercubeException {
public:
    explicit DatabaseError(const std::string& message,
                          const std::string& context = "",
                          const std::string& suggestion = "")
        : HypercubeException(ErrorCode::CONNECTION_FAILED, message, context, suggestion) {}
};

class NumericalError : public HypercubeException {
public:
    explicit NumericalError(const std::string& message,
                           const std::string& context = "",
                           const std::string& suggestion = "")
        : HypercubeException(ErrorCode::NUMERICAL_ERROR, message, context, suggestion) {}
};

class IOError : public HypercubeException {
public:
    explicit IOError(const std::string& message,
                    const std::string& context = "",
                    const std::string& suggestion = "")
        : HypercubeException(ErrorCode::FILE_NOT_FOUND, message, context, suggestion) {}
};

// Error handling utilities
class ErrorHandler {
public:
    static void handle_error(const std::exception& e, bool rethrow = true) {
        // Note: Logging integration would go here
        // For now, we just rethrow or ignore
        if (rethrow) {
            throw;
        }
    }

    static void check_condition(bool condition, ErrorCode code,
                               const std::string& message,
                               const std::string& context = "",
                               const std::string& suggestion = "") {
        if (!condition) {
            throw HypercubeException(code, message, context, suggestion);
        }
    }

    static void check_pointer(const void* ptr, const std::string& name) {
        if (!ptr) {
            throw InvalidArgumentError("Null pointer: " + name);
        }
    }
};

// RAII error context
class ErrorContext {
public:
    explicit ErrorContext(const std::string& context) : context_(context) {}
    ~ErrorContext() noexcept = default;

    const std::string& get() const noexcept { return context_; }

private:
    std::string context_;
};

// Macros for common error checking
#define HYPERCUBE_CHECK(condition, code, message) \
    hypercube::ErrorHandler::check_condition(condition, code, message, __func__)

#define HYPERCUBE_CHECK_ARGUMENT(condition, message) \
    HYPERCUBE_CHECK(condition, hypercube::ErrorCode::INVALID_ARGUMENT, message)

#define HYPERCUBE_CHECK_POINTER(ptr, name) \
    hypercube::ErrorHandler::check_pointer(ptr, name)

#define HYPERCUBE_THROW(code, message) \
    throw hypercube::HypercubeException(code, message, __func__)

#define HYPERCUBE_THROW_INVALID_ARG(message) \
    throw hypercube::InvalidArgumentError(message, __func__)

} // namespace hypercube</xai:function_call">The file cpp/include/hypercube/error.hpp was created successfully.