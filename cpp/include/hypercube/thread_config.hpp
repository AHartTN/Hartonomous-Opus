#pragma once
/**
 * Unified Thread Configuration System for Hypercube Operations
 *
 * Provides centralized, workload-aware thread count management across all components.
 * Eliminates hardcoded thread values and provides consistent threading behavior.
 */

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <vector>

namespace hypercube {

enum class WorkloadType {
    COMPUTE_BOUND,      // CPU-intensive: matrix ops, encoding, hashing
    IO_BOUND,          // I/O intensive: database ops, file I/O, network
    HYBRID            // Mixed workloads: ingestion, processing pipelines
};

class ThreadConfig {
public:
    // Singleton access
    static ThreadConfig& instance();

    // Core thread count management
    size_t get_thread_count(WorkloadType type = WorkloadType::COMPUTE_BOUND) const;
    void set_thread_count_override(size_t count);
    void clear_thread_count_override();

    // Hardware detection and constraints
    size_t get_hardware_concurrency() const;
    size_t get_recommended_max_threads() const;

    // Workload-specific configurations
    size_t get_compute_threads() const;
    size_t get_io_threads() const;
    size_t get_hybrid_threads() const;

    // Library integration
    void configure_openmp() const;
    void configure_mkl() const;
    void configure_std_thread_pool() const;

    // Runtime adjustments
    bool adjust_for_system_load(double load_factor);
    void set_dynamic_scaling(bool enabled);

    // Validation
    bool validate_configuration() const;
    std::vector<std::string> get_validation_errors() const;

private:
    ThreadConfig();
    ~ThreadConfig() = default;

    // Prevent copying
    ThreadConfig(const ThreadConfig&) = delete;
    ThreadConfig& operator=(const ThreadConfig&) = delete;

    // Hardware detection
    void detect_hardware_capabilities();

    // Configuration loading
    void load_from_environment();
    void load_from_config();

    // Internal state
    mutable std::mutex mutex_;
    size_t hardware_concurrency_;
    std::atomic<size_t> max_threads_override_;
    bool dynamic_scaling_enabled_;

    // Workload caps (as fractions of hardware_concurrency)
    double compute_bound_cap_;    // Default: 1.0 (100% of cores)
    double io_bound_cap_;         // Default: 2.0 (200% of cores for I/O)
    double hybrid_cap_;           // Default: 1.5 (150% of cores)
};

// Workload classifier utility
class WorkloadClassifier {
public:
    static WorkloadType classify_operation(const std::string& operation_name);

    // Pre-defined classifications
    static constexpr WorkloadType ENCODING = WorkloadType::COMPUTE_BOUND;
    static constexpr WorkloadType HASHING = WorkloadType::COMPUTE_BOUND;
    static constexpr WorkloadType MATRIX_OPS = WorkloadType::COMPUTE_BOUND;
    static constexpr WorkloadType DATABASE_QUERY = WorkloadType::IO_BOUND;
    static constexpr WorkloadType FILE_IO = WorkloadType::IO_BOUND;
    static constexpr WorkloadType INGESTION = WorkloadType::HYBRID;
};

// Validation result structure
struct ValidationResult {
    bool valid;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

// Runtime adjuster for dynamic thread management
class RuntimeAdjuster {
public:
    static bool should_adjust_threads(double system_load);
    static size_t calculate_optimal_threads(size_t current_count, double load);

    // System monitoring
    static double get_system_load_average();
    static size_t get_memory_pressure_mb();
};

// Runtime monitoring and adjustment
void monitor_and_adjust_threading();

} // namespace hypercube