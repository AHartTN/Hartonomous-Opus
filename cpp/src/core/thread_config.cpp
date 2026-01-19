/**
 * Unified Thread Configuration System Implementation
 *
 * Provides centralized thread management with workload-aware allocation.
 */

#include "hypercube/thread_config.hpp"
#include "hypercube/logging.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MKL_ILP64
#include <mkl.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

namespace hypercube {

// Environment variable names
static constexpr const char* ENV_MAX_THREADS = "HC_MAX_THREADS";
static constexpr const char* ENV_COMPUTE_THREADS = "HC_COMPUTE_THREADS";
static constexpr const char* ENV_IO_THREADS = "HC_IO_THREADS";
static constexpr const char* ENV_HYBRID_THREADS = "HC_HYBRID_THREADS";
static constexpr const char* ENV_THREAD_SCALING = "HC_THREAD_SCALING";
static constexpr const char* ENV_THREAD_PRIORITY = "HC_THREAD_PRIORITY";

// Default workload caps
static constexpr double DEFAULT_COMPUTE_CAP = 1.0;
static constexpr double DEFAULT_IO_CAP = 2.0;
static constexpr double DEFAULT_HYBRID_CAP = 1.5;

ThreadConfig& ThreadConfig::instance() {
    static ThreadConfig instance;
    return instance;
}

ThreadConfig::ThreadConfig()
    : hardware_concurrency_(0)
    , max_threads_override_(0)
    , dynamic_scaling_enabled_(true)
    , compute_bound_cap_(DEFAULT_COMPUTE_CAP)
    , io_bound_cap_(DEFAULT_IO_CAP)
    , hybrid_cap_(DEFAULT_HYBRID_CAP) {

    detect_hardware_capabilities();
    load_from_environment();

    // Validate configuration
    if (!validate_configuration()) {
        auto errors = get_validation_errors();
        for (const auto& error : errors) {
            std::cerr << "ThreadConfig Error: " << error << std::endl;
        }
    }
}

size_t ThreadConfig::get_thread_count(WorkloadType type) const {
    size_t base_count = get_hardware_concurrency();
    if (base_count == 0) base_count = 4;  // Safe fallback

    // Check for global override
    size_t override = max_threads_override_.load(std::memory_order_acquire);
    if (override > 0) {
        return std::min(override, base_count);
    }

    // Apply workload-specific caps
    double cap_factor;
    switch (type) {
        case WorkloadType::COMPUTE_BOUND:
            cap_factor = compute_bound_cap_;
            break;
        case WorkloadType::IO_BOUND:
            cap_factor = io_bound_cap_;
            break;
        case WorkloadType::HYBRID:
            cap_factor = hybrid_cap_;
            break;
        default:
            cap_factor = DEFAULT_COMPUTE_CAP;
    }

    size_t calculated = static_cast<size_t>(base_count * cap_factor);
    return std::max(size_t(1), std::min(calculated, base_count));
}

void ThreadConfig::set_thread_count_override(size_t count) {
    max_threads_override_.store(count, std::memory_order_release);
}

void ThreadConfig::clear_thread_count_override() {
    max_threads_override_.store(0, std::memory_order_release);
}

size_t ThreadConfig::get_hardware_concurrency() const {
    return hardware_concurrency_;
}

size_t ThreadConfig::get_recommended_max_threads() const {
    size_t hw = get_hardware_concurrency();
    return std::max(size_t(1), hw - 1);  // Leave one core for system
}

size_t ThreadConfig::get_compute_threads() const {
    return get_thread_count(WorkloadType::COMPUTE_BOUND);
}

size_t ThreadConfig::get_io_threads() const {
    return get_thread_count(WorkloadType::IO_BOUND);
}

size_t ThreadConfig::get_hybrid_threads() const {
    return get_thread_count(WorkloadType::HYBRID);
}

void ThreadConfig::configure_openmp() const {
    size_t thread_count = get_compute_threads();
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(thread_count));
    omp_set_dynamic(0);  // Disable dynamic adjustment for consistency
#endif
}

void ThreadConfig::configure_mkl() const {
    size_t thread_count = get_compute_threads();
#ifdef MKL_ILP64
    mkl_set_num_threads(static_cast<int>(thread_count));
    mkl_set_dynamic(0);  // Disable dynamic adjustment
#endif
}

void ThreadConfig::configure_std_thread_pool() const {
    // This is a no-op for now, as the ThreadPool will query get_thread_count()
    // when needed. Future enhancement could reconfigure existing pools.
}

bool ThreadConfig::adjust_for_system_load(double load_factor) {
    if (!dynamic_scaling_enabled_) return false;

    // Simple load-based adjustment
    if (load_factor > 0.8) {
        // High load - reduce threads by 25%
        size_t current = max_threads_override_.load();
        if (current == 0) current = get_hardware_concurrency();
        size_t adjusted = current * 3 / 4;
        adjusted = std::max(size_t(1), adjusted);
        set_thread_count_override(adjusted);
        return true;
    } else if (load_factor < 0.3) {
        // Low load - can use more threads
        size_t hw = get_hardware_concurrency();
        size_t adjusted = std::min(hw, static_cast<size_t>(hw * 1.2));
        set_thread_count_override(adjusted);
        return true;
    }

    return false;
}

void ThreadConfig::set_dynamic_scaling(bool enabled) {
    dynamic_scaling_enabled_ = enabled;
}

bool ThreadConfig::validate_configuration() const {
    auto errors = get_validation_errors();
    return errors.empty();
}

std::vector<std::string> ThreadConfig::get_validation_errors() const {
    std::vector<std::string> errors;

    size_t hw = get_hardware_concurrency();
    if (hw == 0) {
        errors.push_back("Unable to detect hardware concurrency");
    }

    size_t override = max_threads_override_.load();
    if (override > hw && override > 0) {
        errors.push_back("Thread count override (" + std::to_string(override) +
                        ") exceeds hardware concurrency (" + std::to_string(hw) + ")");
    }

    if (compute_bound_cap_ <= 0.0 || compute_bound_cap_ > 4.0) {
        errors.push_back("Invalid compute_bound_cap: " + std::to_string(compute_bound_cap_));
    }

    if (io_bound_cap_ <= 0.0 || io_bound_cap_ > 8.0) {
        errors.push_back("Invalid io_bound_cap: " + std::to_string(io_bound_cap_));
    }

    if (hybrid_cap_ <= 0.0 || hybrid_cap_ > 6.0) {
        errors.push_back("Invalid hybrid_cap: " + std::to_string(hybrid_cap_));
    }

    return errors;
}

void ThreadConfig::detect_hardware_capabilities() {
    hardware_concurrency_ = std::thread::hardware_concurrency();

    // Fallback detection methods if std::thread::hardware_concurrency fails
    if (hardware_concurrency_ == 0) {
#ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        hardware_concurrency_ = sysinfo.dwNumberOfProcessors;
#else
        // Try sysconf
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        if (nprocs > 0) {
            hardware_concurrency_ = static_cast<size_t>(nprocs);
        } else {
            // Last resort
            hardware_concurrency_ = 4;
        }
#endif
    }
}

void ThreadConfig::load_from_environment() {
    // Load global override
    if (const char* env = std::getenv(ENV_MAX_THREADS)) {
        try {
            size_t value = std::stoul(env);
            if (value > 0) {
                max_threads_override_.store(value, std::memory_order_relaxed);
            }
        } catch (const std::exception&) {
            std::cerr << "Warning: Invalid " << ENV_MAX_THREADS << " value: " << env << std::endl;
        }
    }

    // Load workload-specific caps
    auto load_cap = [](const char* env_var, double& cap) {
        if (const char* env = std::getenv(env_var)) {
            try {
                double value = std::stod(env);
                if (value > 0.0) {
                    cap = value;
                }
            } catch (const std::exception&) {
                std::cerr << "Warning: Invalid " << env_var << " value: " << env << std::endl;
            }
        }
    };

    load_cap(ENV_COMPUTE_THREADS, compute_bound_cap_);
    load_cap(ENV_IO_THREADS, io_bound_cap_);
    load_cap(ENV_HYBRID_THREADS, hybrid_cap_);

    // Load dynamic scaling setting
    if (const char* env = std::getenv(ENV_THREAD_SCALING)) {
        std::string value = env;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        dynamic_scaling_enabled_ = (value == "true" || value == "1" || value == "yes");
    }
}

void ThreadConfig::load_from_config() {
    // Future: Load from configuration file
    // For now, environment variables are sufficient
}

// =============================================================================
// WorkloadClassifier Implementation
// =============================================================================

WorkloadType WorkloadClassifier::classify_operation(const std::string& operation_name) {
    // Convert to lowercase for case-insensitive matching
    std::string lower_op = operation_name;
    std::transform(lower_op.begin(), lower_op.end(), lower_op.begin(), ::tolower);

    // Database operations
    if (lower_op.find("query") != std::string::npos ||
        lower_op.find("select") != std::string::npos ||
        lower_op.find("insert") != std::string::npos ||
        lower_op.find("update") != std::string::npos ||
        lower_op.find("delete") != std::string::npos ||
        lower_op.find("database") != std::string::npos) {
        return WorkloadType::IO_BOUND;
    }

    // File I/O operations
    if (lower_op.find("file") != std::string::npos ||
        lower_op.find("read") != std::string::npos ||
        lower_op.find("write") != std::string::npos ||
        lower_op.find("load") != std::string::npos ||
        lower_op.find("save") != std::string::npos) {
        return WorkloadType::IO_BOUND;
    }

    // Compute-intensive operations
    if (lower_op.find("matrix") != std::string::npos ||
        lower_op.find("eigen") != std::string::npos ||
        lower_op.find("linear") != std::string::npos ||
        lower_op.find("algebra") != std::string::npos ||
        lower_op.find("compute") != std::string::npos ||
        lower_op.find("hash") != std::string::npos ||
        lower_op.find("encode") != std::string::npos ||
        lower_op.find("decode") != std::string::npos) {
        return WorkloadType::COMPUTE_BOUND;
    }

    // Ingestion and processing pipelines
    if (lower_op.find("ingest") != std::string::npos ||
        lower_op.find("process") != std::string::npos ||
        lower_op.find("pipeline") != std::string::npos ||
        lower_op.find("batch") != std::string::npos) {
        return WorkloadType::HYBRID;
    }

    // Default to compute-bound for unknown operations
    return WorkloadType::COMPUTE_BOUND;
}

// =============================================================================
// RuntimeAdjuster Implementation
// =============================================================================

bool RuntimeAdjuster::should_adjust_threads(double system_load) {
    // Adjust if load is very high (>80%) or very low (<20%)
    return system_load > 0.8 || system_load < 0.2;
}

// Runtime monitoring and adjustment function
void monitor_and_adjust_threading() {
    static auto last_check = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();

    // Check every 30 seconds
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_check).count() < 30) {
        return;
    }

    last_check = now;

    double load = RuntimeAdjuster::get_system_load_average();
    if (RuntimeAdjuster::should_adjust_threads(load)) {
        ThreadConfig::instance().adjust_for_system_load(load);
    }
}

size_t RuntimeAdjuster::calculate_optimal_threads(size_t current_count, double load) {
    if (load > 0.8) {
        // High load - reduce by 25%
        return std::max(size_t(1), current_count * 3 / 4);
    } else if (load < 0.2) {
        // Low load - increase by 20%
        size_t hw = std::thread::hardware_concurrency();
        return std::min(hw, static_cast<size_t>(current_count * 1.2));
    }
    return current_count;
}

double RuntimeAdjuster::get_system_load_average() {
#ifdef _WIN32
    // Windows implementation
    return 0.5;  // Placeholder - would need Windows-specific load average API
#else
    // Linux implementation
    double load[3];
    if (getloadavg(load, 3) != -1) {
        size_t hw = std::thread::hardware_concurrency();
        if (hw > 0) {
            return load[0] / static_cast<double>(hw);  // 1-minute load average
        }
    }
    return 0.5;  // Default to 50% load if unable to detect
#endif
}

size_t RuntimeAdjuster::get_memory_pressure_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);

    size_t used_mb = (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024 * 1024);
    return used_mb;
#else
    // Simple implementation using sysinfo
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        size_t used_mb = (info.totalram - info.freeram) * info.mem_unit / (1024 * 1024);
        return used_mb;
    }
    return 0;
#endif
}

} // namespace hypercube