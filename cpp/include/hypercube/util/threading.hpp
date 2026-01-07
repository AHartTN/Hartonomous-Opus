/**
 * Threading Context and Utilities for Parallel Processing
 *
 * Provides thread-safe utilities for parallel processing with integrated
 * cancellation, progress tracking, and exception propagation.
 */

#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <exception>
#include <memory>
#include <future>
#include <thread>

#include "hypercube/thread_pool.hpp"

namespace hypercube {

/**
 * Cancellation token for cooperative cancellation across threads
 */
class CancellationToken {
public:
    CancellationToken() : cancelled_(false) {}

    // Request cancellation
    void cancel() {
        cancelled_.store(true, std::memory_order_release);
    }

    // Check if cancellation was requested
    bool is_cancelled() const {
        return cancelled_.load(std::memory_order_acquire);
    }

    // Reset the token
    void reset() {
        cancelled_.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> cancelled_;
};

/**
 * Progress tracker for monitoring task completion
 */
class ProgressTracker {
public:
    explicit ProgressTracker(size_t total_tasks)
        : total_(total_tasks), completed_(0) {}

    // Mark a task as completed
    void increment() {
        size_t current = completed_.fetch_add(1, std::memory_order_release) + 1;
        if (current == total_) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.notify_all();
        }
    }

    // Get completion ratio (0.0 to 1.0)
    double progress() const {
        return static_cast<double>(completed_.load(std::memory_order_acquire)) / total_;
    }

    // Get number of completed tasks
    size_t completed() const {
        return completed_.load(std::memory_order_acquire);
    }

    // Get total number of tasks
    size_t total() const { return total_; }

    // Wait for completion
    void wait_for_completion() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() {
            return completed_.load(std::memory_order_acquire) >= total_;
        });
    }

private:
    size_t total_;
    std::atomic<size_t> completed_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

/**
 * Exception propagator for collecting and re-throwing exceptions from worker threads
 */
class ExceptionPropagator {
public:
    ExceptionPropagator() : has_exception_(false) {}

    // Store an exception for later propagation
    void set_exception(std::exception_ptr ex) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!has_exception_) {
            exception_ = ex;
            has_exception_ = true;
        }
    }

    // Re-throw stored exception if any
    void propagate() {
        if (has_exception_) {
            std::exception_ptr ex;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ex = exception_;
            }
            std::rethrow_exception(ex);
        }
    }

    // Check if any exception was stored
    bool has_exception() const {
        return has_exception_;
    }

    // Reset for reuse
    void reset() {
        std::unique_lock<std::mutex> lock(mutex_);
        has_exception_ = false;
        exception_ = nullptr;
    }

private:
    std::exception_ptr exception_;
    std::atomic<bool> has_exception_;
    mutable std::mutex mutex_;
};

/**
 * Threading context providing high-level parallel processing utilities
 * with cancellation, progress tracking, and exception handling
 */
class ThreadingContext {
public:
    explicit ThreadingContext(ThreadPool& pool = ThreadPool::instance())
        : pool_(pool) {}

    /**
     * Parallel map operation with cancellation, progress, and exception handling
     */
    template<typename InputIt, typename OutputIt, typename Func>
    void parallel_map(InputIt first, InputIt last, OutputIt result,
                     Func&& func,
                     CancellationToken* cancel = nullptr,
                     ProgressTracker* progress = nullptr,
                     ExceptionPropagator* exceptions = nullptr) {
        const size_t num_elements = std::distance(first, last);
        if (num_elements == 0) return;

        const size_t num_workers = pool_.num_threads();
        const size_t chunk_size = (num_elements + num_workers - 1) / num_workers;

        std::vector<std::future<void>> futures;
        futures.reserve(num_workers);

        for (size_t worker_id = 0; worker_id < num_workers; ++worker_id) {
            size_t chunk_begin = worker_id * chunk_size;
            size_t chunk_end = std::min(chunk_begin + chunk_size, num_elements);

            if (chunk_begin >= num_elements) break;

            futures.push_back(pool_.submit([&, chunk_begin, chunk_end, worker_id]() {
                try {
                    auto input_it = first;
                    auto output_it = result;
                    std::advance(input_it, chunk_begin);
                    std::advance(output_it, chunk_begin);

                    for (size_t i = chunk_begin; i < chunk_end; ++i) {
                        if (cancel && cancel->is_cancelled()) {
                            break;
                        }

                        *output_it = func(*input_it);

                        if (progress) {
                            progress->increment();
                        }

                        ++input_it;
                        ++output_it;
                    }
                } catch (...) {
                    if (exceptions) {
                        exceptions->set_exception(std::current_exception());
                    }
                    throw;
                }
            }));
        }

        // Wait for all tasks and propagate any exceptions
        for (auto& future : futures) {
            try {
                future.get();
            } catch (...) {
                if (exceptions) {
                    exceptions->set_exception(std::current_exception());
                } else {
                    throw;
                }
            }
        }

        if (exceptions) {
            exceptions->propagate();
        }
    }

    /**
     * Parallel for-each with cancellation and progress tracking
     */
    template<typename InputIt, typename Func>
    void parallel_for_each(InputIt first, InputIt last, Func&& func,
                          CancellationToken* cancel = nullptr,
                          ProgressTracker* progress = nullptr,
                          ExceptionPropagator* exceptions = nullptr) {
        const size_t num_elements = std::distance(first, last);
        if (num_elements == 0) return;

        const size_t num_workers = pool_.num_threads();
        const size_t chunk_size = (num_elements + num_workers - 1) / num_workers;

        std::vector<std::future<void>> futures;
        futures.reserve(num_workers);

        for (size_t worker_id = 0; worker_id < num_workers; ++worker_id) {
            size_t chunk_begin = worker_id * chunk_size;
            size_t chunk_end = std::min(chunk_begin + chunk_size, num_elements);

            if (chunk_begin >= num_elements) break;

            futures.push_back(pool_.submit([&, chunk_begin, chunk_end, worker_id]() {
                try {
                    auto it = first;
                    std::advance(it, chunk_begin);

                    for (size_t i = chunk_begin; i < chunk_end; ++i) {
                        if (cancel && cancel->is_cancelled()) {
                            break;
                        }

                        func(*it);

                        if (progress) {
                            progress->increment();
                        }

                        ++it;
                    }
                } catch (...) {
                    if (exceptions) {
                        exceptions->set_exception(std::current_exception());
                    }
                    throw;
                }
            }));
        }

        // Wait for completion and handle exceptions
        for (auto& future : futures) {
            try {
                future.get();
            } catch (...) {
                if (exceptions) {
                    exceptions->set_exception(std::current_exception());
                } else {
                    throw;
                }
            }
        }

        if (exceptions) {
            exceptions->propagate();
        }
    }

    /**
     * Synchronization barrier for coordinating multiple threads
     */
    class Barrier {
    public:
        explicit Barrier(size_t count) : count_(count), current_(0), generation_(0) {}

        void wait() {
            std::unique_lock<std::mutex> lock(mutex_);
            size_t gen = generation_;
            if (++current_ == count_) {
                current_ = 0;
                ++generation_;
                cv_.notify_all();
            } else {
                cv_.wait(lock, [this, gen]() { return gen != generation_; });
            }
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        size_t count_;
        size_t current_;
        size_t generation_;
    };

    /**
     * Semaphore for limiting concurrent access
     */
    class Semaphore {
    public:
        explicit Semaphore(size_t count) : count_(count) {}

        void acquire() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return count_ > 0; });
            --count_;
        }

        void release() {
            std::unique_lock<std::mutex> lock(mutex_);
            ++count_;
            cv_.notify_one();
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        size_t count_;
    };

    ThreadPool& thread_pool() { return pool_; }

private:
    ThreadPool& pool_;
};

} // namespace hypercube