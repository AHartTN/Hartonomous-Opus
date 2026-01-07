/**
 * High-Performance Thread Pool for Hypercube Operations
 *
 * Efficient thread pool for parallel batch processing.
 * Reusable across all C++ tools - no thread creation overhead per operation.
 */

#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>
#include <optional>
#include <array>
#include <cstdlib>  // for rand()

namespace hypercube {

/**
 * Lock-free work-stealing deque with efficient push/pop operations
 */
class WorkStealingDeque {
private:
    static constexpr size_t BUFFER_SIZE = 256;  // Smaller buffer for better cache performance
    std::array<std::function<void()>, BUFFER_SIZE> buffer_;
    std::atomic<size_t> top_{0};
    std::atomic<size_t> bottom_{0};

public:
    // Push work to bottom (owner thread)
    void push_bottom(std::function<void()> task) {
        size_t bottom = bottom_.load(std::memory_order_relaxed);
        size_t top = top_.load(std::memory_order_acquire);

        if (bottom - top >= BUFFER_SIZE) {
            // Buffer full - grow or drop (for now we drop)
            return;
        }

        buffer_[bottom % BUFFER_SIZE] = std::move(task);
        bottom_.store(bottom + 1, std::memory_order_release);
    }

    // Pop work from bottom (owner thread - LIFO)
    std::optional<std::function<void()>> pop_bottom() {
        size_t bottom = bottom_.load(std::memory_order_relaxed) - 1;
        size_t top = top_.load(std::memory_order_acquire);

        if (bottom < top) {
            bottom_.store(top, std::memory_order_relaxed);
            return std::nullopt;
        }

        std::function<void()> task = std::move(buffer_[bottom % BUFFER_SIZE]);

        if (bottom > top) {
            bottom_.store(bottom, std::memory_order_release);
            return task;
        }

        // Race with steal - use CAS
        if (!top_.compare_exchange_strong(top, top + 1,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
            return std::nullopt;
        }

        bottom_.store(bottom + 1, std::memory_order_release);
        return task;
    }

    // Steal work from top (other threads - FIFO)
    std::optional<std::function<void()>> steal() {
        size_t top = top_.load(std::memory_order_acquire);
        size_t bottom = bottom_.load(std::memory_order_acquire);

        if (top >= bottom) {
            return std::nullopt;
        }

        std::function<void()> task = std::move(buffer_[top % BUFFER_SIZE]);

        if (!top_.compare_exchange_strong(top, top + 1,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
            return std::nullopt;
        }

        return task;
    }

    bool empty() const {
        size_t top = top_.load(std::memory_order_acquire);
        size_t bottom = bottom_.load(std::memory_order_acquire);
        return top >= bottom;
    }
};

/**
 * Global work-stealing thread pool singleton - reused across all operations
 */
class ThreadPool {
public:
    using Task = std::function<void()>;

    // Get singleton instance (created with hardware_concurrency threads)
    static ThreadPool& instance() {
        static ThreadPool pool(std::thread::hardware_concurrency());
        return pool;
    }

    // Get instance with specific thread count
    static ThreadPool& instance(size_t num_threads) {
        static ThreadPool pool(num_threads);
        return pool;
    }

    // Submit a task and get a future for the result
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<ReturnType> result = task->get_future();

        // Submit to current thread's deque if available, otherwise distribute
        auto wrapped_task = [task]() { (*task)(); };
        size_t thread_id = get_current_thread_id();

        if (thread_id < deques_.size() && thread_id < num_threads_) {
            deques_[thread_id]->push_bottom(std::move(wrapped_task));
        } else {
            // Distribute to random deque for load balancing
            size_t target = static_cast<size_t>(rand()) % num_threads_;
            deques_[target]->push_bottom(std::move(wrapped_task));
        }

        return result;
    }

    // Submit batch of tasks and wait for all to complete
    template<typename Func>
    void parallel_for(size_t begin, size_t end, Func&& func) {
        if (begin >= end) return;

        const size_t total = end - begin;
        const size_t chunk_size = std::max(size_t(1), total / (num_threads_ * 8));  // Smaller chunks

        std::vector<std::future<void>> futures;
        futures.reserve(total / chunk_size + 1);

        for (size_t i = begin; i < end; i += chunk_size) {
            size_t chunk_end = std::min(i + chunk_size, end);
            futures.push_back(submit([&func, i, chunk_end]() {
                for (size_t j = i; j < chunk_end; ++j) {
                    func(j);
                }
            }));
        }

        for (auto& future : futures) {
            future.get();
        }
    }

    // Parallel for with index and thread_id
    template<typename Func>
    void parallel_for_indexed(size_t begin, size_t end, Func&& func) {
        if (begin >= end) return;

        const size_t total = end - begin;
        const size_t chunk_size = std::max(size_t(1), total / (num_threads_ * 8));

        std::vector<std::future<void>> futures;
        futures.reserve(total / chunk_size + 1);

        for (size_t i = begin; i < end; i += chunk_size) {
            size_t chunk_end = std::min(i + chunk_size, end);
            futures.push_back(submit([this, &func, i, chunk_end]() {
                size_t thread_id = get_current_thread_id();
                for (size_t j = i; j < chunk_end; ++j) {
                    func(j, thread_id);
                }
            }));
        }

        for (auto& future : futures) {
            future.get();
        }
    }

    // Parallel reduce
    template<typename T, typename ReduceFunc, typename CombineFunc>
    T parallel_reduce(size_t begin, size_t end, T init, ReduceFunc&& reduce, CombineFunc&& combine) {
        if (begin >= end) return init;

        const size_t total = end - begin;
        const size_t chunk_size = std::max(size_t(1), total / (num_threads_ * 8));

        std::vector<std::future<T>> futures;
        futures.reserve(total / chunk_size + 1);

        for (size_t i = begin; i < end; i += chunk_size) {
            size_t chunk_end = std::min(i + chunk_size, end);
            futures.push_back(submit([&reduce, init, i, chunk_end]() {
                T local = init;
                for (size_t j = i; j < chunk_end; ++j) {
                    local = reduce(local, j);
                }
                return local;
            }));
        }

        T result = init;
        for (auto& future : futures) {
            result = combine(result, future.get());
        }
        return result;
    }

    size_t num_threads() const { return num_threads_; }

    ~ThreadPool() {
        stop_ = true;
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Non-copyable, non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

private:
    explicit ThreadPool(size_t num_threads) : num_threads_(num_threads), stop_(false) {
        num_threads_ = std::max(size_t(1), num_threads);
        workers_.reserve(num_threads_);
        deques_.reserve(num_threads_);

        for (size_t i = 0; i < num_threads_; ++i) {
            deques_.push_back(std::make_unique<WorkStealingDeque>());
        }

        for (size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back(&ThreadPool::worker_loop, this, i);
        }
    }

    void worker_loop(size_t thread_id) {
        current_thread_id_ = thread_id;

        while (!stop_) {
            std::function<void()> task;

            // First, try to pop from our own deque
            if (auto local_task = deques_[thread_id]->pop_bottom()) {
                task = std::move(*local_task);
            } else {
                // Try to steal from other threads
                bool stole = false;
                for (size_t attempt = 0; attempt < num_threads_ * 2 && !stole; ++attempt) {
                    size_t victim = (thread_id + attempt + 1) % num_threads_;
                    if (auto stolen_task = deques_[victim]->steal()) {
                        task = std::move(*stolen_task);
                        stole = true;
                    }
                }

                if (!stole) {
                    // No work available, yield to avoid busy waiting
                    std::this_thread::yield();
                    continue;
                }
            }

            if (task) {
                task();
            }
        }
    }

    size_t get_current_thread_id() const {
        return current_thread_id_;
    }

    size_t num_threads_;
    std::vector<std::thread> workers_;
    std::vector<std::unique_ptr<WorkStealingDeque>> deques_;
    std::atomic<bool> stop_;

    // Thread-local storage
    static thread_local size_t current_thread_id_;
};

// Thread-local storage for current thread ID
thread_local size_t ThreadPool::current_thread_id_ = 0;

/**
 * Parallel batch operations with pre-allocated per-thread storage
 */
template<typename T>
class ParallelBatch {
public:
    explicit ParallelBatch(size_t batch_size = 10000)
        : batch_size_(batch_size) {
        // Pre-allocate per-thread storage
        size_t num_threads = ThreadPool::instance().num_threads();
        thread_local_storage_.resize(num_threads);
    }

    // Access per-thread storage
    std::vector<T>& local_storage(size_t thread_id) {
        return thread_local_storage_[thread_id];
    }

    // Merge all thread-local results
    std::vector<T> merge() {
        size_t total = 0;
        for (const auto& storage : thread_local_storage_) {
            total += storage.size();
        }

        std::vector<T> result;
        result.reserve(total);

        for (auto& storage : thread_local_storage_) {
            result.insert(result.end(),
                         std::make_move_iterator(storage.begin()),
                         std::make_move_iterator(storage.end()));
            storage.clear();
        }

        return result;
    }

private:
    size_t batch_size_;
    std::vector<std::vector<T>> thread_local_storage_;
};

} // namespace hypercube
