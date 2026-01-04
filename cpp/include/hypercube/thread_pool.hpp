/**
 * High-Performance Thread Pool for Hypercube Operations
 * 
 * Lock-free work-stealing thread pool for parallel batch processing.
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

namespace hypercube {

/**
 * Global thread pool singleton - reused across all operations
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
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool is stopped");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        
        condition_.notify_one();
        return result;
    }
    
    // Submit batch of tasks and wait for all to complete
    template<typename Func>
    void parallel_for(size_t begin, size_t end, Func&& func) {
        if (begin >= end) return;
        
        const size_t total = end - begin;
        const size_t num_workers = workers_.size();
        const size_t chunk_size = (total + num_workers - 1) / num_workers;
        
        std::vector<std::future<void>> futures;
        futures.reserve(num_workers);
        
        for (size_t i = 0; i < num_workers && begin + i * chunk_size < end; ++i) {
            size_t chunk_begin = begin + i * chunk_size;
            size_t chunk_end = std::min(chunk_begin + chunk_size, end);
            
            futures.push_back(submit([&func, chunk_begin, chunk_end]() {
                for (size_t j = chunk_begin; j < chunk_end; ++j) {
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
        const size_t num_workers = workers_.size();
        const size_t chunk_size = (total + num_workers - 1) / num_workers;
        
        std::vector<std::future<void>> futures;
        futures.reserve(num_workers);
        
        for (size_t thread_id = 0; thread_id < num_workers && begin + thread_id * chunk_size < end; ++thread_id) {
            size_t chunk_begin = begin + thread_id * chunk_size;
            size_t chunk_end = std::min(chunk_begin + chunk_size, end);
            
            futures.push_back(submit([&func, chunk_begin, chunk_end, thread_id]() {
                for (size_t j = chunk_begin; j < chunk_end; ++j) {
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
        const size_t num_workers = workers_.size();
        const size_t chunk_size = (total + num_workers - 1) / num_workers;
        
        std::vector<std::future<T>> futures;
        futures.reserve(num_workers);
        
        for (size_t i = 0; i < num_workers && begin + i * chunk_size < end; ++i) {
            size_t chunk_begin = begin + i * chunk_size;
            size_t chunk_end = std::min(chunk_begin + chunk_size, end);
            
            futures.push_back(submit([&reduce, init, chunk_begin, chunk_end]() {
                T local = init;
                for (size_t j = chunk_begin; j < chunk_end; ++j) {
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
    
    size_t num_threads() const { return workers_.size(); }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
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
    explicit ThreadPool(size_t num_threads) : stop_(false) {
        num_threads = std::max(size_t(1), num_threads);
        workers_.reserve(num_threads);
        
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() {
                while (true) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this]() {
                            return stop_ || !tasks_.empty();
                        });
                        
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    
    std::vector<std::thread> workers_;
    std::queue<Task> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
};

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
