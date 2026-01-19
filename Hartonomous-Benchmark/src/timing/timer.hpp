#pragma once

#include <chrono>
#include <string>
#include <iostream>

using Duration = std::chrono::nanoseconds;

class Timer {
private:
    std::chrono::steady_clock::time_point start_time_;
    bool running_ = false;

public:
    void start() {
        start_time_ = std::chrono::steady_clock::now();
        running_ = true;
    }

    void stop() {
        if (!running_) return;
        running_ = false;
    }

    Duration get_elapsed() const {
        if (running_) {
            return std::chrono::duration_cast<Duration>(
                std::chrono::steady_clock::now() - start_time_);
        }
        return Duration::zero();
    }

    bool is_running() const { return running_; }
};

class ScopedTimer {
private:
    Timer& timer_;
    std::string label_;

public:
    ScopedTimer(Timer& timer, const std::string& label = "")
        : timer_(timer), label_(label) {
        timer_.start();
    }

    ~ScopedTimer() {
        timer_.stop();
        if (!label_.empty()) {
            std::cout << label_ << ": " << timer_.get_elapsed().count() << " ns" << std::endl;
        }
    }
};