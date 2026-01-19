#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "../benchmark/benchmark_base.hpp"

using Duration = std::chrono::nanoseconds;

struct AggregatedResults {
    std::string benchmark_name;
    size_t total_runs = 0;
    Duration min_duration = Duration::max();
    Duration max_duration = Duration::zero();
    Duration avg_duration = Duration::zero();
    double std_dev = 0.0;
    double throughput_avg = 0.0;
};

class ResultAggregator {
private:
    std::vector<BenchmarkResult> results_;

public:
    void add_result(const BenchmarkResult& result) {
        results_.push_back(result);
    }

    AggregatedResults aggregate() const {
        if (results_.empty()) return {};

        AggregatedResults agg;
        agg.benchmark_name = results_.front().name;
        agg.total_runs = results_.size();

        Duration sum = Duration::zero();
        double throughput_sum = 0.0;

        for (const auto& result : results_) {
            agg.min_duration = std::min(agg.min_duration, result.duration);
            agg.max_duration = std::max(agg.max_duration, result.duration);
            sum += result.duration;
            throughput_sum += result.throughput;
        }

        agg.avg_duration = sum / results_.size();
        agg.throughput_avg = throughput_sum / results_.size();

        // Calculate standard deviation
        double variance = 0.0;
        for (const auto& result : results_) {
            double diff = (result.duration - agg.avg_duration).count();
            variance += diff * diff;
        }
        agg.std_dev = std::sqrt(variance / results_.size());

        return agg;
    }

    void export_csv(const std::string& filename) const {
        std::ofstream file(filename);
        file << "name,duration_ns,throughput,memory_used,success,error_message\n";

        for (const auto& result : results_) {
            file << result.name << ","
                 << result.duration.count() << ","
                 << result.throughput << ","
                 << result.memory_used << ","
                 << (result.success ? "true" : "false") << ","
                 << "\"" << result.error_message << "\"\n";
        }
    }
};