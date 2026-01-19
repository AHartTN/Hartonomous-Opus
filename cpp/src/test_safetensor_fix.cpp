// Test program to verify SafeTensor parsing fixes
#include <iostream>
#include <filesystem>
#include "hypercube/ingest/parsing.hpp"
#include "hypercube/ingest/safetensor.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: test_safetensor_fix <safetensor_file>\n";
        return 1;
    }

    std::string filepath = argv[1];
    if (!std::filesystem::exists(filepath)) {
        std::cerr << "File does not exist: " << filepath << "\n";
        return 1;
    }

    std::cerr << "Testing SafeTensor parsing fixes on: " << filepath << "\n";

    hypercube::ingest::IngestContext ctx;
    ctx.verbose = true;

    // Set cache limit
    hypercube::safetensor::MappedFileCache::instance().set_max_size(5);

    bool success = hypercube::ingest::parse_safetensor_header(ctx, filepath);

    if (success) {
        std::cerr << "SUCCESS: Parsed " << ctx.tensors.size() << " tensors\n";
        for (const auto& [name, meta] : ctx.tensors) {
            std::cerr << "  " << name << " [" << meta.shape[0] << " x " << meta.shape[1] << "] " << meta.dtype << "\n";
        }
        return 0;
    } else {
        std::cerr << "FAILED: Could not parse SafeTensor file\n";
        return 1;
    }
}