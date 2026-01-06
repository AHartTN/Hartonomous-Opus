// =============================================================================
// parsing.hpp - Safetensor & Tokenizer Parsing Functions
// =============================================================================
// Modular parsing functions that operate on IngestContext rather than globals.
// =============================================================================

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>

#include "hypercube/ingest/context.hpp"
#include "hypercube/ingest/safetensor.hpp"

namespace fs = std::filesystem;

namespace hypercube {
namespace ingest {

// =============================================================================
// Safetensor Header Parsing
// =============================================================================

inline bool parse_safetensor_header(
    IngestContext& ctx,
    const fs::path& path, 
    const std::string& shard_file = ""
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open: " << path << "\n";
        return false;
    }
    
    // Read 8-byte header size (little-endian)
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), 8);
    
    // Read JSON header
    std::vector<char> buf(header_size);
    file.read(buf.data(), header_size);
    std::string json(buf.begin(), buf.end());
    
    // Simple parser: find each tensor entry
    size_t pos = 0;
    while ((pos = json.find("\"dtype\"", pos)) != std::string::npos) {
        size_t entry_start = json.rfind("{", pos);
        size_t name_end = json.rfind("\":", entry_start);
        size_t name_start = json.rfind("\"", name_end - 1);
        
        if (name_start == std::string::npos || name_end == std::string::npos) {
            pos++;
            continue;
        }
        
        std::string name = json.substr(name_start + 1, name_end - name_start - 1);
        
        if (name == "__metadata__" || name == "format") {
            pos++;
            continue;
        }
        
        safetensor::TensorMeta meta;
        meta.name = name;
        meta.shard_file = shard_file.empty() ? path.string() : shard_file;
        
        // Extract dtype
        size_t dtype_pos = json.find(":", pos) + 1;
        size_t dtype_q1 = json.find("\"", dtype_pos);
        size_t dtype_q2 = json.find("\"", dtype_q1 + 1);
        meta.dtype = json.substr(dtype_q1 + 1, dtype_q2 - dtype_q1 - 1);
        
        // Extract shape
        size_t shape_pos = json.find("\"shape\"", pos);
        size_t shape_start = json.find("[", shape_pos);
        size_t shape_end = json.find("]", shape_start);
        std::string shape_str = json.substr(shape_start + 1, shape_end - shape_start - 1);
        std::stringstream ss(shape_str);
        std::string dim;
        while (std::getline(ss, dim, ',')) {
            meta.shape.push_back(std::stoll(dim));
        }
        
        // Extract data offsets
        size_t off_pos = json.find("\"data_offsets\"", pos);
        size_t off_start = json.find("[", off_pos);
        size_t off_end = json.find("]", off_start);
        std::string off_str = json.substr(off_start + 1, off_end - off_start - 1);
        size_t comma = off_str.find(",");
        meta.data_offset_start = std::stoull(off_str.substr(0, comma));
        meta.data_offset_end = std::stoull(off_str.substr(comma + 1));
        
        ctx.tensors[name] = meta;
        pos = off_end;
    }
    
    return true;
}

// =============================================================================
// Parse Sharded Model Index (model.safetensors.index.json)
// =============================================================================

inline bool parse_model_index(IngestContext& ctx, const fs::path& index_path) {
    std::ifstream file(index_path);
    if (!file) return false;
    
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    
    size_t pos = content.find("\"weight_map\"");
    if (pos == std::string::npos) return false;
    
    pos = content.find("{", pos);
    size_t end = pos;
    int depth = 1;
    while (depth > 0 && end < content.size()) {
        end++;
        if (content[end] == '{') depth++;
        else if (content[end] == '}') depth--;
    }
    
    std::string map_str = content.substr(pos + 1, end - pos - 1);
    
    std::unordered_set<std::string> shards_to_parse;
    size_t i = 0;
    while (i < map_str.size()) {
        size_t q1 = map_str.find("\"", i);
        if (q1 == std::string::npos) break;
        size_t q2 = map_str.find("\"", q1 + 1);
        if (q2 == std::string::npos) break;
        
        size_t q3 = map_str.find("\"", q2 + 1);
        size_t q4 = map_str.find("\"", q3 + 1);
        if (q3 == std::string::npos || q4 == std::string::npos) break;
        std::string shard = map_str.substr(q3 + 1, q4 - q3 - 1);
        
        shards_to_parse.insert(shard);
        i = q4 + 1;
    }
    
    fs::path parent = index_path.parent_path();
    for (const auto& shard : shards_to_parse) {
        fs::path shard_path = parent / shard;
        if (fs::exists(shard_path)) {
            std::cerr << "  Parsing shard: " << shard << "\n";
            parse_safetensor_header(ctx, shard_path, shard_path.string());
        }
    }
    
    std::cerr << "[INDEX] Parsed " << shards_to_parse.size() << " shards, " 
              << ctx.tensors.size() << " tensors\n";
    return true;
}

// =============================================================================
// Parse Tokenizer (BPE Merges + Vocab from tokenizer.json)
// =============================================================================

inline bool parse_tokenizer(IngestContext& ctx, const fs::path& tokenizer_path) {
    std::ifstream file(tokenizer_path);
    if (!file) {
        std::cerr << "Cannot open tokenizer: " << tokenizer_path << "\n";
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    
    // Parse merges array
    size_t pos = content.find("\"merges\"");
    if (pos != std::string::npos) {
        pos = content.find("[", pos);
        size_t end = content.find("]", pos);
        if (pos != std::string::npos && end != std::string::npos) {
            std::string arr = content.substr(pos + 1, end - pos - 1);
            
            size_t i = 0;
            while (i < arr.size()) {
                size_t q1 = arr.find("\"", i);
                if (q1 == std::string::npos) break;
                size_t q2 = arr.find("\"", q1 + 1);
                if (q2 == std::string::npos) break;
                
                std::string merge = arr.substr(q1 + 1, q2 - q1 - 1);
                size_t sp = merge.find(" ");
                if (sp != std::string::npos) {
                    ctx.bpe_merges.emplace_back(merge.substr(0, sp), merge.substr(sp + 1));
                }
                i = q2 + 1;
            }
        }
    }
    
    // Parse vocab
    pos = content.find("\"vocab\"");
    if (pos != std::string::npos) {
        pos = content.find("{", pos);
        size_t end = pos;
        int depth = 1;
        while (depth > 0 && end < content.size()) {
            end++;
            if (content[end] == '{') depth++;
            else if (content[end] == '}') depth--;
        }
        
        std::string vocab_str = content.substr(pos + 1, end - pos - 1);
        size_t i = 0;
        while (i < vocab_str.size()) {
            size_t q1 = vocab_str.find("\"", i);
            if (q1 == std::string::npos) break;
            size_t q2 = vocab_str.find("\"", q1 + 1);
            if (q2 == std::string::npos) break;
            
            std::string token = vocab_str.substr(q1 + 1, q2 - q1 - 1);
            
            size_t colon = vocab_str.find(":", q2);
            size_t comma = vocab_str.find(",", colon);
            if (comma == std::string::npos) comma = vocab_str.size();
            
            int idx = std::stoi(vocab_str.substr(colon + 1, comma - colon - 1));
            ctx.vocab[token] = idx;
            
            i = comma + 1;
        }
    }
    
    std::cerr << "[TOKENIZER] Loaded " << ctx.bpe_merges.size() << " BPE merges, "
              << ctx.vocab.size() << " vocab entries\n";
    return true;
}

// =============================================================================
// Parse Vocab File (BERT-style vocab.txt)
// Parallelized composition computation
// =============================================================================

inline bool parse_vocab(IngestContext& ctx, const fs::path& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file) {
        std::cerr << "Cannot open vocab: " << vocab_path << "\n";
        return false;
    }
    
    // Phase 1: Read all lines sequentially (I/O bound)
    std::vector<std::string> lines;
    lines.reserve(50000);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            lines.push_back(std::move(line));
        }
    }
    
    size_t total = lines.size();
    ctx.vocab_tokens.resize(total);
    
    // Phase 2: Compute compositions in parallel (CPU bound)
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 16) num_threads = 16;
    
    std::atomic<size_t> idx{0};
    std::atomic<size_t> completed{0};
    auto start = std::chrono::steady_clock::now();
    
    auto worker = [&]() {
        while (true) {
            size_t i = idx.fetch_add(1);
            if (i >= total) break;
            
            TokenInfo info;
            info.text = lines[i];
            info.comp = AtomCalculator::compute_vocab_token(lines[i]);
            ctx.vocab_tokens[i] = std::move(info);
            
            completed.fetch_add(1);
        }
    };
    
    // Progress thread
    std::atomic<bool> done{false};
    std::thread progress([&]() {
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            size_t c = completed.load();
            if (c > 0 && c < total) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                double rate = (elapsed > 0) ? (c * 1000.0 / elapsed) : 0;
                std::cerr << "  [VOCAB] " << c << "/" << total << " (" 
                          << std::fixed << std::setprecision(0) << rate << " tok/s)\r" << std::flush;
            }
        }
    });
    
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back(worker);
    }
    for (auto& th : workers) th.join();
    done.store(true);
    progress.join();
    
    // Build index map (sequential, fast)
    for (size_t i = 0; i < total; ++i) {
        ctx.token_to_idx[ctx.vocab_tokens[i].text] = i;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "\n[VOCAB] Loaded " << total << " tokens in " << elapsed << "ms\n";
    return true;
}

// =============================================================================
// Tensor Path Hierarchy Splitting
// =============================================================================

inline std::vector<std::string> split_tensor_path(const std::string& tensor_name) {
    std::vector<std::string> components;
    std::string current;
    
    for (char c : tensor_name) {
        if (c == '.') {
            if (!current.empty()) {
                if (components.empty()) {
                    components.push_back(current);
                } else {
                    components.push_back(components.back() + "." + current);
                }
                current.clear();
            }
        } else {
            current += c;
        }
    }
    
    if (!current.empty()) {
        if (components.empty()) {
            components.push_back(current);
        } else {
            components.push_back(components.back() + "." + current);
        }
    }
    
    return components;
}

} // namespace ingest
} // namespace hypercube
