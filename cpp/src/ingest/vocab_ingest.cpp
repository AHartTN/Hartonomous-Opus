/**
 * Streaming Vocabulary Ingester
 * 
 * Processes tokens ONE AT A TIME through the PMI engine:
 * - Each token → atom sequence → PMI contraction → Merkle DAG
 * - Run-length encoding for repeated patterns
 * - Deduplication at every level (inherent in content-addressing)
 * 
 * This is the RIGHT way: "walking" shares structure with "walked", "walks"
 * because they share the 'walk' composition in the DAG.
 * 
 * Standalone: no hypercube library dependencies, just blake3 and standard C++20.
 */

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <array>
#include <cmath>
#include <span>

// Use bundled blake3 header-only if available, otherwise inline a simple hash
#if __has_include("blake3.h")
#include "blake3.h"
#define HAS_BLAKE3 1
#else
#define HAS_BLAKE3 0
#endif

namespace fs = std::filesystem;

namespace {

// ============================================================================
// Inline Types (standalone - no hypercube library dependency)
// ============================================================================

struct Hash256 {
    std::array<uint8_t, 32> bytes;
    
    bool operator==(const Hash256& o) const { return bytes == o.bytes; }
    bool operator<(const Hash256& o) const { return bytes < o.bytes; }
};

struct Hash256Hasher {
    size_t operator()(const Hash256& h) const {
        uint64_t v;
        std::memcpy(&v, h.bytes.data(), 8);
        return v;
    }
};

// Simple FNV-1a based hash (if no BLAKE3 available)
inline Hash256 hash_bytes(const uint8_t* data, size_t len) {
#if HAS_BLAKE3
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, data, len);
    Hash256 result;
    blake3_hasher_finalize(&hasher, result.bytes.data(), 32);
    return result;
#else
    // Fallback: FNV-1a extended to 256 bits (4x64)
    Hash256 result;
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME = 0x00000100000001b3ULL;
    
    uint64_t h[4] = {FNV_OFFSET, FNV_OFFSET ^ 1, FNV_OFFSET ^ 2, FNV_OFFSET ^ 3};
    for (size_t i = 0; i < len; ++i) {
        for (int j = 0; j < 4; ++j) {
            h[j] ^= data[i];
            h[j] *= FNV_PRIME;
        }
    }
    std::memcpy(result.bytes.data(), h, 32);
    return result;
#endif
}

inline Hash256 hash_codepoint(uint32_t cp) {
    uint8_t buf[4];
    buf[0] = static_cast<uint8_t>(cp & 0xFF);
    buf[1] = static_cast<uint8_t>((cp >> 8) & 0xFF);
    buf[2] = static_cast<uint8_t>((cp >> 16) & 0xFF);
    buf[3] = static_cast<uint8_t>((cp >> 24) & 0xFF);
    return hash_bytes(buf, 4);
}

// ============================================================================
// Composition Hash (ordered Merkle)
// ============================================================================
Hash256 compute_composition_hash(const std::vector<Hash256>& children) {
    std::vector<uint8_t> buffer;
    buffer.reserve(children.size() * 36);
    
    for (size_t i = 0; i < children.size(); ++i) {
        uint32_t ordinal = static_cast<uint32_t>(i);
        buffer.push_back(static_cast<uint8_t>(ordinal & 0xFF));
        buffer.push_back(static_cast<uint8_t>((ordinal >> 8) & 0xFF));
        buffer.push_back(static_cast<uint8_t>((ordinal >> 16) & 0xFF));
        buffer.push_back(static_cast<uint8_t>((ordinal >> 24) & 0xFF));
        buffer.insert(buffer.end(), children[i].bytes.begin(), children[i].bytes.end());
    }
    
    return hash_bytes(buffer.data(), buffer.size());
}

// ============================================================================
// Streaming PMI Context - maintains state across tokens
// ============================================================================
class StreamingPMI {
public:
    // Atom-level statistics
    std::unordered_map<Hash256, uint64_t, Hash256Hasher> atom_counts;
    
    // Pair statistics for PMI
    std::unordered_map<uint64_t, uint64_t> pair_counts;
    
    // Known composition hashes (dedup)
    std::set<Hash256> known_compositions;
    
    uint64_t total_atoms = 0;
    uint64_t total_tokens = 0;
    uint64_t total_compositions = 0;
    uint64_t dedupe_saves = 0;
    
    // PMI threshold for merging
    double pmi_threshold = 0.0;
    size_t min_pair_count = 2;
    
    // Process a single token
    size_t process_token(const std::string& token) {
        if (token.empty()) return 0;
        
        // Step 1: Convert token to atom sequence
        std::vector<Hash256> atom_hashes;
        
        for (size_t i = 0; i < token.size(); ) {
            uint32_t codepoint;
            if ((static_cast<unsigned char>(token[i]) & 0x80) == 0) {
                codepoint = static_cast<unsigned char>(token[i]);
                i++;
            } else if ((static_cast<unsigned char>(token[i]) & 0xE0) == 0xC0 && i + 1 < token.size()) {
                codepoint = ((token[i] & 0x1F) << 6) | (token[i+1] & 0x3F);
                i += 2;
            } else if ((static_cast<unsigned char>(token[i]) & 0xF0) == 0xE0 && i + 2 < token.size()) {
                codepoint = ((token[i] & 0x0F) << 12) | 
                           ((token[i+1] & 0x3F) << 6) | (token[i+2] & 0x3F);
                i += 3;
            } else if ((static_cast<unsigned char>(token[i]) & 0xF8) == 0xF0 && i + 3 < token.size()) {
                codepoint = ((token[i] & 0x07) << 18) | 
                           ((token[i+1] & 0x3F) << 12) |
                           ((token[i+2] & 0x3F) << 6) | (token[i+3] & 0x3F);
                i += 4;
            } else {
                codepoint = static_cast<unsigned char>(token[i]);
                i++;
            }
            
            // Compute deterministic atom hash
            Hash256 atom_hash = hash_codepoint(codepoint);
            atom_hashes.push_back(atom_hash);
            
            // Update atom counts
            atom_counts[atom_hash]++;
            total_atoms++;
        }
        
        if (atom_hashes.empty()) return 0;
        
        // Step 2: Update pair statistics
        for (size_t i = 0; i + 1 < atom_hashes.size(); ++i) {
            uint64_t h1, h2;
            std::memcpy(&h1, atom_hashes[i].bytes.data(), 8);
            std::memcpy(&h2, atom_hashes[i+1].bytes.data(), 8);
            pair_counts[h1 ^ h2]++;
        }
        
        // Step 3: PMI contraction
        std::vector<Hash256> current = atom_hashes;
        size_t new_compositions = 0;
        
        bool contracted = true;
        while (contracted && current.size() > 1) {
            contracted = false;
            double best_pmi = pmi_threshold;
            size_t best_idx = 0;
            
            for (size_t i = 0; i + 1 < current.size(); ++i) {
                uint64_t h1, h2;
                std::memcpy(&h1, current[i].bytes.data(), 8);
                std::memcpy(&h2, current[i+1].bytes.data(), 8);
                
                uint64_t pair_count = pair_counts[h1 ^ h2];
                if (pair_count < min_pair_count) continue;
                
                double p_a = static_cast<double>(atom_counts[current[i]]) / total_atoms;
                double p_b = static_cast<double>(atom_counts[current[i+1]]) / total_atoms;
                double p_ab = static_cast<double>(pair_count) / static_cast<double>(std::max(static_cast<uint64_t>(1), total_atoms - total_tokens));
                
                if (p_a > 0 && p_b > 0 && p_ab > 0) {
                    double pmi = std::log2(p_ab / (p_a * p_b));
                    if (pmi > best_pmi) {
                        best_pmi = pmi;
                        best_idx = i;
                    }
                }
            }
            
            if (best_pmi > pmi_threshold) {
                std::vector<Hash256> children = {current[best_idx], current[best_idx + 1]};
                Hash256 comp_hash = compute_composition_hash(children);
                
                if (known_compositions.find(comp_hash) == known_compositions.end()) {
                    known_compositions.insert(comp_hash);
                    total_compositions++;
                    new_compositions++;
                } else {
                    dedupe_saves++;
                }
                
                // Replace pair with composition
                current[best_idx] = comp_hash;
                current.erase(current.begin() + best_idx + 1);
                contracted = true;
            }
        }
        
        // Step 4: If still multiple elements, create final composition
        if (current.size() > 1) {
            Hash256 final_hash = compute_composition_hash(current);
            
            if (known_compositions.find(final_hash) == known_compositions.end()) {
                known_compositions.insert(final_hash);
                total_compositions++;
                new_compositions++;
            } else {
                dedupe_saves++;
            }
        }
        
        total_tokens++;
        return new_compositions;
    }
    
    void print_stats(std::ostream& out) const {
        out << "\n=== Streaming PMI Stats ===\n";
        out << "Tokens processed: " << total_tokens << "\n";
        out << "Atoms seen: " << total_atoms << "\n";
        out << "Unique atoms: " << atom_counts.size() << "\n";
        out << "Compositions created: " << total_compositions << "\n";
        out << "Dedup saves: " << dedupe_saves << "\n";
        out << "Compression ratio: " << (total_atoms > 0 ? 
            static_cast<double>(total_compositions + atom_counts.size()) / total_atoms : 0) << "\n";
    }
};

// ============================================================================
// Vocabulary Parser (streaming from JSON)
// ============================================================================
class VocabStream {
public:
    std::string json_content;
    size_t json_pos = 0;
    bool in_vocab_section = false;
    bool finished = false;
    std::ifstream txt_file;
    std::string source_type;
    
    bool open(const fs::path& model_path) {
        fs::path tokenizer_json = model_path / "tokenizer.json";
        fs::path vocab_txt = model_path / "vocab.txt";
        
        // Check snapshot directories
        if (!fs::exists(tokenizer_json) && !fs::exists(vocab_txt)) {
            for (const auto& entry : fs::recursive_directory_iterator(model_path)) {
                if (entry.path().filename() == "tokenizer.json" && 
                    entry.path().string().find(".cache") == std::string::npos) {
                    tokenizer_json = entry.path();
                }
                if (entry.path().filename() == "vocab.txt" && 
                    entry.path().string().find(".cache") == std::string::npos) {
                    vocab_txt = entry.path();
                }
            }
        }
        
        if (fs::exists(tokenizer_json)) {
            std::ifstream file(tokenizer_json);
            if (file) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                json_content = buffer.str();
                // Strip BOM if present
                if (json_content.size() >= 3 &&
                    static_cast<unsigned char>(json_content[0]) == 0xEF &&
                    static_cast<unsigned char>(json_content[1]) == 0xBB &&
                    static_cast<unsigned char>(json_content[2]) == 0xBF) {
                    json_content = json_content.substr(3);
                    std::cerr << "[VOCAB] Stripped BOM from tokenizer.json" << std::endl;
                }
                source_type = "tokenizer_json";
                
                size_t vocab_start = json_content.find("\"vocab\"");
                if (vocab_start == std::string::npos) return false;
                json_pos = json_content.find('{', vocab_start);
                if (json_pos == std::string::npos) return false;
                json_pos++;
                in_vocab_section = true;
                return true;
            }
        }
        
        if (fs::exists(vocab_txt)) {
            txt_file.open(vocab_txt);
            if (txt_file) {
                // Check for BOM in vocab.txt
                char bom_check[3];
                if (txt_file.read(bom_check, 3) && txt_file.gcount() == 3) {
                    if (static_cast<unsigned char>(bom_check[0]) == 0xEF &&
                        static_cast<unsigned char>(bom_check[1]) == 0xBB &&
                        static_cast<unsigned char>(bom_check[2]) == 0xBF) {
                        std::cerr << "[VOCAB] BOM detected in vocab.txt, but getline will handle it" << std::endl;
                    } else {
                        // Not BOM, rewind
                        txt_file.seekg(0);
                    }
                }
                source_type = "vocab_txt";
                return true;
            }
        }
        
        return false;
    }
    
    std::string next_token() {
        if (finished) return "";
        
        if (source_type == "vocab_txt") {
            std::string line;
            if (std::getline(txt_file, line)) {
                return line;
            }
            finished = true;
            return "";
        }
        
        if (source_type == "tokenizer_json") {
            std::string token;
            bool in_string = false;
            bool found_colon = false;
            int brace_depth = 1;
            
            while (json_pos < json_content.size() && brace_depth > 0) {
                char c = json_content[json_pos];
                
                if (c == '\\' && json_pos + 1 < json_content.size() && in_string) {
                    char next = json_content[json_pos + 1];
                    if (next == 'n') token += '\n';
                    else if (next == 't') token += '\t';
                    else if (next == 'r') token += '\r';
                    else if (next == '"') token += '"';
                    else if (next == '\\') token += '\\';
                    else if (next == 'u' && json_pos + 5 < json_content.size()) {
                        std::string hex = json_content.substr(json_pos + 2, 4);
                        try {
                            int cp = std::stoi(hex, nullptr, 16);
                            if (cp < 0x80) {
                                token += static_cast<char>(cp);
                            } else if (cp < 0x800) {
                                token += static_cast<char>(0xC0 | (cp >> 6));
                                token += static_cast<char>(0x80 | (cp & 0x3F));
                            } else {
                                token += static_cast<char>(0xE0 | (cp >> 12));
                                token += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                token += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                        } catch (...) {}
                        json_pos += 4;
                    } else {
                        token += next;
                    }
                    json_pos += 2;
                    continue;
                }
                
                if (c == '"') {
                    if (in_string) {
                        if (!found_colon) {
                            in_string = false;
                        } else {
                            in_string = false;
                            token.clear();
                        }
                    } else {
                        in_string = true;
                    }
                } else if (in_string) {
                    if (!found_colon) {
                        token += c;
                    }
                } else if (c == ':') {
                    found_colon = true;
                } else if (c == ',' || c == '}') {
                    if (c == '}') brace_depth--;
                    
                    if (!token.empty() && found_colon) {
                        json_pos++;
                        return token;
                    }
                    token.clear();
                    found_colon = false;
                } else if (c == '{') {
                    brace_depth++;
                }
                
                json_pos++;
            }
            
            finished = true;
            return "";
        }
        
        return "";
    }
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <model_path>\n"
              << "\n"
              << "Streams vocabulary through PMI engine one token at a time.\n"
              << "Builds Merkle DAG with full deduplication.\n"
              << "\n"
              << "Supports MULTIPLE model paths - state accumulates across all,\n"
              << "so compositions are shared between models!\n"
              << "\n"
              << "Options:\n"
              << "  --pmi-threshold <val>  Minimum PMI for merging (default: 0.0)\n"
              << "  --min-pairs <n>        Minimum pair occurrences (default: 2)\n"
              << "  --batch <n>            Batch size for stats output (default: 10000)\n"
              << "  --verbose              Print progress\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog << " model1 model2 model3  # Process all, share state\n";
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    double pmi_threshold = 0.0;
    size_t min_pairs = 2;
    size_t batch_size = 10000;
    bool verbose = false;
    std::vector<std::string> model_paths;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pmi-threshold" && i + 1 < argc) {
            pmi_threshold = std::stod(argv[++i]);
        } else if (arg == "--min-pairs" && i + 1 < argc) {
            min_pairs = std::stoull(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoull(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            model_paths.push_back(arg);
        }
    }
    
    if (model_paths.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Single PMI engine shared across ALL models
    StreamingPMI pmi;
    pmi.pmi_threshold = pmi_threshold;
    pmi.min_pair_count = min_pairs;
    
    auto global_start = std::chrono::high_resolution_clock::now();
    size_t total_tokens = 0;
    size_t models_processed = 0;
    
    for (const auto& model_path : model_paths) {
        fs::path path(model_path);
        if (!fs::exists(path)) {
            std::cerr << "Warning: Path not found: " << model_path << "\n";
            continue;
        }
        
        VocabStream vocab;
        if (!vocab.open(path)) {
            std::cerr << "Warning: No vocabulary found in " << model_path << "\n";
            continue;
        }
        
        std::cerr << "\n=== Processing: " << model_path << " ===\n";
        std::cerr << "Source: " << vocab.source_type << "\n";
        
        auto model_start = std::chrono::high_resolution_clock::now();
        size_t model_tokens = 0;
        size_t model_compositions_before = pmi.total_compositions;
        size_t model_dedup_before = pmi.dedupe_saves;
        
        std::string token;
        while (!(token = vocab.next_token()).empty()) {
            pmi.process_token(token);
            model_tokens++;
            total_tokens++;
            
            if (verbose && model_tokens % batch_size == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - model_start).count();
                double tps = model_tokens * 1000.0 / static_cast<double>(std::max(1LL, static_cast<long long>(elapsed)));
                
                std::cerr << "  " << model_tokens << " tokens, " 
                          << (pmi.total_compositions - model_compositions_before) << " new comps, "
                          << (pmi.dedupe_saves - model_dedup_before) << " dedup (" << tps << " tok/s)\n";
            }
        }
        
        auto model_end = std::chrono::high_resolution_clock::now();
        auto model_duration = std::chrono::duration_cast<std::chrono::milliseconds>(model_end - model_start).count();
        
        std::cerr << "Model complete: " << model_tokens << " tokens, "
                  << (pmi.total_compositions - model_compositions_before) << " NEW compositions, "
                  << (pmi.dedupe_saves - model_dedup_before) << " dedup in " << model_duration << "ms\n";
        
        models_processed++;
    }
    
    auto global_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(global_end - global_start).count();
    
    std::cerr << "\n";
    pmi.print_stats(std::cerr);
    std::cerr << "\n=== Multi-Model Summary ===\n";
    std::cerr << "Models processed: " << models_processed << "\n";
    std::cerr << "Total tokens: " << total_tokens << "\n";
    std::cerr << "Total time: " << total_duration << " ms\n";
    std::cerr << "Overall rate: " << (total_tokens * 1000.0 / static_cast<double>(std::max(1LL, static_cast<long long>(total_duration)))) << " tok/s\n";
    
    // JSON output
    std::cout << "{\n";
    std::cout << "  \"models_processed\": " << models_processed << ",\n";
    std::cout << "  \"total_tokens\": " << total_tokens << ",\n";
    std::cout << "  \"unique_atoms\": " << pmi.atom_counts.size() << ",\n";
    std::cout << "  \"compositions_created\": " << pmi.total_compositions << ",\n";
    std::cout << "  \"dedup_saves\": " << pmi.dedupe_saves << ",\n";
    std::cout << "  \"duration_ms\": " << total_duration << "\n";
    std::cout << "}\n";
    
    return 0;
}
