/**
 * Model Discovery - Universal HuggingFace Model Scanner
 * 
 * Scans configured paths for HuggingFace model directories and registers them.
 * Models are detected by presence of config.json with "model_type" field.
 * 
 * All model vocabs are ingested as TEXT - the universal substrate doesn't care
 * if tokens come from Llama, BERT, GPT, or anywhere else. Text is text.
 */

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <optional>

namespace fs = std::filesystem;

namespace hypercube::ingest {

struct ModelInfo {
    std::string path;           // Absolute path to snapshot dir
    std::string name;           // e.g., "sentence-transformers/all-MiniLM-L6-v2"
    std::string model_type;     // e.g., "bert", "llama4", "llama"
    std::string tokenizer_type; // e.g., "BPE", "WordPiece", "SentencePiece"
    size_t vocab_size = 0;
    size_t hidden_size = 0;
    size_t num_layers = 0;
    size_t num_experts = 0;     // For MoE models
    bool is_multimodal = false;
    
    std::string safetensor_path() const {
        fs::path p(path);
        // Check for single file or sharded
        if (fs::exists(p / "model.safetensors")) {
            return (p / "model.safetensors").string();
        }
        // Look for first shard
        for (const auto& entry : fs::directory_iterator(p)) {
            if (entry.path().filename().string().find("model-00001") != std::string::npos &&
                entry.path().extension() == ".safetensors") {
                return entry.path().string();
            }
        }
        return "";
    }
    
    std::string tokenizer_json_path() const {
        fs::path p(path);
        if (fs::exists(p / "tokenizer.json")) {
            return (p / "tokenizer.json").string();
        }
        return "";
    }
    
    std::string vocab_txt_path() const {
        fs::path p(path);
        if (fs::exists(p / "vocab.txt")) {
            return (p / "vocab.txt").string();
        }
        return "";
    }
};

// Simple JSON value extraction (no external deps)
std::optional<std::string> extract_json_string(const std::string& json, const std::string& key) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1].str();
    }
    return std::nullopt;
}

std::optional<size_t> extract_json_int(const std::string& json, const std::string& key) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stoull(match[1].str());
    }
    return std::nullopt;
}

std::optional<ModelInfo> parse_model_config(const fs::path& config_path) {
    std::ifstream file(config_path);
    if (!file) return std::nullopt;
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    ModelInfo info;
    info.path = config_path.parent_path().string();
    
    // Extract model_type
    auto model_type = extract_json_string(json, "model_type");
    if (!model_type) return std::nullopt;  // Not a valid model config
    info.model_type = *model_type;
    
    // Extract dimensions
    info.vocab_size = extract_json_int(json, "vocab_size").value_or(0);
    info.hidden_size = extract_json_int(json, "hidden_size").value_or(0);
    info.num_layers = extract_json_int(json, "num_hidden_layers").value_or(0);
    info.num_experts = extract_json_int(json, "num_local_experts").value_or(0);
    
    // Check for multimodal (vision_config present)
    info.is_multimodal = json.find("\"vision_config\"") != std::string::npos;
    
    // Extract name from path (models--org--name pattern)
    std::regex name_pattern("models--([^/\\\\]+)--([^/\\\\]+)");
    std::smatch match;
    std::string path_str = config_path.string();
    if (std::regex_search(path_str, match, name_pattern)) {
        info.name = match[1].str() + "/" + match[2].str();
    } else {
        info.name = config_path.parent_path().filename().string();
    }
    
    return info;
}

std::optional<std::string> detect_tokenizer_type(const ModelInfo& model) {
    std::string tokenizer_path = model.tokenizer_json_path();
    if (tokenizer_path.empty()) {
        // Check for tokenizer.model (SentencePiece)
        fs::path p(model.path);
        if (fs::exists(p / "tokenizer.model")) {
            return "SentencePiece";
        }
        return std::nullopt;
    }
    
    std::ifstream file(tokenizer_path);
    if (!file) return std::nullopt;
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    // Look for tokenizer model type
    if (json.find("\"type\": \"BPE\"") != std::string::npos ||
        json.find("\"type\":\"BPE\"") != std::string::npos) {
        return "BPE";
    }
    if (json.find("\"type\": \"WordPiece\"") != std::string::npos ||
        json.find("\"type\":\"WordPiece\"") != std::string::npos) {
        return "WordPiece";
    }
    if (json.find("\"type\": \"Unigram\"") != std::string::npos ||
        json.find("\"type\":\"Unigram\"") != std::string::npos) {
        return "Unigram";
    }
    
    return "Unknown";
}

std::vector<ModelInfo> discover_models(const std::vector<std::string>& search_paths) {
    std::vector<ModelInfo> models;
    
    for (const auto& base_path : search_paths) {
        if (!fs::exists(base_path)) {
            std::cerr << "[SKIP] Path not found: " << base_path << "\n";
            continue;
        }
        
        std::cerr << "[SCAN] " << base_path << "\n";

        // Recursively find config.json files (with error handling for permission-denied)
        try {
            for (const auto& entry : fs::recursive_directory_iterator(
                    base_path, fs::directory_options::skip_permission_denied)) {
                try {
                    if (entry.is_regular_file() && entry.path().filename() == "config.json") {
                        // Skip .cache directories
                        std::string path_str = entry.path().string();
                        if (path_str.find(".cache") != std::string::npos) continue;

                        // Prefer snapshot directories
                        if (path_str.find("snapshots") == std::string::npos &&
                            path_str.find("snapshot") == std::string::npos) {
                            // Only accept if no snapshots dir exists nearby
                            fs::path parent = entry.path().parent_path();
                            if (fs::exists(parent / "snapshots")) continue;
                        }

                        auto info = parse_model_config(entry.path());
                        if (info) {
                            // Detect tokenizer type
                            auto tok_type = detect_tokenizer_type(*info);
                            if (tok_type) info->tokenizer_type = *tok_type;

                            models.push_back(std::move(*info));
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Skipping entry: " << e.what() << "\n";
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Warning: Directory scan error in " << base_path << ": " << e.what() << "\n";
        }
    }
    
    return models;
}

std::vector<std::string> split_paths(const std::string& paths) {
    std::vector<std::string> result;
    std::string delim = ";";  // Windows default, also works on Linux
    
    size_t start = 0;
    size_t end = paths.find(delim);
    while (end != std::string::npos) {
        std::string path = paths.substr(start, end - start);
        if (!path.empty()) result.push_back(path);
        start = end + 1;
        end = paths.find(delim, start);
    }
    if (start < paths.length()) {
        result.push_back(paths.substr(start));
    }
    
    return result;
}

} // namespace hypercube::ingest

// Standalone discovery tool
int main(int argc, char* argv[]) {
    using namespace hypercube::ingest;
    
    std::cerr << "=== Hypercube Model Discovery ===\n\n";
    
    std::vector<std::string> search_paths;
    
    // Get paths from environment or args
    std::string env_paths_str;
#if defined(_WIN32)
    char* env_paths = nullptr;
    size_t len;
    if (_dupenv_s(&env_paths, &len, "HC_MODEL_PATHS") == 0 && env_paths != nullptr) {
        env_paths_str = env_paths;
        free(env_paths);
    }
#else
    if (const char* env_paths = std::getenv("HC_MODEL_PATHS")) {
        env_paths_str = env_paths;
    }
#endif
    if (!env_paths_str.empty()) {
        search_paths = split_paths(env_paths_str);
    }
    
    // Add command line paths
    for (int i = 1; i < argc; ++i) {
        search_paths.push_back(argv[i]);
    }
    
    if (search_paths.empty()) {
        std::cerr << "Usage: " << argv[0] << " [path1] [path2] ...\n";
        std::cerr << "Or set HC_MODEL_PATHS environment variable\n";
        return 1;
    }
    
    auto models = discover_models(search_paths);
    
    std::cerr << "\n=== Discovered Models ===\n";
    for (const auto& m : models) {
        std::cout << "Model: " << m.name << "\n";
        std::cout << "  Type: " << m.model_type << "\n";
        std::cout << "  Tokenizer: " << m.tokenizer_type << "\n";
        std::cout << "  Vocab size: " << m.vocab_size << "\n";
        std::cout << "  Hidden size: " << m.hidden_size << "\n";
        std::cout << "  Layers: " << m.num_layers << "\n";
        if (m.num_experts > 0) {
            std::cout << "  Experts: " << m.num_experts << " (MoE)\n";
        }
        if (m.is_multimodal) {
            std::cout << "  Multimodal: yes (vision+text)\n";
        }
        std::cout << "  Path: " << m.path << "\n";
        std::cout << "  Safetensors: " << (m.safetensor_path().empty() ? "NOT FOUND" : "found") << "\n";
        std::cout << "\n";
    }
    
    std::cout << "Total: " << models.size() << " models\n";
    
    return 0;
}
