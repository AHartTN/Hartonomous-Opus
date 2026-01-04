/**
 * Model Vocab Extractor - Universal Vocabulary Ingestion
 * 
 * Extracts vocabulary from ANY HuggingFace model and converts it to plain text
 * for ingestion via the universal PMI ingester.
 * 
 * The key insight: vocab tokens are just TEXT. 
 * - WordPiece "##ing" → text "##ing"
 * - BPE "Ġhello" → text "Ġhello" (or decode to " hello")
 * - Any token → just the string
 * 
 * The substrate doesn't care about the encoding - it will discover patterns.
 */

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <set>
#include <unordered_map>

namespace fs = std::filesystem;

namespace {

// Extract vocab from tokenizer.json (works for BPE, WordPiece, Unigram)
std::vector<std::pair<std::string, size_t>> extract_vocab_from_tokenizer_json(const std::string& path) {
    std::vector<std::pair<std::string, size_t>> vocab;
    
    std::ifstream file(path);
    if (!file) return vocab;
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    // Find the vocab section: "vocab": { "token": id, ... }
    size_t vocab_start = json.find("\"vocab\"");
    if (vocab_start == std::string::npos) return vocab;
    
    size_t brace_start = json.find('{', vocab_start);
    if (brace_start == std::string::npos) return vocab;
    
    // Parse vocab entries - simple state machine
    size_t pos = brace_start + 1;
    int brace_depth = 1;
    bool in_string = false;
    bool in_key = false;
    std::string current_token;
    std::string current_id;
    
    while (pos < json.size() && brace_depth > 0) {
        char c = json[pos];
        
        if (c == '\\' && pos + 1 < json.size()) {
            // Handle escape sequences
            if (in_string) {
                if (in_key) current_token += json.substr(pos, 2);
                else current_id += json.substr(pos, 2);
            }
            pos += 2;
            continue;
        }
        
        if (c == '"') {
            in_string = !in_string;
            if (in_string && !in_key && current_token.empty()) {
                // Starting a new key
                in_key = true;
            } else if (!in_string && in_key) {
                // Finished reading key
                in_key = false;
            }
        } else if (in_string) {
            if (in_key) current_token += c;
        } else if (c == ':') {
            // About to read value
        } else if (std::isdigit(c) || c == '-') {
            current_id += c;
        } else if (c == ',' || c == '}') {
            // End of entry
            if (!current_token.empty() && !current_id.empty()) {
                try {
                    size_t id = std::stoull(current_id);
                    vocab.emplace_back(current_token, id);
                } catch (...) {}
            }
            current_token.clear();
            current_id.clear();
            
            if (c == '}') brace_depth--;
        } else if (c == '{') {
            brace_depth++;
        }
        
        pos++;
    }
    
    return vocab;
}

// Extract vocab from vocab.txt (simple line-based format, used by BERT)
std::vector<std::pair<std::string, size_t>> extract_vocab_from_txt(const std::string& path) {
    std::vector<std::pair<std::string, size_t>> vocab;
    
    std::ifstream file(path);
    if (!file) return vocab;
    
    std::string line;
    size_t id = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            vocab.emplace_back(line, id);
        }
        id++;
    }
    
    return vocab;
}

// Decode BPE byte-level tokens (Ġ = space, etc.)
std::string decode_bpe_token(const std::string& token) {
    std::string result;
    for (size_t i = 0; i < token.size(); ) {
        // Check for Ġ (U+0120, encoded as C4 A0 in UTF-8)
        if (i + 1 < token.size() && 
            static_cast<unsigned char>(token[i]) == 0xC4 && 
            static_cast<unsigned char>(token[i+1]) == 0xA0) {
            result += ' ';
            i += 2;
        }
        // Check for Ċ (U+010A, encoded as C4 8A in UTF-8) - newline
        else if (i + 1 < token.size() && 
                 static_cast<unsigned char>(token[i]) == 0xC4 && 
                 static_cast<unsigned char>(token[i+1]) == 0x8A) {
            result += '\n';
            i += 2;
        }
        else {
            result += token[i];
            i++;
        }
    }
    return result;
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <model_path>\n"
              << "\n"
              << "Extracts vocabulary from a HuggingFace model and outputs as text.\n"
              << "Output can be piped to ingest.exe for composition creation.\n"
              << "\n"
              << "Options:\n"
              << "  --decode         Decode BPE byte-level tokens to readable form\n"
              << "  --one-per-line   Output one token per line (default)\n"
              << "  --as-text        Output as continuous text (space-separated)\n"
              << "  --stats          Print vocabulary statistics to stderr\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog << " path/to/model > vocab.txt\n"
              << "  " << prog << " path/to/model | ingest.exe -\n";
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    bool decode_bpe = false;
    bool one_per_line = true;
    bool print_stats = false;
    std::string model_path;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--decode") decode_bpe = true;
        else if (arg == "--one-per-line") one_per_line = true;
        else if (arg == "--as-text") one_per_line = false;
        else if (arg == "--stats") print_stats = true;
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg[0] != '-') model_path = arg;
    }
    
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    fs::path path(model_path);
    if (!fs::exists(path)) {
        std::cerr << "Error: Path not found: " << model_path << "\n";
        return 1;
    }
    
    // Find tokenizer files
    std::string tokenizer_json = (path / "tokenizer.json").string();
    std::string vocab_txt = (path / "vocab.txt").string();
    
    // Also check snapshot directories
    if (!fs::exists(tokenizer_json) && !fs::exists(vocab_txt)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.path().filename() == "tokenizer.json" && 
                entry.path().string().find(".cache") == std::string::npos) {
                tokenizer_json = entry.path().string();
            }
            if (entry.path().filename() == "vocab.txt" && 
                entry.path().string().find(".cache") == std::string::npos) {
                vocab_txt = entry.path().string();
            }
        }
    }
    
    std::vector<std::pair<std::string, size_t>> vocab;
    std::string source;
    
    // Try tokenizer.json first (more complete for modern models)
    if (fs::exists(tokenizer_json)) {
        vocab = extract_vocab_from_tokenizer_json(tokenizer_json);
        source = tokenizer_json;
    }
    // Fall back to vocab.txt
    if (vocab.empty() && fs::exists(vocab_txt)) {
        vocab = extract_vocab_from_txt(vocab_txt);
        source = vocab_txt;
    }
    
    if (vocab.empty()) {
        std::cerr << "Error: No vocabulary found in " << model_path << "\n";
        std::cerr << "Checked: tokenizer.json, vocab.txt\n";
        return 1;
    }
    
    // Sort by token ID
    std::sort(vocab.begin(), vocab.end(), 
        [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Stats
    if (print_stats) {
        std::set<std::string> unique_tokens;
        size_t special_count = 0;
        size_t subword_count = 0;
        size_t byte_token_count = 0;
        
        for (const auto& [token, id] : vocab) {
            unique_tokens.insert(token);
            if (token.find('<') != std::string::npos || 
                token.find('[') != std::string::npos) {
                special_count++;
            }
            if (token.size() >= 2 && token[0] == '#' && token[1] == '#') {
                subword_count++;  // WordPiece continuation
            }
            // Check for BPE byte tokens (single high-byte chars)
            if (token.size() == 1 && static_cast<unsigned char>(token[0]) >= 0x80) {
                byte_token_count++;
            }
        }
        
        std::cerr << "=== Vocabulary Stats ===\n";
        std::cerr << "Source: " << source << "\n";
        std::cerr << "Total tokens: " << vocab.size() << "\n";
        std::cerr << "Unique tokens: " << unique_tokens.size() << "\n";
        std::cerr << "Special tokens: " << special_count << "\n";
        std::cerr << "Subword tokens (##): " << subword_count << "\n";
        std::cerr << "Byte tokens: " << byte_token_count << "\n\n";
    }
    
    // Output tokens
    for (const auto& [token, id] : vocab) {
        std::string output = decode_bpe ? decode_bpe_token(token) : token;
        
        if (one_per_line) {
            std::cout << output << "\n";
        } else {
            std::cout << output << " ";
        }
    }
    
    if (!one_per_line) std::cout << "\n";
    
    return 0;
}
