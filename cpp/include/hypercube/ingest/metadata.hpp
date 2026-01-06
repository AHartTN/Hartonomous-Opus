// =============================================================================
// metadata.hpp - Model Metadata Ingestion (JSON configs, tokenizers, etc.)
// =============================================================================
// ALL metadata is content. Config files, tokenizer definitions, vocab files -
// these are semantic structures that belong in the substrate as queryable atoms,
// compositions, and relations.
// 
// Key insight: BPE merges ARE composition relations. Config hierarchies ARE
// the model's self-description. Vocab indices encode frequency/importance.
// =============================================================================

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>

#include "hypercube/types.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/ingest/context.hpp"

namespace fs = std::filesystem;

namespace hypercube {
namespace ingest {
namespace metadata {

// =============================================================================
// MetadataAtom - A configuration value as an ingestible unit
// =============================================================================

struct MetadataAtom {
    std::string path;       // JSON path: "vision_config.dim_embed.0"
    std::string key;        // Leaf key: "dim_embed"
    std::string value;      // String representation of value
    std::string value_type; // "string", "number", "boolean", "array", "object"
    int depth;              // Nesting depth in JSON tree
    
    // Parent path for hierarchy (e.g., "vision_config" for "vision_config.dim_embed")
    std::string parent_path() const {
        size_t last_dot = path.rfind('.');
        return (last_dot != std::string::npos) ? path.substr(0, last_dot) : "";
    }
};

// =============================================================================
// BPEMerge - A merge rule as an explicit composition relation
// =============================================================================
// BPE merge: token_a + token_b → merged_token
// This IS our cascading n-gram merkle-dag structure!

struct BPEMerge {
    std::string token_a;    // First component
    std::string token_b;    // Second component  
    std::string merged;     // Result (token_a + token_b concatenated)
    int priority;           // Merge order/priority (lower = more common)
    
    BPEMerge(const std::string& a, const std::string& b, int p)
        : token_a(a), token_b(b), merged(a + b), priority(p) {}
};

// =============================================================================
// SpecialToken - Added tokens with semantic roles
// =============================================================================

struct SpecialToken {
    std::string content;    // Token text: "<s>", "</s>", "<mask>", etc.
    int id;                 // Vocab index
    bool is_special;        // True for control tokens
    std::string role;       // Inferred role: "bos", "eos", "pad", "unk", "mask", "sep", "cls"
    
    // Infer role from content
    static std::string infer_role(const std::string& content) {
        if (content == "<s>" || content == "[CLS]" || content == "<bos>") return "bos";
        if (content == "</s>" || content == "[SEP]" || content == "<eos>") return "eos";
        if (content == "<pad>" || content == "[PAD]") return "pad";
        if (content == "<unk>" || content == "[UNK]") return "unk";
        if (content == "<mask>" || content == "[MASK]") return "mask";
        return "special";
    }
};

// =============================================================================
// VocabToken - A vocabulary entry with frequency/PMI metadata
// =============================================================================

struct VocabToken {
    std::string text;       // Token text (may include Ġ prefix for word-start)
    int index;              // Vocab index (lower = more frequent in BPE)
    bool is_word_start;     // Has Ġ/▁ prefix (word boundary)
    bool is_subword;        // Continuation token (##, no prefix)
    int byte_length;        // UTF-8 byte length
    int char_count;         // Unicode character count
    
    VocabToken(const std::string& t, int idx) 
        : text(t), index(idx) {
        // Detect word boundary markers
        is_word_start = (!t.empty() && (t[0] == '\xc4' && t.size() > 1 && t[1] == '\xa0'))  // Ġ (GPT-2 style)
                     || (!t.empty() && t[0] == '\xe2' && t.size() > 2 && t[1] == '\x96' && t[2] == '\x81');  // ▁ (SentencePiece)
        is_subword = (!t.empty() && t.size() >= 2 && t[0] == '#' && t[1] == '#');  // ## (BERT style)
        byte_length = static_cast<int>(t.size());
        
        // Count Unicode characters (not bytes)
        char_count = 0;
        for (size_t i = 0; i < t.size(); ) {
            unsigned char c = t[i];
            if ((c & 0x80) == 0) i += 1;
            else if ((c & 0xE0) == 0xC0) i += 2;
            else if ((c & 0xF0) == 0xE0) i += 3;
            else if ((c & 0xF8) == 0xF0) i += 4;
            else i += 1;  // Invalid UTF-8, skip byte
            char_count++;
        }
    }
};

// =============================================================================
// ModelMetadata - Complete parsed metadata for a model
// =============================================================================

struct ModelMetadata {
    std::string model_name;             // Directory name or explicit name
    std::string model_type;             // From config.json: "florence2", "bert", etc.
    
    // Config hierarchy as atoms
    std::vector<MetadataAtom> config_atoms;
    
    // Tokenizer structure
    std::vector<BPEMerge> bpe_merges;
    std::vector<SpecialToken> special_tokens;
    std::vector<VocabToken> vocab_tokens;
    
    // Key architectural parameters (extracted from config)
    int vocab_size = 0;
    int hidden_dim = 0;
    int num_layers = 0;
    int num_heads = 0;
    int max_position = 0;
    
    // File sources
    std::vector<std::string> source_files;
    
    void clear() {
        model_name.clear();
        model_type.clear();
        config_atoms.clear();
        bpe_merges.clear();
        special_tokens.clear();
        vocab_tokens.clear();
        vocab_size = hidden_dim = num_layers = num_heads = max_position = 0;
        source_files.clear();
    }
};

// =============================================================================
// JSON Parsing Helpers (minimal, no external deps)
// =============================================================================

namespace json {

// Skip whitespace
inline size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
        pos++;
    return pos;
}

// Find end of JSON string (handling escapes)
inline size_t find_string_end(const std::string& s, size_t start) {
    for (size_t i = start; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;  // Skip escaped char
            continue;
        }
        if (s[i] == '"') return i;
    }
    return std::string::npos;
}

// Unescape JSON string
inline std::string unescape(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char next = s[i + 1];
            switch (next) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case '/':  result += '/';  break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                default:   result += next; break;
            }
            ++i;
        } else {
            result += s[i];
        }
    }
    return result;
}

// Find matching bracket/brace (handles nesting and strings)
inline size_t find_matching(const std::string& s, size_t start, char open, char close) {
    if (start >= s.size() || s[start] != open) return std::string::npos;
    int depth = 1;
    bool in_string = false;
    for (size_t i = start + 1; i < s.size(); ++i) {
        if (s[i] == '\\' && in_string && i + 1 < s.size()) {
            ++i;
            continue;
        }
        if (s[i] == '"') in_string = !in_string;
        else if (!in_string) {
            if (s[i] == open) depth++;
            else if (s[i] == close) {
                depth--;
                if (depth == 0) return i;
            }
        }
    }
    return std::string::npos;
}

} // namespace json

// =============================================================================
// Config Parser - Extract atoms from JSON config
// =============================================================================

inline void parse_config_recursive(
    const std::string& content,
    size_t start,
    size_t end,
    const std::string& path_prefix,
    int depth,
    std::vector<MetadataAtom>& atoms
) {
    size_t pos = json::skip_ws(content, start);
    
    if (pos >= end) return;
    
    // Object: { key: value, ... }
    if (content[pos] == '{') {
        size_t obj_end = json::find_matching(content, pos, '{', '}');
        if (obj_end == std::string::npos) return;
        
        pos = json::skip_ws(content, pos + 1);
        
        while (pos < obj_end) {
            // Parse key
            if (content[pos] != '"') break;
            size_t key_end = json::find_string_end(content, pos + 1);
            if (key_end == std::string::npos) break;
            std::string key = json::unescape(content.substr(pos + 1, key_end - pos - 1));
            
            pos = json::skip_ws(content, key_end + 1);
            if (pos >= obj_end || content[pos] != ':') break;
            pos = json::skip_ws(content, pos + 1);
            
            std::string full_path = path_prefix.empty() ? key : (path_prefix + "." + key);
            
            // Parse value
            if (content[pos] == '"') {
                // String value
                size_t val_end = json::find_string_end(content, pos + 1);
                if (val_end != std::string::npos) {
                    std::string value = json::unescape(content.substr(pos + 1, val_end - pos - 1));
                    atoms.push_back({full_path, key, value, "string", depth});
                    pos = val_end + 1;
                }
            } else if (content[pos] == '{') {
                // Nested object
                size_t nested_end = json::find_matching(content, pos, '{', '}');
                if (nested_end != std::string::npos) {
                    atoms.push_back({full_path, key, "{...}", "object", depth});
                    parse_config_recursive(content, pos, nested_end + 1, full_path, depth + 1, atoms);
                    pos = nested_end + 1;
                }
            } else if (content[pos] == '[') {
                // Array
                size_t arr_end = json::find_matching(content, pos, '[', ']');
                if (arr_end != std::string::npos) {
                    std::string arr_content = content.substr(pos + 1, arr_end - pos - 1);
                    atoms.push_back({full_path, key, "[" + arr_content + "]", "array", depth});
                    
                    // Parse array elements as indexed children
                    size_t elem_pos = 0;
                    int elem_idx = 0;
                    while (elem_pos < arr_content.size()) {
                        elem_pos = json::skip_ws(arr_content, elem_pos);
                        if (elem_pos >= arr_content.size()) break;
                        
                        std::string elem_path = full_path + "." + std::to_string(elem_idx);
                        
                        if (arr_content[elem_pos] == '"') {
                            size_t elem_end = json::find_string_end(arr_content, elem_pos + 1);
                            if (elem_end != std::string::npos) {
                                std::string val = json::unescape(arr_content.substr(elem_pos + 1, elem_end - elem_pos - 1));
                                atoms.push_back({elem_path, std::to_string(elem_idx), val, "string", depth + 1});
                                elem_pos = elem_end + 1;
                            }
                        } else if (arr_content[elem_pos] == '{') {
                            size_t elem_end = json::find_matching(arr_content, elem_pos, '{', '}');
                            if (elem_end != std::string::npos) {
                                parse_config_recursive(arr_content, elem_pos, elem_end + 1, elem_path, depth + 1, atoms);
                                elem_pos = elem_end + 1;
                            }
                        } else {
                            // Number/bool/null
                            size_t comma = arr_content.find(',', elem_pos);
                            if (comma == std::string::npos) comma = arr_content.size();
                            std::string val = arr_content.substr(elem_pos, comma - elem_pos);
                            // Trim
                            size_t vstart = val.find_first_not_of(" \t\n\r");
                            size_t vend = val.find_last_not_of(" \t\n\r");
                            if (vstart != std::string::npos && vend != std::string::npos) {
                                val = val.substr(vstart, vend - vstart + 1);
                            }
                            if (!val.empty()) {
                                atoms.push_back({elem_path, std::to_string(elem_idx), val, "number", depth + 1});
                            }
                            elem_pos = comma + 1;
                        }
                        
                        // Skip comma
                        elem_pos = json::skip_ws(arr_content, elem_pos);
                        if (elem_pos < arr_content.size() && arr_content[elem_pos] == ',') elem_pos++;
                        elem_idx++;
                    }
                    
                    pos = arr_end + 1;
                }
            } else {
                // Number, boolean, null
                size_t comma = content.find(',', pos);
                size_t brace = content.find('}', pos);
                size_t val_end = std::min(comma, brace);
                if (val_end > obj_end) val_end = obj_end;
                
                std::string val = content.substr(pos, val_end - pos);
                // Trim
                size_t vstart = val.find_first_not_of(" \t\n\r");
                size_t vend = val.find_last_not_of(" \t\n\r");
                if (vstart != std::string::npos && vend != std::string::npos) {
                    val = val.substr(vstart, vend - vstart + 1);
                }
                
                std::string vtype = "number";
                if (val == "true" || val == "false") vtype = "boolean";
                else if (val == "null") vtype = "null";
                
                atoms.push_back({full_path, key, val, vtype, depth});
                pos = val_end;
            }
            
            // Skip comma
            pos = json::skip_ws(content, pos);
            if (pos < obj_end && content[pos] == ',') pos++;
            pos = json::skip_ws(content, pos);
        }
    }
}

// =============================================================================
// Parse config.json into metadata atoms
// =============================================================================

inline bool parse_config_json(const fs::path& path, ModelMetadata& meta) {
    std::ifstream file(path);
    if (!file) return false;
    
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    
    meta.source_files.push_back(path.string());
    
    // Extract key fields directly
    auto extract_int = [&](const std::string& key) -> int {
        std::string search = "\"" + key + "\":";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return 0;
        pos = json::skip_ws(content, pos + search.size());
        size_t end = pos;
        while (end < content.size() && (isdigit(content[end]) || content[end] == '-')) end++;
        if (end > pos) return std::stoi(content.substr(pos, end - pos));
        return 0;
    };
    
    auto extract_string = [&](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\":";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return "";
        pos = json::skip_ws(content, pos + search.size());
        if (pos < content.size() && content[pos] == '"') {
            size_t end = json::find_string_end(content, pos + 1);
            if (end != std::string::npos) {
                return json::unescape(content.substr(pos + 1, end - pos - 1));
            }
        }
        return "";
    };
    
    meta.model_type = extract_string("model_type");
    meta.vocab_size = extract_int("vocab_size");
    meta.hidden_dim = extract_int("d_model");
    if (meta.hidden_dim == 0) meta.hidden_dim = extract_int("hidden_size");
    if (meta.hidden_dim == 0) meta.hidden_dim = extract_int("projection_dim");
    meta.num_layers = extract_int("num_hidden_layers");
    if (meta.num_layers == 0) meta.num_layers = extract_int("encoder_layers");
    meta.num_heads = extract_int("num_attention_heads");
    if (meta.num_heads == 0) meta.num_heads = extract_int("encoder_attention_heads");
    meta.max_position = extract_int("max_position_embeddings");
    
    // Parse full config hierarchy
    parse_config_recursive(content, 0, content.size(), "", 0, meta.config_atoms);
    
    std::cerr << "[CONFIG] Parsed " << meta.config_atoms.size() << " config atoms, "
              << "model_type=" << meta.model_type << ", vocab_size=" << meta.vocab_size << "\n";
    
    return true;
}

// =============================================================================
// Parse tokenizer.json - Extract BPE merges, vocab, and special tokens
// =============================================================================

inline bool parse_tokenizer_json(const fs::path& path, ModelMetadata& meta) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "[TOKENIZER] Cannot open: " << path << "\n";
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    
    meta.source_files.push_back(path.string());
    
    // Parse added_tokens
    size_t pos = content.find("\"added_tokens\"");
    if (pos != std::string::npos) {
        pos = content.find("[", pos);
        if (pos != std::string::npos) {
            size_t end = json::find_matching(content, pos, '[', ']');
            if (end != std::string::npos) {
                std::string arr = content.substr(pos + 1, end - pos - 1);
                
                // Parse each {id:..., content:..., special:...}
                size_t obj_pos = 0;
                while ((obj_pos = arr.find("{", obj_pos)) != std::string::npos) {
                    size_t obj_end = json::find_matching(arr, obj_pos, '{', '}');
                    if (obj_end == std::string::npos) break;
                    
                    std::string obj = arr.substr(obj_pos, obj_end - obj_pos + 1);
                    
                    // Extract fields
                    SpecialToken tok;
                    
                    // content
                    size_t cpos = obj.find("\"content\"");
                    if (cpos != std::string::npos) {
                        cpos = obj.find("\"", cpos + 9);
                        if (cpos != std::string::npos) {
                            size_t cend = json::find_string_end(obj, cpos + 1);
                            if (cend != std::string::npos) {
                                tok.content = json::unescape(obj.substr(cpos + 1, cend - cpos - 1));
                            }
                        }
                    }
                    
                    // id
                    size_t ipos = obj.find("\"id\"");
                    if (ipos != std::string::npos) {
                        ipos = obj.find(":", ipos);
                        if (ipos != std::string::npos) {
                            ipos = json::skip_ws(obj, ipos + 1);
                            size_t iend = ipos;
                            while (iend < obj.size() && (isdigit(obj[iend]) || obj[iend] == '-')) iend++;
                            if (iend > ipos) tok.id = std::stoi(obj.substr(ipos, iend - ipos));
                        }
                    }
                    
                    // special
                    tok.is_special = (obj.find("\"special\":true") != std::string::npos);
                    tok.role = SpecialToken::infer_role(tok.content);
                    
                    if (!tok.content.empty()) {
                        meta.special_tokens.push_back(tok);
                    }
                    
                    obj_pos = obj_end + 1;
                }
            }
        }
    }
    
    // Parse merges (BPE)
    pos = content.find("\"merges\"");
    if (pos != std::string::npos) {
        pos = content.find("[", pos);
        if (pos != std::string::npos) {
            size_t end = json::find_matching(content, pos, '[', ']');
            if (end != std::string::npos) {
                std::string arr = content.substr(pos + 1, end - pos - 1);
                
                int priority = 0;
                size_t i = 0;
                while (i < arr.size()) {
                    size_t q1 = arr.find("\"", i);
                    if (q1 == std::string::npos) break;
                    size_t q2 = json::find_string_end(arr, q1 + 1);
                    if (q2 == std::string::npos) break;
                    
                    std::string merge = json::unescape(arr.substr(q1 + 1, q2 - q1 - 1));
                    size_t sp = merge.find(" ");
                    if (sp != std::string::npos) {
                        meta.bpe_merges.emplace_back(
                            merge.substr(0, sp),
                            merge.substr(sp + 1),
                            priority++
                        );
                    }
                    i = q2 + 1;
                }
            }
        }
    }
    
    // Parse vocab (inside model object for HuggingFace format)
    pos = content.find("\"vocab\"");
    if (pos != std::string::npos) {
        pos = content.find("{", pos);
        if (pos != std::string::npos) {
            size_t end = pos + 1;
            int depth = 1;
            bool in_string = false;
            while (depth > 0 && end < content.size()) {
                if (content[end] == '\\' && in_string && end + 1 < content.size()) {
                    end += 2;
                    continue;
                }
                if (content[end] == '"') in_string = !in_string;
                else if (!in_string) {
                    if (content[end] == '{') depth++;
                    else if (content[end] == '}') depth--;
                }
                end++;
            }
            
            std::string vocab_str = content.substr(pos + 1, end - pos - 2);
            size_t i = 0;
            while (i < vocab_str.size()) {
                size_t q1 = vocab_str.find("\"", i);
                if (q1 == std::string::npos) break;
                size_t q2 = json::find_string_end(vocab_str, q1 + 1);
                if (q2 == std::string::npos) break;
                
                std::string token = json::unescape(vocab_str.substr(q1 + 1, q2 - q1 - 1));
                
                size_t colon = vocab_str.find(":", q2);
                if (colon == std::string::npos) break;
                
                size_t comma = vocab_str.find(",", colon);
                if (comma == std::string::npos) comma = vocab_str.size();
                
                try {
                    std::string idx_str = vocab_str.substr(colon + 1, comma - colon - 1);
                    size_t start = idx_str.find_first_not_of(" \t\n\r");
                    size_t last = idx_str.find_last_not_of(" \t\n\r");
                    if (start != std::string::npos && last != std::string::npos) {
                        idx_str = idx_str.substr(start, last - start + 1);
                    }
                    int idx = std::stoi(idx_str);
                    meta.vocab_tokens.emplace_back(token, idx);
                } catch (...) {
                    // Skip malformed entries
                }
                
                i = comma + 1;
            }
        }
    }
    
    std::cerr << "[TOKENIZER] Loaded " << meta.special_tokens.size() << " special tokens, "
              << meta.bpe_merges.size() << " BPE merges, "
              << meta.vocab_tokens.size() << " vocab entries\n";
    
    return true;
}

// =============================================================================
// Parse all metadata for a model directory
// =============================================================================

inline bool parse_model_metadata(const fs::path& model_dir, ModelMetadata& meta) {
    meta.clear();
    meta.model_name = model_dir.filename().string();
    
    std::cerr << "\n=== Parsing Model Metadata: " << meta.model_name << " ===\n";
    
    // Config files
    if (fs::exists(model_dir / "config.json")) {
        parse_config_json(model_dir / "config.json", meta);
    }
    
    // Tokenizer
    if (fs::exists(model_dir / "tokenizer.json")) {
        parse_tokenizer_json(model_dir / "tokenizer.json", meta);
    }
    
    // Additional configs
    if (fs::exists(model_dir / "tokenizer_config.json")) {
        // Could parse for additional tokenizer settings
        meta.source_files.push_back((model_dir / "tokenizer_config.json").string());
    }
    if (fs::exists(model_dir / "preprocessor_config.json")) {
        meta.source_files.push_back((model_dir / "preprocessor_config.json").string());
    }
    
    std::cerr << "[METADATA] Total: " << meta.config_atoms.size() << " config atoms, "
              << meta.vocab_tokens.size() << " vocab tokens, "
              << meta.bpe_merges.size() << " merges, "
              << meta.special_tokens.size() << " special tokens\n";
    
    return true;
}

} // namespace metadata
} // namespace ingest
} // namespace hypercube
