/**
 * Model Vocabulary Ingestion Tool
 * 
 * Ingests AI model vocabularies (vocab.txt, tokenizer.json) into the hypercube substrate.
 * Each token becomes a trajectory (LINESTRINGZM) through the 4D atom space.
 * 
 * Usage:
 *   ingest_model -d hypercube --vocab /path/to/vocab.txt --name "all-MiniLM-L6-v2"
 *   ingest_model -d hypercube --tokenizer /path/to/tokenizer.json --name "llama3"
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <libpq-fe.h>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

// For JSON parsing (nlohmann/json or simdjson would be better, using simple parsing here)
#include <regex>

using namespace hypercube;

// Configuration
static constexpr int BATCH_SIZE = 1000;
static constexpr int NUM_CONNECTIONS = 4;

struct Token {
    int32_t index;
    std::string text;
    bool is_special;
    bool is_subword;
    Blake3Hash hash;
    double centroid_x, centroid_y, centroid_z, centroid_m;
    int64_t hilbert_lo, hilbert_hi;
};

// Parse vocab.txt format (one token per line)
std::vector<Token> parse_vocab_txt(const std::string& filepath) {
    std::vector<Token> tokens;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filepath << std::endl;
        return tokens;
    }
    
    std::string line;
    int32_t idx = 0;
    
    while (std::getline(file, line)) {
        Token tok;
        tok.index = idx++;
        tok.text = line;
        tok.is_special = (line.front() == '[' && line.back() == ']') ||
                         (line.front() == '<' && line.back() == '>');
        tok.is_subword = line.rfind("##", 0) == 0;
        tokens.push_back(tok);
    }
    
    return tokens;
}

// Parse tokenizer.json format (extract vocab from model.vocab object)
std::vector<Token> parse_tokenizer_json(const std::string& filepath) {
    std::vector<Token> tokens;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filepath << std::endl;
        return tokens;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    // Simple regex-based extraction of vocab entries
    // Pattern: "token":index or "token": index
    std::regex vocab_pattern(R"("([^"\\]|\\.)*"\s*:\s*(\d+))");
    
    // Find vocab section
    size_t vocab_start = content.find("\"vocab\"");
    if (vocab_start == std::string::npos) {
        std::cerr << "Error: Cannot find vocab section in tokenizer.json" << std::endl;
        return tokens;
    }
    
    // Find opening brace
    size_t brace_start = content.find('{', vocab_start);
    if (brace_start == std::string::npos) {
        return tokens;
    }
    
    // Find matching closing brace
    int brace_count = 1;
    size_t brace_end = brace_start + 1;
    while (brace_end < content.size() && brace_count > 0) {
        if (content[brace_end] == '{') brace_count++;
        else if (content[brace_end] == '}') brace_count--;
        brace_end++;
    }
    
    std::string vocab_section = content.substr(brace_start, brace_end - brace_start);
    
    // Extract token:index pairs
    std::unordered_map<int32_t, std::string> idx_to_token;
    int32_t max_idx = 0;
    
    std::sregex_iterator iter(vocab_section.begin(), vocab_section.end(), vocab_pattern);
    std::sregex_iterator end;
    
    while (iter != end) {
        std::string match_str = (*iter)[0].str();
        
        // Find the colon separator
        size_t colon = match_str.rfind(':');
        if (colon != std::string::npos) {
            std::string token_quoted = match_str.substr(0, colon);
            std::string idx_str = match_str.substr(colon + 1);
            
            // Remove quotes and whitespace
            if (token_quoted.size() >= 2) {
                std::string token = token_quoted.substr(1, token_quoted.size() - 2);
                
                // Unescape common escapes
                size_t pos = 0;
                while ((pos = token.find("\\\"", pos)) != std::string::npos) {
                    token.replace(pos, 2, "\"");
                }
                pos = 0;
                while ((pos = token.find("\\\\", pos)) != std::string::npos) {
                    token.replace(pos, 2, "\\");
                }
                
                int32_t idx = std::stoi(idx_str);
                idx_to_token[idx] = token;
                if (idx > max_idx) max_idx = idx;
            }
        }
        ++iter;
    }
    
    // Convert to ordered vector
    tokens.resize(max_idx + 1);
    for (const auto& [idx, text] : idx_to_token) {
        tokens[idx].index = idx;
        tokens[idx].text = text;
        tokens[idx].is_special = (text.front() == '[' && text.back() == ']') ||
                                  (text.front() == '<' && text.back() == '>');
        tokens[idx].is_subword = text.rfind("##", 0) == 0;
    }
    
    return tokens;
}

// Compute token trajectory centroid and Hilbert index
void compute_token_geometry(Token& tok, PGconn* conn) {
    // Strip subword prefix for atom lookup
    std::string clean_text = tok.text;
    if (tok.is_subword && clean_text.size() >= 2) {
        clean_text = clean_text.substr(2);
    }
    
    if (clean_text.empty() || tok.is_special) {
        // Special tokens go to center
        tok.centroid_x = 0.5;
        tok.centroid_y = 0.5;
        tok.centroid_z = 0.5;
        tok.centroid_m = 0.5;
        
        Point4D center(UINT32_MAX / 2, UINT32_MAX / 2, UINT32_MAX / 2, UINT32_MAX / 2);
        HilbertIndex hilbert = HilbertCurve::coords_to_index(center);
        tok.hilbert_lo = static_cast<int64_t>(hilbert.lo);
        tok.hilbert_hi = static_cast<int64_t>(hilbert.hi);
        return;
    }
    
    // Extract codepoints and compute centroid
    double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
    int count = 0;
    
    // UTF-8 decode and process each character
    const char* ptr = clean_text.c_str();
    const char* end = ptr + clean_text.size();
    
    while (ptr < end) {
        uint32_t cp;
        unsigned char c = static_cast<unsigned char>(*ptr);
        
        if (c < 0x80) {
            cp = c;
            ptr += 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = (c & 0x1F) << 6;
            cp |= (static_cast<unsigned char>(ptr[1]) & 0x3F);
            ptr += 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = (c & 0x0F) << 12;
            cp |= (static_cast<unsigned char>(ptr[1]) & 0x3F) << 6;
            cp |= (static_cast<unsigned char>(ptr[2]) & 0x3F);
            ptr += 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = (c & 0x07) << 18;
            cp |= (static_cast<unsigned char>(ptr[1]) & 0x3F) << 12;
            cp |= (static_cast<unsigned char>(ptr[2]) & 0x3F) << 6;
            cp |= (static_cast<unsigned char>(ptr[3]) & 0x3F);
            ptr += 4;
        } else {
            ptr += 1;
            continue;
        }
        
        // Map codepoint to 4D coordinates
        Point4D coords = CoordinateMapper::map_codepoint(cp);
        
        // Normalize to [0, 1]
        constexpr double COORD_SCALE = 1.0 / 4294967295.0;
        sum_x += static_cast<double>(coords.x) * COORD_SCALE;
        sum_y += static_cast<double>(coords.y) * COORD_SCALE;
        sum_z += static_cast<double>(coords.z) * COORD_SCALE;
        sum_m += static_cast<double>(coords.m) * COORD_SCALE;
        count++;
    }
    
    if (count > 0) {
        tok.centroid_x = sum_x / count;
        tok.centroid_y = sum_y / count;
        tok.centroid_z = sum_z / count;
        tok.centroid_m = sum_m / count;
        
        // Compute Hilbert index from centroid
        Point4D centroid(
            static_cast<Coord32>(tok.centroid_x * UINT32_MAX),
            static_cast<Coord32>(tok.centroid_y * UINT32_MAX),
            static_cast<Coord32>(tok.centroid_z * UINT32_MAX),
            static_cast<Coord32>(tok.centroid_m * UINT32_MAX)
        );
        HilbertIndex hilbert = HilbertCurve::coords_to_index(centroid);
        tok.hilbert_lo = static_cast<int64_t>(hilbert.lo);
        tok.hilbert_hi = static_cast<int64_t>(hilbert.hi);
    } else {
        tok.centroid_x = tok.centroid_y = tok.centroid_z = tok.centroid_m = 0.5;
        tok.hilbert_lo = tok.hilbert_hi = 0;
    }
}

// Generate token hash (model_id + index + text)
void compute_token_hash(Token& tok, const Blake3Hash& model_id) {
    Blake3Hasher hasher;
    hasher.update(model_id.bytes, 32);
    
    // Add index as 4 bytes big-endian
    uint8_t idx_bytes[4];
    idx_bytes[0] = (tok.index >> 24) & 0xFF;
    idx_bytes[1] = (tok.index >> 16) & 0xFF;
    idx_bytes[2] = (tok.index >> 8) & 0xFF;
    idx_bytes[3] = tok.index & 0xFF;
    hasher.update(idx_bytes, 4);
    
    hasher.update(reinterpret_cast<const uint8_t*>(tok.text.data()), tok.text.size());
    tok.hash = hasher.finalize();
}

// Register model and return its ID
Blake3Hash register_model(PGconn* conn, const std::string& name, const std::string& type, int vocab_size, const std::string& source_path) {
    std::string query = "SELECT hypercube_register_model($1, $2, $3, NULL, $4)";
    
    const char* params[4];
    params[0] = name.c_str();
    params[1] = type.c_str();
    
    std::string vocab_str = std::to_string(vocab_size);
    params[2] = vocab_str.c_str();
    params[3] = source_path.c_str();
    
    PGresult* res = PQexecParams(conn, query.c_str(), 4, nullptr, params, nullptr, nullptr, 1);
    
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to register model: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return Blake3Hash();
    }
    
    Blake3Hash model_id;
    if (PQntuples(res) > 0 && PQgetlength(res, 0, 0) == 32) {
        std::memcpy(model_id.bytes, PQgetvalue(res, 0, 0), 32);
    }
    
    PQclear(res);
    return model_id;
}

// COPY tokens to database
bool copy_tokens(PGconn* conn, const Blake3Hash& model_id, const std::vector<Token>& tokens, size_t start, size_t end) {
    std::string copy_cmd = 
        "COPY vocab_token (id, model_id, token_index, token_text, is_special, is_subword, "
        "coords, hilbert_lo, hilbert_hi) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')";
    
    PGresult* res = PQexec(conn, copy_cmd.c_str());
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    // Build EWKB for POINTZM
    static const char hex_chars[] = "0123456789abcdef";
    
    auto double_to_hex = [&](double val, char* out) {
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = (bits >> (i * 8)) & 0xFF;
            out[i * 2] = hex_chars[byte >> 4];
            out[i * 2 + 1] = hex_chars[byte & 0x0F];
        }
    };
    
    std::string batch;
    batch.reserve(1 << 18);  // 256KB
    
    char ewkb[75];
    std::memcpy(ewkb, "01010000c0", 10);  // POINTZM little-endian, SRID=0
    ewkb[74] = '\0';
    
    char num_buf[32];
    std::string model_hex = model_id.to_hex();
    
    for (size_t i = start; i < end && i < tokens.size(); ++i) {
        const Token& tok = tokens[i];
        
        if (tok.text.empty()) continue;
        
        // Build EWKB geometry
        double_to_hex(tok.centroid_x, ewkb + 10);
        double_to_hex(tok.centroid_y, ewkb + 26);
        double_to_hex(tok.centroid_z, ewkb + 42);
        double_to_hex(tok.centroid_m, ewkb + 58);
        
        // id (bytea)
        batch += "\\\\x";
        batch += tok.hash.to_hex();
        batch += '\t';
        
        // model_id (bytea)
        batch += "\\\\x";
        batch += model_hex;
        batch += '\t';
        
        // token_index
        snprintf(num_buf, sizeof(num_buf), "%d", tok.index);
        batch += num_buf;
        batch += '\t';
        
        // token_text (escape tabs, newlines, backslashes)
        for (char c : tok.text) {
            if (c == '\t') batch += "\\t";
            else if (c == '\n') batch += "\\n";
            else if (c == '\r') batch += "\\r";
            else if (c == '\\') batch += "\\\\";
            else batch += c;
        }
        batch += '\t';
        
        // is_special
        batch += tok.is_special ? "t" : "f";
        batch += '\t';
        
        // is_subword
        batch += tok.is_subword ? "t" : "f";
        batch += '\t';
        
        // coords (EWKB hex)
        batch += ewkb;
        batch += '\t';
        
        // hilbert_lo
        snprintf(num_buf, sizeof(num_buf), "%ld", tok.hilbert_lo);
        batch += num_buf;
        batch += '\t';
        
        // hilbert_hi
        snprintf(num_buf, sizeof(num_buf), "%ld", tok.hilbert_hi);
        batch += num_buf;
        batch += '\n';
        
        // Send when buffer full
        if (batch.size() > (1 << 17)) {
            if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
                std::cerr << "COPY data failed\n";
                PQputCopyEnd(conn, "error");
                return false;
            }
            batch.clear();
        }
    }
    
    // Send remaining
    if (!batch.empty()) {
        if (PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size())) != 1) {
            std::cerr << "COPY final failed\n";
            PQputCopyEnd(conn, "error");
            return false;
        }
    }
    
    if (PQputCopyEnd(conn, nullptr) != 1) {
        std::cerr << "COPY end failed: " << PQerrorMessage(conn) << std::endl;
        return false;
    }
    
    res = PQgetResult(conn);
    bool success = (PQresultStatus(res) == PGRES_COMMAND_OK);
    if (!success) {
        std::cerr << "COPY result: " << PQerrorMessage(conn) << std::endl;
    }
    PQclear(res);
    
    return success;
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -d, --dbname NAME      Database name (default: hypercube)\n"
              << "  -h, --host HOST        Database host\n"
              << "  -p, --port PORT        Database port\n"
              << "  -U, --user USER        Database user\n"
              << "  --vocab PATH           Path to vocab.txt file\n"
              << "  --tokenizer PATH       Path to tokenizer.json file\n"
              << "  --name NAME            Model name (required)\n"
              << "  --type TYPE            Model type (default: auto-detected)\n"
              << "  --help                 Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string dbname = "hypercube";
    std::string host = "";
    std::string port = "";
    std::string user = "";
    std::string vocab_path = "";
    std::string tokenizer_path = "";
    std::string model_name = "";
    std::string model_type = "";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--dbname") && i + 1 < argc) {
            dbname = argv[++i];
        } else if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = argv[++i];
        } else if ((arg == "-U" || arg == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (arg == "--name" && i + 1 < argc) {
            model_name = argv[++i];
        } else if (arg == "--type" && i + 1 < argc) {
            model_type = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (model_name.empty()) {
        std::cerr << "Error: --name is required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    if (vocab_path.empty() && tokenizer_path.empty()) {
        std::cerr << "Error: Either --vocab or --tokenizer is required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    std::string conninfo = "dbname=" + dbname;
    if (!host.empty()) conninfo += " host=" + host;
    if (!port.empty()) conninfo += " port=" + port;
    if (!user.empty()) conninfo += " user=" + user;
    
    std::cerr << "=== Model Vocabulary Ingestion ===\n";
    std::cerr << "Connection: " << conninfo << "\n";
    std::cerr << "Model: " << model_name << "\n\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Parse vocabulary
    std::cerr << "[1/4] Parsing vocabulary...\n";
    std::vector<Token> tokens;
    std::string source_path;
    
    if (!vocab_path.empty()) {
        tokens = parse_vocab_txt(vocab_path);
        source_path = vocab_path;
        if (model_type.empty()) model_type = "wordpiece";
    } else {
        tokens = parse_tokenizer_json(tokenizer_path);
        source_path = tokenizer_path;
        if (model_type.empty()) model_type = "bpe";
    }
    
    if (tokens.empty()) {
        std::cerr << "Error: No tokens parsed\n";
        return 1;
    }
    
    std::cerr << "      Parsed " << tokens.size() << " tokens\n";
    
    auto parse_time = std::chrono::high_resolution_clock::now();
    
    // Connect to database
    std::cerr << "[2/4] Connecting to database...\n";
    PGconn* conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return 1;
    }
    
    // Register model
    std::cerr << "[3/4] Registering model and computing geometry...\n";
    Blake3Hash model_id = register_model(conn, model_name, model_type, static_cast<int>(tokens.size()), source_path);
    
    if (model_id.bytes[0] == 0 && model_id.bytes[1] == 0) {
        std::cerr << "Error: Failed to register model\n";
        PQfinish(conn);
        return 1;
    }
    
    // Compute geometry for all tokens
    for (auto& tok : tokens) {
        compute_token_hash(tok, model_id);
        compute_token_geometry(tok, conn);
    }
    
    auto geom_time = std::chrono::high_resolution_clock::now();
    auto geom_ms = std::chrono::duration_cast<std::chrono::milliseconds>(geom_time - parse_time).count();
    std::cerr << "      Computed geometry in " << geom_ms << " ms\n";
    
    // Insert tokens
    std::cerr << "[4/4] Inserting tokens...\n";
    
    // Drop indexes for bulk insert
    PQexec(conn, "DROP INDEX IF EXISTS idx_vocab_token_model");
    PQexec(conn, "DROP INDEX IF EXISTS idx_vocab_token_text");
    PQexec(conn, "DROP INDEX IF EXISTS idx_vocab_token_coords");
    PQexec(conn, "DROP INDEX IF EXISTS idx_vocab_token_hilbert");
    
    bool success = copy_tokens(conn, model_id, tokens, 0, tokens.size());
    
    if (!success) {
        std::cerr << "Error: Failed to insert tokens\n";
        PQfinish(conn);
        return 1;
    }
    
    // Rebuild indexes
    std::cerr << "      Rebuilding indexes...\n";
    PQexec(conn, "CREATE INDEX IF NOT EXISTS idx_vocab_token_model ON vocab_token(model_id)");
    PQexec(conn, "CREATE INDEX IF NOT EXISTS idx_vocab_token_text ON vocab_token(token_text)");
    PQexec(conn, "CREATE INDEX IF NOT EXISTS idx_vocab_token_coords ON vocab_token USING GIST(coords)");
    PQexec(conn, "CREATE INDEX IF NOT EXISTS idx_vocab_token_hilbert ON vocab_token(hilbert_hi, hilbert_lo)");
    PQexec(conn, "ANALYZE vocab_token");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cerr << "\n=== Complete ===\n";
    std::cerr << "Tokens ingested: " << tokens.size() << "\n";
    std::cerr << "Total time: " << total_ms << " ms (" << (total_ms / 1000.0) << " s)\n";
    std::cerr << "Rate: " << (tokens.size() * 1000 / std::max(total_ms, 1L)) << " tokens/sec\n";
    
    PQfinish(conn);
    return 0;
}
