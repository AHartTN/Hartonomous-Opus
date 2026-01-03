// Safetensor Package Ingester
// Ingests complete HuggingFace model packages:
// - vocab.txt -> token compositions  
// - tokenizer.json -> BPE merge rules -> semantic edges
// - model.safetensors -> weight matrix -> sparse semantic edges
//
// The embedding vectors are IGNORED - we only extract relationships.
// Weights become edge strengths (M coordinate).

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <libpq-fe.h>
#include "hypercube/types.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"

namespace fs = std::filesystem;
using namespace hypercube;

struct IngestConfig {
    std::string conninfo;
    float weight_threshold = 0.1f;
    int top_k_per_token = 100;
    bool verbose = false;
};

struct TokenInfo {
    std::string text;
    Blake3Hash hash;
    int32_t coord_x, coord_y, coord_z, coord_m;
    int64_t hilbert_lo, hilbert_hi;
};

struct SemanticEdge {
    Blake3Hash from_token;
    Blake3Hash to_token;
    float weight;
};

static std::vector<TokenInfo> g_vocab;
static std::unordered_map<std::string, size_t> g_token_to_idx;

// Forward declarations
std::vector<SemanticEdge> extract_safetensor_weights(const fs::path& safetensor_path, const IngestConfig& config);

bool parse_vocab(const fs::path& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file) {
        std::cerr << "Cannot open vocab: " << vocab_path << "\n";
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        TokenInfo info;
        info.text = line;
        info.hash = Blake3Hasher::hash(std::string_view(line));
        
        g_token_to_idx[line] = g_vocab.size();
        g_vocab.push_back(info);
    }
    
    std::cerr << "[VOCAB] Loaded " << g_vocab.size() << " tokens\n";
    return true;
}

std::vector<std::pair<std::string, std::string>> parse_bpe_merges(const fs::path& tokenizer_path) {
    std::vector<std::pair<std::string, std::string>> merges;
    
    std::ifstream file(tokenizer_path);
    if (!file) {
        std::cerr << "Cannot open tokenizer: " << tokenizer_path << "\n";
        return merges;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    
    size_t pos = content.find("\"merges\"");
    if (pos == std::string::npos) {
        std::cerr << "[BPE] No merges found in tokenizer.json\n";
        return merges;
    }
    
    pos = content.find('[', pos);
    if (pos == std::string::npos) return merges;
    
    size_t end = content.find(']', pos);
    if (end == std::string::npos) return merges;
    
    std::string arr = content.substr(pos + 1, end - pos - 1);
    
    size_t i = 0;
    while (i < arr.size()) {
        size_t q1 = arr.find('"', i);
        if (q1 == std::string::npos) break;
        size_t q2 = arr.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        
        std::string merge = arr.substr(q1 + 1, q2 - q1 - 1);
        
        size_t sp = merge.find(' ');
        if (sp != std::string::npos) {
            merges.emplace_back(merge.substr(0, sp), merge.substr(sp + 1));
        }
        
        i = q2 + 1;
    }
    
    std::cerr << "[BPE] Loaded " << merges.size() << " merge rules\n";
    return merges;
}

std::vector<SemanticEdge> merges_to_edges(
    const std::vector<std::pair<std::string, std::string>>& merges
) {
    std::vector<SemanticEdge> edges;
    
    float max_weight = static_cast<float>(merges.size());
    
    for (size_t i = 0; i < merges.size(); ++i) {
        const auto& [left, right] = merges[i];
        
        float weight = (max_weight - static_cast<float>(i)) / max_weight;
        
        Blake3Hash left_hash = Blake3Hasher::hash(std::string_view(left));
        Blake3Hash right_hash = Blake3Hasher::hash(std::string_view(right));
        
        edges.push_back({left_hash, right_hash, weight});
    }
    
    std::cerr << "[BPE] Created " << edges.size() << " semantic edges\n";
    return edges;
}

bool insert_edges(PGconn* conn, const std::vector<SemanticEdge>& edges, float threshold) {
    if (edges.empty()) return true;
    
    std::vector<const SemanticEdge*> filtered;
    for (const auto& e : edges) {
        if (e.weight >= threshold) {
            filtered.push_back(&e);
        }
    }
    
    std::cerr << "[EDGES] Inserting " << filtered.size() << " edges (threshold=" << threshold << ")\n";
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_edge ("
        "  from_id BYTEA,"
        "  to_id BYTEA,"
        "  weight REAL"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp_edge failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    res = PQexec(conn, "COPY tmp_edge FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY start failed\n";
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 20);
    
    for (const auto* e : filtered) {
        batch += "\\\\x";
        batch += e->from_token.to_hex();
        batch += "\t\\\\x";
        batch += e->to_token.to_hex();
        batch += "\t";
        batch += std::to_string(e->weight);
        batch += "\n";
        
        if (batch.size() > (1 << 19)) {
            PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
            batch.clear();
        }
    }
    
    if (!batch.empty()) {
        PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
    }
    
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);
    
    res = PQexec(conn,
        "INSERT INTO atom (id, geom, children, hilbert_lo, hilbert_hi, depth, atom_count) "
        "SELECT "
        "  hypercube_blake3(from_id || to_id), "
        "  ST_SetSRID(ST_MakeLine("
        "    (SELECT ST_SetSRID(ST_MakePoint("
        "      ST_X(geom), ST_Y(geom), ST_Z(geom), e.weight"
        "    ), 0) FROM atom WHERE id = e.from_id), "
        "    (SELECT ST_SetSRID(ST_MakePoint("
        "      ST_X(geom), ST_Y(geom), ST_Z(geom), e.weight"
        "    ), 0) FROM atom WHERE id = e.to_id)"
        "  ), 0), "
        "  ARRAY[from_id, to_id], "
        "  0, 0, 1, 2 "
        "FROM tmp_edge e "
        "WHERE EXISTS (SELECT 1 FROM atom WHERE id = e.from_id) "
        "  AND EXISTS (SELECT 1 FROM atom WHERE id = e.to_id) "
        "ON CONFLICT (id) DO NOTHING");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Edge insert failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    
    int inserted = atoi(PQcmdTuples(res));
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    std::cerr << "[EDGES] Inserted " << inserted << " new semantic edges\n";
    return true;
}

int main(int argc, char* argv[]) {
    IngestConfig config;
    std::string model_dir;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            config.conninfo = "dbname=" + std::string(argv[++i]);
        } else if (arg == "-U" && i + 1 < argc) {
            config.conninfo += " user=" + std::string(argv[++i]);
        } else if (arg == "-h" && i + 1 < argc) {
            config.conninfo += " host=" + std::string(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            config.weight_threshold = std::stof(argv[++i]);
        } else if (arg == "-v") {
            config.verbose = true;
        } else if (arg[0] != '-') {
            model_dir = arg;
        }
    }
    
    if (model_dir.empty()) {
        std::cerr << "Usage: ingest_safetensor [-d db] [-U user] [-h host] [-t threshold] <model_dir>\n";
        return 1;
    }
    
    fs::path dir(model_dir);
    if (!fs::is_directory(dir)) {
        std::cerr << "Not a directory: " << model_dir << "\n";
        return 1;
    }
    
    std::cerr << "=== Safetensor Package Ingester ===\n";
    std::cerr << "Directory: " << model_dir << "\n";
    std::cerr << "Threshold: " << config.weight_threshold << "\n\n";
    
    fs::path vocab_path, tokenizer_path, model_path;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        if (name == "vocab.txt") vocab_path = entry.path();
        else if (name == "tokenizer.json") tokenizer_path = entry.path();
        else if (name.find(".safetensors") != std::string::npos) model_path = entry.path();
    }
    
    PGconn* conn = PQconnectdb(config.conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }
    
    if (!vocab_path.empty()) {
        std::cerr << "[1] Parsing vocab: " << vocab_path << "\n";
        parse_vocab(vocab_path);
    }
    
    if (!tokenizer_path.empty()) {
        std::cerr << "[2] Parsing BPE merges: " << tokenizer_path << "\n";
        auto merges = parse_bpe_merges(tokenizer_path);
        auto edges = merges_to_edges(merges);
        insert_edges(conn, edges, config.weight_threshold);
    }
    
    if (!model_path.empty()) {
        std::cerr << "[3] Extracting semantic edges from: " << model_path << "\n";
        auto weight_edges = extract_safetensor_weights(model_path, config);
        std::cerr << "[WEIGHTS] Extracted " << weight_edges.size() << " semantic edges\n";
        insert_edges(conn, weight_edges, config.weight_threshold);
    }
    
    PQfinish(conn);
    std::cerr << "\n=== Complete ===\n";
    return 0;
}

std::vector<SemanticEdge> extract_safetensor_weights(const fs::path& safetensor_path, const IngestConfig& config) {
    std::vector<SemanticEdge> edges;
    std::ifstream file(safetensor_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open: " << safetensor_path << "\n";
        return edges;
    }
    
    // Read 8-byte header size
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), 8);
    
    // Read JSON header
    std::vector<char> header_buf(header_size);
    file.read(header_buf.data(), header_size);
    std::string header_json(header_buf.begin(), header_buf.end());
    
    // Parse to find embedding layer tensor name and shape
    // Common patterns: "embeddings.weight", "word_embeddings.weight", "token_embedding.weight"
    size_t embed_pos = header_json.find("\"embeddings\"");
    if (embed_pos == std::string::npos) {
        embed_pos = header_json.find("\"word_embeddings\"");
    }
    if (embed_pos == std::string::npos) {
        std::cerr << "[WEIGHTS] No embedding layer found\n";
        return edges;
    }
    
    // Extract shape: [vocab_size, embedding_dim]
    size_t shape_start = header_json.find("\"shape\"", embed_pos);
    size_t bracket_start = header_json.find("[", shape_start);
    size_t bracket_end = header_json.find("]", bracket_start);
    std::string shape_str = header_json.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    
    size_t comma_pos = shape_str.find(",");
    int vocab_size = std::stoi(shape_str.substr(0, comma_pos));
    int embed_dim = std::stoi(shape_str.substr(comma_pos + 1));
    
    std::cerr << "[WEIGHTS] Found embeddings: " << vocab_size << " tokens x " << embed_dim << " dims\n";
    
    // Extract data offset
    size_t offset_start = header_json.find("\"data_offsets\"", embed_pos);
    size_t offset_bracket = header_json.find("[", offset_start);
    size_t offset_end = header_json.find("]", offset_bracket);
    std::string offset_str = header_json.substr(offset_bracket + 1, offset_end - offset_bracket - 1);
    comma_pos = offset_str.find(",");
    uint64_t data_start = std::stoull(offset_str.substr(0, comma_pos));
    
    // Seek to embedding weights
    file.seekg(8 + header_size + data_start);
    
    // Compute cosine similarity between all token pairs (sparse: only top-k per token)
    std::cerr << "[WEIGHTS] Computing top-" << config.top_k_per_token << " similarities per token...\n";
    
    for (int i = 0; i < std::min(vocab_size, (int)g_vocab.size()); i++) {
        if (i % 1000 == 0) std::cerr << "  " << i << "/" << vocab_size << "\r" << std::flush;
        
        // Read embedding for token i
        std::vector<float> embed_i(embed_dim);
        file.read(reinterpret_cast<char*>(embed_i.data()), embed_dim * sizeof(float));
        
        float norm_i = 0.0f;
        for (float v : embed_i) norm_i += v * v;
        norm_i = std::sqrt(norm_i);
        
        // Track top-k most similar
        std::vector<std::pair<float, int>> top_k;
        
        // Compare with all other tokens
        auto current_pos = file.tellg();
        file.seekg(8 + header_size + data_start);
        
        for (int j = 0; j < std::min(vocab_size, (int)g_vocab.size()); j++) {
            if (i == j) {
                file.seekg(embed_dim * sizeof(float), std::ios::cur);
                continue;
            }
            
            std::vector<float> embed_j(embed_dim);
            file.read(reinterpret_cast<char*>(embed_j.data()), embed_dim * sizeof(float));
            
            float norm_j = 0.0f;
            float dot = 0.0f;
            for (int k = 0; k < embed_dim; k++) {
                dot += embed_i[k] * embed_j[k];
                norm_j += embed_j[k] * embed_j[k];
            }
            norm_j = std::sqrt(norm_j);
            
            float cosine = dot / (norm_i * norm_j + 1e-8f);
            
            if (cosine > config.weight_threshold) {
                top_k.push_back({cosine, j});
                if (top_k.size() > config.top_k_per_token) {
                    std::sort(top_k.begin(), top_k.end(), std::greater<>());
                    top_k.resize(config.top_k_per_token);
                }
            }
        }
        
        // Create edges
        for (auto [weight, j] : top_k) {
            edges.push_back({g_vocab[i].hash, g_vocab[j].hash, weight});
        }
        
        file.seekg(current_pos);
    }
    
    std::cerr << "\n";
    return edges;
}
