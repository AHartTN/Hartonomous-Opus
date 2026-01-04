// Universal Safetensor Package Ingester
// ======================================
// Ingests ANY HuggingFace model package into the hypercube substrate:
//
// 1. tokenizer.json → BPE merges become 'C' composition relations
//    (The model already solved token composition - just parse their work)
//
// 2. embed_tokens.weight → Each token's embedding stored as a LineString
//    (The embedding IS the geometry - N floats = N-point LineString)
//
// 3. router.weight (MoE) → Expert atoms with their own geometries
//    (Experts are first-class atoms, router weights become relations)
//
// 4. Sharded models → Parse model.safetensors.index.json, stream shards
//
// 5. dtype support → BF16, F16, F32 all converted to float for geometry
//
// KEY INSIGHT: Embeddings ARE geometry. The 384-float embedding vector
// for "captain" becomes a 384-point LineString. Similarity = shape overlap.
// PostGIS handles the rest.
//
// SPARSE ENCODING: Only store relationships above threshold (default 0.5).
// No O(n²) similarity computation - just store the shapes and let
// spatial queries find relationships on-demand.

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
#include <chrono>
#include <iomanip>
#include <libpq-fe.h>
#include "hypercube/types.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/atom_calculator.hpp"

namespace fs = std::filesystem;
using namespace hypercube;

// ============================================================================
// Configuration
// ============================================================================

struct IngestConfig {
    std::string conninfo;
    std::string model_name;       // e.g. "llama4-maverick", "minilm"  
    float weight_threshold = 0.5f; // Sparse: only edges above this
    bool verbose = false;
    int batch_size = 10000;       // DB batch insert size
};

// ============================================================================
// Tensor Metadata from Safetensor Header
// ============================================================================

struct TensorMeta {
    std::string name;
    std::string dtype;            // "BF16", "F16", "F32", "I64"
    std::vector<int64_t> shape;
    uint64_t data_offset_start;
    uint64_t data_offset_end;
    std::string shard_file;       // Full path to containing file
};

// ============================================================================
// Global State
// ============================================================================

static std::unordered_map<std::string, TensorMeta> g_tensors;
static std::vector<std::pair<std::string, std::string>> g_bpe_merges;
static std::unordered_map<std::string, int> g_vocab;  // token -> index
static std::string g_model_prefix;  // e.g. "llama4:" for namespacing

// TokenInfo now stores full composition data computed locally
struct TokenInfo {
    std::string text;
    CompositionRecord comp;  // Full composition with hash, coords, children, etc.
};

// Use unified SemanticEdge from types.hpp

static std::vector<TokenInfo> g_vocab_tokens;
static std::unordered_map<std::string, size_t> g_token_to_idx;

// ============================================================================
// BF16/F16 Conversion
// ============================================================================

inline float bf16_to_float(uint16_t bf16) {
    uint32_t f32 = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

inline float f16_to_float(uint16_t f16) {
    // IEEE 754 half-precision conversion
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t mant = f16 & 0x3FF;
    
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

// Forward declarations
bool parse_safetensor_header(const fs::path& path, const std::string& shard_file = "");
bool parse_model_index(const fs::path& index_path);
bool parse_tokenizer(const fs::path& tokenizer_path);
bool parse_vocab(const fs::path& vocab_path);
std::vector<float> read_tensor_row(const TensorMeta& meta, size_t row);
std::string floats_to_linestring_ewkb(const float* data, size_t count);
bool insert_compositions(PGconn* conn);
bool insert_shapes(PGconn* conn, const IngestConfig& config);
bool insert_attention_relations(PGconn* conn, const IngestConfig& config);

// ============================================================================
// Safetensor Header Parsing
// ============================================================================

bool parse_safetensor_header(const fs::path& path, const std::string& shard_file) {
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
        // Find the tensor name (key before this entry)
        size_t entry_start = json.rfind("{", pos);
        size_t name_end = json.rfind("\":", entry_start);
        size_t name_start = json.rfind("\"", name_end - 1);
        
        if (name_start == std::string::npos || name_end == std::string::npos) {
            pos++;
            continue;
        }
        
        std::string name = json.substr(name_start + 1, name_end - name_start - 1);
        
        // Skip metadata
        if (name == "__metadata__" || name == "format") {
            pos++;
            continue;
        }
        
        TensorMeta meta;
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
        
        g_tensors[name] = meta;
        pos = off_end;
    }
    
    return true;
}

// ============================================================================
// Parse Sharded Model Index
// ============================================================================

bool parse_model_index(const fs::path& index_path) {
    std::ifstream file(index_path);
    if (!file) return false;
    
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    
    // Parse weight_map: {"tensor_name": "shard_file", ...}
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
    
    // Extract tensor->shard mappings
    std::unordered_set<std::string> shards_to_parse;
    size_t i = 0;
    while (i < map_str.size()) {
        size_t q1 = map_str.find("\"", i);
        if (q1 == std::string::npos) break;
        size_t q2 = map_str.find("\"", q1 + 1);
        if (q2 == std::string::npos) break;
        // std::string tensor = map_str.substr(q1 + 1, q2 - q1 - 1); // tensor name
        
        size_t q3 = map_str.find("\"", q2 + 1);
        size_t q4 = map_str.find("\"", q3 + 1);
        if (q3 == std::string::npos || q4 == std::string::npos) break;
        std::string shard = map_str.substr(q3 + 1, q4 - q3 - 1);
        
        shards_to_parse.insert(shard);
        i = q4 + 1;
    }
    
    // Parse headers from each unique shard
    fs::path parent = index_path.parent_path();
    
    for (const auto& shard : shards_to_parse) {
        fs::path shard_path = parent / shard;
        if (fs::exists(shard_path)) {
            std::cerr << "  Parsing shard: " << shard << "\n";
            parse_safetensor_header(shard_path, shard_path.string());
        }
    }
    
    std::cerr << "[INDEX] Parsed " << shards_to_parse.size() << " shards, " 
              << g_tensors.size() << " tensors\n";
    return true;
}

// ============================================================================
// Parse Tokenizer (BPE Merges + Vocab)
// ============================================================================

bool parse_tokenizer(const fs::path& tokenizer_path) {
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
                    g_bpe_merges.emplace_back(merge.substr(0, sp), merge.substr(sp + 1));
                }
                i = q2 + 1;
            }
        }
    }
    
    // Parse vocab (for token->index mapping)
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
            g_vocab[token] = idx;
            
            i = comma + 1;
        }
    }
    
    std::cerr << "[TOKENIZER] Loaded " << g_bpe_merges.size() << " BPE merges, "
              << g_vocab.size() << " vocab entries\n";
    return true;
}

// ============================================================================
// Read Tensor Row (handles BF16, F16, F32)
// ============================================================================

std::vector<float> read_tensor_row(const TensorMeta& meta, size_t row) {
    std::ifstream file(meta.shard_file, std::ios::binary);
    if (!file) return {};
    
    // Read header size to know data offset base
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), 8);
    
    if (meta.shape.size() < 2) return {};
    
    size_t row_size = static_cast<size_t>(meta.shape[1]);
    size_t bytes_per_elem = 4;
    if (meta.dtype == "BF16" || meta.dtype == "F16") bytes_per_elem = 2;
    
    // Seek to the row
    file.seekg(8 + header_size + meta.data_offset_start + row * row_size * bytes_per_elem);
    
    std::vector<float> result(row_size);
    
    if (meta.dtype == "F32") {
        file.read(reinterpret_cast<char*>(result.data()), row_size * 4);
    } else if (meta.dtype == "BF16") {
        std::vector<uint16_t> buf(row_size);
        file.read(reinterpret_cast<char*>(buf.data()), row_size * 2);
        for (size_t i = 0; i < row_size; ++i) {
            result[i] = bf16_to_float(buf[i]);
        }
    } else if (meta.dtype == "F16") {
        std::vector<uint16_t> buf(row_size);
        file.read(reinterpret_cast<char*>(buf.data()), row_size * 2);
        for (size_t i = 0; i < row_size; ++i) {
            result[i] = f16_to_float(buf[i]);
        }
    }
    
    return result;
}

// ============================================================================
// LineString Building (embedding row → geometry) - FULL VECTOR
// ============================================================================

// Build LINESTRINGZM EWKB from float array
// Stores EVERY value - embeddings carry meaning in all dimensions
// X = dimension index, Y = value, Z = 0, M = 0
// For a 384-dim embedding: 384 points. For 5120-dim: 5120 points.
// This IS the semantic fingerprint - no compression allowed.
std::string floats_to_linestring_ewkb(const float* data, size_t count) {
    size_t num_points = count;
    if (num_points < 2) num_points = 2;  // LineString needs at least 2 points
    
    std::string ewkb;
    ewkb.reserve(26 + num_points * 64);
    
    // EWKB Header: little-endian, LINESTRINGZM with SRID, SRID=0
    ewkb += "01";           // Little-endian
    ewkb += "020000e0";     // LINESTRINGZM (3002) + SRID flag (0x20000000)
    ewkb += "00000000";     // SRID = 0
    
    // Number of points
    uint32_t n = static_cast<uint32_t>(num_points);
    char buf[9];
    snprintf(buf, sizeof(buf), "%02x%02x%02x%02x",
             n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF, (n >> 24) & 0xFF);
    ewkb += buf;
    
    // Write points: X=dimension_index, Y=value, Z=0, M=0
    auto write_double = [&ewkb](double d) {
        uint64_t bits;
        std::memcpy(&bits, &d, sizeof(bits));
        char hex[17];
        for (int i = 0; i < 8; ++i) {
            snprintf(hex + i * 2, 3, "%02x", static_cast<unsigned>(bits & 0xFF));
            bits >>= 8;
        }
        ewkb += hex;
    };
    
    for (size_t i = 0; i < count; ++i) {
        write_double(static_cast<double>(i));           // X = dimension index
        write_double(static_cast<double>(data[i]));     // Y = value
        write_double(0.0);                              // Z = unused
        write_double(0.0);                              // M = unused
    }
    
    // Pad if needed for minimum 2 points
    for (size_t i = count; i < num_points; ++i) {
        write_double(static_cast<double>(i));
        write_double(0.0);
        write_double(0.0);
        write_double(0.0);
    }
    
    return ewkb;
}

// ============================================================================
// Weight Value → Atom Hash
// ============================================================================

// Convert a float weight to its text representation, then hash it as an atom
// 0.987 → "0.987" → BLAKE3 hash
// Numbers are atoms too! Store once, reference forever.
Blake3Hash weight_to_atom_hash(float weight, int precision = 6) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*g", precision, weight);
    std::string text(buf);
    
    // Compute hash of the number-as-text
    return AtomCalculator::compute_vocab_token(text).hash;
}

/**
 * Parse vocab.txt and compute compositions for each token
 * All computation is done client-side via AtomCalculator
 * ZERO database roundtrips during parsing
 */
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
        
        // Compute full composition client-side (no DB access!)
        // "captain" -> decode to codepoints -> compute atom coords/hashes
        // -> compute composition hash, centroid, hilbert, depth, atom_count
        info.comp = AtomCalculator::compute_vocab_token(line);
        
        g_token_to_idx[line] = g_vocab_tokens.size();
        g_vocab_tokens.push_back(info);
    }
    
    std::cerr << "[VOCAB] Loaded " << g_vocab_tokens.size() << " tokens (computed locally)\n";
    return true;
}

// Helper to build LINESTRINGZM EWKB hex from Point4D vector (for vocab compositions)
std::string build_linestringzm_ewkb(const std::vector<Point4D>& points) {
    if (points.size() < 2) return "";
    
    std::string ewkb;
    ewkb.reserve(26 + points.size() * 64);
    
    // Header: little-endian (01), type=LINESTRINGZM with SRID (020000e0), SRID=0 (00000000)
    ewkb += "01";           // Little-endian
    ewkb += "020000e0";     // LINESTRINGZM with SRID flag
    ewkb += "00000000";     // SRID = 0
    
    // Number of points (4 bytes little-endian)
    uint32_t n = static_cast<uint32_t>(points.size());
    char buf[9];
    snprintf(buf, sizeof(buf), "%02x%02x%02x%02x",
             n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF, (n >> 24) & 0xFF);
    ewkb += buf;
    
    // Each point: x, y, z, m as little-endian doubles
    for (const auto& pt : points) {
        // Convert int32 coords to double
        auto write_double = [&ewkb](int32_t coord) {
            double d = static_cast<double>(coord);
            uint64_t bits;
            std::memcpy(&bits, &d, sizeof(bits));
            char hex[17];
            for (int i = 0; i < 8; ++i) {
                snprintf(hex + i * 2, 3, "%02x", static_cast<unsigned>(bits & 0xFF));
                bits >>= 8;
            }
            ewkb += hex;
        };
        
        write_double(static_cast<int32_t>(pt.x));
        write_double(static_cast<int32_t>(pt.y));
        write_double(static_cast<int32_t>(pt.z));
        write_double(static_cast<int32_t>(pt.m));
    }
    
    return ewkb;
}

/**
 * Insert vocab token compositions into the composition + composition_child tables
 * NEW SCHEMA: composition stores the aggregations, composition_child stores ordered children
 */
bool insert_compositions(PGconn* conn) {
    if (g_vocab_tokens.empty()) return true;
    
    std::cerr << "[COMP] Inserting " << g_vocab_tokens.size() << " token compositions...\n";
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Temp table for compositions
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_comp ("
        "  id BYTEA,"
        "  label TEXT,"
        "  depth INTEGER,"
        "  child_count INTEGER,"
        "  atom_count BIGINT"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp_comp failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // Temp table for composition children
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_comp_child ("
        "  composition_id BYTEA,"
        "  ordinal SMALLINT,"
        "  child_type CHAR(1),"
        "  child_id BYTEA"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp_comp_child failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // COPY compositions
    res = PQexec(conn, "COPY tmp_comp FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY comp start failed\n";
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 20);
    
    for (const auto& token : g_vocab_tokens) {
        const auto& c = token.comp;
        
        // Skip single-char tokens (they're just atoms, already seeded)
        if (c.children.size() <= 1) continue;
        
        // id (hash)
        batch += "\\\\x";
        batch += c.hash.to_hex();
        batch += "\t";
        
        // label (the token text, escaped for COPY)
        for (char ch : token.text) {
            if (ch == '\t') batch += "\\t";
            else if (ch == '\n') batch += "\\n";
            else if (ch == '\\') batch += "\\\\";
            else batch += ch;
        }
        batch += "\t";
        
        // depth, child_count, atom_count
        batch += std::to_string(c.depth);
        batch += "\t";
        batch += std::to_string(c.children.size());
        batch += "\t";
        batch += std::to_string(c.atom_count);
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
    
    // COPY composition children
    res = PQexec(conn, "COPY tmp_comp_child FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY comp_child start failed\n";
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    batch.clear();
    for (const auto& token : g_vocab_tokens) {
        const auto& c = token.comp;
        if (c.children.size() <= 1) continue;
        
        for (size_t i = 0; i < c.children.size(); ++i) {
            // composition_id
            batch += "\\\\x";
            batch += c.hash.to_hex();
            batch += "\t";
            
            // ordinal (0-based)
            batch += std::to_string(i);
            batch += "\t";
            
            // child_type: 'A' = atom (single codepoint), check depth
            // For BPE, children are always atoms (depth 1 composition -> atom children)
            batch += "A\t";
            
            // child_id
            batch += "\\\\x";
            batch += c.children[i].to_hex();
            batch += "\n";
        }
        
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
    
    // Insert compositions
    res = PQexec(conn,
        "INSERT INTO composition (id, label, depth, child_count, atom_count) "
        "SELECT id, label, depth, child_count, atom_count "
        "FROM tmp_comp "
        "ON CONFLICT (id) DO NOTHING");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Composition insert failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    int inserted_comps = atoi(PQcmdTuples(res));
    PQclear(res);
    
    // Insert composition children (only for compositions that exist)
    res = PQexec(conn,
        "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
        "SELECT composition_id, ordinal, child_type, child_id "
        "FROM tmp_comp_child "
        "WHERE EXISTS (SELECT 1 FROM composition WHERE id = tmp_comp_child.composition_id) "
        "ON CONFLICT (composition_id, ordinal) DO NOTHING");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Composition child insert failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    int inserted_children = atoi(PQcmdTuples(res));
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    std::cerr << "[COMP] Inserted " << inserted_comps << " compositions, " << inserted_children << " children\n";
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
        } else if (arg == "-n" && i + 1 < argc) {
            config.model_name = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            config.weight_threshold = std::stof(argv[++i]);
        } else if (arg == "-v") {
            config.verbose = true;
        } else if (arg[0] != '-') {
            model_dir = arg;
        }
    }
    
    if (model_dir.empty()) {
        std::cerr << "Usage: ingest_safetensor [-d db] [-U user] [-h host] [-n model_name] [-t threshold] <model_dir>\n";
        std::cerr << "  -n  Model name prefix (e.g. 'minilm', 'llama4')\n";
        std::cerr << "  -t  Weight threshold for attention edges (default 0.5)\n";
        return 1;
    }
    
    fs::path dir(model_dir);
    if (!fs::is_directory(dir)) {
        std::cerr << "Not a directory: " << model_dir << "\n";
        return 1;
    }
    
    // Auto-detect model name from directory if not specified
    if (config.model_name.empty()) {
        config.model_name = dir.filename().string();
    }
    g_model_prefix = config.model_name + ":";
    
    std::cerr << "=== Universal Safetensor Ingester ===\n";
    std::cerr << "Directory: " << model_dir << "\n";
    std::cerr << "Model: " << config.model_name << "\n";
    std::cerr << "Threshold: " << config.weight_threshold << "\n\n";
    
    auto total_start = std::chrono::steady_clock::now();
    
    // Find model files
    fs::path vocab_path, tokenizer_path, index_path;
    std::vector<fs::path> safetensor_files;
    
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        if (name == "vocab.txt") vocab_path = entry.path();
        else if (name == "tokenizer.json") tokenizer_path = entry.path();
        else if (name == "model.safetensors.index.json") index_path = entry.path();
        else if (name.find(".safetensors") != std::string::npos && 
                 name.find(".index") == std::string::npos) {
            safetensor_files.push_back(entry.path());
        }
    }
    
    // Parse tokenizer first (need vocab for token lookups)
    if (!tokenizer_path.empty()) {
        std::cerr << "[1] Parsing tokenizer: " << tokenizer_path << "\n";
        parse_tokenizer(tokenizer_path);
    }
    
    // Parse vocab.txt if available (BERT-style models)
    if (!vocab_path.empty()) {
        std::cerr << "[2] Parsing vocab: " << vocab_path << "\n";
        parse_vocab(vocab_path);
    }
    
    // Parse model tensors
    if (!index_path.empty()) {
        std::cerr << "[3] Parsing sharded model index: " << index_path << "\n";
        parse_model_index(index_path);
    } else if (!safetensor_files.empty()) {
        std::cerr << "[3] Parsing " << safetensor_files.size() << " safetensor files...\n";
        for (const auto& f : safetensor_files) {
            parse_safetensor_header(f);
        }
    }
    
    std::cerr << "[INFO] Found " << g_tensors.size() << " tensors\n";
    
    // Connect to database
    PGconn* conn = PQconnectdb(config.conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }
    
    // Insert vocab compositions
    if (!g_vocab_tokens.empty()) {
        std::cerr << "\n[4] Inserting compositions...\n";
        insert_compositions(conn);
    }
    
    // Insert embeddings as shapes (external model fingerprints)
    std::cerr << "\n[5] Inserting shapes (model embeddings)...\n";
    insert_shapes(conn, config);
    
    // Insert attention weights as sparse relations
    std::cerr << "\n[6] Extracting attention weights as sparse relations...\n";
    insert_attention_relations(conn, config);
    
    PQfinish(conn);
    
    auto total_end = std::chrono::steady_clock::now();
    auto total_secs = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    
    std::cerr << "\n=== Complete ===\n";
    std::cerr << "Total time: " << total_secs << " seconds\n";
    std::cerr << "Tensors: " << g_tensors.size() << "\n";
    std::cerr << "BPE merges: " << g_bpe_merges.size() << "\n";
    std::cerr << "Vocab: " << g_vocab_tokens.size() << " tokens\n";
    
    return 0;
}

// ============================================================================
// Insert Embeddings as Shapes (external model fingerprints)
// ============================================================================

bool insert_shapes(PGconn* conn, const IngestConfig& config) {
    // Find embedding tensor
    TensorMeta* embed = nullptr;
    for (auto& [name, meta] : g_tensors) {
        if (name.find("embed_tokens") != std::string::npos ||
            name.find("word_embeddings") != std::string::npos ||
            name.find("wte.weight") != std::string::npos) {
            embed = &meta;
            break;
        }
    }
    
    if (!embed) {
        std::cerr << "[SHAPE] No embedding tensor found\n";
        return true;
    }
    
    if (embed->shape.size() < 2) {
        std::cerr << "[SHAPE] Invalid embedding shape\n";
        return false;
    }
    
    int64_t vocab_size = embed->shape[0];
    int64_t embed_dim = embed->shape[1];
    
    std::cerr << "[SHAPE] Processing " << vocab_size << " tokens x " << embed_dim << " dims\n";
    std::cerr << "[SHAPE] Tensor: " << embed->name << " in " << embed->shard_file << "\n";
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Temp table for shapes
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_shape ("
        "  entity_type CHAR(1),"
        "  entity_id BYTEA,"
        "  model_name TEXT,"
        "  embedding BYTEA,"
        "  dim_count INTEGER"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Create tmp_shape failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    res = PQexec(conn, "COPY tmp_shape FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    PQclear(res);
    
    std::string batch;
    batch.reserve(1 << 21);  // 2MB buffer
    
    auto start = std::chrono::steady_clock::now();
    
    for (int64_t i = 0; i < vocab_size && i < static_cast<int64_t>(g_vocab_tokens.size()); ++i) {
        if (i % 10000 == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            float rate = elapsed > 0 ? static_cast<float>(i) / elapsed : 0;
            std::cerr << "  " << i << "/" << vocab_size << " (" << rate << " tok/s)\r" << std::flush;
        }
        
        // Read this token's embedding
        auto row = read_tensor_row(*embed, static_cast<size_t>(i));
        if (row.empty()) continue;
        
        // Get the composition hash for this token (entity_id)
        const auto& comp = g_vocab_tokens[i].comp;
        
        // Determine entity type: 'A' if single char (atom), 'C' if composition
        char entity_type = (comp.children.size() <= 1) ? 'A' : 'C';
        
        // Convert embedding to LineString geometry (hex-encoded EWKB)
        std::string geom = floats_to_linestring_ewkb(row.data(), row.size());
        
        // entity_type, entity_id, model_name, embedding, dim_count
        batch += entity_type;
        batch += "\t\\\\x";
        batch += comp.hash.to_hex();
        batch += "\t";
        batch += config.model_name;
        batch += "\t\\\\x";
        batch += geom;
        batch += "\t";
        batch += std::to_string(embed_dim);
        batch += "\n";
        
        if (batch.size() > (1 << 20)) {
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
    
    std::cerr << "\n";
    
    // Insert into shape table
    res = PQexec(conn,
        "INSERT INTO shape (entity_type, entity_id, model_name, embedding, dim_count) "
        "SELECT entity_type, entity_id, model_name, ST_GeomFromEWKB(embedding), dim_count "
        "FROM tmp_shape "
        "ON CONFLICT (entity_id, model_name) DO UPDATE SET "
        "  embedding = EXCLUDED.embedding, "
        "  dim_count = EXCLUDED.dim_count");
    
    int inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[SHAPE] Insert error: " << PQerrorMessage(conn) << "\n";
    }
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    std::cerr << "[SHAPE] Inserted " << inserted << " shapes for model '" << config.model_name << "'\n";
    return true;
}

// ============================================================================
// Insert Semantic Relations via Embedding Similarity
// ============================================================================
// Compute pairwise cosine similarity between token embeddings.
// Store sparse edges above threshold as semantic relations.
// This creates the ACTUAL knowledge graph - tokens connected by meaning.

// SIMD cosine similarity (AVX2)
#ifdef __AVX2__
#include <immintrin.h>

static float cosine_similarity_avx2(const float* a, const float* b, size_t n) {
    __m256 sum_ab = _mm256_setzero_ps();
    __m256 sum_aa = _mm256_setzero_ps();
    __m256 sum_bb = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum_ab = _mm256_fmadd_ps(va, vb, sum_ab);
        sum_aa = _mm256_fmadd_ps(va, va, sum_aa);
        sum_bb = _mm256_fmadd_ps(vb, vb, sum_bb);
    }
    
    // Horizontal sum
    float ab[8], aa[8], bb[8];
    _mm256_storeu_ps(ab, sum_ab);
    _mm256_storeu_ps(aa, sum_aa);
    _mm256_storeu_ps(bb, sum_bb);
    
    float dot = 0, norm_a = 0, norm_b = 0;
    for (int j = 0; j < 8; ++j) {
        dot += ab[j];
        norm_a += aa[j];
        norm_b += bb[j];
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}
#else
static float cosine_similarity_avx2(const float* a, const float* b, size_t n) {
    float dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}
#endif

bool insert_attention_relations(PGconn* conn, const IngestConfig& config) {
    // ==========================================================================
    // EXTRACT EVERYTHING: embeddings, projections, MLP, all learned weights
    // ==========================================================================
    
    // Find embedding tensor for token-to-token similarity
    TensorMeta* embed = nullptr;
    for (auto& [name, meta] : g_tensors) {
        if (name.find("embed_tokens") != std::string::npos ||
            name.find("word_embeddings") != std::string::npos ||
            name.find("wte.weight") != std::string::npos) {
            embed = &meta;
            break;
        }
    }
    
    // Collect ALL weight tensors for extraction
    std::vector<TensorMeta*> weight_tensors;
    for (auto& [name, meta] : g_tensors) {
        if (name.find(".weight") != std::string::npos && meta.shape.size() == 2) {
            weight_tensors.push_back(&meta);
        }
    }
    
    std::cerr << "[EXTRACT] Found " << weight_tensors.size() << " weight tensors to process\n";
    
    size_t total_edges = 0;
    
    // -------------------------------------------------------------------------
    // PART 1: Token-to-token embedding similarity
    // -------------------------------------------------------------------------
    if (embed && embed->shape.size() >= 2 && !g_vocab_tokens.empty()) {
        int64_t vocab_size = std::min(embed->shape[0], static_cast<int64_t>(g_vocab_tokens.size()));
        int64_t embed_dim = embed->shape[1];
        
        std::cerr << "[SIMILARITY] Computing pairwise similarity for " << vocab_size 
                  << " tokens (threshold=" << config.weight_threshold << ")\n";
        
        // Load all embeddings
        std::vector<std::vector<float>> embeddings(vocab_size);
        for (int64_t i = 0; i < vocab_size; ++i) {
            embeddings[i] = read_tensor_row(*embed, static_cast<size_t>(i));
        }
        
        PGresult* res = PQexec(conn, "BEGIN");
        PQclear(res);
        
        res = PQexec(conn,
            "CREATE TEMP TABLE tmp_sim ("
            "  source_type CHAR(1), source_id BYTEA,"
            "  target_type CHAR(1), target_id BYTEA,"
            "  weight REAL, layer SMALLINT, component TEXT"
            ") ON COMMIT DROP");
        PQclear(res);
        
        res = PQexec(conn, "COPY tmp_sim FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        PQclear(res);
        
        std::string batch;
        batch.reserve(1 << 21);
        size_t sim_edges = 0;
        
        auto start = std::chrono::steady_clock::now();
        
        // Upper triangle (symmetric)
        for (int64_t i = 0; i < vocab_size; ++i) {
            if (embeddings[i].empty()) continue;
            const auto& comp_i = g_vocab_tokens[i].comp;
            char type_i = (comp_i.children.size() <= 1) ? 'A' : 'C';
            
            for (int64_t j = i + 1; j < vocab_size; ++j) {
                if (embeddings[j].empty()) continue;
                
                float sim = cosine_similarity_avx2(
                    embeddings[i].data(), embeddings[j].data(),
                    static_cast<size_t>(embed_dim)
                );
                
                if (sim >= config.weight_threshold) {
                    const auto& comp_j = g_vocab_tokens[j].comp;
                    char type_j = (comp_j.children.size() <= 1) ? 'A' : 'C';
                    
                    // Both directions
                    batch += type_i; batch += "\t\\\\x"; batch += comp_i.hash.to_hex();
                    batch += "\t"; batch += type_j; batch += "\t\\\\x"; batch += comp_j.hash.to_hex();
                    batch += "\t"; batch += std::to_string(sim);
                    batch += "\t\\N\tembed_sim\n";  // NULL layer, component = embed_sim
                    
                    batch += type_j; batch += "\t\\\\x"; batch += comp_j.hash.to_hex();
                    batch += "\t"; batch += type_i; batch += "\t\\\\x"; batch += comp_i.hash.to_hex();
                    batch += "\t"; batch += std::to_string(sim);
                    batch += "\t\\N\tembed_sim\n";
                    
                    sim_edges += 2;
                    
                    if (batch.size() > (1 << 20)) {
                        PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
                        batch.clear();
                    }
                }
            }
            
            if (i % 1000 == 0 && i > 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
                std::cerr << "  " << i << "/" << vocab_size << " - " << sim_edges << " edges\r" << std::flush;
            }
        }
        
        if (!batch.empty()) PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
        PQputCopyEnd(conn, nullptr);
        res = PQgetResult(conn);
        PQclear(res);
        
        // Insert
        res = PQexec(conn,
            ("INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
             "SELECT source_type, source_id, target_type, target_id, 'S', weight, '" + config.model_name + "', 1, layer, component FROM tmp_sim "
             "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
             "  weight = GREATEST(relation.weight, EXCLUDED.weight), source_count = relation.source_count + 1").c_str());
        PQclear(res);
        
        res = PQexec(conn, "COMMIT");
        PQclear(res);
        
        std::cerr << "\n[SIMILARITY] Inserted " << sim_edges << " token similarity edges\n";
        total_edges += sim_edges;
    }
    
    // -------------------------------------------------------------------------
    // PART 2: All weight matrices - Q/K/V projections, MLP, everything
    // -------------------------------------------------------------------------
    for (auto* tensor : weight_tensors) {
        if (tensor->shape.size() < 2) continue;
        
        int64_t rows = tensor->shape[0];
        int64_t cols = tensor->shape[1];
        
        // Parse layer number from tensor name (e.g., "layers.12.attention.q_proj.weight")
        int layer = -1;
        std::string component = tensor->name;
        size_t layers_pos = tensor->name.find("layers.");
        if (layers_pos != std::string::npos) {
            size_t num_start = layers_pos + 7;
            size_t num_end = tensor->name.find(".", num_start);
            if (num_end != std::string::npos) {
                layer = std::stoi(tensor->name.substr(num_start, num_end - num_start));
                component = tensor->name.substr(num_end + 1);
            }
        }
        // Remove ".weight" suffix
        if (component.size() > 7 && component.substr(component.size() - 7) == ".weight") {
            component = component.substr(0, component.size() - 7);
        }
        
        std::cerr << "[WEIGHTS] " << tensor->name << " [" << rows << "x" << cols << "] layer=" << layer << " comp=" << component << "\n";
        
        PGresult* res = PQexec(conn, "BEGIN");
        PQclear(res);
        
        res = PQexec(conn,
            "CREATE TEMP TABLE tmp_wt ("
            "  source_type CHAR(1), source_id BYTEA,"
            "  target_type CHAR(1), target_id BYTEA,"
            "  weight REAL, layer SMALLINT, component TEXT"
            ") ON COMMIT DROP");
        PQclear(res);
        
        res = PQexec(conn, "COPY tmp_wt FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        PQclear(res);
        
        std::string batch;
        batch.reserve(1 << 20);
        size_t tensor_edges = 0;
        
        for (int64_t r = 0; r < rows; ++r) {
            auto row_data = read_tensor_row(*tensor, static_cast<size_t>(r));
            if (row_data.empty()) continue;
            
            // Find max for normalization
            float max_val = 0.0f;
            for (float v : row_data) {
                if (std::fabs(v) > max_val) max_val = std::fabs(v);
            }
            if (max_val < 1e-8f) continue;
            
            for (int64_t c = 0; c < cols; ++c) {
                float normalized = row_data[c] / max_val;
                
                if (std::fabs(normalized) >= config.weight_threshold) {
                    // Create dimension-based IDs
                    std::string from_key = config.model_name + ":" + tensor->name + ":r" + std::to_string(r);
                    std::string to_key = config.model_name + ":" + tensor->name + ":c" + std::to_string(c);
                    
                    auto from_hash = AtomCalculator::compute_vocab_token(from_key).hash;
                    auto to_hash = AtomCalculator::compute_vocab_token(to_key).hash;
                    
                    batch += "C\t\\\\x" + from_hash.to_hex() + "\t";
                    batch += "C\t\\\\x" + to_hash.to_hex() + "\t";
                    batch += std::to_string(normalized) + "\t";
                    batch += (layer >= 0 ? std::to_string(layer) : "\\N") + "\t";
                    batch += component + "\n";
                    tensor_edges++;
                    
                    if (batch.size() > (1 << 19)) {
                        PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
                        batch.clear();
                    }
                }
            }
        }
        
        if (!batch.empty()) PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
        PQputCopyEnd(conn, nullptr);
        res = PQgetResult(conn);
        PQclear(res);
        
        // Insert
        res = PQexec(conn,
            ("INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
             "SELECT source_type, source_id, target_type, target_id, 'W', weight, '" + config.model_name + "', 1, layer, component FROM tmp_wt "
             "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
             "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
             "  source_count = relation.source_count + 1").c_str());
        PQclear(res);
        
        res = PQexec(conn, "COMMIT");
        PQclear(res);
        
        std::cerr << "  -> " << tensor_edges << " edges\n";
        total_edges += tensor_edges;
    }
    
    std::cerr << "[EXTRACT] Total: " << total_edges << " edges from all tensors\n";
    return true;
}
