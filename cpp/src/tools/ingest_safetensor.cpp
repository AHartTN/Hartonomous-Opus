// Universal Safetensor Package Ingester
// ======================================
// Ingests ANY HuggingFace model package into the hypercube substrate:
//
// ARCHITECTURE (CORRECT):
//   Atoms:       Unicode codepoints with YOUR deterministic 4D coordinates
//                (semantic_order → Hilbert decode → radial project to S³)
//   Compositions: Aggregations with centroids = average of atom children
//                (computed from atoms, NOT from model embeddings)
//   Relations:   Edges from multiple sources - THE KNOWLEDGE GRAPH:
//                - 'E' = Embedding k-NN similarity (token semantics)
//                - 'R' = Router weights (MoE expert routing)
//                - 'W' = Weight similarity (Q/K/V/O/MLP projection patterns)
//                - 'D' = Dimension activation (token→dimension mappings)
//                - 'C' = BPE composition relations
//
// INGESTION FLOW:
// 1. tokenizer.json → BPE merges become composition children
// 2. Compositions inserted with centroids computed from atom children
// 3. Embedding tensors → k-NN similarity → relation edges (type='E')
// 4. Router tensors (MoE) → expert routing → relation edges (type='R')
// 5. Attention projections (Q/K/V/O) → weight similarity → edges (type='W')
// 6. FFN/MLP projections (gate/up/down) → weight similarity → edges (type='W')
// 7. Token embeddings → top-k dimension activation → edges (type='D')
//
// THE "BEATEN PATH": By accumulating weight similarities across layers,
// we discover which dimensions co-activate frequently - these are the
// learned pathways that tokens follow through the model.
//
// KEY INSIGHT: Every weight matrix encodes relationships. We extract ALL
// of them as graph edges, building a comprehensive knowledge graph that
// captures how the model learned to process tokens.

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
#include <thread>
#include <mutex>
#include <atomic>
#include <libpq-fe.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include "hypercube/types.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/atom_calculator.hpp"
#include "hypercube/laplacian_4d.hpp"
#include "hypercube/embedding_ops.hpp"  // Centralized SIMD cosine similarity

#ifdef HAS_HNSWLIB
#include "hnswlib/hnswlib/hnswlib.h"
#endif

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
// Memory-Mapped File Cache (zero-copy tensor access)
// ============================================================================

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { unmap(); }
    
    bool map(const std::string& path) {
        if (data_) unmap();
        
#ifdef _WIN32
        file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) return false;
        
        LARGE_INTEGER size_li;
        GetFileSizeEx(file_handle_, &size_li);
        size_ = static_cast<size_t>(size_li.QuadPart);
        
        mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_handle_) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
        
        data_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
        if (!data_) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            mapping_handle_ = nullptr;
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }
#else
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        
        struct stat st;
        if (fstat(fd, &st) < 0) {
            close(fd);
            return false;
        }
        size_ = static_cast<size_t>(st.st_size);
        
        data_ = static_cast<const uint8_t*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0));
        close(fd);
        
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            return false;
        }
#endif
        return true;
    }
    
    void unmap() {
        if (!data_) return;
#ifdef _WIN32
        UnmapViewOfFile(data_);
        if (mapping_handle_) CloseHandle(mapping_handle_);
        if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        mapping_handle_ = nullptr;
#else
        munmap(const_cast<uint8_t*>(data_), size_);
#endif
        data_ = nullptr;
        size_ = 0;
    }
    
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    
private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#endif
};

// Global cache of memory-mapped files
static std::unordered_map<std::string, std::unique_ptr<MappedFile>> g_mmap_cache;
static std::mutex g_mmap_mutex;

// Get or create a memory-mapped file
static const MappedFile* get_mapped_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(g_mmap_mutex);
    auto it = g_mmap_cache.find(path);
    if (it != g_mmap_cache.end()) {
        return it->second.get();
    }
    auto mf = std::make_unique<MappedFile>();
    if (!mf->map(path)) {
        return nullptr;
    }
    auto ptr = mf.get();
    g_mmap_cache[path] = std::move(mf);
    return ptr;
}

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
std::vector<float> read_tensor_slice(const TensorMeta& meta, size_t slice_idx, size_t row);
bool insert_compositions(PGconn* conn);
bool insert_tensor_hierarchy(PGconn* conn, const IngestConfig& config);
bool extract_embedding_relations(PGconn* conn, const IngestConfig& config);
bool insert_attention_relations(PGconn* conn, const IngestConfig& config);

// ============================================================================
// Tensor Name Hierarchy Parsing
// ============================================================================
// Splits "encoder.layer.0.attention.self.query.weight" into path components:
//   encoder
//   encoder.layer
//   encoder.layer.0
//   encoder.layer.0.attention
//   encoder.layer.0.attention.self
//   encoder.layer.0.attention.self.query
//   encoder.layer.0.attention.self.query.weight
//
// Each becomes a composition with parent->child relationships.

std::vector<std::string> split_tensor_path(const std::string& tensor_name) {
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
    
    // Add final component
    if (!current.empty()) {
        if (components.empty()) {
            components.push_back(current);
        } else {
            components.push_back(components.back() + "." + current);
        }
    }
    
    return components;
}

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
// Read Tensor Row (handles BF16, F16, F32) - MEMORY MAPPED VERSION
// Zero-copy access to tensor data via mmap
// ============================================================================

std::vector<float> read_tensor_row(const TensorMeta& meta, size_t row) {
    if (meta.shape.size() < 2) return {};
    
    const MappedFile* mf = get_mapped_file(meta.shard_file);
    if (!mf || !mf->data()) return {};
    
    // First 8 bytes contain header size
    uint64_t header_size;
    std::memcpy(&header_size, mf->data(), 8);
    
    size_t row_size = static_cast<size_t>(meta.shape[1]);
    size_t bytes_per_elem = 4;
    if (meta.dtype == "BF16" || meta.dtype == "F16") bytes_per_elem = 2;
    
    // Calculate offset to the row
    size_t offset = 8 + header_size + meta.data_offset_start + row * row_size * bytes_per_elem;
    if (offset + row_size * bytes_per_elem > mf->size()) return {};
    
    const uint8_t* ptr = mf->data() + offset;
    std::vector<float> result(row_size);
    
    if (meta.dtype == "F32") {
        std::memcpy(result.data(), ptr, row_size * 4);
    } else if (meta.dtype == "BF16") {
        const uint16_t* buf = reinterpret_cast<const uint16_t*>(ptr);
        for (size_t i = 0; i < row_size; ++i) {
            result[i] = bf16_to_float(buf[i]);
        }
    } else if (meta.dtype == "F16") {
        const uint16_t* buf = reinterpret_cast<const uint16_t*>(ptr);
        for (size_t i = 0; i < row_size; ++i) {
            result[i] = f16_to_float(buf[i]);
        }
    }
    
    return result;
}

// Read a row from a 3D tensor: tensor[slice_idx, row, :]
// For experts.gate_up_proj [128, 5120, 16384]: slice_idx=expert, row=input_dim
std::vector<float> read_tensor_slice(const TensorMeta& meta, size_t slice_idx, size_t row) {
    if (meta.shape.size() < 3) return read_tensor_row(meta, row);  // Fallback to 2D
    
    const MappedFile* mf = get_mapped_file(meta.shard_file);
    if (!mf || !mf->data()) return {};
    
    uint64_t header_size;
    std::memcpy(&header_size, mf->data(), 8);
    
    size_t dim1 = static_cast<size_t>(meta.shape[1]);  // rows per slice
    size_t dim2 = static_cast<size_t>(meta.shape[2]);  // cols
    size_t bytes_per_elem = 4;
    if (meta.dtype == "BF16" || meta.dtype == "F16") bytes_per_elem = 2;
    
    // Offset: skip to slice, then to row within slice
    size_t slice_size = dim1 * dim2 * bytes_per_elem;
    size_t row_size_bytes = dim2 * bytes_per_elem;
    size_t offset = 8 + header_size + meta.data_offset_start + 
                    slice_idx * slice_size + row * row_size_bytes;
    
    if (offset + row_size_bytes > mf->size()) return {};
    
    const uint8_t* ptr = mf->data() + offset;
    std::vector<float> result(dim2);
    
    if (meta.dtype == "F32") {
        std::memcpy(result.data(), ptr, dim2 * 4);
    } else if (meta.dtype == "BF16") {
        const uint16_t* buf = reinterpret_cast<const uint16_t*>(ptr);
        for (size_t i = 0; i < dim2; ++i) {
            result[i] = bf16_to_float(buf[i]);
        }
    } else if (meta.dtype == "F16") {
        const uint16_t* buf = reinterpret_cast<const uint16_t*>(ptr);
        for (size_t i = 0; i < dim2; ++i) {
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
 * PARALLELIZED: Read all lines first, then compute compositions in parallel
 */
bool parse_vocab(const fs::path& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file) {
        std::cerr << "Cannot open vocab: " << vocab_path << "\n";
        return false;
    }
    
    // Phase 1: Read all lines sequentially (I/O bound)
    std::vector<std::string> lines;
    lines.reserve(50000);  // Pre-allocate for typical vocab size
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            lines.push_back(std::move(line));
        }
    }
    
    size_t total = lines.size();
    g_vocab_tokens.resize(total);
    
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
            g_vocab_tokens[i] = std::move(info);
            
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
                std::cerr << "  [VOCAB] " << c << "/" << total << " (" << std::fixed << std::setprecision(0) << rate << " tok/s)\r" << std::flush;
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
        g_token_to_idx[g_vocab_tokens[i].text] = i;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "\n[VOCAB] Loaded " << total << " tokens in " << elapsed << "ms using " << num_threads << " threads\n";
    return true;
}

// Lambda for writing a double as little-endian hex
inline void write_double_hex(std::string& out, double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    char hex[17];
    for (int i = 0; i < 8; ++i) {
        snprintf(hex + i * 2, 3, "%02x", static_cast<unsigned>(bits & 0xFF));
        bits >>= 8;
    }
    out += hex;
}

// Helper to build LINESTRINGZM EWKB hex from Point4D vector (for vocab compositions)
// Point4D stores uint32 coordinates - store them directly as doubles
std::string build_composition_linestringzm_ewkb(const std::vector<Point4D>& points) {
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
    // Point4D.x etc are uint32 but represent signed values via offset
    // Store raw uint32 as doubles (PostGIS double has 53-bit mantissa, plenty for 32-bit)
    for (const auto& pt : points) {
        write_double_hex(ewkb, static_cast<double>(pt.x));
        write_double_hex(ewkb, static_cast<double>(pt.y));
        write_double_hex(ewkb, static_cast<double>(pt.z));
        write_double_hex(ewkb, static_cast<double>(pt.m));
    }
    
    return ewkb;
}

// Helper to build POINTZM EWKB hex from Point4D (for composition centroid)
std::string build_composition_pointzm_ewkb(const Point4D& pt) {
    std::string ewkb;
    ewkb.reserve(74);
    
    // Header: little-endian (01), type=POINTZM with SRID (010000e0), SRID=0 (00000000)
    ewkb += "01";           // Little-endian
    ewkb += "010000e0";     // POINTZM with SRID flag
    ewkb += "00000000";     // SRID = 0
    
    // Store coordinates as raw uint32 values converted to double
    write_double_hex(ewkb, static_cast<double>(pt.x));
    write_double_hex(ewkb, static_cast<double>(pt.y));
    write_double_hex(ewkb, static_cast<double>(pt.z));
    write_double_hex(ewkb, static_cast<double>(pt.m));
    
    return ewkb;
}

/**
 * Insert vocab token compositions into the composition + composition_child tables
 * NEW SCHEMA: composition stores the aggregations, composition_child stores ordered children
 * NOW INCLUDES: geom, centroid, hilbert_lo, hilbert_hi computed client-side
 * PARALLELIZED: Build batch strings in parallel, stream to DB
 */
bool insert_compositions(PGconn* conn) {
    if (g_vocab_tokens.empty()) return true;
    
    size_t total = g_vocab_tokens.size();
    std::cerr << "[COMP] Inserting " << total << " token compositions...\n";
    
    // Count compositions (skip single-char tokens)
    size_t comp_count = 0;
    for (const auto& token : g_vocab_tokens) {
        if (token.comp.children.size() > 1) comp_count++;
    }
    std::cerr << "[COMP] " << comp_count << " multi-char compositions to insert\n";
    
    // Phase 1: Build batch strings in parallel
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 16) num_threads = 16;
    
    std::cerr << "[COMP] Building batch strings with " << num_threads << " threads...\n";
    
    // Thread-local buffers for compositions and children
    std::vector<std::string> comp_batches(num_threads);
    std::vector<std::string> child_batches(num_threads);
    for (auto& b : comp_batches) b.reserve(1 << 20);
    for (auto& b : child_batches) b.reserve(1 << 20);
    
    std::atomic<size_t> idx{0};
    std::atomic<size_t> processed{0};
    auto start = std::chrono::steady_clock::now();
    
    // Progress reporter
    std::atomic<bool> done{false};
    std::thread progress_thread([&]() {
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            size_t p = processed.load();
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            double rate = (elapsed_ms > 0) ? (p * 1000.0 / elapsed_ms) : 0;
            std::cerr << "  [BUILD] " << p << "/" << total << " tokens (" 
                      << std::fixed << std::setprecision(0) << rate << "/s)\r" << std::flush;
        }
    });
    
    auto worker = [&](unsigned tid) {
        auto& comp_batch = comp_batches[tid];
        auto& child_batch = child_batches[tid];
        
        while (true) {
            size_t i = idx.fetch_add(1);
            if (i >= total) break;
            
            const auto& token = g_vocab_tokens[i];
            const auto& c = token.comp;
            
            processed.fetch_add(1);
            
            // Skip single-char tokens (they're just atoms, already seeded)
            if (c.children.size() <= 1) continue;
            
            // Build composition row
            comp_batch += "\\\\x";
            comp_batch += c.hash.to_hex();
            comp_batch += "\t";
            
            // label (the token text, escaped for COPY)
            for (char ch : token.text) {
                if (ch == '\t') comp_batch += "\\t";
                else if (ch == '\n') comp_batch += "\\n";
                else if (ch == '\\') comp_batch += "\\\\";
                else comp_batch += ch;
            }
            comp_batch += "\t";
            
            // depth, child_count, atom_count
            comp_batch += std::to_string(c.depth);
            comp_batch += "\t";
            comp_batch += std::to_string(c.children.size());
            comp_batch += "\t";
            comp_batch += std::to_string(c.atom_count);
            comp_batch += "\t";
            
            // geom (LINESTRINGZM from child coordinates)
            std::string geom_ewkb = build_composition_linestringzm_ewkb(c.child_coords);
            if (!geom_ewkb.empty()) {
                comp_batch += geom_ewkb;
            } else {
                comp_batch += "\\N";
            }
            comp_batch += "\t";
            
            // centroid (POINTZM)
            std::string centroid_ewkb = build_composition_pointzm_ewkb(c.centroid);
            comp_batch += centroid_ewkb;
            comp_batch += "\t";
            
            // hilbert_lo, hilbert_hi
            comp_batch += std::to_string(static_cast<int64_t>(c.hilbert.lo));
            comp_batch += "\t";
            comp_batch += std::to_string(static_cast<int64_t>(c.hilbert.hi));
            comp_batch += "\n";
            
            // Build child rows
            for (size_t j = 0; j < c.children.size(); ++j) {
                child_batch += "\\\\x";
                child_batch += c.hash.to_hex();
                child_batch += "\t";
                child_batch += std::to_string(j);
                child_batch += "\tA\t\\\\x";
                child_batch += c.children[j].to_hex();
                child_batch += "\n";
            }
        }
    };
    
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back(worker, t);
    }
    for (auto& th : workers) th.join();
    done.store(true);
    progress_thread.join();
    
    auto build_end = std::chrono::steady_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - start).count();
    
    // Calculate total batch sizes
    size_t comp_total = 0, child_total = 0;
    for (const auto& b : comp_batches) comp_total += b.size();
    for (const auto& b : child_batches) child_total += b.size();
    
    std::cerr << "\n[COMP] Built " << (comp_total / 1024) << "KB compositions + " 
              << (child_total / 1024) << "KB children in " << build_ms << "ms\n";
    std::cerr << "[COMP] Streaming to database...\n";
    
    // Phase 2: Stream to database
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Temp table for compositions WITH GEOMETRY COLUMNS
    std::cerr << "[COMP] Creating temp tables...\n";
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_comp ("
        "  id BYTEA,"
        "  label TEXT,"
        "  depth INTEGER,"
        "  child_count INTEGER,"
        "  atom_count BIGINT,"
        "  geom GEOMETRY(LINESTRINGZM, 0),"
        "  centroid GEOMETRY(POINTZM, 0),"
        "  hilbert_lo BIGINT,"
        "  hilbert_hi BIGINT"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[COMP] Create tmp_comp failed: " << PQerrorMessage(conn) << "\n";
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
        std::cerr << "[COMP] Create tmp_comp_child failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // COPY compositions
    std::cerr << "[COMP] Copying compositions to temp table...\n";
    res = PQexec(conn, "COPY tmp_comp FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "[COMP] COPY comp start failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    for (size_t i = 0; i < comp_batches.size(); ++i) {
        if (!comp_batches[i].empty()) {
            std::cerr << "  [COPY] Batch " << (i+1) << "/" << comp_batches.size() 
                      << " (" << (comp_batches[i].size()/1024) << "KB)\r" << std::flush;
            PQputCopyData(conn, comp_batches[i].c_str(), static_cast<int>(comp_batches[i].size()));
        }
    }
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "\n[COMP] COPY comp failed: " << PQerrorMessage(conn) << "\n";
    }
    PQclear(res);
    std::cerr << "\n";
    
    // COPY composition children
    std::cerr << "[COMP] Copying composition children to temp table...\n";
    res = PQexec(conn, "COPY tmp_comp_child FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "[COMP] COPY comp_child start failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        return false;
    }
    PQclear(res);
    
    for (size_t i = 0; i < child_batches.size(); ++i) {
        if (!child_batches[i].empty()) {
            std::cerr << "  [COPY] Batch " << (i+1) << "/" << child_batches.size() 
                      << " (" << (child_batches[i].size()/1024) << "KB)\r" << std::flush;
            PQputCopyData(conn, child_batches[i].c_str(), static_cast<int>(child_batches[i].size()));
        }
    }
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "\n[COMP] COPY child failed: " << PQerrorMessage(conn) << "\n";
    }
    PQclear(res);
    std::cerr << "\n";

    // Insert compositions WITH geometry columns
    std::cerr << "[COMP] Inserting into composition table...\n";
    res = PQexec(conn,
        "INSERT INTO composition (id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi) "
        "SELECT id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi "
        "FROM tmp_comp "
        "ON CONFLICT (id) DO UPDATE SET "
        "  geom = EXCLUDED.geom, "
        "  centroid = EXCLUDED.centroid, "
        "  hilbert_lo = EXCLUDED.hilbert_lo, "
        "  hilbert_hi = EXCLUDED.hilbert_hi "
        "WHERE composition.geom IS NULL");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[COMP] Composition insert failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    int inserted_comps = atoi(PQcmdTuples(res));
    PQclear(res);
    std::cerr << "[COMP] Inserted " << inserted_comps << " compositions\n";
    
    // Insert composition children (only for compositions that exist)
    std::cerr << "[COMP] Inserting into composition_child table...\n";
    res = PQexec(conn,
        "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
        "SELECT composition_id, ordinal, child_type, child_id "
        "FROM tmp_comp_child "
        "WHERE EXISTS (SELECT 1 FROM composition WHERE id = tmp_comp_child.composition_id) "
        "ON CONFLICT (composition_id, ordinal) DO NOTHING");
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[COMP] Composition child insert failed: " << PQerrorMessage(conn) << "\n";
        PQclear(res);
        PQexec(conn, "ROLLBACK");
        return false;
    }
    int inserted_children = atoi(PQcmdTuples(res));
    PQclear(res);
    std::cerr << "[COMP] Inserted " << inserted_children << " children\n";
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    auto end = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[COMP] Inserted " << inserted_comps << " compositions, " << inserted_children << " children in " << total_ms << "ms\n";
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
        } else if (arg == "-p" && i + 1 < argc) {
            config.conninfo += " port=" + std::string(argv[++i]);
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
        std::cerr << "Usage: ingest_safetensor [-d db] [-U user] [-h host] [-p port] [-n model_name] [-t threshold] <model_dir>\n";
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
        std::string path_str = entry.path().string();
        
        // Skip hidden directories and cache folders
        if (path_str.find("\\.") != std::string::npos || 
            path_str.find("/.") != std::string::npos ||
            path_str.find(".cache") != std::string::npos) {
            continue;
        }
        
        if (name == "vocab.txt") vocab_path = entry.path();
        else if (name == "tokenizer.json") tokenizer_path = entry.path();
        else if (name == "model.safetensors.index.json") index_path = entry.path();
        else if (name.ends_with(".safetensors")) {  // Must END with .safetensors
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
            std::cerr << "  Parsing: " << f << "\n";
            if (!parse_safetensor_header(f)) {
                std::cerr << "  [ERROR] Failed to parse: " << f << "\n";
                return 1;
            }
        }
    }
    
    if (g_tensors.empty()) {
        std::cerr << "[ERROR] No tensors found!\n";
        return 1;
    }
    
    std::cerr << "[INFO] Found " << g_tensors.size() << " tensors\n";
    
    // Connect to database
    PGconn* conn = PQconnectdb(config.conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return 1;
    }
    
    // === BUILD THE COMPOSITION HIERARCHY ===
    // This is the structural skeleton of the model
    
    // Step 4: Insert tensor hierarchy as compositions
    // "encoder.layer.0.attention.self.query.weight" becomes a tree
    std::cerr << "\n[4] Building tensor name hierarchy...\n";
    if (!g_tensors.empty()) {
        insert_tensor_hierarchy(conn, config);
    }
    
    // Step 5: Insert vocab token compositions (BPE tokens)
    if (!g_vocab_tokens.empty()) {
        std::cerr << "\n[5] Inserting token compositions...\n";
        insert_compositions(conn);
    }
    
    // Step 6: Compute composition centroids FROM CHILDREN
    // - Leaf compositions with atom children: centroid = avg of atom coords
    // - Parent compositions: centroid = avg of child composition centroids
    // YOUR coordinate system: atoms have deterministic 4D coords from Unicode→Hilbert→S³
    std::cerr << "\n[6] Computing composition centroids hierarchically...\n";
    {
        // First pass: compositions with atom children
        PGresult* res = PQexec(conn, "SELECT recompute_composition_centroids()");
        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            std::cerr << "[CENTROID] Failed: " << PQerrorMessage(conn) << "\n";
        } else {
            int updated = atoi(PQgetvalue(res, 0, 0));
            std::cerr << "[CENTROID] Updated " << updated << " composition centroids from atoms\n";
        }
        PQclear(res);
        
        // Second pass: compositions with composition children (hierarchical)
        // This propagates centroids up the tree
        res = PQexec(conn,
            "WITH RECURSIVE comp_tree AS ("
            "  SELECT id, centroid, 1 as level FROM composition WHERE centroid IS NOT NULL "
            "  UNION ALL "
            "  SELECT c.id, "
            "    st_centroid_4d(ST_Collect(child.centroid)), "
            "    ct.level + 1 "
            "  FROM composition c "
            "  JOIN composition_child cc ON cc.composition_id = c.id AND cc.child_type = 'C' "
            "  JOIN comp_tree ct ON ct.id = cc.child_id "
            "  JOIN composition child ON child.id = cc.child_id "
            "  WHERE c.centroid IS NULL AND child.centroid IS NOT NULL "
            "  GROUP BY c.id "
            ") "
            "UPDATE composition SET centroid = comp_tree.centroid "
            "FROM comp_tree WHERE composition.id = comp_tree.id");
        
        int hier_updated = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
        if (PQresultStatus(res) == PGRES_COMMAND_OK) {
            std::cerr << "[CENTROID] Updated " << hier_updated << " hierarchical composition centroids\n";
        }
        PQclear(res);
    }
    
    // === EXTRACT RELATIONS (MODEL-SPECIFIC EDGES) ===
    
    // Step 7: Extract k-NN similarity edges from model embeddings as RELATIONS
    // Embeddings are model-specific addresses - extract SIMILARITY GRAPH as edges
    // This does NOT modify composition centroids - those come from atoms only
    std::cerr << "\n[7] Extracting embedding k-NN similarity as relations...\n";
    if (!g_tensors.empty()) {
        extract_embedding_relations(conn, config);
    } else {
        std::cerr << "[EMBED] No tensors found, skipping relation extraction\n";
    }
    
    // Step 8: Extract router/attention relations (MoE models, attention weights)
    std::cerr << "\n[8] Extracting weight-based relations (router, attention, MLP)...\n";
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
// Insert Tensor Name Hierarchy as Compositions
// ============================================================================
// Builds the hierarchical composition tree from tensor names:
//   "encoder.layer.0.attention.self.query.weight" becomes:
//     encoder (depth=1)
//       └─ encoder.layer (depth=2)
//           └─ encoder.layer.0 (depth=3)
//               └─ encoder.layer.0.attention (depth=4)
//                   └─ encoder.layer.0.attention.self (depth=5)
//                       └─ encoder.layer.0.attention.self.query (depth=6)
//                           └─ encoder.layer.0.attention.self.query.weight (depth=7)
//
// This is the structural skeleton of the model - the backbone for:
//   - correct centroid computation at every level
//   - correct semantic grouping
//   - model navigation and comparison
//   - cross-model structural alignment

bool insert_tensor_hierarchy(PGconn* conn, const IngestConfig& config) {
    if (g_tensors.empty()) return true;
    
    std::cerr << "[HIER] Building tensor hierarchy from " << g_tensors.size() << " tensors\n";
    
    // Collect all unique path components across all tensors
    std::unordered_map<std::string, int> path_to_depth;  // path -> depth
    std::unordered_map<std::string, std::string> path_to_parent;  // path -> parent path
    
    for (const auto& [tensor_name, meta] : g_tensors) {
        std::vector<std::string> components = split_tensor_path(tensor_name);
        
        std::string parent_path;
        for (size_t i = 0; i < components.size(); ++i) {
            const std::string& path = components[i];
            int depth = static_cast<int>(i) + 1;
            
            // Record this path if not seen or update depth if deeper
            auto it = path_to_depth.find(path);
            if (it == path_to_depth.end()) {
                path_to_depth[path] = depth;
                if (!parent_path.empty()) {
                    path_to_parent[path] = parent_path;
                }
            }
            
            parent_path = path;
        }
    }
    
    std::cerr << "[HIER] Found " << path_to_depth.size() << " unique hierarchy nodes\n";
    
    // Build composition batch
    std::string comp_batch;
    comp_batch.reserve(path_to_depth.size() * 256);
    
    // Build parent->child edge batch
    std::string child_batch;
    child_batch.reserve(path_to_depth.size() * 128);
    
    // Track full composition records for linking and atom children
    std::unordered_map<std::string, CompositionRecord> path_to_comp;
    
    // Batch for atom children (the characters of each path)
    std::string atom_child_batch;
    atom_child_batch.reserve(path_to_depth.size() * 256);  // Avg ~20 chars per path
    
    for (const auto& [path, depth] : path_to_depth) {
        // Compute FULL composition from path - includes atom children
        CompositionRecord comp = AtomCalculator::compute_vocab_token(path);
        path_to_comp[path] = comp;
        
        // Build composition row with geometry:
        // id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi
        comp_batch += "\\\\x";
        comp_batch += comp.hash.to_hex();
        comp_batch += "\t";
        
        // Escape path for COPY
        for (char ch : path) {
            if (ch == '\t') comp_batch += "\\t";
            else if (ch == '\n') comp_batch += "\\n";
            else if (ch == '\\') comp_batch += "\\\\";
            else comp_batch += ch;
        }
        comp_batch += "\t";
        comp_batch += std::to_string(depth);
        comp_batch += "\t";
        comp_batch += std::to_string(comp.children.size());  // child_count = atom count
        comp_batch += "\t";
        comp_batch += std::to_string(comp.atom_count);
        comp_batch += "\t";
        
        // geom (LINESTRINGZM from child coordinates)
        std::string geom_ewkb = build_composition_linestringzm_ewkb(comp.child_coords);
        if (!geom_ewkb.empty()) {
            comp_batch += geom_ewkb;
        } else {
            comp_batch += "\\N";
        }
        comp_batch += "\t";
        
        // centroid (POINTZM)
        std::string centroid_ewkb = build_composition_pointzm_ewkb(comp.centroid);
        comp_batch += centroid_ewkb;
        comp_batch += "\t";
        
        // hilbert_lo, hilbert_hi
        comp_batch += std::to_string(static_cast<int64_t>(comp.hilbert.lo));
        comp_batch += "\t";
        comp_batch += std::to_string(static_cast<int64_t>(comp.hilbert.hi));
        comp_batch += "\n";
        
        // Build atom children - each character in the path is an atom child
        for (size_t j = 0; j < comp.children.size(); ++j) {
            atom_child_batch += "\\\\x";
            atom_child_batch += comp.hash.to_hex();
            atom_child_batch += "\t";
            atom_child_batch += std::to_string(j);
            atom_child_batch += "\tA\t\\\\x";  // 'A' = atom child
            atom_child_batch += comp.children[j].to_hex();
            atom_child_batch += "\n";
        }
    }
    
    std::cerr << "[HIER] Built " << path_to_comp.size() << " compositions with atom children\n";
    
    // Track how many composition children each parent has (for ordinal tracking)
    std::unordered_map<std::string, size_t> parent_child_count;
    
    // Now build parent->child edges (composition -> composition)
    size_t edge_count = 0;
    for (const auto& [path, parent_path] : path_to_parent) {
        auto child_it = path_to_comp.find(path);
        auto parent_it = path_to_comp.find(parent_path);
        
        if (child_it != path_to_comp.end() && parent_it != path_to_comp.end()) {
            // Parent composition -> child composition
            // Format: composition_id, ordinal, child_type, child_id
            // Ordinal offset by parent's atom count so it doesn't collide with atom children
            size_t atom_count = parent_it->second.children.size();  // Number of atoms in parent's path
            size_t comp_ordinal = atom_count + parent_child_count[parent_path];
            parent_child_count[parent_path]++;
            
            child_batch += "\\\\x";
            child_batch += parent_it->second.hash.to_hex();  // parent's id
            child_batch += "\t";
            child_batch += std::to_string(comp_ordinal);  // ordinal offset past atoms
            child_batch += "\tC\t\\\\x";  // 'C' = composition child
            child_batch += child_it->second.hash.to_hex();  // child's id
            child_batch += "\n";
            edge_count++;
        }
    }
    
    std::cerr << "[HIER] Built " << edge_count << " composition->composition edges\n";
    
    // Stream to database
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    // Create temp table for compositions WITH GEOMETRY
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_hier_comp ("
        "  id BYTEA,"
        "  label TEXT,"
        "  depth INTEGER,"
        "  child_count INTEGER,"
        "  atom_count BIGINT,"
        "  geom GEOMETRY(LINESTRINGZM, 0),"
        "  centroid GEOMETRY(POINTZM, 0),"
        "  hilbert_lo BIGINT,"
        "  hilbert_hi BIGINT"
        ") ON COMMIT DROP");
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[HIER] Create temp table failed: " << PQerrorMessage(conn) << "\n";
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    // COPY compositions
    res = PQexec(conn, "COPY tmp_hier_comp FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "[HIER] COPY start failed: " << PQerrorMessage(conn) << "\n";
        PQexec(conn, "ROLLBACK");
        return false;
    }
    PQclear(res);
    
    if (!comp_batch.empty()) {
        PQputCopyData(conn, comp_batch.c_str(), static_cast<int>(comp_batch.size()));
    }
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);
    
    // Insert into composition table WITH GEOMETRY
    res = PQexec(conn,
        "INSERT INTO composition (id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi) "
        "SELECT id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi FROM tmp_hier_comp "
        "ON CONFLICT (id) DO UPDATE SET "
        "  label = EXCLUDED.label, "
        "  depth = GREATEST(composition.depth, EXCLUDED.depth), "
        "  geom = COALESCE(EXCLUDED.geom, composition.geom), "
        "  centroid = COALESCE(EXCLUDED.centroid, composition.centroid), "
        "  hilbert_lo = COALESCE(EXCLUDED.hilbert_lo, composition.hilbert_lo), "
        "  hilbert_hi = COALESCE(EXCLUDED.hilbert_hi, composition.hilbert_hi)");
    
    int comp_inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "[HIER] Insert compositions failed: " << PQerrorMessage(conn) << "\n";
    }
    PQclear(res);
    
    std::cerr << "[HIER] Inserted/updated " << comp_inserted << " hierarchy compositions\n";
    
    // Insert ATOM children FIRST (the characters that make up each hierarchy path)
    // These get ordinals 0..N-1 where N is the path length
    if (!atom_child_batch.empty()) {
        res = PQexec(conn,
            "CREATE TEMP TABLE tmp_hier_atom_child ("
            "  composition_id BYTEA,"
            "  ordinal SMALLINT,"
            "  child_type CHAR(1),"
            "  child_id BYTEA"
            ") ON COMMIT DROP");
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Create atom child temp table failed: " << PQerrorMessage(conn) << "\n";
        }
        PQclear(res);
        
        res = PQexec(conn, "COPY tmp_hier_atom_child FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        PQclear(res);
        
        PQputCopyData(conn, atom_child_batch.c_str(), static_cast<int>(atom_child_batch.size()));
        PQputCopyEnd(conn, nullptr);
        res = PQgetResult(conn);
        PQclear(res);
        
        // Insert atom children
        res = PQexec(conn,
            "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
            "SELECT composition_id, ordinal, child_type, child_id FROM tmp_hier_atom_child "
            "ON CONFLICT (composition_id, ordinal) DO NOTHING");
        
        int atom_edges = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Insert atom children failed: " << PQerrorMessage(conn) << "\n";
        }
        PQclear(res);
        
        std::cerr << "[HIER] Inserted " << atom_edges << " atom children\n";
    }
    
    // Now insert parent->child composition edges
    // These get ordinals N..N+M-1 where N is path length and M is number of sub-compositions
    if (!child_batch.empty()) {
        res = PQexec(conn,
            "CREATE TEMP TABLE tmp_hier_child ("
            "  composition_id BYTEA,"
            "  ordinal SMALLINT,"
            "  child_type CHAR(1),"
            "  child_id BYTEA"
            ") ON COMMIT DROP");
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Create child temp table failed: " << PQerrorMessage(conn) << "\n";
        }
        PQclear(res);
        
        res = PQexec(conn, "COPY tmp_hier_child FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        PQclear(res);
        
        PQputCopyData(conn, child_batch.c_str(), static_cast<int>(child_batch.size()));
        PQputCopyEnd(conn, nullptr);
        res = PQgetResult(conn);
        PQclear(res);
        
        // Insert composition->composition edges using pre-computed ordinals (offset past atoms)
        res = PQexec(conn,
            "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
            "SELECT composition_id, ordinal, child_type, child_id "
            "FROM tmp_hier_child "
            "ON CONFLICT (composition_id, ordinal) DO NOTHING");
        
        int edges_inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "[HIER] Insert composition edges failed: " << PQerrorMessage(conn) << "\n";
        }
        PQclear(res);
        
        std::cerr << "[HIER] Inserted " << edges_inserted << " composition->composition edges\n";
    }
    
    // Update child counts on parent compositions
    res = PQexec(conn,
        "UPDATE composition c SET child_count = sub.cnt "
        "FROM (SELECT composition_id, COUNT(*) as cnt FROM composition_child GROUP BY composition_id) sub "
        "WHERE c.id = sub.composition_id");
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    return true;
}

// ============================================================================
// Extract Embedding k-NN Similarity as Relations
// ============================================================================
// Builds k-NN similarity graph from model embeddings and inserts edges as
// relation rows. This extracts the SIMILARITY STRUCTURE from model embeddings
// WITHOUT overwriting composition centroids (those come from atoms only).
//
// Architecture:
//   - Atoms: Unicode codepoints with YOUR deterministic 4D coordinates
//   - Compositions: centroids = average of atom children (computed separately)
//   - Relations: edges from various sources including embedding k-NN similarity
//
// This function handles: generation models, embedding models, vision encoders,
// object detection models - ANY model with token/patch embeddings.
// ============================================================================

bool extract_embedding_relations(PGconn* conn, const IngestConfig& config) {
    // Find embedding tensors - support multiple model architectures
    std::vector<std::pair<std::string, TensorMeta*>> embed_tensors;
    
    for (auto& [name, meta] : g_tensors) {
        // Language model embeddings (semantic - high cosine values)
        if (name.find("embed_tokens") != std::string::npos ||
            name.find("word_embeddings") != std::string::npos ||
            name.find("wte.weight") != std::string::npos ||
            name.find("token_embedding") != std::string::npos) {
            embed_tensors.emplace_back("token", &meta);
        }
        // Vision model patch embeddings (moderate cosine values)
        else if (name.find("patch_embed") != std::string::npos ||
                 name.find("patch_embedding") != std::string::npos ||
                 name.find("proj.weight") != std::string::npos && name.find("patch") != std::string::npos) {
            embed_tensors.emplace_back("patch", &meta);
        }
        // CLIP/multimodal projections (moderate cosine values)
        else if (name.find("text_projection") != std::string::npos ||
                 name.find("visual_projection") != std::string::npos) {
            embed_tensors.emplace_back("projection", &meta);
        }
        // Position embeddings (near-orthogonal by design - very low cosine values)
        else if (name.find("position_embed") != std::string::npos ||
                 name.find("pos_embed") != std::string::npos ||
                 name.find("query_position") != std::string::npos) {
            embed_tensors.emplace_back("position", &meta);
        }
    }
    
    if (embed_tensors.empty()) {
        std::cerr << "[EMBED] No embedding tensors found\n";
        return true;
    }
    
    std::cerr << "[EMBED] Found " << embed_tensors.size() << " embedding tensor(s)\n";
    
    // Per-embedding-type thresholds
    auto get_threshold = [&](const std::string& embed_type) -> float {
        if (embed_type == "token") return 0.45f;
        if (embed_type == "patch") return 0.25f;
        if (embed_type == "position") return 0.02f;
        if (embed_type == "projection") return 0.15f;
        return 0.30f;
    };
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 16) num_threads = 16;
    
    auto total_start = std::chrono::steady_clock::now();
    
    // =========================================================================
    // PHASE 1: Process all embedding tensors in parallel, accumulate edges
    // =========================================================================
    std::mutex batch_mutex;
    std::string global_batch;
    global_batch.reserve(16 << 20);  // 16MB initial
    std::atomic<size_t> total_edges{0};
    
    auto process_embedding = [&](const std::string& embed_type, TensorMeta* embed) {
        if (embed->shape.size() < 2) return;
        
        int64_t num_items = embed->shape[0];
        int64_t embed_dim = embed->shape[1];
        float threshold = get_threshold(embed_type);
        
        if (embed_type == "token" && !g_vocab_tokens.empty()) {
            num_items = std::min(num_items, static_cast<int64_t>(g_vocab_tokens.size()));
        }
        
        std::cerr << "[EMBED] Processing " << embed->name << " [" << embed_type << ", thresh=" << threshold << "]: " 
                  << num_items << " x " << embed_dim << " dims\n";
        
        auto start = std::chrono::steady_clock::now();
        
        // Read embeddings
        std::vector<std::vector<float>> embeddings(num_items);
        std::atomic<int64_t> read_idx{0};
        auto read_worker = [&]() {
            while (true) {
                int64_t i = read_idx.fetch_add(1);
                if (i >= num_items) break;
                embeddings[i] = read_tensor_row(*embed, static_cast<size_t>(i));
            }
        };
        std::vector<std::thread> readers;
        for (unsigned t = 0; t < num_threads; ++t) readers.emplace_back(read_worker);
        for (auto& th : readers) th.join();
        
        auto read_end = std::chrono::steady_clock::now();
        auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - start).count();
        std::cerr << "[EMBED] Read embeddings in " << read_ms << "ms\n";
        
        // Build k-NN using HNSWLIB
        const int k_neighbors = 15;
        std::vector<std::vector<std::tuple<size_t, size_t, float>>> thread_edges(num_threads);
        
#ifdef HAS_HNSWLIB
        hnswlib::InnerProductSpace space(embed_dim);
        hnswlib::HierarchicalNSW<float> hnsw(&space, num_items, 16, 200);
        hnsw.setEf(50);
        
        // Normalize for cosine similarity
        std::vector<std::vector<float>> normalized(num_items);
        for (int64_t i = 0; i < num_items; ++i) {
            if (embeddings[i].empty()) continue;
            normalized[i].resize(embed_dim);
            float norm = 0;
            for (size_t d = 0; d < static_cast<size_t>(embed_dim); ++d) 
                norm += embeddings[i][d] * embeddings[i][d];
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (size_t d = 0; d < static_cast<size_t>(embed_dim); ++d) 
                    normalized[i][d] = embeddings[i][d] / norm;
            }
        }
        
        // Parallel index construction
        std::atomic<int64_t> add_idx{0};
        auto add_worker = [&]() {
            while (true) {
                int64_t i = add_idx.fetch_add(1);
                if (i >= num_items) break;
                if (normalized[i].empty()) continue;
                hnsw.addPoint(normalized[i].data(), i);
            }
        };
        std::vector<std::thread> adders;
        for (unsigned t = 0; t < num_threads; ++t) adders.emplace_back(add_worker);
        for (auto& th : adders) th.join();
        
        // Parallel k-NN queries
        std::atomic<int64_t> knn_idx{0};
        auto knn_worker = [&](unsigned tid) {
            auto& local_edges = thread_edges[tid];
            while (true) {
                int64_t i = knn_idx.fetch_add(1);
                if (i >= num_items) break;
                if (normalized[i].empty()) continue;
                
                auto result = hnsw.searchKnn(normalized[i].data(), k_neighbors + 1);
                while (!result.empty()) {
                    auto [dist, j] = result.top();
                    result.pop();
                    if (static_cast<int64_t>(j) == i) continue;
                    float sim = 1.0f - dist;
                    if (sim >= threshold && static_cast<size_t>(i) < j) {
                        local_edges.emplace_back(static_cast<size_t>(i), j, sim);
                    }
                }
            }
        };
        std::vector<std::thread> knn_workers;
        for (unsigned t = 0; t < num_threads; ++t) knn_workers.emplace_back(knn_worker, t);
        for (auto& th : knn_workers) th.join();
#endif
        
        auto knn_end = std::chrono::steady_clock::now();
        auto knn_ms = std::chrono::duration_cast<std::chrono::milliseconds>(knn_end - read_end).count();
        
        size_t edge_count = 0;
        for (const auto& edges : thread_edges) edge_count += edges.size();
        std::cerr << "[EMBED] Built k-NN graph: " << edge_count << " edges in " << knn_ms << "ms\n";
        
        if (edge_count == 0) {
            std::cerr << "[EMBED] No edges above threshold " << threshold << " for " << embed_type << "\n";
            return;
        }
        
        // Build batch string for this tensor
        std::string local_batch;
        local_batch.reserve(edge_count * 128);
        
        for (const auto& edges : thread_edges) {
            for (const auto& [i, j, sim] : edges) {
                Blake3Hash source_hash, target_hash;
                char source_type = 'C', target_type = 'C';
                
                if (embed_type == "token" && i < g_vocab_tokens.size() && j < g_vocab_tokens.size()) {
                    const auto& src = g_vocab_tokens[i];
                    const auto& tgt = g_vocab_tokens[j];
                    source_hash = src.comp.hash;
                    target_hash = tgt.comp.hash;
                    source_type = (src.comp.children.size() <= 1) ? 'A' : 'C';
                    target_type = (tgt.comp.children.size() <= 1) ? 'A' : 'C';
                } else {
                    std::string src_key = embed_type + ":" + std::to_string(i);
                    std::string tgt_key = embed_type + ":" + std::to_string(j);
                    source_hash = AtomCalculator::compute_vocab_token(src_key).hash;
                    target_hash = AtomCalculator::compute_vocab_token(tgt_key).hash;
                }
                
                local_batch += source_type;
                local_batch += "\t\\\\x" + source_hash.to_hex() + "\t";
                local_batch += target_type;
                local_batch += "\t\\\\x" + target_hash.to_hex() + "\t";
                local_batch += std::to_string(sim) + "\t";
                local_batch += embed_type + "\n";
            }
        }
        
        // Append to global batch under lock
        {
            std::lock_guard<std::mutex> lock(batch_mutex);
            global_batch += local_batch;
        }
        total_edges += edge_count;
    };
    
    // Process all embedding tensors (could parallelize but tensor I/O may bottleneck)
    for (auto& [embed_type, embed] : embed_tensors) {
        process_embedding(embed_type, embed);
    }
    
    auto process_end = std::chrono::steady_clock::now();
    auto process_ms = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - total_start).count();
    std::cerr << "[EMBED] Processed all tensors in " << process_ms << "ms, " << total_edges << " edges\n";
    
    if (total_edges == 0) {
        std::cerr << "[EMBED] No embedding relations to insert\n";
        return true;
    }
    
    // =========================================================================
    // PHASE 2: Single bulk insert of all accumulated edges
    // =========================================================================
    auto insert_start = std::chrono::steady_clock::now();
    
    PGresult* res = PQexec(conn, "BEGIN");
    PQclear(res);
    
    res = PQexec(conn,
        "CREATE TEMP TABLE tmp_embed_rel ("
        "  source_type CHAR(1), source_id BYTEA,"
        "  target_type CHAR(1), target_id BYTEA,"
        "  weight REAL, embed_type TEXT"
        ") ON COMMIT DROP");
    PQclear(res);
    
    res = PQexec(conn, "COPY tmp_embed_rel FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
    PQclear(res);
    
    // Stream in chunks to avoid memory issues
    const size_t chunk_size = 4 << 20;  // 4MB chunks
    for (size_t offset = 0; offset < global_batch.size(); offset += chunk_size) {
        size_t len = std::min(chunk_size, global_batch.size() - offset);
        PQputCopyData(conn, global_batch.data() + offset, static_cast<int>(len));
    }
    PQputCopyEnd(conn, nullptr);
    res = PQgetResult(conn);
    PQclear(res);
    
    // Bulk insert with UPSERT
    res = PQexec(conn,
        ("INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
         "SELECT source_type, source_id, target_type, target_id, 'E', weight, '" + config.model_name + "', 1, -1, embed_type "
         "FROM tmp_embed_rel "
         "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
         "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
         "  source_count = relation.source_count + 1").c_str());
    
    int inserted = (PQresultStatus(res) == PGRES_COMMAND_OK) ? atoi(PQcmdTuples(res)) : 0;
    PQclear(res);
    
    res = PQexec(conn, "COMMIT");
    PQclear(res);
    
    auto insert_end = std::chrono::steady_clock::now();
    auto insert_ms = std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - insert_start).count();
    
    std::cerr << "[EMBED] Bulk inserted " << inserted << " relations in " << insert_ms << "ms\n";
    std::cerr << "[EMBED] Total: " << total_edges.load() << " embedding similarity relations\n";
    return true;
}

// ============================================================================
// Insert Semantic Relations via Embedding Similarity
// ============================================================================
// Compute pairwise cosine similarity between token embeddings.
// Store sparse edges above threshold as semantic relations.
// This creates the ACTUAL knowledge graph - tokens connected by meaning.
// 
// NOTE: Cosine similarity is now provided by embedding::cosine_similarity()
// from hypercube/embedding_ops.hpp with automatic SIMD dispatch (AVX512/AVX2/SSE/scalar)

bool insert_attention_relations(PGconn* conn, const IngestConfig& config) {
    // ==========================================================================
    // EXTRACT SPARSE RELATIONS FROM MODEL TENSORS
    // - Router weights: token -> expert routing (MoE models)
    // - Token relationships emerge from Laplacian projection, not raw weights
    // ==========================================================================
    
    size_t total_edges = 0;
    
    // -------------------------------------------------------------------------
    // PART 1: Router weights for MoE models - sparse token-to-expert routing
    // Only extract router.weight tensors which define which experts handle which tokens
    // -------------------------------------------------------------------------
    std::vector<TensorMeta*> router_tensors;
    for (auto& [name, meta] : g_tensors) {
        if (name.find("router.weight") != std::string::npos && meta.shape.size() >= 2) {
            router_tensors.push_back(&meta);
        }
    }
    
    if (!router_tensors.empty()) {
        std::cerr << "[ROUTER] Found " << router_tensors.size() << " router tensors\n";
        
        for (auto* router : router_tensors) {
            // router.weight is [num_experts, hidden_dim]
            // Each row is an expert's routing vector
            int64_t num_experts = router->shape[0];
            int64_t hidden_dim = router->shape[1];
            
            // Parse layer number from tensor name
            int layer = -1;
            size_t layers_pos = router->name.find("layers.");
            if (layers_pos != std::string::npos) {
                size_t num_start = layers_pos + 7;
                size_t num_end = router->name.find(".", num_start);
                if (num_end != std::string::npos) {
                    layer = std::stoi(router->name.substr(num_start, num_end - num_start));
                }
            }
            
            std::cerr << "[ROUTER] " << router->name << " [" << num_experts << " experts x " << hidden_dim << " dims] layer=" << layer << "\n";
            
            // For each expert, find which tokens route to it above threshold
            // This creates expert atoms and token->expert edges
            PGresult* res = PQexec(conn, "BEGIN");
            PQclear(res);
            
            res = PQexec(conn,
                "CREATE TEMP TABLE tmp_router ("
                "  source_type CHAR(1), source_id BYTEA,"
                "  target_type CHAR(1), target_id BYTEA,"
                "  weight REAL, layer SMALLINT, component TEXT"
                ") ON COMMIT DROP");
            PQclear(res);
            
            res = PQexec(conn, "COPY tmp_router FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
            PQclear(res);
            
            std::string batch;
            batch.reserve(1 << 20);
            size_t router_edges = 0;
            
            for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                auto expert_row = read_tensor_row(*router, static_cast<size_t>(expert_idx));
                if (expert_row.empty()) continue;
                
                // Create expert atom hash
                std::string expert_key = "expert:" + std::to_string(layer) + ":" + std::to_string(expert_idx);
                auto expert_hash = AtomCalculator::compute_vocab_token(expert_key).hash;
                
                // Find significant routing weights (use lower threshold 0.1 for router)
                for (int64_t d = 0; d < hidden_dim && d < static_cast<int64_t>(g_vocab_tokens.size()); ++d) {
                    float weight = expert_row[d];
                    if (std::fabs(weight) >= 0.1f) {
                        // Create edge from token to expert
                        const auto& token = g_vocab_tokens[d];
                        char token_type = (token.comp.children.size() <= 1) ? 'A' : 'C';
                        
                        batch += token_type;
                        batch += "\t\\\\x" + token.comp.hash.to_hex() + "\t";
                        batch += "E\t\\\\x" + expert_hash.to_hex() + "\t";
                        batch += std::to_string(weight) + "\t";
                        batch += std::to_string(layer) + "\trouter\n";
                        router_edges++;
                        
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
                 "SELECT source_type, source_id, target_type, target_id, 'R', weight, '" + config.model_name + "', 1, layer, component FROM tmp_router "
                 "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
                 "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
                 "  source_count = relation.source_count + 1").c_str());
            PQclear(res);
            
            res = PQexec(conn, "COMMIT");
            PQclear(res);
            
            std::cerr << "  -> " << router_edges << " routing edges\n";
            total_edges += router_edges;
        }
    } else {
        std::cerr << "[ROUTER] No router tensors found (not an MoE model)\n";
    }
    
    // -------------------------------------------------------------------------
    // PART 2: Attention projections (Q/K/V/O) - token transformation relationships
    // These weights show how tokens relate through the attention mechanism:
    // - Q (query): what a token is looking for
    // - K (key): what a token offers to be found
    // - V (value): what information a token carries
    // - O (output): how attention results are projected back
    // 
    // We extract ROW SIMILARITY: rows that are similar = tokens that transform similarly
    // -------------------------------------------------------------------------
    std::cerr << "\n[ATTN] Extracting attention projection similarities...\n";
    
    struct TensorGroup {
        std::string component;
        std::vector<TensorMeta*> tensors;
    };
    std::vector<TensorGroup> attn_groups = {
        // Attention projections
        {"q_proj", {}}, {"k_proj", {}}, {"v_proj", {}}, {"o_proj", {}},
        {"query", {}}, {"key", {}}, {"value", {}},  // Alternative naming
        {"qkv_proj", {}},  // Fused QKV
        
        // FFN/MLP layers (LLaMA style)
        {"gate_proj", {}}, {"up_proj", {}}, {"down_proj", {}},
        
        // FFN/MLP layers (GPT style)
        {"fc1", {}}, {"fc2", {}}, {"fc_in", {}}, {"fc_out", {}},
        {"c_fc", {}}, {"c_proj", {}},  // GPT-2
        
        // Output projections
        {"lm_head", {}}, {"output", {}}, {"classifier", {}},
        
        // Vision transformers
        {"patch_embed", {}}, {"cls_token", {}},
    };
    
    for (auto& [name, meta] : g_tensors) {
        for (auto& group : attn_groups) {
            if (name.find(group.component + ".weight") != std::string::npos && meta.shape.size() == 2) {
                group.tensors.push_back(&meta);
            }
        }
    }
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 8) num_threads = 8;  // Limit for memory
    
    for (auto& group : attn_groups) {
        if (group.tensors.empty()) continue;
        
        std::cerr << "[ATTN] Processing " << group.tensors.size() << " " << group.component << " tensors\n";
        
        for (auto* tensor : group.tensors) {
            // Parse layer number
            int layer = -1;
            size_t layers_pos = tensor->name.find("layers.");
            if (layers_pos != std::string::npos) {
                size_t num_start = layers_pos + 7;
                size_t num_end = tensor->name.find(".", num_start);
                if (num_end != std::string::npos) {
                    layer = std::stoi(tensor->name.substr(num_start, num_end - num_start));
                }
            }
            
            int64_t out_dim = tensor->shape[0];
            int64_t in_dim = tensor->shape[1];
            
            // For large tensors, sample rows to keep computation tractable
            // We're looking for which OUTPUT dimensions are related
            int64_t max_rows = std::min(out_dim, static_cast<int64_t>(2048));
            int64_t stride = std::max(static_cast<int64_t>(1), out_dim / max_rows);
            
            std::cerr << "  " << tensor->name << " [" << out_dim << " x " << in_dim << "] layer=" << layer;
            if (stride > 1) std::cerr << " (sampling every " << stride << " rows)";
            std::cerr << "\n";
            
            // Read sampled rows
            std::vector<std::vector<float>> rows;
            std::vector<int64_t> row_indices;
            rows.reserve(max_rows);
            row_indices.reserve(max_rows);
            
            for (int64_t i = 0; i < out_dim; i += stride) {
                auto row = read_tensor_row(*tensor, static_cast<size_t>(i));
                if (!row.empty()) {
                    rows.push_back(std::move(row));
                    row_indices.push_back(i);
                }
            }
            
            if (rows.size() < 2) continue;
            
            // Build k-NN similarity for these weight rows
            // Similar rows = dimensions that behave similarly = related features
            const int k_neighbors = 10;
            std::vector<std::vector<std::tuple<size_t, size_t, float>>> thread_edges(num_threads);
            std::atomic<size_t> knn_idx{0};
            
            auto knn_worker = [&](unsigned tid) {
                auto& local_edges = thread_edges[tid];
                std::vector<std::pair<float, size_t>> neighbors;
                neighbors.reserve(rows.size());
                
                while (true) {
                    size_t i = knn_idx.fetch_add(1);
                    if (i >= rows.size()) break;
                    
                    neighbors.clear();
                    for (size_t j = 0; j < rows.size(); ++j) {
                        if (i == j) continue;
                        float sim = static_cast<float>(embedding::cosine_similarity(rows[i].data(), rows[j].data(), in_dim));
                        neighbors.emplace_back(sim, j);
                    }
                    
                    std::partial_sort(neighbors.begin(),
                                      neighbors.begin() + std::min(static_cast<size_t>(k_neighbors), neighbors.size()),
                                      neighbors.end(),
                                      [](auto& a, auto& b) { return a.first > b.first; });
                    
                    for (size_t k = 0; k < std::min(static_cast<size_t>(k_neighbors), neighbors.size()); ++k) {
                        float sim = neighbors[k].first;
                        size_t j = neighbors[k].second;
                        // Use lower threshold for weight matrices (0.15 vs 0.5 for embeddings)
                        if (sim >= 0.15f && i < j) {
                            local_edges.emplace_back(i, j, sim);
                        }
                    }
                }
            };
            
            std::vector<std::thread> workers;
            for (unsigned t = 0; t < num_threads; ++t) {
                workers.emplace_back(knn_worker, t);
            }
            for (auto& th : workers) th.join();
            
            // Count edges
            size_t edge_count = 0;
            for (const auto& edges : thread_edges) edge_count += edges.size();
            
            if (edge_count == 0) continue;
            
            // Insert edges as relations
            // These represent "dimension i and dimension j behave similarly in this layer"
            // This helps identify the "beaten path" - frequently co-activated features
            PGresult* res = PQexec(conn, "BEGIN");
            PQclear(res);
            
            res = PQexec(conn,
                "CREATE TEMP TABLE tmp_attn ("
                "  source_type CHAR(1), source_id BYTEA,"
                "  target_type CHAR(1), target_id BYTEA,"
                "  weight REAL, layer SMALLINT, component TEXT"
                ") ON COMMIT DROP");
            PQclear(res);
            
            res = PQexec(conn, "COPY tmp_attn FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
            PQclear(res);
            
            std::string batch;
            batch.reserve(1 << 20);
            
            for (const auto& edges : thread_edges) {
                for (const auto& [i, j, sim] : edges) {
                    // Create dimension atoms (these represent learned features)
                    std::string src_key = group.component + ":" + std::to_string(layer) + ":dim" + std::to_string(row_indices[i]);
                    std::string tgt_key = group.component + ":" + std::to_string(layer) + ":dim" + std::to_string(row_indices[j]);
                    
                    auto src_hash = AtomCalculator::compute_vocab_token(src_key).hash;
                    auto tgt_hash = AtomCalculator::compute_vocab_token(tgt_key).hash;
                    
                    batch += "C\t\\\\x" + src_hash.to_hex() + "\t";
                    batch += "C\t\\\\x" + tgt_hash.to_hex() + "\t";
                    batch += std::to_string(sim) + "\t";
                    batch += std::to_string(layer) + "\t" + group.component + "\n";
                    
                    if (batch.size() > (1 << 19)) {
                        PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
                        batch.clear();
                    }
                }
            }
            
            if (!batch.empty()) PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
            PQputCopyEnd(conn, nullptr);
            res = PQgetResult(conn);
            PQclear(res);
            
            // Insert with relation_type='W' for weight similarity
            res = PQexec(conn,
                ("INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
                 "SELECT source_type, source_id, target_type, target_id, 'W', weight, '" + config.model_name + "', 1, layer, component FROM tmp_attn "
                 "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
                 "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
                 "  source_count = relation.source_count + 1").c_str());
            PQclear(res);
            
            res = PQexec(conn, "COMMIT");
            PQclear(res);
            
            std::cerr << "    -> " << edge_count << " weight similarity edges\n";
            total_edges += edge_count;
        }
    }
    
    // -------------------------------------------------------------------------
    // PART 3: Token-to-dimension mapping via embedding * projection
    // This shows which tokens activate which dimensions most strongly
    // -------------------------------------------------------------------------
    std::cerr << "\n[TOKEN-DIM] Extracting token->dimension activation patterns...\n";
    
    // Find embedding tensor
    TensorMeta* embed = nullptr;
    for (auto& [name, meta] : g_tensors) {
        if ((name.find("embed_tokens") != std::string::npos ||
             name.find("word_embeddings") != std::string::npos) && meta.shape.size() == 2) {
            embed = &meta;
            break;
        }
    }
    
    if (embed && !g_vocab_tokens.empty()) {
        int64_t vocab_size = std::min(embed->shape[0], static_cast<int64_t>(g_vocab_tokens.size()));
        int64_t embed_dim = embed->shape[1];
        
        // For each token, find which dimensions it activates most strongly
        // This creates token->dimension edges
        std::cerr << "[TOKEN-DIM] Processing " << vocab_size << " tokens x " << embed_dim << " dims\n";
        
        PGresult* res = PQexec(conn, "BEGIN");
        PQclear(res);
        
        res = PQexec(conn,
            "CREATE TEMP TABLE tmp_tokdim ("
            "  source_type CHAR(1), source_id BYTEA,"
            "  target_type CHAR(1), target_id BYTEA,"
            "  weight REAL, component TEXT"
            ") ON COMMIT DROP");
        PQclear(res);
        
        res = PQexec(conn, "COPY tmp_tokdim FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        PQclear(res);
        
        std::string batch;
        batch.reserve(1 << 21);
        size_t tokdim_edges = 0;
        
        // Top-k dimensions per token
        const int top_k = 20;
        
        for (int64_t tok_idx = 0; tok_idx < vocab_size; ++tok_idx) {
            auto emb = read_tensor_row(*embed, static_cast<size_t>(tok_idx));
            if (emb.empty()) continue;
            
            const auto& token = g_vocab_tokens[tok_idx];
            if (token.comp.hash.is_zero()) continue;
            
            char token_type = (token.comp.children.size() <= 1) ? 'A' : 'C';
            
            // Find top-k dimensions by absolute value
            std::vector<std::pair<float, int64_t>> dim_vals;
            dim_vals.reserve(embed_dim);
            for (int64_t d = 0; d < embed_dim; ++d) {
                dim_vals.emplace_back(std::fabs(emb[d]), d);
            }
            std::partial_sort(dim_vals.begin(), dim_vals.begin() + std::min(static_cast<int64_t>(top_k), embed_dim),
                              dim_vals.end(), [](auto& a, auto& b) { return a.first > b.first; });
            
            for (int k = 0; k < std::min(static_cast<int64_t>(top_k), embed_dim); ++k) {
                float val = emb[dim_vals[k].second];  // Keep sign
                // Lower threshold for token->dimension activations (0.3)
                if (std::fabs(val) < 0.3f) continue;
                
                int64_t dim_idx = dim_vals[k].second;
                std::string dim_key = "embed:dim" + std::to_string(dim_idx);
                auto dim_hash = AtomCalculator::compute_vocab_token(dim_key).hash;
                
                batch += token_type;
                batch += "\t\\\\x" + token.comp.hash.to_hex() + "\t";
                batch += "C\t\\\\x" + dim_hash.to_hex() + "\t";
                batch += std::to_string(val) + "\tembed\n";
                tokdim_edges++;
                
                if (batch.size() > (1 << 20)) {
                    PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
                    batch.clear();
                }
            }
        }
        
        if (!batch.empty()) PQputCopyData(conn, batch.c_str(), static_cast<int>(batch.size()));
        PQputCopyEnd(conn, nullptr);
        res = PQgetResult(conn);
        PQclear(res);
        
        // Insert with relation_type='D' for dimension activation
        res = PQexec(conn,
            ("INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, source_count, layer, component) "
             "SELECT source_type, source_id, target_type, target_id, 'D', weight, '" + config.model_name + "', 1, -1, component FROM tmp_tokdim "
             "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET "
             "  weight = (relation.weight * relation.source_count + EXCLUDED.weight) / (relation.source_count + 1), "
             "  source_count = relation.source_count + 1").c_str());
        PQclear(res);
        
        res = PQexec(conn, "COMMIT");
        PQclear(res);
        
        std::cerr << "[TOKEN-DIM] Created " << tokdim_edges << " token->dimension edges\n";
        total_edges += tokdim_edges;
    }
    
    std::cerr << "\n[EXTRACT] Total: " << total_edges << " relation edges from model weights\n";
    return true;
}
