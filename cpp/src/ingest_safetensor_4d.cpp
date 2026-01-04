/**
 * Safetensor Ingestion with 4D Laplacian Projection
 * =================================================
 * 
 * This tool ingests model embeddings from safetensor files and projects them
 * directly into 4D hypercube coordinates using Laplacian Eigenmaps + Gram-Schmidt.
 * 
 * Key differences from the original ingest_safetensor:
 * - NO shape table: embeddings are projected to 4D during ingestion
 * - Direct insertion into atom.geom and composition.centroid
 * - Hilbert indices computed and stored alongside coordinates
 * - Aligns model tokens to the same hypersphere as Unicode atoms
 * 
 * Algorithm:
 * 1. Parse safetensor files and extract token embeddings
 * 2. Build k-NN similarity graph over embeddings
 * 3. Compute unnormalized Laplacian L = D - W
 * 4. Find 4 smallest non-zero eigenvectors
 * 5. Apply Gram-Schmidt orthonormalization
 * 6. Normalize to [0, 2^32-1]^4 hypercube
 * 7. Optionally project onto hypersphere
 * 8. Insert coordinates into atom/composition tables
 */

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
#include "hypercube/ingest/projection_db.hpp"

namespace fs = std::filesystem;
using namespace hypercube;

// =============================================================================
// Configuration
// =============================================================================

struct IngestConfig {
    std::string conninfo;
    std::string model_name;
    
    // Laplacian projection parameters
    int k_neighbors = 15;
    float similarity_threshold = 0.0f;
    int power_iterations = 200;
    bool project_to_sphere = true;
    
    // DB parameters
    int batch_size = 10000;
    bool update_existing = true;
    bool verbose = false;
};

// =============================================================================
// Tensor Metadata
// =============================================================================

struct TensorMeta {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t data_offset_start;
    uint64_t data_offset_end;
    std::string shard_file;
};

// =============================================================================
// Memory-Mapped File
// =============================================================================

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

// =============================================================================
// Global State
// =============================================================================

static std::unordered_map<std::string, TensorMeta> g_tensors;
static std::unordered_map<std::string, std::unique_ptr<MappedFile>> g_mmap_cache;
static std::mutex g_mmap_mutex;

// Token info with pre-computed composition data
struct TokenInfo {
    std::string text;
    CompositionRecord comp;
};

static std::vector<TokenInfo> g_vocab_tokens;
static std::unordered_map<std::string, size_t> g_token_to_idx;

// =============================================================================
// BF16/F16 Conversion
// =============================================================================

inline float bf16_to_float(uint16_t bf16) {
    uint32_t f32 = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

inline float f16_to_float(uint16_t f16) {
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

// =============================================================================
// File Utilities
// =============================================================================

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

static std::vector<float> read_tensor_row(const TensorMeta& meta, size_t row) {
    if (meta.shape.size() < 2) return {};
    
    const MappedFile* mf = get_mapped_file(meta.shard_file);
    if (!mf || !mf->data()) return {};
    
    uint64_t header_size;
    std::memcpy(&header_size, mf->data(), 8);
    
    size_t row_size = static_cast<size_t>(meta.shape[1]);
    size_t bytes_per_elem = 4;
    if (meta.dtype == "BF16" || meta.dtype == "F16") bytes_per_elem = 2;
    
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

// =============================================================================
// Parsing Functions
// =============================================================================

bool parse_safetensor_header(const fs::path& path, const std::string& shard_file = "") {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open: " << path << "\n";
        return false;
    }
    
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), 8);
    
    std::vector<char> buf(header_size);
    file.read(buf.data(), header_size);
    std::string json(buf.begin(), buf.end());
    
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

bool parse_model_index(const fs::path& index_path) {
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
            parse_safetensor_header(shard_path, shard_path.string());
        }
    }
    
    std::cerr << "[INDEX] Parsed " << shards_to_parse.size() << " shards, " 
              << g_tensors.size() << " tensors\n";
    return true;
}

bool parse_vocab(const fs::path& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file) {
        std::cerr << "Cannot open vocab: " << vocab_path << "\n";
        return false;
    }
    
    std::vector<std::string> lines;
    lines.reserve(50000);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            lines.push_back(std::move(line));
        }
    }
    
    size_t total = lines.size();
    g_vocab_tokens.resize(total);
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 16) num_threads = 16;
    
    std::atomic<size_t> idx{0};
    auto start = std::chrono::steady_clock::now();
    
    auto worker = [&]() {
        while (true) {
            size_t i = idx.fetch_add(1);
            if (i >= total) break;
            
            TokenInfo info;
            info.text = lines[i];
            info.comp = AtomCalculator::compute_vocab_token(lines[i]);
            g_vocab_tokens[i] = std::move(info);
        }
    };
    
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back(worker);
    }
    for (auto& th : workers) th.join();
    
    for (size_t i = 0; i < total; ++i) {
        g_token_to_idx[g_vocab_tokens[i].text] = i;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[VOCAB] Loaded " << total << " tokens in " << elapsed << "ms\n";
    return true;
}

// =============================================================================
// Main Ingestion Pipeline with Laplacian Projection
// =============================================================================

bool ingest_with_laplacian_projection(PGconn* conn, const IngestConfig& config) {
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
        std::cerr << "[INGEST] No embedding tensor found\n";
        return false;
    }
    
    if (embed->shape.size() < 2) {
        std::cerr << "[INGEST] Invalid embedding shape\n";
        return false;
    }
    
    int64_t vocab_size = std::min(embed->shape[0], static_cast<int64_t>(g_vocab_tokens.size()));
    int64_t embed_dim = embed->shape[1];
    
    std::cerr << "\n=== Laplacian Projection Ingestion ===\n";
    std::cerr << "Model: " << config.model_name << "\n";
    std::cerr << "Vocab size: " << vocab_size << "\n";
    std::cerr << "Embedding dim: " << embed_dim << "\n\n";
    
    auto total_start = std::chrono::steady_clock::now();
    
    // Step 1: Load all embeddings
    std::cerr << "[1] Loading embeddings from safetensor...\n";
    std::vector<std::vector<float>> embeddings(vocab_size);
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::atomic<int64_t> load_idx{0};
    auto load_worker = [&]() {
        while (true) {
            int64_t i = load_idx.fetch_add(1);
            if (i >= vocab_size) break;
            embeddings[i] = read_tensor_row(*embed, static_cast<size_t>(i));
        }
    };
    
    std::vector<std::thread> load_threads;
    for (unsigned t = 0; t < num_threads; ++t) {
        load_threads.emplace_back(load_worker);
    }
    for (auto& th : load_threads) th.join();
    
    std::cerr << "  Loaded " << vocab_size << " embeddings\n";
    
    // Step 2: Run Laplacian projection
    std::cerr << "\n[2] Running Laplacian Eigenmap projection...\n";
    
    LaplacianConfig lap_config;
    lap_config.k_neighbors = config.k_neighbors;
    lap_config.similarity_threshold = config.similarity_threshold;
    lap_config.power_iterations = config.power_iterations;
    lap_config.project_to_sphere = config.project_to_sphere;
    lap_config.num_threads = static_cast<int>(num_threads);
    
    LaplacianProjector projector(lap_config);
    
    // Optional progress callback
    projector.set_progress_callback([](const std::string& stage, size_t current, size_t total) {
        if (current % 1000 == 0 || current == total) {
            std::cerr << "  [" << stage << "] " << current << "/" << total << "\r" << std::flush;
        }
    });
    
    // Collect labels for projection
    std::vector<std::string> labels(vocab_size);
    for (int64_t i = 0; i < vocab_size; ++i) {
        labels[i] = g_vocab_tokens[i].text;
    }
    
    ProjectionResult result = projector.project(embeddings, labels);
    
    if (result.coords.empty()) {
        std::cerr << "[INGEST] Projection failed\n";
        return false;
    }
    
    // Step 3: Persist to database
    std::cerr << "\n[3] Persisting to database...\n";
    
    // Collect token data
    std::vector<std::string> tok_labels(vocab_size);
    std::vector<Blake3Hash> tok_hashes(vocab_size);
    std::vector<bool> tok_is_atom(vocab_size);
    
    for (int64_t i = 0; i < vocab_size; ++i) {
        tok_labels[i] = g_vocab_tokens[i].text;
        tok_hashes[i] = g_vocab_tokens[i].comp.hash;
        tok_is_atom[i] = (g_vocab_tokens[i].comp.children.size() <= 1);
    }
    
    db::PersistConfig db_config;
    db_config.model_name = config.model_name;
    db_config.batch_size = config.batch_size;
    db_config.update_existing = config.update_existing;
    db_config.verbose = config.verbose;
    
    db::ProjectionPersister persister(conn, db_config);
    size_t updated = persister.persist(tok_labels, tok_hashes, tok_is_atom, result);
    
    auto total_end = std::chrono::steady_clock::now();
    auto total_secs = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    
    std::cerr << "\n=== Ingestion Complete ===\n";
    std::cerr << "Total time: " << total_secs << " seconds\n";
    std::cerr << "Tokens processed: " << vocab_size << "\n";
    std::cerr << "Database rows updated: " << updated << "\n";
    std::cerr << "k-NN edges: " << result.edge_count << "\n";
    
    return true;
}

// =============================================================================
// Main
// =============================================================================

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
        } else if (arg == "-k" && i + 1 < argc) {
            config.k_neighbors = std::stoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            config.similarity_threshold = std::stof(argv[++i]);
        } else if (arg == "--no-sphere") {
            config.project_to_sphere = false;
        } else if (arg == "--update") {
            config.update_existing = true;
        } else if (arg == "-v") {
            config.verbose = true;
        } else if (arg == "--help") {
            std::cerr << "Usage: ingest_safetensor_4d [options] <model_dir>\n"
                      << "  -d DB         Database name\n"
                      << "  -U USER       Database user\n"
                      << "  -h HOST       Database host\n"
                      << "  -n NAME       Model name\n"
                      << "  -k K          k for k-NN graph (default: 15)\n"
                      << "  -t THRESH     Similarity threshold (default: 0.0)\n"
                      << "  --no-sphere   Don't project to hypersphere\n"
                      << "  --update      Update existing coordinates\n"
                      << "  -v            Verbose output\n";
            return 0;
        } else if (arg[0] != '-') {
            model_dir = arg;
        }
    }
    
    if (model_dir.empty()) {
        std::cerr << "Usage: ingest_safetensor_4d [options] <model_dir>\n";
        return 1;
    }
    
    fs::path dir(model_dir);
    if (!fs::is_directory(dir)) {
        std::cerr << "Not a directory: " << model_dir << "\n";
        return 1;
    }
    
    if (config.model_name.empty()) {
        config.model_name = dir.filename().string();
    }
    
    std::cerr << "=== Safetensor Ingestion with 4D Laplacian Projection ===\n";
    std::cerr << "Directory: " << model_dir << "\n";
    std::cerr << "Model: " << config.model_name << "\n";
    std::cerr << "k-NN: " << config.k_neighbors << "\n";
    std::cerr << "Sphere projection: " << (config.project_to_sphere ? "yes" : "no") << "\n\n";
    
    // Find model files
    fs::path vocab_path, index_path;
    std::vector<fs::path> safetensor_files;
    
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        if (name == "vocab.txt") vocab_path = entry.path();
        else if (name == "model.safetensors.index.json") index_path = entry.path();
        else if (name.find(".safetensors") != std::string::npos && 
                 name.find(".index") == std::string::npos) {
            safetensor_files.push_back(entry.path());
        }
    }
    
    // Parse vocab
    if (!vocab_path.empty()) {
        std::cerr << "[1] Parsing vocab: " << vocab_path << "\n";
        parse_vocab(vocab_path);
    } else {
        std::cerr << "No vocab.txt found\n";
        return 1;
    }
    
    // Parse model tensors
    if (!index_path.empty()) {
        std::cerr << "[2] Parsing sharded model index: " << index_path << "\n";
        parse_model_index(index_path);
    } else if (!safetensor_files.empty()) {
        std::cerr << "[2] Parsing " << safetensor_files.size() << " safetensor files...\n";
        for (const auto& f : safetensor_files) {
            parse_safetensor_header(f);
        }
    } else {
        std::cerr << "No safetensor files found\n";
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
    
    // Run ingestion with Laplacian projection
    bool success = ingest_with_laplacian_projection(conn, config);
    
    PQfinish(conn);
    
    return success ? 0 : 1;
}
