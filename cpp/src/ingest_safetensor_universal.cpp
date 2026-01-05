/**
 * Universal Safetensor Ingestion
 * ==============================
 * 
 * Ingests ALL tensors from ANY safetensor file - vision, language, multimodal.
 * No vocab requirement. Every tensor key becomes a composition.
 * 
 * Algorithm:
 * 1. Parse ALL tensor keys from safetensor
 * 2. For each tensor:
 *    - Flatten to 1D vector
 *    - Create composition with label = tensor key name
 *    - Store flattened weights (sparse: skip zeros)
 * 3. Build k-NN similarity graph across ALL tensors
 * 4. Create relation edges for similar tensors (P = proximity)
 * 5. Project tensor embeddings to 4D via Laplacian Eigenmaps
 * 6. Store 4D centroids on compositions
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
#include <queue>
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
#include "hypercube/laplacian_4d.hpp"

namespace fs = std::filesystem;
using namespace hypercube;

// =============================================================================
// Configuration
// =============================================================================

struct UniversalConfig {
    std::string conninfo;
    std::string model_path;
    std::string model_name;
    
    // Similarity threshold for creating relations
    float similarity_threshold = 0.3f;
    int k_neighbors = 15;
    int power_iterations = 200;
    bool project_to_sphere = true;
    
    // DB parameters
    int batch_size = 1000;
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
    int64_t numel;  // Total elements
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
        if (fstat(fd, &st) < 0) { close(fd); return false; }
        size_ = static_cast<size_t>(st.st_size);
        
        data_ = static_cast<const uint8_t*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0));
        close(fd);
        if (data_ == MAP_FAILED) { data_ = nullptr; return false; }
#endif
        return true;
    }
    
    void unmap() {
        if (!data_) return;
#ifdef _WIN32
        UnmapViewOfFile(data_);
        if (mapping_handle_) CloseHandle(mapping_handle_);
        if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
        mapping_handle_ = nullptr;
        file_handle_ = INVALID_HANDLE_VALUE;
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

static MappedFile g_mmap;
static uint64_t g_header_size = 0;
static std::vector<TensorMeta> g_tensors;

// =============================================================================
// Type Conversion
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
// Safetensor Parsing
// =============================================================================

bool parse_safetensor(const fs::path& path) {
    if (!g_mmap.map(path.string())) {
        std::cerr << "Cannot mmap: " << path << "\n";
        return false;
    }
    
    std::memcpy(&g_header_size, g_mmap.data(), 8);
    
    std::string json(reinterpret_cast<const char*>(g_mmap.data() + 8), g_header_size);
    
    // Parse all tensor entries
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
        
        meta.numel = 1;
        if (!shape_str.empty()) {
            std::stringstream ss(shape_str);
            std::string dim;
            while (std::getline(ss, dim, ',')) {
                int64_t d = std::stoll(dim);
                meta.shape.push_back(d);
                meta.numel *= d;
            }
        }
        
        // Extract data offsets
        size_t off_pos = json.find("\"data_offsets\"", pos);
        size_t off_start = json.find("[", off_pos);
        size_t off_end = json.find("]", off_start);
        std::string off_str = json.substr(off_start + 1, off_end - off_start - 1);
        
        size_t comma = off_str.find(",");
        meta.data_offset_start = std::stoull(off_str.substr(0, comma));
        meta.data_offset_end = std::stoull(off_str.substr(comma + 1));
        
        g_tensors.push_back(std::move(meta));
        pos = off_end;
    }
    
    std::cerr << "[PARSE] Found " << g_tensors.size() << " tensors\n";
    return true;
}

// =============================================================================
// Read Flattened Tensor
// =============================================================================

std::vector<float> read_tensor_flat(const TensorMeta& meta) {
    size_t offset = 8 + g_header_size + meta.data_offset_start;
    const uint8_t* ptr = g_mmap.data() + offset;
    
    std::vector<float> result(meta.numel);
    
    if (meta.dtype == "F32") {
        std::memcpy(result.data(), ptr, meta.numel * 4);
    } else if (meta.dtype == "BF16") {
        const uint16_t* buf = reinterpret_cast<const uint16_t*>(ptr);
        for (int64_t i = 0; i < meta.numel; ++i) {
            result[i] = bf16_to_float(buf[i]);
        }
    } else if (meta.dtype == "F16") {
        const uint16_t* buf = reinterpret_cast<const uint16_t*>(ptr);
        for (int64_t i = 0; i < meta.numel; ++i) {
            result[i] = f16_to_float(buf[i]);
        }
    } else if (meta.dtype == "I64") {
        const int64_t* buf = reinterpret_cast<const int64_t*>(ptr);
        for (int64_t i = 0; i < meta.numel; ++i) {
            result[i] = static_cast<float>(buf[i]);
        }
    } else if (meta.dtype == "I32") {
        const int32_t* buf = reinterpret_cast<const int32_t*>(ptr);
        for (int64_t i = 0; i < meta.numel; ++i) {
            result[i] = static_cast<float>(buf[i]);
        }
    }
    
    return result;
}

// =============================================================================
// Compute Summary Vector for k-NN
// =============================================================================

// For large tensors, compute a fixed-size summary for similarity comparison
std::vector<float> compute_tensor_summary(const std::vector<float>& data, size_t summary_dim = 256) {
    if (data.size() <= summary_dim) {
        // Pad with zeros if smaller
        std::vector<float> result = data;
        result.resize(summary_dim, 0.0f);
        return result;
    }
    
    // Downsample by averaging chunks
    std::vector<float> result(summary_dim);
    size_t chunk_size = data.size() / summary_dim;
    
    for (size_t i = 0; i < summary_dim; ++i) {
        float sum = 0.0f;
        size_t start = i * chunk_size;
        size_t end = (i == summary_dim - 1) ? data.size() : (i + 1) * chunk_size;
        for (size_t j = start; j < end; ++j) {
            sum += data[j];
        }
        result[i] = sum / static_cast<float>(end - start);
    }
    
    return result;
}

// =============================================================================
// Cosine Similarity
// =============================================================================

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 0.0f;
    
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a < 1e-9f || norm_b < 1e-9f) return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// =============================================================================
// Database Operations
// =============================================================================

bool create_composition(PGconn* conn, const std::string& label, const uint8_t* hash,
                        const std::array<uint32_t, 4>& coords, const std::string& model_name) {
    // Format geometry
    char geom[256];
    snprintf(geom, sizeof(geom), "POINTZM(%u %u %u %u)",
             coords[0], coords[1], coords[2], coords[3]);
    
    // Compute Hilbert index
    Point4D pt = {coords[0], coords[1], coords[2], coords[3]};
    HilbertIndex hilbert = HilbertCurve::coords_to_index(pt);
    
    // Format strings that must stay in scope
    char hash_hex[65];
    for (int i = 0; i < 32; ++i) {
        snprintf(hash_hex + i*2, 3, "%02x", hash[i]);
    }
    char hlo_str[32], hhi_str[32];
    snprintf(hlo_str, sizeof(hlo_str), "%lld", static_cast<long long>(hilbert.lo));
    snprintf(hhi_str, sizeof(hhi_str), "%lld", static_cast<long long>(hilbert.hi));
    
    const char* params[8] = {
        hash_hex,
        label.c_str(),
        "1",  // depth
        "0",  // child_count
        "0",  // atom_count
        geom,
        hlo_str,
        hhi_str
    };
    
    PGresult* res = PQexecParams(conn,
        "INSERT INTO composition (id, label, depth, child_count, atom_count, centroid, hilbert_lo, hilbert_hi) "
        "VALUES (decode($1, 'hex'), $2, $3, $4, $5, ST_GeomFromText($6, 0), $7, $8) "
        "ON CONFLICT (id) DO NOTHING",
        8, nullptr, params, nullptr, nullptr, 0);
    
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "DB ERROR: " << PQresultErrorMessage(res) << "\n";
        std::cerr << "  Query params: hash=" << hash_hex << " label=" << label << " geom=" << geom << "\n";
    }
    
    bool ok = PQresultStatus(res) == PGRES_COMMAND_OK;
    PQclear(res);
    return ok;
}

bool create_relation(PGconn* conn, const uint8_t* source_hash, const uint8_t* target_hash,
                     float weight, const std::string& model_name) {
    char src_hex[65], tgt_hex[65];
    for (int i = 0; i < 32; ++i) {
        snprintf(src_hex + i*2, 3, "%02x", source_hash[i]);
        snprintf(tgt_hex + i*2, 3, "%02x", target_hash[i]);
    }
    
    char weight_str[32];
    snprintf(weight_str, sizeof(weight_str), "%.6f", weight);
    
    const char* params[5] = { src_hex, tgt_hex, weight_str, model_name.c_str(), "tensor" };
    
    PGresult* res = PQexecParams(conn,
        "INSERT INTO relation (source_type, source_id, target_type, target_id, relation_type, weight, source_model, component) "
        "VALUES ('C', decode($1, 'hex'), 'C', decode($2, 'hex'), 'P', $3, $4, $5) "
        "ON CONFLICT (source_id, target_id, relation_type, source_model, layer, component) DO UPDATE SET weight = EXCLUDED.weight",
        5, nullptr, params, nullptr, nullptr, 0);
    
    bool ok = PQresultStatus(res) == PGRES_COMMAND_OK;
    PQclear(res);
    return ok;
}

// =============================================================================
// Main Ingestion Pipeline
// =============================================================================

bool ingest_universal(const UniversalConfig& config) {
    // Parse safetensor
    fs::path model_path(config.model_path);
    fs::path safetensor_path;
    
    // Find safetensor file
    if (fs::is_directory(model_path)) {
        for (const auto& entry : fs::directory_iterator(model_path)) {
            if (entry.path().extension() == ".safetensors") {
                safetensor_path = entry.path();
                break;
            }
        }
    } else if (model_path.extension() == ".safetensors") {
        safetensor_path = model_path;
    }
    
    if (safetensor_path.empty() || !fs::exists(safetensor_path)) {
        std::cerr << "No safetensor file found in: " << model_path << "\n";
        return false;
    }
    
    std::cerr << "\n=== Universal Safetensor Ingestion ===\n";
    std::cerr << "File: " << safetensor_path << "\n";
    
    if (!parse_safetensor(safetensor_path)) {
        return false;
    }
    
    std::cerr << "Tensors: " << g_tensors.size() << "\n\n";
    
    // Connect to database
    PGconn* conn = PQconnectdb(config.conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "DB connection failed: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
        return false;
    }
    
    auto total_start = std::chrono::steady_clock::now();
    
    // Step 1: Read all tensors and compute summaries IN PARALLEL
    std::cerr << "[1] Reading tensors and computing summaries (parallel)...\n";
    
    std::vector<std::vector<float>> summaries(g_tensors.size());
    std::vector<std::array<uint8_t, 32>> hashes(g_tensors.size());
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::atomic<size_t> read_idx{0};
    std::atomic<size_t> read_done{0};
    
    auto read_worker = [&]() {
        while (true) {
            size_t i = read_idx.fetch_add(1);
            if (i >= g_tensors.size()) break;
            
            auto& meta = g_tensors[i];
            auto data = read_tensor_flat(meta);
            summaries[i] = compute_tensor_summary(data);
            
            std::span<const uint8_t> data_bytes(reinterpret_cast<const uint8_t*>(data.data()), 
                                                data.size() * sizeof(float));
            Blake3Hash hash = Blake3Hasher::hash(data_bytes);
            std::copy(hash.bytes.begin(), hash.bytes.end(), hashes[i].begin());
            
            read_done.fetch_add(1);
        }
    };
    
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back(read_worker);
    }
    
    // Progress reporter
    while (read_done.load() < g_tensors.size()) {
        std::cerr << "  " << read_done.load() << "/" << g_tensors.size() << " tensors\r" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    for (auto& t : workers) t.join();
    std::cerr << "  " << g_tensors.size() << "/" << g_tensors.size() << " tensors          \n";
    
    // Step 2: Build k-NN similarity graph and project to 4D
    std::cerr << "\n[2] Running Laplacian projection on tensor summaries...\n";
    
    LaplacianConfig lap_config;
    lap_config.k_neighbors = config.k_neighbors;
    lap_config.similarity_threshold = 0.0f;
    lap_config.power_iterations = config.power_iterations;
    lap_config.project_to_sphere = config.project_to_sphere;
    lap_config.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    
    LaplacianProjector projector(lap_config);
    
    std::vector<std::string> labels(g_tensors.size());
    for (size_t i = 0; i < g_tensors.size(); ++i) {
        labels[i] = g_tensors[i].name;
    }
    
    auto result = projector.project(summaries, labels);
    auto& projections = result.coords;
    
    std::cerr << "  Projected " << projections.size() << " tensors to 4D\n";
    
    // Step 3: Insert compositions
    std::cerr << "\n[3] Inserting compositions...\n";
    
    PQexec(conn, "BEGIN");
    
    size_t inserted = 0;
    for (size_t i = 0; i < g_tensors.size(); ++i) {
        if (create_composition(conn, g_tensors[i].name, hashes[i].data(),
                               projections[i], config.model_name)) {
            inserted++;
        }
        
        if (inserted % config.batch_size == 0) {
            PQexec(conn, "COMMIT");
            PQexec(conn, "BEGIN");
            std::cerr << "  " << inserted << "/" << g_tensors.size() << "\r" << std::flush;
        }
    }
    
    PQexec(conn, "COMMIT");
    std::cerr << "  Inserted " << inserted << " compositions\n";
    
    // Step 4: Create k-NN relations (parallel computation, batched insert)
    std::cerr << "\n[4] Creating k-NN relations (parallel)...\n";
    
    // Pre-compute all neighbors in parallel
    std::vector<std::vector<std::pair<float, size_t>>> all_neighbors(g_tensors.size());
    std::atomic<size_t> knn_idx{0};
    std::atomic<size_t> knn_done{0};
    
    auto knn_worker = [&]() {
        while (true) {
            size_t i = knn_idx.fetch_add(1);
            if (i >= g_tensors.size()) break;
            
            std::vector<std::pair<float, size_t>> neighbors;
            neighbors.reserve(config.k_neighbors * 2);
            
            for (size_t j = 0; j < g_tensors.size(); ++j) {
                if (i == j) continue;
                float sim = cosine_similarity(summaries[i], summaries[j]);
                if (sim >= config.similarity_threshold) {
                    neighbors.emplace_back(sim, j);
                }
            }
            
            std::sort(neighbors.begin(), neighbors.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
            
            if (neighbors.size() > static_cast<size_t>(config.k_neighbors)) {
                neighbors.resize(config.k_neighbors);
            }
            
            all_neighbors[i] = std::move(neighbors);
            knn_done.fetch_add(1);
        }
    };
    
    workers.clear();
    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back(knn_worker);
    }
    
    while (knn_done.load() < g_tensors.size()) {
        std::cerr << "  Computing neighbors: " << knn_done.load() << "/" << g_tensors.size() << "\r" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    for (auto& t : workers) t.join();
    std::cerr << "  Computing neighbors: " << g_tensors.size() << "/" << g_tensors.size() << "          \n";
    
    // Batch insert relations
    PQexec(conn, "BEGIN");
    
    size_t relations_created = 0;
    for (size_t i = 0; i < g_tensors.size(); ++i) {
        for (auto& [sim, j] : all_neighbors[i]) {
            if (create_relation(conn, hashes[i].data(), hashes[j].data(), sim, config.model_name)) {
                relations_created++;
            }
        }
        
        if (i % 500 == 0) {
            PQexec(conn, "COMMIT");
            PQexec(conn, "BEGIN");
            std::cerr << "  Inserting relations: " << relations_created << "\r" << std::flush;
        }
    }
    
    PQexec(conn, "COMMIT");
    std::cerr << "  Created " << relations_created << " relations          \n";
    
    auto total_end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    
    std::cerr << "\n=== Ingestion Complete ===\n";
    std::cerr << "Tensors:    " << g_tensors.size() << "\n";
    std::cerr << "Compositions: " << inserted << "\n";
    std::cerr << "Relations:  " << relations_created << "\n";
    std::cerr << "Time:       " << elapsed << "s\n";
    
    PQfinish(conn);
    return true;
}

// =============================================================================
// Main
// =============================================================================

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <model_path>\n\n"
              << "Options:\n"
              << "  -d, --database <name>     Database name (default: hypercube)\n"
              << "  -U, --user <user>         Database user (default: hypercube)\n"
              << "  -h, --host <host>         Database host (default: localhost)\n"
              << "  -p, --port <port>         Database port (default: 5432)\n"
              << "  -n, --name <model_name>   Model name for source_model field\n"
              << "  -k, --neighbors <k>       k-NN neighbors (default: 15)\n"
              << "  -t, --threshold <t>       Similarity threshold (default: 0.3)\n"
              << "  -v, --verbose             Verbose output\n"
              << "\n"
              << "Example:\n"
              << "  " << prog << " -d hypercube D:\\Models\\detection_models\\DETR-ResNet-101\n";
}

int main(int argc, char* argv[]) {
    UniversalConfig config;
    std::string db_name = "hypercube";
    std::string db_user = "hypercube";
    std::string db_host = "localhost";
    std::string db_port = "5432";
    std::string db_pass;
    
    // Get password from environment
    if (const char* p = std::getenv("PGPASSWORD")) {
        db_pass = p;
    }
    
    std::string model_path;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if ((arg == "-d" || arg == "--database") && i + 1 < argc) {
            db_name = argv[++i];
        } else if ((arg == "-U" || arg == "--user") && i + 1 < argc) {
            db_user = argv[++i];
        } else if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            db_host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            db_port = argv[++i];
        } else if ((arg == "-n" || arg == "--name") && i + 1 < argc) {
            config.model_name = argv[++i];
        } else if ((arg == "-k" || arg == "--neighbors") && i + 1 < argc) {
            config.k_neighbors = std::stoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--threshold") && i + 1 < argc) {
            config.similarity_threshold = std::stof(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            model_path = arg;
        }
    }
    
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    config.model_path = model_path;
    
    if (config.model_name.empty()) {
        // Extract model name from path
        fs::path p(model_path);
        config.model_name = p.filename().string();
    }
    
    // Build connection string
    std::ostringstream conninfo;
    conninfo << "host=" << db_host << " port=" << db_port
             << " dbname=" << db_name << " user=" << db_user;
    if (!db_pass.empty()) {
        conninfo << " password=" << db_pass;
    }
    config.conninfo = conninfo.str();
    
    return ingest_universal(config) ? 0 : 1;
}
