/**
 * Unicode Atom Seeder - Bulk generates all Unicode atoms
 * 
 * Outputs CSV or binary format for fast PostgreSQL COPY ingestion.
 * Generates ~1.1M atoms in <1 second on modern hardware.
 */

#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstring>

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/coordinates.hpp"
#include "hypercube/blake3.hpp"

using namespace hypercube;

struct AtomRecord {
    uint32_t codepoint;
    AtomCategory category;
    uint32_t x, y, z, m;
    uint64_t hilbert_lo, hilbert_hi;
    Blake3Hash hash;
};

// Process a range of codepoints
void process_range(uint32_t start, uint32_t end, std::vector<AtomRecord>& out) {
    out.reserve(end - start);
    
    for (uint32_t cp = start; cp < end; ++cp) {
        // Skip surrogates
        if (cp >= constants::SURROGATE_START && cp <= constants::SURROGATE_END) {
            continue;
        }
        
        AtomRecord rec;
        rec.codepoint = cp;
        rec.category = CoordinateMapper::categorize(cp);
        
        Point4D coords = CoordinateMapper::map_codepoint(cp);
        rec.x = coords.x;
        rec.y = coords.y;
        rec.z = coords.z;
        rec.m = coords.m;
        
        HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
        rec.hilbert_lo = hilbert.lo;
        rec.hilbert_hi = hilbert.hi;
        
        rec.hash = Blake3Hasher::hash_codepoint(cp);
        
        out.push_back(rec);
    }
}

// Output CSV for PostgreSQL COPY directly into atom table
// Format: hash, codepoint, category, coords(WKT), hilbert_lo, hilbert_hi
// Matches atom table: id, codepoint, category, coords, hilbert_lo, hilbert_hi
void write_csv(const std::vector<AtomRecord>& atoms, std::ostream& out) {
    for (const auto& a : atoms) {
        // Normalize to [0,1] for PostGIS POINTZM
        double nx = static_cast<double>(a.x) / 4294967295.0;
        double ny = static_cast<double>(a.y) / 4294967295.0;
        double nz = static_cast<double>(a.z) / 4294967295.0;
        double nm = static_cast<double>(a.m) / 4294967295.0;
        
        // Output format for COPY into atom table with geometry parser
        // SRID=0;POINT ZM (x y z m)
        out << "\\\\x" << a.hash.to_hex() << '\t'
            << a.codepoint << '\t'
            << category_to_string(a.category) << '\t'
            << "SRID=0;POINT ZM (" << std::fixed << std::setprecision(15) 
            << nx << ' ' << ny << ' ' << nz << ' ' << nm << ")\t"
            << static_cast<int64_t>(a.hilbert_lo) << '\t'
            << static_cast<int64_t>(a.hilbert_hi) << '\n';
    }
}

// Output SQL INSERT statements (batched)
void write_sql(const std::vector<AtomRecord>& atoms, std::ostream& out, size_t batch_size = 1000) {
    out << "-- Auto-generated atom seeding SQL\n";
    out << "BEGIN;\n\n";
    
    for (size_t i = 0; i < atoms.size(); i += batch_size) {
        out << "INSERT INTO atom (id, codepoint, category, coords, hilbert_lo, hilbert_hi) VALUES\n";
        
        size_t end = std::min(i + batch_size, atoms.size());
        for (size_t j = i; j < end; ++j) {
            const auto& a = atoms[j];
            
            // Normalize coords to [0,1] for POINTZM
            double nx = static_cast<double>(a.x) / 4294967295.0;
            double ny = static_cast<double>(a.y) / 4294967295.0;
            double nz = static_cast<double>(a.z) / 4294967295.0;
            double nm = static_cast<double>(a.m) / 4294967295.0;
            
            out << (j == i ? "  " : ", ")
                << "('\\x" << a.hash.to_hex() << "', "
                << a.codepoint << ", '"
                << category_to_string(a.category) << "'::atom_category, "
                << "ST_MakePoint(" << nx << ", " << ny << ", " << nz << ", " << nm << "), "
                << static_cast<int64_t>(a.hilbert_lo) << ", "
                << static_cast<int64_t>(a.hilbert_hi) << ")\n";
        }
        out << "ON CONFLICT (codepoint) DO NOTHING;\n\n";
    }
    
    out << "COMMIT;\n";
}

// Convert double to hex string (little-endian, 8 bytes)
inline void double_to_hex(double val, char* out) {
    static const char hex_chars[] = "0123456789abcdef";
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    // Little-endian output
    for (int i = 0; i < 8; ++i) {
        uint8_t byte = (bits >> (i * 8)) & 0xFF;
        out[i * 2] = hex_chars[byte >> 4];
        out[i * 2 + 1] = hex_chars[byte & 0x0F];
    }
}

// Output EWKB format for direct COPY into atom table
// EWKB POINTZM (no SRID) = 01 (little-endian) + 010000c0 (type POINTZM = 0xC0000001 LE) + 4x8-byte doubles
// Format: hash, codepoint, category, ewkb_geometry, hilbert_lo, hilbert_hi
void write_ewkb(const std::vector<AtomRecord>& atoms, std::ostream& out) {
    constexpr double SCALE = 1.0 / 4294967295.0;
    
    // EWKB header for POINTZM: "01010000c0" (little-endian byte order, type=0xC0000001)
    constexpr char ewkb_header[] = "01010000c0";  // 10 chars
    
    // Pre-allocate EWKB string: header(10) + 4*16(coords) = 74 chars
    char ewkb[75];
    std::memcpy(ewkb, ewkb_header, 10);
    ewkb[74] = '\0';
    
    for (const auto& a : atoms) {
        double nx = static_cast<double>(a.x) * SCALE;
        double ny = static_cast<double>(a.y) * SCALE;
        double nz = static_cast<double>(a.z) * SCALE;
        double nm = static_cast<double>(a.m) * SCALE;
        
        // Fill in coordinates as hex
        double_to_hex(nx, ewkb + 10);
        double_to_hex(ny, ewkb + 26);
        double_to_hex(nz, ewkb + 42);
        double_to_hex(nm, ewkb + 58);
        
        out << "\\\\x" << a.hash.to_hex() << '\t'
            << a.codepoint << '\t'
            << category_to_string(a.category) << '\t'
            << ewkb << '\t'
            << static_cast<int64_t>(a.hilbert_lo) << '\t'
            << static_cast<int64_t>(a.hilbert_hi) << '\n';
    }
}

// Output raw format for staging table (faster COPY)
// Format: hash, codepoint, category, x, y, z, m (normalized doubles), hilbert_lo, hilbert_hi
// Optimized: 10-digit precision is sufficient for 32-bit integer->double->integer roundtrip
void write_raw(const std::vector<AtomRecord>& atoms, std::ostream& out) {
    // Use a large buffer for better I/O performance
    constexpr size_t BUFFER_SIZE = 1 << 20;  // 1MB buffer
    std::vector<char> buffer(BUFFER_SIZE);
    out.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
    
    // Pre-compute reciprocal for faster division
    constexpr double SCALE = 1.0 / 4294967295.0;
    
    for (const auto& a : atoms) {
        // Normalize to [0,1] for PostGIS POINTZM
        double nx = static_cast<double>(a.x) * SCALE;
        double ny = static_cast<double>(a.y) * SCALE;
        double nz = static_cast<double>(a.z) * SCALE;
        double nm = static_cast<double>(a.m) * SCALE;
        
        // Use 10-digit precision - sufficient for 32-bit coordinate roundtrip
        out << "\\\\x" << a.hash.to_hex() << '\t'
            << a.codepoint << '\t'
            << category_to_string(a.category) << '\t'
            << std::fixed << std::setprecision(10) 
            << nx << '\t' << ny << '\t' << nz << '\t' << nm << '\t'
            << static_cast<int64_t>(a.hilbert_lo) << '\t'
            << static_cast<int64_t>(a.hilbert_hi) << '\n';
    }
}

int main(int argc, char* argv[]) {
    std::string format = "csv";
    std::string output_file = "-";
    int num_threads = std::thread::hardware_concurrency();
    
    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sql") format = "sql";
        else if (arg == "--csv") format = "csv";
        else if (arg == "--raw") format = "raw";
        else if (arg == "--ewkb") format = "ewkb";
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "-j" && i + 1 < argc) num_threads = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " [--csv|--sql|--raw|--ewkb] [-o output] [-j threads]\n";
            return 0;
        }
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Parallel generation
    std::vector<std::vector<AtomRecord>> thread_results(num_threads);
    std::vector<std::thread> threads;
    
    uint32_t total_codepoints = constants::MAX_CODEPOINT + 1;
    uint32_t chunk_size = (total_codepoints + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        uint32_t start = t * chunk_size;
        uint32_t end = std::min(start + chunk_size, total_codepoints);
        
        threads.emplace_back([t, start, end, &thread_results]() {
            process_range(start, end, thread_results[t]);
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    // Merge results
    std::vector<AtomRecord> all_atoms;
    size_t total = 0;
    for (const auto& v : thread_results) total += v.size();
    all_atoms.reserve(total);
    
    for (auto& v : thread_results) {
        all_atoms.insert(all_atoms.end(), 
                         std::make_move_iterator(v.begin()),
                         std::make_move_iterator(v.end()));
    }
    
    auto gen_time = std::chrono::high_resolution_clock::now();
    auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_time - start_time).count();
    
    std::cerr << "Generated " << all_atoms.size() << " atoms in " << gen_ms << " ms\n";
    
    // Output
    std::ostream* out = &std::cout;
    std::ofstream file_out;
    if (output_file != "-") {
        file_out.open(output_file);
        out = &file_out;
    }
    
    if (format == "csv") {
        write_csv(all_atoms, *out);
    } else if (format == "raw") {
        write_raw(all_atoms, *out);
    } else if (format == "ewkb") {
        write_ewkb(all_atoms, *out);
    } else {
        write_sql(all_atoms, *out);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cerr << "Total time: " << total_ms << " ms\n";
    
    return 0;
}
