/**
 * @file compositions.cpp
 * @brief Insert vocab token compositions into database
 * 
 * Parallel batch building and COPY streaming for high-throughput insertion
 * of composition records with computed geometry.
 */

#include "hypercube/ingest/db_operations.hpp"
#include "hypercube/db/operations.hpp"
#include "hypercube/db/helpers.hpp"

namespace hypercube {
namespace ingest {
namespace db {

bool insert_compositions(PGconn* conn, IngestContext& ctx) {
    using namespace hypercube::db;
    
    if (ctx.vocab_tokens.empty()) return true;
    
    size_t total = ctx.vocab_tokens.size();
    std::cerr << "[COMP] Inserting " << total << " token compositions...\n";
    
    // Count compositions (skip single-char tokens)
    size_t comp_count = 0;
    for (const auto& token : ctx.vocab_tokens) {
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
            
            const auto& token = ctx.vocab_tokens[i];
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
    
    // Phase 2: Stream to database with transaction
    Transaction tx(conn);

    // Drop idx_comp_label to prevent corruption during bulk insert
    std::cerr << "[COMP] Dropping idx_comp_label to prevent corruption...\n";
    exec(conn, "DROP INDEX IF EXISTS idx_comp_label");

    // Temp table for compositions WITH GEOMETRY COLUMNS
    std::cerr << "[COMP] Creating temp tables...\n";
    if (!create_temp_table(conn, "tmp_comp", schema::composition())) {
        std::cerr << "[COMP] Create tmp_comp failed\n";
        return false;
    }
    
    // Temp table for composition children
    if (!create_temp_table(conn, "tmp_comp_child", schema::composition_child())) {
        std::cerr << "[COMP] Create tmp_comp_child failed\n";
        return false;
    }
    
    // COPY compositions using CopyStream
    std::cerr << "[COMP] Copying compositions to temp table...\n";
    {
        CopyStream copy(conn, "COPY tmp_comp FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        if (!copy.ok()) {
            std::cerr << "[COMP] COPY comp start failed: " << copy.error() << "\n";
            return false;
        }
        
        for (size_t i = 0; i < comp_batches.size(); ++i) {
            if (!comp_batches[i].empty()) {
                std::cerr << "  [COPY] Batch " << (i+1) << "/" << comp_batches.size() 
                          << " (" << (comp_batches[i].size()/1024) << "KB)\r" << std::flush;
                if (!copy.put(comp_batches[i])) {
                    std::cerr << "\n[COMP] COPY comp failed: " << copy.error() << "\n";
                    return false;
                }
            }
        }
        
        if (!copy.end()) {
            std::cerr << "\n[COMP] COPY comp end failed: " << copy.error() << "\n";
            return false;
        }
    }
    std::cerr << "\n";
    
    // COPY composition children using CopyStream
    std::cerr << "[COMP] Copying composition children to temp table...\n";
    {
        CopyStream copy(conn, "COPY tmp_comp_child FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')");
        if (!copy.ok()) {
            std::cerr << "[COMP] COPY comp_child start failed: " << copy.error() << "\n";
            return false;
        }
        
        for (size_t i = 0; i < child_batches.size(); ++i) {
            if (!child_batches[i].empty()) {
                std::cerr << "  [COPY] Batch " << (i+1) << "/" << child_batches.size() 
                          << " (" << (child_batches[i].size()/1024) << "KB)\r" << std::flush;
                if (!copy.put(child_batches[i])) {
                    std::cerr << "\n[COMP] COPY child failed: " << copy.error() << "\n";
                    return false;
                }
            }
        }
        
        if (!copy.end()) {
            std::cerr << "\n[COMP] COPY child end failed: " << copy.error() << "\n";
            return false;
        }
    }
    std::cerr << "\n";

    // Insert compositions WITH geometry columns
    std::cerr << "[COMP] Inserting into composition table...\n";
    Result res = exec(conn,
        "INSERT INTO composition (id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi) "
        "SELECT id, label, depth, child_count, atom_count, geom, centroid, hilbert_lo, hilbert_hi "
        "FROM tmp_comp "
        "ON CONFLICT (id) DO UPDATE SET "
        "  geom = EXCLUDED.geom, "
        "  centroid = EXCLUDED.centroid, "
        "  hilbert_lo = EXCLUDED.hilbert_lo, "
        "  hilbert_hi = EXCLUDED.hilbert_hi "
        "WHERE composition.geom IS NULL");
    
    if (!res.ok()) {
        std::cerr << "[COMP] Composition insert failed: " << res.error_message() << "\n";
        return false;
    }
    int inserted_comps = cmd_tuples(res);
    std::cerr << "[COMP] Inserted " << inserted_comps << " compositions\n";
    
    // Insert composition children (only for compositions that exist)
    std::cerr << "[COMP] Inserting into composition_child table...\n";
    res = exec(conn,
        "INSERT INTO composition_child (composition_id, ordinal, child_type, child_id) "
        "SELECT composition_id, ordinal, child_type, child_id "
        "FROM tmp_comp_child "
        "WHERE EXISTS (SELECT 1 FROM composition WHERE id = tmp_comp_child.composition_id) "
        "ON CONFLICT (composition_id, ordinal) DO NOTHING");
    
    if (!res.ok()) {
        std::cerr << "[COMP] Composition child insert failed: " << res.error_message() << "\n";
        return false;
    }
    int inserted_children = cmd_tuples(res);
    std::cerr << "[COMP] Inserted " << inserted_children << " children\n";
    
    // Recreate idx_comp_label
    std::cerr << "[COMP] Recreating idx_comp_label...\n";
    exec(conn, "CREATE INDEX idx_comp_label ON composition(label)");

    tx.commit();

    auto end = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[COMP] Inserted " << inserted_comps << " compositions, " 
              << inserted_children << " children in " << total_ms << "ms\n";
    return true;
}

} // namespace db
} // namespace ingest
} // namespace hypercube
