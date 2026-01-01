#pragma once

#include "hypercube/types.hpp"
#include <span>
#include <vector>
#include <memory>

namespace hypercube {

/**
 * BLAKE3 hashing for content addressing
 * 
 * Used for:
 * 1. Atom hashing: hash of UTF-8 encoded codepoint
 * 2. Relation hashing: hash of ordered child hashes (Merkle DAG)
 * 
 * BLAKE3 properties:
 * - 256-bit output (32 bytes)
 * - Extremely fast (faster than MD5 while being cryptographically secure)
 * - Supports incremental hashing
 * - SIMD optimized
 */
class Blake3Hasher {
public:
    /**
     * Hash arbitrary data
     * @param data Input bytes
     * @return 32-byte BLAKE3 hash
     */
    static Blake3Hash hash(std::span<const uint8_t> data) noexcept;
    
    /**
     * Hash a string
     */
    static Blake3Hash hash(std::string_view str) noexcept;
    
    /**
     * Hash a Unicode codepoint (as UTF-8)
     * @param codepoint Unicode codepoint
     * @return BLAKE3 hash of UTF-8 encoding
     */
    static Blake3Hash hash_codepoint(uint32_t codepoint) noexcept;
    
    /**
     * Hash a sequence of child hashes (for Merkle DAG)
     * @param children Ordered sequence of child hashes
     * @return BLAKE3 hash of concatenated children
     */
    static Blake3Hash hash_children(std::span<const Blake3Hash> children) noexcept;
    
    /**
     * Hash children with ordinals (position-sensitive)
     * Format: [ordinal_0][hash_0][ordinal_1][hash_1]...
     */
    static Blake3Hash hash_children_ordered(std::span<const Blake3Hash> children) noexcept;
    
    /**
     * Incremental hasher for streaming data
     */
    class Incremental {
    public:
        Incremental() noexcept;
        ~Incremental();
        
        Incremental(const Incremental&) = delete;
        Incremental& operator=(const Incremental&) = delete;
        Incremental(Incremental&&) noexcept;
        Incremental& operator=(Incremental&&) noexcept;
        
        void update(std::span<const uint8_t> data) noexcept;
        void update(std::string_view str) noexcept;
        Blake3Hash finalize() noexcept;
        void reset() noexcept;
        
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    
    /**
     * Keyed hashing for MAC
     */
    static Blake3Hash keyed_hash(std::span<const uint8_t> key,  // Must be 32 bytes
                                  std::span<const uint8_t> data) noexcept;
    
    /**
     * Key derivation
     */
    static Blake3Hash derive_key(std::string_view context,
                                  std::span<const uint8_t> key_material) noexcept;

    /**
     * Encode a Unicode codepoint as UTF-8
     * @param codepoint Unicode codepoint
     * @return UTF-8 bytes (1-4 bytes)
     */
    static std::vector<uint8_t> encode_utf8(uint32_t codepoint) noexcept;
};

} // namespace hypercube
