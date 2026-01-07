#include "hypercube/blake3.hpp"

// BLAKE3 reference implementation (portable C)
// In production, link against the official BLAKE3 library with SIMD optimizations
// This is a minimal implementation for bootstrapping

namespace {

[[maybe_unused]] constexpr uint32_t BLAKE3_KEY_LEN = 32;
constexpr uint32_t BLAKE3_OUT_LEN = 32;
constexpr uint32_t BLAKE3_BLOCK_LEN = 64;
constexpr uint32_t BLAKE3_CHUNK_LEN = 1024;

constexpr uint32_t IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

constexpr uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

enum blake3_flags {
    CHUNK_START = 1 << 0,
    CHUNK_END = 1 << 1,
    PARENT = 1 << 2,
    ROOT = 1 << 3,
    KEYED_HASH = 1 << 4,
    DERIVE_KEY_CONTEXT = 1 << 5,
    DERIVE_KEY_MATERIAL = 1 << 6,
};

inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t load32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline void store32_le(uint8_t* p, uint32_t x) {
    p[0] = static_cast<uint8_t>(x);
    p[1] = static_cast<uint8_t>(x >> 8);
    p[2] = static_cast<uint8_t>(x >> 16);
    p[3] = static_cast<uint8_t>(x >> 24);
}

void g(uint32_t* state, size_t a, size_t b, size_t c, size_t d, uint32_t mx, uint32_t my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotr32(state[b] ^ state[c], 7);
}

void round_fn(uint32_t* state, const uint32_t* msg, size_t round) {
    const uint8_t* schedule = MSG_SCHEDULE[round];
    g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
    g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
    g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
    g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);
    g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
    g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
    g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
    g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

void compress(const uint32_t cv[8], const uint8_t block[BLAKE3_BLOCK_LEN],
              uint8_t block_len, uint64_t counter, uint8_t flags,
              uint32_t out[16]) {
    uint32_t msg[16];
    for (size_t i = 0; i < 16; ++i) {
        msg[i] = load32_le(&block[i * 4]);
    }

    uint32_t state[16] = {
        cv[0], cv[1], cv[2], cv[3],
        cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3],
        static_cast<uint32_t>(counter),
        static_cast<uint32_t>(counter >> 32),
        block_len,
        flags
    };

    for (size_t r = 0; r < 7; ++r) {
        round_fn(state, msg, r);
    }

    for (size_t i = 0; i < 8; ++i) {
        out[i] = state[i] ^ state[i + 8];
        out[i + 8] = state[i + 8] ^ cv[i];
    }
}

struct blake3_chunk_state {
    uint32_t cv[8];
    uint64_t chunk_counter;
    uint8_t buf[BLAKE3_BLOCK_LEN];
    uint8_t buf_len;
    uint8_t blocks_compressed;
    uint8_t flags;
};

struct blake3_hasher {
    uint32_t key[8];
    blake3_chunk_state chunk;
    uint8_t cv_stack_len;
    uint8_t cv_stack[54 * 32]; // 54 levels * 32 bytes
};

void chunk_state_init(blake3_chunk_state* self, const uint32_t key[8], uint8_t flags) {
    for (int i = 0; i < 8; ++i) self->cv[i] = key[i];
    self->chunk_counter = 0;
    std::memset(self->buf, 0, BLAKE3_BLOCK_LEN);
    self->buf_len = 0;
    self->blocks_compressed = 0;
    self->flags = flags;
}

void chunk_state_reset(blake3_chunk_state* self, const uint32_t key[8], uint64_t chunk_counter) {
    for (int i = 0; i < 8; ++i) self->cv[i] = key[i];
    self->chunk_counter = chunk_counter;
    std::memset(self->buf, 0, BLAKE3_BLOCK_LEN);
    self->buf_len = 0;
    self->blocks_compressed = 0;
}

size_t chunk_state_len(const blake3_chunk_state* self) {
    return BLAKE3_BLOCK_LEN * static_cast<size_t>(self->blocks_compressed) + 
           static_cast<size_t>(self->buf_len);
}

uint8_t chunk_state_start_flag(const blake3_chunk_state* self) {
    return self->blocks_compressed == 0 ? CHUNK_START : 0;
}

void chunk_state_update(blake3_chunk_state* self, const uint8_t* input, size_t input_len) {
    while (input_len > 0) {
        if (self->buf_len == BLAKE3_BLOCK_LEN) {
            uint32_t out[16];
            compress(self->cv, self->buf, BLAKE3_BLOCK_LEN, self->chunk_counter,
                    self->flags | chunk_state_start_flag(self), out);
            for (int i = 0; i < 8; ++i) self->cv[i] = out[i];
            self->blocks_compressed++;
            self->buf_len = 0;
            std::memset(self->buf, 0, BLAKE3_BLOCK_LEN);
        }
        
        size_t take = BLAKE3_BLOCK_LEN - self->buf_len;
        if (take > input_len) take = input_len;
        std::memcpy(&self->buf[self->buf_len], input, take);
        self->buf_len += static_cast<uint8_t>(take);
        input += take;
        input_len -= take;
    }
}

void chunk_state_finalize(const blake3_chunk_state* self, bool is_root, uint32_t out[8]) {
    uint8_t flags = self->flags | chunk_state_start_flag(self) | CHUNK_END;
    if (is_root) flags |= ROOT;
    
    uint32_t full_out[16];
    compress(self->cv, self->buf, self->buf_len, self->chunk_counter, flags, full_out);
    for (int i = 0; i < 8; ++i) out[i] = full_out[i];
}

void hasher_init(blake3_hasher* self) {
    chunk_state_init(&self->chunk, IV, 0);
    for (int i = 0; i < 8; ++i) self->key[i] = IV[i];
    self->cv_stack_len = 0;
}

void hasher_push_cv(blake3_hasher* self, const uint32_t cv[8]) {
    for (int i = 0; i < 8; ++i) {
        store32_le(&self->cv_stack[self->cv_stack_len * 32 + i * 4], cv[i]);
    }
    self->cv_stack_len++;
}

void hasher_pop_cv(blake3_hasher* self, uint32_t cv[8]) {
    self->cv_stack_len--;
    for (int i = 0; i < 8; ++i) {
        cv[i] = load32_le(&self->cv_stack[self->cv_stack_len * 32 + i * 4]);
    }
}

void parent_cv(const uint32_t left[8], const uint32_t right[8], 
               const uint32_t key[8], uint8_t flags, uint32_t out[8]) {
    uint8_t block[BLAKE3_BLOCK_LEN];
    for (int i = 0; i < 8; ++i) {
        store32_le(&block[i * 4], left[i]);
        store32_le(&block[32 + i * 4], right[i]);
    }
    uint32_t full_out[16];
    compress(key, block, BLAKE3_BLOCK_LEN, 0, flags | PARENT, full_out);
    for (int i = 0; i < 8; ++i) out[i] = full_out[i];
}

void hasher_add_chunk_cv(blake3_hasher* self, uint32_t new_cv[8], uint64_t total_chunks) {
    while ((total_chunks & 1) == 0) {
        uint32_t parent[8];
        uint32_t popped[8];
        hasher_pop_cv(self, popped);
        parent_cv(popped, new_cv, self->key, 0, parent);
        for (int i = 0; i < 8; ++i) new_cv[i] = parent[i];
        total_chunks >>= 1;
    }
    hasher_push_cv(self, new_cv);
}

void hasher_update(blake3_hasher* self, const uint8_t* input, size_t input_len) {
    while (input_len > 0) {
        if (chunk_state_len(&self->chunk) == BLAKE3_CHUNK_LEN) {
            uint32_t cv[8];
            chunk_state_finalize(&self->chunk, false, cv);
            uint64_t total_chunks = self->chunk.chunk_counter + 1;
            hasher_add_chunk_cv(self, cv, total_chunks);
            chunk_state_reset(&self->chunk, self->key, total_chunks);
        }
        
        size_t take = BLAKE3_CHUNK_LEN - chunk_state_len(&self->chunk);
        if (take > input_len) take = input_len;
        chunk_state_update(&self->chunk, input, take);
        input += take;
        input_len -= take;
    }
}

void hasher_finalize(const blake3_hasher* self, uint8_t out[BLAKE3_OUT_LEN]) {
    uint32_t cv[8];
    chunk_state_finalize(&self->chunk, self->cv_stack_len == 0, cv);
    
    uint32_t parent[8];
    for (int i = 0; i < 8; ++i) parent[i] = cv[i];
    
    for (int i = self->cv_stack_len - 1; i >= 0; --i) {
        uint32_t popped[8];
        for (int j = 0; j < 8; ++j) {
            popped[j] = load32_le(&self->cv_stack[i * 32 + j * 4]);
        }
        uint8_t flags = (i == 0) ? ROOT : 0;
        parent_cv(popped, parent, self->key, flags, parent);
    }
    
    for (int i = 0; i < 8; ++i) {
        store32_le(&out[i * 4], parent[i]);
    }
}

} // anonymous namespace

namespace hypercube {

Blake3Hash Blake3Hasher::hash(std::span<const uint8_t> data) noexcept {
    blake3_hasher hasher;
    hasher_init(&hasher);
    hasher_update(&hasher, data.data(), data.size());

    Blake3Hash result;
    hasher_finalize(&hasher, result.bytes.data());
    return result;
}

Blake3Hash Blake3Hasher::hash(std::string_view str) noexcept {
    return hash(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(str.data()), str.size()));
}

std::vector<uint8_t> Blake3Hasher::encode_utf8(uint32_t codepoint) noexcept {
    std::vector<uint8_t> result;
    
    if (codepoint <= 0x7F) {
        result.push_back(static_cast<uint8_t>(codepoint));
    } else if (codepoint <= 0x7FF) {
        result.push_back(static_cast<uint8_t>(0xC0 | (codepoint >> 6)));
        result.push_back(static_cast<uint8_t>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        result.push_back(static_cast<uint8_t>(0xE0 | (codepoint >> 12)));
        result.push_back(static_cast<uint8_t>(0x80 | ((codepoint >> 6) & 0x3F)));
        result.push_back(static_cast<uint8_t>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0x10FFFF) {
        result.push_back(static_cast<uint8_t>(0xF0 | (codepoint >> 18)));
        result.push_back(static_cast<uint8_t>(0x80 | ((codepoint >> 12) & 0x3F)));
        result.push_back(static_cast<uint8_t>(0x80 | ((codepoint >> 6) & 0x3F)));
        result.push_back(static_cast<uint8_t>(0x80 | (codepoint & 0x3F)));
    }
    
    return result;
}

Blake3Hash Blake3Hasher::hash_codepoint(uint32_t codepoint) noexcept {
    auto utf8 = encode_utf8(codepoint);
    return hash(utf8);
}

Blake3Hash Blake3Hasher::hash_children(std::span<const Blake3Hash> children) noexcept {
    if (children.empty()) {
        return hash(std::vector<uint8_t>{});
    }
    
    blake3_hasher hasher;
    hasher_init(&hasher);
    
    for (const auto& child : children) {
        hasher_update(&hasher, child.bytes.data(), child.bytes.size());
    }
    
    Blake3Hash result;
    hasher_finalize(&hasher, result.bytes.data());
    return result;
}

Blake3Hash Blake3Hasher::hash_children_ordered(std::span<const Blake3Hash> children) noexcept {
    blake3_hasher hasher;
    hasher_init(&hasher);
    
    uint32_t ordinal = 0;
    for (const auto& child : children) {
        // Add ordinal as 4 bytes little-endian
        uint8_t ord_bytes[4];
        ord_bytes[0] = static_cast<uint8_t>(ordinal);
        ord_bytes[1] = static_cast<uint8_t>(ordinal >> 8);
        ord_bytes[2] = static_cast<uint8_t>(ordinal >> 16);
        ord_bytes[3] = static_cast<uint8_t>(ordinal >> 24);
        hasher_update(&hasher, ord_bytes, 4);
        
        hasher_update(&hasher, child.bytes.data(), child.bytes.size());
        ++ordinal;
    }
    
    Blake3Hash result;
    hasher_finalize(&hasher, result.bytes.data());
    return result;
}

// Incremental hasher implementation
struct Blake3Hasher::Incremental::Impl {
    blake3_hasher hasher;
};

Blake3Hasher::Incremental::Incremental() noexcept 
    : impl_(std::make_unique<Impl>()) {
    hasher_init(&impl_->hasher);
}

Blake3Hasher::Incremental::~Incremental() = default;

Blake3Hasher::Incremental::Incremental(Incremental&&) noexcept = default;
Blake3Hasher::Incremental& Blake3Hasher::Incremental::operator=(Incremental&&) noexcept = default;

void Blake3Hasher::Incremental::update(std::span<const uint8_t> data) noexcept {
    hasher_update(&impl_->hasher, data.data(), data.size());
}

void Blake3Hasher::Incremental::update(std::string_view str) noexcept {
    std::vector<uint8_t> data(str.begin(), str.end());
    update(data);
}

Blake3Hash Blake3Hasher::Incremental::finalize() noexcept {
    Blake3Hash result;
    hasher_finalize(&impl_->hasher, result.bytes.data());
    return result;
}

void Blake3Hasher::Incremental::reset() noexcept {
    hasher_init(&impl_->hasher);
}

Blake3Hash Blake3Hasher::keyed_hash(std::span<const uint8_t> key,
                                     std::span<const uint8_t> data) noexcept {
    // Simplified keyed hash - uses key as IV
    blake3_hasher hasher;
    hasher_init(&hasher);
    
    if (key.size() >= 32) {
        for (int i = 0; i < 8; ++i) {
            hasher.key[i] = load32_le(&key[i * 4]);
            hasher.chunk.cv[i] = hasher.key[i];
        }
        hasher.chunk.flags = KEYED_HASH;
    }
    
    hasher_update(&hasher, data.data(), data.size());
    
    Blake3Hash result;
    hasher_finalize(&hasher, result.bytes.data());
    return result;
}

Blake3Hash Blake3Hasher::derive_key(std::string_view context,
                                     std::span<const uint8_t> key_material) noexcept {
    // Derive context key
    blake3_hasher context_hasher;
    hasher_init(&context_hasher);
    context_hasher.chunk.flags = DERIVE_KEY_CONTEXT;
    hasher_update(&context_hasher, 
                  reinterpret_cast<const uint8_t*>(context.data()), 
                  context.size());
    
    uint8_t context_key[32];
    hasher_finalize(&context_hasher, context_key);
    
    // Derive output key
    blake3_hasher derive_hasher;
    hasher_init(&derive_hasher);
    for (int i = 0; i < 8; ++i) {
        derive_hasher.key[i] = load32_le(&context_key[i * 4]);
        derive_hasher.chunk.cv[i] = derive_hasher.key[i];
    }
    derive_hasher.chunk.flags = DERIVE_KEY_MATERIAL;
    
    hasher_update(&derive_hasher, key_material.data(), key_material.size());
    
    Blake3Hash result;
    hasher_finalize(&derive_hasher, result.bytes.data());
    return result;
}

} // namespace hypercube
