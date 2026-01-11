#include "hypercube/ingest/context.hpp"
#include "hypercube/ingest/parsing.hpp"
#include <iostream>
#include <filesystem>

int main() {
    hypercube::ingest::IngestContext ctx;
    ctx.verbose = true;

    std::filesystem::path vocab_path = "d:/temp/small_vocab.txt";
    std::cout << "Testing vocab parsing with small vocab file: " << vocab_path << std::endl;

    bool result = hypercube::ingest::parse_vocab(ctx, vocab_path);
    std::cout << "Parse result: " << (result ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Tokens loaded: " << ctx.vocab_tokens.size() << std::endl;

    for (size_t i = 0; i < ctx.vocab_tokens.size(); ++i) {
        std::cout << "Token " << i << ": '" << ctx.vocab_tokens[i].text << "' -> "
                  << ctx.vocab_tokens[i].comp.hash.to_hex().substr(0, 8) << "..." << std::endl;
    }

    return result ? 0 : 1;
}