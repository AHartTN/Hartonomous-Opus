#include "hypercube/unicode_categorization.hpp"
#include "hypercube/types.hpp"

namespace hypercube {

AtomCategory UnicodeCategorizer::categorize(uint32_t codepoint) noexcept
{
    if ((codepoint & 0xFFFF) >= 0xFFFE) {
        return AtomCategory::Noncharacter;
    }

    size_t lo = 0, hi = num_unicode_blocks;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (codepoint > unicode_blocks[mid].end) {
            lo = mid + 1;
        } else if (codepoint < unicode_blocks[mid].start) {
            hi = mid;
        } else {
            return unicode_blocks[mid].category;
        }
    }

    if (codepoint <= constants::MAX_CODEPOINT) {
        return AtomCategory::LetterOther;
    }
    return AtomCategory::SymbolOther;
}

} // namespace hypercube