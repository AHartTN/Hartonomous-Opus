#pragma once

#include "hypercube/types.hpp"
#include <cstdint>

namespace hypercube {

/**
 * Unicode categorization utilities
 * Provides categorization of Unicode codepoints into semantic categories
 */
class UnicodeCategorizer {
public:
    /**
     * Categorize a Unicode codepoint into its semantic category
     * @param codepoint Unicode codepoint to categorize
     * @return AtomCategory enum value
     */
    static AtomCategory categorize(uint32_t codepoint) noexcept;

private:
    // Unicode block definitions for categorization
    struct UnicodeBlock {
        uint32_t start;
        uint32_t end;
        AtomCategory category;
    };

    static constexpr UnicodeBlock unicode_blocks[] = {
        {0x0000, 0x001F, AtomCategory::Control},
        {0x0020, 0x0020, AtomCategory::Space},
        // 0x21-0x2F: Split for proper categorization
        {0x0021, 0x0027, AtomCategory::PunctuationOther}, // ! " # $ % & '
        {0x0028, 0x0028, AtomCategory::PunctuationOpen},  // (
        {0x0029, 0x0029, AtomCategory::PunctuationClose}, // )
        {0x002A, 0x002A, AtomCategory::PunctuationOther}, // *
        {0x002B, 0x002B, AtomCategory::MathSymbol},       // +
        {0x002C, 0x002E, AtomCategory::PunctuationOther}, // , - .
        {0x002F, 0x002F, AtomCategory::PunctuationOther}, // /
        {0x0030, 0x0039, AtomCategory::Digit},
        {0x003A, 0x003B, AtomCategory::PunctuationOther}, // : ;
        {0x003C, 0x003C, AtomCategory::MathSymbol},       // <
        {0x003D, 0x003D, AtomCategory::MathSymbol},       // =
        {0x003E, 0x003E, AtomCategory::MathSymbol},       // >
        {0x003F, 0x0040, AtomCategory::PunctuationOther}, // ? @
        {0x0041, 0x005A, AtomCategory::LetterUpper},
        {0x005B, 0x005B, AtomCategory::PunctuationOpen},
        {0x005C, 0x005C, AtomCategory::PunctuationOther},
        {0x005D, 0x005D, AtomCategory::PunctuationClose},
        {0x005E, 0x0060, AtomCategory::PunctuationOther},
        {0x0061, 0x007A, AtomCategory::LetterLower},
        {0x007B, 0x007B, AtomCategory::PunctuationOpen},
        {0x007C, 0x007C, AtomCategory::PunctuationOther},
        {0x007D, 0x007D, AtomCategory::PunctuationClose},
        {0x007E, 0x007E, AtomCategory::PunctuationOther},
        {0x007F, 0x009F, AtomCategory::Control},
        {0x00A0, 0x00A0, AtomCategory::Space},
        {0x00A1, 0x00AA, AtomCategory::PunctuationOther},
        {0x00AB, 0x00AB, AtomCategory::PunctuationOpen},
        {0x00AC, 0x00AC, AtomCategory::MathSymbol},
        {0x00AD, 0x00B0, AtomCategory::PunctuationOther},
        {0x00B1, 0x00B1, AtomCategory::MathSymbol},
        {0x00B2, 0x00BA, AtomCategory::PunctuationOther},
        {0x00BB, 0x00BB, AtomCategory::PunctuationClose},
        {0x00BC, 0x00BF, AtomCategory::PunctuationOther},
        {0x00C0, 0x00D6, AtomCategory::LetterUpper},
        {0x00D7, 0x00D7, AtomCategory::MathSymbol},
        {0x00D8, 0x00DE, AtomCategory::LetterUpper},
        {0x00DF, 0x00F6, AtomCategory::LetterLower},
        {0x00F7, 0x00F7, AtomCategory::MathSymbol},
        {0x00F8, 0x00FF, AtomCategory::LetterLower},
        {0x0100, 0x024F, AtomCategory::LetterOther},
        {0x0250, 0x02AF, AtomCategory::LetterOther},
        {0x02B0, 0x02FF, AtomCategory::LetterModifier},
        {0x0300, 0x036F, AtomCategory::MarkNonspacing},
        {0x0370, 0x03FF, AtomCategory::LetterOther},
        {0x0400, 0x04FF, AtomCategory::LetterOther},
        {0x0590, 0x05FF, AtomCategory::LetterOther},
        {0x0600, 0x06FF, AtomCategory::LetterOther},
        {0x0900, 0x097F, AtomCategory::LetterOther},
        {0x2000, 0x200A, AtomCategory::Space},
        {0x200B, 0x200F, AtomCategory::Format},
        {0x2010, 0x2015, AtomCategory::PunctuationOther},
        {0x2016, 0x2016, AtomCategory::MathSymbol},
        {0x2017, 0x2017, AtomCategory::PunctuationOther},
        {0x2018, 0x2018, AtomCategory::PunctuationOpen},
        {0x2019, 0x2019, AtomCategory::PunctuationClose},
        {0x201A, 0x201A, AtomCategory::PunctuationClose},
        {0x201B, 0x201B, AtomCategory::PunctuationOpen},
        {0x201C, 0x201C, AtomCategory::PunctuationOpen},
        {0x201D, 0x201D, AtomCategory::PunctuationClose},
        {0x201E, 0x201E, AtomCategory::PunctuationClose},
        {0x201F, 0x201F, AtomCategory::PunctuationOpen},
        {0x2020, 0x2027, AtomCategory::PunctuationOther},
        {0x2028, 0x2029, AtomCategory::Separator},
        {0x202A, 0x202E, AtomCategory::Format},
        {0x202F, 0x202F, AtomCategory::Space},
        {0x2030, 0x2038, AtomCategory::PunctuationOther},
        {0x2039, 0x2039, AtomCategory::PunctuationOpen},
        {0x203A, 0x203A, AtomCategory::PunctuationClose},
        {0x203B, 0x205E, AtomCategory::PunctuationOther},
        {0x205F, 0x205F, AtomCategory::Space},
        {0x2060, 0x206F, AtomCategory::Format},
        {0x2190, 0x21FF, AtomCategory::SymbolOther},
        {0x2200, 0x22FF, AtomCategory::MathSymbol},
        {0x2500, 0x257F, AtomCategory::SymbolOther},
        {0x25A0, 0x25FF, AtomCategory::SymbolOther},
        {0x2600, 0x26FF, AtomCategory::SymbolOther},
        {0x2700, 0x27BF, AtomCategory::SymbolOther},
        {0x2A00, 0x2AFF, AtomCategory::MathSymbol},
        {0x3400, 0x4DBF, AtomCategory::LetterOther},
        {0x4E00, 0x9FFF, AtomCategory::LetterOther},
        {0x20A0, 0x20CF, AtomCategory::Currency},
        {0xAC00, 0xD7AF, AtomCategory::LetterOther},
        {0xD800, 0xDFFF, AtomCategory::Surrogate},
        {0xE000, 0xF8FF, AtomCategory::PrivateUse},
        {0xFFFE, 0xFFFF, AtomCategory::Noncharacter},
        {0x1F300, 0x1F5FF, AtomCategory::SymbolOther},
        {0x1F600, 0x1F64F, AtomCategory::SymbolOther},
        {0x1F680, 0x1F6FF, AtomCategory::SymbolOther},
        {0x1F900, 0x1F9FF, AtomCategory::SymbolOther},
        {0x20000, 0x2A6DF, AtomCategory::LetterOther},
        {0xF0000, 0xFFFFD, AtomCategory::PrivateUse},
        {0x100000, 0x10FFFD, AtomCategory::PrivateUse},
    };

    static constexpr size_t num_unicode_blocks = sizeof(unicode_blocks) / sizeof(unicode_blocks[0]);
};

} // namespace hypercube