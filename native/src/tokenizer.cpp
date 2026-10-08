#include "dengine/tokenizer.hpp"

#include "nlohmann/json.hpp"

#include <unicode/uchar.h>
#include <unicode/uregex.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace dengine {
namespace {

using json = nlohmann::json;
constexpr std::size_t kMaxTokenizerJsonBytes = 8 * 1024 * 1024;
constexpr std::size_t kMaxTokenizerConfigBytes = 64 * 1024;
constexpr std::size_t kSelectedVocabSize = 49152;
constexpr std::size_t kSelectedMergeCount = 48900;

// Exact template in the pinned tokenizer_config.json. Chat formatting is
// implemented below rather than evaluating Jinja inside the native process.
constexpr const char* kChatTemplate =
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}"
    "{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

// The GPT-2 ByteLevel pattern used by the model's ByteLevel pre-tokenizer.
constexpr const char* kByteLevelPattern =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

std::string read_text(const std::filesystem::path& path, std::size_t limit) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size > limit) {
        throw std::runtime_error("tokenizer file missing, empty, or too large: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open tokenizer file: " + path.string());
    std::string contents(static_cast<std::size_t>(size), '\0');
    input.read(contents.data(), static_cast<std::streamsize>(size));
    if (input.gcount() != static_cast<std::streamsize>(size) || input.peek() != EOF) {
        throw std::runtime_error("tokenizer file changed or truncated: " + path.string());
    }
    return contents;
}

json parse_json_file(const std::filesystem::path& path, std::size_t limit) {
    const auto contents = read_text(path, limit);
    auto parsed = json::parse(contents, nullptr, false);
    if (!parsed.is_object()) throw std::invalid_argument("invalid tokenizer JSON: " + path.string());
    return parsed;
}

void require_equal(const json& object, const char* key, const json& expected) {
    const auto found = object.find(key);
    if (found == object.end() || *found != expected) {
        throw std::invalid_argument(std::string("unsupported tokenizer field: ") + key);
    }
}

std::uint32_t token_id(const json& value) {
    if (!value.is_number_unsigned()) throw std::invalid_argument("invalid tokenizer ID");
    const auto id = value.get<std::uint64_t>();
    if (id >= kSelectedVocabSize) throw std::invalid_argument("tokenizer ID outside vocabulary");
    return static_cast<std::uint32_t>(id);
}

std::string pair_key(const std::string& left, const std::string& right) {
    std::string result = left;
    result.push_back('\0');
    result += right;
    return result;
}

std::vector<UChar> to_utf16(std::string_view utf8) {
    if (utf8.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("text is too long to tokenize");
    }
    UErrorCode status = U_ZERO_ERROR;
    int32_t length = 0;
    u_strFromUTF8(nullptr, 0, &length, utf8.data(), static_cast<int32_t>(utf8.size()), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        throw std::invalid_argument("text must be valid UTF-8");
    }
    status = U_ZERO_ERROR;
    std::vector<UChar> converted(static_cast<std::size_t>(length) + 1);
    u_strFromUTF8(converted.data(), length + 1, &length, utf8.data(),
                  static_cast<int32_t>(utf8.size()), &status);
    if (U_FAILURE(status)) throw std::invalid_argument("text must be valid UTF-8");
    converted.resize(static_cast<std::size_t>(length));
    return converted;
}

std::string to_utf8(const UChar* utf16, int32_t length) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t needed = 0;
    u_strToUTF8(nullptr, 0, &needed, utf16, length, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        throw std::runtime_error("Unicode conversion failed");
    }
    status = U_ZERO_ERROR;
    std::string result(static_cast<std::size_t>(needed), '\0');
    u_strToUTF8(result.data(), needed, nullptr, utf16, length, &status);
    if (U_FAILURE(status)) throw std::runtime_error("Unicode conversion failed");
    return result;
}

// The byte-level decoder replaces incomplete UTF-8 sequences, matching the
// behavior of the reference decoder for arbitrary generated token sequences.
std::string utf8_lossy(std::string_view bytes) {
    if (bytes.empty()) return {};
    UErrorCode status = U_ZERO_ERROR;
    int32_t length = 0;
    u_strFromUTF8WithSub(nullptr, 0, &length, bytes.data(),
                         static_cast<int32_t>(bytes.size()), 0xfffd, nullptr, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        throw std::runtime_error("could not decode token bytes");
    }
    status = U_ZERO_ERROR;
    std::vector<UChar> utf16(static_cast<std::size_t>(length) + 1);
    u_strFromUTF8WithSub(utf16.data(), length + 1, &length, bytes.data(),
                         static_cast<int32_t>(bytes.size()), 0xfffd, nullptr, &status);
    if (U_FAILURE(status)) throw std::runtime_error("could not decode token bytes");
    return to_utf8(utf16.data(), length);
}

struct RegexDeleter {
    void operator()(URegularExpression* regex) const { uregex_close(regex); }
};

using Regex = std::unique_ptr<URegularExpression, RegexDeleter>;

Regex make_byte_level_regex() {
    UErrorCode status = U_ZERO_ERROR;
    Regex regex(uregex_openC(kByteLevelPattern, 0, nullptr, &status));
    if (U_FAILURE(status) || !regex) {
        throw std::runtime_error("could not compile ByteLevel regex");
    }
    return regex;
}

bool is_number(UChar32 codepoint) {
    const auto type = u_charType(codepoint);
    return type == U_DECIMAL_DIGIT_NUMBER || type == U_LETTER_NUMBER ||
           type == U_OTHER_NUMBER;
}

}  // namespace

Tokenizer Tokenizer::load(const std::filesystem::path& model_directory) {
    if (model_directory.empty()) throw std::invalid_argument("model directory is required");
    const auto config = parse_json_file(model_directory / "tokenizer_config.json",
                                        kMaxTokenizerConfigBytes);
    require_equal(config, "tokenizer_class", "GPT2Tokenizer");
    require_equal(config, "vocab_size", kSelectedVocabSize);
    require_equal(config, "add_prefix_space", false);
    require_equal(config, "clean_up_tokenization_spaces", false);
    require_equal(config, "model_max_length", 8192);
    require_equal(config, "chat_template", kChatTemplate);
    require_equal(config, "bos_token", "<|im_start|>");
    require_equal(config, "eos_token", "<|im_end|>");
    require_equal(config, "unk_token", "<|endoftext|>");

    const auto data = parse_json_file(model_directory / "tokenizer.json",
                                      kMaxTokenizerJsonBytes);
    require_equal(data, "normalizer", nullptr);
    require_equal(data, "post_processor", nullptr);
    const auto& pre = data.at("pre_tokenizer");
    require_equal(pre, "type", "Sequence");
    const auto& steps = pre.at("pretokenizers");
    if (!steps.is_array() || steps.size() != 2) {
        throw std::invalid_argument("unsupported tokenizer pre-tokenizer sequence");
    }
    require_equal(steps[0], "type", "Digits");
    require_equal(steps[0], "individual_digits", true);
    require_equal(steps[1], "type", "ByteLevel");
    require_equal(steps[1], "add_prefix_space", false);
    require_equal(steps[1], "trim_offsets", true);
    require_equal(steps[1], "use_regex", true);
    require_equal(data.at("decoder"), "type", "ByteLevel");
    require_equal(data.at("decoder"), "add_prefix_space", true);
    require_equal(data.at("decoder"), "trim_offsets", true);
    require_equal(data.at("decoder"), "use_regex", true);

    const auto& model = data.at("model");
    require_equal(model, "type", "BPE");
    require_equal(model, "dropout", nullptr);
    require_equal(model, "unk_token", nullptr);
    require_equal(model, "continuing_subword_prefix", nullptr);
    require_equal(model, "end_of_word_suffix", nullptr);
    require_equal(model, "byte_fallback", false);
    require_equal(model, "fuse_unk", false);
    require_equal(model, "ignore_merges", false);
    const auto& vocab = model.at("vocab");
    const auto& merges = model.at("merges");
    if (!vocab.is_object() || vocab.size() != kSelectedVocabSize ||
        !merges.is_array() || merges.size() != kSelectedMergeCount) {
        throw std::invalid_argument("tokenizer vocabulary or merge count does not match model");
    }

    Tokenizer tokenizer;
    tokenizer.id_to_token_.resize(kSelectedVocabSize);
    tokenizer.is_special_.resize(kSelectedVocabSize, false);
    std::vector<bool> seen(kSelectedVocabSize, false);
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const auto id = token_id(it.value());
        if (seen[id]) throw std::invalid_argument("duplicate tokenizer ID");
        seen[id] = true;
        tokenizer.id_to_token_[id] = it.key();
        tokenizer.token_to_id_.emplace(it.key(), id);
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        throw std::invalid_argument("tokenizer IDs are not contiguous");
    }

    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        if (!merges[rank].is_string()) throw std::invalid_argument("invalid BPE merge");
        const auto pair = merges[rank].get<std::string>();
        const auto separator = pair.find(' ');
        if (separator == std::string::npos || separator == 0 ||
            separator == pair.size() - 1 || pair.find(' ', separator + 1) != std::string::npos) {
            throw std::invalid_argument("invalid BPE merge pair");
        }
        const auto left = pair.substr(0, separator);
        const auto right = pair.substr(separator + 1);
        if (tokenizer.token_to_id_.find(left + right) == tokenizer.token_to_id_.end() ||
            !tokenizer.merge_rank_.emplace(pair_key(left, right), rank).second) {
            throw std::invalid_argument("duplicate or unknown BPE merge");
        }
    }

    const auto& added = data.at("added_tokens");
    if (!added.is_array() || added.size() != 17) {
        throw std::invalid_argument("unexpected special-token list");
    }
    for (const auto& entry : added) {
        const auto id = token_id(entry.at("id"));
        const auto content = entry.at("content").get<std::string>();
        if (entry.at("special") != true || entry.at("single_word") != false ||
            entry.at("lstrip") != false || entry.at("rstrip") != false ||
            entry.at("normalized") != false || tokenizer.id_to_token_[id] != content ||
            tokenizer.is_special_[id]) {
            throw std::invalid_argument("unsupported special-token entry");
        }
        tokenizer.is_special_[id] = true;
        tokenizer.special_tokens_.emplace_back(content, id);
    }
    std::sort(tokenizer.special_tokens_.begin(), tokenizer.special_tokens_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    std::uint32_t next = 256;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) ||
                             (byte >= 161 && byte <= 172) || (byte >= 174);
        const std::uint32_t symbol = visible ? byte : next++;
        char encoded[4];
        int32_t length = 0;
        U8_APPEND_UNSAFE(encoded, length, symbol);
        tokenizer.byte_to_symbol_[byte] = std::string(encoded, static_cast<std::size_t>(length));
        tokenizer.symbol_to_byte_.emplace(symbol, static_cast<std::uint8_t>(byte));
    }
    return tokenizer;
}

void Tokenizer::encode_pretoken(std::string_view piece,
                                std::vector<std::uint32_t>& ids) const {
    std::string mapped;
    for (unsigned char byte : piece) mapped += byte_to_symbol_[byte];
    std::vector<std::string> symbols;
    for (int32_t index = 0; index < static_cast<int32_t>(mapped.size());) {
        const auto start = index;
        UChar32 codepoint;
        U8_NEXT(mapped.data(), index, static_cast<int32_t>(mapped.size()), codepoint);
        if (codepoint < 0) throw std::runtime_error("invalid internal byte symbol");
        auto symbol = mapped.substr(static_cast<std::size_t>(start),
                                    static_cast<std::size_t>(index - start));
        // The reference BPE drops unknown base symbols before considering
        // adjacent merges. This matters for rare control bytes.
        if (token_to_id_.find(symbol) != token_to_id_.end()) {
            symbols.push_back(std::move(symbol));
        }
    }
    while (symbols.size() > 1) {
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        std::string best_pair;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto pair = pair_key(symbols[i], symbols[i + 1]);
            const auto found = merge_rank_.find(pair);
            if (found != merge_rank_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_pair = pair;
            }
        }
        if (best_rank == std::numeric_limits<std::size_t>::max()) break;
        std::vector<std::string> merged;
        for (std::size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() && pair_key(symbols[i], symbols[i + 1]) == best_pair) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.push_back(std::move(symbols[i]));
                ++i;
            }
        }
        symbols = std::move(merged);
    }
    for (const auto& symbol : symbols) {
        const auto found = token_to_id_.find(symbol);
        if (found == token_to_id_.end()) throw std::runtime_error("BPE produced unknown token");
        ids.push_back(found->second);
    }
}

std::vector<std::uint32_t> Tokenizer::encode_regular(std::string_view text) const {
    std::vector<std::uint32_t> ids;
    if (text.empty()) return ids;
    auto regex = make_byte_level_regex();
    const auto split_with_regex = [&](std::string_view segment) {
        if (segment.empty()) return;
        auto utf16 = to_utf16(segment);
        UErrorCode status = U_ZERO_ERROR;
        uregex_setText(regex.get(), utf16.data(), static_cast<int32_t>(utf16.size()), &status);
        if (U_FAILURE(status)) throw std::runtime_error("could not set ByteLevel text");
        int32_t cursor = 0;
        while (uregex_findNext(regex.get(), &status)) {
            const auto start = uregex_start(regex.get(), 0, &status);
            const auto end = uregex_end(regex.get(), 0, &status);
            if (U_FAILURE(status) || start != cursor || end <= start) {
                throw std::runtime_error("ByteLevel regex did not cover input");
            }
            const auto piece = to_utf8(utf16.data() + start, end - start);
            encode_pretoken(piece, ids);
            cursor = end;
        }
        if (U_FAILURE(status) || cursor != static_cast<int32_t>(utf16.size())) {
            throw std::runtime_error("ByteLevel regex did not cover input");
        }
    };

    std::size_t segment_start = 0;
    for (int32_t index = 0; index < static_cast<int32_t>(text.size());) {
        const auto start = index;
        UChar32 codepoint;
        U8_NEXT(text.data(), index, static_cast<int32_t>(text.size()), codepoint);
        if (codepoint < 0) throw std::invalid_argument("text must be valid UTF-8");
        if (is_number(codepoint)) {
            split_with_regex(text.substr(segment_start, static_cast<std::size_t>(start) - segment_start));
            encode_pretoken(text.substr(static_cast<std::size_t>(start),
                                        static_cast<std::size_t>(index - start)), ids);
            segment_start = static_cast<std::size_t>(index);
        }
    }
    split_with_regex(text.substr(segment_start));
    return ids;
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<std::uint32_t> ids;
    std::size_t cursor = 0;
    while (cursor < text.size()) {
        std::size_t next = std::string_view::npos;
        const std::pair<std::string, std::uint32_t>* chosen = nullptr;
        for (const auto& special : special_tokens_) {
            const auto found = text.find(special.first, cursor);
            if (found < next) {
                next = found;
                chosen = &special;
            }
        }
        const auto regular = encode_regular(text.substr(cursor, next == std::string_view::npos
                                                              ? next : next - cursor));
        ids.insert(ids.end(), regular.begin(), regular.end());
        if (!chosen) break;
        ids.push_back(chosen->second);
        cursor = next + chosen->first.size();
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<std::uint32_t>& ids,
                              bool skip_special_tokens) const {
    std::string output;
    std::string bytes;
    const auto flush = [&] {
        output += utf8_lossy(bytes);
        bytes.clear();
    };
    for (auto id : ids) {
        if (id >= id_to_token_.size()) throw std::out_of_range("token ID outside vocabulary");
        if (is_special_[id]) {
            flush();
            if (!skip_special_tokens) output += id_to_token_[id];
            continue;
        }
        const auto& token = id_to_token_[id];
        for (int32_t index = 0; index < static_cast<int32_t>(token.size());) {
            UChar32 symbol;
            U8_NEXT(token.data(), index, static_cast<int32_t>(token.size()), symbol);
            const auto found = symbol_to_byte_.find(static_cast<std::uint32_t>(symbol));
            if (found == symbol_to_byte_.end()) {
                throw std::runtime_error("token contains unknown byte-level symbol");
            }
            bytes.push_back(static_cast<char>(found->second));
        }
    }
    flush();
    return output;
}

std::string Tokenizer::format_chat(const std::vector<ChatMessage>& messages,
                                   bool add_generation_prompt) const {
    std::string formatted;
    if (!messages.empty() && messages.front().role != "system") {
        formatted = "<|im_start|>system\n"
                    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
                    "<|im_end|>\n";
    }
    for (const auto& message : messages) {
        formatted += "<|im_start|>" + message.role + "\n" + message.content +
                     "<|im_end|>\n";
    }
    if (add_generation_prompt) formatted += "<|im_start|>assistant\n";
    return formatted;
}

std::vector<std::uint32_t> Tokenizer::encode_chat(const std::vector<ChatMessage>& messages,
                                                   bool add_generation_prompt) const {
    return encode(format_chat(messages, add_generation_prompt));
}

}  // namespace dengine
