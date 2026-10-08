#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dengine {

struct ChatMessage {
    std::string role;
    std::string content;
};

// Implements the selected model's Digits -> ByteLevel -> BPE pipeline.
// ICU supplies Unicode regex/UTF-8 operations, not token IDs or BPE merges.
class Tokenizer {
public:
    static Tokenizer load(const std::filesystem::path& model_directory);

    std::vector<std::uint32_t> encode(std::string_view text) const;
    std::string decode(const std::vector<std::uint32_t>& ids,
                       bool skip_special_tokens = false) const;
    std::string format_chat(const std::vector<ChatMessage>& messages,
                            bool add_generation_prompt) const;
    std::vector<std::uint32_t> encode_chat(const std::vector<ChatMessage>& messages,
                                           bool add_generation_prompt) const;
    std::size_t vocab_size() const noexcept { return id_to_token_.size(); }

private:
    Tokenizer() = default;
    std::vector<std::uint32_t> encode_regular(std::string_view text) const;
    void encode_pretoken(std::string_view piece, std::vector<std::uint32_t>& ids) const;
    std::array<std::string, 256> byte_to_symbol_;
    std::unordered_map<std::uint32_t, std::uint8_t> symbol_to_byte_;
    std::unordered_map<std::string, std::uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::size_t> merge_rank_;
    std::vector<std::pair<std::string, std::uint32_t>> special_tokens_;
    std::vector<bool> is_special_;
};

}  // namespace dengine
