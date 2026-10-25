#pragma once

#include "dengine/inference_model.hpp"
#include "dengine/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dengine {

constexpr std::size_t kDefaultNewTokens = 8;
constexpr std::size_t kMaxNewTokens = 64;
constexpr std::uint32_t kEndTokenId = 2;

struct GenerationResult {
    std::string text;
    std::vector<std::uint32_t> generated_ids;
    std::size_t prompt_tokens;
    std::string finish_reason;
};

class Generator {
public:
    Generator(const InferenceModel& model, const Tokenizer& tokenizer)
        : model_(model), tokenizer_(tokenizer) {}

    GenerationResult generate(std::string_view prompt, std::size_t max_new_tokens,
                              std::string_view stop = {}) const;

private:
    const InferenceModel& model_;
    const Tokenizer& tokenizer_;
};

}  // namespace dengine
