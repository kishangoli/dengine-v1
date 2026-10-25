#include "dengine/generator.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace dengine {

GenerationResult Generator::generate(std::string_view prompt, std::size_t max_new_tokens,
                                     std::string_view stop) const {
    if (max_new_tokens == 0 || max_new_tokens > kMaxNewTokens) {
        throw std::invalid_argument("max_new_tokens must be between 1 and 64");
    }
    auto context = tokenizer_.encode(prompt);
    if (context.empty()) throw std::invalid_argument("prompt produced no tokens");
    if (context.size() > kInitialContextLimit ||
        max_new_tokens - 1 > kInitialContextLimit - context.size()) {
        throw std::out_of_range("prompt and requested output exceed the 1024-token context limit");
    }
    GenerationResult result{"", {}, context.size(), "max_new_tokens"};
    result.generated_ids.reserve(max_new_tokens);
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        const auto scores = model_.next_token_scores(context);
        const auto next = static_cast<std::uint32_t>(
            std::max_element(scores.values().begin(), scores.values().end()) - scores.values().begin());
        result.generated_ids.push_back(next);
        context.push_back(next);
        result.text = tokenizer_.decode(result.generated_ids, true);
        if (next == kEndTokenId) {
            result.finish_reason = "end_token";
            break;
        }
        if (!stop.empty()) {
            const auto at = result.text.find(stop);
            if (at != std::string::npos) {
                result.text.resize(at);
                result.finish_reason = "stop_string";
                break;
            }
        }
    }
    return result;
}

}  // namespace dengine
