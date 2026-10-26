#include "dengine/inference_model.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dengine {

InferenceModel::InferenceModel(const std::filesystem::path& directory)
    : model_(load_model(directory)),
      embeddings_(weight_to_f32(model_.tensor("model.embed_tokens.weight"))),
      final_norm_(weight_to_f32(model_.tensor("model.norm.weight"))) {
    blocks_.reserve(model_.config().layers);
    for (std::size_t layer = 0; layer < model_.config().layers; ++layer) {
        blocks_.emplace_back(model_, layer);
    }
}

F32Tensor InferenceModel::next_token_scores(const std::vector<std::uint32_t>& input_ids) const {
    if (input_ids.empty()) throw std::invalid_argument("input_ids must contain at least one token");
    if (input_ids.size() > kInitialContextLimit) {
        throw std::invalid_argument("input_ids exceed the 1024-token context limit");
    }
    const auto width = model_.config().hidden_size;
    F32Tensor hidden(TensorShape({input_ids.size(), width}));
    for (std::size_t token = 0; token < input_ids.size(); ++token) {
        if (input_ids[token] >= model_.config().vocab_size) {
            throw std::invalid_argument("input_ids contain a token outside the vocabulary");
        }
        std::copy_n(embeddings_.values().begin() + input_ids[token] * width, width,
                    hidden.data() + token * width);
    }
    for (const auto& block : blocks_) hidden = block.forward(hidden);

    // Each token is normalized independently. Only the final row is needed
    // to score the next token.
    const auto last = hidden.values().end() - width;
    const F32Tensor final_row(TensorShape({width}), std::vector<float>(last, hidden.values().end()));
    return score_last_hidden(final_row);
}

F32Tensor InferenceModel::score_last_hidden(const F32Tensor& hidden) const {
    const auto normalized = rms_norm(hidden, final_norm_, 1e-5f);
    // SmolLM2 ties this output projection to the input embedding table.
    return matvec(embeddings_, normalized);
}

}  // namespace dengine
