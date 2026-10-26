#pragma once

#include "dengine/transformer_block.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace dengine {

class CachedSession;

// Loads the selected model once and computes next-token scores for token IDs.
// This is an uncached, single-request CPU correctness path; it does not choose
// a token or decode text.
class InferenceModel {
public:
    explicit InferenceModel(const std::filesystem::path& directory);

    F32Tensor next_token_scores(const std::vector<std::uint32_t>& input_ids) const;
    const ModelConfig& config() const noexcept { return model_.config(); }

private:
    friend class CachedSession;
    F32Tensor score_last_hidden(const F32Tensor& hidden) const;

    LoadedModel model_;
    F32Tensor embeddings_;
    F32Tensor final_norm_;
    std::vector<TransformerBlock> blocks_;
};

}  // namespace dengine
