#pragma once

#include "dengine/cpu_math.hpp"
#include "dengine/model_loader.hpp"

#include <cstddef>

namespace dengine {

class LayerKvCache;

struct BlockResult {
    F32Tensor normalized_input;
    F32Tensor attention_output;
    F32Tensor output;
};

// One uncached layer of the pinned SmolLM2 model. Input and output are
// [sequence length, hidden width]. The first sequence row is at start_position.
class TransformerBlock {
public:
    TransformerBlock(const LoadedModel& model, std::size_t layer);

    F32Tensor forward(const F32Tensor& input, std::size_t start_position = 0) const;
    BlockResult forward_detailed(const F32Tensor& input, std::size_t start_position = 0) const;
    F32Tensor forward_token(const F32Tensor& input, LayerKvCache& cache,
                            std::size_t position) const;
    std::size_t layer() const noexcept { return layer_; }

private:
    ModelConfig config_;
    std::size_t layer_;
    F32Tensor input_norm_;
    F32Tensor post_attention_norm_;
    F32Tensor query_;
    F32Tensor key_;
    F32Tensor value_;
    F32Tensor attention_output_;
    F32Tensor gate_;
    F32Tensor up_;
    F32Tensor down_;
};

}  // namespace dengine
