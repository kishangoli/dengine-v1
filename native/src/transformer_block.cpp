#include "dengine/transformer_block.hpp"

#include "dengine/kv_cache.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dengine {
namespace {

constexpr float kNormEpsilon = 1e-5f;
constexpr double kRopeTheta = 100000.0;

std::size_t checked_layer(const ModelConfig& config, std::size_t layer) {
    if (layer >= config.layers) throw std::out_of_range("transformer layer index is out of range");
    return layer;
}

F32Tensor project_rows(const F32Tensor& weight, const F32Tensor& rows) {
    const auto count = rows.shape().dimensions()[0];
    const auto width = rows.shape().dimensions()[1];
    const auto output_width = weight.shape().dimensions()[0];
    F32Tensor output(TensorShape({count, output_width}));
    for (std::size_t row = 0; row < count; ++row) {
        std::vector<float> values(rows.values().begin() + row * width,
                                  rows.values().begin() + (row + 1) * width);
        auto projected = matvec(weight, F32Tensor(TensorShape({width}), std::move(values)));
        for (std::size_t column = 0; column < output_width; ++column) {
            output.at({row, column}) = projected.values()[column];
        }
    }
    return output;
}

void rotate_positions(F32Tensor& projected, std::size_t heads, std::size_t head_width,
                      std::size_t start_position) {
    const auto count = projected.shape().dimensions()[0];
    const auto width = heads * head_width;
    const auto half = head_width / 2;
    for (std::size_t token = 0; token < count; ++token) {
        for (std::size_t pair = 0; pair < half; ++pair) {
            const double frequency = 1.0 / std::pow(kRopeTheta, double(2 * pair) / head_width);
            const double angle = double(start_position + token) * frequency;
            const float cosine = static_cast<float>(std::cos(angle));
            const float sine = static_cast<float>(std::sin(angle));
            for (std::size_t head = 0; head < heads; ++head) {
                const auto first = token * width + head * head_width + pair;
                const auto second = first + half;
                const float a = projected.values()[first];
                const float b = projected.values()[second];
                projected.data()[first] = a * cosine - b * sine;
                projected.data()[second] = b * cosine + a * sine;
            }
        }
    }
}

F32Tensor attention(const F32Tensor& query, const F32Tensor& key,
                    const F32Tensor& value, const ModelConfig& config) {
    const std::size_t count = query.shape().dimensions()[0];
    const std::size_t head_width = config.head_size();
    const std::size_t group_size = config.query_heads / config.kv_heads;
    const double scaling = 1.0 / std::sqrt(double(head_width));
    const float masked = -std::numeric_limits<float>::infinity();
    F32Tensor result(TensorShape({count, config.hidden_size}));
    for (std::size_t head = 0; head < config.query_heads; ++head) {
        const std::size_t kv_head = head / group_size;
        for (std::size_t token = 0; token < count; ++token) {
            std::vector<float> scores(count, masked);
            for (std::size_t previous = 0; previous <= token; ++previous) {
                double score = 0.0;
                for (std::size_t column = 0; column < head_width; ++column) {
                    score += double(query.values()[token * config.hidden_size + head * head_width + column]) *
                             key.values()[previous * config.kv_heads * head_width + kv_head * head_width + column];
                }
                scores[previous] = static_cast<float>(score * scaling);
            }
            const auto probabilities = softmax(F32Tensor(TensorShape({count}), std::move(scores)));
            for (std::size_t column = 0; column < head_width; ++column) {
                double sum = 0.0;
                for (std::size_t previous = 0; previous <= token; ++previous) {
                    sum += double(probabilities.values()[previous]) *
                           value.values()[previous * config.kv_heads * head_width + kv_head * head_width + column];
                }
                result.at({token, head * head_width + column}) = static_cast<float>(sum);
            }
        }
    }
    return result;
}

F32Tensor attention_token(const F32Tensor& query, const F32Tensor& key,
                          const F32Tensor& value, const LayerKvCache& cache,
                          const ModelConfig& config) {
    const auto count = cache.length() + 1;
    const auto head_width = config.head_size();
    const auto group_size = config.query_heads / config.kv_heads;
    const double scaling = 1.0 / std::sqrt(double(head_width));
    F32Tensor result(TensorShape({config.hidden_size}));
    for (std::size_t head = 0; head < config.query_heads; ++head) {
        const auto kv_head = head / group_size;
        std::vector<float> scores(count);
        for (std::size_t previous = 0; previous < count; ++previous) {
            const float* key_row = previous == cache.length()
                ? key.values().data() : cache.key_row(previous);
            double score = 0.0;
            for (std::size_t column = 0; column < head_width; ++column) {
                score += double(query.values()[head * head_width + column]) *
                         key_row[kv_head * head_width + column];
            }
            scores[previous] = static_cast<float>(score * scaling);
        }
        const auto probabilities = softmax(F32Tensor(TensorShape({count}), std::move(scores)));
        for (std::size_t column = 0; column < head_width; ++column) {
            double sum = 0.0;
            for (std::size_t previous = 0; previous < count; ++previous) {
                const float* value_row = previous == cache.length()
                    ? value.values().data() : cache.value_row(previous);
                sum += double(probabilities.values()[previous]) *
                       value_row[kv_head * head_width + column];
            }
            result.at({head * head_width + column}) = static_cast<float>(sum);
        }
    }
    return result;
}

}  // namespace

TransformerBlock::TransformerBlock(const LoadedModel& model, std::size_t layer)
    : config_(model.config()), layer_(checked_layer(config_, layer)),
      input_norm_(weight_to_f32(model.tensor("model.layers." + std::to_string(layer) + ".input_layernorm.weight"))),
      post_attention_norm_(weight_to_f32(model.tensor("model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight"))),
      query_(weight_to_f32(model.tensor("model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight"))),
      key_(weight_to_f32(model.tensor("model.layers." + std::to_string(layer) + ".self_attn.k_proj.weight"))),
      value_(weight_to_f32(model.tensor("model.layers." + std::to_string(layer) + ".self_attn.v_proj.weight"))),
      attention_output_(weight_to_f32(model.tensor("model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight"))),
      gate_(weight_to_f32(model.tensor("model.layers." + std::to_string(layer) + ".mlp.gate_proj.weight"))),
      up_(weight_to_f32(model.tensor("model.layers." + std::to_string(layer) + ".mlp.up_proj.weight"))),
      down_(weight_to_f32(model.tensor("model.layers." + std::to_string(layer) + ".mlp.down_proj.weight"))) {
    if (config_.head_size() % 2 != 0) throw std::invalid_argument("rotary head width must be even");
}

F32Tensor TransformerBlock::forward(const F32Tensor& input, std::size_t start_position) const {
    return forward_detailed(input, start_position).output;
}

BlockResult TransformerBlock::forward_detailed(const F32Tensor& input,
                                                std::size_t start_position) const {
    const auto& shape = input.shape().dimensions();
    if (shape.size() != 2 || shape[1] != config_.hidden_size) {
        throw std::invalid_argument("transformer input must be [tokens, hidden width]");
    }
    if (start_position >= kInitialContextLimit || shape[0] > kInitialContextLimit - start_position ||
        start_position >= config_.max_positions || shape[0] > config_.max_positions - start_position) {
        throw std::out_of_range("transformer positions exceed context limit");
    }
    const auto normalized = rms_norm(input, input_norm_, kNormEpsilon);
    auto queries = project_rows(query_, normalized);
    auto keys = project_rows(key_, normalized);
    const auto values = project_rows(value_, normalized);
    rotate_positions(queries, config_.query_heads, config_.head_size(), start_position);
    rotate_positions(keys, config_.kv_heads, config_.head_size(), start_position);
    const auto mixed = attention(queries, keys, values, config_);
    const auto attention_projection = project_rows(attention_output_, mixed);
    const auto after_attention = add(input, attention_projection);
    const auto feed_input = rms_norm(after_attention, post_attention_norm_, kNormEpsilon);
    const auto gated = multiply(silu(project_rows(gate_, feed_input)),
                                project_rows(up_, feed_input));
    return {normalized, attention_projection, add(after_attention, project_rows(down_, gated))};
}

F32Tensor TransformerBlock::forward_token(const F32Tensor& input, LayerKvCache& cache,
                                           std::size_t position) const {
    if (input.shape().dimensions() != std::vector<std::size_t>{config_.hidden_size}) {
        throw std::invalid_argument("cached transformer input must have hidden width");
    }
    if (cache.width() != config_.kv_heads * config_.head_size() ||
        cache.length() != position) {
        throw std::invalid_argument("KV cache shape or position does not match layer");
    }
    if (position >= cache.capacity() || position >= kInitialContextLimit ||
        position >= config_.max_positions) {
        throw std::out_of_range("cached transformer position exceeds context limit");
    }
    const auto normalized = rms_norm(input, input_norm_, kNormEpsilon);
    auto queries = matvec(query_, normalized);
    auto keys = matvec(key_, normalized);
    const auto values = matvec(value_, normalized);
    queries.reshape(TensorShape({1, config_.hidden_size}));
    keys.reshape(TensorShape({1, cache.width()}));
    rotate_positions(queries, config_.query_heads, config_.head_size(), position);
    rotate_positions(keys, config_.kv_heads, config_.head_size(), position);
    queries.reshape(TensorShape({config_.hidden_size}));
    keys.reshape(TensorShape({cache.width()}));
    const auto mixed = attention_token(queries, keys, values, cache, config_);
    const auto after_attention = add(input, matvec(attention_output_, mixed));
    const auto feed_input = rms_norm(after_attention, post_attention_norm_, kNormEpsilon);
    const auto gated = multiply(silu(matvec(gate_, feed_input)), matvec(up_, feed_input));
    auto output = add(after_attention, matvec(down_, gated));
    cache.append(keys, values);
    return output;
}

}  // namespace dengine
