#include "dengine/kv_cache.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace dengine {

LayerKvCache::LayerKvCache(std::size_t width, std::size_t capacity)
    : width_(width), capacity_(capacity) {
    if (width == 0 || capacity == 0 || capacity > kInitialContextLimit ||
        width > kMaxKvBytes / (2 * sizeof(float) * capacity)) {
        throw std::invalid_argument("invalid KV cache dimensions or capacity");
    }
    keys_.reserve(width * capacity);
    values_.reserve(width * capacity);
}

void LayerKvCache::clear() noexcept {
    keys_.clear();
    values_.clear();
}

const float* LayerKvCache::key_row(std::size_t position) const {
    if (position >= length()) throw std::out_of_range("KV key position is out of range");
    return keys_.data() + position * width_;
}

const float* LayerKvCache::value_row(std::size_t position) const {
    if (position >= length()) throw std::out_of_range("KV value position is out of range");
    return values_.data() + position * width_;
}

void LayerKvCache::append(const F32Tensor& key, const F32Tensor& value) {
    if (length() >= capacity_) throw std::out_of_range("KV cache is full");
    if (key.shape().dimensions() != std::vector<std::size_t>{width_} ||
        value.shape().dimensions() != std::vector<std::size_t>{width_}) {
        throw std::invalid_argument("KV row width does not match cache");
    }
    keys_.insert(keys_.end(), key.values().begin(), key.values().end());
    values_.insert(values_.end(), value.values().begin(), value.values().end());
}

CachedSession::CachedSession(const InferenceModel& model, std::size_t capacity)
    : model_(model), capacity_(capacity) {
    const auto& config = model_.config();
    if (capacity == 0 || capacity > kInitialContextLimit) {
        throw std::invalid_argument("KV cache capacity must be between 1 and 1024 tokens");
    }
    const auto width = config.kv_heads * config.head_size();
    const auto bytes_per_layer = 2 * sizeof(float) * width * capacity;
    if (config.layers > kMaxKvBytes / bytes_per_layer) {
        throw std::length_error("KV cache exceeds the 256 MiB limit");
    }
    layers_.reserve(config.layers);
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
        layers_.emplace_back(width, capacity);
    }
}

void CachedSession::reset() noexcept {
    for (auto& layer : layers_) layer.clear();
    last_hidden_.reset();
    token_count_ = 0;
    block_token_evaluations_ = 0;
}

void CachedSession::prefill(const std::vector<std::uint32_t>& input_ids) {
    if (token_count_ != 0) throw std::invalid_argument("prefill requires an empty session");
    if (input_ids.empty()) throw std::invalid_argument("prefill requires at least one token");
    if (input_ids.size() > capacity_) throw std::out_of_range("prompt exceeds KV cache capacity");
    try {
        for (const auto id : input_ids) append_token(id);
    } catch (...) {
        reset();
        throw;
    }
}

void CachedSession::append_token(std::uint32_t token_id) {
    if (token_count_ >= capacity_) throw std::out_of_range("KV cache is full");
    if (token_id >= model_.config().vocab_size) {
        throw std::invalid_argument("token ID outside the model vocabulary");
    }
    const auto width = model_.config().hidden_size;
    const auto row = model_.embeddings_.values().begin() + token_id * width;
    F32Tensor hidden(TensorShape({width}), std::vector<float>(row, row + width));
    try {
        for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
            hidden = model_.blocks_[layer].forward_token(hidden, layers_[layer], token_count_);
            ++block_token_evaluations_;
        }
        last_hidden_.emplace(std::move(hidden));
        ++token_count_;
    } catch (...) {
        reset();
        throw;
    }
}

F32Tensor CachedSession::next_token_scores() const {
    if (!last_hidden_) throw std::invalid_argument("cannot score an empty KV cache");
    return model_.score_last_hidden(*last_hidden_);
}

}  // namespace dengine
