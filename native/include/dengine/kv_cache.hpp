#pragma once

#include "dengine/inference_model.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace dengine {

// Past key/value rows for one layer and one request. Rows are appended in
// token order and retained until the owning session is reset or destroyed.
class LayerKvCache {
public:
    LayerKvCache(std::size_t width, std::size_t capacity);

    std::size_t length() const noexcept { return keys_.size() / width_; }
    std::size_t width() const noexcept { return width_; }
    std::size_t capacity() const noexcept { return capacity_; }
    void clear() noexcept;
    const float* key_row(std::size_t position) const;
    const float* value_row(std::size_t position) const;

private:
    friend class TransformerBlock;
    void append(const F32Tensor& key, const F32Tensor& value);

    std::size_t width_;
    std::size_t capacity_;
    std::vector<float> keys_;
    std::vector<float> values_;
};

// Owns all layer caches for one request. The model's weights stay shared and
// read-only; token history and working state stay private to this session.
class CachedSession {
public:
    explicit CachedSession(const InferenceModel& model,
                           std::size_t capacity = kInitialContextLimit);

    void prefill(const std::vector<std::uint32_t>& input_ids);
    void append_token(std::uint32_t token_id);
    F32Tensor next_token_scores() const;
    void reset() noexcept;

    std::size_t token_count() const noexcept { return token_count_; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t block_token_evaluations() const noexcept { return block_token_evaluations_; }

private:
    const InferenceModel& model_;
    std::size_t capacity_;
    std::vector<LayerKvCache> layers_;
    std::optional<F32Tensor> last_hidden_;
    std::size_t token_count_ = 0;
    std::size_t block_token_evaluations_ = 0;
};

}  // namespace dengine
