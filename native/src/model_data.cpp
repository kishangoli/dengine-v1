#include "dengine/model_data.hpp"

#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace dengine {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "FP32 tensor access requires IEEE-754 32-bit float");
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Stage 2 tensor scalar access requires a little-endian host"
#endif

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("byte count overflow");
    }
    return a + b;
}

std::uint64_t checked_multiply(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("byte count overflow");
    }
    return a * b;
}

template <typename T>
T read_scalar(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    T result;
    std::memcpy(&result, bytes.data() + offset, sizeof(T));
    return result;
}

template <typename T>
void write_scalar(std::vector<std::uint8_t>& bytes, std::size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void require_type(ElementType actual, ElementType expected) {
    if (actual != expected) {
        throw std::invalid_argument("tensor element type does not match accessor");
    }
}

std::filesystem::path require_regular_file(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) {
        throw std::invalid_argument("model file missing or not regular: " + path.string());
    }
    return path;
}

}  // namespace

std::size_t element_size(ElementType type) {
    switch (type) {
        case ElementType::bf16: return 2;
        case ElementType::f32: return 4;
    }
    throw std::invalid_argument("unsupported tensor element type");
}

TensorShape::TensorShape(std::vector<std::size_t> dimensions)
    : dimensions_(std::move(dimensions)), element_count_(1) {
    if (dimensions_.empty() || dimensions_.size() > 8) {
        throw std::invalid_argument("tensor rank must be between 1 and 8");
    }
    for (std::size_t dimension : dimensions_) {
        if (dimension == 0) {
            throw std::invalid_argument("tensor dimensions must be nonzero");
        }
        if (dimension > std::numeric_limits<std::size_t>::max() / element_count_) {
            throw std::overflow_error("tensor element count overflow");
        }
        element_count_ *= dimension;
    }
}

std::size_t TensorShape::byte_count(ElementType type) const {
    const std::size_t width = element_size(type);
    if (element_count_ > std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error("tensor byte count overflow");
    }
    return element_count_ * width;
}

std::size_t TensorShape::flat_index(std::initializer_list<std::size_t> indices) const {
    if (indices.size() != dimensions_.size()) {
        throw std::out_of_range("tensor index rank does not match shape");
    }
    std::size_t flat = 0;
    std::size_t axis = 0;
    for (std::size_t index : indices) {
        if (index >= dimensions_[axis]) {
            throw std::out_of_range("tensor index outside dimension");
        }
        // Since flat < product of previous dimensions, this cannot overflow.
        flat = flat * dimensions_[axis] + index;
        ++axis;
    }
    return flat;
}

WeightTensor::WeightTensor(std::shared_ptr<const std::vector<std::uint8_t>> bytes,
                           std::size_t offset, TensorShape shape, ElementType type)
    : bytes_(std::move(bytes)), offset_(offset), shape_(std::move(shape)), type_(type) {}

std::uint16_t WeightTensor::bf16_bits(std::initializer_list<std::size_t> indices) const {
    require_type(type_, ElementType::bf16);
    return read_scalar<std::uint16_t>(*bytes_, offset_ + shape_.flat_index(indices) * 2);
}

float WeightTensor::f32(std::initializer_list<std::size_t> indices) const {
    require_type(type_, ElementType::f32);
    return read_scalar<float>(*bytes_, offset_ + shape_.flat_index(indices) * 4);
}

WeightStorage::WeightStorage(std::vector<std::uint8_t> bytes) {
    if (bytes.size() > kMaxWeightStorageBytes) {
        throw std::length_error("weight storage exceeds 1 GiB limit");
    }
    try {
        bytes_ = std::make_shared<const std::vector<std::uint8_t>>(std::move(bytes));
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("could not allocate weight storage");
    }
}

WeightTensor WeightStorage::tensor(std::size_t offset, TensorShape shape,
                                   ElementType type) const {
    const std::size_t width = element_size(type);
    if (offset % width != 0) {
        throw std::invalid_argument("tensor byte offset must align to element size");
    }
    const std::size_t count = shape.byte_count(type);
    if (offset > bytes_->size() || count > bytes_->size() - offset) {
        throw std::out_of_range("tensor extends beyond weight storage");
    }
    return WeightTensor(bytes_, offset, std::move(shape), type);
}

ScratchTensor::ScratchTensor(TensorShape shape, ElementType type)
    : shape_(std::move(shape)), type_(type) {
    const std::size_t count = shape_.byte_count(type_);
    if (count > kMaxScratchBytes) {
        throw std::length_error("scratch tensor exceeds 512 MiB limit");
    }
    try {
        bytes_.resize(count, 0);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("could not allocate scratch tensor");
    }
}

void ScratchTensor::set_f32(std::initializer_list<std::size_t> indices, float value) {
    require_type(type_, ElementType::f32);
    write_scalar(bytes_, shape_.flat_index(indices) * 4, value);
}

float ScratchTensor::f32(std::initializer_list<std::size_t> indices) const {
    require_type(type_, ElementType::f32);
    return read_scalar<float>(bytes_, shape_.flat_index(indices) * 4);
}

void ScratchTensor::set_bf16_bits(std::initializer_list<std::size_t> indices,
                                  std::uint16_t value) {
    require_type(type_, ElementType::bf16);
    write_scalar(bytes_, shape_.flat_index(indices) * 2, value);
}

std::uint16_t ScratchTensor::bf16_bits(std::initializer_list<std::size_t> indices) const {
    require_type(type_, ElementType::bf16);
    return read_scalar<std::uint16_t>(bytes_, shape_.flat_index(indices) * 2);
}

void ModelConfig::validate() const {
    if (vocab_size == 0 || hidden_size == 0 || intermediate_size == 0 || layers == 0 ||
        query_heads == 0 || kv_heads == 0 || max_positions == 0) {
        throw std::invalid_argument("model configuration dimensions must be nonzero");
    }
    if (hidden_size % query_heads != 0 || query_heads % kv_heads != 0) {
        throw std::invalid_argument("attention heads do not divide model dimensions");
    }
}

std::size_t ModelConfig::head_size() const {
    validate();
    return hidden_size / query_heads;
}

std::uint64_t ModelConfig::parameter_count() const {
    validate();
    const auto embedding = checked_multiply(vocab_size, hidden_size);
    const auto q = checked_multiply(hidden_size, hidden_size);
    const auto kv = checked_multiply(checked_multiply(kv_heads, head_size()), hidden_size);
    const auto mlp = checked_multiply(intermediate_size, hidden_size);
    std::uint64_t per_layer = checked_add(checked_add(q, q), checked_add(kv, kv));
    per_layer = checked_add(per_layer, checked_multiply(3, mlp));
    per_layer = checked_add(per_layer, checked_multiply(2, hidden_size));
    return checked_add(checked_add(embedding, checked_multiply(layers, per_layer)), hidden_size);
}

ModelConfig smollm2_135m_config() {
    // Pinned Stage 1 config.json; Stage 3 will read and compare the file itself.
    return {49152, 576, 1536, 30, 9, 3, 8192};
}

ModelPaths inspect_model_directory(const std::filesystem::path& directory) {
    if (directory.empty()) {
        throw std::invalid_argument("model directory path must be explicit");
    }
    std::error_code error;
    const auto absolute = std::filesystem::absolute(directory, error);
    if (error || !std::filesystem::is_directory(absolute, error) || error) {
        throw std::invalid_argument("model directory missing or not a directory");
    }
    const auto config = require_regular_file(absolute / "config.json");
    const auto weights = require_regular_file(absolute / "model.safetensors");
    const auto size = std::filesystem::file_size(weights, error);
    if (error || size == 0 || size > kMaxModelFileBytes) {
        throw std::invalid_argument("model weight file is empty, unreadable, or exceeds 1 GiB");
    }
    return {absolute.lexically_normal(), config, weights, size};
}

MemoryEstimate estimate_memory(const ModelConfig& config,
                               std::uint64_t model_file_bytes,
                               std::uint64_t scratch_bytes,
                               std::size_t context_tokens,
                               std::size_t active_requests) {
    config.validate();
    if (model_file_bytes == 0 || model_file_bytes > kMaxModelFileBytes) {
        throw std::invalid_argument("model file exceeds 1 GiB limit or is empty");
    }
    if (scratch_bytes > kMaxScratchBytes) {
        throw std::invalid_argument("scratch reservation exceeds 512 MiB limit");
    }
    if (context_tokens == 0 || context_tokens > kInitialContextLimit ||
        context_tokens > config.max_positions) {
        throw std::invalid_argument("context length exceeds initial 1024-token limit");
    }
    if (active_requests == 0 || active_requests > kInitialActiveRequests) {
        throw std::invalid_argument("only one active request is supported initially");
    }

    const auto converted = checked_multiply(config.parameter_count(), 4);
    const auto kv = checked_multiply(
        checked_multiply(checked_multiply(checked_multiply(
            checked_multiply(config.layers, 2), config.kv_heads), config.head_size()),
            context_tokens), checked_multiply(active_requests, 4));
    const auto weights = checked_add(model_file_bytes, converted);
    if (weights > kMaxWeightStorageBytes || kv > kMaxKvBytes) {
        throw std::length_error("weight or KV reservation exceeds memory limit");
    }
    const auto total = checked_add(checked_add(weights, scratch_bytes), kv);
    if (total > kMaxPlannedBytes) {
        throw std::length_error("planned model memory exceeds 2 GiB limit");
    }
    return {model_file_bytes, converted, scratch_bytes, kv, total};
}

}  // namespace dengine
