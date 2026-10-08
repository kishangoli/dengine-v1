#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <vector>

namespace dengine {

// Stage 2 limits are conservative process-planning limits for a 16 GiB machine.
// They do not describe memory currently allocated by the echo executable.
constexpr std::uint64_t kMiB = 1024ULL * 1024;
constexpr std::uint64_t kGiB = 1024ULL * kMiB;
constexpr std::uint64_t kMaxModelFileBytes = kGiB;
constexpr std::uint64_t kMaxWeightStorageBytes = kGiB;
constexpr std::uint64_t kMaxScratchBytes = 512 * kMiB;
constexpr std::uint64_t kMaxKvBytes = 256 * kMiB;
constexpr std::uint64_t kMaxPlannedBytes = 2 * kGiB;
constexpr std::size_t kInitialContextLimit = 1024;
constexpr std::size_t kInitialActiveRequests = 1;

enum class ElementType { bf16, f32 };
std::size_t element_size(ElementType type);

// All tensors are dense and row-major: the last dimension changes fastest.
// Dimensions must be nonzero; rank is between 1 and 8 inclusive.
class TensorShape {
public:
    explicit TensorShape(std::vector<std::size_t> dimensions);
    const std::vector<std::size_t>& dimensions() const noexcept { return dimensions_; }
    std::size_t element_count() const noexcept { return element_count_; }
    std::size_t byte_count(ElementType type) const;
    std::size_t flat_index(std::initializer_list<std::size_t> indices) const;

private:
    std::vector<std::size_t> dimensions_;
    std::size_t element_count_;
};

// A weight view owns a shared reference to its byte storage. Callers can read
// weights but cannot mutate them through the public interface.
class WeightTensor {
public:
    ElementType type() const noexcept { return type_; }
    const TensorShape& shape() const noexcept { return shape_; }
    std::uint16_t bf16_bits(std::initializer_list<std::size_t> indices) const;
    float f32(std::initializer_list<std::size_t> indices) const;

private:
    friend class WeightStorage;
    WeightTensor(std::shared_ptr<const std::vector<std::uint8_t>> bytes,
                 std::size_t offset, TensorShape shape, ElementType type);
    std::shared_ptr<const std::vector<std::uint8_t>> bytes_;
    std::size_t offset_;
    TensorShape shape_;
    ElementType type_;
};

class WeightStorage {
public:
    explicit WeightStorage(std::vector<std::uint8_t> bytes);
    std::size_t size() const noexcept { return bytes_->size(); }
    WeightTensor tensor(std::size_t offset, TensorShape shape, ElementType type) const;

private:
    std::shared_ptr<const std::vector<std::uint8_t>> bytes_;
};

// Scratch is separately owned, mutable, and bounded. It never aliases weights.
class ScratchTensor {
public:
    ScratchTensor(TensorShape shape, ElementType type);
    const TensorShape& shape() const noexcept { return shape_; }
    ElementType type() const noexcept { return type_; }
    std::size_t size_bytes() const noexcept { return bytes_.size(); }
    void set_f32(std::initializer_list<std::size_t> indices, float value);
    float f32(std::initializer_list<std::size_t> indices) const;
    void set_bf16_bits(std::initializer_list<std::size_t> indices, std::uint16_t value);
    std::uint16_t bf16_bits(std::initializer_list<std::size_t> indices) const;

private:
    TensorShape shape_;
    ElementType type_;
    std::vector<std::uint8_t> bytes_;
};

struct ModelConfig {
    std::size_t vocab_size;
    std::size_t hidden_size;
    std::size_t intermediate_size;
    std::size_t layers;
    std::size_t query_heads;
    std::size_t kv_heads;
    std::size_t max_positions;
    void validate() const;
    std::uint64_t parameter_count() const;
    std::size_t head_size() const;
};

ModelConfig smollm2_135m_config();

struct ModelPaths {
    std::filesystem::path directory;
    std::filesystem::path config;
    std::filesystem::path weights;
    std::uint64_t model_file_bytes;
};

// An explicit directory is required. This checks paths and file sizes only;
// Stage 3 will parse and authenticate contents.
ModelPaths inspect_model_directory(const std::filesystem::path& directory);

struct MemoryEstimate {
    std::uint64_t model_file_bytes;
    std::uint64_t converted_weight_bytes;
    std::uint64_t scratch_bytes;
    std::uint64_t kv_cache_bytes;
    std::uint64_t total_bytes;
};

// A conservative reservation estimate, not a measured peak or allocation.
MemoryEstimate estimate_memory(const ModelConfig& config,
                               std::uint64_t model_file_bytes,
                               std::uint64_t scratch_bytes,
                               std::size_t context_tokens,
                               std::size_t active_requests);

}  // namespace dengine
