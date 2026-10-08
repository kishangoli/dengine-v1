#pragma once

#include "dengine/model_data.hpp"

#include <filesystem>
#include <map>
#include <string>

namespace dengine {

struct TensorDescriptor {
    TensorShape shape;
    ElementType type;
    std::size_t file_offset;
};

// Owns the original Safetensors file bytes. Tensor views share that storage;
// no FP32 conversion or model computation happens during loading.
class LoadedModel {
public:
    LoadedModel(ModelConfig config, WeightStorage storage,
                std::map<std::string, TensorDescriptor> tensors);
    const ModelConfig& config() const noexcept { return config_; }
    const std::map<std::string, TensorDescriptor>& tensors() const noexcept { return tensors_; }
    std::size_t file_bytes() const noexcept { return storage_.size(); }
    WeightTensor tensor(const std::string& name) const;

private:
    ModelConfig config_;
    WeightStorage storage_;
    std::map<std::string, TensorDescriptor> tensors_;
};

// Supports the pinned SmolLM2-135M-Instruct config and its single BF16
// Safetensors file. Throws with a clear message for other layouts/files.
LoadedModel load_model(const std::filesystem::path& directory);

}  // namespace dengine
