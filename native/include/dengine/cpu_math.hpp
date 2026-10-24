#pragma once

#include "dengine/model_data.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace dengine {

// Owned, dense FP32 working data. Layout is row-major. Reshape changes only
// dimensions, never the values or their order in memory.
class F32Tensor {
public:
    explicit F32Tensor(TensorShape shape);
    F32Tensor(TensorShape shape, std::vector<float> values);

    const TensorShape& shape() const noexcept { return shape_; }
    const std::vector<float>& values() const noexcept { return values_; }
    float* data() noexcept { return values_.data(); }
    float& at(std::initializer_list<std::size_t> indices);
    float at(std::initializer_list<std::size_t> indices) const;
    void reshape(TensorShape shape);

private:
    TensorShape shape_;
    std::vector<float> values_;
};

// Expand the original BF16 bits to FP32 without changing their numerical value.
// This first path handles the selected model's 1D and 2D weight tensors.
F32Tensor weight_to_f32(const WeightTensor& weight);

F32Tensor matvec(const F32Tensor& matrix, const F32Tensor& vector);
F32Tensor matmul(const F32Tensor& left, const F32Tensor& right);
F32Tensor add(const F32Tensor& left, const F32Tensor& right);
F32Tensor multiply(const F32Tensor& left, const F32Tensor& right);
F32Tensor scale(const F32Tensor& input, float factor);
F32Tensor silu(const F32Tensor& input);

// RMSNorm and softmax operate independently over each slice of the last axis.
F32Tensor rms_norm(const F32Tensor& input, const F32Tensor& weight, float epsilon);
// Negative infinity is permitted as a mask; every slice needs a finite value.
F32Tensor softmax(const F32Tensor& scores);

}  // namespace dengine
