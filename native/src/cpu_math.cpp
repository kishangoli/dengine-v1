#include "dengine/cpu_math.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace dengine {
namespace {

void require_f32_capacity(const TensorShape& shape) {
    if (shape.byte_count(ElementType::f32) > kMaxScratchBytes) {
        throw std::length_error("FP32 tensor exceeds 512 MiB working-buffer limit");
    }
}

void require_same_shape(const F32Tensor& left, const F32Tensor& right) {
    if (left.shape().dimensions() != right.shape().dimensions()) {
        throw std::invalid_argument("tensor shapes must match");
    }
}

std::size_t last_width(const F32Tensor& input) {
    return input.shape().dimensions().back();
}

}  // namespace

F32Tensor::F32Tensor(TensorShape shape) : shape_(std::move(shape)) {
    require_f32_capacity(shape_);
    try {
        values_.resize(shape_.element_count(), 0.0f);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("could not allocate FP32 working tensor");
    }
}

F32Tensor::F32Tensor(TensorShape shape, std::vector<float> values)
    : shape_(std::move(shape)), values_(std::move(values)) {
    require_f32_capacity(shape_);
    if (values_.size() != shape_.element_count()) {
        throw std::invalid_argument("FP32 value count does not match tensor shape");
    }
}

float& F32Tensor::at(std::initializer_list<std::size_t> indices) {
    return values_[shape_.flat_index(indices)];
}

float F32Tensor::at(std::initializer_list<std::size_t> indices) const {
    return values_[shape_.flat_index(indices)];
}

void F32Tensor::reshape(TensorShape shape) {
    if (shape.element_count() != shape_.element_count()) {
        throw std::invalid_argument("reshape must preserve element count");
    }
    shape_ = std::move(shape);
}

F32Tensor weight_to_f32(const WeightTensor& weight) {
    const auto& dimensions = weight.shape().dimensions();
    if (dimensions.size() != 1 && dimensions.size() != 2) {
        throw std::invalid_argument("only 1D and 2D weight tensors are supported");
    }
    F32Tensor output(weight.shape());
    for (std::size_t index = 0; index < output.values().size(); ++index) {
        const auto read = [&](std::size_t row, std::size_t column) {
            if (weight.type() == ElementType::f32) {
                return dimensions.size() == 1 ? weight.f32({column}) : weight.f32({row, column});
            }
            if (weight.type() != ElementType::bf16) {
                throw std::invalid_argument("unsupported weight element type");
            }
            const std::uint16_t bits = dimensions.size() == 1
                ? weight.bf16_bits({column}) : weight.bf16_bits({row, column});
            const std::uint32_t expanded = std::uint32_t(bits) << 16;
            float value;
            std::memcpy(&value, &expanded, sizeof(value));
            return value;
        };
        const auto width = dimensions.back();
        output.data()[index] = read(index / width, index % width);
    }
    return output;
}

F32Tensor matvec(const F32Tensor& matrix, const F32Tensor& vector) {
    const auto& dims = matrix.shape().dimensions();
    if (dims.size() != 2 || vector.shape().dimensions().size() != 1 ||
        dims[1] != vector.shape().dimensions()[0]) {
        throw std::invalid_argument("matvec requires [rows, columns] times [columns]");
    }
    F32Tensor output(TensorShape({dims[0]}));
    for (std::size_t row = 0; row < dims[0]; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < dims[1]; ++column) {
            sum += static_cast<double>(matrix.values()[row * dims[1] + column]) *
                   vector.values()[column];
        }
        output.data()[row] = static_cast<float>(sum);
    }
    return output;
}

F32Tensor matmul(const F32Tensor& left, const F32Tensor& right) {
    const auto& a = left.shape().dimensions();
    const auto& b = right.shape().dimensions();
    if (a.size() != 2 || b.size() != 2 || a[1] != b[0]) {
        throw std::invalid_argument("matmul requires [M, K] times [K, N]");
    }
    F32Tensor output(TensorShape({a[0], b[1]}));
    for (std::size_t row = 0; row < a[0]; ++row) {
        for (std::size_t column = 0; column < b[1]; ++column) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < a[1]; ++inner) {
                sum += static_cast<double>(left.values()[row * a[1] + inner]) *
                       right.values()[inner * b[1] + column];
            }
            output.data()[row * b[1] + column] = static_cast<float>(sum);
        }
    }
    return output;
}

F32Tensor add(const F32Tensor& left, const F32Tensor& right) {
    require_same_shape(left, right);
    F32Tensor output(left.shape());
    for (std::size_t i = 0; i < left.values().size(); ++i) {
        output.data()[i] = left.values()[i] + right.values()[i];
    }
    return output;
}

F32Tensor multiply(const F32Tensor& left, const F32Tensor& right) {
    require_same_shape(left, right);
    F32Tensor output(left.shape());
    for (std::size_t i = 0; i < left.values().size(); ++i) {
        output.data()[i] = left.values()[i] * right.values()[i];
    }
    return output;
}

F32Tensor scale(const F32Tensor& input, float factor) {
    if (!std::isfinite(factor)) throw std::invalid_argument("scale factor must be finite");
    F32Tensor output(input.shape());
    for (std::size_t i = 0; i < input.values().size(); ++i) {
        output.data()[i] = input.values()[i] * factor;
    }
    return output;
}

F32Tensor silu(const F32Tensor& input) {
    F32Tensor output(input.shape());
    for (std::size_t i = 0; i < input.values().size(); ++i) {
        const double value = input.values()[i];
        if (!std::isfinite(value)) throw std::invalid_argument("SiLU input must be finite");
        const double sigmoid = value >= 0 ? 1.0 / (1.0 + std::exp(-value))
                                          : std::exp(value) / (1.0 + std::exp(value));
        output.data()[i] = static_cast<float>(value * sigmoid);
    }
    return output;
}

F32Tensor rms_norm(const F32Tensor& input, const F32Tensor& weight, float epsilon) {
    const auto width = last_width(input);
    if (weight.shape().dimensions() != std::vector<std::size_t>{width}) {
        throw std::invalid_argument("RMSNorm weight must match the last dimension");
    }
    if (!std::isfinite(epsilon) || epsilon <= 0) {
        throw std::invalid_argument("RMSNorm epsilon must be positive and finite");
    }
    F32Tensor output(input.shape());
    for (std::size_t offset = 0; offset < input.values().size(); offset += width) {
        double square_sum = 0.0;
        for (std::size_t i = 0; i < width; ++i) {
            const double value = input.values()[offset + i];
            if (!std::isfinite(value) || !std::isfinite(weight.values()[i])) {
                throw std::invalid_argument("RMSNorm values must be finite");
            }
            square_sum += value * value;
        }
        const double inverse_rms = 1.0 / std::sqrt(square_sum / width + epsilon);
        for (std::size_t i = 0; i < width; ++i) {
            output.data()[offset + i] = static_cast<float>(
                static_cast<double>(input.values()[offset + i]) * inverse_rms * weight.values()[i]);
        }
    }
    return output;
}

F32Tensor softmax(const F32Tensor& scores) {
    const auto width = last_width(scores);
    F32Tensor output(scores.shape());
    for (std::size_t offset = 0; offset < scores.values().size(); offset += width) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < width; ++i) {
            const float value = scores.values()[offset + i];
            if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
                throw std::invalid_argument("softmax scores must be finite or -infinity");
            }
            if (value > maximum) maximum = value;
        }
        if (!std::isfinite(maximum)) {
            throw std::invalid_argument("softmax slice has no unmasked value");
        }
        double sum = 0.0;
        for (std::size_t i = 0; i < width; ++i) {
            const double exponential = std::exp(static_cast<double>(scores.values()[offset + i]) - maximum);
            output.data()[offset + i] = static_cast<float>(exponential);
            sum += exponential;
        }
        for (std::size_t i = 0; i < width; ++i) {
            output.data()[offset + i] = static_cast<float>(output.values()[offset + i] / sum);
        }
    }
    return output;
}

}  // namespace dengine
