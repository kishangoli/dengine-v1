#include "dengine/cpu_math.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using dengine::F32Tensor;
using dengine::TensorShape;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void close(float actual, double expected, double tolerance, const char* message) {
    if (!std::isfinite(actual) || std::abs(static_cast<double>(actual) - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void rejects(Function function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_storage_and_weight_conversion() {
    F32Tensor values(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
    check(values.at({1, 2}) == 6, "row-major indexing");
    values.reshape(TensorShape({3, 2}));
    check(values.at({2, 1}) == 6 && values.values()[0] == 1,
          "reshape preserves storage order");
    values.at({0, 1}) = -7;
    check(values.values()[1] == -7, "working tensor is mutable");
    rejects<std::invalid_argument>([&] { values.reshape(TensorShape({7})); },
                                   "reshape changing count rejected");
    rejects<std::invalid_argument>([] {
        F32Tensor(TensorShape({2, 2}), {1, 2, 3});
    }, "wrong value count rejected");
    rejects<std::length_error>([] {
        F32Tensor(TensorShape({dengine::kMaxScratchBytes / 4 + 1}));
    }, "oversized FP32 allocation rejected before allocation");

    // Raw BF16 bits for [1, -2, 0.5, -0.25], stored little-endian.
    dengine::WeightStorage storage({0x80, 0x3f, 0x00, 0xc0,
                                    0x00, 0x3f, 0x80, 0xbe});
    auto converted = dengine::weight_to_f32(
        storage.tensor(0, TensorShape({2, 2}), dengine::ElementType::bf16));
    check(converted.shape().dimensions() == std::vector<std::size_t>({2, 2}) &&
          converted.at({0, 0}) == 1 && converted.at({0, 1}) == -2 &&
          converted.at({1, 0}) == 0.5f && converted.at({1, 1}) == -0.25f,
          "BF16 bits expand exactly to FP32");

    const float sample = 1.25f;
    std::vector<std::uint8_t> raw(sizeof(sample));
    std::memcpy(raw.data(), &sample, sizeof(sample));
    dengine::WeightStorage fp32(std::move(raw));
    check(dengine::weight_to_f32(
              fp32.tensor(0, TensorShape({1}), dengine::ElementType::f32)).at({0}) == sample,
          "FP32 weight conversion preserves value");
}

void test_matrix_and_vector_operations() {
    const F32Tensor matrix(TensorShape({2, 3}), {1, 2, -1, 0, 3, 4});
    const F32Tensor vector(TensorShape({3}), {2, -1, 0.5f});
    const auto product = dengine::matvec(matrix, vector);
    check(product.shape().dimensions() == std::vector<std::size_t>({2}) &&
          product.at({0}) == -0.5f && product.at({1}) == -1,
          "matvec matches hand-calculated dot products");
    check(dengine::matvec(matrix, vector).values() == product.values(),
          "matvec repeats deterministically");

    const F32Tensor left(TensorShape({2, 2}), {1, 2, 3, 4});
    const F32Tensor right(TensorShape({2, 3}), {5, 6, 7, 8, 9, 10});
    const auto result = dengine::matmul(left, right);
    check(result.shape().dimensions() == std::vector<std::size_t>({2, 3}) &&
          result.values() == std::vector<float>({21, 24, 27, 47, 54, 61}),
          "matmul matches hand-calculated rows and columns");

    const auto added = dengine::add(left, F32Tensor(TensorShape({2, 2}), {4, 3, 2, 1}));
    check(added.values() == std::vector<float>({5, 5, 5, 5}), "elementwise add");
    const auto multiplied = dengine::multiply(left, left);
    check(multiplied.values() == std::vector<float>({1, 4, 9, 16}),
          "elementwise multiply");
    check(dengine::scale(left, -0.5f).values() ==
              std::vector<float>({-0.5f, -1, -1.5f, -2}), "scale");

    rejects<std::invalid_argument>([&] {
        dengine::matvec(matrix, F32Tensor(TensorShape({2}), {1, 2}));
    }, "matvec incompatible dimensions rejected");
    rejects<std::invalid_argument>([&] { dengine::matmul(matrix, right); },
                                   "matmul incompatible dimensions rejected");
    rejects<std::invalid_argument>([&] { dengine::add(left, vector); },
                                   "elementwise mismatched shapes rejected");
    rejects<std::invalid_argument>([&] {
        dengine::scale(left, std::numeric_limits<float>::infinity());
    }, "nonfinite scale rejected");
}

void test_activation_normalization_and_softmax() {
    const F32Tensor input(TensorShape({5}), {-10, -1, 0, 1, 10});
    const auto activated = dengine::silu(input);
    const std::vector<double> expected_silu = {
        -0.0004539786870243439, -0.2689414213699951, 0,
        0.7310585786300049, 9.999546021312975};
    for (std::size_t i = 0; i < expected_silu.size(); ++i) {
        close(activated.values()[i], expected_silu[i], 2e-6, "SiLU reference value");
    }
    close(dengine::silu(F32Tensor(TensorShape({2}), {-1000, 1000})).at({0}),
          0, 1e-6, "SiLU is stable for large negative input");

    const F32Tensor rows(TensorShape({3, 2}), {3, 4, 0, 5, -2, 1});
    const F32Tensor weights(TensorShape({2}), {2, 0.5f});
    const auto normalized = dengine::rms_norm(rows, weights, 1e-5f);
    const std::vector<double> expected_norm = {
        1.6970555960256115, 0.5656851986752038,
        0, 0.7071064983440047,
        -2.529817068505626, 0.31622713356320326};
    for (std::size_t i = 0; i < expected_norm.size(); ++i) {
        close(normalized.values()[i], expected_norm[i], 2e-6,
              "RMSNorm reference value");
    }
    rejects<std::invalid_argument>([&] { dengine::rms_norm(rows, weights, 0); },
                                   "nonpositive epsilon rejected");
    rejects<std::invalid_argument>([&] {
        dengine::rms_norm(rows, F32Tensor(TensorShape({3}), {1, 1, 1}), 1e-5f);
    }, "RMSNorm mismatched weight rejected");

    const float negative_infinity = -std::numeric_limits<float>::infinity();
    const F32Tensor scores(TensorShape({2, 3}), {1000, 1001, 1002,
                                                 0, negative_infinity, 0});
    const auto probabilities = dengine::softmax(scores);
    close(probabilities.at({0, 0}), 0.09003057317038046, 1e-6,
          "stable softmax first score");
    close(probabilities.at({0, 1}), 0.24472847105479764, 1e-6,
          "stable softmax second score");
    close(probabilities.at({0, 2}), 0.6652409557748218, 1e-6,
          "stable softmax third score");
    check(probabilities.at({1, 0}) == 0.5f && probabilities.at({1, 1}) == 0 &&
          probabilities.at({1, 2}) == 0.5f, "masked position has zero probability");
    close(probabilities.at({0, 0}) + probabilities.at({0, 1}) +
          probabilities.at({0, 2}), 1.0, 1e-6, "softmax slice sums to one");
    rejects<std::invalid_argument>([&] {
        dengine::softmax(F32Tensor(TensorShape({2}),
                                   {negative_infinity, negative_infinity}));
    }, "fully masked slice rejected");
    rejects<std::invalid_argument>([] {
        dengine::softmax(F32Tensor(TensorShape({2}),
            {0, std::numeric_limits<float>::quiet_NaN()}));
    }, "NaN score rejected");
    rejects<std::invalid_argument>([] {
        dengine::softmax(F32Tensor(TensorShape({2}),
            {0, std::numeric_limits<float>::infinity()}));
    }, "positive infinity score rejected");
}

}  // namespace

int main() {
    try {
        test_storage_and_weight_conversion();
        test_matrix_and_vector_operations();
        test_activation_normalization_and_softmax();
        std::cout << "CPU math tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CPU math test failed: " << error.what() << '\n';
        return 1;
    }
}
