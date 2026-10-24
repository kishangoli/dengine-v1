#include "dengine/transformer_block.hpp"

#include "nlohmann/json.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using dengine::F32Tensor;
using dengine::TensorShape;
using json = nlohmann::json;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void compare(const std::vector<float>& actual, const json& expected,
             double absolute_tolerance, double relative_tolerance,
             const std::string& label) {
    check(actual.size() == expected.size(), "reference value count differs");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double target = expected.at(i).get<double>();
        const double difference = std::abs(double(actual[i]) - target);
        if (!std::isfinite(actual[i]) || difference > absolute_tolerance + relative_tolerance * std::abs(target)) {
            throw std::runtime_error(label + " differs from Python at index " + std::to_string(i) +
                                     ": C++=" + std::to_string(actual[i]) +
                                     ", Python=" + std::to_string(target));
        }
    }
}

F32Tensor embeddings(const dengine::LoadedModel& model, const json& ids) {
    const auto weights = model.tensor("model.embed_tokens.weight");
    const auto width = model.config().hidden_size;
    F32Tensor result(TensorShape({ids.size(), width}));
    for (std::size_t row = 0; row < ids.size(); ++row) {
        const auto id = ids.at(row).get<std::size_t>();
        check(id < model.config().vocab_size, "fixture token ID out of range");
        for (std::size_t column = 0; column < width; ++column) {
            const auto bits = std::uint32_t(weights.bf16_bits({id, column})) << 16;
            float value;
            static_assert(sizeof(value) == sizeof(bits));
            std::memcpy(&value, &bits, sizeof(value));
            result.at({row, column}) = value;
        }
    }
    return result;
}

template <typename Function>
void rejects(Function function, const char* label) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::out_of_range&) {
        return;
    }
    throw std::runtime_error(label);
}

void verify_case(const dengine::LoadedModel& model, const dengine::TransformerBlock& block,
                 const json& fixture) {
    const auto input = embeddings(model, fixture.at("input_ids"));
    const auto result = block.forward_detailed(input, fixture.at("start_position").get<std::size_t>());
    const auto count = fixture.at("input_ids").size();
    check(result.output.shape().dimensions() == std::vector<std::size_t>({count, model.config().hidden_size}),
          "block output shape");
    compare({result.normalized_input.values().begin(), result.normalized_input.values().begin() + 8},
            fixture.at("normalized_first_8"), 3e-5, 3e-5, "input normalization");
    compare({result.attention_output.values().begin(), result.attention_output.values().begin() + 8},
            fixture.at("attention_output_first_8"), 3e-4, 3e-4, "attention output");
    for (std::size_t row = 0; row < count; ++row) {
        const std::vector<float> output_row(result.output.values().begin() + row * model.config().hidden_size,
            result.output.values().begin() + (row + 1) * model.config().hidden_size);
        compare(output_row, fixture.at("output").at(row), 5e-4, 5e-4,
                fixture.at("name").get<std::string>() + " output row " + std::to_string(row));
    }
    const auto repeated = block.forward(input, fixture.at("start_position").get<std::size_t>());
    check(repeated.values() == result.output.values(), "block should be deterministic");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("expected MODEL_DIRECTORY FIXTURE_JSON");
        const auto model = dengine::load_model(argv[1]);
        std::ifstream input(argv[2]);
        if (!input) throw std::runtime_error("cannot open block fixture");
        const auto fixture = json::parse(input);
        check(fixture.at("layer").get<std::size_t>() == 0, "fixture layer mismatch");
        const dengine::TransformerBlock block(model, 0);
        for (const auto& sample : fixture.at("cases")) verify_case(model, block, sample);

        const auto& samples = fixture.at("cases");
        const auto full = block.forward(embeddings(model, samples.at(1).at("input_ids")));
        for (std::size_t prefix_length = 1; prefix_length < samples.at(1).at("input_ids").size(); ++prefix_length) {
            json prefix_ids = json::array();
            for (std::size_t i = 0; i < prefix_length; ++i) {
                prefix_ids.push_back(samples.at(1).at("input_ids").at(i));
            }
            const auto prefix = block.forward(embeddings(model, prefix_ids));
            for (std::size_t i = 0; i < prefix_length * model.config().hidden_size; ++i) {
                if (std::abs(prefix.values()[i] - full.values()[i]) > 1e-6f) {
                    throw std::runtime_error("future tokens changed an earlier token's output");
                }
            }
        }
        rejects([&] { block.forward(F32Tensor(TensorShape({2, 2}))); }, "bad input width accepted");
        rejects([&] { block.forward(embeddings(model, samples.at(0).at("input_ids")), 1024); },
                "context beyond current limit accepted");
        rejects([&] { dengine::TransformerBlock(model, model.config().layers); },
                "invalid layer accepted");
        std::cout << "transformer block matches pinned Python reference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
