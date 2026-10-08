#include "dengine/model_data.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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

void test_shapes_and_views() {
    using namespace dengine;
    TensorShape shape({2, 3});
    check(shape.element_count() == 6 && shape.byte_count(ElementType::f32) == 24,
          "row-major shape size");
    check(shape.flat_index({1, 2}) == 5, "last dimension changes fastest");
    rejects<std::invalid_argument>([] { TensorShape({}); }, "empty shape rejected");
    rejects<std::invalid_argument>([] { TensorShape({2, 0}); }, "zero dimension rejected");
    rejects<std::invalid_argument>([] { TensorShape({1, 1, 1, 1, 1, 1, 1, 1, 1}); },
                                   "rank above eight rejected");
    rejects<std::overflow_error>([] {
        TensorShape({std::numeric_limits<std::size_t>::max(), 2});
    }, "element overflow rejected");
    rejects<std::overflow_error>([] {
        TensorShape({std::numeric_limits<std::size_t>::max()}).byte_count(ElementType::f32);
    }, "byte overflow rejected");
    rejects<std::out_of_range>([&] { shape.flat_index({2, 0}); }, "axis bound rejected");
    rejects<std::out_of_range>([&] { shape.flat_index({0}); }, "wrong rank rejected");
    rejects<std::invalid_argument>([] { element_size(static_cast<ElementType>(9)); },
                                   "unknown type rejected");

    WeightTensor retained = [] {
        // Two little-endian BF16 values, each represented as its original 16 bits.
        WeightStorage storage({0x80, 0x3f, 0x00, 0x40});
        return storage.tensor(0, TensorShape({2}), ElementType::bf16);
    }();
    check(retained.bf16_bits({0}) == 0x3f80 && retained.bf16_bits({1}) == 0x4000,
          "BF16 weight view survives storage lifetime");
    rejects<std::invalid_argument>([&] { retained.f32({0}); }, "type mismatch rejected");
    rejects<std::out_of_range>([&] { retained.bf16_bits({2}); }, "weight index rejected");

    float sample = -1.25f;
    std::vector<std::uint8_t> bytes(sizeof(sample));
    std::memcpy(bytes.data(), &sample, sizeof(sample));
    WeightStorage fp32(std::move(bytes));
    check(fp32.tensor(0, TensorShape({1}), ElementType::f32).f32({0}) == sample,
          "FP32 weight round-trip");
    rejects<std::invalid_argument>([&] {
        fp32.tensor(1, TensorShape({1}), ElementType::bf16);
    }, "misaligned offset rejected");
    rejects<std::out_of_range>([&] {
        fp32.tensor(4, TensorShape({1}), ElementType::f32);
    }, "out-of-bounds tensor rejected");
    rejects<std::out_of_range>([&] {
        fp32.tensor(std::numeric_limits<std::size_t>::max() - 3, TensorShape({1}),
                    ElementType::f32);
    }, "huge offset rejected without wraparound");
}

void test_scratch_is_separate() {
    using namespace dengine;
    ScratchTensor scratch(TensorShape({2, 2}), ElementType::f32);
    check(scratch.f32({0, 0}) == 0, "scratch starts zeroed");
    scratch.set_f32({1, 0}, -3.5f);
    check(scratch.f32({1, 0}) == -3.5f && scratch.f32({0, 1}) == 0,
          "FP32 scratch round-trip and row-major position");
    rejects<std::out_of_range>([&] { scratch.set_f32({2, 0}, 1); },
                               "scratch index rejected");
    rejects<std::invalid_argument>([&] { scratch.bf16_bits({0, 0}); },
                                   "scratch accessor type checked");

    ScratchTensor bf16(TensorShape({2}), ElementType::bf16);
    bf16.set_bf16_bits({1}, 0x3f80);
    check(bf16.bf16_bits({1}) == 0x3f80, "BF16 scratch round-trip");
    rejects<std::length_error>([] {
        ScratchTensor(TensorShape({dengine::kMaxScratchBytes / 4 + 1}), ElementType::f32);
    }, "oversized scratch rejected before allocation");
}

void test_config_and_budget() {
    using namespace dengine;
    const auto config = smollm2_135m_config();
    config.validate();
    check(config.head_size() == 64 && config.parameter_count() == 134515008,
          "selected model layout and tied embedding count");
    auto invalid = config;
    invalid.query_heads = 8;
    rejects<std::invalid_argument>([&] { invalid.validate(); }, "bad head division rejected");
    invalid = config;
    invalid.layers = 0;
    rejects<std::invalid_argument>([&] { invalid.parameter_count(); },
                                   "zero layers rejected");
    invalid = config;
    invalid.vocab_size = std::numeric_limits<std::size_t>::max();
    rejects<std::overflow_error>([&] { invalid.parameter_count(); },
                                 "parameter overflow rejected");

    const auto estimate = estimate_memory(config, 269060552, 256 * kMiB, 1024, 1);
    check(estimate.model_file_bytes == 269060552 &&
          estimate.converted_weight_bytes == 538060032 &&
          estimate.scratch_bytes == 256 * kMiB &&
          estimate.kv_cache_bytes == 47185920 &&
          estimate.total_bytes == 1122741960,
          "memory estimate separates file, converted weights, scratch, and KV");
    rejects<std::invalid_argument>([&] {
        estimate_memory(config, 0, 0, 1, 1);
    }, "empty model rejected");
    rejects<std::invalid_argument>([&] {
        estimate_memory(config, kMaxModelFileBytes + 1, 0, 1, 1);
    }, "large model file rejected");
    rejects<std::invalid_argument>([&] {
        estimate_memory(config, 1, kMaxScratchBytes + 1, 1, 1);
    }, "large scratch reservation rejected");
    rejects<std::invalid_argument>([&] {
        estimate_memory(config, 1, 0, 1025, 1);
    }, "large context rejected");
    rejects<std::invalid_argument>([&] {
        estimate_memory(config, 1, 0, 1, 2);
    }, "multiple active requests rejected for initial policy");
    invalid = config;
    invalid.vocab_size = 400000000;
    rejects<std::length_error>([&] {
        estimate_memory(invalid, 1, 0, 1, 1);
    }, "large converted weights rejected");
}

void test_model_paths() {
    using namespace dengine;
    rejects<std::invalid_argument>([] { inspect_model_directory({}); },
                                   "empty model path rejected");
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
                           ("dengine-stage2-" + std::to_string(unique));
    std::filesystem::create_directory(directory);
    try {
        rejects<std::invalid_argument>([&] { inspect_model_directory(directory); },
                                       "missing files rejected");
        std::ofstream(directory / "config.json") << "{}";
        std::ofstream(directory / "model.safetensors") << "x";
        auto paths = inspect_model_directory(directory);
        check(paths.directory.is_absolute() && paths.model_file_bytes == 1 &&
              paths.config.filename() == "config.json" &&
              paths.weights.filename() == "model.safetensors",
              "explicit model directory resolved");
        std::filesystem::resize_file(paths.weights, 0);
        rejects<std::invalid_argument>([&] { inspect_model_directory(directory); },
                                       "empty weight file rejected");
        // A sparse file tests the path-size guard without writing a GiB of data.
        std::filesystem::resize_file(paths.weights, kMaxModelFileBytes + 1);
        rejects<std::invalid_argument>([&] { inspect_model_directory(directory); },
                                       "oversized weight file rejected");
        std::filesystem::remove(paths.weights);
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
    try {
        test_shapes_and_views();
        test_scratch_is_separate();
        test_config_and_budget();
        test_model_paths();
        std::cout << "model data tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model data test failed: " << error.what() << '\n';
        return 1;
    }
}
