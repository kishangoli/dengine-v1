#include "dengine/model_loader.hpp"

#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using json = nlohmann::json;

template <typename Function>
void rejects(Function function, const char* message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

json selected_config() {
    return {{"architectures", json::array({"LlamaForCausalLM"})},
            {"model_type", "llama"}, {"hidden_act", "silu"},
            {"tie_word_embeddings", true}, {"attention_bias", false},
            {"mlp_bias", false}, {"rope_interleaved", false},
            {"rope_scaling", nullptr}, {"rope_theta", 100000},
            {"rms_norm_eps", 1e-5}, {"torch_dtype", "bfloat16"},
            {"pretraining_tp", 1}, {"bos_token_id", 1}, {"eos_token_id", 2},
            {"vocab_size", 49152}, {"hidden_size", 576},
            {"intermediate_size", 1536}, {"num_hidden_layers", 30},
            {"num_attention_heads", 9}, {"num_key_value_heads", 3},
            {"max_position_embeddings", 8192}};
}

void write_text(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::binary);
    if (!output.write(contents.data(), static_cast<std::streamsize>(contents.size()))) {
        throw std::runtime_error("could not write test file");
    }
}

void write_safetensors(const std::filesystem::path& path, const std::string& header) {
    std::ofstream output(path, std::ios::binary);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((size >> shift) & 0xff));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!output) throw std::runtime_error("could not write Safetensors fixture");
}

void test_rejections() {
    using dengine::load_model;
    rejects([] { load_model({}); }, "explicit path required");
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
                           ("dengine-loader-" + std::to_string(unique));
    std::filesystem::create_directory(directory);
    try {
        rejects([&] { load_model(directory); }, "missing config and weights rejected");
        const auto config_path = directory / "config.json";
        const auto weight_path = directory / "model.safetensors";
        auto config = selected_config();
        write_text(config_path, config.dump());
        rejects([&] { load_model(directory); }, "missing weights rejected");
        write_safetensors(weight_path, "{}");

        write_text(config_path, "not JSON");
        rejects([&] { load_model(directory); }, "invalid config JSON rejected");
        config["num_attention_heads"] = 8;
        write_text(config_path, config.dump());
        rejects([&] { load_model(directory); }, "incompatible config rejected");
        config = selected_config();
        config["rope_theta"] = 50000;
        write_text(config_path, config.dump());
        rejects([&] { load_model(directory); }, "unsupported position encoding rejected");
        write_text(config_path, selected_config().dump());

        write_text(weight_path, "short");
        rejects([&] { load_model(directory); }, "truncated header prefix rejected");
        write_safetensors(weight_path, "not JSON");
        rejects([&] { load_model(directory); }, "invalid header JSON rejected");
        write_safetensors(weight_path, "{}");
        rejects([&] { load_model(directory); }, "missing tensors rejected");
        write_safetensors(weight_path, R"({"same":1,"same":2})");
        try {
            load_model(directory);
            throw std::runtime_error("duplicate keys were accepted");
        } catch (const std::invalid_argument& error) {
            if (std::string(error.what()).find("duplicate JSON keys") == std::string::npos) {
                throw;
            }
        }
        write_safetensors(weight_path, R"({"__metadata__":{"format":1}})");
        rejects([&] { load_model(directory); }, "invalid metadata rejected");

        // Header length says 100 bytes, but only 2 bytes follow.
        write_safetensors(weight_path, "{}");
        std::fstream bad_length(weight_path, std::ios::in | std::ios::out | std::ios::binary);
        bad_length.put(static_cast<char>(100));
        bad_length.close();
        rejects([&] { load_model(directory); }, "header outside file rejected");
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
    try {
        test_rejections();
        std::cout << "model loader rejection tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model loader test failed: " << error.what() << '\n';
        return 1;
    }
}
