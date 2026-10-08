#include "dengine/model_loader.hpp"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dengine {
namespace {

using json = nlohmann::json;
constexpr std::size_t kMaxConfigBytes = 64 * 1024;
constexpr std::size_t kMaxHeaderBytes = 16 * 1024 * 1024;
constexpr std::uint64_t kInitialScratchReservation = 256 * kMiB;

std::vector<std::uint8_t> read_file(const std::filesystem::path& path, std::uint64_t limit) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size > limit ||
        size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("file missing, empty, or too large: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open file: " + path.string());
    std::vector<std::uint8_t> bytes;
    try {
        bytes.resize(static_cast<std::size_t>(size));
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("not enough memory to read: " + path.string());
    }
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (input.gcount() != static_cast<std::streamsize>(size) || input.peek() != EOF) {
        throw std::runtime_error("file changed or truncated while reading: " + path.string());
    }
    return bytes;
}

json parse_object(const std::uint8_t* data, std::size_t size, const char* label) {
    std::vector<std::set<std::string>> object_keys;
    bool duplicate = false;
    const auto callback = [&](int, json::parse_event_t event, json& parsed) {
        if (event == json::parse_event_t::object_start) {
            object_keys.emplace_back();
        } else if (event == json::parse_event_t::key) {
            if (!object_keys.back().insert(parsed.get<std::string>()).second) {
                duplicate = true;
            }
        } else if (event == json::parse_event_t::object_end) {
            object_keys.pop_back();
        }
        return true;
    };
    json value = json::parse(data, data + size, callback, false);
    if (duplicate) {
        throw std::invalid_argument(std::string(label) + " contains duplicate JSON keys");
    }
    if (!value.is_object()) {
        throw std::invalid_argument(std::string(label) + " must be a valid JSON object");
    }
    return value;
}

const json& field(const json& object, const char* name) {
    const auto it = object.find(name);
    if (it == object.end()) {
        throw std::invalid_argument(std::string("missing field: ") + name);
    }
    return *it;
}

std::size_t unsigned_number(const json& value, const char* label) {
    if (!value.is_number_unsigned()) {
        throw std::invalid_argument(std::string(label) + " must be a nonnegative integer");
    }
    const auto number = value.get<std::uint64_t>();
    if (number > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(std::string(label) + " does not fit in size_t");
    }
    return static_cast<std::size_t>(number);
}

void require_equal(const json& config, const char* key, const json& expected) {
    if (field(config, key) != expected) {
        throw std::invalid_argument(std::string("unsupported model config field: ") + key);
    }
}

ModelConfig parse_config(const std::vector<std::uint8_t>& bytes) {
    const auto config = parse_object(bytes.data(), bytes.size(), "config.json");
    require_equal(config, "architectures", json::array({"LlamaForCausalLM"}));
    require_equal(config, "model_type", "llama");
    require_equal(config, "hidden_act", "silu");
    require_equal(config, "tie_word_embeddings", true);
    require_equal(config, "attention_bias", false);
    require_equal(config, "mlp_bias", false);
    require_equal(config, "rope_interleaved", false);
    require_equal(config, "rope_scaling", nullptr);
    require_equal(config, "rope_theta", 100000);
    require_equal(config, "rms_norm_eps", 1e-5);
    require_equal(config, "torch_dtype", "bfloat16");
    require_equal(config, "pretraining_tp", 1);
    require_equal(config, "bos_token_id", 1);
    require_equal(config, "eos_token_id", 2);

    const ModelConfig found{
        unsigned_number(field(config, "vocab_size"), "vocab_size"),
        unsigned_number(field(config, "hidden_size"), "hidden_size"),
        unsigned_number(field(config, "intermediate_size"), "intermediate_size"),
        unsigned_number(field(config, "num_hidden_layers"), "num_hidden_layers"),
        unsigned_number(field(config, "num_attention_heads"), "num_attention_heads"),
        unsigned_number(field(config, "num_key_value_heads"), "num_key_value_heads"),
        unsigned_number(field(config, "max_position_embeddings"), "max_position_embeddings")};
    found.validate();
    const auto expected = smollm2_135m_config();
    if (found.vocab_size != expected.vocab_size || found.hidden_size != expected.hidden_size ||
        found.intermediate_size != expected.intermediate_size || found.layers != expected.layers ||
        found.query_heads != expected.query_heads || found.kv_heads != expected.kv_heads ||
        found.max_positions != expected.max_positions) {
        throw std::invalid_argument("config dimensions do not match SmolLM2-135M-Instruct");
    }
    return found;
}

std::map<std::string, std::vector<std::size_t>> expected_shapes(const ModelConfig& config) {
    const auto hidden = config.hidden_size;
    const auto kv_width = config.kv_heads * config.head_size();
    const auto mlp = config.intermediate_size;
    std::map<std::string, std::vector<std::size_t>> expected;
    expected.emplace("model.embed_tokens.weight", std::vector<std::size_t>{config.vocab_size, hidden});
    expected.emplace("model.norm.weight", std::vector<std::size_t>{hidden});
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        expected.emplace(prefix + "self_attn.q_proj.weight", std::vector<std::size_t>{hidden, hidden});
        expected.emplace(prefix + "self_attn.k_proj.weight", std::vector<std::size_t>{kv_width, hidden});
        expected.emplace(prefix + "self_attn.v_proj.weight", std::vector<std::size_t>{kv_width, hidden});
        expected.emplace(prefix + "self_attn.o_proj.weight", std::vector<std::size_t>{hidden, hidden});
        expected.emplace(prefix + "mlp.gate_proj.weight", std::vector<std::size_t>{mlp, hidden});
        expected.emplace(prefix + "mlp.up_proj.weight", std::vector<std::size_t>{mlp, hidden});
        expected.emplace(prefix + "mlp.down_proj.weight", std::vector<std::size_t>{hidden, mlp});
        expected.emplace(prefix + "input_layernorm.weight", std::vector<std::size_t>{hidden});
        expected.emplace(prefix + "post_attention_layernorm.weight", std::vector<std::size_t>{hidden});
    }
    return expected;
}

std::uint64_t header_length(const std::vector<std::uint8_t>& file) {
    if (file.size() < 8) throw std::invalid_argument("Safetensors file is truncated");
    std::uint64_t length = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        length |= std::uint64_t(file[shift / 8]) << shift;
    }
    if (length == 0 || length > kMaxHeaderBytes || length > file.size() - 8) {
        throw std::invalid_argument("invalid Safetensors header length");
    }
    return length;
}

struct Range {
    std::size_t begin;
    std::size_t end;
};

std::map<std::string, TensorDescriptor> parse_tensors(const std::vector<std::uint8_t>& file,
                                                        const ModelConfig& config) {
    const auto length = header_length(file);
    const auto header = parse_object(file.data() + 8, static_cast<std::size_t>(length),
                                     "Safetensors header");
    const auto expected = expected_shapes(config);
    const auto metadata = header.find("__metadata__");
    if (metadata != header.end()) {
        if (!metadata->is_object()) throw std::invalid_argument("invalid Safetensors metadata");
        for (auto it = metadata->begin(); it != metadata->end(); ++it) {
            if (!it.value().is_string()) {
                throw std::invalid_argument("Safetensors metadata values must be strings");
            }
        }
    }
    if (header.size() != expected.size() + (metadata != header.end() ? 1 : 0)) {
        throw std::invalid_argument("Safetensors tensor count does not match model");
    }
    const std::size_t data_begin = 8 + static_cast<std::size_t>(length);
    const std::size_t data_size = file.size() - data_begin;
    std::map<std::string, TensorDescriptor> tensors;
    std::vector<Range> ranges;
    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const auto wanted = expected.find(it.key());
        if (wanted == expected.end()) {
            throw std::invalid_argument("unexpected tensor: " + it.key());
        }
        const auto& entry = it.value();
        if (!entry.is_object() || entry.size() != 3 || field(entry, "dtype") != "BF16") {
            throw std::invalid_argument("unsupported tensor entry or dtype: " + it.key());
        }
        const auto& shape_json = field(entry, "shape");
        if (!shape_json.is_array()) throw std::invalid_argument("invalid tensor shape: " + it.key());
        std::vector<std::size_t> dims;
        for (const auto& dim : shape_json) dims.push_back(unsigned_number(dim, "tensor dimension"));
        TensorShape shape(std::move(dims));
        if (shape.dimensions() != wanted->second) {
            throw std::invalid_argument("wrong tensor shape: " + it.key());
        }
        const auto& offsets = field(entry, "data_offsets");
        if (!offsets.is_array() || offsets.size() != 2) {
            throw std::invalid_argument("invalid tensor offsets: " + it.key());
        }
        const auto start = unsigned_number(offsets[0], "tensor offset");
        const auto end = unsigned_number(offsets[1], "tensor offset");
        if (start > end || end > data_size || end - start != shape.byte_count(ElementType::bf16)) {
            throw std::invalid_argument("tensor range is outside file or has wrong size: " + it.key());
        }
        if (start % 2 != 0 || data_begin % 2 != 0) {
            throw std::invalid_argument("BF16 tensor offset is not aligned: " + it.key());
        }
        ranges.push_back({start, end});
        tensors.emplace(it.key(), TensorDescriptor{std::move(shape), ElementType::bf16,
                                                    data_begin + start});
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) {
        return a.begin < b.begin;
    });
    std::size_t cursor = 0;
    for (const auto& range : ranges) {
        if (range.begin != cursor) {
            throw std::invalid_argument("Safetensors data contains a gap or overlap");
        }
        cursor = range.end;
    }
    if (cursor != data_size) {
        throw std::invalid_argument("Safetensors file has unindexed data");
    }
    if (cursor != config.parameter_count() * 2) {
        throw std::invalid_argument("Safetensors parameter count does not match config");
    }
    return tensors;
}

}  

LoadedModel::LoadedModel(ModelConfig config, WeightStorage storage,
                         std::map<std::string, TensorDescriptor> tensors)
    : config_(config), storage_(std::move(storage)), tensors_(std::move(tensors)) {}

WeightTensor LoadedModel::tensor(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::out_of_range("unknown tensor: " + name);
    return storage_.tensor(it->second.file_offset, it->second.shape, it->second.type);
}

LoadedModel load_model(const std::filesystem::path& directory) {
    const auto paths = inspect_model_directory(directory);
    const auto config_bytes = read_file(paths.config, kMaxConfigBytes);
    const auto config = parse_config(config_bytes);
    // Reserve for the original file, a later FP32 conversion, scratch, and KV.
    estimate_memory(config, paths.model_file_bytes, kInitialScratchReservation,
                    kInitialContextLimit, kInitialActiveRequests);
    auto file = read_file(paths.weights, kMaxModelFileBytes);
    if (file.size() != paths.model_file_bytes) {
        throw std::runtime_error("model file changed while loading");
    }
    auto tensors = parse_tensors(file, config);
    return LoadedModel(config, WeightStorage(std::move(file)), std::move(tensors));
}

} 
