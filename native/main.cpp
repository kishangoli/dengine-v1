#include <cctype>
#include <iostream>
#include <string>

#include "nlohmann/json.hpp"
#include "dengine/model_loader.hpp"

namespace {

using json = nlohmann::json;

// A cap on each wire message bounds memory even when a sender omits newlines.
constexpr std::size_t kMaxLineBytes = 64 * 1024;

enum class ReadResult { line, too_large, end, failure };

ReadResult read_line(std::string& line) {
    line.clear();
    bool too_large = false;

    for (;;) {
        const auto next = std::cin.get();
        if (next == std::char_traits<char>::eof()) {
            if (std::cin.bad()) {
                return ReadResult::failure;
            }
            if (too_large) {
                return ReadResult::too_large;
            }
            return line.empty() ? ReadResult::end : ReadResult::line;
        }
        if (next == '\n') {
            if (too_large) {
                return ReadResult::too_large;
            }
            // Tolerate CRLF framing while counting the CR toward the byte limit.
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return ReadResult::line;
        }
        if (!too_large) {
            if (line.size() == kMaxLineBytes) {
                too_large = true;
                line.clear();
            } else {
                line.push_back(static_cast<char>(next));
            }
        }
        // Once over the limit, drain through the next newline before responding.
    }
}

bool nonblank(const std::string& value) {
    for (unsigned char c : value) {
        if (!std::isspace(c)) {
            return true;
        }
    }
    return false;
}

json error_response(const json& id, const char* code, const char* message) {
    return {{"id", id}, {"ok", false},
            {"error", {{"code", code}, {"message", message}}}};
}

json process(const std::string& line) {
    const json request = json::parse(line, nullptr, false);
    if (request.is_discarded()) {
        return error_response(nullptr, "invalid_json", "request must be valid JSON");
    }
    if (!request.is_object()) {
        return error_response(nullptr, "invalid_request", "request must be a JSON object");
    }

    json id = nullptr;
    const auto id_field = request.find("id");
    if (id_field != request.end() && id_field->is_string()) {
        const std::string candidate = id_field->get<std::string>();
        if (nonblank(candidate)) {
            id = candidate;
        }
    }
    if (id.is_null()) {
        return error_response(nullptr, "invalid_request", "id must be a nonempty string");
    }

    const auto prompt_field = request.find("prompt");
    if (prompt_field == request.end() || !prompt_field->is_string() ||
        !nonblank(prompt_field->get_ref<const std::string&>())) {
        return error_response(id, "invalid_request", "prompt must be a nonempty string");
    }

    return {{"id", id}, {"ok", true},
            {"text", "echo: " + prompt_field->get<std::string>()}};
}

bool write_response(const json& response) {
    std::cout << response.dump() << '\n' << std::flush;
    return static_cast<bool>(std::cout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 1) {
        if (argc != 3 || std::string(argv[1]) != "--inspect-model") {
            std::cerr << "usage: dengine-native [--inspect-model MODEL_DIRECTORY]\n";
            return 2;
        }
        try {
            const auto model = dengine::load_model(argv[2]);
            json tensors = json::array();
            for (const auto& item : model.tensors()) {
                const auto first_bits = model.tensor(item.first).bf16_bits(
                    item.second.shape.dimensions().size() == 1
                        ? std::initializer_list<std::size_t>{0}
                        : std::initializer_list<std::size_t>{0, 0});
                tensors.push_back({{"name", item.first},
                                   {"shape", item.second.shape.dimensions()},
                                   {"dtype", "BF16"},
                                   {"first_bf16_bits", first_bits}});
            }
            const auto& config = model.config();
            std::cout << json({{"model", "SmolLM2-135M-Instruct"},
                               {"tensor_count", tensors.size()},
                               {"file_bytes", model.file_bytes()},
                               {"parameters", config.parameter_count()},
                               {"config", {{"layers", config.layers},
                                           {"hidden_size", config.hidden_size},
                                           {"query_heads", config.query_heads},
                                           {"kv_heads", config.kv_heads},
                                           {"vocab_size", config.vocab_size}}},
                               {"tensors", std::move(tensors)}}).dump() << '\n';
            return std::cout ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "model inspection failed: " << error.what() << '\n';
            return 1;
        }
    }
    std::string line;
    for (;;) {
        switch (read_line(line)) {
            case ReadResult::end:
                return 0;
            case ReadResult::failure:
                std::cerr << "failed to read stdin\n";
                return 1;
            case ReadResult::too_large:
                if (!write_response(error_response(
                        nullptr, "request_too_large", "request exceeds 65536 bytes"))) {
                    std::cerr << "failed to write stdout\n";
                    return 1;
                }
                break;
            case ReadResult::line:
                if (!write_response(process(line))) {
                    std::cerr << "failed to write stdout\n";
                    return 1;
                }
                break;
        }
    }
}
