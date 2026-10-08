#include "dengine/tokenizer.hpp"

#include "nlohmann/json.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json process(const dengine::Tokenizer& tokenizer, const std::string& line) {
    const auto request = json::parse(line);
    if (!request.is_object()) throw std::invalid_argument("request must be a JSON object");
    const auto operation = request.at("op").get<std::string>();
    if (operation == "encode") {
        const auto text = request.at("text").get<std::string>();
        const auto ids = tokenizer.encode(text);
        return {{"ok", true}, {"ids", ids}};
    }
    if (operation == "decode") {
        const auto& items = request.at("ids");
        if (!items.is_array()) throw std::invalid_argument("ids must be an array");
        std::vector<std::uint32_t> ids;
        for (const auto& item : items) {
            if (!item.is_number_unsigned() || item.get<std::uint64_t>() >= tokenizer.vocab_size()) {
                throw std::invalid_argument("invalid token ID");
            }
            ids.push_back(item.get<std::uint32_t>());
        }
        return {{"ok", true},
                {"text", tokenizer.decode(ids, request.value("skip_special_tokens", false))}};
    }
    if (operation == "chat") {
        const auto& items = request.at("messages");
        if (!items.is_array()) throw std::invalid_argument("messages must be an array");
        std::vector<dengine::ChatMessage> messages;
        for (const auto& item : items) {
            messages.push_back({item.at("role").get<std::string>(),
                                item.at("content").get<std::string>()});
        }
        const auto formatted = tokenizer.format_chat(
            messages, request.value("add_generation_prompt", true));
        return {{"ok", true}, {"formatted", formatted},
                {"ids", tokenizer.encode(formatted)}};
    }
    throw std::invalid_argument("unknown operation");
}

}  

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: dengine-tokenizer MODEL_DIRECTORY\n";
        return 2;
    }
    try {
        const auto tokenizer = dengine::Tokenizer::load(argv[1]);
        std::string line;
        while (std::getline(std::cin, line)) {
            json response;
            try {
                response = process(tokenizer, line);
            } catch (const std::exception& error) {
                response = {{"ok", false}, {"error", error.what()}};
            }
            std::cout << response.dump() << '\n' << std::flush;
            if (!std::cout) return 1;
        }
        return std::cin.bad() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "tokenizer startup failed: " << error.what() << '\n';
        return 1;
    }
}
