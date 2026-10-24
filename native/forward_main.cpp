#include "dengine/inference_model.hpp"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json process(const dengine::InferenceModel& model, const std::string& line) {
    const auto request = json::parse(line);
    if (!request.is_object()) throw std::invalid_argument("request must be a JSON object");
    const auto& ids = request.at("input_ids");
    if (!ids.is_array()) throw std::invalid_argument("input_ids must be an array");
    if (ids.empty() || ids.size() > dengine::kInitialContextLimit) {
        throw std::invalid_argument("input_ids must contain 1 to 1024 tokens");
    }
    std::vector<std::uint32_t> input_ids;
    input_ids.reserve(ids.size());
    for (const auto& id : ids) {
        if (!id.is_number_unsigned() || id.get<std::uint64_t>() >= model.config().vocab_size) {
            throw std::invalid_argument("input_ids contain an invalid token ID");
        }
        input_ids.push_back(id.get<std::uint32_t>());
    }
    const auto option = request.find("full_scores");
    if (option != request.end() && !option->is_boolean()) {
        throw std::invalid_argument("full_scores must be a boolean");
    }
    const auto scores = model.next_token_scores(input_ids);
    const auto& values = scores.values();
    std::vector<std::size_t> rank(values.size());
    std::iota(rank.begin(), rank.end(), 0);
    constexpr std::size_t kTopCount = 10;
    std::partial_sort(rank.begin(), rank.begin() + kTopCount, rank.end(),
                      [&](std::size_t a, std::size_t b) {
                          return values[a] == values[b] ? a < b : values[a] > values[b];
                      });
    rank.resize(kTopCount);
    std::vector<float> top_scores;
    top_scores.reserve(kTopCount);
    for (const auto id : rank) top_scores.push_back(values[id]);
    json response = {{"ok", true}, {"top10_ids", rank}, {"top10_scores", top_scores}};
    if (option != request.end() && option->get<bool>()) response["scores"] = values;
    return response;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: dengine-forward MODEL_DIRECTORY\n";
        return 2;
    }
    try {
        const dengine::InferenceModel model(argv[1]);
        std::string line;
        while (std::getline(std::cin, line)) {
            json response;
            try {
                response = process(model, line);
            } catch (const std::exception& error) {
                response = {{"ok", false}, {"error", error.what()}};
            }
            std::cout << response.dump() << '\n' << std::flush;
            if (!std::cout) return 1;
        }
        return std::cin.bad() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "model startup failed: " << error.what() << '\n';
        return 1;
    }
}
