#include "dengine/generator.hpp"
#include "dengine/kv_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
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

void compare_scores(const dengine::F32Tensor& cached, const dengine::F32Tensor& uncached,
                    const char* message) {
    check(cached.values().size() == uncached.values().size(), "score count differs");
    double maximum_difference = 0.0;
    for (std::size_t i = 0; i < cached.values().size(); ++i) {
        check(std::isfinite(cached.values()[i]), "cached score is not finite");
        maximum_difference = std::max(maximum_difference,
            std::abs(double(cached.values()[i]) - uncached.values()[i]));
    }
    if (maximum_difference > 1e-3) {
        throw std::runtime_error(std::string(message) + ": max score difference " +
                                 std::to_string(maximum_difference));
    }
    const auto top = [](const dengine::F32Tensor& scores) {
        return std::max_element(scores.values().begin(), scores.values().end()) -
               scores.values().begin();
    };
    check(top(cached) == top(uncached), "cached and uncached top token differ");
}

void test_session(const dengine::InferenceModel& model) {
    rejects<std::invalid_argument>([&] { dengine::CachedSession invalid(model, 0); },
                                   "zero cache capacity accepted");
    rejects<std::invalid_argument>([&] { dengine::CachedSession invalid(model, 1025); },
                                   "oversized cache capacity accepted");
    dengine::CachedSession session(model, 16);
    rejects<std::invalid_argument>([&] { session.next_token_scores(); },
                                   "empty cache produced scores");

    const std::vector<std::uint32_t> ids{19556, 28, 905, 17};
    session.prefill(ids);
    check(session.capacity() == 16 && session.token_count() == ids.size(),
          "prompt entries not saved");
    check(session.block_token_evaluations() == ids.size() * model.config().layers,
          "prompt should execute each token in each layer once");
    compare_scores(session.next_token_scores(), model.next_token_scores(ids),
                   "prompt scores");
    const auto work_after_prefill = session.block_token_evaluations();
    session.next_token_scores();
    check(session.block_token_evaluations() == work_after_prefill,
          "rescoring recomputed old token blocks");

    session.append_token(198);
    auto extended = ids;
    extended.push_back(198);
    check(session.token_count() == extended.size() &&
          session.block_token_evaluations() == work_after_prefill + model.config().layers,
          "one new token should execute one block per layer");
    compare_scores(session.next_token_scores(), model.next_token_scores(extended),
                   "one-token extension scores");
    rejects<std::invalid_argument>([&] { session.prefill(ids); },
                                   "prefill on nonempty cache accepted");
    rejects<std::invalid_argument>([&] { session.append_token(49152); },
                                   "invalid token ID accepted");
    check(session.token_count() == extended.size(), "invalid ID changed cached history");

    dengine::CachedSession independent(model, 16);
    independent.prefill({216, 378, 2365});
    compare_scores(session.next_token_scores(), model.next_token_scores(extended),
                   "another request changed first cache");

    session.reset();
    check(session.token_count() == 0 && session.block_token_evaluations() == 0,
          "reset did not clear cache state and counters");
    rejects<std::invalid_argument>([&] { session.next_token_scores(); },
                                   "reset cache produced stale scores");
    rejects<std::invalid_argument>([&] { session.prefill({19556, 49152}); },
                                   "bad prefill token accepted");
    check(session.token_count() == 0, "failed prefill retained partial state");
    session.prefill(ids);
    compare_scores(session.next_token_scores(), model.next_token_scores(ids),
                   "reset and reuse scores");

    dengine::CachedSession small(model, 1);
    small.prefill({19556});
    rejects<std::out_of_range>([&] { small.append_token(198); },
                               "full cache accepted another token");
    check(small.token_count() == 1, "capacity error damaged existing cache");
}

void test_generation(const dengine::InferenceModel& model,
                     const dengine::Tokenizer& tokenizer) {
    const dengine::Generator generator(model, tokenizer);
    for (const auto& prompt : {std::string("Hello, world!"),
                               std::string("  The quick brown fox\n")}) {
        const auto cached = generator.generate(prompt, 8);
        const auto uncached = generator.generate_uncached(prompt, 8);
        check(cached.generated_ids == uncached.generated_ids &&
              cached.text == uncached.text &&
              cached.finish_reason == uncached.finish_reason,
              "cached and uncached generation differ");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected MODEL_DIRECTORY");
        const dengine::InferenceModel model(argv[1]);
        const auto tokenizer = dengine::Tokenizer::load(argv[1]);
        test_session(model);
        test_generation(model, tokenizer);
        std::cout << "cached and uncached inference match; old tokens are not recomputed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
