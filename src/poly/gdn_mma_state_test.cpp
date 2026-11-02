// Optional real-model test: MODEL PROMPT_JSON PROFILE (STRATA_GDN_MMA=1 enables). Requires an MTP model.
// The prompt has at least 8192 numeric IDs. Use a fresh profile copied from the same seed for each arm.
#include "engine.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
using namespace strata::poly;
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
std::vector<int32_t> prompt_ids(const char* path) {
    std::ifstream in(path);
    require(bool(in), "cannot open prompt");
    std::string text((std::istreambuf_iterator<char>(in)), {});
    for (char& c : text) if (c < '0' || c > '9') c = ' ';
    std::istringstream words(text);
    std::vector<int32_t> ids;
    for (int32_t id; words >> id;) ids.push_back(id);
    require(ids.size() >= 8192, "prompt requires at least 8192 numeric IDs");
    return ids;
}
std::vector<uint8_t> saved(Engine& engine) {
    std::vector<uint8_t> bytes(engine.state_bytes());
    engine.save_state(bytes.data());
    return bytes;
}
std::vector<float> scores(Engine& engine) { return {engine.logits(), engine.logits() + engine.n_vocab()}; }
void same_logits(const std::vector<float>& expected, const float* actual) {
    require(!std::memcmp(expected.data(), actual, expected.size() * sizeof(float)), "logits differ");
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 4,
                "usage: poly_gdn_mma_state_test MODEL PROMPT_JSON PROFILE (STRATA_GDN_MMA=1 enables)");
        Options options;
        options.model = argv[1]; options.profile = argv[3];
        options.max_context = 110000; options.threads = 6; options.prompt_batch = 2048;
        options.vram_reserve_mib = 768; options.kv_format = KvFmt::kQ8_0;
        options.cpu_share = 0; options.expert_swaps = 0; options.window = 8;
        auto tokens = prompt_ids(argv[2]);
        auto engine = Engine::open(options);
        require(engine->drafts(), "this test requires a loaded MTP block");
        // A first single-token call may capture decode graphs before any eligible prefill.
        engine->eval(tokens.data(), 1, true);
        require(engine->read(tokens.data() + 1, 4095), "prefix read failed");
        engine->eval(tokens.data() + 4096, 1, true);
        const auto prefix = saved(*engine);
        const auto position = engine->pos();
        const int32_t known = tokens[5000];
        auto verify = [&](const char* phase) {
            engine->load_state(prefix.data(), position);
            engine->eval(&known, 1, true);
            const auto expected = saved(*engine);
            const auto output = scores(*engine);
            const int32_t wrong = (most_likely(output.data(), engine->n_vocab()) + 1) % engine->n_vocab();
            for (int width : {2, 3, 4, 8}) {
                engine->load_state(prefix.data(), position);
                int32_t guess[8] = {known, wrong};
                for (int j = 2; j < width; ++j) guess[j] = tokens[5100 + j];
                engine->window(guess, width);
                same_logits(output, engine->logits_of(0));
                require(most_likely(engine->logits_of(0), engine->n_vocab()) != wrong, "draft must be wrong");
                engine->keep(1);
                require(engine->pos() == position + 1 && saved(*engine) == expected, "rollback state differs");
                std::printf("MATCH %s wrong-draft width=%d\n", phase, width);
            }
        };
        verify("initial");
        engine->load_state(prefix.data(), position);
        require(engine->read(tokens.data() + 5000, 65), "continuation read failed");
        engine->eval(tokens.data() + 5065, 1, true);
        const auto continued = saved(*engine);
        const auto continued_logits = scores(*engine);
        engine->load_state(prefix.data(), position);
        require(engine->read(tokens.data() + 5000, 65), "restored continuation failed");
        engine->eval(tokens.data() + 5065, 1, true);
        require(saved(*engine) == continued, "checkpoint continuation state differs");
        same_logits(continued_logits, engine->logits());
        engine->resize_cache(896);
        verify("resized");
        engine->load_state(prefix.data(), position);
        int calls = 0;
        const std::function<bool(int,int)> cancel = [&](int done, int) { ++calls; return done < 5; };
        require(!engine->read(tokens.data(), 4096, nullptr, 0, &cancel) && calls >= 5, "cancel read failed");
        verify("cancel recovery");
        std::puts("PASS GDN model state/rollback/continuation/residency/cancel");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
