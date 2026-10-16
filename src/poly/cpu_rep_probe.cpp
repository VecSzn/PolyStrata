// src/poly/cpu_rep_probe.cpp - cpu_experts() bitwise reproducibility, standalone.
#include "poly/cpu_experts.hpp"

using namespace strata::poly;

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("ggml_time_ms=%lld\n", (long long) ggml_time_ms());
    std::mt19937_64 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int EMBD = 2048, FF = 1024, NEXP = 8, ROWS = 1;
    for (int nthr : {1, 16}) {
        if (nthr == 16) break;   // bisect: only threads=1 arm, and skip Workers entirely
        // Workers workers(nthr);   // bisect: Workers skipped
        std::vector<float> xin((size_t) EMBD);
        for (float& v : xin) v = nd(rng);
        std::vector<float> gsrc((size_t) NEXP * EMBD), usrc((size_t) NEXP * EMBD), dsrc((size_t) NEXP * FF);
        for (float& v : gsrc) v = nd(rng);
        for (float& v : usrc) v = nd(rng);
        for (float& v : dsrc) v = nd(rng);
        const size_t g_bytes = ggml_quantize_chunk(GGML_TYPE_Q6_K, gsrc.data(), nullptr, 0, NEXP, EMBD, nullptr);
        const size_t d_bytes = ggml_quantize_chunk(GGML_TYPE_Q6_K, dsrc.data(), nullptr, 0, NEXP, FF, nullptr);
        std::vector<uint8_t> gate(g_bytes), up(g_bytes), down(d_bytes);
        ggml_quantize_chunk(GGML_TYPE_Q6_K, gsrc.data(), gate.data(), 0, NEXP, EMBD, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q6_K, usrc.data(), up.data(), 0, NEXP, EMBD, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q6_K, dsrc.data(), down.data(), 0, NEXP, FF, nullptr);
        std::vector<CpuExpert> experts(NEXP);
        for (int e = 0; e < NEXP; ++e) {
            experts[e].gate = gate.data() + (size_t) e * (g_bytes / NEXP);
            experts[e].up = up.data() + (size_t) e * (g_bytes / NEXP);
            experts[e].down = down.data() + (size_t) e * (d_bytes / NEXP);
            experts[e].weight = 0.1f + 0.02f * e;
            experts[e].row = 0;
        }
        std::vector<float> out((size_t) ROWS * EMBD);
        std::vector<size_t> hashes;
        for (int r = 0; r < 33; ++r) {
            std::fill(out.begin(), out.end(), 0.f);
            if (r == 0) {
                std::printf("skip cpu_experts call for bisect
");
                continue;
            }
            // cpu_experts(workers, experts.data(), NEXP, ROWS, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, EMBD, FF, xin.data(),
                        30.f, false, out.data(), nullptr);
            size_t h = 1469598103934665603ull;
            const unsigned char* pb = (const unsigned char*) out.data();
            for (size_t i = 0; i < out.size() * sizeof(float); ++i) {
                h ^= pb[i];
                h *= 1099511628211ull;
            }
            hashes.push_back(h);
        }
        const int distinct = (int) std::set<size_t>(hashes.begin(), hashes.end()).size();
        std::printf("threads=%d  distinct=%d/33  %s' + NL + '", nthr, distinct,
                    distinct == 1 ? "stable" : "!!! NON-DETERMINISTIC !!!");
        for (size_t h : std::set<size_t>(hashes.begin(), hashes.end()))
            std::printf("  %016zx' + NL + '", h);
    }
    return 0;
}
