// src/poly/route_test.cu - route_rows() against the hosts' routing: the same experts for every token, and the same
// weights up to a few units in the last place.  The host side is the architectures' route() written out once more
// (softmax: Qwen3.6, Gemma 4's with its scale and each expert's own; sigmoid with a bias: GLM 5.3 Flash and
// LFM2.5; sqrt of softplus with a bias: DeepSeek V4; sigmoid with the pick on the bare scores: Kolibri 1).  Random
// scores, and some tokens whose top scores tie.
//
// usage: poly_route_test   (exit 0: every pick the same and every weight close)
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

// What the hosts do, gate 0 softmax / 1 sigmoid / 2 sqrt(softplus), as their route() functions have it.  `by`
// gets what the experts were picked by.
void host_route(const float* lg, int n_expert, int n_used, int gate, float in_scale, const float* bias,
                const float* escale, bool norm, float w_scale, int* ids, float* w, std::vector<float>& by) {
    std::vector<float> p(n_expert), pick(n_expert);
    double all = 0.0;
    float mx = -INFINITY;
    for (int e = 0; e < n_expert; ++e) mx = std::max(mx, lg[e] * in_scale);
    for (int e = 0; e < n_expert; ++e) {
        const float x = lg[e] * in_scale;
        // gate 3 (SIGMOID_LOGIT_ADD, Kolibri 1): the weight is the sigmoid of the score, the pick is on the score
        p[e] = gate == 0 ? std::exp(x - mx)
             : gate == 2 ? std::sqrt(x > 20.0f ? x : std::log(1.0f + std::exp(x)))
                         : 1.f / (1.f + std::exp(-x));
        all += p[e];
        pick[e] = (gate == 3 ? x : p[e]) + (bias ? bias[e] : 0.f);
    }
    by = pick;
    const float inv = gate == 0 ? (float) (1.0 / all) : 1.f;
    float sum = 0.f;
    for (int k = 0; k < n_used; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = p[best] * inv;
        sum += w[k];
    }
    for (int k = 0; k < n_used; ++k) {
        if (norm) w[k] /= std::max(sum, 6.103515625e-5f);
        if (escale) w[k] *= escale[ids[k]];
        if (w_scale != 0.f) w[k] *= w_scale;
    }
}

int check(int gate, int n_expert, int n_used, float in_scale, bool with_bias, bool with_escale, bool norm,
          float w_scale, std::mt19937& rng) {
    const int rows = 4096;
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> lg((size_t) rows * n_expert), bias(n_expert), escale(n_expert);
    for (float& x : lg) x = nd(rng);
    for (int t = 0; t < rows; t += 7)   // ties at the top: the lower expert goes first on both sides
        lg[(size_t) t * n_expert + 5] = lg[(size_t) t * n_expert + 9] = 9.f;
    for (float& b : bias) b = with_bias ? 0.05f * nd(rng) : 0.f;
    for (float& s : escale) s = 0.5f + std::abs(nd(rng)) * 0.1f;
    float *d_lg, *d_bias, *d_escale, *d_w;
    int32_t* d_ids;
    cudaMalloc(&d_lg, lg.size() * sizeof(float));
    cudaMalloc(&d_bias, n_expert * sizeof(float));
    cudaMalloc(&d_escale, n_expert * sizeof(float));
    cudaMalloc(&d_w, (size_t) rows * n_used * sizeof(float));
    cudaMalloc(&d_ids, (size_t) rows * n_used * sizeof(int32_t));
    cudaMemcpy(d_lg, lg.data(), lg.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), n_expert * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_escale, escale.data(), n_expert * sizeof(float), cudaMemcpyHostToDevice);
    route_rows(d_lg, n_expert, n_used, rows, gate, in_scale, with_bias ? d_bias : nullptr,
               with_escale ? d_escale : nullptr, norm, w_scale, d_ids, d_w, nullptr);
    std::vector<int32_t> ids((size_t) rows * n_used);
    std::vector<float> w((size_t) rows * n_used);
    cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(w.data(), d_w, w.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_lg), cudaFree(d_bias), cudaFree(d_escale), cudaFree(d_w), cudaFree(d_ids);
    // A pick may go the other way where two experts' scores are a rounding apart: the card's expf and the host's
    // std::exp differ in the last bit now and then (MSVC's more often than glibc's).  That is a tie, and the token's
    // later picks are not compared.
    int bad_ids = 0, ties = 0;
    double worst = 0.0;
    std::vector<int> hi(n_used);
    std::vector<float> hw(n_used), by;
    for (int t = 0; t < rows; ++t) {
        host_route(lg.data() + (size_t) t * n_expert, n_expert, n_used, gate, in_scale,
                   with_bias ? bias.data() : nullptr, with_escale ? escale.data() : nullptr, norm, w_scale, hi.data(),
                   hw.data(), by);
        for (int k = 0; k < n_used; ++k) {
            const size_t i = (size_t) t * n_used + k;
            if (ids[i] != hi[k]) {
                const double a = by[(size_t) hi[k]], b = by[(size_t) ids[i]];
                if (std::abs(a - b) <= 4e-7 * std::max(std::abs(a), std::abs(b))) ++ties;
                else ++bad_ids;
                break;
            }
            worst = std::max(worst, std::abs((double) w[i] - hw[k]) / std::max(1e-12, (double) std::abs(hw[k])));
        }
    }
    const bool ok = bad_ids == 0 && worst < 1e-5;
    std::printf("gate %d, %d experts, %d used: %d picks differ, %d rounding ties, weights within %.2g relative: %s\n",
                gate, n_expert, n_used, bad_ids, ties, worst, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(1234);
    int bad = 0;
    bad += check(0, 256, 8, 1.f, false, false, true, 0.f, rng);                     // Qwen3.6
    bad += check(0, 128, 8, 1.f / std::sqrt(2816.f), false, true, true, 0.f, rng);  // Gemma 4
    bad += check(1, 256, 8, 1.f, true, false, true, 2.5f, rng);                    // GLM 5.3 Flash
    bad += check(2, 256, 6, 1.f, true, false, true, 1.5f, rng);                    // DeepSeek V4
    bad += check(1, 64, 4, 1.f, true, false, true, 0.f, rng);                      // LFM2.5-24B-A2B
    bad += check(3, 384, 6, 1.f, true, false, false, 0.f, rng);                    // Kolibri 1
    if (cudaGetLastError() != cudaSuccess) {
        std::printf("CUDA error\n");
        return 2;
    }
    return bad ? 1 : 0;
}
