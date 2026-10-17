// src/poly/mxfp4_test.cu - the card's MXFP4 product (kernels::native_mmvq(), type 39) against ggml's own
// dequantizer: random blocks with exponents around a half's range, times float activations the card quantizes to
// Q8_1, for 1 to 8 columns.  To the activations as the card quantized them the product must be exact but for the
// order of its float sums; the slots launch (native_mmvq_slots()) must give the bits of the single launches.
//
// usage: poly_mxfp4_test   (exit 0: all agree)
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata;

constexpr int kMxfp4 = 39, kBlock = 17, kQ8Block = 36;   // a Q8_1 block: a half2 (scale, sum) and 32 int8

double most = 0;   // the largest error to the quantized activations, as a share of the largest output

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

// Returns the failures of one shape.
int check(int n_in, int n_out, std::mt19937& rng) {
    const int blocks = n_in / 32;
    std::vector<uint8_t> w((size_t) n_out * blocks * kBlock);
    for (size_t b = 0; b < w.size() / kBlock; ++b) {
        w[b * kBlock] = (uint8_t) (117 + rng() % 20);
        for (int j = 1; j < kBlock; ++j) w[b * kBlock + j] = (uint8_t) rng();
    }
    std::vector<float> wf((size_t) n_out * n_in);
    const ggml_type_traits* t = ggml_get_type_traits(GGML_TYPE_MXFP4);
    for (int r = 0; r < n_out; ++r)
        t->to_float(w.data() + (size_t) r * blocks * kBlock, wf.data() + (size_t) r * n_in, n_in);

    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) 8 * n_in);
    for (float& v : x) v = nd(rng);

    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    uint8_t* dw = upload(w);
    float* dx = upload(x);
    void* xq = nullptr;
    cudaMalloc(&xq, kernels::native_q8_1_bytes(n_in, 8));
    float* dy = nullptr;
    cudaMalloc(&dy, (size_t) 8 * n_out * sizeof(float));

    int fails = 0;
    std::vector<float> one;   // the single column's output, for the slots launch
    for (int ncols = 1; ncols <= 8; ++ncols) {
        kernels::native_quantize_q8_1(dx, xq, n_in, ncols, s);
        kernels::native_mmvq(kMxfp4, dw, xq, dy, n_in, n_out, ncols, s);
        cudaStreamSynchronize(s);
        if (cudaGetLastError() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: %d x %d, %d columns: the launch failed\n", n_out, n_in, ncols);
            return fails + 1;
        }
        const std::vector<float> y = download(dy, (size_t) ncols * n_out);
        if (ncols == 1) one = y;
        const std::vector<uint8_t> q = download((const uint8_t*) xq, kernels::native_q8_1_bytes(n_in, ncols));
        double worst = 0, worst_q = 0, scale = 0;
        for (int c = 0; c < ncols; ++c)
            for (int r = 0; r < n_out; ++r) {
                const float* wr = wf.data() + (size_t) r * n_in;
                double ref = 0, ref_q = 0;
                for (int i = 0; i < n_in; ++i) ref += (double) wr[i] * x[(size_t) c * n_in + i];
                for (int b = 0; b < blocks; ++b) {
                    const uint8_t* blk = q.data() + ((size_t) c * blocks + b) * kQ8Block;
                    __half d;
                    std::memcpy(&d, blk, sizeof d);
                    double sum = 0;
                    for (int i = 0; i < 32; ++i) sum += (double) wr[b * 32 + i] * (int8_t) blk[4 + i];
                    ref_q += sum * (double) __half2float(d);
                }
                const double got = y[(size_t) c * n_out + r];
                worst = std::max(worst, std::fabs(ref - got));
                worst_q = std::max(worst_q, std::fabs(ref_q - got));
                scale = std::max(scale, std::fabs(ref));
            }
        // to the floats the product is off by the activations' rounding, about 1/254 of a block's largest
        most = std::max(most, worst_q / scale);
        if (!(worst_q <= 1e-5 * scale) || !(worst <= 2e-2 * scale)) {
            std::fprintf(stderr, "FAIL: %d x %d, %d columns: off by %.3g (%.3g to the floats) of %.3g\n", n_out, n_in,
                         ncols, worst_q, worst, scale);
            ++fails;
        }
    }

    // three experts in one launch, each the same matrix and column as the single launch: the same bits
    kernels::native_quantize_q8_1(dx, xq, n_in, 1, s);
    kernels::MmvqSlots slots{};
    float* dys = nullptr;
    cudaMalloc(&dys, (size_t) 3 * n_out * sizeof(float));
    for (int j = 0; j < 3; ++j) {
        slots.w[j] = dw;
        slots.x[j] = xq;
        slots.y[j] = dys + (size_t) j * n_out;
    }
    if (!kernels::native_mmvq_slots(kMxfp4, slots, 3, n_in, n_out, s)) {
        std::fprintf(stderr, "FAIL: %d x %d: no slots kernel for MXFP4\n", n_out, n_in);
        ++fails;
    } else {
        cudaStreamSynchronize(s);
        const std::vector<float> ys = download(dys, (size_t) 3 * n_out);
        for (int j = 0; j < 3; ++j)
            if (std::memcmp(ys.data() + (size_t) j * n_out, one.data(), (size_t) n_out * sizeof(float)) != 0) {
                std::fprintf(stderr, "FAIL: %d x %d: slot %d differs from the single launch\n", n_out, n_in, j);
                ++fails;
            }
    }
    cudaFree(dys);
    cudaFree(dy);
    cudaFree(xq);
    cudaFree(dx);
    cudaFree(dw);
    cudaStreamDestroy(s);
    return fails;
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    int fails = 0;
    // DeepSeek V4 Flash's down experts (2048 in, 4096 out) and smaller shapes, the small-K kernel's among them
    for (const auto& [n_in, n_out] : {std::pair{2048, 4096}, std::pair{512, 256}, std::pair{64, 96}})
        fails += check(n_in, n_out, rng);
    std::printf("poly_mxfp4_test: %d failures; the largest error %.2g of the largest output\n", fails, most);
    return fails ? 1 : 0;
}
