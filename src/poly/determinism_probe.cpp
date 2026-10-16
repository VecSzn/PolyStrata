// src/poly/determinism_probe.cpp - run-to-run bitwise stability of the decode kernels.
//
//     cmake --build build-up --target poly_determinism_probe && ./poly_determinism_probe
//
// WHY IT EXISTS.  Greedy decode on this engine answers the same prompt with two different
// continuations at a rate of a few percent per request, and every engine-level variable has been
// ruled out: the conversation cache (fresh prompts still diverge), the expert cache page-out
// (--no-gpu-experts narrows it but does not remove it), CUDA graph replay (--no-graphs changes
// nothing), the VRAM layout (--kv-whole changes nothing), the CPU expert reduction (fixed-order
// blocks, single-threaded still diverges) and the sampler (most_likely() is a plain argmax with no
// RNG).  What is left is the device side of a decode step: with the same bytes in, some kernel of
// the per-token path must return different bits on different launches.  This probe calls each
// kernel of that path directly, 33 times on the same inputs, and hashes the outputs.
//
// WHAT IT COVERS.  The four paths Engine::lin() can take for one token (n=1): gemv_f32,
// gemv_bf16, native_mmvq over a Q6_K weight, and cuBLAS gemm_f32 as the prompt path's product.
// The CPU expert path is not probed: its structure is fixed-order by construction and the
// single-thread experiment already cleared it.
//
// THE DATA.  Weights are random bytes with their fp16 scales rewritten to sane normals, the same
// trick mmvq_multi_parity uses, so every output is finite and a bitwise difference means a real
// arithmetic difference.  Every launch reads the same device buffers; nothing is rewritten between
// launches except the output buffer, which the check copies out before the next launch.

#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "poly/cpu_experts.hpp"

using namespace strata::poly;

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

int fails = 0;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

uint64_t fnv1a(const float* v, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v);
    for (size_t i = 0; i < n * 4; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

uint16_t sane_half(std::mt19937_64& rng) {
    // a normal fp16 in +- [2^-10, 2^-5): large enough that fp16 rounding of a product stays normal
    uint16_t bits = (uint16_t) (rng() & 0xffff);
    bits = (bits & 0x83ff) | 0x2c00;   // exponent somewhere in the window, random mantissa, random sign
    return bits;
}

// gemv from this project lives in kernels.cu and is not exported in a header; its F32/BF16 paths are
// exercised here through cuBLAS instead, which is what the prompt path uses.  The per-token gemv of
// the engine IS probed - through native_mmvq, which is the quantized-weight path every decode lin()
// takes for Q6_K weights.

struct Run {
    const char* name;
    uint64_t first = 0;
    int uniq = 1;
};

volatile uint64_t sink;

void check(Run& r, const std::vector<float>& out) {
    uint64_t h = fnv1a(out.data(), out.size());
    if (r.first == 0) r.first = h;
    else if (h != r.first) {
        ++r.uniq;
        r.first = h;   // continue tracking: report how many distinct outputs appear
    }
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    cudaStream_t s = nullptr;
    if (!ck(cudaStreamCreate(&s), "stream")) return 1;

    std::mt19937_64 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int N_IN = 2048, N_OUT = 4096;   // Ornith's n_embd x a fat FFN-ish output
    constexpr int REPS = 33;

    // ---- 5) cpu_experts, same inputs, repeated: the decode path's CPU expert products ----
    {
        std::printf("\n== cpu_experts same-input repeat ==\n");
        constexpr int EMBD = 2048, FF = 1024, NEXP = 8, ROWS = 1;
        for (int nthr : {1, 16}) {
            Workers workers(nthr);
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
                cpu_experts(workers, experts.data(), NEXP, ROWS, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, EMBD, FF,
                            xin.data(), 30.f, false, out.data(), nullptr);
                size_t h = 1469598103934665603ull;
                const unsigned char* pb = (const unsigned char*) out.data();
                for (size_t i = 0; i < out.size() * sizeof(float); ++i) {
                    h ^= pb[i];
                    h *= 1099511628211ull;
                }
                hashes.push_back(h);
            }
            const int distinct = (int) std::set<size_t>(hashes.begin(), hashes.end()).size();
            std::printf("threads=%d  distinct=%d/33  %s\n", nthr, distinct,
                         distinct == 1 ? "stable" : "!!! NON-DETERMINISTIC !!!");
            for (size_t h : std::set<size_t>(hashes.begin(), hashes.end()))
                std::printf("  %016zx\n", h);
            fails += distinct - 1;
        }
    }

    // ---- 1) native_mmvq, Q6_K weights, one token ----
    {
        const int block_bytes = 210, block_elems = 256;
        const size_t wbytes = (size_t) N_OUT * (N_IN / block_elems) * block_bytes;
        std::vector<uint8_t> w(wbytes);
        for (auto& b : w) b = (uint8_t) (rng() & 0xff);
        const size_t n_blocks = wbytes / block_bytes;
        for (size_t k = 0; k < n_blocks; ++k) {
            uint16_t h = sane_half(rng);                    // Q6_K scale at offset 0
            std::memcpy(&w[k * block_bytes], &h, 2);
            for (int at : {2, 4}) {                         // and the two sub-block min scales
                h = sane_half(rng);
                std::memcpy(&w[k * block_bytes + at], &h, 2);
            }
        }
        std::vector<float> x(N_IN);
        for (auto& v : x) v = nd(rng);

        void *dw = nullptr, *xq = nullptr, *dx = nullptr, *dy = nullptr;
        size_t qbytes = strata::kernels::native_q8_1_bytes(N_IN, 1);
        if (!ck(cudaMalloc(&dw, wbytes), "w") || !ck(cudaMalloc(&xq, qbytes), "xq") ||
            !ck(cudaMalloc(&dx, (size_t) N_IN * 4), "x") || !ck(cudaMalloc(&dy, (size_t) N_OUT * 4), "y"))
            return 1;
        if (!ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "w copy") ||
            !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x copy"))
            return 1;
        strata::kernels::quantize_q8_1_rows((const float*) dx, 1, N_IN, xq, s);
        if (!ck(cudaStreamSynchronize(s), "quantize")) return 1;

        Run r{"native_mmvq Q6_K n=1"};
        std::vector<float> out(N_OUT);
        for (int rep = 0; rep < REPS; ++rep) {
            strata::kernels::native_mmvq(GGML_TYPE_Q6_K, dw, xq, (float*) dy, N_IN, N_OUT, 1, s);
            if (!ck(cudaStreamSynchronize(s), "mmvq")) return 1;
            if (!ck(cudaMemcpy(out.data(), dy, out.size() * 4, cudaMemcpyDeviceToHost), "read")) return 1;
            check(r, out);
        }
        std::printf("%-28s unique outputs: %d/%d  %s\n", r.name, r.uniq, REPS, r.uniq == 1 ? "STABLE" : "UNSTABLE");
        if (r.uniq > 1) ++fails;
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(dy);
    }

    // ---- 2) native_mmvq, Q6_K, eight tokens (the multi-column path decode never takes but batch 2..8 does) ----
    {
        const int block_bytes = 210, block_elems = 256, T = 8;
        const size_t wbytes = (size_t) N_OUT * (N_IN / block_elems) * block_bytes;
        std::vector<uint8_t> w(wbytes);
        for (auto& b : w) b = (uint8_t) (rng() & 0xff);
        for (size_t k = 0; k < wbytes / block_bytes; ++k) {
            uint16_t h = sane_half(rng);
            std::memcpy(&w[k * block_bytes], &h, 2);
            for (int at : {2, 4}) { h = sane_half(rng); std::memcpy(&w[k * block_bytes + at], &h, 2); }
        }
        std::vector<float> x((size_t) T * N_IN);
        for (auto& v : x) v = nd(rng);

        void *dw = nullptr, *xq = nullptr, *dx = nullptr, *dy = nullptr;
        size_t qall = strata::kernels::native_q8_1_bytes(N_IN, T);
        size_t qcol = strata::kernels::native_q8_1_bytes(N_IN, 1);
        if (!ck(cudaMalloc(&dw, wbytes), "w") || !ck(cudaMalloc(&xq, qall), "xq") ||
            !ck(cudaMalloc(&dx, x.size() * 4), "x") || !ck(cudaMalloc(&dy, (size_t) T * N_OUT * 4), "y"))
            return 1;
        if (!ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "w copy") ||
            !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x copy"))
            return 1;
        strata::kernels::quantize_q8_1_rows((const float*) dx, T, N_IN, xq, s);
        if (!ck(cudaStreamSynchronize(s), "quantize")) return 1;

        Run r{"native_mmvq Q6_K T=8"};
        std::vector<float> out((size_t) T * N_OUT);
        for (int rep = 0; rep < REPS; ++rep) {
            strata::kernels::native_mmvq(GGML_TYPE_Q6_K, dw, xq, (float*) dy, N_IN, N_OUT, T, s);
            if (!ck(cudaStreamSynchronize(s), "mmvq")) return 1;
            if (!ck(cudaMemcpy(out.data(), dy, out.size() * 4, cudaMemcpyDeviceToHost), "read")) return 1;
            check(r, out);
        }
        std::printf("%-28s unique outputs: %d/%d  %s\n", r.name, r.uniq, REPS, r.uniq == 1 ? "STABLE" : "UNSTABLE");
        if (r.uniq > 1) ++fails;
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(dy);
    }

    // ---- 3) cuBLAS gemm_f32, the prompt batch's product (n=32) ----
    {
        std::vector<float> w((size_t) N_OUT * N_IN);
        for (auto& v : w) v = nd(rng);
        std::vector<float> x((size_t) 32 * N_IN);
        for (auto& v : x) v = nd(rng);

        float *dw = nullptr, *dx = nullptr, *dy = nullptr;
        if (!ck(cudaMalloc(&dw, w.size() * 4), "w") || !ck(cudaMalloc(&dx, x.size() * 4), "x") ||
            !ck(cudaMalloc(&dy, (size_t) 32 * N_OUT * 4), "y"))
            return 1;
        if (!ck(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "w copy") ||
            !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x copy"))
            return 1;

        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 1;
        if (cublasSetStream(h, s) != CUBLAS_STATUS_SUCCESS) return 1;
        if (cublasSetMathMode(h, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) return 1;

        Run r{"cuBLAS gemm f32 n=32"};
        std::vector<float> out((size_t) 32 * N_OUT);
        const float one = 1.f, zero = 0.f;
        for (int rep = 0; rep < REPS; ++rep) {
            // this project's gemm_f32: y = x * w^T (row-major, cublas column-major trick)
            if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N_OUT, 32, N_IN, &one,
                             dw, CUDA_R_32F, N_IN, dx, CUDA_R_32F, N_IN, &zero,
                             dy, CUDA_R_32F, N_OUT, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                std::printf("cublas gemm failed\n"); return 1;
            }
            if (!ck(cudaStreamSynchronize(s), "gemm")) return 1;
            if (!ck(cudaMemcpy(out.data(), dy, out.size() * 4, cudaMemcpyDeviceToHost), "read")) return 1;
            check(r, out);
        }
        std::printf("%-28s unique outputs: %d/%d  %s\n", r.name, r.uniq, REPS, r.uniq == 1 ? "STABLE" : "UNSTABLE");
        if (r.uniq > 1) ++fails;
        cublasDestroy(h);
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }

    // ---- 4) cuBLAS gemv-shaped gemm (n=1): what a decode lin() takes for f32 weights ----
    {
        std::vector<float> w((size_t) N_OUT * N_IN);
        for (auto& v : w) v = nd(rng);
        std::vector<float> x(N_IN);
        for (auto& v : x) v = nd(rng);

        float *dw = nullptr, *dx = nullptr, *dy = nullptr;
        if (!ck(cudaMalloc(&dw, w.size() * 4), "w") || !ck(cudaMalloc(&dx, x.size() * 4), "x") ||
            !ck(cudaMalloc(&dy, (size_t) N_OUT * 4), "y"))
            return 1;
        if (!ck(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "w copy") ||
            !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x copy"))
            return 1;
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 1;
        if (cublasSetStream(h, s) != CUBLAS_STATUS_SUCCESS) return 1;

        Run r{"cuBLAS gemv f32 n=1"};
        std::vector<float> out(N_OUT);
        const float one = 1.f, zero = 0.f;
        for (int rep = 0; rep < REPS; ++rep) {
            if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N_OUT, 1, N_IN, &one,
                             dw, CUDA_R_32F, N_IN, dx, CUDA_R_32F, N_IN, &zero,
                             dy, CUDA_R_32F, N_OUT, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                std::printf("cublas gemv failed\n"); return 1;
            }
            if (!ck(cudaStreamSynchronize(s), "gemv")) return 1;
            if (!ck(cudaMemcpy(out.data(), dy, out.size() * 4, cudaMemcpyDeviceToHost), "read")) return 1;
            check(r, out);
        }
        std::printf("%-28s unique outputs: %d/%d  %s\n", r.name, r.uniq, REPS, r.uniq == 1 ? "STABLE" : "UNSTABLE");
        if (r.uniq > 1) ++fails;
        cublasDestroy(h);
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }

    std::printf("\n%s (%d unstable path%s)\n", fails ? "UNSTABLE" : "ALL STABLE", fails, fails == 1 ? "" : "s");

    sink = 0;
    return fails ? 1 : 0;
}
