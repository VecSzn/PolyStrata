// src/poly/kv_int8_test.cpp - store_int8()/attend_int8() against ggml's own quantizer.
//
// --kv int8 keeps an attention layer's K and V as 8-bit codes with a delta a block of kQ8_0Values values, which is
// llama.cpp's Q8_0 rule (delta = the block's largest magnitude / 127, code = roundf(x / delta)) in two arrays
// instead of ggml's 34-byte block_q8_0 - see kernels.hpp.  So the writer must still agree with
// quantize_row_q8_0_ref to the bit, which is what this checks block by block against ggml_quantize_chunk: the same
// entry point engine.cpp uses for its expert weights, so the reference is the library's own code.
//
//   cmake --build build-pr4 --target poly_kv_int8_test && ./poly_kv_int8_test
#include "strata/poly/kernels.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::poly;

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

void* dev_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(%zu) failed\n", bytes);
        std::exit(2);
    }
    return p;
}

uint16_t half_bits(float f) {
    __half h = __float2half(f);
    uint16_t b;
    std::memcpy(&b, &h, sizeof b);
    return b;
}

float half_value(uint16_t b) {   // through memcpy: HIP has no host __ushort_as_half
    __half h;
    std::memcpy(&h, &b, sizeof h);
    return __half2float(h);
}

/// ggml's own 34-byte blocks for these rows, taken apart into the two arrays --kv int8 keeps.
struct Reference {
    std::vector<int8_t> code;
    std::vector<uint16_t> delta;
};

Reference ggml_reference(const std::vector<float>& src, int64_t rows, int64_t per_row) {
    const int64_t blocks = rows * per_row / kQ8_0Values;
    std::vector<uint8_t> raw((size_t) blocks * 34);   // ggml's block_q8_0: a half delta, then 32 codes
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), raw.data(), 0, rows, per_row, nullptr);
    Reference r;
    r.code.resize((size_t) blocks * kQ8_0Values);
    r.delta.resize((size_t) blocks);
    for (int64_t b = 0; b < blocks; ++b) {
        std::memcpy(&r.delta[(size_t) b], raw.data() + b * 34, sizeof(uint16_t));
        std::memcpy(r.code.data() + b * kQ8_0Values, raw.data() + b * 34 + 2, kQ8_0Values);
    }
    return r;
}

/// The spread of values a quantizer gets wrong: an all-zero row (delta 0), a row of ±1e4, small denormals beside a
/// large value, and the plain normal case.
std::vector<float> make_rows(int n_kv, int head_dim, int rows, int kind) {
    const size_t n = (size_t) n_kv * head_dim * rows;
    std::vector<float> v(n, 0.0f);
    std::mt19937 rng(1234u + (unsigned) kind);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (int r = 0; r < rows; ++r) {
        float* row = v.data() + (size_t) r * n_kv * head_dim;
        switch (kind % 4) {
            case 0:   // zeros
                break;
            case 1:   // extremes, every block near the top of the range
                for (int i = 0; i < n_kv * head_dim; ++i) row[i] = (i % 2 ? -1.0f : 1.0f) * 1e4f;
                break;
            case 2:   // a wide dynamic range inside one block
                for (int i = 0; i < n_kv * head_dim; ++i)
                    row[i] = (i % kQ8_0Values) ? 1e-6f * g(rng) : 1e3f * g(rng);
                break;
            default:  // normal
                for (int i = 0; i < n_kv * head_dim; ++i) row[i] = g(rng);
                break;
        }
    }
    return v;
}

/// store_int8() at row `pos`, against the reference, code by code and delta by delta.
void case_plain(const char* name, int n_kv, int head_dim, int rows, int kind) {
    const int per_row = n_kv * head_dim / kQ8_0Values;
    const std::vector<float> src = make_rows(n_kv, head_dim, rows, kind);
    const Reference ref = ggml_reference(src, rows, (int64_t) n_kv * head_dim);

    const size_t n_codes = (size_t) rows * n_kv * head_dim, n_deltas = (size_t) rows * per_row;
    float* dx = (float*) dev_alloc(src.size() * sizeof(float));
    int8_t* dcode = (int8_t*) dev_alloc(n_codes);
    uint16_t* ddelta = (uint16_t*) dev_alloc(n_deltas * sizeof(uint16_t));
    cudaMemcpy(dx, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dcode, 0x7F, n_codes);
    cudaMemset(ddelta, 0x11, n_deltas * sizeof(uint16_t));

    store_int8(dx, dcode, ddelta, n_kv, head_dim, 0, rows, nullptr);
    cudaDeviceSynchronize();

    std::vector<int8_t> code(n_codes, 0);
    std::vector<uint16_t> delta(n_deltas, 0);
    cudaMemcpy(code.data(), dcode, n_codes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta.data(), ddelta, n_deltas * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    size_t bad = 0;
    for (size_t i = 0; i < n_codes; ++i)
        if (code[i] != ref.code[i]) ++bad;
    for (size_t i = 0; i < n_deltas; ++i)
        if (delta[i] != ref.delta[i]) ++bad;
    if (bad) std::fprintf(stderr, "  %s: %zu differ\n", name, bad);
    check(bad == 0, name);

    cudaFree(dx);
    cudaFree(dcode);
    cudaFree(ddelta);
}

/// store_int8() from a position that is not zero: the same blocks, written further along.
void case_pos(const char* name, int n_kv, int head_dim, int rows, int64_t pos, int kind) {
    const std::vector<float> src = make_rows(n_kv, head_dim, rows, kind);
    const Reference ref = ggml_reference(src, rows, (int64_t) n_kv * head_dim);
    const size_t rc = (size_t) n_kv * head_dim, rd = (size_t) n_kv * head_dim / kQ8_0Values;

    float* dx = (float*) dev_alloc(src.size() * sizeof(float));
    int8_t* dcode = (int8_t*) dev_alloc((size_t) (pos + rows) * rc);
    uint16_t* ddelta = (uint16_t*) dev_alloc((size_t) (pos + rows) * rd * sizeof(uint16_t));
    cudaMemcpy(dx, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice);
    store_int8(dx, dcode, ddelta, n_kv, head_dim, pos, rows, nullptr);
    cudaDeviceSynchronize();

    std::vector<int8_t> code((size_t) (pos + rows) * rc, 0);
    std::vector<uint16_t> delta((size_t) (pos + rows) * rd, 0);
    cudaMemcpy(code.data(), dcode, code.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(delta.data(), ddelta, delta.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    size_t bad = 0;
    for (size_t i = 0; i < ref.code.size(); ++i)
        if (code[(size_t) pos * rc + i] != ref.code[i]) ++bad;
    for (size_t i = 0; i < ref.delta.size(); ++i)
        if (delta[(size_t) pos * rd + i] != ref.delta[i]) ++bad;
    if (bad) std::fprintf(stderr, "  %s: %zu differ at pos %lld\n", name, bad, (long long) pos);
    check(bad == 0, name);

    cudaFree(dx);
    cudaFree(dcode);
    cudaFree(ddelta);
}

/// store_int8_at(): the row comes from the card, and with `ring` the destination wraps.
void case_at(const char* name, int n_kv, int head_dim, int rows, long long at, int ring, int kind) {
    const std::vector<float> src = make_rows(n_kv, head_dim, rows, kind);
    const Reference ref = ggml_reference(src, rows, (int64_t) n_kv * head_dim);
    const size_t rc = (size_t) n_kv * head_dim, rd = (size_t) n_kv * head_dim / kQ8_0Values;
    const int64_t slots = ring > 0 ? ring : (int64_t) at + rows;

    float* dx = (float*) dev_alloc(src.size() * sizeof(float));
    int8_t* dcode = (int8_t*) dev_alloc((size_t) slots * rc);
    uint16_t* ddelta = (uint16_t*) dev_alloc((size_t) slots * rd * sizeof(uint16_t));
    long long* d_at = (long long*) dev_alloc(sizeof(long long));
    cudaMemcpy(dx, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dcode, 0x7F, (size_t) slots * rc);
    cudaMemset(ddelta, 0x11, (size_t) slots * rd * sizeof(uint16_t));
    cudaMemcpy(d_at, &at, sizeof(long long), cudaMemcpyHostToDevice);

    store_int8_at(dx, dcode, ddelta, n_kv, head_dim, rows, d_at, ring, nullptr);
    cudaDeviceSynchronize();

    std::vector<int8_t> code((size_t) slots * rc, 0);
    std::vector<uint16_t> delta((size_t) slots * rd, 0);
    cudaMemcpy(code.data(), dcode, code.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(delta.data(), ddelta, delta.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    size_t bad = 0;
    for (int r = 0; r < rows; ++r) {
        const int64_t dst = ring > 0 ? (at + r) % ring : at + r;
        for (size_t i = 0; i < rc; ++i)
            if (code[(size_t) dst * rc + i] != ref.code[(size_t) r * rc + i]) ++bad;
        for (size_t i = 0; i < rd; ++i)
            if (delta[(size_t) dst * rd + i] != ref.delta[(size_t) r * rd + i]) ++bad;
    }
    if (bad) std::fprintf(stderr, "  %s: %zu differ (at %lld, ring %d)\n", name, bad, at, ring);
    check(bad == 0, name);

    cudaFree(dx);
    cudaFree(dcode);
    cudaFree(ddelta);
    cudaFree(d_at);
}

/// The reader's own sum (code * delta, one delta a block) stays within half a step of what was written.
void case_error(const char* name, int n_kv, int head_dim, int rows) {
    const size_t n = (size_t) n_kv * head_dim * rows, blocks = n / kQ8_0Values;
    const std::vector<float> src = make_rows(n_kv, head_dim, rows, 3);
    float* dx = (float*) dev_alloc(n * sizeof(float));
    int8_t* dcode = (int8_t*) dev_alloc(n);
    uint16_t* ddelta = (uint16_t*) dev_alloc(blocks * sizeof(uint16_t));
    cudaMemcpy(dx, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    store_int8(dx, dcode, ddelta, n_kv, head_dim, 0, rows, nullptr);
    cudaDeviceSynchronize();

    std::vector<int8_t> code(n);
    std::vector<uint16_t> delta(blocks);
    cudaMemcpy(code.data(), dcode, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta.data(), ddelta, blocks * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    float worst = 0.0f;
    for (size_t b = 0; b < blocks; ++b) {
        const float d = half_value(delta[b]);
        float amax = 0.0f;
        for (int j = 0; j < kQ8_0Values; ++j)
            amax = std::fmax(amax, std::fabs(src[b * kQ8_0Values + (size_t) j]));
        // half a code's step, plus the delta's own fp16 rounding (a relative 2^-11 of the block's largest value)
        const float bound = amax * (0.5f / 127.0f + 2.0f / 2048.0f) + 1e-7f;
        for (int j = 0; j < kQ8_0Values; ++j)
            worst = std::fmax(worst, std::fabs((float) code[b * kQ8_0Values + (size_t) j] * d -
                                               src[b * kQ8_0Values + (size_t) j]) - bound);
    }
    if (worst > 0.0f) std::fprintf(stderr, "  %s: a value is %g past the half-step bound\n", name, worst);
    check(worst <= 0.0f, name);

    cudaFree(dx);
    cudaFree(dcode);
    cudaFree(ddelta);
}

/// Both caches over the same values, for the attend comparisons below.
struct Caches {
    std::vector<uint16_t> kh, vh;   // fp16
    std::vector<int8_t> kc, vc;     // codes
    std::vector<uint16_t> kd, vd;   // deltas
    std::vector<float> qf;
    float scale = 0.0f;
};

Caches make_caches(int dim, int n_head, int kv_heads, int n_rows) {
    Caches c;
    const int kvw = kv_heads * dim;
    std::mt19937 rng(4242u);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> kf((size_t) n_rows * kvw), vf((size_t) n_rows * kvw);
    c.qf.resize((size_t) n_head * dim);
    for (float& x : kf) x = g(rng);
    for (float& x : vf) x = g(rng);
    for (float& x : c.qf) x = g(rng);
    c.scale = 1.0f / std::sqrt((float) dim);
    c.kh.resize(kf.size());
    c.vh.resize(vf.size());
    for (size_t i = 0; i < kf.size(); ++i) c.kh[i] = half_bits(kf[i]);
    for (size_t i = 0; i < vf.size(); ++i) c.vh[i] = half_bits(vf[i]);
    const Reference rk = ggml_reference(kf, n_rows, kvw), rv = ggml_reference(vf, n_rows, kvw);
    c.kc = rk.code;
    c.kd = rk.delta;
    c.vc = rv.code;
    c.vd = rv.delta;
    return c;
}

// attend() and attend_int8() over the same numbers: the fp16 caches and the int8 ones hold the same K and V, so a
// token's output must agree to within the quantization's own error.  This is what says the dp4a path - the
// query's codes against the key's, scaled by the two blocks' deltas - is the attention it replaced.
// dequant_int8_to_f16 against the same thing done on the host, half by half: the kernel that puts a Q8_0 cache back
// onto the fp16 path (llama.cpp converts a quantized cache to f16 the same way, for the same reason).
// What the pass costs: it is O(rows), against an attention that is O(rows^2), so the question is only whether it
// is small next to the GEMM it buys back.
void bench_dequant(int dim, int kv_heads, int n_rows) {
    const int pd = dim / kQ8_0Values;
    const int64_t n_blocks = (int64_t) n_rows * kv_heads * pd;
    const size_t nh = (size_t) n_blocks * kQ8_0Values;
    int8_t* dc = (int8_t*) dev_alloc(nh);
    uint16_t* dd = (uint16_t*) dev_alloc((size_t) n_blocks * sizeof(uint16_t));
    uint16_t* dh = (uint16_t*) dev_alloc(nh * sizeof(uint16_t));
    cudaMemset(dc, 1, nh);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) dequant_int8_to_f16(dc, dd, dh, n_blocks, nullptr);
    cudaEventRecord(a, nullptr);
    for (int i = 0; i < 20; ++i) dequant_int8_to_f16(dc, dd, dh, n_blocks, nullptr);
    cudaEventRecord(b, nullptr);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    ms /= 20.0f;
    std::printf("  dequant one layer of %d tokens: %.3f ms (%.1f MiB of halves)\n", n_rows, ms,
                (double) nh * 2.0 / 1048576.0);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    for (void* p : {(void*) dc, (void*) dd, (void*) dh})
        cudaFree(p);
}

void case_dequant_kernel(int dim, int n_head, int kv_heads, int n_rows) {
    const Caches c = make_caches(dim, n_head, kv_heads, n_rows);
    const int pd = dim / kQ8_0Values;
    const int64_t n_blocks = (int64_t) n_rows * kv_heads * pd;
    const size_t nh = (size_t) n_blocks * kQ8_0Values;

    int8_t* dc = (int8_t*) dev_alloc(c.kc.size());
    uint16_t* dd = (uint16_t*) dev_alloc(c.kd.size() * sizeof(uint16_t));
    uint16_t* dh = (uint16_t*) dev_alloc(nh * sizeof(uint16_t));
    cudaMemcpy(dc, c.kc.data(), c.kc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dd, c.kd.data(), c.kd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    dequant_int8_to_f16(dc, dd, dh, n_blocks, nullptr);
    std::vector<uint16_t> got(nh);
    cudaMemcpy(got.data(), dh, nh * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int64_t b = 0; b < n_blocks; ++b)
        for (int j = 0; j < kQ8_0Values; ++j) {
            const uint16_t want = half_bits((float) c.kc[b * kQ8_0Values + j] * half_value(c.kd[b]));
            if (got[b * kQ8_0Values + j] != want) ++bad;
        }
    std::printf("  dequant_int8_to_f16, %lld blocks: %d of %zu halves differ from the host's\n",
                (long long) n_blocks, bad, nh);
    check(bad == 0, "the dequantizing kernel writes the halves the host computes");
    for (void* p : {(void*) dc, (void*) dd, (void*) dh})
        cudaFree(p);
}

// Is the loss the quantization's or the reader's?  Dequantize the int8 cache back to fp16 on the host and run the
// fp16 kernel over it.  If that reproduces the fp16 cache's own output, the codes and the deltas are right and the
// int8 reader is what is wrong; if it does not, the layout the reader expects is not the one the writer made.
void case_dequant(const char* name, int dim, int n_head, int kv_heads, int n_rows) {
    const Caches c = make_caches(dim, n_head, kv_heads, n_rows);
    const int pd = dim / kQ8_0Values;
    std::vector<uint16_t> kdq(c.kh.size()), vdq(c.vh.size());
    for (int r = 0; r < n_rows; ++r)
        for (int h = 0; h < kv_heads; ++h)
            for (int b = 0; b < pd; ++b) {
                const size_t blk = (size_t) (r * kv_heads + h) * pd + b;
                const float dk = half_value(c.kd[blk]), dv = half_value(c.vd[blk]);
                for (int j = 0; j < kQ8_0Values; ++j) {
                    const size_t at = (size_t) (r * kv_heads + h) * dim + b * kQ8_0Values + j;
                    kdq[at] = half_bits((float) c.kc[at] * dk);
                    vdq[at] = half_bits((float) c.vc[at] * dv);
                }
            }

    float* dq = (float*) dev_alloc(c.qf.size() * sizeof(float));
    uint16_t* dkh = (uint16_t*) dev_alloc(c.kh.size() * sizeof(uint16_t));
    uint16_t* dvh = (uint16_t*) dev_alloc(c.vh.size() * sizeof(uint16_t));
    uint16_t* dkq = (uint16_t*) dev_alloc(kdq.size() * sizeof(uint16_t));
    uint16_t* dvq = (uint16_t*) dev_alloc(vdq.size() * sizeof(uint16_t));
    float* dsc = (float*) dev_alloc((size_t) n_head * n_rows * sizeof(float));
    float* doh = (float*) dev_alloc((size_t) n_head * dim * sizeof(float));
    float* doq = (float*) dev_alloc((size_t) n_head * dim * sizeof(float));
    cudaMemcpy(dq, c.qf.data(), c.qf.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkh, c.kh.data(), c.kh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvh, c.vh.data(), c.vh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dkq, kdq.data(), kdq.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvq, vdq.data(), vdq.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    attend(dq, dkh, dvh, n_rows - 1, 1, dsc, 1, doh, dim, n_head, kv_heads, c.scale, nullptr, 0, 0, false, nullptr, 0);
    attend(dq, dkq, dvq, n_rows - 1, 1, dsc, 1, doq, dim, n_head, kv_heads, c.scale, nullptr, 0, 0, false, nullptr, 0);
    cudaDeviceSynchronize();

    std::vector<float> a((size_t) n_head * dim), b((size_t) n_head * dim);
    cudaMemcpy(a.data(), doh, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), doq, b.size() * sizeof(float), cudaMemcpyDeviceToHost);
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    const double cos = na > 0 && nb > 0 ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    std::printf("  %s: cosine(fp16 cache, int8 cache dequantized) = %.6f\n", name, cos);
    check(cos > 0.999, "the int8 codes and deltas carry the fp16 cache's own values");
    for (void* p : {(void*) dq, (void*) dkh, (void*) dvh, (void*) dkq, (void*) dvq, (void*) dsc, (void*) doh,
                    (void*) doq})
        cudaFree(p);
}

void case_attend_c(const char* name, int dim, int n_head, int kv_heads, int n_rows, const Caches& c) {
    const int pd = dim / kQ8_0Values;

    float* dq = (float*) dev_alloc(c.qf.size() * sizeof(float));
    uint16_t* dkh = (uint16_t*) dev_alloc(c.kh.size() * sizeof(uint16_t));
    uint16_t* dvh = (uint16_t*) dev_alloc(c.vh.size() * sizeof(uint16_t));
    int8_t* dkc = (int8_t*) dev_alloc(c.kc.size());
    uint16_t* dkd = (uint16_t*) dev_alloc(c.kd.size() * sizeof(uint16_t));
    int8_t* dvc = (int8_t*) dev_alloc(c.vc.size());
    uint16_t* dvd = (uint16_t*) dev_alloc(c.vd.size() * sizeof(uint16_t));
    int8_t* dqc = (int8_t*) dev_alloc((size_t) n_head * dim);
    uint16_t* dqd = (uint16_t*) dev_alloc((size_t) n_head * pd * sizeof(uint16_t));
    float* dscore = (float*) dev_alloc((size_t) n_head * n_rows * sizeof(float));
    float* dout_h = (float*) dev_alloc((size_t) n_head * dim * sizeof(float));
    float* dout_q = (float*) dev_alloc((size_t) n_head * dim * sizeof(float));
    cudaMemcpy(dq, c.qf.data(), c.qf.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkh, c.kh.data(), c.kh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvh, c.vh.data(), c.vh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dkc, c.kc.data(), c.kc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dkd, c.kd.data(), c.kd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc, c.vc.data(), c.vc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvd, c.vd.data(), c.vd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // the last row of the cache, so the span a score kernel walks is n_rows and not one: the old call here passed
    // pos 0 with a single row and never exercised a multi-key attention at all
    attend(dq, dkh, dvh, n_rows - 1, 1, dscore, 1, dout_h, dim, n_head, kv_heads, c.scale, nullptr, 0, 0, false,
           nullptr, 0);
    attend_int8(dq, KvInt8{dqc, dqd}, KvInt8{dkc, dkd}, KvInt8{dvc, dvd}, n_rows - 1, 1, dscore, 1, dout_q, dim,
              n_head, kv_heads, c.scale, nullptr, 0, 0, false, nullptr, 0);
    cudaDeviceSynchronize();

    std::vector<float> oh((size_t) n_head * dim), oq((size_t) n_head * dim);
    cudaMemcpy(oh.data(), dout_h, oh.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(oq.data(), dout_q, oq.size() * sizeof(float), cudaMemcpyDeviceToHost);

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < oh.size(); ++i) {
        dot += (double) oh[i] * oq[i];
        na += (double) oh[i] * oh[i];
        nb += (double) oq[i] * oq[i];
    }
    const double cos = na > 0 && nb > 0 ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    std::printf("  %s: cosine(attend, attend_int8) = %.6f\n", name, cos);
    check(cos > 0.999, "attend_int8 is the attention attend() is");

    for (void* p : {(void*) dq, (void*) dkh, (void*) dvh, (void*) dkc, (void*) dkd, (void*) dvc, (void*) dvd,
                    (void*) dqc, (void*) dqd, (void*) dscore, (void*) dout_h, (void*) dout_q})
        cudaFree(p);
}

void case_attend(const char* name, int dim, int n_head, int kv_heads, int n_rows) {
    case_attend_c(name, dim, n_head, kv_heads, n_rows, make_caches(dim, n_head, kv_heads, n_rows));
}

// The query side's own quantization, with the cache's storage error taken out of the comparison: the fp16
// caches hold exactly what the int8 codes represent (each value quantized, then dequantized on the host, the
// same rounding the dequant kernel does), so the only difference the two calls see is the query's own codes.
// If this holds as well as the storage cases did, the query side needs nothing.
void case_attend_q(const char* name, int dim, int n_head, int kv_heads, int n_rows) {
    Caches c = make_caches(dim, n_head, kv_heads, n_rows);
    for (size_t i = 0; i < c.kh.size(); ++i) c.kh[i] = half_bits((float) c.kc[i] * half_value(c.kd[i / kQ8_0Values]));
    for (size_t i = 0; i < c.vh.size(); ++i) c.vh[i] = half_bits((float) c.vc[i] * half_value(c.vd[i / kQ8_0Values]));
    case_attend_c(name, dim, n_head, kv_heads, n_rows, c);
}

// case_attend() with a prompt batch in the call instead of one token: `rows` tokens at a time, in `chunk`-sized
// pieces.  A single token's call has nq = per queries, which the 16-query block of the mma path holds inside one
// token; a batch spreads one block over 16 / per tokens, and a group of fewer than eight queries reaches rows the
// one-token cases never do.  Gemma 4's window layers are that case (16 heads of 256, 8 of them holding keys).
void case_attend_rows(const char* name, int dim, int n_head, int kv_heads, int n_rows, int rows, int chunk) {
    Caches c = make_caches(dim, n_head, kv_heads, n_rows);
    c.qf.resize((size_t) rows * n_head * dim);
    std::mt19937 rng(31337u);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (float& x : c.qf) x = g(rng);
    const int pd = dim / kQ8_0Values;

    float* dq = (float*) dev_alloc(c.qf.size() * sizeof(float));
    uint16_t* dkh = (uint16_t*) dev_alloc(c.kh.size() * sizeof(uint16_t));
    uint16_t* dvh = (uint16_t*) dev_alloc(c.vh.size() * sizeof(uint16_t));
    int8_t* dkc = (int8_t*) dev_alloc(c.kc.size());
    uint16_t* dkd = (uint16_t*) dev_alloc(c.kd.size() * sizeof(uint16_t));
    int8_t* dvc = (int8_t*) dev_alloc(c.vc.size());
    uint16_t* dvd = (uint16_t*) dev_alloc(c.vd.size() * sizeof(uint16_t));
    int8_t* dqc = (int8_t*) dev_alloc((size_t) rows * n_head * dim);
    uint16_t* dqd = (uint16_t*) dev_alloc((size_t) rows * n_head * pd * sizeof(uint16_t));
    // the scratch the score kernels need: a chunk's tokens against every key (the engine gives chunk_ * n_head *
    // max_context).  Sized as n_head * n_rows it holds a single token's row and nothing else, and a multi-row call
    // then writes past it - which is what the numbers this case first printed were measuring.
    const size_t scores = (size_t) rows * n_head * n_rows;
    float* dscore = (float*) dev_alloc(scores * sizeof(float));
    float* dout_h = (float*) dev_alloc((size_t) rows * n_head * dim * sizeof(float));
    float* dout_q = (float*) dev_alloc((size_t) rows * n_head * dim * sizeof(float));
    cudaMemcpy(dq, c.qf.data(), c.qf.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkh, c.kh.data(), c.kh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvh, c.vh.data(), c.vh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dkc, c.kc.data(), c.kc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dkd, c.kd.data(), c.kd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc, c.vc.data(), c.vc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvd, c.vd.data(), c.vd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    const int64_t pos = n_rows - rows;   // the batch ends on the cache's last row
    attend(dq, dkh, dvh, pos, rows, dscore, chunk, dout_h, dim, n_head, kv_heads, c.scale, nullptr, 0, 0, false,
           nullptr, 0);
    attend_int8(dq, KvInt8{dqc, dqd}, KvInt8{dkc, dkd}, KvInt8{dvc, dvd}, pos, rows, dscore, chunk, dout_q, dim,
              n_head, kv_heads, c.scale, nullptr, 0, 0, false, nullptr, 0);
    cudaDeviceSynchronize();

    std::vector<float> oh((size_t) rows * n_head * dim), oq(oh.size());
    cudaMemcpy(oh.data(), dout_h, oh.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(oq.data(), dout_q, oq.size() * sizeof(float), cudaMemcpyDeviceToHost);

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < oh.size(); ++i) {
        dot += (double) oh[i] * oq[i];
        na += (double) oh[i] * oh[i];
        nb += (double) oq[i] * oq[i];
    }
    const double cos = na > 0 && nb > 0 ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    std::printf("  %s: cosine(attend, attend_int8) = %.6f\n", name, cos);
    check(cos > 0.999, "attend_int8 is the attention attend() is, with more rows than one too");

    for (void* p : {(void*) dq, (void*) dkh, (void*) dvh, (void*) dkc, (void*) dkd, (void*) dvc, (void*) dvd,
                    (void*) dqc, (void*) dqd, (void*) dscore, (void*) dout_h, (void*) dout_q})
        cudaFree(p);
}

// The same rows against attend_batch, which is the attention the ENGINE uses for every call of kStageRows or more
// (32) and the one the long-context runs were validated with.  attend() and attend_int8() are asked for fewer rows
// than that - a decode's one, and with --window the two to eight tokens a commit of accepted guesses reads - so
// this is the call that says which of the two sides is the one that is off.
void case_attend_rows_batch(const char* name, int dim, int n_head, int kv_heads, int n_rows, int rows,
                            int window = 0, int ring = 0) {
    Caches c = make_caches(dim, n_head, kv_heads, n_rows);
    c.qf.resize((size_t) rows * n_head * dim);
    std::mt19937 rng(31337u);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (float& x : c.qf) x = g(rng);
    const int pd = dim / kQ8_0Values;

    float* dq = (float*) dev_alloc(c.qf.size() * sizeof(float));
    uint16_t* dkh = (uint16_t*) dev_alloc(c.kh.size() * sizeof(uint16_t));
    uint16_t* dvh = (uint16_t*) dev_alloc(c.vh.size() * sizeof(uint16_t));
    int8_t* dkc = (int8_t*) dev_alloc(c.kc.size());
    uint16_t* dkd = (uint16_t*) dev_alloc(c.kd.size() * sizeof(uint16_t));
    int8_t* dvc = (int8_t*) dev_alloc(c.vc.size());
    uint16_t* dvd = (uint16_t*) dev_alloc(c.vd.size() * sizeof(uint16_t));
    int8_t* dqc = (int8_t*) dev_alloc((size_t) rows * n_head * dim);
    uint16_t* dqd = (uint16_t*) dev_alloc((size_t) rows * n_head * pd * sizeof(uint16_t));
    const size_t nout = (size_t) rows * n_head * dim;
    const size_t scores = (size_t) rows * n_head * n_rows;
    float* dscore = (float*) dev_alloc(scores * sizeof(float));
    float* dwork = (float*) dev_alloc((nout + nout / 2 + 3 * (size_t) rows * n_head + 64) * sizeof(float));
    float* dout_a = (float*) dev_alloc(nout * sizeof(float));
    float* dout_q = (float*) dev_alloc(nout * sizeof(float));
    float* dout_b = (float*) dev_alloc(nout * sizeof(float));
    cudaMemcpy(dq, c.qf.data(), c.qf.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkh, c.kh.data(), c.kh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvh, c.vh.data(), c.vh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dkc, c.kc.data(), c.kc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dkd, c.kd.data(), c.kd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc, c.vc.data(), c.vc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvd, c.vd.data(), c.vd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    const int64_t pos = n_rows - rows;
    void* blas = blas_create(nullptr);
    attend(dq, dkh, dvh, pos, rows, dscore, rows, dout_a, dim, n_head, kv_heads, c.scale, nullptr, window, ring,
           false, nullptr, 0);
    attend_int8(dq, KvInt8{dqc, dqd}, KvInt8{dkc, dkd}, KvInt8{dvc, dvd}, pos, rows, dscore, rows, dout_q, dim,
              n_head, kv_heads, c.scale, nullptr, window, ring, false, nullptr, 0);
    attend_batch(blas, dq, dkh, dvh, pos, rows, dscore, scores, dwork, dout_b, dim, n_head, kv_heads,
                 c.scale, nullptr, window, ring, false);
    cudaDeviceSynchronize();
    blas_destroy(blas);

    std::vector<float> oa(nout), oq(nout), ob(nout);
    cudaMemcpy(oa.data(), dout_a, nout * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(oq.data(), dout_q, nout * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ob.data(), dout_b, nout * sizeof(float), cudaMemcpyDeviceToHost);
    const auto cos_of = [&](const std::vector<float>& x, const std::vector<float>& y) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            dot += (double) x[i] * y[i];
            na += (double) x[i] * x[i];
            nb += (double) y[i] * y[i];
        }
        return na > 0 && nb > 0 ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    };
    std::printf("  %s: cos(attend, batch) = %.6f   cos(attend_int8, batch) = %.6f\n", name, cos_of(oa, ob),
                cos_of(oq, ob));

    for (void* p : {(void*) dq, (void*) dkh, (void*) dvh, (void*) dkc, (void*) dkd, (void*) dvc, (void*) dvd,
                    (void*) dqc, (void*) dqd, (void*) dscore, (void*) dwork, (void*) dout_a, (void*) dout_q,
                    (void*) dout_b})
        cudaFree(p);
}
void bench_attend(int dim, int n_head, int kv_heads, int n_keys, int reps, int rows = 1) {
    const Caches c = make_caches(dim, n_head, kv_heads, n_keys);
    const int pd = dim / kQ8_0Values;

    float* dq = (float*) dev_alloc(c.qf.size() * sizeof(float));
    uint16_t* dkh = (uint16_t*) dev_alloc(c.kh.size() * sizeof(uint16_t));
    uint16_t* dvh = (uint16_t*) dev_alloc(c.vh.size() * sizeof(uint16_t));
    int8_t* dkc = (int8_t*) dev_alloc(c.kc.size());
    uint16_t* dkd = (uint16_t*) dev_alloc(c.kd.size() * sizeof(uint16_t));
    int8_t* dvc = (int8_t*) dev_alloc(c.vc.size());
    uint16_t* dvd = (uint16_t*) dev_alloc(c.vd.size() * sizeof(uint16_t));
    int8_t* dqc = (int8_t*) dev_alloc((size_t) n_head * dim);
    uint16_t* dqd = (uint16_t*) dev_alloc((size_t) n_head * pd * sizeof(uint16_t));
    float* dscore = (float*) dev_alloc((size_t) rows * n_head * n_keys * sizeof(float));
    float* dout = (float*) dev_alloc((size_t) rows * n_head * dim * sizeof(float));
    cudaMemcpy(dq, c.qf.data(), c.qf.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkh, c.kh.data(), c.kh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvh, c.vh.data(), c.vh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dkc, c.kc.data(), c.kc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dkd, c.kd.data(), c.kd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc, c.vc.data(), c.vc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvd, c.vd.data(), c.vd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // a decode: one token at the end of n_keys rows (the span the score kernels walk comes from pos and rows)
    const int64_t pos = n_keys - rows;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    const auto once = [&](bool q8) {
        if (q8)
            attend_int8(dq, KvInt8{dqc, dqd}, KvInt8{dkc, dkd}, KvInt8{dvc, dvd}, pos, rows, dscore, rows, dout, dim,
                      n_head, kv_heads, c.scale, nullptr, 0, 0, false, nullptr, 0);
        else
            attend(dq, dkh, dvh, pos, rows, dscore, rows, dout, dim, n_head, kv_heads, c.scale, nullptr, 0, 0, false,
                   nullptr, 0);
    };
    const auto time_it = [&](bool q8) {
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) once(q8);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        return ms / (float) reps;
    };
    for (int i = 0; i < 20; ++i) once(false), once(true);   // warm up, and let the clock settle
    // A round is a few hundred launches of tens of microseconds, so the card's clock and whatever else is on it
    // move a mean by more than the difference being measured.  Alternate which side goes first and compare the two
    // of a round - they then saw the same clock - reporting the middle pair and, for reference, the best of each.
    constexpr int ROUNDS = 7;
    float fp16[ROUNDS], q8ms[ROUNDS];
    for (int r = 0; r < ROUNDS; ++r) {
        if (r % 2 == 0) {
            fp16[r] = time_it(false);
            q8ms[r] = time_it(true);
        } else {
            q8ms[r] = time_it(true);
            fp16[r] = time_it(false);
        }
    }
    float ratio[ROUNDS];
    for (int r = 0; r < ROUNDS; ++r) ratio[r] = q8ms[r] / fp16[r];
    std::sort(ratio, ratio + ROUNDS);
    const float h = *std::min_element(fp16, fp16 + ROUNDS), q = *std::min_element(q8ms, q8ms + ROUNDS);
    std::printf("  %d keys x %d q: fp16 %.3f ms, q8 %.3f ms -> q8 is %.2fx  (median of %d; best %.2fx)\n",
                n_keys, rows, h, q, ratio[ROUNDS / 2], ROUNDS, q / h);

    cudaEventDestroy(a);
    cudaEventDestroy(b);
    for (void* p : {(void*) dq, (void*) dkh, (void*) dvh, (void*) dkc, (void*) dkd, (void*) dvc, (void*) dvd,
                    (void*) dqc, (void*) dqd, (void*) dscore, (void*) dout})
        cudaFree(p);
}

// The prompt path's two ways, head to head.  A batch of `rows` queries at the end of `n_keys` rows: cuBLAS over the
// fp16 cache (attend_batch -- what --kv fp16 does, and what the int8 cache is converted onto, so it is the number
// the quantized path has to beat once its conversion pass is counted) against the q8 kernels over the quantized
// cache (attend_int8, queries quantized first as the engine does).  Both get the engine's own scores scratch,
// chunk * n_head * max_context floats (Qwen35's chunk_ at the context the run uses).
// Which resource each one is up against is what makes the difference readable: bytes moved over the card's
// bandwidth says whether a cache half the size is worth anything at all.
void bench_attend_batch(int dim, int n_head, int kv_heads, int n_keys, int rows, int chunk, int reps) {
    const Caches c = make_caches(dim, n_head, kv_heads, n_keys);
    const int pd = dim / kQ8_0Values;
    const size_t qn = (size_t) rows * n_head * dim;
    const size_t score_floats = (size_t) chunk * n_head * n_keys;
    const size_t work_floats = (size_t) rows * n_head * (dim + dim / 2 + 3) + 1;

    // a batch's queries: make_caches() has one token's worth
    std::vector<float> qf(qn);
    {
        std::mt19937 rng(909u);
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (float& x : qf) x = g(rng);
    }

    float* dq = (float*) dev_alloc(qn * sizeof(float));
    uint16_t* dkh = (uint16_t*) dev_alloc(c.kh.size() * sizeof(uint16_t));
    uint16_t* dvh = (uint16_t*) dev_alloc(c.vh.size() * sizeof(uint16_t));
    int8_t* dkc = (int8_t*) dev_alloc(c.kc.size());
    uint16_t* dkd = (uint16_t*) dev_alloc(c.kd.size() * sizeof(uint16_t));
    int8_t* dvc = (int8_t*) dev_alloc(c.vc.size());
    uint16_t* dvd = (uint16_t*) dev_alloc(c.vd.size() * sizeof(uint16_t));
    int8_t* dqc = (int8_t*) dev_alloc(qn);
    uint16_t* dqd = (uint16_t*) dev_alloc((size_t) rows * n_head * pd * sizeof(uint16_t));
    float* dscore = (float*) dev_alloc(score_floats * sizeof(float));
    float* dwork = (float*) dev_alloc(work_floats * sizeof(float));
    float* dout = (float*) dev_alloc(qn * sizeof(float));

    cudaMemcpy(dq, qf.data(), qn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkh, c.kh.data(), c.kh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvh, c.vh.data(), c.vh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dkc, c.kc.data(), c.kc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dkd, c.kd.data(), c.kd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc, c.vc.data(), c.vc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvd, c.vd.data(), c.vd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    void* blas = nullptr;
    if (cublasCreate((cublasHandle_t*) &blas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        std::exit(2);
    }

    const int64_t pos = n_keys - rows;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    const auto once_batch = [&] {
        attend_batch(blas, dq, dkh, dvh, pos, rows, dscore, score_floats, dwork, dout, dim, n_head, kv_heads, c.scale,
                     nullptr, 0, 0, false);
    };
    const auto once_q8 = [&] {
        store_int8(dq, dqc, dqd, n_head, dim, 0, rows, nullptr);   // the batch's queries in the cache's own form
        attend_int8(dq, KvInt8{dqc, dqd}, KvInt8{dkc, dkd}, KvInt8{dvc, dvd}, pos, rows, dscore, chunk, dout, dim,
                  n_head, kv_heads, c.scale, nullptr, 0, 0, false, nullptr, 0);
    };
    const auto time_it = [&](bool q8) {
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) {
            if (q8) once_q8();
            else once_batch();
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        return ms / (float) reps;
    };
    for (int i = 0; i < 3; ++i) once_batch(), once_q8();   // warm up, and let the clock settle
    constexpr int ROUNDS = 5;
    float fp16[ROUNDS], q8ms[ROUNDS];
    for (int r = 0; r < ROUNDS; ++r) {
        if (r % 2 == 0) {
            fp16[r] = time_it(false);
            q8ms[r] = time_it(true);
        } else {
            q8ms[r] = time_it(true);
            fp16[r] = time_it(false);
        }
    }
    float ratio[ROUNDS];
    for (int r = 0; r < ROUNDS; ++r) ratio[r] = q8ms[r] / fp16[r];
    std::sort(ratio, ratio + ROUNDS);
    const float h = *std::min_element(fp16, fp16 + ROUNDS), q = *std::min_element(q8ms, q8ms + ROUNDS);
    std::printf("  prompt %d keys x %d q (chunk %d): cuBLAS %8.3f ms, q8 %8.3f ms -> q8 is %.2fx"
                "  (median of %d; best %.2fx)\n",
                n_keys, rows, chunk, h, q, ratio[ROUNDS / 2], ROUNDS, q / h);

    cublasDestroy((cublasHandle_t) blas);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    for (void* p : {(void*) dq, (void*) dkh, (void*) dvh, (void*) dkc, (void*) dkd, (void*) dvc, (void*) dvd,
                    (void*) dqc, (void*) dqd, (void*) dscore, (void*) dwork, (void*) dout})
        cudaFree(p);
}

}  // namespace

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device\n");
        return 2;
    }
    // n_kv = 2 and head_dim = 256 are Qwen3.6's; 256/32 tells how many blocks a head has
    const int n_kv = 2, head_dim = 256;
#ifndef STRATA_USE_HIP   // no int8 mma tier on HIP or before Ampere: attend_int8 never takes it there
    if (int8_tensor_cores()) check(probe_mma_layout(), "the m16n8k32 fragment layout is what I think it is");
#endif
    // FIRST, while the card still has room for anything: attend() and attend_int8() against attend_batch, which is
    // the attention the engine uses for every call of kStageRows (32) rows or more - a prompt chunk and, with
    // --window, a commit of accepted guesses.  attend() is asked for fewer rows than that, and this is the only
    // place that says whether its multi-row form is the one that is off or the q8 readers are.
    case_attend_rows_batch("rows  1, 16 heads of 256 over 8", 256, 16, 8, 128, 1);
    case_attend_rows_batch("rows  8, 16 heads of 256 over 8", 256, 16, 8, 128, 8);
    case_attend_rows_batch("rows 32, 16 heads of 256 over 8", 256, 16, 8, 128, 32);
    case_attend_rows_batch("rows  8, 16 heads of 512 over 2", 512, 16, 2, 128, 8);
    case_attend_rows_batch("rows  8, 32 heads of 64 over 8", 64, 32, 8, 128, 8);
    case_attend_rows_batch("rows  8, 16 heads of 256 over 8, 2048 keys", 256, 16, 8, 2048, 8);
    // A sliding window: Gemma 4's 24 window layers decode with window 1024 over a ring of 2048 rows, and no case in
    // this file has ever set either - n_rows == ring here, so the ring never wraps and what is under test is the
    // span a windowed call reads.
    case_attend_rows_batch("rows  1, window 512 of a 1024 ring", 256, 16, 8, 1024, 1, 512, 1024);
    case_attend_rows_batch("rows  8, window 512 of a 1024 ring", 256, 16, 8, 1024, 8, 512, 1024);
    case_attend_rows_batch("rows  1, window 256 of a 1024 ring", 256, 16, 8, 1024, 1, 256, 1024);
    case_attend_rows_batch("rows  8, window 256 of a 1024 ring", 256, 16, 8, 1024, 8, 256, 1024);
#ifndef STRATA_USE_HIP
    if (int8_tensor_cores()) {
        double tm = 0.0, td = 0.0;
        probe_mma_speed(&tm, &td);
        std::printf("  one 16x8x256 product x20000 x384 warps: mma %.1f ms, dp4a %.1f ms -> mma is %.1fx\n",
                    tm, td, td / tm);
    }
#endif
    for (int kind = 0; kind < 4; ++kind) {
        char name[96];
        std::snprintf(name, sizeof name, "store_int8 rows=5 kind=%d", kind);
        case_plain(name, n_kv, head_dim, 5, kind);
    }
    case_pos("store_int8 pos=13", n_kv, head_dim, 4, 13, 3);
    case_at("store_int8_at at=0 ring=0", n_kv, head_dim, 4, 0, 0, 1);
    case_at("store_int8_at at=0 ring=8", n_kv, head_dim, 4, 0, 8, 2);
    case_at("store_int8_at at=3 ring=5", n_kv, head_dim, 4, 3, 5, 3);    // wraps: rows 0..3 -> 3,4,0,1
    case_error("the half-step bound", n_kv, head_dim, 4);
    case_dequant_kernel(256, 16, 2, 64);
    case_dequant_kernel(256, 16, 2, 3000);
    std::printf("=== what the dequantizing pass costs ===\n");
    bench_dequant(256, 2, 8192);
    bench_dequant(256, 2, 32768);
    case_dequant("dequant round trip, 8 keys", head_dim, 16, n_kv, 8);
    case_dequant("dequant round trip, 512 keys", head_dim, 16, n_kv, 512);
    case_attend("attend vs attend_int8, 8 keys", head_dim, 16, n_kv, 8);
    // the query's codes alone: the same shapes with the cache's storage error taken out
    case_attend_q("query codes alone, 512 keys", head_dim, 16, n_kv, 512);
    case_attend_q("query codes alone, 512-value heads over 2", 512, 16, 2, 512);
    case_attend("attend vs attend_int8, 64 keys", head_dim, 16, n_kv, 64);
    case_attend("attend vs attend_int8, 512 keys", head_dim, 16, n_kv, 512);
    case_attend("attend vs attend_int8, 3000 keys", head_dim, 16, n_kv, 3000);
    // Gemma 4's window layers: 16 heads of 256 values with 8 of them holding keys - a group of TWO queries, where
    // every 256-value case above is n_head 16 over 2 (a group of eight).
    case_dequant_kernel(256, 16, 8, 64);
    case_attend("attend vs attend_int8, 16 heads of 256 over 8, 64 keys", 256, 16, 8, 64);
    case_attend("attend vs attend_int8, 16 heads of 256 over 8, 512 keys", 256, 16, 8, 512);
    // and the same shape in a batch, which is where its 16-query block spans several tokens
    case_attend_rows("batch of 64, 16 heads of 256 over 2 (per 8)", 256, 16, 2, 512, 64, 64);
    case_attend_rows("batch of 64, 16 heads of 256 over 8 (per 2)", 256, 16, 8, 512, 64, 64);
    // The sizes the engine actually hands attend_int8: kStageRows is 32, so every call below that - the decode's 1
    // and, with --window, the 2..8 tokens a commit of accepted guesses reads at once - goes to the q8 readers.
    case_attend_rows("batch of 8, 16 heads of 256 over 8 (per 2)", 256, 16, 8, 512, 8, 8);
    case_attend_rows("batch of 8, 16 heads of 256 over 2 (per 8)", 256, 16, 2, 512, 8, 8);
    // Gemma 4's full layers: 16 heads of 512 values with 2 of them holding keys - the widest head the q8 readers
    // have (attn_score_int8_kernel<512 / WARP>, a lane 16 values).  Nothing above covers it.
    case_attend("attend vs attend_int8, 16 heads of 512 over 2, 512 keys", 512, 16, 2, 512);
    // 24B-A2B (LFM2.5): 32 heads of 64 values, 8 of them holding keys.  A head narrower than a warp takes
    // attn_score_int8_kernel<8, 8> - eight lanes own it and the warp covers the group's four queries in one round -
    // so these are the cases that say the 64-value form is the same attention.
    case_dequant_kernel(64, 32, 8, 64);
    case_dequant("dequant round trip, 64-value heads, 8 keys", 64, 32, 8, 8);
    case_dequant("dequant round trip, 64-value heads, 512 keys", 64, 32, 8, 512);
    case_attend("attend vs attend_int8, 64-value heads, 8 keys", 64, 32, 8, 8);
    case_attend("attend vs attend_int8, 64-value heads, 64 keys", 64, 32, 8, 64);
    case_attend("attend vs attend_int8, 64-value heads, 512 keys", 64, 32, 8, 512);
    case_attend("attend vs attend_int8, 64-value heads, 3000 keys", 64, 32, 8, 3000);
    std::printf("=== the score path, fp16 vs %d-value blocks (16 heads over 2, dim 256) ===\n", kQ8_0Values);
    bench_attend(head_dim, 16, n_kv, 512, 300);
    bench_attend(head_dim, 16, n_kv, 2048, 200);
    bench_attend(head_dim, 16, n_kv, 4096, 80);
    bench_attend(head_dim, 16, n_kv, 16384, 30);
    bench_attend(head_dim, 16, n_kv, 8192, 50);
    bench_attend(head_dim, 16, n_kv, 32768, 20);
    std::printf("=== 24B-A2B's shape: 32 heads of 64 values, 8 of them holding keys ===\n");
    bench_attend(64, 32, 8, 2048, 100);
    bench_attend(64, 32, 8, 8192, 50);
    bench_attend(64, 32, 8, 32768, 20);
    bench_attend(64, 32, 8, 8192, 40, 8);
    std::printf("=== a prompt batch: 8 queries at a time ===\n");
    bench_attend(head_dim, 16, n_kv, 2048, 60, 8);
    bench_attend(head_dim, 16, n_kv, 4096, 60, 8);
    bench_attend(head_dim, 16, n_kv, 16384, 25, 8);
    bench_attend(head_dim, 16, n_kv, 8192, 40, 8);
    bench_attend(head_dim, 16, n_kv, 32768, 12, 8);
    std::printf("=== how far does a bigger chunk go? (8192 keys) ===\n");
    bench_attend(head_dim, 16, n_kv, 8192, 30, 1);
    bench_attend(head_dim, 16, n_kv, 8192, 20, 16);
    bench_attend(head_dim, 16, n_kv, 8192, 15, 32);
    bench_attend(head_dim, 16, n_kv, 8192, 10, 64);

    // The prompt path's two ways.  `rows` 512 is the engine's token block: kAttendResults caps cuBLAS's chunk of
    // queries there, and `chunk` 32 is Qwen35's chunk_ at a 32k context, the scratch both sides have to fit in.
    std::printf("=== the prompt path: cuBLAS over the halves against the q8 kernels over the codes ===\n");
    bench_attend_batch(head_dim, 16, n_kv, 8192, 512, 32, 10);
    bench_attend_batch(head_dim, 16, n_kv, 16384, 512, 32, 8);
    bench_attend_batch(head_dim, 16, n_kv, 32768, 512, 32, 5);
    bench_attend_batch(head_dim, 16, n_kv, 32768, 512, 64, 5);

    std::printf("ok: %d checks\n", checks);
    return 0;
}
