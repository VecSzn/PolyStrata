// src/poly/kernels.cu - see the header.  The reference for every formula is llama.cpp's graph of the model and the
// ggml ops it builds.  Where a kernel takes several tokens, blockIdx.y is the token unless it says otherwise.
#include "strata/poly/kernels.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

constexpr int WARP = 32;
constexpr int HC = 4;   // hyper-connection streams; the model's hc_mult, checked at load
constexpr int KEY_BLOCK = 16;   // cache rows one block of dsa_keys_kernel copies
// attn_score_kernel: a block's warps, a row and a group each; the values of a key a thread reads at a step; the
// queries a key's values are kept for; attn_score_reg_kernel's rows a warp
constexpr int SCORE_WARPS = 8, SCORE_VALUES = 4, SCORE_QUERIES = 4, SCORE_KEYS = 8;
// attend_batch(): the bytes of a chunk's results, which every block of rows reads and writes once more: in the
// card's L2 cache (32 MiB on the 4070 Laptop's AD106) that costs little, from its memory a lot
constexpr size_t kAttendResults = size_t(8) << 20;
// attn_value_kernel: a block's value dimensions, a sum's shares, a block's queries
constexpr int VALUE_DIMS = 32, VALUE_SHARES = 32, VALUE_QUERIES = 8;
constexpr int KDA_TERMS = 16, KDA_COLUMNS = 16;   // kda_state_kernel: a thread's values, a block's columns
// the indexer: a key's values; idx_score_kernel's keys a tile and tokens a block; idx_select_kernel's threads
constexpr int IDX_DIM = 128, IDX_TILE = 32, IDX_WARPS = 8, IDX_SELECT = 1024;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

// One value per thread summed over the block.  blockDim.x is a multiple of 32, and `part` has a float a warp: 8
// where a block is 256 threads at most, 32 where it can be 1024.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();   // part is free again
    return s;
}

// A value summed over each run of `lanes` threads of a warp; lanes is a power of two.
__device__ __forceinline__ float lanes_sum(float x, int lanes) {
    for (int o = lanes / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

__device__ __forceinline__ float block_max(float x, float* part) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = part[0];
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float ldw(const float* w, size_t i) { return w[i]; }
__device__ __forceinline__ float ldw(const uint16_t* w, size_t i) { return __uint_as_float(uint32_t(w[i]) << 16); }

template<typename W>
__global__ void gemv_kernel(const W* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int n_in,
                            int n_out, int rows_per_x) {
    __shared__ float part[32];
    const int o = blockIdx.x;
    const size_t tok = blockIdx.y;
    const W* wr = w + size_t(o) * n_in;
    const float* xr = x + (tok * size_t(n_out / rows_per_x) + size_t(o / rows_per_x)) * n_in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += ldw(wr, i) * xr[i];
    acc = block_sum(acc, part);
    if (threadIdx.x == 0) y[tok * n_out + o] = acc;
}

// The threads of a block that sums over n values.  A thread reads its values one after the other, and a kernel
// this small takes as long as those reads in a row: four a thread at most, while a block has the threads.  With
// 256 threads for 4096 values, GLM 5.3 Flash's layer norms, small products and stream mixing were 2.7 of the 11
// ms a token its card worked before a layer's router had its scores.
int sum_threads(int n) {
    int threads = 128;
    while (threads < 1024 && threads * 4 < n) threads *= 2;
    return threads;
}

template<typename W>
void gemv(const W* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream) {
    const int threads = n_in >= 128 ? sum_threads(n_in) : 32;
    gemv_kernel<W><<<dim3(n_out, rows), threads, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, rows_per_x);
}

__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ w, float* __restrict__ out, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ldw(w, i);
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* y, int n,
                                float eps, int y_stride) {
    __shared__ float part[32];
    const float* xr = x + size_t(blockIdx.x) * n;
    float* yr = y + size_t(blockIdx.x) * y_stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
}

__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* y, int n, float eps) {
    __shared__ float part[8];
    const float* xr = x + size_t(blockIdx.x) * n;
    float* yr = y + size_t(blockIdx.x) * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, part) / float(n);
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float scale = 1.0f / sqrtf(block_sum(v, part) / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
}

// mix[o] = fn row o . r for o < n_mix; mix[n_mix] = sum of r^2 (the flat RMS norm's input)
__global__ void hc_mix_kernel(const float* __restrict__ r, const float* __restrict__ fn, float* __restrict__ mix,
                              int n, int n_mix) {
    __shared__ float part[32];
    const int o = blockIdx.x;
    r += size_t(blockIdx.y) * n;
    mix += size_t(blockIdx.y) * (n_mix + 1);
    float acc = 0.0f;
    if (o < n_mix) {
        const float* fr = fn + size_t(o) * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) acc += fr[i] * r[i];
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) acc += r[i] * r[i];
    }
    acc = block_sum(acc, part);
    if (threadIdx.x == 0) mix[o] = acc;
}

__device__ void comb_norm_cols(float* comb, float eps) {
    for (int dst = 0; dst < HC; ++dst) {
        float sum = eps;
        for (int src = 0; src < HC; ++src) sum += comb[dst + HC * src];
        const float inv = 1.0f / sum;
        for (int src = 0; src < HC; ++src) comb[dst + HC * src] *= inv;
    }
}

__device__ void comb_norm_rows(float* comb, float eps) {
    for (int src = 0; src < HC; ++src) {
        float sum = eps;
        for (int dst = 0; dst < HC; ++dst) sum += comb[dst + HC * src];
        const float inv = 1.0f / sum;
        for (int dst = 0; dst < HC; ++dst) comb[dst + HC * src] *= inv;
    }
}

__global__ void hc_read_kernel(const float* __restrict__ r, const float* __restrict__ mix,
                               const float* __restrict__ scale, const float* __restrict__ base,
                               float* __restrict__ hcw, float* __restrict__ cur, int n_embd, float norm_eps,
                               float hc_eps, int iters) {
    __shared__ float pre[HC];
    r += size_t(blockIdx.y) * HC * n_embd;
    mix += size_t(blockIdx.y) * (HC * (2 + HC) + 1);
    hcw += size_t(blockIdx.y) * (HC + HC * HC);
    cur += size_t(blockIdx.y) * n_embd;
    if (threadIdx.x == 0) {
        const float rs = 1.0f / sqrtf(mix[HC * (2 + HC)] / float(HC * n_embd) + norm_eps);
        for (int h = 0; h < HC; ++h) pre[h] = sigmoid(mix[h] * rs * scale[0] + base[h]) + hc_eps;
        if (blockIdx.x == 0) {
            for (int h = 0; h < HC; ++h) hcw[h] = 2.0f * sigmoid(mix[HC + h] * rs * scale[1] + base[HC + h]);
            float comb[HC * HC];
            for (int src = 0; src < HC; ++src) {   // softmax over dst
                float mx = -INFINITY;
                for (int dst = 0; dst < HC; ++dst) {
                    const int k = dst + HC * src;
                    comb[k] = mix[2 * HC + k] * rs * scale[2] + base[2 * HC + k];
                    mx = fmaxf(mx, comb[k]);
                }
                float sum = 0.0f;
                for (int dst = 0; dst < HC; ++dst) {
                    const int k = dst + HC * src;
                    comb[k] = expf(comb[k] - mx);
                    sum += comb[k];
                }
                const float inv = 1.0f / sum;
                for (int dst = 0; dst < HC; ++dst) comb[dst + HC * src] = comb[dst + HC * src] * inv + hc_eps;
            }
            comb_norm_cols(comb, hc_eps);
            for (int it = 1; it < iters; ++it) {
                comb_norm_rows(comb, hc_eps);
                comb_norm_cols(comb, hc_eps);
            }
            for (int k = 0; k < HC * HC; ++k) hcw[HC + k] = comb[k];
        }
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float s = 0.0f;
    for (int h = 0; h < HC; ++h) s += r[size_t(h) * n_embd + i] * pre[h];
    cur[i] = s;
}

__global__ void hc_post_kernel(const float* __restrict__ x, const float* __restrict__ hcw, float* r, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    x += size_t(blockIdx.y) * n_embd;
    hcw += size_t(blockIdx.y) * (HC + HC * HC);
    r += size_t(blockIdx.y) * HC * n_embd;
    float in[HC], out[HC];
    for (int s = 0; s < HC; ++s) in[s] = r[size_t(s) * n_embd + i];
    const float xi = x[i];
    for (int dst = 0; dst < HC; ++dst) {
        float v = xi * hcw[dst];
        for (int src = 0; src < HC; ++src) v += in[src] * hcw[HC + dst + HC * src];
        out[dst] = v;
    }
    for (int s = 0; s < HC; ++s) r[size_t(s) * n_embd + i] = out[s];
}

__global__ void hc_mean_kernel(const float* __restrict__ r, float* __restrict__ y, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    r += size_t(blockIdx.y) * HC * n_embd;
    float acc = r[i];
    for (int s = 1; s < HC; ++s) acc += r[size_t(s) * n_embd + i];
    y[size_t(blockIdx.y) * n_embd + i] = acc * (1.0f / HC);
}

// d_conv is 4: three past inputs per channel, then the new one.  A thread is a channel and walks the tokens.
// The state as it is after each of the first n_snap tokens goes to snap, a state `stride` floats after the other.
__global__ void kda_conv_kernel(const float* __restrict__ proj, float* __restrict__ state,
                                const float* __restrict__ wq, const float* __restrict__ wk,
                                const float* __restrict__ wv, float* __restrict__ out, int d_inner, int rows,
                                float* __restrict__ snap, size_t stride, int n_snap) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3 * d_inner) return;
    const int m = t / d_inner, c = t % d_inner;
    const float* w = (m == 0 ? wq : m == 1 ? wk : wv) + 4 * size_t(c);
    float* st = state + size_t(t) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const size_t first = size_t(m) * rows * d_inner + c;
    for (int row = 0; row < rows; ++row) {
        const size_t at = first + size_t(row) * d_inner;
        const float x = proj[at];
        const float y = s0 * w[0] + s1 * w[1] + s2 * w[2] + x * w[3];
        s0 = s1;
        s1 = s2;
        s2 = x;
        out[at] = y * sigmoid(y);
        if (row < n_snap) {
            float* m = snap + size_t(row) * stride + size_t(t) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

// One block per head, and a column of the state to every `lanes` threads of it: a thread holds every lanes-th
// value of its column, so a column's two sums are a few terms a thread and a sum over its threads.  A thread a
// column read the column's 128 values one after the other, twice, and that was 25 microseconds a token.  The
// block walks the tokens.  s[i + hd * j] holds key dimension i of column j.  blockDim.x is hd * lanes.
__global__ void kda_step_kernel(const float* __restrict__ qkv, const float* __restrict__ g1,
                                const float* __restrict__ a, const float* __restrict__ beta, float* __restrict__ s,
                                const float* __restrict__ g2, const float* __restrict__ o_norm,
                                float* __restrict__ out, int hd, int n_head, int rows, float lower_bound,
                                float norm_eps, float* __restrict__ snap, size_t stride, int n_snap) {
    __shared__ float part[32];
    __shared__ float qn[256], kn[256], decay[256];
    const int lanes = int(blockDim.x) / hd;
    const int h = blockIdx.x, j = threadIdx.x / lanes, lane = threadIdx.x % lanes;
    const bool first = lane == 0;   // the thread that stands for its column in the sums over the head
    const size_t d_inner = size_t(hd) * n_head, plane = d_inner * rows;
    float* col = s + (size_t(h) * hd + j) * hd;
    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * d_inner + size_t(h) * hd + j;
        const float q = qkv[at], k = qkv[plane + at], v = qkv[2 * plane + at];
        const float qs = block_sum(first ? q * q : 0.0f, part);
        const float ks = block_sum(first ? k * k : 0.0f, part);
        if (first) {
            qn[j] = q / sqrtf(qs + 1e-6f) * (1.0f / sqrtf(float(hd)));
            kn[j] = k / sqrtf(ks + 1e-6f);
            // the decay gate of key dimension j: exp(lower_bound * sigmoid(-(g1 * a)))
            decay[j] = expf(lower_bound / (1.0f + expf(g1[at] * a[h])));
        }
        __syncthreads();
        // Which product is fused into which sum is spelled out, here and in kda_state_kernel: left to the
        // compiler, the two came out with other last bits.
        float sk = 0.0f;
        for (int i = lane; i < hd; i += lanes) sk = fmaf(__fmul_rn(col[i], decay[i]), kn[i], sk);
        sk = lanes_sum(sk, lanes);
        const float d = (v - sk) * sigmoid(beta[size_t(row) * n_head + h]);
        float o = 0.0f;
        float* m = row < n_snap ? snap + size_t(row) * stride + (size_t(h) * hd + j) * hd : nullptr;
        for (int i = lane; i < hd; i += lanes) {
            const float n = fmaf(col[i], decay[i], __fmul_rn(kn[i], d));
            col[i] = n;
            if (m) m[i] = n;
            o = fmaf(n, qn[i], o);
        }
        o = lanes_sum(o, lanes);
        // the sum's last barrier also ends this token's reads of qn, kn and decay
        const float os = block_sum(first ? o * o : 0.0f, part);
        if (first) out[at] = o / sqrtf(os / float(hd) + norm_eps) * o_norm[j] * sigmoid(g2[at]);
    }
}

// kda_step_kernel in three for a prompt batch.  There a head's block walks the tokens with every column's values
// in the card's memory, and waits for its warps three times a token, for sums that need no token before them.
// Here the norms of q and k come first for every token at once, the state's columns walk the tokens in registers,
// a few columns a block so that the blocks spread over the card, and the output's norm comes last.  The sums add
// in kda_step_kernel's order, so a token comes out with the same bits.

// The sum over a head's values that kda_step_kernel's block_sum makes of them, a thread a value here.  There they
// lie `lanes` threads apart, so a warp sums WARP / lanes of them (`group`) and the block adds the warps' sums in
// turn.  part has a float a group.
__device__ __forceinline__ float kda_head_sum(float x, float* part, int group) {
    x = lanes_sum(x, group);
    if (threadIdx.x % group == 0) part[threadIdx.x / group] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / group; ++k) s += part[k];
    __syncthreads();   // part is free again
    return s;
}

// One block a token and head, a thread a value: q and k normed in place, and g1 made the decay of its key
// dimension.
__global__ void kda_gates_kernel(float* __restrict__ qkv, float* __restrict__ g1, const float* __restrict__ a,
                                 int n_head, int rows, float lower_bound, int group) {
    __shared__ float part[32];
    const int hd = blockDim.x;
    const size_t at = size_t(blockIdx.x) * hd + threadIdx.x, plane = size_t(hd) * n_head * rows;
    const float q = qkv[at], k = qkv[plane + at];
    // the squares are rounded before they are added, as there
    const float qs = kda_head_sum(__fmul_rn(q, q), part, group);
    const float ks = kda_head_sum(__fmul_rn(k, k), part, group);
    qkv[at] = q / sqrtf(qs + 1e-6f) * (1.0f / sqrtf(float(hd)));
    qkv[plane + at] = k / sqrtf(ks + 1e-6f);
    g1[at] = expf(lower_bound / (1.0f + expf(g1[at] * a[blockIdx.x % n_head])));
}

// The state's update with what kda_gates_kernel left.  A column has `lanes` threads as in kda_step_kernel, each
// with KDA_TERMS of its values, and a block has KDA_COLUMNS of a head's columns.  out gets the output before its
// norm.
__global__ void kda_state_kernel(const float* __restrict__ qkv, const float* __restrict__ decay,
                                 const float* __restrict__ beta, float* __restrict__ s, float* __restrict__ out,
                                 int n_head, int rows) {
    const int lanes = int(blockDim.x) / KDA_COLUMNS, hd = lanes * KDA_TERMS;
    const int per_head = hd / KDA_COLUMNS;   // blocks a head
    const int h = blockIdx.x / per_head, j = (blockIdx.x % per_head) * KDA_COLUMNS + threadIdx.x / lanes;
    const int lane = threadIdx.x % lanes;
    const size_t d_inner = size_t(hd) * n_head, plane = d_inner * rows;
    float* col = s + (size_t(h) * hd + j) * hd + lane;
    float c[KDA_TERMS];
#pragma unroll
    for (int t = 0; t < KDA_TERMS; ++t) c[t] = col[t * lanes];
    for (int row = 0; row < rows; ++row) {
        const size_t head = size_t(row) * d_inner + size_t(h) * hd;
        const float* qn = qkv + head + lane;
        const float* kn = qkv + plane + head + lane;
        const float* dc = decay + head + lane;
        const float v = qkv[2 * plane + head + j];
        float sk = 0.0f;   // kda_step_kernel's fused products
#pragma unroll
        for (int t = 0; t < KDA_TERMS; ++t) sk = fmaf(__fmul_rn(c[t], dc[t * lanes]), kn[t * lanes], sk);
        sk = lanes_sum(sk, lanes);
        const float d = (v - sk) * sigmoid(beta[size_t(row) * n_head + h]);
        float o = 0.0f;
#pragma unroll
        for (int t = 0; t < KDA_TERMS; ++t) {
            const float n = fmaf(c[t], dc[t * lanes], __fmul_rn(kn[t * lanes], d));
            c[t] = n;
            o = fmaf(n, qn[t * lanes], o);
        }
        o = lanes_sum(o, lanes);
        if (lane == 0) out[head + j] = o;
    }
#pragma unroll
    for (int t = 0; t < KDA_TERMS; ++t) col[t * lanes] = c[t];
}

// One block a token and head, a thread a value: the output's norm, its weights and the gate.
__global__ void kda_out_kernel(float* __restrict__ out, const float* __restrict__ g2,
                               const float* __restrict__ o_norm, float norm_eps, int group) {
    __shared__ float part[32];
    const int hd = blockDim.x;
    const size_t at = size_t(blockIdx.x) * hd + threadIdx.x;
    const float o = out[at];
    const float os = kda_head_sum(__fmul_rn(o, o), part, group);
    out[at] = o / sqrtf(os / float(hd) + norm_eps) * o_norm[threadIdx.x] * sigmoid(g2[at]);
}

__global__ void store_f16_kernel(const float* __restrict__ x, half* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(x[i]);
}

// Rows from the position the card holds when the launch runs.
__global__ void store_f16_at_kernel(const float* __restrict__ x, half* __restrict__ cache, int n, int rows,
                                    const long long* __restrict__ at, int ring) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * rows) return;
    const long long row = ring > 0 ? (*at + i / n) % ring : *at + i / n;
    cache[size_t(row) * n + i % n] = __float2half(x[i]);
}

__global__ void store_int8_kernel(const float* __restrict__ x, int8_t* __restrict__ cache_q,
                                  uint16_t* __restrict__ cache_scale, int kv_heads, int head_dim,
                                  long long pos, size_t n_groups) {
    const size_t grp = blockIdx.x;
    if (grp >= n_groups) return;
    const int G = head_dim / 64;
    const size_t r = grp / (size_t(kv_heads) * G);
    const int rem = int(grp % (size_t(kv_heads) * G));
    const int h = rem / G, g = rem % G;
    const int t = threadIdx.x;

    const size_t in_idx = ((r * size_t(kv_heads) + h) * head_dim) + g * 64 + t;
    const float val = x[in_idx];

    float a = fabsf(val);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float s_max[2];
    if ((t & 31) == 0) s_max[t >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(s_max[0], s_max[1]);
    const uint16_t sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);

    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(val / sf);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }

    const long long row = pos + r;
    const size_t q_idx = ((size_t(row) * kv_heads + h) * head_dim) + g * 64 + t;
    cache_q[q_idx] = (int8_t) q;
    if (t == 0) {
        const size_t scale_idx = ((size_t(row) * kv_heads + h) * G) + g;
        cache_scale[scale_idx] = sbits;
    }
}

__global__ void store_int8_at_kernel(const float* __restrict__ x, int8_t* __restrict__ cache_q,
                                     uint16_t* __restrict__ cache_scale, int kv_heads, int head_dim,
                                     const long long* __restrict__ at, int ring, size_t n_groups) {
    const size_t grp = blockIdx.x;
    if (grp >= n_groups) return;
    const int G = head_dim / 64;
    const size_t r = grp / (size_t(kv_heads) * G);
    const int rem = int(grp % (size_t(kv_heads) * G));
    const int h = rem / G, g = rem % G;
    const int t = threadIdx.x;

    const size_t in_idx = ((r * size_t(kv_heads) + h) * head_dim) + g * 64 + t;
    const float val = x[in_idx];

    float a = fabsf(val);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float s_max[2];
    if ((t & 31) == 0) s_max[t >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(s_max[0], s_max[1]);
    const uint16_t sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);

    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(val / sf);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }

    const long long cur_pos = *at;
    const long long row = ring > 0 ? (cur_pos + r) % ring : (cur_pos + r);
    const size_t q_idx = ((size_t(row) * kv_heads + h) * head_dim) + g * 64 + t;
    cache_q[q_idx] = (int8_t) q;
    if (t == 0) {
        const size_t scale_idx = ((size_t(row) * kv_heads + h) * G) + g;
        cache_scale[scale_idx] = sbits;
    }
}

// How many cache rows token `tok` of a call from position pos attends to: every row up to its own while it is
// dense, then its list.
__host__ __device__ inline int dsa_keys(int tok, long long pos, int dense_rows, int top_rows, int kpool) {
    const long long n = pos + tok + 1;
    return tok >= dense_rows ? top_rows + int(n % kpool) : int(n);
}

// The keys of token first + blockIdx.y as float rows, n_keys of them; the rows past its last key are zero.  A
// block copies KEY_BLOCK rows.
__global__ void dsa_keys_kernel(const half* __restrict__ cache, const int32_t* __restrict__ sel_all, int sel_stride,
                                int top_rows, int kpool, long long pos, int first, int dense_rows,
                                float* __restrict__ keys, int rank, int n_keys) {
    const int tok = first + blockIdx.y;
    const int n_sel = dsa_keys(tok, pos, dense_rows, top_rows, kpool);
    const int32_t* sel = tok >= dense_rows ? sel_all + size_t(tok - dense_rows) * sel_stride : nullptr;
    float* dst = keys + size_t(blockIdx.y) * n_keys * rank;
    const int j0 = blockIdx.x * KEY_BLOCK;
    for (int i = threadIdx.x; i < KEY_BLOCK * rank; i += blockDim.x) {
        const int j = j0 + i / rank, r = i % rank;
        if (j >= n_keys) return;
        dst[size_t(j) * rank + r] = j < n_sel ? __half2float(cache[size_t(sel ? sel[j] : j) * rank + r]) : 0.0f;
    }
}

// Softmax in place over a head's scores of its token's keys; the entries past the last key become zero.  One
// block per head and token.
__global__ void dsa_softmax_kernel(float* score, int n_keys, int top_rows, int kpool, long long pos, int first,
                                   int dense_rows) {
    __shared__ float part[8];
    const int n_sel = dsa_keys(first + blockIdx.y, pos, dense_rows, top_rows, kpool);
    float* row = score + (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * n_keys;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n_sel; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n_sel; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    const float inv = 1.0f / block_sum(sum, part);
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) row[j] = j < n_sel ? row[j] * inv : 0.0f;
}

// One block per token from position pos.
__global__ void idx_store_kernel(const float* __restrict__ key, const float* __restrict__ gate,
                                 float* __restrict__ ring, int ring_rows, int dim, long long pos) {
    const int d = threadIdx.x;
    if (d >= dim) return;
    float* row = ring + size_t((pos + blockIdx.x) % ring_rows) * 2 * dim;
    row[d] = key[size_t(blockIdx.x) * dim + d];
    row[dim + d] = gate[size_t(blockIdx.x) * dim + d];
}

// One block per token from position pos; the ones that do not end a pool have nothing to do.
__global__ void idx_pool_kernel(const float* __restrict__ ring, int ring_rows, float* __restrict__ pool,
                                const float* __restrict__ ape, int dim, int kpool, long long pos) {
    const int d = threadIdx.x;
    const long long last = pos + blockIdx.x;
    if (d >= dim || (last + 1) % kpool != 0) return;
    const auto row = [&](int k) { return ring + size_t((last - kpool + 1 + k) % ring_rows) * 2 * dim; };
    float mx = -INFINITY;
    for (int k = 0; k < kpool; ++k) mx = fmaxf(mx, row(k)[dim + d] + ape[d + dim * k]);
    float sum = 0.0f, acc = 0.0f;
    for (int k = 0; k < kpool; ++k) {
        const float e = expf(row(k)[dim + d] + ape[d + dim * k] - mx);
        sum += e;
        acc += e * row(k)[d];
    }
    pool[size_t(last / kpool) * dim + d] = acc / sum;
}

// A block's warps are tokens, one each, that score a run of pools together, IDX_TILE keys at a time through shared
// memory; lane h holds head h's query.  A score is summed in one order in any company: a head's product along the
// key, then the heads by warp_sum().
template <int DIM>
__global__ void __launch_bounds__(IDX_WARPS * WARP)
    idx_score_kernel(const float* __restrict__ q, const float* __restrict__ w, const float* __restrict__ pool,
                     float* __restrict__ score, int score_stride, int n_head, int kpool, long long pos, float wscale,
                     int run, int rows) {
    __shared__ float4 tile[IDX_TILE * DIM / 4];
    const int warps = blockDim.x / WARP, h = threadIdx.x % WARP;
    const int tok = blockIdx.y * warps + threadIdx.x / WARP;
    const int first = blockIdx.x * run;
    const int last = min(first + run, int((pos + min(rows, int(blockIdx.y + 1) * warps)) / kpool));
    if (first >= last) return;   // not a whole pool at the block's last token yet
    const bool live = tok < rows;
    const int n_pool = live ? int((pos + tok + 1) / kpool) : 0;
    float qh[DIM];
    float wh = 0.0f;
    if (live && h < n_head) {
        const float4* src = reinterpret_cast<const float4*>(q + (size_t(tok) * n_head + h) * DIM);
#pragma unroll
        for (int d = 0; d < DIM / 4; ++d) {
            const float4 v = src[d];
            qh[4 * d] = v.x;
            qh[4 * d + 1] = v.y;
            qh[4 * d + 2] = v.z;
            qh[4 * d + 3] = v.w;
        }
        wh = w[size_t(tok) * n_head + h] * wscale;
    } else {
#pragma unroll
        for (int d = 0; d < DIM; ++d) qh[d] = 0.0f;
    }
    float* out = score + size_t(tok) * score_stride;
    const float4* keys = reinterpret_cast<const float4*>(pool);
    for (int p0 = first; p0 < last; p0 += IDX_TILE) {
        const int n = min(IDX_TILE, last - p0);
#pragma unroll 4
        for (int i = threadIdx.x; i < n * DIM / 4; i += blockDim.x) tile[i] = keys[size_t(p0) * (DIM / 4) + i];
        __syncthreads();
        const int m = min(n, n_pool - p0);   // the token's own pools in the tile
        for (int j = 0; j < m; j += 2) {   // two at a time
            const int j1 = min(j + 1, m - 1);
            const float4 *k0 = tile + j * (DIM / 4), *k1 = tile + j1 * (DIM / 4);
            float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
            for (int d = 0; d < DIM / 4; ++d) {
                const float4 a = k0[d], b = k1[d];
                d0 += qh[4 * d] * a.x;
                d1 += qh[4 * d] * b.x;
                d0 += qh[4 * d + 1] * a.y;
                d1 += qh[4 * d + 1] * b.y;
                d0 += qh[4 * d + 2] * a.z;
                d1 += qh[4 * d + 2] * b.z;
                d0 += qh[4 * d + 3] * a.w;
                d1 += qh[4 * d + 3] * b.w;
            }
            float s0 = 0.0f, s1 = 0.0f;
            if (h < n_head) {
                s0 += fmaxf(d0, 0.0f) * wh;
                s1 += fmaxf(d1, 0.0f) * wh;
            }
            s0 = warp_sum(s0);
            s1 = warp_sum(s1);
            if (h == 0) {
                out[p0 + j] = s0;
                if (j1 > j) out[p0 + j1] = s1;
            }
        }
        __syncthreads();
    }
}

// idx_score_kernel's sum over the heads for a prompt batch, whose products cuBLAS made: dot holds q[h] . pooled(p)
// for a token and head's n_pool pools together.  A thread a pool, blockIdx.y the token.
__global__ void idx_weigh_kernel(const float* __restrict__ dot, const float* __restrict__ w,
                                 float* __restrict__ score, int score_stride, int n_pool, int n_head, int kpool,
                                 long long pos, float wscale) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tok = blockIdx.y;
    if (p >= (pos + tok + 1) / kpool) return;   // not a whole pool at this token yet
    dot += tok * n_head * n_pool + p;
    w += tok * n_head;
    float acc = 0.0f;
    for (int h = 0; h < n_head; ++h) acc += fmaxf(dot[size_t(h) * n_pool], 0.0f) * (w[h] * wscale);
    score[tok * score_stride + p] = acc;
}

// A score as an integer: a higher score is a larger one, and -0 is +0, as float comparisons have them.
__device__ __forceinline__ uint32_t idx_key(float s) {
    const uint32_t b = __float_as_uint(s + 0.0f);
    return b & 0x80000000u ? ~b : b | 0x80000000u;
}

// x summed over the threads before this one, and over all of them in *all.  part has a slot a warp.
__device__ __forceinline__ int block_before(int x, int* part, int* all) {
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, warps = blockDim.x / WARP;
    int in = x;
#pragma unroll
    for (int o = 1; o < WARP; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffff, in, o, WARP);
        if (lane >= o) in += y;
    }
    if (lane == WARP - 1) part[wid] = in;
    __syncthreads();
    if (wid == 0) {
        int v = lane < warps ? part[lane] : 0;
#pragma unroll
        for (int o = 1; o < WARP; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffff, v, o, WARP);
            if (lane >= o) v += y;
        }
        if (lane < warps) part[lane] = v;
    }
    __syncthreads();
    const int before = (wid ? part[wid - 1] : 0) + in - x;
    *all = part[warps - 1];
    __syncthreads();   // part is free again
    return before;
}

// A token's n_top pools with the highest scores, the lower pool first among equals, as rows of its list in that
// order, then the rows no pool has yet.  One block of IDX_SELECT a token.  The cut, the n_top-th highest key, comes
// a byte at a time: each warp counts the next byte of the keys that agree with it so far.  Then a thread has a run
// of the token's pools.  The block's shared memory holds the chosen pools and then their keys, n_top each.
__global__ void __launch_bounds__(IDX_SELECT)
    idx_select_kernel(const float* __restrict__ score, int score_stride, int32_t* __restrict__ sel, int sel_stride,
                      int n_top, int kpool, long long pos) {
    extern __shared__ uint32_t chosen[];
    __shared__ int hist[IDX_SELECT / WARP][256], part[IDX_SELECT / WARP], pick[2];
    const long long at = pos + blockIdx.x;
    const int n_pool = int((at + 1) / kpool);
    const float* s = score + size_t(blockIdx.x) * score_stride;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    uint32_t cut = 0, mask = 0;
    int want = n_top;   // the cut's place among the keys that agree with it on mask
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int d = lane; d < 256; d += WARP) hist[wid][d] = 0;
        __syncwarp();
        for (int i = threadIdx.x; i < n_pool; i += blockDim.x) {
            const uint32_t k = idx_key(s[i]);
            if ((k & mask) == cut) atomicAdd(&hist[wid][(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (threadIdx.x < 256) {
            int t = 0;
            for (int k = 0; k < IDX_SELECT / WARP; ++k) t += hist[k][threadIdx.x];
            hist[0][threadIdx.x] = t;   // a thread reads and writes its own column only
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int d = 255, above = 0;
            for (; d > 0 && above + hist[0][d] < want; --d) above += hist[0][d];
            pick[0] = d;
            pick[1] = above;
        }
        __syncthreads();
        cut |= uint32_t(pick[0]) << shift;
        mask |= 255u << shift;
        want -= pick[1];
        __syncthreads();
    }
    // the pools above the cut, and of those on it the lowest, in the order of the pools
    const int per = (n_pool + int(blockDim.x) - 1) / int(blockDim.x);
    const int lo = threadIdx.x * per, hi = min(lo + per, n_pool);
    int g = 0, e = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = idx_key(s[i]);
        g += k > cut;
        e += k == cut;
    }
    int g_all, e_all;   // above and on the cut in all the threads
    const int g0 = block_before(g, part, &g_all);   // in the threads before this one
    int e0 = block_before(e, part, &e_all);
    const int need = n_top - g_all;
    int out = g0 + min(e0, need);
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = idx_key(s[i]);
        if (k > cut || (k == cut && e0++ < need)) {
            chosen[out] = i;
            chosen[n_top + out++] = k;
        }
    }
    __syncthreads();
    // a chosen pool's place: behind those with a higher score and the lower pools with its own
    for (int c = threadIdx.x; c < n_top; c += blockDim.x) {
        const uint32_t mine = chosen[n_top + c];
        int rank = 0;
        for (int k = 0; k < n_top; ++k) rank += chosen[n_top + k] > mine || (chosen[n_top + k] == mine && k < c);
        int32_t* row = sel + size_t(blockIdx.x) * sel_stride + size_t(rank) * kpool;
        for (int j = 0; j < kpool; ++j) row[j] = int32_t(chosen[c]) * kpool + j;
    }
    if (threadIdx.x != 0) return;
    int32_t* tail = sel + size_t(blockIdx.x) * sel_stride + size_t(n_top) * kpool;
    for (long long p = (long long) n_pool * kpool; p <= at; ++p) *tail++ = int32_t(p);
}

__global__ void swiglu_clamp_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                    float* __restrict__ out, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    out[i] = g / (1.0f + expf(-g)) * u;
}

__global__ void axpy_kernel(const float* __restrict__ x, float a, float* __restrict__ acc, int n, bool add) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] = add ? acc[i] + x[i] * a : x[i] * a;
}

__global__ void copy2_kernel(const uint32_t* __restrict__ x0, uint32_t* __restrict__ y0, int n0,
                             const uint32_t* __restrict__ x1, uint32_t* __restrict__ y1, int n1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n0) y0[i] = x0[i];
    else if (i < n0 + n1) y1[i - n0] = x1[i - n0];
}

// The sums may be one and the same, so a thread adds its value of each vector in turn.
__global__ void add_scaled_kernel(ScaledRows rows, int count, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = 0; k < count; ++k) rows.acc[k][i] = rows.acc[k][i] + rows.x[k][i] * rows.w[k];
}

__global__ void swiglu_clamp_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n,
                                         float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float g = fminf(row[i], limit);
    const float u = fminf(fmaxf(row[n + i], -limit), limit);
    out[size_t(blockIdx.y) * n + i] = g / (1.0f + expf(-g)) * u;
}

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}

__global__ void geglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gelu(gate[i]) * up[i];
}

__global__ void geglu_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    out[size_t(blockIdx.y) * n + i] = gelu(row[i]) * row[n + i];
}

__global__ void moe_sum_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ acc,
                               int n, int n_used) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t first = size_t(blockIdx.y) * n_used;
    float s = 0.0f;
    for (int k = 0; k < n_used; ++k) s += w[first + k] * x[(first + k) * n + i];
    acc[size_t(blockIdx.y) * n + i] += s;
}

__global__ void add_bias_kernel(const float* __restrict__ b, float* __restrict__ acc, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[size_t(blockIdx.y) * n + i] += b[i];
}

// One block per head and token.  The pairs of the rotation are i and i + n_rot / 2 for i below n_rot / 2.  With
// axes, a token has a position on each of three axes, and pair j turns by the one on axis (j % sections) % 3.
// With `at`, the first token's position is what the card holds there when the launch runs.
__global__ void head_norm_rope_kernel(const float* __restrict__ x, int x_stride, const float* __restrict__ w,
                                      float* __restrict__ y, int dim, int heads, int n_rot, float theta_scale,
                                      const float* __restrict__ freq, long long pos, float eps,
                                      const int* __restrict__ axes, int sections,
                                      const long long* __restrict__ at) {
    __shared__ float part[8];
    const float* xr = x + size_t(blockIdx.x) * x_stride;
    float* yr = y + size_t(blockIdx.x) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(dim) + eps);
    const int tok = blockIdx.x / heads;
    const float own = float((at ? *at : pos) + tok);
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i >= n_rot) {
            yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
            continue;
        }
        const int j = i < half ? i : i - half;
        const float p = axes ? float(axes[size_t(tok) * 3 + (j % sections) % 3]) : own;
        const float turn = p * powf(theta_scale, float(j));
        const float theta = freq ? turn / freq[j] : turn;
        const float x0 = w ? xr[j] * scale * w[j] : xr[j] * scale;
        const float x1 = w ? xr[j + half] * scale * w[j + half] : xr[j + half] * scale;
        yr[i] = i < half ? x0 * cosf(theta) - x1 * sinf(theta) : x0 * sinf(theta) + x1 * cosf(theta);
    }
}

// The rows the token at position p attends: how many, and the first of them.  They end at its own row, or at row
// `last` when that is given: the tokens of the call then see each other.
__device__ __forceinline__ int attn_rows(long long p, int window, long long last, long long* lo) {
    *lo = window > 0 && p >= window ? p - window + 1 : 0;
    return int((last < 0 ? p : last) - *lo) + 1;
}

// The rows the `tokens` tokens from position p attend between them: how many, and the first of them.  A launch's
// scores lie in that order, so that its tokens read a row of the caches once between them.
__device__ __forceinline__ int attn_span(long long p, int tokens, int window, long long last, long long* lo) {
    long long lo_last;
    const int n = attn_rows(p + tokens - 1, window, last, &lo_last);
    attn_rows(p, window, last, lo);
    return int(lo_last - *lo) + n;
}

// The scores of the key k with N queries that lie one after the other from q, for the warp of attn_score_kernel.
// score: where the first of them has its score of the key's row; the others lie n_keys floats apart.
template <int N>
__device__ __forceinline__ void attn_scores(const float* __restrict__ q, const half* __restrict__ k,
                                            float* __restrict__ score, int n_keys, int dim, float scale) {
    static_assert(SCORE_VALUES == 4, "a step is four halves of the key in one read and a float4 of a query");
    const int lane = threadIdx.x % WARP;
    const float* qi[N];
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        qi[i] = q + size_t(i) * dim;
        acc[i] = 0.0f;
    }
    for (int d = SCORE_VALUES * lane; d < dim; d += SCORE_VALUES * WARP) {
        const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(k + d);
        float x[SCORE_VALUES];
#pragma unroll
        for (int j = 0; j < SCORE_VALUES; ++j)
            x[j] = __half2float(__ushort_as_half((unsigned short) (bits >> 16 * j)));
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const float4 y = *reinterpret_cast<const float4*>(qi[i] + d);
            acc[i] += y.x * x[0];
            acc[i] += y.y * x[1];
            acc[i] += y.z * x[2];
            acc[i] += y.w * x[3];
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
        for (int off = WARP / 2; off > 0; off >>= 1) acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off);
        if (lane == 0) score[size_t(i) * n_keys] = acc[i] * scale;
    }
}

// attn_scores() for n queries: N at a time, then what is left by halves of N.
template <int N>
__device__ __forceinline__ void attn_scores_by(int n, const float* __restrict__ q, const half* __restrict__ k,
                                               float* __restrict__ score, int n_keys, int dim, float scale) {
    for (; n >= N; n -= N, q += size_t(N) * dim, score += size_t(N) * n_keys)
        attn_scores<N>(q, k, score, n_keys, dim, scale);
    if constexpr (N > 1) attn_scores_by<N / 2>(n, q, k, score, n_keys, dim, scale);
}

// score[token][head][i] = scale * (q[token][head] . the key of the head's group in the i-th row of the launch's
// span).  A warp has a row and one of the row's groups of heads.  Its threads read the key side by side,
// SCORE_VALUES values each a step, so that a step reads one place of the cache, and the launch's tokens read it
// once between them: the cache of a long context is what the kernel waits for.  A thread sums its products with
// a query, for up to SCORE_QUERIES heads of a token at a time, and a query's 32 sums are added in pairs.  A
// token gets scores for the rows of the span it does not attend as well, which the softmax leaves out.
__global__ void attn_score_kernel(const float* __restrict__ q, const half* __restrict__ kcache,
                                  float* __restrict__ score, int n_keys, int dim, int n_head, int kv_heads,
                                  long long pos, int first, int tokens, float scale, int window, int ring,
                                  long long last, const long long* __restrict__ at) {
    const int unit = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;   // a row and a group
    const int key = unit / kv_heads, g = unit % kv_heads, per = n_head / kv_heads;
    long long lo;
    if (key >= attn_span((at ? *at : pos) + first, tokens, window, last, &lo)) return;
    const long long row = ring > 0 ? (lo + key) % ring : lo + key;
    const half* k = kcache + (size_t(row) * kv_heads + g) * dim;
    q += (size_t(first) * n_head + g * per) * dim;   // the group's heads of the launch's first token
    score += size_t(g) * per * n_keys + key;         // and their scores of the row
    for (int t = 0; t < tokens; ++t, q += size_t(n_head) * dim, score += size_t(n_head) * n_keys)
        attn_scores_by<SCORE_QUERIES>(per, q, k, score, n_keys, dim, scale);
}

// The sums of N scores a lane holds parts of, added over the warp in the tree attn_scores() adds one score in: at
// each of its steps a lane adds the part a lane `off` away has.  While a lane holds more than one score it keeps
// half of them and gives the other half away, so one exchange serves two scores; then the four lanes of a score
// finish it.  A sum is the other tree's to the bit, since a + b is b + a.  Returns the lane's score, `which` of the N.
template <int N>
__device__ __forceinline__ float scores_sum(float* v, int lane, int* which) {
    int idx = 0;
#pragma unroll
    for (int off = WARP / 2, m = N; off > 0; off >>= 1) {
        if (m > 1) {
            const int half = m / 2;
            const bool upper = lane & off;
#pragma unroll
            for (int i = 0; i < half; ++i) {
                const float send = upper ? v[i] : v[i + half], keep = upper ? v[i + half] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
            }
            idx = 2 * idx + (upper ? 1 : 0);
            m = half;
        } else {
            v[0] += __shfl_xor_sync(0xffffffffu, v[0], off);
        }
    }
    *which = idx;
    return v[0];
}

// attn_score_kernel for heads of 128 * STEPS values, N of a group's heads at a time: the same products in the same
// order and the same sums, but a warp has SCORE_KEYS rows of a group, and a token's queries stay in its registers
// while it goes through them.  The other kernel read them again for every row, and passed every score's parts from lane to
// lane on its own: those reads and moves were what it waited for.
template <int N, int STEPS>
__global__ void __launch_bounds__(SCORE_WARPS * WARP)
attn_score_reg_kernel(const float* __restrict__ q, const half* __restrict__ kcache, float* __restrict__ score,
                      int n_keys, int n_head, int kv_heads, long long pos, int first, int tokens, float scale,
                      int window, int ring, long long last, const long long* __restrict__ at) {
    static_assert(SCORE_VALUES == 4, "a step is four halves of the key in one read and a float4 of a query");
    constexpr int dim = SCORE_VALUES * WARP * STEPS;
    const int unit = blockIdx.x * SCORE_WARPS + threadIdx.x / WARP;   // a group's block of rows
    const int g = unit % kv_heads, k0 = unit / kv_heads * SCORE_KEYS, lane = threadIdx.x % WARP;
    long long lo;
    const int span = attn_span((at ? *at : pos) + first, tokens, window, last, &lo);
    if (k0 >= span) return;
    const int rows = min(SCORE_KEYS, span - k0), per = n_head / kv_heads;
    for (int th = 0; th < tokens * per; th += N) {   // N heads of a token
        const int t = th / per, h = g * per + th % per;
        const float* qt = q + ((size_t(first) + t) * n_head + h) * dim + SCORE_VALUES * lane;
        float4 y[N][STEPS];
#pragma unroll
        for (int i = 0; i < N; ++i)
#pragma unroll
            for (int s = 0; s < STEPS; ++s) y[i][s] = *reinterpret_cast<const float4*>(qt + i * dim + 128 * s);
        float* st = score + (size_t(t) * n_head + h) * n_keys + k0;
        long long row = ring > 0 ? (lo + k0) % ring : lo + k0;
        for (int r = 0; r < rows; ++r) {
            const half* k = kcache + (size_t(row) * kv_heads + g) * dim + SCORE_VALUES * lane;
            float acc[N];
#pragma unroll
            for (int i = 0; i < N; ++i) acc[i] = 0.0f;
#pragma unroll
            for (int s = 0; s < STEPS; ++s) {
                const unsigned long long bits = *reinterpret_cast<const unsigned long long*>(k + 128 * s);
                float x[SCORE_VALUES];
#pragma unroll
                for (int j = 0; j < SCORE_VALUES; ++j)
                    x[j] = __half2float(__ushort_as_half((unsigned short) (bits >> 16 * j)));
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    acc[i] += y[i][s].x * x[0];
                    acc[i] += y[i][s].y * x[1];
                    acc[i] += y[i][s].z * x[2];
                    acc[i] += y[i][s].w * x[3];
                }
            }
            int which;
            const float sum = scores_sum<N>(acc, lane, &which);
            if ((lane & (WARP / N - 1)) == 0) st[size_t(which) * n_keys + r] = sum * scale;
            if (++row == ring) row = 0;
        }
    }
}

// Softmax in place over a head's scores of the rows its token attends; the other rows of the launch's span get
// zero, which adds nothing to the token's values.  One block per head and token.  With `wt` the weights go there
// instead, in the order attn_value_t_kernel reads them: for each of the span's rows, the VALUE_QUERIES queries of
// a block of the group's queries side by side (per: the group's heads).
__global__ void attn_softmax_kernel(float* score, int n_keys, long long pos, int first, int tokens, int window,
                                    long long last, const long long* __restrict__ at, float* __restrict__ wt,
                                    int per) {
    __shared__ float part[8];
    const long long p = (at ? *at : pos) + first;
    long long lo, lo0;
    const int span = attn_span(p, tokens, window, last, &lo0);
    const int n = attn_rows(p + blockIdx.y, window, last, &lo), own = int(lo - lo0);
    float* all = score + (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * n_keys;
    float* row = all + own;   // the token's rows of the span
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    const float inv = 1.0f / block_sum(sum, part);
    if (!wt) {
        for (int j = threadIdx.x; j < n; j += blockDim.x) row[j] *= inv;
        for (int j = threadIdx.x; j < own; j += blockDim.x) all[j] = 0.0f;
        for (int j = own + n + threadIdx.x; j < span; j += blockDim.x) all[j] = 0.0f;
        return;
    }
    const int h = blockIdx.x, query = blockIdx.y * per + h % per;   // the query among its group's
    const int blocks = (tokens * per + VALUE_QUERIES - 1) / VALUE_QUERIES;
    float* o = wt + (size_t(h / per) * blocks + query / VALUE_QUERIES) * n_keys * VALUE_QUERIES + query % VALUE_QUERIES;
    for (int j = threadIdx.x; j < own; j += blockDim.x) o[size_t(j) * VALUE_QUERIES] = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) o[size_t(own + j) * VALUE_QUERIES] = row[j] * inv;
    for (int j = own + n + threadIdx.x; j < span; j += blockDim.x) o[size_t(j) * VALUE_QUERIES] = 0.0f;
}

// A thread's sums of attn_value_kernel for N queries: over its share of the span's rows from `row` of the cache
// on, of a row's value with each query's weight there.  w: the weights of the lane's query; the warp's first N
// lanes have the N queries between them.
template <int N>
__device__ __forceinline__ void attn_values(const float* __restrict__ w, const half* __restrict__ v, size_t width,
                                            long long row, int ring, int span, float* sums) {
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    v += size_t(row) * width;
    for (int j = threadIdx.x / VALUE_DIMS; j < span; j += VALUE_SHARES) {
        const float x = __half2float(*v), weight = w[j];
#pragma unroll
        for (int i = 0; i < N; ++i) acc[i] += __shfl_sync(0xffffffffu, weight, i) * x;
        v += VALUE_SHARES * width;
        row += VALUE_SHARES;
        if (ring > 0 && row >= ring) row -= ring, v -= size_t(ring) * width;
    }
#pragma unroll
    for (int i = 0; i < N; ++i) sums[i] = acc[i];
}

// attn_values() for n queries, with N if no smaller power of two holds them.
template <int N>
__device__ __forceinline__ void attn_values_for(int n, const float* __restrict__ w, const half* __restrict__ v,
                                                size_t width, long long row, int ring, int span, float* sums) {
    if constexpr (N > 1)
        if (n <= N / 2) return attn_values_for<N / 2>(n, w, v, width, row, ring, span, sums);
    attn_values<N>(w, v, width, row, ring, span, sums);
}

// out[token][head][d] = sum over the rows of score[token][head][row] * the value of the head's group in that
// row.  A block has a group of heads, VALUE_DIMS value dimensions (blockIdx.x) and up to VALUE_QUERIES of the
// group's queries (blockIdx.y): the group's heads of the launch's first token, then of its second, and so on.
// A thread has one of the dimensions and one of VALUE_SHARES shares of the span's rows, every VALUE_SHARES-th
// row from its own, so that many reads of the cache are on their way at a time.  It reads a row's value once
// for the block's queries; their weights of the row come in one read as well, a lane a query, and the warp
// passes them around.  A token's shares are counted from its own first row and added in their order, so that it
// comes out the same whatever tokens the launch has beside it.
__global__ void attn_value_kernel(const float* __restrict__ score, const half* __restrict__ vcache,
                                  float* __restrict__ out, int n_keys, int dim, int n_head, int kv_heads,
                                  long long pos, int first, int tokens, int window, int ring, long long last,
                                  const long long* __restrict__ at) {
    static_assert(VALUE_DIMS == WARP && VALUE_QUERIES <= WARP, "a warp is a share and reads a query's weight a lane");
    __shared__ float share[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS];
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int g = blockIdx.x / dims, q0 = blockIdx.y * VALUE_QUERIES;
    const int live = min(VALUE_QUERIES, tokens * per - q0);   // the block's queries
    const int lane = threadIdx.x % VALUE_DIMS, s = threadIdx.x / VALUE_DIMS;
    const int d = (blockIdx.x % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    // the lane's query, and for a lane past the head's last value a value to read all the same
    const int mine = q0 + min(lane % VALUE_QUERIES, live - 1);
    float sums[VALUE_QUERIES];
    attn_values_for<VALUE_QUERIES>(live, score + (size_t(mine / per) * n_head + g * per + mine % per) * n_keys,
                                  vcache + size_t(g) * dim + min(d, dim - 1), size_t(kv_heads) * dim,
                                  ring > 0 ? (lo + s) % ring : lo + s, ring, span, sums);
    for (int i = 0; i < live; ++i) {
        long long own;   // the first row of the query's token: the thread's rows are this share of the token's
        attn_rows(p + (q0 + i) / per, window, last, &own);
        share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = sums[i];
    }
    __syncthreads();
    if (s != 0 || d >= dim) return;
    for (int i = 0; i < live; ++i) {
        float sum = share[i][0][lane];
        for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
        const int query = q0 + i;
        out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
    }
}

// attn_value_kernel with the weights as attn_softmax_kernel leaves them with `wt`: a row's queries side by side,
// so a warp reads a row's weights in two loads that every lane makes alike, where the other passes them from lane
// to lane, and so waited on the passing (Nsight Compute).  A block has TILES of the blocks of VALUE_QUERIES
// queries, which read the cache's values once between them.  The products go in the rows' order: the sums are
// the other kernel's to the bit.
template <int TILES>
__global__ void __launch_bounds__(VALUE_DIMS * VALUE_SHARES)
attn_value_t_kernel(const float* __restrict__ wt, const half* __restrict__ vcache, float* __restrict__ out, int n_keys,
                    int dim, int n_head, int kv_heads, long long pos, int first, int tokens, int window, int ring,
                    long long last, const long long* __restrict__ at) {
    static_assert(VALUE_DIMS == WARP && VALUE_QUERIES == 8, "a warp is a share, and a row's weights are two float4");
    __shared__ float share[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS];
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int blocks = (tokens * per + VALUE_QUERIES - 1) / VALUE_QUERIES;   // the group's blocks of queries
    const int g = blockIdx.x / dims, b0 = blockIdx.y * TILES;
    const int lane = threadIdx.x % VALUE_DIMS, s = threadIdx.x / VALUE_DIMS;
    const int d = (blockIdx.x % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    const size_t width = size_t(kv_heads) * dim;
    const half* v = vcache + size_t(g) * dim + min(d, dim - 1);   // a lane past the last value reads one all the same
    const float* w[TILES];   // a tile past the group's last reads the block's first again, for nothing
#pragma unroll
    for (int t = 0; t < TILES; ++t)
        w[t] = wt + (size_t(g) * blocks + (b0 + t < blocks ? b0 + t : b0)) * n_keys * VALUE_QUERIES;
    float acc[TILES][VALUE_QUERIES];
#pragma unroll
    for (int t = 0; t < TILES; ++t)
#pragma unroll
        for (int i = 0; i < VALUE_QUERIES; ++i) acc[t][i] = 0.0f;
    long long row = ring > 0 ? (lo + s) % ring : lo + s;
    for (int j = s; j < span; j += VALUE_SHARES) {
        const float x = __half2float(v[size_t(row) * width]);
#pragma unroll
        for (int t = 0; t < TILES; ++t) {
            const float4 a = *reinterpret_cast<const float4*>(w[t] + size_t(j) * VALUE_QUERIES);
            const float4 b = *reinterpret_cast<const float4*>(w[t] + size_t(j) * VALUE_QUERIES + 4);
            acc[t][0] += a.x * x;
            acc[t][1] += a.y * x;
            acc[t][2] += a.z * x;
            acc[t][3] += a.w * x;
            acc[t][4] += b.x * x;
            acc[t][5] += b.y * x;
            acc[t][6] += b.z * x;
            acc[t][7] += b.w * x;
        }
        row += VALUE_SHARES;
        if (ring > 0 && row >= ring) row -= ring;
    }
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
        const int q0 = (b0 + t) * VALUE_QUERIES, live = min(VALUE_QUERIES, tokens * per - q0);
        if (live <= 0) break;
        if (t > 0) __syncthreads();   // the tile before has its sums out of share
        for (int i = 0; i < live; ++i) {
            long long own;   // the first row of the query's token: the thread's rows are this share of the token's
            attn_rows(p + (q0 + i) / per, window, last, &own);
            share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = acc[t][i];
        }
        __syncthreads();
        if (s == 0 && d < dim)
            for (int i = 0; i < live; ++i) {
                float sum = share[i][0][lane];
                for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
                const int query = q0 + i;
                out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
            }
    }
}

template <int N>
__device__ __forceinline__ void attn_scores_int8(const float* __restrict__ q,
                                                 const int8_t* __restrict__ k_code,
                                                 const uint16_t* __restrict__ k_scale,
                                                 float* __restrict__ score, int n_keys, int dim, float scale) {
    static_assert(SCORE_VALUES == 4, "a step is four int8 codes of the key and a float4 of a query");
    const int lane = threadIdx.x % WARP;
    const float* qi[N];
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        qi[i] = q + size_t(i) * dim;
        acc[i] = 0.0f;
    }
    for (int d = SCORE_VALUES * lane; d < dim; d += SCORE_VALUES * WARP) {
        const char4 c = *reinterpret_cast<const char4*>(k_code + d);
        const float sf = f32_from_f16(k_scale[d / 64]);
        const float x0 = (float) c.x * sf;
        const float x1 = (float) c.y * sf;
        const float x2 = (float) c.z * sf;
        const float x3 = (float) c.w * sf;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const float4 y = *reinterpret_cast<const float4*>(qi[i] + d);
            acc[i] += y.x * x0 + y.y * x1 + y.z * x2 + y.w * x3;
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
        for (int off = WARP / 2; off > 0; off >>= 1) acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off);
        if (lane == 0) score[size_t(i) * n_keys] = acc[i] * scale;
    }
}

template <int N>
__device__ __forceinline__ void attn_scores_int8_by(int n, const float* __restrict__ q,
                                                    const int8_t* __restrict__ k_code,
                                                    const uint16_t* __restrict__ k_scale,
                                                    float* __restrict__ score, int n_keys, int dim, float scale) {
    for (; n >= N; n -= N, q += size_t(N) * dim, score += size_t(N) * n_keys)
        attn_scores_int8<N>(q, k_code, k_scale, score, n_keys, dim, scale);
    if constexpr (N > 1) attn_scores_int8_by<N / 2>(n, q, k_code, k_scale, score, n_keys, dim, scale);
}

__global__ void attn_score_int8_kernel(const float* __restrict__ q, const int8_t* __restrict__ kcache_q,
                                       const uint16_t* __restrict__ kcache_scale,
                                       float* __restrict__ score, int n_keys, int dim, int n_head, int kv_heads,
                                       long long pos, int first, int tokens, float scale, int window, int ring,
                                       long long last, const long long* __restrict__ at) {
    const int unit = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    const int key = unit / kv_heads, g = unit % kv_heads, per = n_head / kv_heads;
    long long lo;
    if (key >= attn_span((at ? *at : pos) + first, tokens, window, last, &lo)) return;
    const long long row = ring > 0 ? (lo + key) % ring : lo + key;
    const int8_t* k_code = kcache_q + (size_t(row) * kv_heads + g) * dim;
    const uint16_t* k_scale = kcache_scale + (size_t(row) * kv_heads + g) * (dim / 64);
    q += (size_t(first) * n_head + g * per) * dim;
    score += size_t(g) * per * n_keys + key;
    for (int t = 0; t < tokens; ++t, q += size_t(n_head) * dim, score += size_t(n_head) * n_keys)
        attn_scores_int8_by<SCORE_QUERIES>(per, q, k_code, k_scale, score, n_keys, dim, scale);
}

template <int N>
__device__ __forceinline__ void attn_values_int8(const float* __restrict__ w, const int8_t* __restrict__ v_code,
                                                 const uint16_t* __restrict__ v_scale, size_t width_q,
                                                 size_t width_s, long long row, int ring, int span, float* sums) {
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    v_code += size_t(row) * width_q;
    v_scale += size_t(row) * width_s;
    for (int j = threadIdx.x / VALUE_DIMS; j < span; j += VALUE_SHARES) {
        const float sf = f32_from_f16(*v_scale);
        const float x = (float)(*v_code) * sf, weight = w[j];
#pragma unroll
        for (int i = 0; i < N; ++i) acc[i] += __shfl_sync(0xffffffffu, weight, i) * x;
        v_code += VALUE_SHARES * width_q;
        v_scale += VALUE_SHARES * width_s;
        row += VALUE_SHARES;
        if (ring > 0 && row >= ring) {
            row -= ring;
            v_code -= size_t(ring) * width_q;
            v_scale -= size_t(ring) * width_s;
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) sums[i] = acc[i];
}

template <int N>
__device__ __forceinline__ void attn_values_int8_for(int n, const float* __restrict__ w,
                                                     const int8_t* __restrict__ v_code,
                                                     const uint16_t* __restrict__ v_scale,
                                                     size_t width_q, size_t width_s,
                                                     long long row, int ring, int span, float* sums) {
    if constexpr (N > 1)
        if (n <= N / 2) return attn_values_int8_for<N / 2>(n, w, v_code, v_scale, width_q, width_s, row, ring, span, sums);
    attn_values_int8<N>(w, v_code, v_scale, width_q, width_s, row, ring, span, sums);
}

__global__ void attn_value_int8_kernel(const float* __restrict__ score, const int8_t* __restrict__ vcache_q,
                                       const uint16_t* __restrict__ vcache_scale,
                                       float* __restrict__ out, int n_keys, int dim, int n_head, int kv_heads,
                                       long long pos, int first, int tokens, int window, int ring, long long last,
                                       const long long* __restrict__ at) {
    static_assert(VALUE_DIMS == WARP && VALUE_QUERIES <= WARP, "a warp is a share and reads a query's weight a lane");
    __shared__ float share[VALUE_QUERIES][VALUE_SHARES][VALUE_DIMS];
    const int per = n_head / kv_heads, dims = (dim + VALUE_DIMS - 1) / VALUE_DIMS;
    const int g = blockIdx.x / dims, q0 = blockIdx.y * VALUE_QUERIES;
    const int live = min(VALUE_QUERIES, tokens * per - q0);
    const int lane = threadIdx.x % VALUE_DIMS, s = threadIdx.x / VALUE_DIMS;
    const int d = (blockIdx.x % dims) * VALUE_DIMS + lane;
    const long long p = (at ? *at : pos) + first;
    long long lo;
    const int span = attn_span(p, tokens, window, last, &lo);
    const int mine = q0 + min(lane % VALUE_QUERIES, live - 1);
    float sums[VALUE_QUERIES];
    const size_t width_q = size_t(kv_heads) * dim;
    const size_t width_s = size_t(kv_heads) * (dim / 64);
    const int d_clamped = min(d, dim - 1);
    const int g_group = d_clamped / 64;
    attn_values_int8_for<VALUE_QUERIES>(live, score + (size_t(mine / per) * n_head + g * per + mine % per) * n_keys,
                                        vcache_q + size_t(g) * dim + d_clamped,
                                        vcache_scale + size_t(g) * (dim / 64) + g_group,
                                        width_q, width_s,
                                        ring > 0 ? (lo + s) % ring : lo + s, ring, span, sums);
    for (int i = 0; i < live; ++i) {
        long long own;
        attn_rows(p + (q0 + i) / per, window, last, &own);
        share[i][(s + VALUE_SHARES - int((own - lo) % VALUE_SHARES)) % VALUE_SHARES][lane] = sums[i];
    }
    __syncthreads();
    if (s != 0 || d >= dim) return;
    for (int i = 0; i < live; ++i) {
        float sum = share[i][0][lane];
        for (int k = 1; k < VALUE_SHARES; ++k) sum += share[i][k][lane];
        const int query = q0 + i;
        out[((size_t(first) + query / per) * n_head + g * per + query % per) * dim + d] = sum;
    }
}

__global__ void dequant_q8_to_f16_kernel(const int8_t* __restrict__ src_q,
                                         const uint16_t* __restrict__ src_scale,
                                         half* __restrict__ dst,
                                         int m, int kv_heads, int head_dim,
                                         long long row0, int ring) {
    const int per = head_dim / 4;
    const int total = m * kv_heads * per;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int r = i / (kv_heads * per);
    const int rem = i % (kv_heads * per);
    const int h = rem / per;
    const int q4 = rem % per;
    const int d = q4 * 4;
    const int G = head_dim / 64;
    const long long row = ring > 0 ? (row0 + r) % ring : row0 + r;
    const size_t q_idx = (size_t(row) * kv_heads + h) * head_dim + d;
    const size_t scale_idx = (size_t(row) * kv_heads + h) * G + d / 64;
    const float sf = f32_from_f16(src_scale[scale_idx]);
    const char4 c = *reinterpret_cast<const char4*>(src_q + q_idx);
    ushort4 out;
    out.x = f16_from_f32((float) c.x * sf);
    out.y = f16_from_f32((float) c.y * sf);
    out.z = f16_from_f32((float) c.z * sf);
    out.w = f16_from_f32((float) c.w * sf);
    const size_t dst_idx = (size_t(r) * kv_heads + h) * head_dim + d;
    *reinterpret_cast<ushort4*>(dst + dst_idx) = out;
}
// A chunk's queries and its results lie in two orders: the engine's (token, head) and the one a product per
// group of heads takes (group, token, head of the group).  This is where the engine's `row` lies in the second.
__device__ __forceinline__ size_t attn_by_group(int row, int n_head, int kv_heads, int tokens) {
    const int per = n_head / kv_heads, t = row / n_head, h = row % n_head;
    return (size_t(h / per) * tokens + t) * per + h % per;
}

// The queries to the order of the groups, as the halves the products take.  blockIdx.y is the token and the head.
__global__ void attn_group_kernel(const float* __restrict__ src, half* __restrict__ dst, int dim, int n_head,
                                  int kv_heads, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const float x = src[size_t(blockIdx.y) * dim + i];
    dst[attn_by_group(blockIdx.y, n_head, kv_heads, tokens) * dim + i] = __float2half(x);
}

// The results back to the engine's order.
__global__ void attn_ungroup_kernel(const float* __restrict__ src, float* __restrict__ dst, int dim, int n_head,
                                    int kv_heads, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    dst[size_t(blockIdx.y) * dim + i] = src[attn_by_group(blockIdx.y, n_head, kv_heads, tokens) * dim + i];
}

// A softmax over a chunk's rows a block of rows at a time, for scores in the order of the groups: blockIdx.x
// counts (group, token, head of the group), and the block's rows start at row0 of the sequence.  A score's weight
// is its exp against the largest its token and head have had (top), zero for a row the token does not attend, as
// the half the values' product takes.  sum is the sum of the weights against that top, and shrink what the
// weights of the blocks before are worth against it.  With `fresh` there was no block before.
__global__ void attn_softmax_block_kernel(const float* __restrict__ score, half* __restrict__ weight, int n_keys,
                                          long long row0, long long pos, int first, int window, int tokens, int per,
                                          long long last, float* top, float* sum, float* shrink, bool fresh) {
    __shared__ float part[8];
    const int tok = first + (blockIdx.x % (tokens * per)) / per;
    long long lo;
    const int n = attn_rows(pos + tok, window, last, &lo);
    const long long end = lo + n;   // the token attends the sequence's rows from lo to before end
    const int j0 = lo > row0 ? int(lo - row0) : 0, j1 = end - row0 < n_keys ? int(end - row0) : n_keys;
    const float* row = score + size_t(blockIdx.x) * n_keys;
    half* w = weight + size_t(blockIdx.x) * n_keys;
    float mx = -INFINITY;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, part);
    const float before = fresh ? -INFINITY : top[blockIdx.x], now = fmaxf(before, mx);
    float s = 0.0f;
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) {
        w[j] = __float2half(j >= j0 && j < j1 ? expf(row[j] - now) : 0.0f);
        s += __half2float(w[j]);   // the sum is of the weights as the values get them
    }
    s = block_sum(s, part);
    if (threadIdx.x != 0) return;
    const float f = mx == -INFINITY ? 1.0f : expf(before - now);   // none of the block's rows: nothing changes
    top[blockIdx.x] = now;
    sum[blockIdx.x] = (fresh ? 0.0f : sum[blockIdx.x] * f) + s;
    shrink[blockIdx.x] = f;
}

// A chunk's results in the order of the groups, `dim` values a token and head: times its factor, or divided by it.
__global__ void attn_scale_kernel(float* out, const float* __restrict__ factor, int dim, bool divide) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const float f = factor[blockIdx.y];
    // a row whose largest score this block of keys did not raise is times 1: it is not read or written
    if (!divide && f == 1.0f) return;
    out[size_t(blockIdx.y) * dim + i] *= divide ? 1.0f / f : f;
}

__global__ void softcap_kernel(float* x, size_t n, float cap) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] * (1.0f / cap)) * cap;
}

// One block of 1024; a thread has every 1024th value.
__global__ void likeliest_kernel(const float* __restrict__ x, int n, Likeliest* __restrict__ out) {
    __shared__ float part[32];
    __shared__ int first[32];
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
    m = block_max(m, part);
    int at = n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] == m) at = min(at, i);
        if (x[i] > m - 20.0f) sum += expf(x[i] - m);
    }
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) at = min(at, __shfl_xor_sync(0xffffffff, at, o, WARP));
    if ((threadIdx.x & 31) == 0) first[threadIdx.x >> 5] = at;
    sum = block_sum(sum, part);   // and first is written
    if (threadIdx.x != 0) return;
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) at = min(at, first[k]);
    out->token = at < n ? at : 0;
    out->prob = 1.0f / sum;
}

__global__ void mul_sigmoid_kernel(float* __restrict__ acc, const float* __restrict__ g, int n, int g_stride,
                                   bool one_gate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[size_t(blockIdx.y) * n + i] *= sigmoid(g[size_t(blockIdx.y) * g_stride + (one_gate ? 0 : i)]);
}

// Where a token leaves the recurrent state as it is after it: `snaps` has a bit for each of a call's tokens that
// does, and their states lie in the tokens' order, `stride` floats apart.  Null for a token that leaves none.
__device__ __forceinline__ float* snap_of(float* snap, size_t stride, unsigned snaps, int row) {
    return row < 32 && (snaps >> row & 1u) ? snap + size_t(__popc(snaps & ((1u << row) - 1u))) * stride : nullptr;
}

// Three past inputs per channel, then the new one.  A thread is a channel and walks the tokens.  The state as it
// is after a token of `snaps` goes to snap (snap_of()).
__global__ void conv4_silu_kernel(const float* __restrict__ x, float* __restrict__ state,
                                  const float* __restrict__ w, float* __restrict__ out, int channels, int rows,
                                  float* __restrict__ snap, size_t stride, unsigned snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = w + 4 * size_t(c);
    float* st = state + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * channels + c;
        const float v = x[at];
        const float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3];
        s0 = s1;
        s1 = s2;
        s2 = v;
        out[at] = y * sigmoid(y);
        if (float* m = snap_of(snap, stride, snaps, row)) {
            m += size_t(c) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

// The gated delta net's step, in four launches.  A token's queries and keys are normed without the state, its
// decay and its share of the error are a head's alone, and its output is normed without the tokens after it: all
// of that is done for every token at once, and only the state's update walks the tokens.
//
// One block a token and key head: the head's q and k, normed in place.
__global__ void gdn_norm_kernel(float* __restrict__ qkv, int dim, int k_heads, size_t row_n, float norm_eps) {
    __shared__ float part[8];
    const int hk = blockIdx.x % k_heads, j = threadIdx.x;
    float* x = qkv + size_t(blockIdx.x / k_heads) * row_n;
    float* qp = x + size_t(hk) * dim + j;
    float* kp = x + size_t(dim) * k_heads + size_t(hk) * dim + j;
    const float q = *qp, k = *kp;
    const float qs = block_sum(q * q, part);
    const float ks = block_sum(k * k, part);
    *qp = q / sqrtf(qs + norm_eps) * (1.0f / sqrtf(float(dim)));
    *kp = k / sqrtf(ks + norm_eps);
}

// A token's decay and its share of the error for each head, in place of alpha and beta.
__global__ void gdn_gates_kernel(float* __restrict__ alpha, float* __restrict__ beta, const float* __restrict__ dt,
                                 const float* __restrict__ a, int n_head, int n) {
    const int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n) return;
    const int h = at % n_head;
    const float g = alpha[at] + dt[h];
    alpha[at] = expf((g > 20.0f ? g : logf(1.0f + expf(g))) * a[h]);
    beta[at] = sigmoid(beta[at]);
}

// The state's update, a warp a column: the blocks of a head share its columns among their warps, and a lane holds
// every 32nd value of its warp's column, so a column is read and written as whole lines and its two sums are a
// few terms a lane and a sum over the warp.  The block walks the tokens, and no warp waits for another.
// s[i + dim * j] holds key dimension i of column j.  out gets the output before its norm.  decay and share are
// what gdn_gates_kernel left.  dim is a multiple of 32.
__global__ void gdn_state_kernel(const float* __restrict__ qkv, const float* __restrict__ decay,
                                 const float* __restrict__ share, float* __restrict__ s, float* __restrict__ out,
                                 int dim, int n_head, int k_heads, int rows, float* __restrict__ snap, size_t stride,
                                 unsigned snaps) {
    const int per_head = dim / int(blockDim.x / WARP);   // blocks a head
    const int h = blockIdx.x / per_head, hk = h % k_heads;
    const int j = (blockIdx.x % per_head) * int(blockDim.x / WARP) + int(threadIdx.x / WARP);
    const int lane = threadIdx.x & (WARP - 1);
    if (j >= dim) return;   // a whole warp: the last block of a head whose columns do not fill it
    const size_t keys = size_t(dim) * k_heads, d_inner = size_t(dim) * n_head, row_n = 2 * keys + d_inner;
    const size_t mine = (size_t(h) * dim + j) * dim;
    float* col = s + mine;
    for (int row = 0; row < rows; ++row) {
        const float* x = qkv + size_t(row) * row_n;
        const float* qn = x + size_t(hk) * dim;
        const float* kn = x + keys + size_t(hk) * dim;
        const float v = x[2 * keys + size_t(h) * dim + j];
        const float dc = decay[size_t(row) * n_head + h];
        float sk = 0.0f;
        for (int i = lane; i < dim; i += WARP) sk += col[i] * dc * kn[i];
        sk = warp_sum(sk);
        const float d = (v - sk) * share[size_t(row) * n_head + h];
        float o = 0.0f;
        float* m = snap_of(snap, stride, snaps, row);
        if (m) m += mine;
        for (int i = lane; i < dim; i += WARP) {
            const float n = col[i] * dc + kn[i] * d;
            col[i] = n;
            if (m) m[i] = n;
            o += n * qn[i];
        }
        o = warp_sum(o);
        if (lane == 0) out[size_t(row) * d_inner + size_t(h) * dim + j] = o;
    }
}

// gdn_state_kernel with the warp's column in registers, PER values a lane, for the whole walk over the tokens:
// a token reads and writes no state.  The sums go in the same order, so the result is the same to the bit.
template <int PER>
__global__ void gdn_state_reg_kernel(const float* __restrict__ qkv, const float* __restrict__ decay,
                                     const float* __restrict__ share, float* __restrict__ s, float* __restrict__ out,
                                     int dim, int n_head, int k_heads, int rows, float* __restrict__ snap,
                                     size_t stride, unsigned snaps) {
    const int per_head = dim / int(blockDim.x / WARP);
    const int h = blockIdx.x / per_head, hk = h % k_heads;
    const int j = (blockIdx.x % per_head) * int(blockDim.x / WARP) + int(threadIdx.x / WARP);
    const int lane = threadIdx.x & (WARP - 1);
    if (j >= dim) return;
    const size_t keys = size_t(dim) * k_heads, d_inner = size_t(dim) * n_head, row_n = 2 * keys + d_inner;
    const size_t mine = (size_t(h) * dim + j) * dim;
    float* col = s + mine;
    float c[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) c[k] = col[lane + k * WARP];
    for (int row = 0; row < rows; ++row) {
        const float* x = qkv + size_t(row) * row_n;
        const float* qn = x + size_t(hk) * dim;
        const float* kn = x + keys + size_t(hk) * dim;
        const float v = x[2 * keys + size_t(h) * dim + j];
        const float dc = decay[size_t(row) * n_head + h];
        // the products and sums as gdn_state_kernel's compiled code has them, so that the result is the same bits
        float sk = 0.0f;
#pragma unroll
        for (int k = 0; k < PER; ++k) sk = __fmaf_rn(__fmul_rn(dc, c[k]), kn[lane + k * WARP], sk);
        sk = warp_sum(sk);
        const float d = __fmul_rn(__fsub_rn(v, sk), share[size_t(row) * n_head + h]);
        float o = 0.0f;
        float* m = snap_of(snap, stride, snaps, row);
        if (m) m += mine;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int i = lane + k * WARP;
            const float n = __fmaf_rn(dc, c[k], __fmul_rn(kn[i], d));
            c[k] = n;
            if (m) m[i] = n;
            o = __fmaf_rn(n, qn[i], o);
        }
        o = warp_sum(o);
        if (lane == 0) out[size_t(row) * d_inner + size_t(h) * dim + j] = o;
    }
#pragma unroll
    for (int k = 0; k < PER; ++k) col[lane + k * WARP] = c[k];
}

// One block a token and value head: the output's norm, its weights and the gate.
__global__ void gdn_out_kernel(float* __restrict__ out, const float* __restrict__ z,
                               const float* __restrict__ o_norm, int dim, float norm_eps) {
    __shared__ float part[8];
    const size_t at = size_t(blockIdx.x) * dim + threadIdx.x;
    const float o = out[at];
    const float os = block_sum(o * o, part);
    const float gate = z[at];
    out[at] = o / sqrtf(os / float(dim) + norm_eps) * o_norm[threadIdx.x] * (gate * sigmoid(gate));
}

inline cudaStream_t cs(void* stream) { return (cudaStream_t) stream; }
inline int grid(size_t n, int block) { return int((n + block - 1) / block); }

void blas_ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string((int) s));
}

}  // namespace

void* blas_create(void* stream) {
    cublasHandle_t h = nullptr;
    blas_ck(cublasCreate(&h), "cublasCreate");
    if (const cublasStatus_t s = cublasSetStream(h, cs(stream)); s != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(h);
        blas_ck(s, "cublasSetStream");
    }
    // cuBLAS takes 75 ms over a process's first product of halves, whatever its size: a prompt batch's attention
    // would wait for it.  A product that fails here fails again where its status is read.
    const int n = 16;
    void* p = nullptr;
    if (cudaMalloc(&p, n * n * (2 * sizeof(half) + sizeof(float))) != cudaSuccess) return h;
    cudaMemsetAsync(p, 0, n * n * (2 * sizeof(half) + sizeof(float)), cs(stream));
    const float one = 1.0f, zero = 0.0f;
    const half* a = (const half*) p;
    cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, a, CUDA_R_16F, n, n * n, a + n * n,
                               CUDA_R_16F, n, n * n, &zero, (float*) (a + 2 * n * n), CUDA_R_32F, n, n * n, 1,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaStreamSynchronize(cs(stream));
    cudaFree(p);
    return h;
}

void blas_destroy(void* blas) {
    if (blas) cublasDestroy((cublasHandle_t) blas);
}

void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, rows, stream);
}

void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream) {
    gemv(w, x, y, n_in, n_out, rows_per_x, rows, stream);
}

// A buffer of rows is its transpose to cuBLAS, so y[row] = w x[row] is Y = W^T X.  A group's rows of w, its part
// of every token's x and of every token's y are each a matrix with a stride.
void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows) {
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const int groups = n_out / rows_per_x;
    if (groups == 1) {
        blas_ck(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, n_out, rows, n_in, &one, w, n_in, x, n_in, &zero, y, n_out),
                "a batch's projection");
        return;
    }
    blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, rows_per_x, rows, n_in, &one, w, n_in,
                                      (long long) rows_per_x * n_in, x, groups * n_in, n_in, &zero, y, n_out,
                                      rows_per_x, groups),
            "a batch's projections");
}

void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream) {
    bf16_to_f32_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(w, out, n);
}

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int y_stride) {
    rms_norm_kernel<<<rows, sum_threads(n), 0, cs(stream)>>>(x, w, y, n, eps, y_stride ? y_stride : n);
}

void layer_norm(const float* x, const float* w, const float* b, float* y, int n, int rows, float eps, void* stream) {
    layer_norm_kernel<<<rows, 128, 0, cs(stream)>>>(x, w, b, y, n, eps);
}

void hc_pre(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* hcw,
            float* cur, int n_embd, int rows, float norm_eps, float hc_eps, int sinkhorn_iters, void* stream) {
    const int n_mix = HC * (2 + HC);
    hc_mix_kernel<<<dim3(n_mix + 1, rows), sum_threads(HC * n_embd), 0, cs(stream)>>>(r, fn, mix, HC * n_embd, n_mix);
    hc_read_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(r, mix, scale, base, hcw, cur, n_embd,
                                                                         norm_eps, hc_eps, sinkhorn_iters);
}

void hc_post(const float* x, const float* hcw, float* r, int n_embd, int rows, void* stream) {
    hc_post_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(x, hcw, r, n_embd);
}

void hc_mean(const float* r, float* y, int n_embd, int rows, void* stream) {
    hc_mean_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(r, y, n_embd);
}

void kda_conv(const float* proj, float* state, const float* wq, const float* wk, const float* wv, float* out,
              int d_inner, int rows, void* stream, float* snap, size_t stride, int n_snap) {
    kda_conv_kernel<<<grid(3 * d_inner, 256), 256, 0, cs(stream)>>>(proj, state, wq, wk, wv, out, d_inner, rows,
                                                                    snap, stride, snap ? n_snap : 0);
}

void kda_step(float* qkv, float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, int rows, float gate_lower_bound,
              float norm_eps, void* stream, float* snap, size_t stride, int n_snap) {
    if (head_dim % WARP != 0 || head_dim > 256)
        throw std::invalid_argument("kda_step: a head of 32 to 256 values, in 32s");
    int lanes = WARP;   // threads a column, while a block has them
    while (head_dim * lanes > 1024) lanes /= 2;
    // a few tokens, or a head that is not kda_state_kernel's size: one launch
    if (rows < 32 || snap || head_dim != KDA_TERMS * lanes) {
        kda_step_kernel<<<n_head, head_dim * lanes, 0, cs(stream)>>>(qkv, g1, a, beta, s, g2, o_norm, out, head_dim,
                                                                    n_head, rows, gate_lower_bound, norm_eps, snap,
                                                                    stride, snap ? n_snap : 0);
        return;
    }
    const int group = WARP / lanes;
    kda_gates_kernel<<<rows * n_head, head_dim, 0, cs(stream)>>>(qkv, g1, a, n_head, rows, gate_lower_bound, group);
    kda_state_kernel<<<n_head * (head_dim / KDA_COLUMNS), KDA_COLUMNS * lanes, 0, cs(stream)>>>(qkv, g1, beta, s, out,
                                                                                               n_head, rows);
    kda_out_kernel<<<rows * n_head, head_dim, 0, cs(stream)>>>(out, g2, o_norm, norm_eps, group);
}

void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream) {
    const size_t all = size_t(n) * rows;
    store_f16_kernel<<<grid(all, 256), 256, 0, cs(stream)>>>(x, (half*) cache + size_t(pos) * n, int(all));
}

void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream) {
    store_f16_at_kernel<<<grid(size_t(n) * rows, 256), 256, 0, cs(stream)>>>(x, (half*) cache, n, rows, at, ring);
}

void store_int8(const float* x, int8_t* cache_q, uint16_t* cache_scale, int kv_heads, int head_dim, int64_t pos,
                int rows, void* stream) {
    if (head_dim % 64 != 0) throw std::invalid_argument("store_int8: head_dim must be multiple of 64");
    const int G = head_dim / 64;
    const size_t n_groups = (size_t) rows * kv_heads * G;
    if (n_groups == 0) return;
    store_int8_kernel<<<n_groups, 64, 0, cs(stream)>>>(x, cache_q, cache_scale, kv_heads, head_dim, pos, n_groups);
}

void store_int8_at(const float* x, int8_t* cache_q, uint16_t* cache_scale, int kv_heads, int head_dim,
                   const long long* at, int rows, int ring, void* stream) {
    if (head_dim % 64 != 0) throw std::invalid_argument("store_int8_at: head_dim must be multiple of 64");
    const int G = head_dim / 64;
    const size_t n_groups = (size_t) rows * kv_heads * G;
    if (n_groups == 0) return;
    store_int8_at_kernel<<<n_groups, 64, 0, cs(stream)>>>(x, cache_q, cache_scale, kv_heads, head_dim, at, ring, n_groups);
}

void dsa_attend(void* blas, const float* q, const uint16_t* cache, const int32_t* sel, int sel_stride, int top_rows,
                int kpool, int64_t pos, int rows, int dense_rows, float* keys, float* score, int chunk, float* out,
                int rank, int n_head, float scale, void* stream) {
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const long long head = (long long) n_head * rank;
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        int n_keys = 0;
        for (int t = first; t < first + c; ++t)
            n_keys = std::max(n_keys, dsa_keys(t, (long long) pos, dense_rows, top_rows, kpool));
        const long long key_n = (long long) n_keys * rank, score_n = (long long) n_head * n_keys;
        dsa_keys_kernel<<<dim3(grid(n_keys, KEY_BLOCK), c), 256, 0, cs(stream)>>>(
            (const half*) cache, sel, sel_stride, top_rows, kpool, (long long) pos, first, dense_rows, keys, rank,
            n_keys);
        // score[head][key] = scale * (q[head] . keys[key]).  cuBLAS reads a buffer of rows as the transpose.
        blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, n_keys, n_head, rank, &scale, keys, rank,
                                          key_n, q + size_t(first) * head, rank, head, &zero, score, n_keys,
                                          score_n, c),
                "attention scores");
        dsa_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, n_keys, top_rows, kpool, (long long) pos,
                                                                   first, dense_rows);
        // out[head] = sum over the keys of score[head][key] * keys[key]
        blas_ck(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, rank, n_head, n_keys, &one, keys, rank, key_n,
                                          score, n_keys, score_n, &zero, out + size_t(first) * head, rank, head, c),
                "attention values");
    }
}

void idx_store(const float* key, const float* gate, float* ring, int ring_rows, int dim, int64_t pos, int rows,
               void* stream) {
    idx_store_kernel<<<rows, 256, 0, cs(stream)>>>(key, gate, ring, ring_rows, dim, (long long) pos);
}

void idx_pool(const float* ring, int ring_rows, float* pool, const float* ape, int dim, int kpool, int64_t pos,
              int rows, void* stream) {
    idx_pool_kernel<<<rows, 256, 0, cs(stream)>>>(ring, ring_rows, pool, ape, dim, kpool, (long long) pos);
}

void idx_score(const float* q, const float* w, const float* pool, float* score, int score_stride, int dim,
               int n_head, int kpool, int64_t pos, int rows, void* stream) {
    if (dim != IDX_DIM || n_head > WARP) throw std::invalid_argument("idx_score: 128 values a key, 32 heads at most");
    const int n_pool = int((pos + rows) / kpool);   // the last token sees the most
    if (n_pool == 0) return;
    // whole tiles, and some 500 blocks at a long context
    const int run = std::max(IDX_TILE, std::min(1024, n_pool / 512 / IDX_TILE * IDX_TILE));
    const int warps = std::min(rows, IDX_WARPS);
    idx_score_kernel<IDX_DIM><<<dim3((n_pool + run - 1) / run, (rows + warps - 1) / warps), warps * WARP, 0,
                                cs(stream)>>>(q, w, pool, score, score_stride, n_head, kpool, (long long) pos,
                                              1.0f / std::sqrt(float(dim) * float(n_head)), run, rows);
}

void idx_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int kpool,
                int64_t pos, int rows, void* stream) {
    idx_select_kernel<<<rows, IDX_SELECT, 2 * size_t(n_top) * sizeof(uint32_t), cs(stream)>>>(
        score, score_stride, sel, sel_stride, n_top, kpool, (long long) pos);
}

void idx_score_batch(void* blas, const float* q, const float* w, const float* pool, float* score, int score_stride,
                     int dim, int n_head, int kpool, int64_t pos, int rows, float* work, size_t work_floats,
                     void* stream) {
    const float one = 1.0f, zero = 0.0f;
    const int n_pool = int((pos + rows) / kpool);   // the last token sees the most
    if (n_pool == 0) return;
    const int chunk = (int) std::min<size_t>(65535, work_floats / (size_t(n_pool) * n_head));
    if (chunk < 1) throw std::invalid_argument("idx_score_batch: no room for a token's products");
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        blas_ck(cublasSgemm((cublasHandle_t) blas, CUBLAS_OP_T, CUBLAS_OP_N, n_pool, c * n_head, dim, &one, pool,
                            dim, q + size_t(first) * n_head * dim, dim, &zero, work, n_pool),
                "indexer scores");
        idx_weigh_kernel<<<dim3(grid(n_pool, 256), c), 256, 0, cs(stream)>>>(
            work, w + size_t(first) * n_head, score + size_t(first) * score_stride, score_stride, n_pool, n_head,
            kpool, (long long) pos + first, 1.0f / std::sqrt(float(dim) * float(n_head)));
    }
}

void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream) {
    swiglu_clamp_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(gate, up, out, n, limit);
}

void axpy(const float* x, float a, float* acc, int n, bool add, void* stream) {
    axpy_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, a, acc, n, add);
}

void copy2(const void* x0, void* y0, int n0, const void* x1, void* y1, int n1, void* stream) {
    copy2_kernel<<<grid(n0 + n1, 256), 256, 0, cs(stream)>>>((const uint32_t*) x0, (uint32_t*) y0, n0,
                                                             (const uint32_t*) x1, (uint32_t*) y1, n1);
}

void add_scaled(const ScaledRows& rows, int count, int n, void* stream) {
    if (count < 1 || count > 32) throw std::invalid_argument("add_scaled: 1 to 32 vectors");
    add_scaled_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(rows, count, n);
}

void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream) {
    swiglu_clamp_rows_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n, limit);
}

void geglu(const float* gate, const float* up, float* out, int n, void* stream) {
    geglu_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(gate, up, out, n);
}

void geglu_rows(const float* gate_up, float* out, int n, int rows, void* stream) {
    geglu_rows_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n);
}

void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream) {
    moe_sum_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(x, w, acc, n, n_used);
}

void add_bias(const float* b, float* acc, int n, int rows, void* stream) {
    add_bias_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(b, acc, n);
}

void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq,
                    const int32_t* axes, int sections, const long long* at) {
    head_norm_rope_kernel<<<rows * heads, dim >= 256 ? 256 : 128, 0, cs(stream)>>>(
        x, x_stride, w, y, dim, heads, n_rot, theta_scale, freq, (long long) pos, eps, axes, sections, at);
}

void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window,
            int ring, bool whole, const long long* at, int room) {
    if (at && (whole || room < 1)) throw std::invalid_argument("attend: a position on the card needs its room");
    if (dim % SCORE_VALUES || uintptr_t(q) % (SCORE_VALUES * sizeof(float)) || uintptr_t(kcache) % (SCORE_VALUES * 2))
        throw std::invalid_argument("attend: queries and keys are read four values at a time");
    const long long last = whole ? pos + rows - 1 : -1;
    const int per = n_head / kv_heads;
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        // the launch's span: from the first row its first token attends to the last its last token does
        const long long p = pos + first, lo = window > 0 && p >= window ? p - window + 1 : 0;
        const int keys = at ? room : int((whole ? last : p + c - 1) - lo) + 1;
        const unsigned units = grid(size_t(keys), SCORE_KEYS) * kv_heads, threads = SCORE_WARPS * WARP;
        const auto reg = [&](auto kernel) {
            kernel<<<grid(units, SCORE_WARPS), threads, 0, cs(stream)>>>(q, (const half*) kcache, score, keys, n_head,
                                                                        kv_heads, (long long) pos, first, c, scale,
                                                                        window, ring, last, at);
        };
        // The queries in registers where a head has 128, 256 or 512 values and a group 4 or 8 heads.  With fewer
        // heads a group the cache is the most of what is read and the other kernel is as fast.
        if (per % 4 == 0 && dim == 128) reg(attn_score_reg_kernel<4, 1>);
        else if (per % 8 == 0 && dim == 256) reg(attn_score_reg_kernel<8, 2>);
        else if (per % 4 == 0 && dim == 256) reg(attn_score_reg_kernel<4, 2>);
        else if (per % 4 == 0 && dim == 512) reg(attn_score_reg_kernel<4, 4>);
        else
            attn_score_kernel<<<grid(size_t(keys) * kv_heads, SCORE_WARPS), threads, 0, cs(stream)>>>(
                q, (const half*) kcache, score, keys, dim, n_head, kv_heads, (long long) pos, first, c, scale,
                window, ring, last, at);
        // the weights as attn_value_t_kernel reads them, behind the scores where the buffer has room for them
        const int blocks = grid(c * per, VALUE_QUERIES);
        float* wt = per >= 4 && size_t(kv_heads) * blocks * VALUE_QUERIES <= size_t(chunk - c) * n_head
                        ? score + size_t(c) * n_head * keys : nullptr;
        attn_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, keys, (long long) pos, first, c, window,
                                                                    last, at, wt, per);
        // Two blocks of queries a launch block read the values half as often, where the launch still has
        // blocks for most of the card's 36 multiprocessors: for Qwen3.6's two tokens they are half as many and
        // slower, for three or four tokens and for Gemma 4's full layers faster.
        const int dims = grid(dim, VALUE_DIMS);
        if (wt && blocks > 1 && kv_heads * dims * grid(blocks, 2) >= 32)
            attn_value_t_kernel<2><<<dim3(kv_heads * dims, grid(blocks, 2)), VALUE_DIMS * VALUE_SHARES,
                                     0, cs(stream)>>>(wt, (const half*) vcache, out, keys, dim, n_head, kv_heads,
                                                      (long long) pos, first, c, window, ring, last, at);
        else if (wt)
            attn_value_t_kernel<1><<<dim3(kv_heads * dims, blocks), VALUE_DIMS * VALUE_SHARES, 0, cs(stream)>>>(
                wt, (const half*) vcache, out, keys, dim, n_head, kv_heads, (long long) pos, first, c, window, ring,
                last, at);
        else
            attn_value_kernel<<<dim3(kv_heads * dims, blocks), VALUE_DIMS * VALUE_SHARES, 0,
                                cs(stream)>>>(score, (const half*) vcache, out, keys, dim, n_head, kv_heads,
                                              (long long) pos, first, c, window, ring, last, at);
    }
}

void attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                  float scale, void* stream, int window, int ring, bool whole) {
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const int per = n_head / kv_heads, width = kv_heads * dim;
    const long long last = whole ? pos + rows - 1 : -1;
    // The products take halves and sum floats, on the tensor cores: the cache's rows as they lie there, the
    // queries and the softmax's weights rounded.  Two thirds of the scratch are for a block's scores, the rest
    // for their weights.
    const size_t cells = score_floats / 3 * 2;
    half* weight = (half*) (score + cells);
    for (int first = 0, c = 0; first < rows; first += c) {
        // every token left, unless a grid's 65535 rows or the scratch for one row's scores is less, or the results
        // would outgrow kAttendResults: every block of rows reads and writes them all again (og)
        c = (int) std::min({size_t(rows - first), size_t(65535 / n_head), cells / n_head,
                            std::max<size_t>(16, kAttendResults / (size_t(n_head) * dim * sizeof(float)))});
        const long long p0 = pos + first;   // the chunk's first position, and the first row its tokens read
        const long long lo0 = window > 0 && p0 >= window ? p0 - window + 1 : 0;
        const int n_keys = int((whole ? last : p0 + c - 1) - lo0) + 1;
        const size_t heads = size_t(c) * n_head;
        half* qg = (half*) work;                      // the queries by group
        float* og = work + (heads * dim + 1) / 2;     // the results by group
        float* top = og + heads * dim;       // a token and head's largest score so far, the sum of its weights
        float* sum = top + heads;            // against it, and what a block of rows makes the sums before it worth
        float* shrink = sum + heads;
        attn_group_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(q + size_t(first) * n_head * dim,
                                                                                   qg, dim, n_head, kv_heads, c);
        // The rows go a block at a time, as many as have their scores in the scratch, so that a product has the
        // chunk's every token for its columns however long the context is: cuBLAS is several times slower on a
        // few columns.  A token's softmax is kept from block to block (top, sum), and its values so far shrink
        // to the new top.
        const long long cols = (long long) c * per;
        const int fit = (int) (cells / heads);
        const int block = fit > 16 ? fit & ~15 : fit;   // the tensor cores are half as fast on a ragged block
        for (int k0 = 0, m = 0; k0 < n_keys; k0 += m) {
            // the block's rows where the cache has them: a block ends where a ring starts over
            const long long row = ring > 0 ? (lo0 + k0) % ring : lo0 + k0;
            m = (int) std::min<long long>(std::min(block, n_keys - k0), ring > 0 ? ring - row : n_keys);
            // A product a group: its heads' queries of every token of the chunk against the group's keys.  cuBLAS
            // reads the cache's rows as the transpose, and the groups' keys lie `dim` apart in a row.
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, (int) cols, dim, &scale,
                                               (const half*) kcache + size_t(row) * width, CUDA_R_16F, width, dim, qg,
                                               CUDA_R_16F, dim, cols * dim, &zero, score, CUDA_R_32F, m, cols * m,
                                               kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention scores");
            attn_softmax_block_kernel<<<c * n_head, 128, 0, cs(stream)>>>(score, weight, m, lo0 + k0, (long long) pos,
                                                                         first, window, c, per, last, top, sum,
                                                                         shrink, k0 == 0);
            if (k0 > 0)
                attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, shrink, dim, false);
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, dim, (int) cols, m, &one,
                                               (const half*) vcache + size_t(row) * width, CUDA_R_16F, width, dim,
                                               weight, CUDA_R_16F, m, cols * m, k0 > 0 ? &one : &zero, og, CUDA_R_32F,
                                               dim, cols * dim, kv_heads, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention values");
        }
        attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, sum, dim, true);
        attn_ungroup_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(
            og, out + size_t(first) * n_head * dim, dim, n_head, kv_heads, c);
    }
}

void attend_int8(const float* q, const int8_t* kcache_q, const uint16_t* kcache_scale,
                 const int8_t* vcache_q, const uint16_t* vcache_scale, int64_t pos, int rows, float* score,
                 int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window,
                 int ring, bool whole, const long long* at, int room) {
    if (at && (whole || room < 1)) throw std::invalid_argument("attend_int8: a position on the card needs its room");
    if (dim % SCORE_VALUES || uintptr_t(q) % (SCORE_VALUES * sizeof(float)) || uintptr_t(kcache_q) % 4)
        throw std::invalid_argument("attend_int8: queries and keys are read four values at a time");
    const long long last = whole ? pos + rows - 1 : -1;
    const int per = n_head / kv_heads;
    for (int first = 0; first < rows; first += chunk) {
        const int c = std::min(chunk, rows - first);
        const long long p = pos + first, lo = window > 0 && p >= window ? p - window + 1 : 0;
        const int keys = at ? room : int((whole ? last : p + c - 1) - lo) + 1;
        attn_score_int8_kernel<<<grid(size_t(keys) * kv_heads, SCORE_WARPS), SCORE_WARPS * WARP, 0, cs(stream)>>>(
            q, kcache_q, kcache_scale, score, keys, dim, n_head, kv_heads, (long long) pos, first, c, scale, window,
            ring, last, at);
        attn_softmax_kernel<<<dim3(n_head, c), 256, 0, cs(stream)>>>(score, keys, (long long) pos, first, c, window,
                                                                    last, at, nullptr, per);
        attn_value_int8_kernel<<<dim3(kv_heads * grid(dim, VALUE_DIMS), grid(c * per, VALUE_QUERIES)),
                                 VALUE_DIMS * VALUE_SHARES, 0, cs(stream)>>>(
            score, vcache_q, vcache_scale, out, keys, dim, n_head, kv_heads, (long long) pos, first, c, window, ring,
            last, at);
    }
}

void attend_batch_int8(void* blas, const float* q, const int8_t* kcache_q, const uint16_t* kcache_scale,
                       const int8_t* vcache_q, const uint16_t* vcache_scale, int64_t pos, int rows,
                       float* score, size_t score_floats, float* work, uint16_t* kdequant, uint16_t* vdequant,
                       float* out, int dim, int n_head, int kv_heads,
                       float scale, void* stream, int window, int ring, bool whole) {
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const int per = n_head / kv_heads, width = kv_heads * dim;
    const long long last = whole ? pos + rows - 1 : -1;
    const size_t cells = score_floats / 3 * 2;
    half* weight = (half*) (score + cells);
    half* k_blk = (half*) kdequant;
    half* v_blk = (half*) vdequant;
    for (int first = 0, c = 0; first < rows; first += c) {
        c = (int) std::min({size_t(rows - first), size_t(65535 / n_head), cells / n_head});
        const long long p0 = pos + first;
        const long long lo0 = window > 0 && p0 >= window ? p0 - window + 1 : 0;
        const int n_keys = int((whole ? last : p0 + c - 1) - lo0) + 1;
        const size_t heads = size_t(c) * n_head;
        half* qg = (half*) work;
        float* og = work + (heads * dim + 1) / 2;
        float* top = og + heads * dim;
        float* sum = top + heads;
        float* shrink = sum + heads;
        attn_group_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(q + size_t(first) * n_head * dim,
                                                                                   qg, dim, n_head, kv_heads, c);
        const long long cols = (long long) c * per;
        const int fit = (int) (cells / heads);
        const int block = std::min(fit > 16 ? fit & ~15 : fit, 2048);
        for (int k0 = 0, m = 0; k0 < n_keys; k0 += m) {
            const long long row = ring > 0 ? (lo0 + k0) % ring : lo0 + k0;
            m = (int) std::min<long long>(std::min(block, n_keys - k0), ring > 0 ? ring - row : n_keys);
            const int total_items = m * kv_heads * (dim / 4);
            dequant_q8_to_f16_kernel<<<grid(total_items, 256), 256, 0, cs(stream)>>>(
                kcache_q, kcache_scale, k_blk, m, kv_heads, dim, row, ring);
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, (int) cols, dim, &scale,
                                               k_blk, CUDA_R_16F, width, dim, qg,
                                               CUDA_R_16F, dim, cols * dim, &zero, score, CUDA_R_32F, m, cols * m,
                                               kv_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention scores");
            attn_softmax_block_kernel<<<c * n_head, 128, 0, cs(stream)>>>(score, weight, m, lo0 + k0, (long long) pos,
                                                                         first, window, c, per, last, top, sum,
                                                                         shrink, k0 == 0);
            if (k0 > 0)
                attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, shrink, dim, false);
            dequant_q8_to_f16_kernel<<<grid(total_items, 256), 256, 0, cs(stream)>>>(
                vcache_q, vcache_scale, v_blk, m, kv_heads, dim, row, ring);
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, dim, (int) cols, m, &one,
                                               v_blk, CUDA_R_16F, width, dim,
                                               weight, CUDA_R_16F, m, cols * m, k0 > 0 ? &one : &zero, og, CUDA_R_32F,
                                               dim, cols * dim, kv_heads, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "attention values");
        }
        attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, sum, dim, true);
        attn_ungroup_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(
            og, out + size_t(first) * n_head * dim, dim, n_head, kv_heads, c);
    }
}

void softcap(float* x, size_t n, float cap, void* stream) {
    softcap_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(x, n, cap);
}

void likeliest(const float* logits, int n, Likeliest* out, void* stream) {
    likeliest_kernel<<<1, 1024, 0, cs(stream)>>>(logits, n, out);
}

void mul_sigmoid(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream) {
    for (int r = 0; r < rows; r += 65535)   // a grid has at most 65535 rows, and 4,096 tokens have 16 heads each
        mul_sigmoid_kernel<<<dim3(grid(n, 256), std::min(65535, rows - r)), 256, 0, cs(stream)>>>(
            acc + size_t(r) * n, g + size_t(r) * g_stride, n, g_stride, one_gate);
}

void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows, void* stream,
                float* snap, size_t stride, unsigned snaps) {
    conv4_silu_kernel<<<grid(channels, 256), 256, 0, cs(stream)>>>(x, state, w, out, channels, rows, snap, stride,
                                                                  snap ? snaps : 0u);
}

void gdn_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* s, const float* z,
              const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows, float norm_eps, void* stream,
              float* snap, size_t stride, unsigned snaps) {
    if (dim % WARP != 0 || dim > 256) throw std::invalid_argument("gdn_step: a head of 32 to 256 values, in 32s");
    const size_t row_n = (2 * size_t(k_heads) + n_head) * dim;
    const int gates = rows * n_head, columns = std::min(dim, 1024 / WARP);   // a block's columns: a warp each
    gdn_norm_kernel<<<rows * k_heads, dim, 0, cs(stream)>>>(qkv, dim, k_heads, row_n, norm_eps);
    gdn_gates_kernel<<<grid(gates, 256), 256, 0, cs(stream)>>>(alpha, beta, dt, a, n_head, gates);
    const int blocks = n_head * ((dim + columns - 1) / columns), threads = columns * WARP;
    const unsigned sn = snap ? snaps : 0u;
    switch (dim / WARP) {
    case 1: gdn_state_reg_kernel<1><<<blocks, threads, 0, cs(stream)>>>(qkv, alpha, beta, s, out, dim, n_head, k_heads,
                                                                          rows, snap, stride, sn); break;
    case 2: gdn_state_reg_kernel<2><<<blocks, threads, 0, cs(stream)>>>(qkv, alpha, beta, s, out, dim, n_head, k_heads,
                                                                          rows, snap, stride, sn); break;
    case 4: gdn_state_reg_kernel<4><<<blocks, threads, 0, cs(stream)>>>(qkv, alpha, beta, s, out, dim, n_head, k_heads,
                                                                          rows, snap, stride, sn); break;
    case 8: gdn_state_reg_kernel<8><<<blocks, threads, 0, cs(stream)>>>(qkv, alpha, beta, s, out, dim, n_head, k_heads,
                                                                          rows, snap, stride, sn); break;
    default: gdn_state_kernel<<<blocks, threads, 0, cs(stream)>>>(qkv, alpha, beta, s, out, dim, n_head, k_heads, rows,
                                                                  snap, stride, sn);
    }
    gdn_out_kernel<<<rows * n_head, dim, 0, cs(stream)>>>(out, z, o_norm, dim, norm_eps);
}

}  // namespace strata::poly
