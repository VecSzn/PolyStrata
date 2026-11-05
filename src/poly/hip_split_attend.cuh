// HIP only: attend() for a few tokens over a long q8_0 cache, heads of 256 values in groups of 8 (Qwen3.5 / Tiel: 16 query heads over 2 KV heads).
//
// The score, softmax and value kernels of attend() launch 16 to 32 blocks for such a call, each walking the whole conversation, and every token
// of the call reads the whole cache again: 3.4 ms a layer at 110K keys, 4 tokens 8 ms, against 0.3 ms the cache's bytes take at the card's bandwidth.
// This is flash-decoding for those shapes: a launch block takes kKeys keys of one KV head and ALL the call's tokens (up to 4: 8 heads x tokens = up to
// 32 queries), so each key and value is read once for all of them; it walks its keys in tiles with an online softmax and writes (sums, max, sum of
// exponentials); join() adds the blocks up.  The addition order differs from the old kernels', so the results are not bitwise theirs.
#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace strata::poly::hip_split_attend {

constexpr int kDim = 256;       // head dimension
constexpr int kG = 8;           // query heads a KV head
constexpr int kKeys = 512;      // keys a launch block
constexpr int kThreads = 256;   // 8 waves of 32
constexpr int kPad = 260;       // a row of the tile in LDS: 256 codes and 4 more, so a lane a row reads without bank conflicts
constexpr int kMaxRows = 4;     // tokens a launch: 8 x 4 = 32 queries

inline bool requested(const char* v) { return v != nullptr && std::atoi(v) != 0; }

/// Whether the call's shapes fit: the partial sums (a token's, a head's, a block's: kDim + 2 floats) must fit in `part_floats`.
inline bool fits(int dim, int n_head, int kv_heads, int keys, int rows, int window, int ring, bool whole, bool sinks, size_t part_floats) {
    if (dim != kDim || kv_heads <= 0 || n_head != kv_heads * kG || window != 0 || ring != 0 || whole || sinks) return false;
    const int c = rows < kMaxRows ? rows : kMaxRows;
    const size_t blocks = (size_t(keys) + kKeys - 1) / kKeys;
    return part_floats >= size_t(c) * n_head * blocks * (kDim + 2);
}

// QG groups of 4 queries (8 x tokens padded up to 8, 16 or 32 queries): a tile of 32 * 8 / QG keys, a wave a (query group, 32 keys) in the scores.
template <int QG>
__global__ void __launch_bounds__(kThreads)
split_q8(const float* __restrict__ q, const int8_t* __restrict__ kc, const uint16_t* __restrict__ ks, const int8_t* __restrict__ vc,
         const uint16_t* __restrict__ vs, float* __restrict__ part, int n_head, int kv_heads, long long pos, int first, int tokens,
         float scale, int blocks, const long long* __restrict__ at) {
    constexpr int Q = 4 * QG, TILE = 32 * (8 / QG);
    constexpr int NU = TILE / 16, NS = TILE / 32;   // 16-byte parts and scale values a thread loads for a tile
    __shared__ __attribute__((aligned(16))) float Qs[kDim * Q];            // [dim][query]: scaled queries
    __shared__ __attribute__((aligned(16))) int8_t Ks[TILE * kPad];        // the tile's key codes, then (after the scores) its value codes
    __shared__ float Kscale[TILE * 8], Vscale[TILE * 8];
    __shared__ float S[Q * (TILE + 1)];                                     // [query][key]
    __shared__ __attribute__((aligned(16))) float P[TILE * Q];             // [key][query]: weights
    __shared__ __attribute__((aligned(16))) float alpha[Q];
    const int g = blockIdx.y, b = blockIdx.x, tid = threadIdx.x, lane = tid % 32, w = tid / 32;
    const long long p = (at ? *at : pos) + first;
    const int k0 = b * kKeys, span = int(p) + tokens;   // the call's tokens attend rows [0, span)
    if (k0 >= span) return;
    const int nk = min(kKeys, span - k0);

    for (int i = tid; i < Q * kDim; i += kThreads) {
        const int qi = i / kDim, d = i % kDim, t = qi / 8, h = qi % 8;
        Qs[d * Q + qi] = t < tokens ? q[((size_t(first) + t) * n_head + g * kG + h) * kDim + d] * scale : 0.0f;
    }
    // The values' tile of a thread: 8 queries (a token's 8 heads) x 4 dimensions.  64 dimension groups x NQG tokens x KS key subsets are 256 threads,
    // so each LDS read of a weight or a value byte feeds 32 multiply-adds (a thread a dimension fed 8 to 32 of them for 3 to 8 reads a key).
    constexpr int NQG = Q / 8, KS = 4 / NQG;
    const int dg = tid & 63, qgrp = (tid >> 6) % NQG, ksub = (tid >> 6) / NQG;
    float O[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) { O[i][0] = 0.0f; O[i][1] = 0.0f; O[i][2] = 0.0f; O[i][3] = 0.0f; }
    float m_run[Q / 8], l_run[Q / 8];
#pragma unroll
    for (int u = 0; u < Q / 8; ++u) { m_run[u] = -INFINITY; l_run[u] = 0.0f; }

    // a tile's rows and scales go global -> registers (issued early, so their latency hides behind the arithmetic of the phase before) -> LDS
    uint4 kr[NU], vr[NU];
    unsigned short ksr[NS], vsr[NS];
    const auto load = [&](const int8_t* codes, const uint16_t* scales, uint4* r, unsigned short* sr, int t0) {
        const int ntile = min(TILE, nk - t0);
#pragma unroll
        for (int i = 0; i < NU; ++i) {
            const int e = tid + kThreads * i;
            r[i] = e < ntile * 16 ? *reinterpret_cast<const uint4*>(codes + (size_t(k0 + t0 + e / 16) * kv_heads + g) * kDim + (e % 16) * 16)
                                  : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const int e = tid + kThreads * i;
            sr[i] = e < ntile * 8 ? scales[(size_t(k0 + t0 + e / 8) * kv_heads + g) * 8 + e % 8] : (unsigned short) 0;
        }
    };
    const auto put = [&](const uint4* r, const unsigned short* sr, float* scale_lds, int t0) {
        const int ntile = min(TILE, nk - t0);
#pragma unroll
        for (int i = 0; i < NU; ++i) {
            const int e = tid + kThreads * i;
            if (e < ntile * 16) {
                int* dst = reinterpret_cast<int*>(Ks + (e / 16) * kPad + (e % 16) * 16);
                dst[0] = int(r[i].x); dst[1] = int(r[i].y); dst[2] = int(r[i].z); dst[3] = int(r[i].w);
            }
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const int e = tid + kThreads * i;
            if (e < ntile * 8) scale_lds[e] = __half2float(__ushort_as_half(sr[i]));
        }
    };

    load(kc, ks, kr, ksr, 0);
    for (int t0 = 0; t0 < nk; t0 += TILE) {
        const int ntile = min(TILE, nk - t0);
        put(kr, ksr, Kscale, t0);
        __syncthreads();
        load(vc, vs, vr, vsr, t0);   // the values arrive while the scores and the softmax are computed
        // (B) scores: wave w takes query group w % QG and the keys 32 * (w / QG) of the tile, a lane a key
        {
            const int qg = w % QG, j = (w / QG) * 32 + lane;
            float s[4] = {0.f, 0.f, 0.f, 0.f};
            if (j < ntile) {
                const int* krow = reinterpret_cast<const int*>(Ks + j * kPad);
                for (int blk = 0; blk < 8; ++blk) {
                    float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int i4 = 0; i4 < 8; ++i4) {
                        const int pk = krow[blk * 8 + i4];
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float kvv = float((pk << (24 - 8 * e)) >> 24);
                            const float4 qv = *reinterpret_cast<const float4*>(&Qs[(blk * 32 + i4 * 4 + e) * Q + qg * 4]);
                            a[0] += kvv * qv.x; a[1] += kvv * qv.y; a[2] += kvv * qv.z; a[3] += kvv * qv.w;
                        }
                    }
                    const float sc = Kscale[j * 8 + blk];
#pragma unroll
                    for (int r = 0; r < 4; ++r) s[r] += a[r] * sc;
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int qi = qg * 4 + r, t = qi / 8;
                const long long own = t < tokens ? p + t + 1 : 0;   // the token attends rows [0, own)
                S[qi * (TILE + 1) + (w / QG) * 32 + lane] = (j < ntile && k0 + t0 + j < own) ? s[r] : -INFINITY;
            }
        }
        __syncthreads();
        // (C) the online softmax: wave w takes the queries w, w + 8, ...; every lane of it holds the query's running max and sum
#pragma unroll
        for (int u = 0; u < Q / 8; ++u) {
            const int qi = w + 8 * u;
            float sv[TILE / 32], tmax = -INFINITY;
#pragma unroll
            for (int i = 0; i < TILE / 32; ++i) { sv[i] = S[qi * (TILE + 1) + lane + 32 * i]; tmax = fmaxf(tmax, sv[i]); }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor(tmax, o, 32));
            const float m_new = fmaxf(m_run[u], tmax);
            const float a = (m_new == -INFINITY) ? 1.0f : expf(m_run[u] - m_new);   // m_run = -inf: 0
            float psum = 0.0f;
#pragma unroll
            for (int i = 0; i < TILE / 32; ++i) {
                const float e = (m_new == -INFINITY) ? 0.0f : expf(sv[i] - m_new);
                P[(lane + 32 * i) * Q + qi] = e;
                psum += e;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor(psum, o, 32);
            l_run[u] = l_run[u] * a + psum;
            m_run[u] = m_new;
            if (lane == 0) alpha[qi] = a;
        }
        put(vr, vsr, Vscale, t0);   // the scores are done with Ks: the values take its place
        __syncthreads();
        if (t0 + TILE < nk) load(kc, ks, kr, ksr, t0 + TILE);   // the next tile's keys arrive during the values
        // (D) the values: thread (dimension group, token, key subset) takes dimensions 4 dg .. 4 dg + 3 of every KS-th row of the tile, for the token's 8 heads
        {
            const float4 a0 = *reinterpret_cast<const float4*>(&alpha[qgrp * 8]), a1 = *reinterpret_cast<const float4*>(&alpha[qgrp * 8 + 4]);
            const float al[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) { O[i][0] *= al[i]; O[i][1] *= al[i]; O[i][2] *= al[i]; O[i][3] *= al[i]; }
        }
        for (int j = ksub; j < ntile; j += KS) {
            const int vi = *reinterpret_cast<const int*>(Ks + j * kPad + 4 * dg);   // 4 value codes
            const float sc = Vscale[j * 8 + (dg >> 3)];
            const float v0 = float((vi << 24) >> 24) * sc, v1 = float((vi << 16) >> 24) * sc, v2 = float((vi << 8) >> 24) * sc, v3 = float(vi >> 24) * sc;
            const float4 pa = *reinterpret_cast<const float4*>(&P[j * Q + qgrp * 8]), pb = *reinterpret_cast<const float4*>(&P[j * Q + qgrp * 8 + 4]);
            const float pw[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) { O[i][0] += pw[i] * v0; O[i][1] += pw[i] * v1; O[i][2] += pw[i] * v2; O[i][3] += pw[i] * v3; }
        }
        __syncthreads();
    }
    // the block's partial sums: (kDim sums, the max, the sum of exponentials) a (token, head).  With KS > 1 the key subsets' sums are added first, through
    // the tile's LDS (the values are done with it).
    if (KS > 1) {
        float* red = reinterpret_cast<float*>(Ks);   // [KS - 1][Q][kDim]
        if (ksub > 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                *reinterpret_cast<float4*>(&red[((ksub - 1) * Q + qgrp * 8 + i) * kDim + 4 * dg]) = make_float4(O[i][0], O[i][1], O[i][2], O[i][3]);
        }
        __syncthreads();
        if (ksub == 0) {
#pragma unroll
            for (int k = 1; k < KS; ++k)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 r = *reinterpret_cast<const float4*>(&red[((k - 1) * Q + qgrp * 8 + i) * kDim + 4 * dg]);
                    O[i][0] += r.x; O[i][1] += r.y; O[i][2] += r.z; O[i][3] += r.w;
                }
        }
    }
    if (ksub == 0 && qgrp < tokens) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float* out = part + ((size_t(qgrp) * n_head + g * kG + i) * blocks + b) * (kDim + 2) + 4 * dg;
            out[0] = O[i][0]; out[1] = O[i][1]; out[2] = O[i][2]; out[3] = O[i][3];
        }
    }
#pragma unroll
    for (int u = 0; u < Q / 8; ++u) {
        const int qi = w + 8 * u, t = qi / 8, h = qi % 8;
        if (lane == 0 && t < tokens) {
            float* out = part + ((size_t(t) * n_head + g * kG + h) * blocks + b) * (kDim + 2);
            out[kDim] = m_run[u];
            out[kDim + 1] = l_run[u];
        }
    }
}

// The blocks added up for a head of a token: each block's sums scaled to the largest max; a block the token sees nothing of has max -inf and weighs 0.
__global__ void __launch_bounds__(kThreads)
join(const float* __restrict__ part, float* __restrict__ out, int n_head, long long pos, int first, int tokens, int blocks,
     const long long* __restrict__ at) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const long long p = (at ? *at : pos) + first;
    const int used = int((p + tokens + kKeys - 1) / kKeys);   // the blocks split_q8 wrote
    const float* base = part + (size_t(t) * n_head + h) * blocks * (kDim + 2);
    float mx = -INFINITY;
    for (int b = 0; b < used; ++b) mx = fmaxf(mx, base[size_t(b) * (kDim + 2) + kDim]);
    float num = 0.0f, den = 0.0f;
    for (int b = 0; b < used; ++b) {
        const float m = base[size_t(b) * (kDim + 2) + kDim];
        if (m == -INFINITY) continue;
        const float wt = expf(m - mx);
        num += wt * base[size_t(b) * (kDim + 2) + d];
        den += wt * base[size_t(b) * (kDim + 2) + kDim + 1];
    }
    out[(size_t(first + t) * n_head + h) * kDim + d] = num / den;
}

/// One launch of up to kMaxRows tokens from row `first` of the call.  `blocks` covers the longest the call can attend (`room` with `at`).
inline void run(const float* q, const int8_t* kc, const uint16_t* ks, const int8_t* vc, const uint16_t* vs, float* part, float* out, int n_head,
                int kv_heads, long long pos, int first, int tokens, float scale, int blocks, const long long* at, hipStream_t stream) {
    const dim3 grid(blocks, kv_heads);
    if (tokens <= 1)
        hipLaunchKernelGGL(split_q8<2>, grid, dim3(kThreads), 0, stream, q, kc, ks, vc, vs, part, n_head, kv_heads, pos, first, tokens, scale, blocks, at);
    else if (tokens == 2)
        hipLaunchKernelGGL(split_q8<4>, grid, dim3(kThreads), 0, stream, q, kc, ks, vc, vs, part, n_head, kv_heads, pos, first, tokens, scale, blocks, at);
    else
        hipLaunchKernelGGL(split_q8<8>, grid, dim3(kThreads), 0, stream, q, kc, ks, vc, vs, part, n_head, kv_heads, pos, first, tokens, scale, blocks, at);
    hipLaunchKernelGGL(join, dim3(n_head, tokens), dim3(kThreads), 0, stream, part, out, n_head, pos, first, tokens, blocks, at);
}

}  // namespace strata::poly::hip_split_attend
