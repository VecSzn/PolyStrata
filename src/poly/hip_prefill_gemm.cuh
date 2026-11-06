// The two products of attend_batch() (prompt attention) as register-tiled FP32 kernels that read the q8_0 cache as it lies:
//   scores:  S[n][m]  = alpha * sum_k  K[row0 + m][g*dim + k] * Q[n][k]          (K = q8 codes x fp16 scales, Q = grouped queries as floats)
//   values:  O[n][d] (+)= sum_k  V[row0 + k][g*dim + d] * W[n][k]               (V = q8 codes x fp16 scales, W = the softmax's weights as floats)
// n runs over a group's columns (tokens x heads of the group), m over the block's keys, g = blockIdx.z the KV head.  Tiles of 128 x 128 x 16, 256 threads
// of 8 x 8 outputs each, the tiles double-buffered in LDS, the next one loaded into registers while the current one is multiplied.  Arithmetic is FP32 with
// FP32 sums, so against SGEMM on the dequantized cache the only difference is the order of the sums.  No FP32 copy of the cache is made.
//
#pragma once
#if __has_include(<cuda_fp16.h>)
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#endif

#include <cstdint>
#include <type_traits>

#ifndef KUNROLL
#define KUNROLL 2
#endif

namespace hip_prefill_gemm {

constexpr int BM = 128, BN = 128, BK = 16, kThreads = 256, kPadM = 4;   // LDS rows are BM + 4 floats: 16-byte aligned, reads and stores spread over the banks

struct Args {
    const int8_t* codes;       // the cache's codes: row r, value i at codes[r * width + i]
    const uint16_t* scales;    // fp16 scales of 32 values: scales[r * (width / 32) + i / 32]
    long long row0;            // the cache row of m = 0 (values: k = 0)
    int width;                 // the cache's row: kv_heads * dim
    int dim;                   // values a head
    const float* B;            // [group][n][ldb]: queries (scores) or weights (values)
    long long b_stride;        // floats between groups of B
    int ldb;                   // floats between the columns n of B (a multiple of 4)
    float* C;                  // [group][n][ldc]
    long long c_stride;
    int ldc;                   // a multiple of 4
    int M, N, K;               // the product's size; scores M = keys, K = dim; values M = dim, K = keys
    float alpha;
    int accumulate;            // values: C += instead of C =
};

__device__ __forceinline__ void unpack8(uint64_t w, float s, float* o) {
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = (float) (int8_t) (w >> (8 * i)) * s;
}

__device__ __forceinline__ uint64_t load_codes(const int8_t* p) { return *reinterpret_cast<const uint64_t*>(p); }

// SCORES = true: A is K (k contiguous in a row), values false: A is V (m contiguous in a row)
template <bool SCORES>
__global__ void __launch_bounds__(kThreads) gemm_q8(const Args a) {
    __shared__ __attribute__((aligned(16))) float As[2][BK][BM + kPadM];
    __shared__ __attribute__((aligned(16))) float Bs[2][BK][BN + kPadM];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int g = blockIdx.z;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const float* __restrict__ B = a.B + (long long) g * a.b_stride;
    float* __restrict__ C = a.C + (long long) g * a.c_stride;
    const int gofs = g * a.dim;   // the group's values within a cache row (a multiple of 32)
    const int swidth = a.width >> 5;

    // what a thread loads for a tile: A: 8 codes; B: 8 floats.  (row, half) = (tid & 127, tid >> 7) for the k-contiguous operands; on RDNA1 (gfx1010 to
    // gfx1012) (tid >> 1, tid & 1): the two lanes of a row are neighbours, so a load instruction of a wave touches 16 rows, not 32 (measured on gfx1010,
    // the scores product 1.00 -> 0.88 ms).  Not measured on other cards, which keep the first mapping.
#if defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__)
    const int lrow = tid >> 1, lhalf = tid & 1;
#else
    const int lrow = tid & (BM - 1), lhalf = tid >> 7;
#endif
    float ra[8], rb[8];

    // the thread's rows, once: B's column and the cache row it reads; per tile only a 32-bit step is added
    const int bn = n0 + lrow;
    const float* __restrict__ bp = B + (long long) (bn < a.N ? bn : 0) * a.ldb + 8 * lhalf;
    const int8_t* __restrict__ ap;
    const uint16_t* __restrict__ sp;
    bool a_ok;
    if (SCORES) {
        const int m = m0 + lrow;
        a_ok = m < a.M;
        const long long r = a.row0 + (a_ok ? m : 0);
        ap = a.codes + r * a.width + gofs + 8 * lhalf;
        sp = a.scales + r * swidth + (gofs >> 5);
    } else {
        const int d = m0 + 8 * (tid & 15);
        a_ok = d < a.M;
        const int dd = a_ok ? d : 0;
        const long long r = a.row0 + (tid >> 4);
        ap = a.codes + r * a.width + gofs + dd;
        sp = a.scales + r * swidth + ((gofs + dd) >> 5);
    }

    auto load = [&](auto full_tag, int k0) {
        constexpr bool FULL = decltype(full_tag)::value;
        const int kb = k0 + 8 * lhalf;
        if (FULL || (bn < a.N && kb + 8 <= a.K)) {
            const float4 x = *reinterpret_cast<const float4*>(bp + k0);
            const float4 y = *reinterpret_cast<const float4*>(bp + k0 + 4);
            rb[0] = x.x; rb[1] = x.y; rb[2] = x.z; rb[3] = x.w; rb[4] = y.x; rb[5] = y.y; rb[6] = y.z; rb[7] = y.w;
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) rb[i] = (bn < a.N && kb + i < a.K) ? bp[k0 + i] : 0.0f;
        }
        if (SCORES) {
            if (FULL || a_ok) {
                const uint64_t w = load_codes(ap + k0);
                const float s = __half2float(__ushort_as_half(sp[(k0 + 8 * lhalf) >> 5]));
                unpack8(w, s, ra);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) ra[i] = 0.0f;
            }
        } else {
            const int key = k0 + (tid >> 4);
            if (FULL || (key < a.K && a_ok)) {
                const uint64_t w = load_codes(ap + (long long) k0 * a.width);
                const float s = __half2float(__ushort_as_half(sp[(long long) k0 * swidth]));
                unpack8(w, s, ra);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) ra[i] = 0.0f;
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 8; ++i) Bs[buf][8 * lhalf + i][lrow] = rb[i];
        if (SCORES) {
#pragma unroll
            for (int i = 0; i < 8; ++i) As[buf][8 * lhalf + i][lrow] = ra[i];
        } else {
            float* p = &As[buf][tid >> 4][8 * (tid & 15)];
            *reinterpret_cast<float4*>(p) = make_float4(ra[0], ra[1], ra[2], ra[3]);
            *reinterpret_cast<float4*>(p + 4) = make_float4(ra[4], ra[5], ra[6], ra[7]);
        }
    };

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    const int tiles = (a.K + BK - 1) / BK;
    auto loop = [&](auto full_tag) {
        load(full_tag, 0);
        store(0);
        __syncthreads();
        for (int t = 0; t < tiles; ++t) {
            const int cur = t & 1;
            if (t + 1 < tiles) load(full_tag, (t + 1) * BK);   // the next tile into registers while this one is multiplied
#pragma unroll KUNROLL
            for (int kk = 0; kk < BK; ++kk) {
                const float4 a0 = *reinterpret_cast<const float4*>(&As[cur][kk][tx * 4]);
                const float4 a1 = *reinterpret_cast<const float4*>(&As[cur][kk][64 + tx * 4]);
                const float4 b0 = *reinterpret_cast<const float4*>(&Bs[cur][kk][ty * 4]);
                const float4 b1 = *reinterpret_cast<const float4*>(&Bs[cur][kk][64 + ty * 4]);
                const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
            }
            if (t + 1 < tiles) store(cur ^ 1);
            __syncthreads();
        }
    };
    // a whole tile: every column, row and k step is inside, so none of the loads asks
    if (n0 + BN <= a.N && m0 + BM <= a.M && a.K % BK == 0) loop(std::true_type{});
    else loop(std::false_type{});

    // C[n][m]: for each of the thread's 8 columns, its two runs of 4 consecutive rows
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int n = n0 + (j < 4 ? ty * 4 + j : 64 + ty * 4 + (j - 4));
        if (n >= a.N) continue;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int m = m0 + (h == 0 ? tx * 4 : 64 + tx * 4);
            if (m >= a.M) continue;
            float* p = C + (long long) n * a.ldc + m;
            float4 v = make_float4(acc[h * 4][j] * a.alpha, acc[h * 4 + 1][j] * a.alpha, acc[h * 4 + 2][j] * a.alpha, acc[h * 4 + 3][j] * a.alpha);
            if (a.accumulate) {
                const float4 o = *reinterpret_cast<const float4*>(p);
                v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
            }
            if (m + 3 < a.M) *reinterpret_cast<float4*>(p) = v;
            else {
                const float vv[4] = {v.x, v.y, v.z, v.w};
                for (int i = 0; m + i < a.M && i < 4; ++i) p[i] = vv[i];
            }
        }
    }
}

}  // namespace hip_prefill_gemm
