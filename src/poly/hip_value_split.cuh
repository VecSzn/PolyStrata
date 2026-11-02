// HIP only: reduce a GQA8 group's weighted values in 1024-key parts, then join
// the parts. The score and softmax kernels, and the KV representation, stay as
// they were. The changed FP32 addition order is not bitwise the old reduction.
#pragma once
#include "hip_value_split_policy.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace strata::poly::hip_value_split {

template <bool Q8>
static __global__ void split(const float* wt, const uint16_t* values, const int8_t* codes, const uint16_t* scales,
                      float* partial, int keys, int heads, int kv_heads, int first, int rows, long long pos,
                      const long long* at, int parts) {
    __shared__ float sums[kGroup][kShares][32];
    const int lane = threadIdx.x % 32, share = threadIdx.x / 32;
    const int group = blockIdx.x / 8, dim = (blockIdx.x % 8) * 32 + lane;
    const int part = blockIdx.y, tok = blockIdx.z;
    const long long end = (at ? *at : pos) + first + tok + 1;
    const long long last = end < (long long) (part + 1) * kKeys ? end : (long long) (part + 1) * kKeys;
    float acc[kGroup] = {};
    const float* wbase = wt + (size_t(group) * rows + tok) * keys * kGroup;
    for (long long row = (long long) part * kKeys + share; row < last; row += kShares) {
        const size_t index = (size_t(row) * kv_heads + group) * kDim + dim;
        float v;
        if constexpr (Q8) v = float(codes[index]) * __half2float(__ushort_as_half(scales[index / 32]));
        else v = __half2float(__ushort_as_half(values[index]));
        const float4 a = *reinterpret_cast<const float4*>(wbase + size_t(row) * kGroup);
        const float4 b = *reinterpret_cast<const float4*>(wbase + size_t(row) * kGroup + 4);
        acc[0] += v * a.x; acc[1] += v * a.y; acc[2] += v * a.z; acc[3] += v * a.w;
        acc[4] += v * b.x; acc[5] += v * b.y; acc[6] += v * b.z; acc[7] += v * b.w;
    }
#pragma unroll
    for (int h = 0; h < kGroup; ++h) sums[h][share][lane] = acc[h];
    __syncthreads();
    if (share == 0) {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            float total = 0.f;
#pragma unroll
            for (int s = 0; s < kShares; ++s) total += sums[h][s][lane];
            partial[((size_t(tok) * heads + group * kGroup + h) * parts + part) * kDim + dim] = total;
        }
    }
}

static __global__ void join(const float* partial, float* out, int heads, int parts, int first) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float* base = partial + (size_t(t) * heads + h) * parts * kDim + d;
    float sum = 0.f;
    for (int p = 0; p < parts; ++p) sum += base[size_t(p) * kDim];
    out[(size_t(first + t) * heads + h) * kDim + d] = sum;
}

static inline void run(const float* wt, const uint16_t* values, const int8_t* codes, const uint16_t* scales,
                float* scratch, float* out, int keys, int heads, int kv_heads, int first, int rows,
                long long pos, const long long* at, hipStream_t stream) {
    const int parts = (keys - 1) / kKeys + 1;
    const dim3 grid(kv_heads * 8, parts, rows), block(32 * kShares);
    if (codes)
        hipLaunchKernelGGL(split<true>, grid, block, 0, stream, wt, values, codes, scales, scratch, keys, heads,
                          kv_heads, first, rows, pos, at, parts);
    else
        hipLaunchKernelGGL(split<false>, grid, block, 0, stream, wt, values, codes, scales, scratch, keys, heads,
                          kv_heads, first, rows, pos, at, parts);
    hipLaunchKernelGGL(join, dim3(heads, rows), dim3(kDim), 0, stream, scratch, out, heads, parts, first);
}
}  // namespace strata::poly::hip_value_split
