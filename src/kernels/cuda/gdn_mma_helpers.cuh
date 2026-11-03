// Adapted from ggml-org/llama.cpp 7182ef71e6d5dd0c8ea76486fac134fcd25025d4.
// Original GDN MMA kernel: PR #29353. MIT; see third_party/ggml/LICENSE.
#pragma once
#include "mma.cuh"
namespace ggml_cuda_mma {
    template <int stride, typename T>
    static __device__ __forceinline__ uint32_t swizzle(const uint32_t offset, const uint32_t i) {
        static_assert(sizeof(T) <= 4, "unsupported type size");
        constexpr int stride_bytes = stride*sizeof(T);
        static_assert(stride_bytes % 16 == 0, "bad stride");
        constexpr uint32_t shift = sizeof(T) == 1 ? 4 : (sizeof(T) == 2 ? 3 : 2);
        if (stride_bytes % 32 != 0) {
            return offset; // Equivalent to padding with 16 bytes.
        }
        if (stride_bytes % 64 != 0) {
            return offset ^ (((i / 4) % 2) << shift);
        }
        if (stride_bytes % 128 != 0) {
            return offset ^ (((i / 2) % 4) << shift);
        }
        return offset ^ ((i % 8) << shift);
    }

    template <int stride, typename T>
    static __device__ __forceinline__ T * swizzle(T * ptr, const uint32_t offset, const uint32_t i) {
        return ptr + swizzle<stride, T>(offset, i);
    }

    template <int stride, int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix_swizzled(
            tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int offset) {
#if defined(TURING_MMA_AVAILABLE)
        static_assert(I == 8 || I == 16, "bad tile width");
        static_assert(J ==  8, "bad tile height");
        const int i = threadIdx.x % t.I;
        const int j = ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
        int offset_ij = offset + i * stride + j;
        offset_ij = swizzle<stride, T>(offset_ij, i);
        int * xi = (int *) t.x;
        if constexpr (I == 16) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
                : "l"(xs0 + offset_ij));
        } else {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                : "=r"(xi[0]), "=r"(xi[1])
                : "l"(xs0 + offset_ij));
        }
#elif defined(VOLTA_MMA_AVAILABLE)
#pragma unroll
        for (int o = 0; o < t.ne; o += 4) {
            const int offset_ij = offset + t.get_i(o) * stride + o;
            ggml_cuda_memcpy_1<4*sizeof(T)>(t.x + o, swizzle<stride>(xs0, offset_ij, t.get_i(o)));
        }
#elif defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
        static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad data layout");
        static_assert(sizeof(t.x) == 32, "bad ne");
        static_assert(I == 16, "bad tile width");
        static_assert(J ==  8, "bad tile height");
#pragma unroll
        for (int o = 0; o < 8; o += 4) {
            const int offset_ij = offset + t.get_i(0) * stride + o;
            ggml_cuda_memcpy_1<16>(t.x + o, swizzle<stride>(xs0, offset_ij, t.get_i(0)));
        }
#else
        static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad data layout");
        static_assert(sizeof(t.x) == 16, "bad ne");
        const int offset_ij = offset + t.get_i(0)*stride + t.get_j(0);
        ggml_cuda_memcpy_1<16>(t.x, swizzle<stride>(xs0, offset_ij, t.get_i(0)));
#endif // RDNA3
#elif defined(AMD_MFMA_AVAILABLE)
        static_assert(sizeof(t.x) == 8, "bad ne");
        const int offset_ij = offset + t.get_i(0)*stride + t.get_j(0);
        ggml_cuda_memcpy_1<8>(t.x, swizzle<stride>(xs0, offset_ij, t.get_i(0)));
#else
        GGML_UNUSED_VARS(t, xs0, offset);
        NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
    }

    template <int stride>
    static __device__ __forceinline__ void load_ldmatrix_swizzled(
            tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int offset) {
#ifdef VOLTA_MMA_AVAILABLE
#pragma unroll
        for (int l0 = 0; l0 < t.ne; l0 += 2) {
            const int offset_ij = offset + t.get_i(l0)*stride + t.get_j(l0);
            ggml_cuda_memcpy_1<2*sizeof(half2)>(t.x + l0, swizzle<stride>(xs0, offset_ij, t.get_i(l0)));
        }
#else
        GGML_UNUSED_VARS(t, xs0, offset);
        NO_DEVICE_CODE;
#endif // VOLTA_MMA_AVAILABLE
    }

    template <int stride, int I, typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix_trans_swizzled(
            tile<I, 8, T, dl> & t, const T * __restrict__ xs0, const int offset) {
#if defined(TURING_MMA_AVAILABLE)
        static_assert(I == 8 || I == 16, "bad tile width");
        static_assert(dl == DATA_LAYOUT_I_MAJOR, "bad data layout");
        const int i = threadIdx.x % 16;
        const int j = I == 16 ? (threadIdx.x / 16) * (t.J / 2) : 0;
        int offset_ij = offset + i * stride + j;
        offset_ij = swizzle<stride, T>(offset_ij, i);
        int * xi = (int *) t.x;
        if constexpr (I == 16) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
                : "l"(xs0 + offset_ij));
        } else {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                : "=r"(xi[0]), "=r"(xi[1])
                : "l"(xs0 + offset_ij));
        }
#elif defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        static_assert(dl == DATA_LAYOUT_I_MAJOR || dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad data layout");
        if constexpr (I == 32) {
#pragma unroll
            for (int l0 = 0; l0 < t.ne/2; ++l0) {
                half2 tmp[2];
#pragma unroll
                for (int o = 0; o < 2; ++o) {
                    const int j = 2*t.get_j(l0) + o;
                    int offset_ij = offset + j*stride + t.get_i(l0)/2;
                    offset_ij = swizzle<stride, T>(offset_ij, j);
                    tmp[o] = xs0[offset_ij];
                }

                t.x[l0]          =  __lows2half2(tmp[0], tmp[1]);
                t.x[l0 + t.ne/2] = __highs2half2(tmp[0], tmp[1]);
            }
        } else {
            half * xh = (half *) t.x;
#pragma unroll
            for (int l = 0; l < t.ne; ++l) {
#pragma unroll
                for (int o = 0; o < 2; ++o) {
                    const int j = 2*t.get_j(l) + o;
                    xh[2*l + o] = ((const half *) xs0)[swizzle<2*stride, half>(2*offset + j*(2*stride) + t.get_i(l), j)];
                }
            }
        }
#else
        GGML_UNUSED_VARS(t, xs0, offset);
        NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
    }

}
