// src/poly/fattn.cu - see the header.  The kernel is fattn_mma.cuh's, launched here in place of llama.cpp's
// launch_fattn(): one CUDA block a tile of 64 query rows (8 tokens x 8 heads, or 4 x 16), attending to a whole
// block of keys. The stream-k fixup is never needed; each tile leaves its softmax max and sum behind
// (fattn_mma.cuh, change 2).
#include "fattn.hpp"

#if !defined(GGML_USE_HIP)
#include "fattn_mma.cuh"
#endif

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>

namespace strata::poly {

#if !defined(GGML_USE_HIP)
namespace {

constexpr int kD = 256, kCols = 64;

struct Setup {
    bool ok = false;
    int nwarps = 0, nbatch_fa = 0;
    size_t smem = 0;
};

int grid(long long n, int per) { return (int) ((n + per - 1) / per); }

// The launch's shape and shared memory as ggml_cuda_flash_attn_ext_mma_f16_case() works them out, for the
// current card.  Ampere or later: on Turing the kernel has no 64-column tile.
template <int Group>
Setup make_setup(int cc) {
    constexpr int kCols1 = kCols / Group, kCols2 = Group;
    const auto kKernel = flash_attn_ext_f16<kD, kD, kCols1, kCols2, false, false, false>;
    Setup r;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !ampere_mma_available(cc)) return r;
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(kD, kD, kCols, cc);
    const int nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(kD, kD, kCols, cc);
    const int nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(kD, kD, kCols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(kD, kD, kCols, cc);
    const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(kD, kD, kCols, cc);
    const int nstages = ggml_cuda_fattn_mma_get_nstages(kD, kD, kCols1, kCols2, cc);
    r.nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(kD, kD, kCols, cc);
    r.nwarps = nthreads / WARP_SIZE;
    const int cols_per_warp = std::min(kCols, get_cols_per_warp(cc));
    // the tile's max and sum reach dst_meta only with one warp a query column (np == 1, fattn_mma.cuh change 2);
    // with more, the blocks of keys could not be merged
    if (r.nwarps * cols_per_warp != kCols) return r;
    const int stride_K = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_K2, cc);
    const int stride_V = ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_V2, cc);
    const size_t kv1 = size_t(r.nbatch_fa) * std::max(stride_K, stride_V) * sizeof(half2);
    const size_t kv2 = size_t(r.nbatch_fa) * (stride_K + stride_V) * sizeof(half2);
    const size_t q = size_t(kCols) * (kD / 2 + 4) * sizeof(half2);
    const size_t mask = size_t(kCols1) * (r.nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t combine = size_t(r.nwarps) * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
    const size_t kv = nstages <= 1 ? kv1 : kv2;
    r.smem = std::max(combine, q_in_reg ? std::max(q, kv + mask) : q + kv + mask);
    r.ok = cudaFuncSetAttribute(kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) r.smem) == cudaSuccess;
    cudaGetLastError();
    return r;
}

// Each card's own, worked out the first time it asks: a layer split may put cards of different generations in one
// process, and cudaFuncSetAttribute() holds for the current card only.
template <int Group>
const Setup& setup() {
    static std::mutex mu;
    static Setup s[GGML_CUDA_MAX_DEVICES];
    static bool done[GGML_CUDA_MAX_DEVICES] = {};
    const int d = ggml_cuda_get_device();
    std::lock_guard<std::mutex> lock(mu);
    if (!done[d]) {
        s[d] = make_setup<Group>(ggml_cuda_info().devices[d].cc);
        done[d] = true;
    }
    return s[d];
}

// The scratch for a batch of n tokens: a block's results and their max and sum, the merged max and sum, the
// causal block's steps, then K and V side by side, each piece 16-byte aligned (offsets in floats).  rows: the keys
// a block takes, a whole number of steps and at most a grid's 65535 rows (dequant_kernel); 0 when the scratch
// cannot hold a block of n + nbatch_fa rows.
struct Layout {
    size_t part = 0, pmeta = 0, ameta = 0, kv_max = 0, kv = 0;
    long long rows = 0;
};

template <int Group>
Layout layout(int n, int dim, int n_head, int kv_heads, int nbatch_fa, size_t scratch_floats) {
    Layout L;
    const size_t heads = size_t(n) * n_head;
    size_t at = 0;
    auto take = [&](size_t floats) {
        const size_t p = at;
        at += (floats + 3) & ~size_t(3);
        return p;
    };
    L.part = take(heads * dim);
    L.pmeta = take(2 * heads);
    L.ameta = take(2 * heads);
    L.kv_max = take((size_t) grid(n, kCols / Group));
    L.kv = at;
    if (at >= scratch_floats) return L;
    // a row of K and one of V take kv_heads * dim floats together
    const size_t width = size_t(kv_heads) * dim;
    const long long rows =
        std::min<long long>((long long) ((scratch_floats - at) / width), 65535) / nbatch_fa * nbatch_fa;
    if (rows >= (long long) n + nbatch_fa) L.rows = rows;
    return L;
}

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("fattn: ") + what + ": " + cudaGetErrorString(e));
}

// rows [row, row + valid) of a q8_0 cache (or an fp16 one, f16) as fp16, and zeros on to `blockIdx.y`'s last row:
// the kernel reads whole steps of keys, and the causal mask removes the zero rows' weights, but a NaN would get
// through it
__global__ void dequant_kernel(const int8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                               const half* __restrict__ f16, half* __restrict__ out, int width, long long row,
                               int valid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const int r = blockIdx.y;
    half x = __float2half(0.0f);
    if (r < valid) {
        const size_t at = size_t(row + r) * width;
        x = f16 ? f16[at + i]
                : __float2half((float) codes[at + i] * __half2float(__ushort_as_half(scales[at / 32 + i / 32])));
    }
    out[size_t(r) * width + i] = x;
}

// The last step of keys a tile of tokens reads in the causal block: token j sees the keys up to j + delta.
template <int Cols1>
__global__ void kv_max_kernel(int* __restrict__ kv_max, int tiles, int n, long long delta, int keys, int step) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tiles) return;
    const int j = t * Cols1 + Cols1 - 1 < n ? t * Cols1 + Cols1 - 1 : n - 1;
    const long long seen = (j + delta + 1 + step - 1) / step * step;
    kv_max[t] = seen < keys ? (int) seen : keys;
}

// A block of keys' results into what the blocks before it made of a token and head: each normalized by its own
// softmax sum, so each weighs in by its sum against the larger max.  One CUDA block a token and head.
__global__ void merge_kernel(float* __restrict__ out, float2* __restrict__ meta, const float* __restrict__ part,
                             const float2* __restrict__ pmeta, int dim) {
    const size_t row = blockIdx.x;
    const float2 a = meta[row], b = pmeta[row];
    const float top = fmaxf(a.x, b.x), wa = a.y * __expf(a.x - top), wb = b.y * __expf(b.x - top);
    const float inv = 1.0f / (wa + wb);
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        out[row * dim + d] = (out[row * dim + d] * wa + part[row * dim + d] * wb) * inv;
    __syncthreads();
    if (threadIdx.x == 0) meta[row] = make_float2(top, wa + wb);
}

// The q8 cache (kq8, vq8) or the fp16 one (keys, vals): an fp16 cache's whole steps before pos are read where they
// lie, in one block, and only the last block is copied into the scratch.
template <int Group>
bool run(const float* q, Kvq8 kq8, Kvq8 vq8, const half* keys, const half* vals, int64_t pos, int n, float* out,
         int dim, int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream) {
    if (n <= 0 || !fattn_fits(dim, n_head, kv_heads, n, scratch_floats)) return false;
    constexpr int kCols1 = kCols / Group;
    const auto kKernel = flash_attn_ext_f16<kD, kD, kCols1, Group, false, false, false>;
    const Setup& S = setup<Group>();
    const int width = kv_heads * dim, tiles = grid(n, kCols1);
    const size_t heads = size_t(n) * n_head;
    const Layout L = layout<Group>(n, dim, n_head, kv_heads, S.nbatch_fa, scratch_floats);
    if (L.rows == 0) return false;
    const long long rows = L.rows;
    float* part = scratch + L.part;
    float2* pmeta = (float2*) (scratch + L.pmeta);
    float2* ameta = (float2*) (scratch + L.ameta);
    int* kv_max = (int*) (scratch + L.kv_max);
    half* kw = (half*) (scratch + L.kv);
    half* vw = kw + size_t(rows) * width;

    const cudaStream_t s = (cudaStream_t) stream;
    const dim3 block(WARP_SIZE, S.nwarps);
    const uint3 ne01 = init_fastdiv_values((uint64_t) n);
    const int32_t nb01 = n_head * dim * (int) sizeof(float), nb02 = dim * (int) sizeof(float);
    const int32_t nb11 = width * (int) sizeof(half), nb12 = dim * (int) sizeof(half);
    const long long total = pos + n;
    const long long whole = pos / S.nbatch_fa * S.nbatch_fa;   // an fp16 cache's rows read in place
    for (long long k0 = 0, b = 0; k0 < total; ++b) {
        // Blocks of whole steps from the start, as long as what is left is more than a block; then the rest,
        // which holds the batch's own rows and starts at or before pos, so that every token sees a key of it.
        const bool last = keys ? k0 == whole : total - k0 <= rows;
        const long long m = last ? total - k0
                          : keys ? whole
                                 : std::min<long long>(rows, (pos - k0) / S.nbatch_fa * S.nbatch_fa);
        const int nk = grid(m, S.nbatch_fa) * S.nbatch_fa;
        const long long delta = pos - k0;
        const half* kb = kw;
        const half* vb = vw;
        if (keys && !last) {
            kb = keys, vb = vals;
        } else {
            dequant_kernel<<<dim3(grid(width, 256), nk), 256, 0, s>>>(kq8.codes, kq8.scales, keys, kw, width, k0,
                                                                      (int) m);
            dequant_kernel<<<dim3(grid(width, 256), nk), 256, 0, s>>>(vq8.codes, vq8.scales, vals, vw, width, k0,
                                                                      (int) m);
        }
        if (last) kv_max_kernel<kCols1><<<grid(tiles, 128), 128, 0, s>>>(kv_max, tiles, n, delta, nk, S.nbatch_fa);
        float* dst = b == 0 ? out : part;
        float2* meta = b == 0 ? ameta : pmeta;
        kKernel<<<tiles * kv_heads, block, S.smem, s>>>(
            (const char*) q, (const char*) kb, (const char*) vb, nullptr, nullptr, last ? kv_max : nullptr, dst, meta,
            scale, 0.0f, 1.0f, 1.0f, 1u, 0.0f,
            dim, ne01, n_head, 1, nb01, nb02, nb01 * n,
            dim, nk, kv_heads, 1, nb11, nb12, (int64_t) nb11 * nk,
            nb11, nb12, (int64_t) nb11 * nk,
            n, 1, 1, (int32_t) (2 * delta), 0, 0);
        ck("attention");
        if (b > 0) merge_kernel<<<(unsigned) heads, dim, 0, s>>>(out, ameta, part, pmeta, dim);
        k0 += m;
    }
    ck("merge");
    return true;
}

// ---- decode (fdec_q8): a few tokens over the whole cache, split by keys, on the tensor cores ----

constexpr int kSplit = 1024;   // keys a CUDA block takes, counted from the cache's first row
constexpr int kTile = 64;      // keys a block takes at a time: 16 a warp
constexpr int kDecWarps = 4;
constexpr int kDecRows = 32;   // 4 tokens x 8 heads, or 2 x 16; the scratch has the same size

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// Two int8 of `w` (each with its top bit flipped) as a half2, exactly: 0x64XX is 1024 + XX.  sel 0x4140 takes
// bytes 0 and 1, 0x4342 bytes 2 and 3.
__device__ __forceinline__ uint32_t i8pair(uint32_t w, uint32_t sel) {
    uint32_t x = __byte_perm(w, 0x64646464u, sel);
    half2 h = __hsub2(*reinterpret_cast<half2*>(&x), __float2half2_rn(1152.0f));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t mul2(uint32_t x, half2 s) {
    half2 h = __hmul2(*reinterpret_cast<half2*>(&x), s);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint4 widen(uint2 v) { return make_uint4(v.x, v.y, 0u, 0u); }

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i < 2 ? (i == 0 ? v.x : v.y) : (i == 2 ? v.z : v.w);
}

__device__ __forceinline__ float scale_of(const uint4& v, int b) {   // half b of 8
    const uint32_t w = b < 4 ? (b < 2 ? v.x : v.y) : (b < 6 ? v.z : v.w);
    return __half2float(__ushort_as_half((unsigned short) (b & 1 ? w >> 16 : w & 0xffff)));
}

// One CUDA block: the keys [split * kSplit, +kSplit) of group blockIdx.y for the launch's tokens (MT * 16 / Group
// at most, row = token * Group + head of the group), as flash attention: per tile of 64 keys, a warp scores 16
// (S = Q K^T, mma m16n8k16, the queries as halves, the key codes exact as halves and each 32-value block's scale
// applied to its sum in floats), the softmax max and sum go on per row, and a warp adds 64 of the 256 value
// dimensions (O += P V, the weights and the values as halves).  It leaves the rows' unnormalized results with
// their max and sum; fdec_merge_kernel adds the splits.  A row's arithmetic never depends on the other rows: a
// token comes out the same to the bit whatever tokens the launch has beside it.  The dimensions are numbered as
// suits the reads (a thread's 8 values of a block of K, its 8 dimensions of V), which the sums don't see.  F16: an
// fp16 cache (kc and vc are halves, no scales), read as it is.
template <int MT, bool F16, int Group>
__global__ void __launch_bounds__(kDecWarps * WARP_SIZE)
fdec_kernel(const float* __restrict__ q, const void* __restrict__ kc, const uint16_t* __restrict__ ks,
            const void* __restrict__ vc, const uint16_t* __restrict__ vs, long long pos,
            const long long* __restrict__ at, int first, int tokens, int n_head, int kv_heads, float scale,
            float* __restrict__ part, float* __restrict__ ml) {
    constexpr int Q = MT * 16, QS = kD + 8, PS = kTile + 8;
    __shared__ __align__(16) half Qs[Q * QS];
    __shared__ __align__(16) half Ps[Q * PS];
    __shared__ float red[2][kDecWarps][Q];
    const int w = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE, g8 = lane / 4, t = lane % 4;
    const int grp = blockIdx.y, split = blockIdx.x;
    const long long p0 = (at ? *at : pos) + first, last = p0 + tokens - 1, r0 = (long long) split * kSplit;
    if (r0 > last) return;
    const long long r1 = r0 + kSplit < last + 1 ? r0 + kSplit : last + 1;
    for (int i = threadIdx.x; i < Q * kD; i += blockDim.x) {
        const int row = i / kD, d = i % kD, tok = row / Group;
        const float x = tok < tokens ? q[((size_t) (first + tok) * n_head + grp * Group + row % Group) * kD + d] : 0.f;
        Qs[row * QS + d] = __float2half(x);
    }
    constexpr int E = F16 ? 2 : 1;                         // bytes a value
    constexpr uint32_t kZero = F16 ? 0u : 0x80808080u;     // a zero (a q8 code's once flipped)
    const size_t width = (size_t) kv_heads * kD, swidth = width / 32;
    const char* kg = (const char*) kc + (size_t) grp * kD * E;
    const char* vg = (const char*) vc + ((size_t) grp * kD + 64 * w + 8 * g8) * E;
    const uint16_t* ksg = F16 ? nullptr : ks + (size_t) grp * (kD / 32);
    const uint16_t* vsg = F16 ? nullptr : vs + (size_t) grp * (kD / 32) + (64 * w + 8 * g8) / 32;
    float m[MT][2], l[MT][2], o[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        m[mt][0] = m[mt][1] = -INFINITY;
        l[mt][0] = l[mt][1] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[mt][j][0] = o[mt][j][1] = o[mt][j][2] = o[mt][j][3] = 0.f;
    }
    __syncthreads();
    for (long long tb = r0; tb < r1; tb += kTile) {
        // the reads first, all of them: K of the warp's 16 keys (n-tile nt: key 8 nt + g8; a thread 8 values of
        // each 32-value block), their scales (keys 8 nt + 2t + j), V of all 64 keys for the warp's dimensions.
        // q8 fills .x and .y (8 codes), fp16 all four words (8 halves).
        uint4 kx[2][8];
        uint4 ksc[2][2] = {};
        uint4 vx[4][4];
        uint16_t vsc[4][4] = {};
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const long long key = tb + 16 * w + 8 * nt + g8;
#pragma unroll
            for (int b = 0; b < 8; ++b)
                kx[nt][b] = key > last ? make_uint4(kZero, kZero, kZero, kZero)
                          : F16 ? *reinterpret_cast<const uint4*>(kg + (key * width + 32 * b + 8 * t) * E)
                                : widen(*reinterpret_cast<const uint2*>(kg + key * width + 32 * b + 8 * t));
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const long long k2 = tb + 16 * w + 8 * nt + 2 * t + j;
                if (!F16 && k2 <= last) ksc[nt][j] = *reinterpret_cast<const uint4*>(ksg + k2 * swidth);
            }
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const long long key = tb + 16 * kk + 2 * t + (i & 1) + 8 * (i >> 1);
                vx[kk][i] = key > last ? make_uint4(kZero, kZero, kZero, kZero)
                          : F16 ? *reinterpret_cast<const uint4*>(vg + key * width * E)
                                : widen(*reinterpret_cast<const uint2*>(vg + key * width));
                if (!F16 && key <= last) vsc[kk][i] = vsg[key * swidth];
            }
        // S = Q K^T, a 32-value block (two k-steps) at a time, then times the keys' scales of the block
        float s[MT][2][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) s[mt][nt][0] = s[mt][nt][1] = s[mt][nt][2] = s[mt][nt][3] = 0.f;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            float c[MT][2][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) c[mt][nt][0] = c[mt][nt][1] = c[mt][nt][2] = c[mt][nt][3] = 0.f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int d = 32 * b + 8 * t + 4 * h;   // the thread's 4 dimensions of the step
                uint32_t bk[2][2];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    if constexpr (F16) {
                        bk[nt][0] = h ? kx[nt][b].z : kx[nt][b].x;
                        bk[nt][1] = h ? kx[nt][b].w : kx[nt][b].y;
                    } else {
                        const uint32_t word = (h ? kx[nt][b].y : kx[nt][b].x) ^ 0x80808080u;
                        bk[nt][0] = i8pair(word, 0x4140);
                        bk[nt][1] = i8pair(word, 0x4342);
                    }
                }
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const uint2 lo = *reinterpret_cast<const uint2*>(&Qs[(mt * 16 + g8) * QS + d]);
                    const uint2 hi = *reinterpret_cast<const uint2*>(&Qs[(mt * 16 + g8 + 8) * QS + d]);
                    const uint32_t a[4] = {lo.x, hi.x, lo.y, hi.y};
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) mma16816(c[mt][nt], a, bk[nt][0], bk[nt][1]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float sc = F16 ? 1.f : scale_of(ksc[nt][j], b);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        s[mt][nt][j] += c[mt][nt][j] * sc;
                        s[mt][nt][2 + j] += c[mt][nt][2 + j] * sc;
                    }
                }
        }
        // the causal mask, the tile's max per row over the four warps, the new max
        float alpha[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r, tok = row / Group;
                float mx = -INFINITY;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const long long key = tb + 16 * w + 8 * nt + 2 * t + j;
                        float& x = s[mt][nt][2 * r + j];
                        x = tok < tokens && key <= p0 + tok ? x * scale : -INFINITY;
                        mx = fmaxf(mx, x);
                    }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                if (t == 0) red[0][w][row] = mx;
            }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r;
                float top = m[mt][r];
#pragma unroll
                for (int k = 0; k < kDecWarps; ++k) top = fmaxf(top, red[0][k][row]);
                alpha[mt][r] = top == -INFINITY ? 1.f : expf(m[mt][r] - top);
                m[mt][r] = top;
                float sum = 0.f;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const float x0 = s[mt][nt][2 * r], x1 = s[mt][nt][2 * r + 1];
                    const half2 p = __floats2half2_rn(x0 == -INFINITY ? 0.f : expf(x0 - top),
                                                      x1 == -INFINITY ? 0.f : expf(x1 - top));
                    sum += __low2float(p) + __high2float(p);
                    *reinterpret_cast<half2*>(&Ps[row * PS + 16 * w + 8 * nt + 2 * t]) = p;
                }
                sum += __shfl_xor_sync(0xffffffffu, sum, 1);
                sum += __shfl_xor_sync(0xffffffffu, sum, 2);
                if (t == 0) red[1][w][row] = sum;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    o[mt][j][2 * r] *= alpha[mt][r];
                    o[mt][j][2 * r + 1] *= alpha[mt][r];
                }
            }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = mt * 16 + g8 + 8 * r;
                float sum = 0.f;
#pragma unroll
                for (int k = 0; k < kDecWarps; ++k) sum += red[1][k][row];
                l[mt][r] = l[mt][r] * alpha[mt][r] + sum;
            }
        // O += P V: k-step kk is keys 16 kk.. of the tile; n-tile j column g8 is dimension 64 w + 8 g8 + j
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t a[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const half* p = &Ps[(mt * 16 + g8) * PS + 16 * kk + 2 * t];
                a[mt][0] = *reinterpret_cast<const uint32_t*>(p);
                a[mt][1] = *reinterpret_cast<const uint32_t*>(p + 8 * PS);
                a[mt][2] = *reinterpret_cast<const uint32_t*>(p + 8);
                a[mt][3] = *reinterpret_cast<const uint32_t*>(p + 8 * PS + 8);
            }
            if constexpr (F16) {
                // n-tile j: half j % 2 of word j / 2 of the four keys' 8 halves
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t sel = j & 1 ? 0x7632 : 0x5410;
                    const uint32_t b0 = __byte_perm(word(vx[kk][0], j / 2), word(vx[kk][1], j / 2), sel);
                    const uint32_t b1 = __byte_perm(word(vx[kk][2], j / 2), word(vx[kk][3], j / 2), sel);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) mma16816(o[mt][j], a[mt], b0, b1);
                }
                continue;
            }
            const half2 s01 = __halves2half2(__ushort_as_half(vsc[kk][0]), __ushort_as_half(vsc[kk][1]));
            const half2 s89 = __halves2half2(__ushort_as_half(vsc[kk][2]), __ushort_as_half(vsc[kk][3]));
#pragma unroll
            for (int wd = 0; wd < 2; ++wd) {
                const uint32_t x0 = (wd ? vx[kk][0].y : vx[kk][0].x) ^ 0x80808080u;
                const uint32_t x1 = (wd ? vx[kk][1].y : vx[kk][1].x) ^ 0x80808080u;
                const uint32_t x2 = (wd ? vx[kk][2].y : vx[kk][2].x) ^ 0x80808080u;
                const uint32_t x3 = (wd ? vx[kk][3].y : vx[kk][3].x) ^ 0x80808080u;
                const uint32_t lo01 = __byte_perm(x0, x1, 0x5140), hi01 = __byte_perm(x0, x1, 0x7362);
                const uint32_t lo89 = __byte_perm(x2, x3, 0x5140), hi89 = __byte_perm(x2, x3, 0x7362);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const uint32_t sel = jj & 1 ? 0x4342 : 0x4140;
                    const uint32_t b0 = mul2(i8pair(jj < 2 ? lo01 : hi01, sel), s01);
                    const uint32_t b1 = mul2(i8pair(jj < 2 ? lo89 : hi89, sel), s89);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) mma16816(o[mt][4 * wd + jj], a[mt], b0, b1);
                }
            }
        }
        __syncthreads();   // Ps and red are the next tile's
    }
    // the rows' results: c0 of n-tile j is dimension 64 w + 16 t + j, c1 the one 8 on
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = mt * 16 + g8 + 8 * r;
            if (row / Group >= tokens) continue;
            const size_t idx = ((size_t) split * kv_heads + grp) * kDecRows + row;
            float4* dst = reinterpret_cast<float4*>(part + idx * kD + 64 * w + 16 * t);
            dst[0] = make_float4(o[mt][0][2 * r], o[mt][1][2 * r], o[mt][2][2 * r], o[mt][3][2 * r]);
            dst[1] = make_float4(o[mt][4][2 * r], o[mt][5][2 * r], o[mt][6][2 * r], o[mt][7][2 * r]);
            dst[2] = make_float4(o[mt][0][2 * r + 1], o[mt][1][2 * r + 1], o[mt][2][2 * r + 1], o[mt][3][2 * r + 1]);
            dst[3] = make_float4(o[mt][4][2 * r + 1], o[mt][5][2 * r + 1], o[mt][6][2 * r + 1], o[mt][7][2 * r + 1]);
            if (w == 0 && t == 0) {
                ml[2 * idx] = m[mt][r];
                ml[2 * idx + 1] = l[mt][r];
            }
        }
}

// The splits of a token and head added in their order, each by its sum against the largest max.  One CUDA block
// a head and token, a thread a dimension.  A token reads only the splits up to its own row.
template <int Group>
__global__ void fdec_merge_kernel(const float* __restrict__ part, const float* __restrict__ ml, float* __restrict__ out,
                                  long long pos, const long long* __restrict__ at, int first, int n_head, int kv_heads) {
    const int h = blockIdx.x, tok = blockIdx.y, d = threadIdx.x, grp = h / Group;
    const long long p = (at ? *at : pos) + first + tok;
    const int splits = int(p / kSplit) + 1, row = tok * Group + h % Group;
    float top = -INFINITY;
    for (int i = 0; i < splits; ++i) top = fmaxf(top, ml[2 * (((size_t) i * kv_heads + grp) * kDecRows + row)]);
    float sum = 0.f, acc = 0.f;
    for (int i = 0; i < splits; ++i) {
        const size_t idx = ((size_t) i * kv_heads + grp) * kDecRows + row;
        const float e = expf(ml[2 * idx] - top);
        sum += ml[2 * idx + 1] * e;
        acc += part[idx * kD + d] * e;
    }
    out[((size_t) (first + tok) * n_head + h) * kD + d] = acc / sum;
}

// splits of 1,024 keys up to `rows`, each with its results for kDecRows query rows and a max and a sum each
size_t fdec_floats(int rows, int kv_heads) { return (size_t) grid(rows, kSplit) * kv_heads * kDecRows * (kD + 2); }

}  // namespace

bool fattn_fits(int dim, int n_head, int kv_heads, int rows, size_t scratch_floats) {
    if (dim != kD || kv_heads <= 0 || rows <= 0) return false;
    if (n_head == 8 * kv_heads) {
        const Setup& S = setup<8>();
        return S.ok && layout<8>(rows, dim, n_head, kv_heads, S.nbatch_fa, scratch_floats).rows > 0;
    }
    if (n_head == 16 * kv_heads) {
        const Setup& S = setup<16>();
        return S.ok && layout<16>(rows, dim, n_head, kv_heads, S.nbatch_fa, scratch_floats).rows > 0;
    }
    return false;
}

bool fattn_q8(const float* q, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n, float* out, int dim, int n_head, int kv_heads,
              float scale, float* scratch, size_t scratch_floats, void* stream) {
    return kq8.codes && vq8.codes &&
           (n_head == 16 * kv_heads
                ? run<16>(q, kq8, vq8, nullptr, nullptr, pos, n, out, dim, n_head, kv_heads, scale, scratch,
                          scratch_floats, stream)
                : run<8>(q, kq8, vq8, nullptr, nullptr, pos, n, out, dim, n_head, kv_heads, scale, scratch,
                         scratch_floats, stream));
}

bool fattn_f16(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t pos, int n, float* out, int dim,
               int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream) {
    return keys && vals &&
           (n_head == 16 * kv_heads
                ? run<16>(q, Kvq8{}, Kvq8{}, (const half*) keys, (const half*) vals, pos, n, out, dim, n_head,
                          kv_heads, scale, scratch, scratch_floats, stream)
                : run<8>(q, Kvq8{}, Kvq8{}, (const half*) keys, (const half*) vals, pos, n, out, dim, n_head,
                         kv_heads, scale, scratch, scratch_floats, stream));
}

bool fdec_fits(int dim, int n_head, int kv_heads, int rows, size_t scratch_floats) {
    if (dim != kD || kv_heads <= 0 || (n_head != 8 * kv_heads && n_head != 16 * kv_heads) || rows <= 0) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ampere_mma_available(cc) && fdec_floats(rows, kv_heads) <= scratch_floats;
}

namespace {

template <bool F16, int Group>
bool run_dec(const float* q, const void* kc, const uint16_t* ks, const void* vc, const uint16_t* vs, int64_t pos, int n,
             const long long* at, int room, float* out, int dim, int n_head, int kv_heads, float scale, float* scratch,
             size_t scratch_floats, void* stream) {
    const int rows = (int) (at ? room : pos + n);   // a recorded launch is sized for `room` rows
    if (n <= 0 || !kc || !vc || !fdec_fits(dim, n_head, kv_heads, rows, scratch_floats)) return false;
    const int splits = grid(rows, kSplit);
    float* ml = scratch + (size_t) splits * kv_heads * kDecRows * kD;
    const cudaStream_t s = (cudaStream_t) stream;
    constexpr int kDecTokens = kDecRows / Group;
    for (int first = 0; first < n; first += kDecTokens) {
        const int c = std::min(kDecTokens, n - first);
        const dim3 blocks(splits, kv_heads);
        const auto kernel = c * Group <= 16 ? fdec_kernel<1, F16, Group> : fdec_kernel<2, F16, Group>;
        kernel<<<blocks, kDecWarps * WARP_SIZE, 0, s>>>(q, kc, ks, vc, vs, (long long) pos, at, first, c, n_head, kv_heads,
                                                        scale, scratch, ml);
        fdec_merge_kernel<Group><<<dim3(n_head, c), kD, 0, s>>>(scratch, ml, out, (long long) pos, at, first, n_head,
                                                         kv_heads);
    }
    ck("decode attention");
    return true;
}

}  // namespace

bool fdec_q8(const float* q, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n, const long long* at, int room, float* out,
             int dim, int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream) {
    return kq8.scales && vq8.scales &&
           (n_head == 16 * kv_heads
                ? run_dec<false, 16>(q, kq8.codes, kq8.scales, vq8.codes, vq8.scales, pos, n, at, room, out, dim,
                                     n_head, kv_heads, scale, scratch, scratch_floats, stream)
                : run_dec<false, 8>(q, kq8.codes, kq8.scales, vq8.codes, vq8.scales, pos, n, at, room, out, dim,
                                    n_head, kv_heads, scale, scratch, scratch_floats, stream));
}

bool fdec_f16(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t pos, int n, const long long* at,
              int room, float* out, int dim, int n_head, int kv_heads, float scale, float* scratch,
              size_t scratch_floats, void* stream) {
    return n_head == 16 * kv_heads
               ? run_dec<true, 16>(q, keys, nullptr, vals, nullptr, pos, n, at, room, out, dim, n_head, kv_heads,
                                  scale, scratch, scratch_floats, stream)
               : run_dec<true, 8>(q, keys, nullptr, vals, nullptr, pos, n, at, room, out, dim, n_head, kv_heads,
                                 scale, scratch, scratch_floats, stream);
}

#else

bool fattn_fits(int, int, int, int, size_t) { return false; }

bool fdec_fits(int, int, int, int, size_t) { return false; }

bool fdec_q8(const float*, Kvq8, Kvq8, int64_t, int, const long long*, int, float*, int, int, int, float, float*, size_t,
             void*) {
    return false;
}

bool fdec_f16(const float*, const uint16_t*, const uint16_t*, int64_t, int, const long long*, int, float*, int, int, int,
              float, float*, size_t, void*) {
    return false;
}

bool fattn_q8(const float*, Kvq8, Kvq8, int64_t, int, float*, int, int, int, float, float*, size_t, void*) {
    return false;
}

bool fattn_f16(const float*, const uint16_t*, const uint16_t*, int64_t, int, float*, int, int, int, float, float*,
               size_t, void*) {
    return false;
}

#endif

}  // namespace strata::poly
